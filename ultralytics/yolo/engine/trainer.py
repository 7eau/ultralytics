# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Train a model on a dataset

Usage:
    $ yolo mode=train model=yolov8n.pt data=coco128.yaml imgsz=640 epochs=100 batch=16
"""
import math
import os
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from ultralytics.nn.tasks import attempt_load_one_weight, attempt_load_weights
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, __version__, callbacks,
                                    clean_url, colorstr, emojis, yaml_save)
from ultralytics.yolo.utils.autobatch import check_train_batch_size
from ultralytics.yolo.utils.checks import check_amp, check_file, check_imgsz, print_args
from ultralytics.yolo.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.yolo.utils.files import get_latest_run, increment_path
from ultralytics.yolo.utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, init_seeds, one_cycle,
                                                select_device, strip_optimizer)


class BaseTrainer:
    """
    BaseTrainer

    A base class for creating trainers.

    Attributes:
        args (SimpleNamespace): 训练的参数配置.
        check_resume (method): 检查是否需要从已保存的权重恢复训练的方法
        validator (BaseValidator): 模型效果评估器.
        model (nn.Module): 模型实例.
        callbacks (defaultdict): 训练回调函数.
        save_dir (Path): 保存训练结果的文件夹.
        wdir (Path): 保存权重的文件夹 Directory to save weights.
        last (Path): 保存最后一个epoch权重的路径 Path to last checkpoint.
        best (Path): 保存最优epoch权重的路径Path to best checkpoint.
        save_period (int): 每x个epoch保存一次权重，x > 1.
        batch_size (int): 训练的Batch size.
        epochs (int): 训练的epoch.
        start_epoch (int): 从第x个epoch开始训练.
        device (torch.device): 用以训练的设备.
        amp (bool): 是否进行自动精度混合 Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): 用以自动混合精度的梯度缩放器 Gradient scaler for AMP.
        data (str): 训练数据的路径 Path to data.
        trainset (torch.utils.data.Dataset): 训练数据集 Training dataset.
        testset (torch.utils.data.Dataset): 特使数据集 Testing dataset.
        ema (nn.Module): 模型权重的指数移动平均 EMA (Exponential Moving Average) of the model.
        lf (nn.Module): 损失函数 Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): 学习率调度器 Learning rate scheduler.
        best_fitness (float): 已完成的训练中最好的fitness The best fitness value achieved.
        fitness (float): 当前的fitness值 Current fitness value.
        loss (float): 当前的损失值 Current loss value.
        tloss (float): 总的损失值 Total loss value.
        loss_names (list): 训练过程计算出的所有的loss名称的列表(包括cls_loss, bos_loss, obj_loss).
        csv (Path): 保存训练过程的csv文件 Path to results CSV file.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        BaseTrainer初始化函数

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)  # 加载参数
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)  # 初始化随机种子

        # 设置保存目录和模型保存路径
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task  # 用户定义的项目目录
        name = self.args.name or f'{self.args.mode}'  # 用户定义的该次运行名称

        if hasattr(self.args, 'save_dir'):  # 如果用户已定义了save_dir则直接使用
            self.save_dir = Path(self.args.save_dir)
        else:  # 否则根据project、name以递增的方式(例如: detect1, detect2, ...)构建出save_dir
            self.save_dir = Path(
                increment_path(Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))
        self.wdir = self.save_dir / 'weights'  # 权重文件保存文件夹
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # 保存训练参数
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # 权重保存路径
        self.save_period = self.args.save_period

        # 初始化epoch部分参数
        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # 在CPU环境下,使用单进程数据加载会更快,不需要额外的workers

        # Model and Dataset
        self.model = self.args.model
        # 根据任务类型调用检查函数:
        # ==============================================================================================================
        # 校验数据集路径是否存在, 如果是yaml文件,将解析并返回dataset字典
        # 获取解析后的dataset,更新到self.data,
        # 解析后的dataset信息, 包含yaml_file: 原始yaml路径， train/val/test路径等
        # 如果dataset不合法, 会raise一个异常
        # ==============================================================================================================
        try:
            if self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            elif self.args.data.endswith('.yaml') or self.args.task in ('detect', 'segment'):
                self.data = check_det_dataset(self.args.data)
                if 'yaml_file' in self.data:
                    self.args.data = self.data['yaml_file']  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e

        # 从self.data中获取训练集和测试集的数据
        # ==============================================================================================================
        # 将data字典抽象为数据集对象,方便后续统一地训练、验证不同的数据集
        # get_dataset需要子类实现,将data解析成针对特定任务的训练集、测试集
        # ==============================================================================================================
        self.trainset, self.testset = self.get_dataset(self.data)
        self.ema = None

        # 优化器部分初始化
        self.lf = None
        self.scheduler = None

        # Epoch层级的metrics指标
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # 回调函数
        # ==============================================================================================================
        # 回调函数在训练循环中的不同时期被调用,以实现辅助操作。
        #
        # 默认回调函数提供基础功能,如EarlyStopping、ModelCheckpoint等。而集成回调函数在主进程中被添加,进行日志记录、
        # 结果写入等辅助工作。这样可以方便地使用回调函数来增强训练循环,而不需要直接修改循环内代码。通过回调函数可以扩展Trainer的功能,
        # 而不影响核心训练逻辑。
        # ==============================================================================================================
        self.callbacks = _callbacks or callbacks.get_default_callbacks()  # 获取默认回调函数组
        if RANK in (-1, 0):  # 如果是主进程(单卡训练或多卡训练中rank=0的进程),添加集成回调函数
            callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback):
        """
        Appends the given callback.
        """
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """
        Overrides the existing callbacks with the given callback.
        """
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """
        训练循环的主要函数
        Allow device='', device=None on Multi-GPU systems to default to device=0.
        """
        # 1. 计算可工作的GPU数量(world_size)
        if isinstance(self.args.device, int) or self.args.device:  # i.e. device=0 or device=[0,1,2,3]
            world_size = torch.cuda.device_count()
        elif torch.cuda.is_available():  # i.e. device=None or device=''
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        # 2. 判断是否需要分布式训练
        if world_size > 1 and 'LOCAL_RANK' not in os.environ:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with Multi-GPU training, setting rect=False")
                self.args.rect = False
            # Command
            cmd, file = generate_ddp_command(world_size, self)
            try:
                LOGGER.info(f'DDP command: {cmd}')
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))
        else:  # 不需要分布式训练，进行普通训练
            self._do_train(world_size)

    def _setup_ddp(self, world_size):
        """Initializes and sets the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device('cuda', RANK)
        LOGGER.info(f'DDP info: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
        os.environ['NCCL_BLOCKING_WAIT'] = '1'  # set to enforce timeout
        dist.init_process_group(
            'nccl' if dist.is_nccl_available() else 'gloo',
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=world_size)

    def _setup_train(self, world_size):
        """
        训练初始化的关键函数，完成训练中需要的各组件的构建和初始化
        """
        # 1. 构建模型
        self.run_callbacks('on_pretrain_routine_start')
        ckpt = self.setup_model()  # 加载或构建模型
        self.model = self.model.to(self.device)  # 将模型放到设备上(CPU/GPU)
        self.set_model_attributes()

        # 2. 初始化AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # 根据args判断是否使用AMP
        if self.amp and RANK in (-1, 0):  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)  # 检查和初始化AMP
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and world_size > 1:  # DDP
            dist.broadcast(self.amp, src=0)  # broadcast the tensor from rank 0 to all other ranks (returns None)
        self.amp = bool(self.amp)  # as boolean
        self.scaler = amp.GradScaler(enabled=self.amp)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[RANK])

        # 3. 设置图片大小
        gs = max(int(self.model.stride.max() if hasattr(self.model, 'stride') else 32), 32)  # 计算模型最大步幅(grid size)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)  # 检查输入图片大小是否可行

        # 4. 设置Batch size
        if self.batch_size == -1:
            if RANK == -1:  # 如果为auto batch,在单GPU下自动选择合适的batch size
                self.args.batch = self.batch_size = check_train_batch_size(self.model, self.args.imgsz, self.amp)
            else:
                SyntaxError('batch=-1 to use AutoBatch is only available in Single-GPU training. '
                            'Please pass a valid batch size value for Multi-GPU DDP training, i.e. batch=16')

        # 5. 构建数据集的Dataloaders
        batch_size = self.batch_size // max(world_size, 1)
        # 调用get_dataloader()构建训练集和验证集的dataloader
        self.train_loader = self.get_dataloader(self.trainset, batch_size=batch_size, rank=RANK, mode='train')
        if RANK in (-1, 0):
            self.test_loader = self.get_dataloader(self.testset, batch_size=batch_size * 2, rank=-1, mode='val')
            self.validator = self.get_validator()
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix='val')
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))  # TODO: init metrics for plot_results()?
            self.ema = ModelEMA(self.model)
            if self.args.plots and not self.args.v5loader:
                self.plot_training_labels()

        # 6. 构建优化器(Optimizer)
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(model=self.model,
                                              name=self.args.optimizer,
                                              lr=self.args.lr0,
                                              momentum=self.args.momentum,
                                              decay=weight_decay,
                                              iterations=iterations)

        # 7. 构建学习率调度器 Scheduler: 创建学习率调度策略(余弦退火或线性衰减)
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

        # 8. 初始化Early Stopping
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False

        # 9. 恢复训练:从之前的检查点恢复训练
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move

        self.run_callbacks('on_pretrain_routine_end')

    def _do_train(self, world_size=1):
        """
        完成训练、评估、绘制指标图, 是组织训练流程的核心函数
        """
        if world_size > 1:
            self._setup_ddp(world_size)

        # 训练组件的初始化
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()

        nb = len(self.train_loader)  # batch的数量(number of batches)
        nw = max(round(self.args.warmup_epochs *
                       nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup的总迭代数(number of warmup iterations)

        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')

        # 控制在训练过程中何时关闭 mosaic 数据增强
        # ============================================[mosaic 数据增强]==================================================
        #   mosaic数据增强是YOLO训练常用的一种技巧, 可以增强模型的泛化能力。它的基本思路是:
        #       1. 随机从数据集中取出4张图片
        #       2. 在一张大图中拼接这4张图片
        #       3. 在拼接图像上进行目标检测训练
        #
        #   这可以让模型看到更多样化的样本, 避免过拟合。 但是过度使用mosaic也会使训练难以收敛, 且Inference时模型不会看到
        #   mosaic过的图像，所以一般在训练中期(如最后10个epochs)会关闭mosaic, 只保留常规的数据增强, 让模型专注正常图像的训练。
        # ==============================================================================================================
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        # 处理恢复已完成训练的模型的边界情况
        # ==============================================================================================================
        #   因为后面会在循环中对epoch计数,当恢复一个已full train的模型时,start_epoch可能等于self.epochs,
        #   如果不预定义epoch,会导致这个循环直接退出,训练流程不完整。
        # ==============================================================================================================
        epoch = self.epochs  # predefine for resume fully trained model edge cases

        # 开始训练循环
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()  # 将PyTorch模型设置为训练模式
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):  # 当epoch达到 总计数 - close_mosaic 的时刻,就会关闭mosaic
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:  # 单个batch训练开始
                self.run_callbacks('on_train_batch_start')
                # Warmup：当迭代数ni在nw内时,会通过线性插值的方式逐步增大学习率,从一个较小值warmup到指定的学习率
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # 前向推理(Forward)
                with torch.cuda.amp.autocast(self.amp):  # 使用自动混合精度(AMP)进行模型的前向计算
                    batch = self.preprocess_batch(batch)  # 对一个batch数据进行预处理
                    self.loss, self.loss_items = self.model(batch)  # 模型前向计算,返回loss和各个损失项
                    if RANK != -1:  # 在分布式训练下对loss进行缩放, 为了在reduce时可以正确聚合结果
                        self.loss *= world_size
                    # 计算正在变化的平均loss
                    # ==================================================================================================
                    # 取前i步的mean loss 和 当前step的loss做平均， 如果self.tloss为空则直接取当前loss
                    # 这是模型训练中常见的一些技巧,可以加速训练,同时让loss曲线更平滑。
                    # ==================================================================================================
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # 反向推理(Backward)
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                # 通过梯度累积来实现更大的batch size
                # ======================================================================================================
                # 主要逻辑是：
                #   不是每次迭代就更新权重,而是累积一定步数的梯度后再做更新。这在显存不足以支持非常大的batch size时很有用。
                # 例如:
                #   每次迭代的batch size是b,但希望的batch size是B(B大于显卡容量), 设置self.accumulate = B // b,
                #   比如B是256,b是64,那么accumulate=4, 就可以每4次迭代累积一次梯度,来模拟B=256的batch size进行更新
                # 通过梯度累积来实现更大的batch size,是一种常用的工程技巧
                # ======================================================================================================
                if ni - last_opt_step >= self.accumulate:  # self.accumulate: 累积梯度的次数,默认为1
                    self.optimizer_step()  # 进行优化器更新,利用累积的梯度来更新权重
                    last_opt_step = ni  # 更新last_opt_step = ni,代表已优化过

                # 日志记录Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                # 执行batch训练结束回调
                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            # 按预定策略调整学习率,帮助模型训练
            self.scheduler.step()

            # 执行epoch训练结束回调
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                # 使用了EMA(指数移动平均)来保存模型的一些属性,其中主要作用是保存模型的yaml配置
                # ======================================================================================================
                # 用EMA更新模型的一些属性值,这些属性在验证和保存模型时会用到
                # 主要有:
                # - yaml: 模型的配置文件
                # - nc: 类别数
                # - args: 模型的训练参数
                # - names: 类别名称
                # - stride: 模型的各层步幅
                # - class_weights: 各类别的 LOSS 权重
                # 之所以要更新这些属性,是因为训练中我们可能会修改模型结构,如更改类别数,这会改变模型实例的 nc 值。
                # 但是我们希望保存的模型和当前模型结构尽可能一致。通过 EMA 更新这些属性值,可以保持保存的模型配置与当前模型一致。
                # ======================================================================================================
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])

                # 计算是否是最后一个epoch
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    # 进行模型验证，得到模型在验证集的指标和fitness，将训练过程的loss等信息保存到self.metrics中
                    self.metrics, self.fitness = self.validate()
                # 保存训练指标到CSV文件里
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                # 使用early stopping判断是否需要停止训练：根据模型的fitness信息,early stopper会判断是否需要stop
                self.stop = self.stopper(epoch + 1, self.fitness)

                # 保存模型
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # 使用best.pt做最后一次验证
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()  # 绘制指标
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        """Save model checkpoints based on various conditions."""
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half(),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Use dill (if exists) to serialize the lambda functions where pickle does not do this
        try:
            import dill as pickle
        except ImportError:
            import pickle

        # Save last, best and delete
        torch.save(ckpt, self.last, pickle_module=pickle)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best, pickle_module=pickle)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt', pickle_module=pickle)
        del ckpt

    @staticmethod
    def get_dataset(data):
        """
        Get train, val path from data dict if it exists. Returns None if data format is not recognized.
        """
        return data['train'], data.get('val') or data.get('test')

    def setup_model(self):
        """
        load/create/download model for any task.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, weights = self.model, None
        ckpt = None
        if str(model).endswith('.pt'):
            weights, ckpt = attempt_load_one_weight(model)
            cfg = ckpt['model'].yaml
        else:
            cfg = model
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """
        Allows custom preprocessing model inputs and ground truths depending on task type.
        """
        return batch

    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop('fitness', -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Returns a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError('get_validator function not implemented in trainer')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """
        Returns dataloader derived from torch.data.Dataloader.
        """
        raise NotImplementedError('get_dataloader function not implemented in trainer')

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build dataset"""
        raise NotImplementedError('build_dataset function not implemented in trainer')

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        return {'loss': loss_items} if loss_items is not None else ['loss']

    def set_model_attributes(self):
        """
        To set or update model parameters before training.
        """
        self.model.names = self.data['names']

    def build_targets(self, preds, targets):
        """Builds target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Returns a string describing training progress."""
        return ''

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plots training samples during YOLOv5 training."""
        pass

    def plot_training_labels(self):
        """Plots training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Saves training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1  # number of cols
        s = '' if self.csv.exists() else (('%23s,' * n % tuple(['epoch'] + keys)).rstrip(',') + '\n')  # header
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([self.epoch] + vals)).rstrip(',') + '\n')

    def plot_metrics(self):
        """Plot and display metrics visually."""
        pass

    def on_plot(self, name, data=None):
        """Registers plots (e.g. to be consumed in callbacks)"""
        self.plots[name] = {'data': data, 'timestamp': time.time()}

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f'\nValidating {f}...')
                    self.metrics = self.validator(model=f)
                    self.metrics.pop('fitness', None)
                    self.run_callbacks('on_fit_epoch_end')

    def check_resume(self):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = attempt_load_weights(last).args
                if not Path(ckpt_args['data']).exists():
                    ckpt_args['data'] = self.args.data

                self.args = get_cfg(ckpt_args)
                self.args.model, resume = str(last), True  # reinstate
            except Exception as e:
                raise FileNotFoundError('Resume checkpoint not found. Please pass a valid checkpoint to resume from, '
                                        "i.e. 'yolo train resume model=path/to/last.pt'") from e
        self.resume = resume

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None:
            return
        best_fitness = 0.0
        start_epoch = ckpt['epoch'] + 1
        if ckpt['optimizer'] is not None:
            self.optimizer.load_state_dict(ckpt['optimizer'])  # optimizer
            best_fitness = ckpt['best_fitness']
        if self.ema and ckpt.get('ema'):
            self.ema.ema.load_state_dict(ckpt['ema'].float().state_dict())  # EMA
            self.ema.updates = ckpt['updates']
        if self.resume:
            assert start_epoch > 0, \
                f'{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n' \
                f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
            LOGGER.info(
                f'Resuming training from {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs')
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs.")
            self.epochs += ckpt['epoch']  # finetune additional epochs
        self.best_fitness = best_fitness
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            LOGGER.info('Closing dataloader mosaic')
            if hasattr(self.train_loader.dataset, 'mosaic'):
                self.train_loader.dataset.mosaic = False
            if hasattr(self.train_loader.dataset, 'close_mosaic'):
                self.train_loader.dataset.close_mosaic(hyp=self.args)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Constructs an optimizer for the given model, based on the specified optimizer name, learning rate,
        momentum, weight decay, and number of iterations.
        根据配置选择优化器类型和参数

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations. Default: 'auto'.
            lr (float, optional): The learning rate for the optimizer. Default: 0.001.
            momentum (float, optional): The momentum factor for the optimizer. Default: 0.9.
            decay (float, optional): The weight decay for the optimizer. Default: 1e-5.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'. Default: 1e5.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """

        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        if name == 'auto':
            nc = getattr(model, 'nc', 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ('SGD', 0.01, 0.9) if iterations > 10000 else ('AdamW', lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        if name in ('Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam'):
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == 'RMSProp':
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers "
                f'[Adam, AdamW, NAdam, RAdam, RMSProp, SGD, auto].'
                'To request support for addition optimizers please visit https://github.com/ultralytics/ultralytics.')

        optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f'{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)')
        return optimizer
