# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DetectH, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment, CoordAtt, StemBlock)
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.utils.loss import v8ClassificationLoss, v8DetectionLoss, v8PoseLoss, v8SegmentationLoss
from ultralytics.utils.plotting import feature_visualization
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync)

try:
    import thop
except ImportError:
    thop = None


class BaseModel(nn.Module):
    """
    YOLO模型的基类,定义了模型的基本接口:前向传播forward,损失计算loss,预测predict等 \n
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x, *args, **kwargs):
        """
        模型前向传播的主入口,输入为图像或图像+标签,返回预测或损失 \n
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor | dict): The input image tensor or a dict including image tensor and gt labels.

        Returns:
            (torch.Tensor): The output of the network.
        """
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x, profile=False, visualize=False, augment=False):
        """
        仅进行前向推理预测,可选择是否profiling和可视化 \n
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): 模型的输入tensor.
            profile (bool):  如果为True打印每层的计算时间，默认为False.
            visualize (bool): 如果为True,则保存模型的feature maps， 默认为False.
            augment (bool): 如果为True,则预测期间保持图像增强，默认为False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        if augment:
            return self._predict_augment(x)
        return self._predict_once(x, profile, visualize)

    def _predict_once(self, x, profile=False, visualize=False):
        """
        单尺度图像的前向推理 \n
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model.
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(f'WARNING ⚠️ {self.__class__.__name__} does not support augmented inference yet. '
                       f'Reverting to single-scale inference instead.')
        """
        多尺度图像的前向推理,默认不实现 \n
        Perform augmentations on input image x and return augmented inference.
        """
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)

    def _profile_one_layer(self, m, x, dt):
        """
        测量并打印某一层的计算时间 \n
        Profile the computation time and FLOPs of a single layer of the model on a given input.
        Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1] and isinstance(x, list)  # is final layer list, copy input as inplace fix
        flops = thop.profile(m, inputs=[x.copy() if c else x], verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {flops:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self, verbose=True):
        """
        将BN和卷积层融合可以加速推理 \n
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the
        computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        if not self.is_fused():
            for m in self.model.modules():
                if isinstance(m, (Conv, Conv2, DWConv)) and hasattr(m, 'bn'):
                    if isinstance(m, Conv2):
                        m.fuse_convs()
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, ConvTranspose) and hasattr(m, 'bn'):
                    m.conv_transpose = fuse_deconv_and_bn(m.conv_transpose, m.bn)
                    delattr(m, 'bn')  # remove batchnorm
                    m.forward = m.forward_fuse  # update forward
                if isinstance(m, RepConv):
                    m.fuse_convs()
                    m.forward = m.forward_fuse  # update forward
            self.info(verbose=verbose)

        return self

    def is_fused(self, thresh=10):
        """
        判断模型是否已经融合。 \n
        Check if the model has less than a certain threshold of BatchNorm layers.

        Args:
            thresh (int, optional): The threshold number of BatchNorm layers. Default is 10.

        Returns:
            (bool): True if the number of BatchNorm layers in the model is less than the threshold, False otherwise.
        """
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        return sum(isinstance(v, bn) for v in self.modules()) < thresh  # True if < 'thresh' BatchNorm layers in model

    def info(self, detailed=False, verbose=True, imgsz=640):
        """
        打印模型信息 \n
        Prints model information

        Args:
            detailed (bool): if True, prints out detailed information about the model. Defaults to False
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.
        在模型的所有参数和缓冲区上应用一个函数 \n
        `_apply()` is a function that applies a function to all the tensors in the model that are not
        parameters or registered buffers

        Args:
            fn (function): the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DetectH, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights, verbose=True):
        """
        从权重文件加载预训练模型 \n
        Load the weights into the model.

        Args:
            weights (dict | torch.nn.Module): The pre-trained weights to be loaded.
            verbose (bool, optional): Whether to log the transfer progress. Defaults to True.
        """
        model = weights['model'] if isinstance(weights, dict) else weights  # torchvision models are not dicts
        csd = model.float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, self.state_dict())  # intersect
        self.load_state_dict(csd, strict=False)  # load
        if verbose:
            LOGGER.info(f'Transferred {len(csd)}/{len(self.model.state_dict())} items from pretrained weights')

    def loss(self, batch, preds=None):
        """计算损失函数,需要在子类中实现criterion \n
        Compute loss

        Args:
            batch (dict): Batch to compute loss on
            preds (torch.Tensor | List[torch.Tensor]): Predictions.
        """
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        preds = self.forward(batch['img']) if preds is None else preds
        return self.criterion(preds, batch)

    def init_criterion(self):
        raise NotImplementedError('compute_loss() needs to be implemented by task heads')


class DetectionModel(BaseModel):
    """YOLOv8 detection model."""

    def __init__(self, cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):  # model, input channels, number of classes
        super().__init__()
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # 获取配置文件dict

        # 定义和构建模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # 获取输入通道数，如果配置文件中未配置，则默认是3
        # 以数据集中定义的类别数为准，如果数据集中的类别数和配置文件中的类别数不一致，将重写配置文件中的类别数
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 构建模型，获取model及savelist
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # 模型配置文件中默认的各类别名称列表
        # inplace是PyTorch的一个参数,决定了某些操作(如激活函数)是否进行inplace操作
        # ======================================================
        # inplace=True意味着会对激活函数的输入进行覆盖从而节省内存。
        # inplace=False意味着不覆盖输入张量而生成新的输出。
        # inplace往往在考虑内存占用与速度之间的tradeoff
        # ======================================================
        self.inplace = self.yaml.get('inplace', True)

        # 计算网络的下采样比例stride,并设置到最后的预测层(如Detect)中
        # 1. 取出网络的最后一层,假设是检测预测层Detect()
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, DetectH, Segment, Pose)):
            # 2. 设置初始下采样步幅s = 256, 是anchor设计的常用基础尺寸
            s = 256  # 2x min stride
            # 3. 将self.inplace赋值给预测层的inplace
            m.inplace = self.inplace
            # 4. 定义前向函数forward,根据网络类型不同稍有变化
            forward = lambda x: self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            # 5. 传入全0输入,获取网络各层输出的shape, 计算各层shape比输入shape的下采样比例,即stride,并赋值给预测层模块m
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # 6. 将算得的stride赋值给网络self
            self.stride = m.stride
            # 7. 调用bias_init()初始化预测层偏置
            m.bias_init()  # only run once
        else:  # 如果最后一层不是预测层,则直接设置默认步幅为32
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # 初始化 weights, biases
        initialize_weights(self)

        # 输出模型的结构和参数信息
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """
        实现测试时增强推理(augmented inference)
        Perform augmentations on input image x and return augmented inference and train outputs.
        """
        # ==========================================================
        # 1. 对于输入图片,根据不同尺度s和翻转方向f,做多尺度和翻转增强。
        # 2. 对每种增强后的图片,都进行预测,得到增强后预测框。
        # 3. 通过_descale_pred()方法对预测框进行反尺度变换和反翻转。
        # 4. 通过_clip_augmented()方法裁剪预测框,只保留有效区域。
        # 5. 将所有增强结果预测框堆叠起来,最终实现测试时的多尺度和翻转推理。
        #
        # 增强推理可以提升小目标的检测效果,是YOLO框架的一大特点
        # ==========================================================
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = super().predict(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, -1), None  # augmented inference, train

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """
        对预测框进行反尺度变换和反翻转
        De-scale predictions following augmented inference (inverse operation).
        """
        p[:, :4] /= scale  # de-scale
        x, y, wh, cls = p.split((1, 1, 2, p.shape[dim] - 4), dim)
        if flips == 2:
            y = img_size[0] - y  # de-flip ud
        elif flips == 3:
            x = img_size[1] - x  # de-flip lr
        return torch.cat((x, y, wh, cls), dim)

    def _clip_augmented(self, y):
        """
        裁剪预测框,只保留有效区域
        Clip YOLOv5 augmented inference tails.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[-1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][..., :-i]  # large
        i = (y[-1].shape[-1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][..., i:]  # small
        return y

    def init_criterion(self):
        """
        初始化用于计算损失的criterion
        Returns: Criterion class for computing Detection training losses

        """
        return v8DetectionLoss(self)


class SegmentationModel(DetectionModel):
    """YOLOv8 segmentation model."""

    def __init__(self, cfg='yolov8n-seg.yaml', ch=3, nc=None, verbose=True):
        """Initialize YOLOv8 segmentation model with given config and parameters."""
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8SegmentationLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)


class PoseModel(DetectionModel):
    """YOLOv8 pose model."""

    def __init__(self, cfg='yolov8n-pose.yaml', ch=3, nc=None, data_kpt_shape=(None, None), verbose=True):
        """Initialize YOLOv8 Pose model."""
        if not isinstance(cfg, dict):
            cfg = yaml_model_load(cfg)  # load model YAML
        if any(data_kpt_shape) and list(data_kpt_shape) != list(cfg['kpt_shape']):
            LOGGER.info(f"Overriding model.yaml kpt_shape={cfg['kpt_shape']} with kpt_shape={data_kpt_shape}")
            cfg['kpt_shape'] = data_kpt_shape
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        return v8PoseLoss(self)

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference."""
        LOGGER.warning(
            f'WARNING ⚠️ {self.__class__.__name__} has not supported augment inference yet! Now using single-scale inference instead.'
        )
        return self._predict_once(x)


class ClassificationModel(BaseModel):
    """YOLOv8 classification model."""

    def __init__(self,
                 cfg=None,
                 model=None,
                 ch=3,
                 nc=None,
                 cutoff=10,
                 verbose=True):  # yaml, model, channels, number of classes, cutoff index, verbose flag
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg, ch, nc, verbose)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Create a YOLOv5 classification model from a YOLOv5 detection model."""
        from ultralytics.nn.autobackend import AutoBackend
        if isinstance(model, AutoBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg, ch, nc, verbose):
        """Set YOLOv8 model configurations and define the model architecture."""
        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        elif not nc and not self.yaml.get('nc', None):
            raise ValueError('nc not specified. Must specify nc in model.yaml or function arguments.')
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=ch, verbose=verbose)  # model, savelist
        self.stride = torch.Tensor([1])  # no stride constraints
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.info()

    @staticmethod
    def reshape_outputs(model, nc):
        """Update a TorchVision classification model to class count 'n' if required."""
        name, m = list((model.model if hasattr(model, 'model') else model).named_children())[-1]  # last module
        if isinstance(m, Classify):  # YOLO Classify() head
            if m.linear.out_features != nc:
                m.linear = nn.Linear(m.linear.in_features, nc)
        elif isinstance(m, nn.Linear):  # ResNet, EfficientNet
            if m.out_features != nc:
                setattr(model, name, nn.Linear(m.in_features, nc))
        elif isinstance(m, nn.Sequential):
            types = [type(x) for x in m]
            if nn.Linear in types:
                i = types.index(nn.Linear)  # nn.Linear index
                if m[i].out_features != nc:
                    m[i] = nn.Linear(m[i].in_features, nc)
            elif nn.Conv2d in types:
                i = types.index(nn.Conv2d)  # nn.Conv2d index
                if m[i].out_channels != nc:
                    m[i] = nn.Conv2d(m[i].in_channels, nc, m[i].kernel_size, m[i].stride, bias=m[i].bias is not None)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        return v8ClassificationLoss()


class RTDETRDetectionModel(DetectionModel):

    def __init__(self, cfg='rtdetr-l.yaml', ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self):
        """Compute the classification loss between predictions and true labels."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()

        img = batch['img']
        # NOTE: preprocess gt_bbox and gt_labels to list.
        bs = len(img)
        batch_idx = batch['batch_idx']
        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': batch['cls'].to(img.device, dtype=torch.long).view(-1),
            'bboxes': batch['bboxes'].to(device=img.device),
            'batch_idx': batch_idx.to(img.device, dtype=torch.long).view(-1),
            'gt_groups': gt_groups}

        preds = self.predict(img, batch=targets) if preds is None else preds
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        loss = self.criterion((dec_bboxes, dec_scores),
                              targets,
                              dn_bboxes=dn_bboxes,
                              dn_scores=dn_scores,
                              dn_meta=dn_meta)
        # NOTE: There are like 12 losses in RTDETR, backward with all losses but only show the main three losses.
        return sum(loss.values()), torch.as_tensor([loss[k].detach() for k in ['loss_giou', 'loss_class', 'loss_bbox']],
                                                   device=img.device)

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False
            batch (dict): A dict including gt boxes and labels from dataloader.

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model[:-1]:  # except the head part
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        head = self.model[-1]
        x = head([y[j] for j in head.f], batch)  # head inference
        return x


class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initialize an ensemble of models."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Function generates the YOLOv5 network's final layer."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 2)  # nms ensemble, y shape(B, HW, C)
        return y, None  # inference, train output


# Functions ------------------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def temporary_modules(modules=None):
    """
    Context manager for temporarily adding or modifying modules in Python's module cache (`sys.modules`).

    This function can be used to change the module paths during runtime. It's useful when refactoring code,
    where you've moved a module from one location to another, but you still want to support the old import
    paths for backwards compatibility.

    Args:
        modules (dict, optional): A dictionary mapping old module paths to new module paths.

    Example:
        ```python
        with temporary_modules({'old.module.path': 'new.module.path'}):
            import old.module.path  # this will now import new.module.path
        ```

    Note:
        The changes are only in effect inside the context manager and are undone once the context manager exits.
        Be aware that directly manipulating `sys.modules` can lead to unpredictable results, especially in larger
        applications or libraries. Use this function with caution.
    """
    if not modules:
        modules = {}

    import importlib
    import sys
    try:
        # Set modules in sys.modules under their old name
        for old, new in modules.items():
            sys.modules[old] = importlib.import_module(new)

        yield
    finally:
        # Remove the temporary module paths
        for old in modules:
            if old in sys.modules:
                del sys.modules[old]


def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    from ultralytics.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        with temporary_modules({
                'ultralytics.yolo.utils': 'ultralytics.utils',
                'ultralytics.yolo.v8': 'ultralytics.models.yolo',
                'ultralytics.yolo.data': 'ultralytics.data'}):  # for legacy 8.0 Classify and Pose models
            return torch.load(file, map_location='cpu'), file  # load

    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'models':
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        return torch.load(file, map_location='cpu'), file  # load


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    """Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a."""

    ensemble = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt, w = torch_safe_load(w)  # load ckpt
        args = {**DEFAULT_CFG_DICT, **ckpt['train_args']} if 'train_args' in ckpt else None  # combined args
        model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

        # Model compatibility updates
        model.args = args  # attach args to model
        model.pt_path = w  # attach *.pt file path to model
        model.task = guess_model_task(model)
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.])

        # Append
        ensemble.append(model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval())  # model in eval mode

    # Module compatibility updates
    for m in ensemble.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, DetectH, Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(ensemble) == 1:
        return ensemble[-1]

    # Return ensemble
    LOGGER.info(f'Ensemble created with {weights}\n')
    for k in 'names', 'nc', 'yaml':
        setattr(ensemble, k, getattr(ensemble[0], k))
    ensemble.stride = ensemble[torch.argmax(torch.tensor([m.stride.max() for m in ensemble])).int()].stride
    assert all(ensemble[0].nc == m.nc for m in ensemble), f'Models differ in class counts {[m.nc for m in ensemble]}'
    return ensemble


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    """Loads a single model weights."""
    ckpt, weight = torch_safe_load(weight)  # load ckpt
    args = {**DEFAULT_CFG_DICT, **(ckpt.get('train_args', {}))}  # combine model and default args, preferring model args
    model = (ckpt.get('ema') or ckpt['model']).to(device).float()  # FP32 model

    # Model compatibility updates
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # attach args to model
    model.pt_path = weight  # attach *.pt file path to model
    model.task = guess_model_task(model)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.])

    model = model.fuse().eval() if fuse and hasattr(model, 'fuse') else model.eval()  # model in eval mode

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Segment):
            m.inplace = inplace  # torch 1.7.0 compatibility
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model and ckpt
    return model, ckpt


def parse_model(d, ch, verbose=True):  # model_dict, input_channels(3)
    """
    将YOLO的模型配置字典解析成一个PyTorch模块网络
    Parse a YOLO model.yaml dictionary into a PyTorch model.
    """
    import ast

    # 从模型中获取需要的参数
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))

    # 从模型配置字典中解析出该模型的尺度(scale)信息,然后根据尺度设置模型的深度(depth)、宽度(width)和最大通道数(max_channels)
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    # 从模型配置字典中解析activation函数的类型, 并重新定义Conv模块的默认activation
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    # 打印日志
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

    # 这段代码的主要目的是基于配置字典,递归地构建每个模块及其参数,并连接到Sequential中
    # ==================================================================================================================
    # f: from索引, n: 重复数, m: 模型类型, args: 参数
    # 具体来看:
    # 1. 从backbone和head中遍历每个模块的配置信息,包括from索引f、重复数n、模块类型m和参数args。
    # 2. 根据m创建实际的模块对象,如果m的类型在nn中就直接创建,否则从全局命名空间获取。
    # 3. 解析args中的字符串为实际变量或表达式。
    # 4. 计算重复数n_。
    # 5. 根据模块类型调整输入输出通道:
    #   - 对于大多数模块,设置输入输出通道为ch[f]、ch[f+1],还可能插入n_。
    #   - 对BN、Concat特殊处理输入通道。
    #   - 对Detect等设置f。
    # ==================================================================================================================
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 1. 从配置字典中获取模块类名,并将字符串转换成实际的模块类
        # ====================================================================
        # 1.1. 从配置字典中取出字符串形式的模块类名m
        # 1.2. 判断m是否以'nn.'开头,即是否是PyTorch内置的模块。
        #   如果是,则用getattr()根据模块路径获取模块类,例如'nn.Conv2d'。如果不是,
        #   则说明是自定义的模块类,位于全局命名空间globals()中。
        # 1.3. 利用globals()可以根据字符串KEY取出实际的类,例如'Focus'。
        # 1.4. m[3:]是为了去掉开头的'nn.'或其他标识。
        # 1.5. 最后将字符串转换成了实际的类对象,保存在m中。
        # ====================================================================
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]

        # 2. 解析模块的参数字符串,将其转换为实际的变量或表达式
        # =============================================================
        # 2.1. 遍历模块参数args
        # 2.2. 判断每个参数a是否为字符串类型
        # 2.3. 如果是,try转换:
        #   2.3.1 如果a在locals()中,表示它是参数名,直接取值
        #   2.3.2 如果不在,使用ast.literal_eval执行字符串表达式,转换成变量
        # 2.4. 将转换后的参数重新赋值给args。
        # =============================================================
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        # 3. 根据depth_multiple来计算每个模块的重复次数n_
        n = n_ = max(round(n * depth), 1) if n > 1 else n

        # 4. 根据模块类型调整输入输出通道:
        # ============================================================
        #   - 对于大多数模块,设置输入输出通道为ch[f]、ch[f+1],还可能插入n_
        #   - 对BN、Concat特殊处理输入通道。
        #   - 对Detect等设置f。
        # ============================================================
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                 BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3,
                 StemBlock, CoordAtt):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, DetectH, Segment, Pose):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        elif m is RTDETRDecoder:  # special case, channels arg must be passed in index 1
            args.insert(1, [ch[x] for x in f])
        else:
            c2 = ch[f]

        # 5. 根据重复次数n和参数args构建模块m的重复容器m_
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)

        # 6. 计算和保存模块m相关信息
        t = str(m)[8:-2].replace('__main__.', '')  # 从模块m的字符串表示中提取出模块类型
        m.np = sum(x.numel() for x in m_.parameters())  # 计算模块m中的可训练参数数量
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type

        # 7. 打印日志
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print

        # 8. 保存模块
        # 构建模型可训练参数的索引表savelist，为了可以方便地实现参数选择性冻结
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
        # 追加模型到layers中
        layers.append(m_)
        # 更新并保存网络每个模块的输出通道数
        if i == 0:
            ch = []
        ch.append(c2)  # 更新输出通道
    return nn.Sequential(*layers), sorted(save)  # 返回Sequential模型和savelist


def yaml_model_load(path):
    """
    从 yaml 配置文件中加载一个YOLOv8模型
    Load a YOLOv8 model from a YAML file.
    """
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_name(new_stem + path.suffix)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d


def guess_model_scale(model_path):
    """
    从YOLO模型的配置文件路径中猜测其模型尺度大小(scale)\n
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str | Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''  # 如果没有找到,返回空字符串


def guess_model_task(model):
    """
    猜测一个PyTorch模型或配置字典的任务类型(task),即检测、分割、分类等
    Guess the task of a PyTorch model from its architecture or configuration.

    Args:
        model (nn.Module | dict): PyTorch model or model configuration in YAML format.

    Returns:
        (str): Task of the model ('detect', 'segment', 'classify', 'pose').

    Raises:
        SyntaxError: If the task of the model could not be determined.
    """

    def cfg2task(cfg):
        """Guess from YAML dictionary."""
        m = cfg['head'][-1][-2].lower()  # output module name
        if m in ('classify', 'classifier', 'cls', 'fc'):
            return 'classify'
        if m == 'detect':
            return 'detect'
        if m == 'segment':
            return 'segment'
        if m == 'pose':
            return 'pose'

    # Guess from model cfg
    if isinstance(model, dict):  # 如果输入是配置字典,根据'head'中的最后一层解析task
        with contextlib.suppress(Exception):
            return cfg2task(model)

    # Guess from PyTorch model
    if isinstance(model, nn.Module):  # 如果是PyTorch模型,遍历模块判断类型,如果有Detect/Segment等模块则解析任务类型
        # 尝试解析模型的yaml文件或args参数来获取任务类型
        for x in 'model.args', 'model.model.args', 'model.model.model.args':
            with contextlib.suppress(Exception):
                return eval(x)['task']
        for x in 'model.yaml', 'model.model.yaml', 'model.model.model.yaml':
            with contextlib.suppress(Exception):
                return cfg2task(eval(x))

        for m in model.modules():
            if isinstance(m, (Detect, DetectH)):
                return 'detect'
            elif isinstance(m, Segment):
                return 'segment'
            elif isinstance(m, Classify):
                return 'classify'
            elif isinstance(m, Pose):
                return 'pose'

    # Guess from model filename
    if isinstance(model, (str, Path)):  # 如果模型路径中包含关键字如'detect',也可判断任务类型
        model = Path(model)
        if '-seg' in model.stem or 'segment' in model.parts:
            return 'segment'
        elif '-cls' in model.stem or 'classify' in model.parts:
            return 'classify'
        elif '-pose' in model.stem or 'pose' in model.parts:
            return 'pose'
        elif 'detect' in model.parts:
            return 'detect'

    # Unable to determine task from model
    LOGGER.warning("WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. "
                   "Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.")
    return 'detect'  # 如果无法判断，默认为 detect
