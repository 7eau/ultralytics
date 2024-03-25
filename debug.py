from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-cadh.yaml')
    model.train(data='shel5kv4.yaml', epochs=1, batch=16)

    # model = YOLO('yolov8n.pt')
    # model.train(data='shel5kv4.yaml', epochs=100, batch=16)