from ultralytics import YOLO

# 假设你的yaml文件名为 'my_yolo11.yaml'
model = YOLO('../ultralytics/cfg/models/11/yolo11.yaml')
model = YOLO("../ultralytics/cfg/models/8/yolov8-MobileViT.yaml")

# 打印模型结构
print(model.model)