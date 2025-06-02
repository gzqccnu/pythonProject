import cv2
import numpy as np
from ultralytics import YOLO

# 加载图像和模型
img_path = r"F:\File\Camellia oleifera\Camellia_oleifera\test_ill.jpg"
img_rain_path = r'F:\File\Camellia oleifera\Camellia_oleifera_rainy\1_rainy.jpg'
model = YOLO(r"F:\Visual Studio Code Proj\VS Code Python\ultralytics-main\weights\best.pt")

# 运行推理
results = model(img_path, conf=0.25)
detections = results[0]

# 提取每个检测框的信息
for box in detections.boxes:
    cls_id = int(box.cls[0])  # 类别ID
    conf = float(box.conf[0])  # 置信度
    class_name = model.names[cls_id]  # 类别名称
    print(f"检测到: {class_name}, 置信度: {conf:.2f}")
