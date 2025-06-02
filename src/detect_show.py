import cv2
import numpy as np
from ultralytics import YOLO
import torch

# 图像路径和模型路径
img_path = r"F:\dataset\dataset\train\images\73.jpg"
img_rain_path = r'F:\File\Camellia oleifera\Camellia_oleifera_rainy\1_rainy.jpg'
ill_path = r'F:\File\Camellia oleifera\dataset\病虫害\炭疽病.jpg'
model_path = r"F:\Visual Studio Code Proj\VS Code Python\ultralytics-main\weights\best.pt"
better_path = r'D:\QQ文件\pre_best.pt'

# 检查CUDA是否可用并设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True  # 启用cudnn加速

# 加载模型
model = YOLO(better_path).to(device)  # 将模型移动到GPU或CPU

# 加载图像并推理
results = model(img_path, conf=0.25, device=device)
detections = results[0].cpu()

# 读取原始图像用于绘制
img = cv2.imread(img_path)

# 遍历每个检测结果，绘制框和文本
for box in detections.boxes:
    # 坐标
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # 类别和置信度
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    class_name = model.names[cls_id]

    # 绘制边框
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 绘制类别名称和置信度
    label = f"{class_name}: {conf:.2f}"
    cv2.putText(img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 显示图像
cv2.imshow("检测结果", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite("F:/桌面/output.jpg", img)