from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
from ultralytics import YOLO
import base64
import torch
import torch_npu  # 新增导入

app = FastAPI()

# 加载NPU模型
device = torch.device('npu', 0)
model_path = "fruit_best.pt"
#model_path = "best_epoch_weights.pth"
model = YOLO(model_path)
model.model.to(device)  # 映射到NPU

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf_threshold: float = 0.25):
    try:
        start_time = time.time()
        
        # 读取并解码图片
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(content={"error": "无法解码上传的图片"}, status_code=400)

        # 推理
        results = model(img, conf=conf_threshold)
        detections = results[0]
        annotated_frame = detections.plot()
        if annotated_frame is None:
            return JSONResponse(content={"error": "绘制检测结果失败"}, status_code=500)

        # 编码为 JPEG，再转 Base64
        success, buffer = cv2.imencode(".jpg", annotated_frame)
        if not success or buffer is None:
            return JSONResponse(content={"error": "JPEG 编码失败"}, status_code=500)
        b64_str = base64.b64encode(buffer).decode("utf-8")

        end_time = time.time()
        # 返回 JSON 包含 Base64 字符串和耗时
        return JSONResponse({
            "image_base64": b64_str,
            "processing_time": f"{end_time - start_time:.2f}s"
        })
    
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

