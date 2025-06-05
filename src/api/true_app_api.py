from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
from ultralytics import YOLO
import base64
import traceback
import torch
from fastapi.middleware.cors import CORSMiddleware
import requests
from io import BytesIO
from typing import Optional
from pydantic import BaseModel

# 定义URL请求模型
class URLRequest(BaseModel):
    image_url: str
    conf_threshold: float = 0.25

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化模型
model_path = r"D:\QQ文件\pre_best.pt"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

def download_image_from_url(url: str) -> np.ndarray:
    """从 URL 下载图片并返回 OpenCV 格式"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        img_data = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法解码下载的图片")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"下载图片失败: {str(e)}")

@app.post("/detect")
async def detect(
    file: Optional[UploadFile] = File(None),  # 可选：文件上传
    image_url: Optional[str] = Form(None),    # 可选：图片 URL (Form)
    conf_threshold: float = Form(0.25)        # 置信度阈值
):
    return await process_detection(file, image_url, conf_threshold)

@app.post("/detect/url")
async def detect_from_url(request: URLRequest):
    """专用于JSON请求中通过URL进行检测的端点"""
    return await process_detection(None, request.image_url, request.conf_threshold)

async def process_detection(file: Optional[UploadFile], image_url: Optional[str], conf_threshold: float):
    """处理检测逻辑的共用函数"""
    try:
        start_time = time.time()
        # 检查输入：必须提供 file 或 image_url，但不能同时提供
        if file is None and image_url is None:
            raise HTTPException(status_code=400, detail="必须提供 file 或 image_url")
        if file is not None and image_url is not None:
            raise HTTPException(status_code=400, detail="不能同时提供 file 和 image_url")
            
        # 方式1：从文件上传读取图片
        if file is not None:
            contents = await file.read()
            img_data = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=400, detail="无法解码上传的图片")
        # 方式2：从 URL 下载图片
        elif image_url is not None:
            img = download_image_from_url(image_url)
            
        # 推理
        results = model(img, conf=conf_threshold)
        detections = results[0]
        annotated_frame = detections.plot()
        if annotated_frame is None:
            raise HTTPException(status_code=500, detail="绘制检测结果失败")
            
        # 编码为 JPEG，再转 Base64
        success, buffer = cv2.imencode(".jpg", annotated_frame)
        if not success:
            raise HTTPException(status_code=500, detail="JPEG 编码失败")
        b64_str = base64.b64encode(buffer).decode("utf-8")
        
        # 获取检测结果详情
        detection_results = []
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = det
            detection_results.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": detections.names[int(class_id)]
            })
            
        end_time = time.time()
        return JSONResponse({
            "image_base64": b64_str,
            "detections": detection_results,
            "processing_time": f"{end_time - start_time:.2f}s",
            "device_used": device
        })
    except HTTPException as e:
        raise e  # 直接抛出 HTTP 异常
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")

