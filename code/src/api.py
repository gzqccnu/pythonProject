from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import time
from ultralytics import YOLO
import base64
import traceback
from typing import List, Dict

app = FastAPI()

# 初始化模型 (在应用启动时加载一次)
try:
    model_path = r"F:\Visual Studio Code Proj\VS Code Python\ultralytics-main\weights\best.pt"
    model = YOLO(model_path)
except Exception as e:
    print(f"模型加载失败: {e}")
    model = None

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf_threshold: float = Query(0.25, description="置信度阈值"),
    return_image: bool = Query(True, description="是否返回标注后的 Base64 编码图片"),
):
    if model is None:
        return JSONResponse(content={"error": "模型未加载成功"}, status_code=500)

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

        response_data = {"processing_time": f"{time.time() - start_time:.2f}s"}

        if return_image:
            annotated_frame = detections.plot()
            if annotated_frame is None:
                return JSONResponse(content={"error": "绘制检测结果失败"}, status_code=500)

            # 编码为 JPEG，再转 Base64
            success, buffer = cv2.imencode(".jpg", annotated_frame)
            if not success or buffer is None:
                return JSONResponse(content={"error": "JPEG 编码失败"}, status_code=500)
            b64_str = base64.b64encode(buffer).decode("utf-8")
            response_data["image_base64"] = b64_str
        else:
            # 只返回检测框和标签
            detection_results: List[Dict[str, any]] = []
            for *xyxy, conf, cls in detections.xyxy:
                detection_results.append({
                    "bbox": [round(x.item()) for x in xyxy],
                    "confidence": round(conf.item(), 3),
                    "class_id": int(cls.item()),
                    "class_name": detections.names[int(cls.item())]
                })
            response_data["detections"] = detection_results

        return JSONResponse(content=response_data)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)