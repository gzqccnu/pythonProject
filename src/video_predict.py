import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

def main():
    try:
        # 设置模型路径 - 提供默认模型和备选方案
        model_path = r"../yolov8n.pt"  # 替换为你的模型路径
        if not os.path.exists(model_path):
            print(f"警告: 找不到模型 {model_path}")
            print("尝试下载官方预训练模型 yolov8n.pt...")
            model_path = "yolov8n.pt"  # 使用官方预训练模型作为备选

        # 加载YOLO模型
        print(f"正在加载模型: {model_path}")
        model = YOLO(model_path)
        print("模型加载成功!")

        # 打开摄像头
        print("正在打开摄像头...")
        cap = cv2.VideoCapture(0)
        
        # 检查摄像头是否成功打开
        if not cap.isOpened():
            raise Exception("无法打开摄像头，请检查设备连接")
        
        print("摄像头已打开。按 'q' 退出，按 's' 保存当前帧")
        
        # 初始化FPS计算
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        # 设置检测置信度阈值
        conf_threshold = 0.25
        
        while True:
            # 读取一帧
            success, frame = cap.read()
            
            if not success:
                print("无法读取视频帧，退出...")
                break
                
            # 每秒更新FPS
            frame_count += 1
            if frame_count >= 10:  # 每10帧更新一次FPS
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # 运行YOLO推理
            results = model(frame, conf=conf_threshold)
            
            # 获取检测结果
            detections = results[0]
            
            # 在原始帧上绘制结果
            annotated_frame = detections.plot()
            
            # 显示FPS
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示检测到的对象数量
            num_objects = len(detections.boxes)
            cv2.putText(annotated_frame, f"检测到: {num_objects} 个对象", (20, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示处理后的帧
            cv2.imshow("YOLO 检测", annotated_frame)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("用户退出")
                break
            elif key == ord("s"):
                # 保存当前帧
                save_path = f"detection_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(save_path, annotated_frame)
                print(f"图像已保存至: {save_path}")
            elif key == ord("+") or key == ord("="):
                # 提高置信度阈值
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"置信度阈值: {conf_threshold:.2f}")
            elif key == ord("-"):
                # 降低置信度阈值
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"置信度阈值: {conf_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 释放资源
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("程序已结束")

if __name__ == "__main__":
    main()