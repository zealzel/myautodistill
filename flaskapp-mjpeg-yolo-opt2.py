import time
import cv2
import numpy as np
import threading
from flask import Flask, Response
from picamera2 import Picamera2
from ultralytics import YOLO

app = Flask(__name__)

# 全域變數用來保存原始與處理後的影格
raw_frame = None
processed_frame = None
frame_lock = threading.Lock()

# 載入 YOLO 模型（可考慮使用輕量化模型）
# model = YOLO("runs/detect/train9/weights/best.pt")
model = YOLO("best_ncnn_model")

# 初始化 Picamera2（使用 video configuration，較適合硬體加速）
picam2 = Picamera2()
config = picam2.create_video_configuration({"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(0.5)  # 預熱
print("Camera resolution:", config["main"]["size"])


def capture_thread():
    """持續捕獲影格，更新 raw_frame"""
    global raw_frame
    while True:
        try:
            frame = picam2.capture_array("main")  # Picamera2 預設回傳 RGB 影格
            with frame_lock:
                raw_frame = frame.copy()
        except Exception as e:
            print("Capture error:", e)
        # 控制捕獲頻率（根據實際效能調整）
        # time.sleep(0.01)
        time.sleep(0.02)


def detection_thread():
    """每隔 200 毫秒進行一次 YOLO 偵測，並更新 processed_frame"""
    global raw_frame, processed_frame
    while True:
        # 每 200 毫秒偵測一次
        # time.sleep(0.2)
        time.sleep(0.5)
        with frame_lock:
            if raw_frame is None:
                continue
            frame = raw_frame.copy()
        # 將 RGB 轉換為 BGR 供 OpenCV 使用
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 執行 YOLO 偵測
        results = model(frame_bgr, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                if cls == 0 and conf > 0.7:
                    label = f"Hand {conf:.2f}"
                    color = (0, 255, 0)
                elif cls == 1 and conf > 0.7:
                    label = f"Bottle {conf:.2f}"
                    color = (255, 0, 0)
                else:
                    continue
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame_bgr,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )
        with frame_lock:
            processed_frame = frame_bgr.copy()


def generate_frames():
    """串流產生器，送出最新的處理後影格，若尚無處理結果則傳送原始影格"""
    global processed_frame, raw_frame
    while True:
        with frame_lock:
            if processed_frame is not None:
                frame_to_send = processed_frame.copy()
            elif raw_frame is not None:
                frame_to_send = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            else:
                continue
        ret, buffer = cv2.imencode(".jpg", frame_to_send)
        if not ret:
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    # 簡單 HTML 頁面
    return """
    <!DOCTYPE html>
    <html lang="zh-Hant">
    <head>
        <meta charset="UTF-8">
        <title>YOLO MJPEG Stream</title>
    </head>
    <body>
        <h1>YOLO MJPEG Camera Stream</h1>
        <img src="/video_feed" style="max-width:100%;">
    </body>
    </html>
    """


if __name__ == "__main__":
    # 啟動捕獲與偵測執行緒
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=detection_thread, daemon=True)
    t1.start()
    t2.start()

    try:
        # 關閉 reloader 避免重複初始化
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    finally:
        picam2.stop()

