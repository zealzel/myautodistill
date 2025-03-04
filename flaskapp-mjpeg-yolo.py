import io
import time
from threading import Condition
import numpy as np
import cv2
from flask import Flask, Response
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs.fileoutput import FileOutput
from ultralytics import YOLO

app = Flask(__name__)

# 載入 YOLO 模型
model = YOLO("runs/detect/train9/weights/best.pt")


# 自訂 StreamingOutput，僅繼承 io.BufferedIOBase（參考官方範例）
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        # 每次寫入時將整個 buf 當作一張完整的 JPEG 影格
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


# 初始化攝影機，使用預設的 video configuration（硬體加速通常採用 video 模式）
try:
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)  # 預熱時間
    print("目前解析度：", config["main"]["size"])
except Exception as e:
    print("Camera initialization error:", e)
    raise

# 建立 StreamingOutput 物件
output = StreamingOutput()
# 啟動 JpegEncoder，並以 FileOutput 將 encoder 輸出寫入 output
try:
    picam2.start_recording(JpegEncoder(), FileOutput(output))
except Exception as e:
    print("Encoder start error:", e)
    raise


def gen_frames():
    while True:
        with output.condition:
            output.condition.wait()  # 等待新的影格產生
            frame_bytes = output.frame
        if frame_bytes is None:
            continue

        # 將 JPEG bytes 解碼為影像陣列
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # 執行 YOLO 偵測
        results = model(frame, verbose=False)
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
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )

        # 重新編碼處理後的影像
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        jpg_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")
        time.sleep(0.01)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    # 簡單 HTML 頁面，用於顯示影像串流
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
    try:
        # 關閉 reloader 避免重複初始化攝影機
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    finally:
        picam2.stop_recording()
        picam2.stop()

