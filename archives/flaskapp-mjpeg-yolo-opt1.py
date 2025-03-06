import io
import time
import numpy as np
import cv2
from threading import Condition
from flask import Flask, Response, render_template
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs.fileoutput import FileOutput
from ultralytics import YOLO

app = Flask(__name__)

# 載入 YOLO 模型
model = YOLO("runs/detect/train9/weights/best.pt")


# 自訂 StreamingOutput 類別，只繼承 io.BufferedIOBase
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


# 全域初始化 Picamera2，使用預設 video configuration
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

# 建立 StreamingOutput 與 JpegEncoder，並啟動 encoder（硬體加速）
output = StreamingOutput()
encoder = JpegEncoder()
try:
    picam2.start_recording(JpegEncoder(), FileOutput(output))
except Exception as e:
    print("Encoder start error:", e)
    raise


def gen_frames():
    frame_count = 0
    while True:
        with output.condition:
            output.condition.wait()  # 等待新影格產生
            frame_bytes = output.frame
        if frame_bytes is None:
            continue

        # 解碼 JPEG 影格成 numpy 陣列
        np_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame_count += 1
        # 每5幀進行一次 YOLO 偵測，其餘幀直接傳輸原始影像
        if frame_count % 5 == 0:
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
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )

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
    # 簡單 HTML 頁面，請確保 templates/index.html 中有 <img src="/video_feed">
    return render_template("index.html")


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    finally:
        picam2.stop_recording()
        picam2.stop()

