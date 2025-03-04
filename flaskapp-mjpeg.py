import io
import time
from threading import Condition
from flask import Flask, Response
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs.fileoutput import FileOutput

app = Flask(__name__)

PAGE = """\
<html>
<head>
  <title>Picamera2 MJPEG Streaming Demo</title>
</head>
<body>
  <h1>Picamera2 MJPEG Streaming Demo</h1>
  <img src="/stream.mjpg" width="640" height="480" />
</body>
</html>
"""


# 自訂 StreamingOutput 類別（簡單捕獲 encoder 輸出的完整 JPEG 影格）
class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        # 當收到完整的 JPEG 影格資料（假設 encoder 每次寫入的是一整張圖片）
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
        return len(buf)


# 初始化攝影機並設定 video configuration
try:
    picam2 = Picamera2()
    # 以預設的 video configuration 並設定主要影像流解析度為 640x480
    config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)  # 預熱時間
except Exception as e:
    print("Camera initialization error:", e)
    raise

# 建立 StreamingOutput 物件
output = StreamingOutput()
# 使用 JpegEncoder（硬體加速）與 FileOutput(output)
picam2.start_recording(JpegEncoder(), FileOutput(output))


@app.route("/")
def index():
    return PAGE


@app.route("/stream.mjpg")
def stream():
    def generate():
        while True:
            with output.condition:
                output.condition.wait()  # 等待新影格產生
                frame = output.frame
            if frame is None:
                continue
            yield (
                b"--FRAME\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + f"{len(frame)}".encode()
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=FRAME")


if __name__ == "__main__":
    try:
        # 建議關閉 reloader 避免重複初始化攝影機
        app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    finally:
        picam2.stop_recording()
        picam2.stop()

