import os
from flask import Flask, Response, render_template
import cv2
import time

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

# 開啟攝像頭
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("攝像頭打開失敗！")

app = Flask(__name__)

cap = cv2.VideoCapture(0)
time.sleep(0.5)


def gen_frames():
    target_fps = 10  # 設定目標幀率，例如10幀/秒
    frame_interval = 1.0 / target_fps  # 每幀之間的最小時間間隔
    while True:
        start_time = time.time()
        ret, frame = video_capture.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        scale_factor = 0.5
        frame = cv2.resize(
            frame, (int(width * scale_factor), int(height * scale_factor))
        )
        height_scaled, width_scaled = frame.shape[:2]
        # frame = cv2.resize(frame, (640, 480))

        # ret2, buffer = cv2.imencode(".jpg", frame)
        ret2, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        # 控制每幀輸出的時間，保持在目標幀率左右
        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    # Main page
    return render_template("index.html")


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=5000, debug=True)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
