from flask import Flask, Response
import cv2
from picamera2 import Picamera2
from picamera2.previews import NullPreview
import time

app = Flask(__name__)


def gen_frames():
    # 初始化 Picamera2 並設定為 NullPreview
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": (640, 480)})
    # config = picam2.create_preview_configuration(
    # main={"size": (640, 480)},
    # preview=NullPreview()
    # )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)  # 預熱時間
    picam2.set_controls({"FrameDurationLimits": (66667, 66667)})

    print("目前解析度：", config["main"]["size"])
    metadata = picam2.capture_metadata()
    frame_duration = metadata.get("FrameDuration", None)
    if frame_duration:
        frame_rate = 1e6 / frame_duration  # 1,000,000 微秒除以每幀時間
        print("目前幀率：", frame_rate, "fps")
    else:
        print("無法取得 FrameDuration 資訊")

    try:
        while True:
            # 取得影像
            frame = picam2.capture_array()
            # 將影像編碼成 JPEG 格式

            # 將 RGB 轉換為 BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            jpg_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")
    finally:
        picam2.stop()


@app.route("/video_feed")
def video_feed():
    # 返回 MJPEG 串流
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # 綁定到 0.0.0.0 使其他機器也能訪問，port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
