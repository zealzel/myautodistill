from flask import Flask, Response, render_template
import cv2
import time
from picamera2 import Picamera2

app = Flask(__name__)

picam2 = Picamera2()
config = picam2.create_preview_configuration({"size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(0.5)


def gen_frames():
    while True:
        frame = picam2.capture_array()
        # Picamera2 預設回傳 RGB
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode(".jpg", frame_bgr)
        if not ret:
            continue
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


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
