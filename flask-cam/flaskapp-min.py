from flask import Flask, Response, render_template
import cv2
import time

# 開啟攝像頭
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("攝像頭打開失敗！")

app = Flask(__name__)

cap = cv2.VideoCapture(0)
time.sleep(0.5)


def gen_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        ret2, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


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
