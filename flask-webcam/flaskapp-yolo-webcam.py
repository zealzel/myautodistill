from flask import Flask, Response, render_template

# from picamera2 import Picamera2
import time
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)
model = YOLO("../runs/detect/train9/weights/best_ncnn_model")
cap = cv2.VideoCapture(0)  # Use 0 for Mac's built-in camera or 1 for an external camera


def gen_frames():
    conf_threshold = 0.5  # Confidence threshold for detections
    conf_threshold_bottle = 0.3  # Confidence threshold for detections

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame, verbose=False)

        # # Draw detection results on the frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls = int(box.cls[0])
                if cls == 0 and conf > conf_threshold:  # cls == 0 indicates "hand"
                    label = f"Hand {conf:.2f}"
                    color = (0, 255, 0)
                elif (
                    cls == 1 and conf > conf_threshold_bottle
                ):  # cls == 1 indicates "bottle"
                    label = f"My Bottle {conf:.2f}"
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

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        jpg_bytes = buffer.tobytes()

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    # Video streaming route
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    # Main page
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
    # app.run(host="0.0.0.0", port=5000, debug=True)
