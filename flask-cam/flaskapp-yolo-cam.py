from flask import Flask, Response, render_template
import numpy as np
import colorsys

# from picamera2 import Picamera2
import time
import cv2
from ultralytics import YOLO
import os

# PROJ = "test01"
# MODEL = "train"
PROJ = "proj1"
MODEL = "train9"

app = Flask(__name__)
# model = YOLO("../runs/detect/train9/weights/best_ncnn_model")
model = YOLO(f"../projects/{PROJ}/runs/detect/{MODEL}/weights/best_ncnn_model")
cap = cv2.VideoCapture(0)  # Use 0 for Mac's built-in camera or 1 for an external camera


def get_classes(classes_path=f"../projects/{PROJ}/dataset/yolov3/classes.txt"):
    with open(classes_path, "r") as f:
        classes = f.read().splitlines()
    return classes


def generate_colors(n):
    """為每個類別生成不同的顏色

    Args:
        n: 類別數量

    Returns:
        list: BGR 顏色列表，每個顏色為 (B,G,R) tuple
    """
    colors = []
    for i in range(n):
        # 使用 HSV 色彩空間來生成顏色，確保顏色夠分散
        hue = i / n
        sat = 0.9 + np.random.random() * 0.1  # 90-100% 飽和度
        val = 0.9 + np.random.random() * 0.1  # 90-100% 亮度

        # 轉換 HSV 到 RGB
        rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(hue, sat, val))
        # 轉換為 BGR (OpenCV 使用 BGR)
        bgr = (rgb[2], rgb[1], rgb[0])
        colors.append(bgr)

    return colors


def put_text(frame, text, x1, y1, conf, color):
    """在框上方顯示文字

    Args:
        frame: 影像幀
        text: 要顯示的文字
        x1, y1: 文字位置
        conf: 信心度
        color: BGR 顏色元組
    """
    cv2.putText(
        frame,
        f"{text} {conf:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,  # 1.0,
        color,
        4,  # 2,
    )
    print(f"{text}: {conf:.2f}")


classes = get_classes()
colors = generate_colors(len(classes))
print("classes", classes)


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

                if conf > conf_threshold:
                    color = colors[cls]  # 使用對應類別的顏色
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    put_text(frame, classes[cls], x1, y1, conf, color)
                    # def put_text(text, x1, y1, conf):

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
