from flask import Flask, Response, render_template
import numpy as np
from picamera2 import Picamera2
import time
import cv2
from ultralytics import YOLO
import os
import colorsys

app = Flask(__name__)

PROJ = "proj1"
MODEL = "train9"
model = YOLO(f"../projects/{PROJ}/runs/detect/{MODEL}/weights/best_ncnn_model")

# Define the video capture object
# cap = cv2.VideoCapture(0)  # Use 0 for Mac's built-in camera or 1 for an external camera

picam2 = Picamera2()

modes = picam2.sensor_modes
mode = modes[0]
print("mode:", mode)
print("raw: ", picam2.camera_configuration()["raw"])

config = picam2.create_preview_configuration({"size": (1280, 960)})
# config = picam2.create_preview_configuration({"size": (640, 480)})
# config = picam2.create_preview_configuration({"size": (320, 240)})
# config = picam2.create_preview_configuration({"size": (160, 120)})
picam2.configure(config)
print("目前解析度：", config["main"]["size"])
# picam2.set_controls({"FrameDurationLimits": (66667, 66667)})
# picam2.set_controls({"FrameDurationLimits": (33333, 100000)})
picam2.set_controls({"FrameRate": 10})
picam2.start()
time.sleep(0.5)  # 預熱時間

metadata = picam2.capture_metadata()
frame_duration = metadata.get("FrameDuration", None)
if frame_duration:
    frame_rate = 1e6 / frame_duration  # 1,000,000 微秒除以每幀時間
    print("目前幀率：", frame_rate, "fps")
else:
    print("無法取得 FrameDuration 資訊")

# Define the directory to save captured frames
save_dir = "image-saved"
os.makedirs(save_dir, exist_ok=True)
frame_count = 0


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
    global frame_count
    conf_threshold = 0.8  # Confidence threshold for detections
    conf_threshold_bottle = 0.3  # Confidence threshold for detections
    target_interval = 1 / 15  # 目標每幀時間（秒）

    while True:
        try:
            # 捕獲影像 (返回 numpy array)
            frame = picam2.capture_array()
        except Exception as e:
            print("捕獲影像失敗:", e)
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Perform object detection
        results = model(frame, verbose=False)
        #
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
