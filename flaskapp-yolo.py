from flask import Flask, Response, render_template
from picamera2 import Picamera2
import time
import cv2
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load the YOLO model
model = YOLO("runs/detect/train9/weights/best.pt")

# Define the video capture object
# cap = cv2.VideoCapture(0)  # Use 0 for Mac's built-in camera or 1 for an external camera

picam2 = Picamera2()
# config = picam2.create_preview_configuration({"size": (640, 480)})
config = picam2.create_preview_configuration({"size": (160, 120)})
# config = picam2.create_preview_configuration({"size": (320, 240)})
picam2.configure(config)
print("目前解析度：", config["main"]["size"])
picam2.start()
time.sleep(0.5)  # 預熱時間
# picam2.set_controls({"FrameDurationLimits": (66667, 66667)})
# picam2.set_controls({"FrameDurationLimits": (33333, 100000)})
# picam2.set_controls({"FrameRate": 15})


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


def gen_frames():
    global frame_count
    conf_threshold = 0.7  # Confidence threshold for detections
    target_interval = 1 / 15  # 目標每幀時間（秒）
    # target_interval = 1 / 5  # 目標每幀時間（秒）

    while True:
        start_time = time.time()
        # frame_count += 1
        # # 每4幀只處理1幀
        # # if frame_count % 4 != 0:
        # if frame_count % 10 != 0:
        #     continue
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
                if cls == 0 and conf > conf_threshold:  # cls == 0 indicates "hand"
                    label = f"Hand {conf:.2f}"
                    color = (0, 255, 0)
                elif cls == 1 and conf > conf_threshold:  # cls == 1 indicates "bottle"
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

        # 控制傳輸頻率
        # elapsed = time.time() - start_time
        # print("elapsed:", elapsed)
        # sleep_time = target_interval - elapsed
        # if sleep_time > 0:
        #     # print("sleep_time!", sleep_time)
        #     time.sleep(sleep_time)

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
