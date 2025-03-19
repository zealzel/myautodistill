import sys
import os
import colorsys

# 將專案根目錄加入 sys.path
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
)

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from ultralytics import YOLO

PROJ = "proj1"
MODEL = "train9"
model = YOLO(f"../../projects/{PROJ}/runs/detect/{MODEL}/weights/best_ncnn_model")
# model = YOLO("../../runs/detect/train9/weights/best_ncnn_model")

app = Flask(__name__)
CORS(app)


def get_classes(classes_path=f"../../projects/{PROJ}/dataset/yolov3/classes.txt"):
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


@app.route("/detect", methods=["POST"])
def detect():
    data = request.get_json()
    image_data = data["image"]
    width = data["width"]
    height = data["height"]
    detected_objects = []

    # 解碼圖片
    if not image_data:
        return jsonify({"detectedObjects": detected_objects})

    image_bytes = base64.b64decode(image_data.split(",")[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 調整圖片大小
    img = cv2.resize(img, (width, height))

    if img is None:
        return jsonify({"error": "Image decoding failed"}), 400

    # 使用 YOLO 進行物體檢測
    results = model(img, verbose=False)

    # 設定信心度閾值
    conf_threshold = 0.5
    conf_threshold_bottle = 0.3

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            if conf > conf_threshold:
                color = colors[cls]  # 使用對應類別的顏色
                # cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # put_text(frame, classes[cls], x1, y1, conf, color)
                detected_objects.append(
                    {
                        "label": classes[cls],
                        "confidence": conf,
                        "location": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    }
                )

            #
            # if cls == 0 and conf > conf_threshold:  # hand
            #     label = f"Hand {conf:.2f}"
            #     detected_objects.append(
            #         {
            #             "label": label,
            #             "confidence": conf,
            #             "location": {"left": x1, "top": y1, "right": x2, "bottom": y2},
            #         }
            #     )
            # elif cls == 1 and conf > conf_threshold_bottle:  # bottle
            #     label = f"My Bottle {conf:.2f}"
            #     detected_objects.append(
            #         {
            #             "label": label,
            #             "confidence": conf,
            #             "location": {"left": x1, "top": y1, "right": x2, "bottom": y2},
            #         }
            #     )

    return jsonify({"detectedObjects": detected_objects})


if __name__ == "__main__":
    print("Starting detection server...")
    app.run(host="0.0.0.0", port=5000)
