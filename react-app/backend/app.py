import sys
import os

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

model = YOLO("../../runs/detect/train9/weights/best_ncnn_model")

app = Flask(__name__)
CORS(app)


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

            if cls == 0 and conf > conf_threshold:  # hand
                label = f"Hand {conf:.2f}"
                detected_objects.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "location": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    }
                )
            elif cls == 1 and conf > conf_threshold_bottle:  # bottle
                label = f"My Bottle {conf:.2f}"
                detected_objects.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "location": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                    }
                )

    return jsonify({"detectedObjects": detected_objects})


if __name__ == "__main__":
    print("Starting detection server...")
    app.run(host="0.0.0.0", port=5000)
