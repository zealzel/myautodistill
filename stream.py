import os
import time
from ultralytics import YOLO
import cv2
import torch
import numpy as np
import colorsys

# Check if MPS is available
if torch.backends.mps.is_available():
    print("MPS is available.")
else:
    print("MPS is not available.")


# Check the device being used
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

# convert model to ncnn format
# yolo export model=runs/detect/train9/weights/best.pt format=ncnn
#
# PROJ = "test01"
# MODEL = "train"
# PROJ = "tissue"
# MODEL = "train"
PROJ = "proj1"
MODEL = "train9"
#
model = YOLO(
    # "runs/detect/train8/weights/best_ncnn_model"
    # "runs/detect/train8/weights/best.pt"
    # "runs/detect/train9/weights/best.pt"
    #
    # "projects/test01/runs/detect/train/weights/best.pt"
    # f"projects/{PROJ}/runs/detect/{MODEL}/weights/best.pt"
    f"projects/{PROJ}/runs/detect/{MODEL}/weights/best_ncnn_model"
)
# model.to(device)

# 開啟攝影機
cap = cv2.VideoCapture(0)  # Mac's Camera
# cap = cv2.VideoCapture(1)  # GoPro

print("is cap opened?", cap.isOpened())
time.sleep(1)
save_dir = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/image-saved"
frame_count = 0


# classes.txt
"""
# classes.txt
object1
object2
object3
"""


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


def get_classes(classes_path=f"projects/{PROJ}/dataset/yolov3/classes.txt"):
    with open(classes_path, "r") as f:
        classes = f.read().splitlines()
    return classes


# 取得類別列表
classes = get_classes()
# 為每個類別生成顏色
colors = generate_colors(len(classes))


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


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    conf_threshold = 0.6

    # 繪製偵測結果
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])

            if conf > conf_threshold:
                color = colors[cls]  # 使用對應類別的顏色
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                put_text(frame, classes[cls], x1, y1, conf, color)

    cv2.imshow("object detection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        filename = os.path.join(save_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(filename, frame)
        print("Saved frame:", filename)
    if key == ord("q"):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()


"""
yolo detect predict model=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/runs/detect/train8/weights/best.pt source=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/abc/dataset/project-9-at-2025-02-19-09-45-87d33f43/train/images/IMG_2872-00001.jpg
"""
