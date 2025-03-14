import os
import time
from ultralytics import YOLO
import cv2
import torch

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
PROJ = "test01"
MODEL = "train"
# PROJ = "tissue"
# MODEL = "train"
# PROJ = "proj1"
# MODEL = "train9"
#
model = YOLO(
    # "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/runs/detect/train/weights/best.pt"
    # "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/my-first-customed/runs/detect/train/weights/best.pt"
    # "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/runs/detect/train8/weights/best.pt",
    #
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


def get_classes(classes_path=f"projects/{PROJ}/dataset/yolov3/classes.txt"):
    with open(classes_path, "r") as f:
        classes = f.read().splitlines()
    return classes


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 進行人物偵測
    results = model(frame, verbose=False)

    classes = get_classes()
    print("classes", classes)

    conf_threshold = 0.6
    # conf_thresholds = [0.7, 0.5]

    def put_text(text, x1, y1, conf):
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        print(f"{text}:  {conf:.2f}")

    # 繪製偵測結果
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            # if cls == 0 and conf > conf_threshold:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     put_text(f"Hand {conf:.2f}", x1, y1, conf)
            # elif cls == 1 and conf > conf_threshold:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #     put_text(f"My Bottle {conf:.2f}", x1, y1, conf)
            for i, each_class in enumerate(classes):
                if cls == i and conf > conf_threshold:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    put_text(f"{each_class} {conf:.2f}", x1, y1, conf)

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
