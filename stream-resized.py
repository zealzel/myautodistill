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

model = YOLO(
    # "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/runs/detect/train/weights/best.pt"
    # "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/my-first-customed/runs/detect/train/weights/best.pt"
    # "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/runs/detect/train8/weights/best.pt",
    # "runs/detect/train9/weights/best.pt"
    "runs/detect/train8/weights/best_ncnn_model"
)
# model.to(device)

print("...2")
# 開啟攝影機
cap = cv2.VideoCapture(0)  # Mac's Camera
# cap = cv2.VideoCapture(1)  # GoPro

print("...3")
print("is cap opened?", cap.isOpened())
time.sleep(1)
save_dir = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/image-saved"
frame_count = 0


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (640, 480))
    results = model(resized_frame, verbose=False)
    # results = model(frame, verbose=False)
    conf_threshold = 0.7

    # 繪製偵測結果
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            if cls == 0 and conf > conf_threshold:  # cls == 0 表示 "person"
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    resized_frame,
                    f"Hand {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                print(f"hand:  {conf:.2f}")
            elif cls == 1 and conf > conf_threshold:  # cls == 0 表示 "person"
                # elif cls == 1 and conf > 0.5:  # cls == 0 表示 "person"
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    resized_frame,
                    f"My Bottle {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                print(f"bottle:  {conf:.2f}")

    cv2.imshow("my bottle detection", resized_frame)
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
