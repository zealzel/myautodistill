import os
from autodistill_yolov8 import YOLOv8
from ultralytics import YOLO
import warnings

warnings.filterwarnings(
    "ignore", "You are using `torch.load` with `weights_only=False`*."
)


HOME = os.getcwd()
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"

target_model = YOLOv8("yolov8n.pt")
# target_model = YOLOv8("yolov8n.yaml").load("yolov8n.pt", weights_only=False)
#
# target_model = YOLO("yolov8n.yaml").load("yolov8n.pt")

# target_model.train(DATA_YAML_PATH, epochs=50, device="mps")
target_model.train(DATA_YAML_PATH, epochs=50)


"""
yolo detect predict model=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/try-autodistill/runs/detect/train3/weights/best.pt \
  source=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/try-autodistill/videos/milk-video-8.mov

yolo detect predict model=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/try-autodistill/runs/detect/train3/weights/best.pt \
  source=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/try-autodistill/videos/milk-video-3.mov
"""
