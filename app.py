import roboflow
import numpy as np

roboflow.login()

dataset = roboflow.download_dataset(
    dataset_url="https://universe.roboflow.com/mohamed-traore-2ekkp/taco-trash-annotations-in-context/model/16",
    model_format="yolov8",
)
