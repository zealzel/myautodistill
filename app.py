import os
from ultralytics import YOLO
import supervision as sv
from pathlib import Path
from autodistill_yolov8 import YOLOv8
from tqdm import tqdm
import roboflow
import numpy as np
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM


class FaceTrain:
    def __init__(self):
        self.home = os.getcwd()
        self.video_dir_path = f"{self.home}/videos"
        self.image_dir_path = f"{self.home}/images"
        self.frame_interval = 100  # ms
        self.video_paths = sv.list_files_with_extensions(
            directory=self.video_dir_path, extensions=["mov", "mp4", "MOV"]
        )
        roboflow.login()

    def download_dataset(self, dataset_url, model_format):
        # dataset = roboflow.download_dataset(
        #     dataset_url="https://universe.roboflow.com/mohamed-traore-2ekkp/taco-trash-annotations-in-context/model/16",
        #     model_format="yolov8",
        # )
        dataset = roboflow.download_dataset(
            datraset_url=dataset_url, model_format=model_format
        )
        return dataset

    def convert_video_to_image(self):
        for video_path in tqdm(self.video_paths):
            video_name = video_path.stem
            image_name_pattern = video_name + "-{:05d}.png"
            fps = sv.VideoInfo.from_video_path(video_path).fps
            frame_stride = round(self.frame_interval / (1000 / fps))
            print(f"fps: {fps} frame/s")
            print(f"interval: {self.frame_interval} s")
            print("frame_stride:", frame_stride)
            with sv.ImageSink(
                target_dir_path=self.image_dir_path,
                image_name_pattern=image_name_pattern,
            ) as sink:
                images_extract = sv.get_video_frames_generator(
                    source_path=str(video_path), stride=frame_stride
                )
                for image in images_extract:
                    sink.save_image(image=image)
                print(f"total frame count: {len(list(images_extract))}")

    def list_files_with_extensions(self):
        image_paths = sv.list_files_with_extensions(
            directory=self.image_dir_path, extensions=["png", "jpg", "jpg"]
        )
        print("image count:", len(image_paths))
        return image_paths

    def init_base_model_autolabel(self, datset_name, annotation_class):
        # annotation_class = {"milk bottle": "bottle", "blue cap": "cap"}
        ontology = CaptionOntology(annotation_class)
        dataset_dir_path = f"{self.home}/dataset/{datset_name}"
        base_model = GroundedSAM(ontology=ontology)
        dataset = base_model.label(
            input_folder=self.image_dir_path,
            extension=".png",
            output_folder=dataset_dir_path,
        )
        return base_model, dataset

    def display_annotation(self, dataset_name):
        dataset_dir_path = f"{self.home}/dataset/{dataset_name}"
        ANNOTATIONS_DIRECTORY_PATH = f"{dataset_dir_path}/train/labels"
        IMAGES_DIRECTORY_PATH = f"{dataset_dir_path}/train/images"
        DATA_YAML_PATH = f"{dataset_dir_path}/data.yaml"
        # ANNOTATIONS_DIRECTORY_PATH = f"{self.home}/dataset/train/labels"
        # IMAGES_DIRECTORY_PATH = f"{self.home}/dataset/train/images"
        # DATA_YAML_PATH = f"{self.home}/dataset/data.yaml"

        print("ANNOTATIONS_DIRECTORY_PATH:", ANNOTATIONS_DIRECTORY_PATH)
        print("IMAGES_DIRECTORY_PATH:", IMAGES_DIRECTORY_PATH)
        print("DATA_YAML_PATH:", DATA_YAML_PATH)

        IMAGE_DIR_PATH = f"{self.home}/images"
        SAMPLE_SIZE = 16
        SAMPLE_GRID_SIZE = (4, 4)
        SAMPLE_PLOT_SIZE = (16, 10)

        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=IMAGES_DIRECTORY_PATH,
            annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
            data_yaml_path=DATA_YAML_PATH,
        )
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        images = []
        image_names = []
        for i, (image_path, image, annotation) in enumerate(dataset):
            if i == SAMPLE_SIZE:
                break
            annotated_image = ismage.copy()
            annotated_image = mask_annotator.annotate(
                scene=annotated_image, detections=annotation
            )
            annotated_image = box_annotator.annotate(
                scene=annotated_image, detections=annotation
            )
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=annotation
            )
            image_names.append(Path(image_path).name)
            images.append(annotated_image)

        sv.plot_images_grid(
            images=images,
            titles=image_names,
            grid_size=SAMPLE_GRID_SIZE,
            size=SAMPLE_PLOT_SIZE,
        )

    def train_yolo(self, dataset_name):
        dataset_dir_path = f"{self.home}/dataset/{dataset_name}"
        ANNOTATIONS_DIRECTORY_PATH = f"{dataset_dir_path}/train/labels"
        IMAGES_DIRECTORY_PATH = f"{dataset_dir_path}/train/images"
        DATA_YAML_PATH = f"{dataset_dir_path}/data.yaml"
        target_model = YOLOv8("yolov8n.pt")
        target_model.train(DATA_YAML_PATH, epochs=50)

    def predict_yolo(images):
        MODEL_NAME = "yolov8n.pt"
        model = YOLO()
        results = model(images)
        predictions = []
        for result in results:
            img_width, img_height = result.orig_shape
            boxes = result.boxes.cpu().numpy()
            prediction = {
                "result": [],
                "score": 0.0,
                "model_version": MODEL_NAME,
            }
            scores = []
            for box, class_id, score in zip(boxes.xywh, boxes.cls, boxes.conf):
                x, y, w, h = box
                prediction["result"].append(
                    {
                        "from_name": "label",
                        "to_name": "img",
                        "original_width": int(img_width),
                        "original_height": int(img_height),
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "rectanglelabels": [result.names[class_id]],
                            "width": w / img_width * 100,
                            "height": h / img_height * 100,
                            "x": (x - 0.5 * w) / img_width * 100,
                            "y": (y - 0.5 * h) / img_height * 100,
                        },
                        "score": float(score),
                        "type": "rectanglelabels",
                    }
                )
                scores.append(float(score))
            prediction["score"] = min(scores) if scores else 0.0
            predictions.append(prediction)
        return predictions


if __name__ == "__main__":
    ft = FaceTrain()
    annotation_class = {
        "normal human hand": "hand",
        "bottle made by silver stain steel with slightly cone shape": "mybottle",
    }
    dataset_name = "my-first-customed"
    ft.convert_video_to_image()
    """
    ft.init_base_model_autolabel(
        datset_name="my-first-customed",
        annotation_class=annotation_class,
    )

    yolo detect predict model=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/runs/detect/train/weights/best.pt \
      source=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed/train/images/IMG_2872-00099.jpg

    """
