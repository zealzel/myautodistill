import os
import enum
import urllib.request
import cv2
import glob
import random
import shutil
from ultralytics import YOLO
import supervision as sv
from pathlib import Path
from autodistill_yolov8 import YOLOv8
from tqdm import tqdm
import roboflow
import numpy as np

from autodistill.helpers import split_data
from autodistill.detection import CaptionOntology
from groundingdino.util.inference import Model

from autodistill.detection import DetectionBaseModel

import autodistill_grounded_sam.helpers
import autodistill.helpers

from autodistill_grounded_sam import GroundedSAM
from label_studio_sdk.converter.imports import yolo as import_yolo
from PIL import Image
import yaml
import torch


if torch.cuda.is_available():
    print("CUDA available")
    DEVICE = torch.device("cuda")
elif torch.mps.is_available():
    print("mps available")
    DEVICE = torch.device("mps")
else:
    print("WARNING: CUDA or MPS not available. GroundingDINO will run very slowly.")

print(f"DEVICE: {DEVICE}")


original_label = DetectionBaseModel.label


class NmsSetting(str, enum.Enum):
    NONE = "no_nms"
    CLASS_SPECIFIC = "class_specific"
    CLASS_AGNOSTIC = "class_agnostic"


def my_detection_label(
    self,
    input_folder: str,
    extension: str = ".jpg",
    output_folder: str | None = None,
    human_in_the_loop: bool = False,
    roboflow_project: str | None = None,
    roboflow_tags: list[str] = ["autodistill"],
    sahi: bool = False,
    record_confidence: bool = False,
    nms_settings: NmsSetting = NmsSetting.NONE,
) -> sv.DetectionDataset:
    """
    Label a dataset with the model.
    """
    print("monkey patch! my_detection_label")
    if output_folder is None:
        output_folder = input_folder + "_labeled"

    os.makedirs(output_folder, exist_ok=True)

    image_paths = glob.glob(input_folder + "/*" + extension)
    detections_map = {}

    if sahi:
        slicer = sv.InferenceSlicer(callback=self.predict)

    progress_bar = tqdm(image_paths, desc="Labeling images")
    for f_path in progress_bar:
        progress_bar.set_description(desc=f"Labeling {f_path}", refresh=True)

        image = cv2.imread(f_path)
        if sahi:
            detections = slicer(image)
        else:
            detections = self.predict(image)

        if nms_settings == NmsSetting.CLASS_SPECIFIC:
            detections = detections.with_nms()
        if nms_settings == NmsSetting.CLASS_AGNOSTIC:
            detections = detections.with_nms(class_agnostic=True)

        detections_map[f_path] = detections

    dataset = sv.DetectionDataset(self.ontology.classes(), image_paths, detections_map)

    dataset.as_yolo(
        output_folder + "/images",
        output_folder + "/annotations",
        min_image_area_percentage=0.01,
        data_yaml_path=output_folder + "/data.yaml",
    )

    if record_confidence:
        image_names = [os.path.basename(f_path) for f_path in image_paths]
        self._record_confidence_in_files(
            output_folder + "/annotations", image_names, detections_map
        )

    # my own overwrite
    split_data(output_folder, split_ratio=1.0, record_confidence=record_confidence)

    if human_in_the_loop:
        roboflow.login()

        rf = roboflow.Roboflow()

        workspace = rf.workspace()

        workspace.upload_dataset(output_folder, project_name=roboflow_project)

    print("Labeled dataset created - ready for distillation.")
    return dataset


DetectionBaseModel.label = my_detection_label


def load_grounding_dino():
    AUTODISTILL_CACHE_DIR = os.path.expanduser("~/.cache/autodistill")

    GROUDNING_DINO_CACHE_DIR = os.path.join(AUTODISTILL_CACHE_DIR, "groundingdino")

    GROUNDING_DINO_CONFIG_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "GroundingDINO_SwinT_OGC.py"
    )
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(
        GROUDNING_DINO_CACHE_DIR, "groundingdino_swint_ogc.pth"
    )

    try:
        print("trying to load grounding dino directly")
        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )
        return grounding_dino_model
    except Exception:
        print("downloading dino model weights")
        if not os.path.exists(GROUDNING_DINO_CACHE_DIR):
            os.makedirs(GROUDNING_DINO_CACHE_DIR)

        if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CHECKPOINT_PATH)

        if not os.path.exists(GROUNDING_DINO_CONFIG_PATH):
            url = "https://raw.githubusercontent.com/roboflow/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            urllib.request.urlretrieve(url, GROUNDING_DINO_CONFIG_PATH)

        grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=DEVICE,
        )

        # grounding_dino_model.to(DEVICE)

        return grounding_dino_model


autodistill_grounded_sam.helpers.load_grounding_dino = load_grounding_dino


def test_base_import_yolo(input_data_dir):
    """Tests generated config and json files for yolo imports
    test_import_yolo_data folder assumes only images in the 'images' folder
    with corresponding labels existing in the 'labes' dir and a 'classes.txt' present.
    (currently 7 images -> 3 png, 2 jpg and 2 jpeg files)
    """
    # input_data_dir = os.path.join(
    #     os.path.abspath(os.path.dirname(__file__)), "data", "test_import_yolo_data"
    # )
    # out_json_file = os.path.join("/tmp", "lsc-pytest", "yolo_exp_test.json")
    out_json_file = "output_test.json"

    image_ext = ".jpg,.jpeg,.png"  # comma seperated string of extns.

    image_root_url = "/data/local-files/?d=train/images"
    import_yolo.convert_yolo_to_ls(
        input_dir=input_data_dir,
        out_file=out_json_file,
        image_ext=image_ext,
        image_root_url=image_root_url,
    )

    #'yolo_exp_test.label_config.xml' and 'yolo_exp_test.json' must be generated.
    # out_config_file = os.path.join(
    #     "/tmp", "lsc-pytest", "yolo_exp_test.label_config.xml"
    # )
    out_config_file = "output_test.label_config.xml"
    assert os.path.exists(out_config_file) and os.path.exists(out_json_file), (
        "> import failed! files not generated."
    )

    # # provided labels from classes.txt
    # with open(os.path.join(input_data_dir, "classes.txt"), "r") as f:
    #     labels = f.read()[:-1].split(
    #         "\n"
    #     )  # [:-1] since last line in classes.txt is empty by convention
    #
    # # generated labels from config xml
    # label_element = ET.parse(out_config_file).getroot()[2]
    # labels_generated = [x.attrib["value"] for x in label_element.getchildren()]
    # assert set(labels) == set(labels_generated), (
    #     "> generated class labels do not match original labels"
    # )
    #
    # # total image files in the input folder
    # img_files = glob.glob(os.path.join(input_data_dir, "images", "*"))
    #
    # with open(out_json_file, "r") as f:
    #     ls_data = json.loads(f.read())

    # assert len(ls_data) == len(img_files), "some file imports did not succeed!"


def seg_to_bbox(seg_info):
    # Example input: 5 0.046875 0.369141 0.0644531 0.384766 0.0800781 0.402344 ...
    class_id, *points = seg_info.split()
    points = [float(p) for p in points]
    x_min, y_min, x_max, y_max = (
        min(points[0::2]),
        min(points[1::2]),
        max(points[0::2]),
        max(points[1::2]),
    )
    width, height = x_max - x_min, y_max - y_min
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    # bbox_info = f"{int(class_id) - 1} {x_center} {y_center} {width} {height}"
    bbox_info = f"{int(class_id)} {x_center} {y_center} {width} {height}"
    return bbox_info


def seg2bbox(label_folder, out_label_folder):
    for file in os.listdir(label_folder):
        if file.endswith(".txt"):
            label_textfile = os.path.join(label_folder, file)
            with open(label_textfile, "r") as f:
                lines = f.readlines()
            bboxes = [seg_to_bbox(line) for line in lines]
        print("bboxes:", bboxes)
        out_label_textfile = os.path.join(out_label_folder, file)
        print("out_label_textfile:", out_label_textfile)
        with open(out_label_textfile, "w") as f:
            f.writelines("\n".join(bboxes))


class YoloFit:
    def __init__(self, projname, test_val_ratio=0.2):
        self.projname = projname
        self.home = os.getcwd()
        self.proj = f"{self.home}/projects/{self.projname}"
        self.video_dir_path = f"{self.proj}/videos"
        self.image_dir_path = f"{self.proj}/images"
        self.frame_dir_path = f"{self.proj}/frames"
        self.frame_interval = 100  # ms
        print("proj:", self.proj)
        print("video_dir_path:", self.video_dir_path)
        print("image_dir_path:", self.image_dir_path)
        print("frame_dir_path:", self.frame_dir_path)
        print("video_paths:", self.video_paths)
        roboflow.login()

    @property
    def video_paths(self):
        return sv.list_files_with_extensions(
            directory=self.video_dir_path, extensions=["mov", "mp4", "MOV"]
        )

    def create_proj(self):
        project_path = Path(self.home) / "projects" / self.projname
        project_path.mkdir(parents=True, exist_ok=True)
        # Create subdirectories
        # (project_path / "dataset").mkdir(exist_ok=True)
        (project_path / "dataset/train/images").mkdir(parents=True, exist_ok=True)
        (project_path / "dataset/train/labels").mkdir(parents=True, exist_ok=True)
        (project_path / "dataset/valid/images").mkdir(parents=True, exist_ok=True)
        (project_path / "dataset/valid/labels").mkdir(parents=True, exist_ok=True)
        (project_path / "frames").mkdir(exist_ok=True)
        (project_path / "runs").mkdir(exist_ok=True)
        (project_path / "videos").mkdir(exist_ok=True)

    def download_dataset(self, dataset_url, model_format):
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
                target_dir_path=self.frame_dir_path,
                image_name_pattern=image_name_pattern,
            ) as sink:
                images_extract = sv.get_video_frames_generator(
                    source_path=str(video_path), stride=frame_stride
                )
                for image in images_extract:
                    sink.save_image(image=image)
                print(f"total frame count: {len(list(images_extract))}")

    def list_images_with_extensions(self):
        image_paths = sv.list_files_with_extensions(
            directory=self.image_dir_path, extensions=["png", "jpg", "jpg"]
        )
        print("image count:", len(image_paths))
        return image_paths

    def list_frames_with_extensions(self):
        frame_paths = sv.list_files_with_extensions(
            directory=self.frame_dir_path, extensions=["png", "jpg", "jpg"]
        )
        print("frame count:", len(frame_paths))
        return frame_paths

    def init_base_model_autolabel(self, annotation_class, from_frames=True):
        # annotation_class = {"milk bottle": "bottle", "blue cap": "cap"}
        ontology = CaptionOntology(annotation_class)
        dataset_dir_path = f"{self.proj}/dataset/"
        base_model = GroundedSAM(ontology=ontology)
        input_folder = self.frame_dir_path if from_frames else self.image_dir_path
        dataset = base_model.label(
            input_folder=input_folder,
            # input_folder=self.image_dir_path,
            extension=".png",
            output_folder=dataset_dir_path,
        )
        return base_model, dataset

    def merge_test_val(self):
        dataset_dir_path = f"{self.proj}/dataset/"
        images_dir_path = Path(dataset_dir_path) / "images"
        images_dir_path.mkdir(exist_ok=True)
        labels_dir_path = Path(dataset_dir_path) / "annotations"
        labels_dir_path.mkdir(exist_ok=True)

        # Move images from train/images and valid/images to images/
        for subdir in ["train/images", "valid/images"]:
            source_dir = Path(dataset_dir_path) / subdir
            for image_file in source_dir.glob("*.*"):
                shutil.move(str(image_file), images_dir_path)

        # Move labels from train/labels and valid/labels to annotations/
        for subdir in ["train/labels", "valid/labels"]:
            source_dir = Path(dataset_dir_path) / subdir
            for label_file in source_dir.glob("*.*"):
                shutil.move(str(label_file), labels_dir_path)

    def rearrnage_test_val(self, split_ratio=0.8):
        dataset_dir_path = f"{self.proj}/dataset/"
        self.merge_test_val()
        split_data(dataset_dir_path, split_ratio=split_ratio)

    def display_annotation(self):
        dataset_dir_path = f"{self.proj}/dataset"
        ANNOTATIONS_DIRECTORY_PATH = f"{dataset_dir_path}/train/labels"
        IMAGES_DIRECTORY_PATH = f"{dataset_dir_path}/train/images"
        DATA_YAML_PATH = f"{dataset_dir_path}/data.yaml"
        print("ANNOTATIONS_DIRECTORY_PATH:", ANNOTATIONS_DIRECTORY_PATH)
        print("IMAGES_DIRECTORY_PATH:", IMAGES_DIRECTORY_PATH)
        print("DATA_YAML_PATH:", DATA_YAML_PATH)

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
            annotated_image = image.copy()
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

    def train_yolo(self):
        dataset_dir_path = f"{self.proj}/dataset"
        DATA_YAML_PATH = f"{dataset_dir_path}/data.yaml"
        target_model = YOLOv8("yolov8n.pt")
        target_model.train(
            DATA_YAML_PATH, epochs=50, device="mps"
        )  # or cuda if supported

    def predict_yolo(images):
        # for importing into label-studio, TBD
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
    projname = "abc"
    yf = YoloFit(projname)
    annotation_class = {
        "normal human hand": "hand",
        "bottle made by silver stain steel with slightly cone shape": "mybottle",
    }
    # yf.convert_video_to_image()
    # yf.init_base_model_autolabel(
    #     annotation_class=annotation_class,
    # )
    # in_path = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed/train/labels-bak/"
    # out_path = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed/train/labels/"
    in_path = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed/train"
    out_path = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed/train"
    # test_base_import_yolo(input_data_dir=in_path)

    tt = lambda xc, yc, w, h: [xc - w / 2, yc - h / 2, w, h]

    # seg2bbox(in_path, out_path)
    """
    yf.init_base_model_autolabel(
        datset_name="my-first-customed",
        annotation_class=annotation_class,
    )
    yf.init_base_model_autolabel(
        annotation_class=annotation_class,
    )
/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/abc

    yolo detect predict model=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/runs/detect/train/weights/best.pt \
      source=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed/train/images/IMG_2872-00099.jpg

    """
