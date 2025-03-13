#!/usr/bin/env python3
import os
import yaml
import cv2
import enum
import json
from datetime import datetime
import glob
from pathlib import Path
from tqdm import tqdm
import roboflow

import torch
import fire

from autodistill.helpers import split_data
from autodistill.detection import CaptionOntology, DetectionBaseModel

# from groundingdino.util.inference import Model # not used now
from ultralytics import YOLO
from autodistill_yolov8 import YOLOv8

# from autodistill_grounded_sam import GroundedSAM # move into class method
from label_studio_sdk.converter.imports import yolo as import_yolo
import supervision as sv

# 檢查可用的運算裝置
if torch.cuda.is_available():
    print("CUDA available")
    DEVICE = torch.device("cuda")
elif torch.mps.is_available():
    print("mps available")
    DEVICE = torch.device("mps")
else:
    print("WARNING: CUDA or MPS not available. GroundingDINO will run very slowly.")
    DEVICE = torch.device("cpu")

print(f"DEVICE: {DEVICE}")


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


# def my_yolo_train(self, dataset_yaml, proj_dir, epochs=200, device="cpu"):
def my_yolo_train(self, dataset_yaml, project, epochs=200, device="cpu"):
    print("my_yolo_train")
    self.yolo.train(data=dataset_yaml, epochs=epochs, device=device, project=project)
    # self.yolo.train(data=dataset_yaml, epochs=epochs, device=device)


YOLOv8.train = my_yolo_train


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


class YoloCLI:
    """YOLO 工具的命令列介面"""

    # def __init__(self, projname, test_val_ratio=0.2):
    #     self.projname = projname
    #     self.home = os.getcwd()
    #     self.proj = f"{self.home}/projects/{self.projname}"
    #     self.video_dir_path = f"{self.proj}/videos"
    #     self.image_dir_path = f"{self.proj}/images"
    #     self.frame_dir_path = f"{self.proj}/frames"
    #     self.frame_interval = 100  # ms
    #     print("proj:", self.proj)
    #     print("video_dir_path:", self.video_dir_path)
    #     print("image_dir_path:", self.image_dir_path)
    #     print("frame_dir_path:", self.frame_dir_path)
    #     print("video_paths:", self.video_paths)
    #     roboflow.login()

    def create_proj(self, projname: str):
        """建立新的專案資料夾結構

        Args:
            projname: 專案名稱
        """
        project_path = Path(os.getcwd()) / "projects" / projname
        project_path.mkdir(parents=True, exist_ok=True)

        # 建立專案資料夾
        for sub in [
            "videos",
            "images",
            "frames",
            "dataset/yolov8/train/images",
            "dataset/yolov8/train/labels",
            "dataset/yolov8/valid/images",
            "dataset/yolov8/valid/labels",
            "dataset/yolov3/labels",
        ]:
            path = project_path / sub
            path.mkdir(parents=True, exist_ok=True)
            print("建立資料夾：", path)

        # 建立 yolov3 的 images symbolic link
        yolov3_images_symlink = project_path / "dataset/yolov3/images"
        yolov8_images_target = project_path / "dataset/yolov8/train/images"
        if yolov3_images_symlink.exists() and yolov3_images_symlink.is_symlink():
            os.remove(yolov3_images_symlink)
        os.symlink(yolov8_images_target, yolov3_images_symlink)
        print("專案建立完成：", project_path)

    def convert_video(self, projname: str, interval: int = 100):
        """將專案中的影片轉換為圖片

        Args:
            projname: 專案名稱
            interval: 擷取影格的時間間隔(毫秒)，預設 100ms
        """
        proj_dir = Path(os.getcwd()) / "projects" / projname
        video_dir = proj_dir / "videos"
        frame_dir = proj_dir / "frames"

        video_files = glob.glob(str(video_dir / "*.[mM][pP]4")) + glob.glob(
            str(video_dir / "*.[mM][oO][vV]")
        )

        if not video_files:
            print("專案內找不到影片檔案")
            return

        for video_path in video_files:
            video_path = Path(video_path)
            video_name = video_path.stem

            # 取得影片資訊
            video_info = sv.VideoInfo.from_video_path(str(video_path))
            fps = video_info.fps
            frame_count = video_info.total_frames
            frame_stride = round(interval / (1000 / fps))

            # 計算預期的總幀數
            expected_frames = frame_count // frame_stride

            print(f"\n處理影片: {video_path.name}")
            print(f"FPS: {fps} frame/s")
            print(f"總幀數: {frame_count}")
            print(f"間隔: {interval} ms")
            print(f"取樣步長: {frame_stride}")
            print(f"預期擷取幀數: {expected_frames}")

            # 建立進度條
            pbar = tqdm(total=expected_frames, desc="擷取影格")

            with sv.ImageSink(
                target_dir_path=frame_dir,
                image_name_pattern=f"{video_name}-{{:05d}}.png",
            ) as sink:
                frame_generator = sv.get_video_frames_generator(
                    source_path=str(video_path), stride=frame_stride
                )

                saved_count = 0
                for frame in frame_generator:
                    sink.save_image(frame)
                    saved_count += 1
                    pbar.update(1)

                pbar.close()
                print(f"實際擷取影格數: {saved_count}")

    def auto_label_test(
        self, projname: str, annotation_class: dict, from_frames: bool = True
    ):
        from autodistill_grounded_sam import GroundedSAM

        print("auto_label")
        print("projname:", projname)
        print("annotation_class:", annotation_class)
        print("type(annotation_class):", type(annotation_class))
        print("from_frames:", from_frames)
        if type(annotation_class) is str:
            print("type(annotation_class) is str")
            annotation_class = json.loads(annotation_class)
            print("type(annotation_class):", type(annotation_class))

    def auto_label(
        self, projname: str, annotation_class: dict, from_frames: bool = True
    ):
        from autodistill_grounded_sam import GroundedSAM

        """使用 GroundedSAM 自動標註影像

        Args:
            projname: 專案名稱
            annotation_class: 標註類別對照字典，例如 "{'normal human hand': 'hand'}"
            from_frames: 是否從 frames 資料夾讀取圖片，預設為 True
        """
        proj_dir = Path(os.getcwd()) / "projects" / projname
        # dataset_dir = proj_dir / "dataset"
        dataset_dir = proj_dir / "dataset" / "yolov8"

        if type(annotation_class) is str:
            annotation_class = json.loads(annotation_class)

        ontology = CaptionOntology(annotation_class)
        base_model = GroundedSAM(ontology=ontology)

        input_folder = str(proj_dir / ("frames" if from_frames else "images"))
        dataset = base_model.label(
            input_folder=input_folder,
            extension=".png",
            output_folder=str(dataset_dir),
        )
        print("自動標註完成")
        return dataset

    def seg2bbox(self, projname):
        """
        projname: 專案名稱
        in_label_folder is a directory, convert all .txt files in it to yolo bbox format in out_label_folder
        if out_label_folder is not provided, the converted files will overwrite the original files

        dataset_path = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/projname/dataset"
        in_path = f"{dataset_path}/yolov8/train/labels"
        out_path = f"{dataset_path}/yolov3/labels"
        """
        dataset_dir = Path(os.getcwd()) / "projects" / projname / "dataset"
        in_label_folder = dataset_dir / "yolov8" / "train" / "labels"
        out_label_folder = dataset_dir / "yolov3" / "labels"
        print("in_label_folder:", in_label_folder)
        print("out_label_folder:", out_label_folder)
        if not out_label_folder:
            out_label_folder = in_label_folder
        label_files = [e for e in os.listdir(in_label_folder) if e.endswith(".txt")]
        if not label_files:
            print("No label files found in the directory.")
        for file in label_files:
            if not file.endswith(".txt"):
                continue
            label_textfile = os.path.join(in_label_folder, file)
            with open(label_textfile, "r") as f:
                lines = f.readlines()
            bboxes = [seg_to_bbox(line) for line in lines]
            print("bboxes:", bboxes)
            out_label_textfile = os.path.join(out_label_folder, file)
            print("out_label_textfile:", out_label_textfile)
            with open(out_label_textfile, "w") as f:
                f.writelines("\n".join(bboxes))

    def convert_format(self, projname: str, source: str, target: str):
        """轉換不同的 YOLO 格式

        Args:
            projname: 專案名稱
            source: 來源格式 ('yolo8' 或 'yolo3')
            target: 目標格式 ('yolo8' 或 'yolo3')
        """

        if source == "yolo8" and target == "yolo3":
            dataset_dir = Path(os.getcwd()) / "projects" / projname / "dataset"
            self._yolo8_to_yolo3(dataset_dir)
        elif source == "yolo3" and target == "yolo8":
            dataset_dir = Path(os.getcwd()) / "projects" / projname / "dataset/reviewed"
            self._yolo3_to_yolo8(dataset_dir)
        else:
            print(f"不支援的格式轉換: {source} -> {target}")

    def _yolo8_to_yolo3(self, data_dir: Path):
        """YOLOv8 轉 YOLOv3 格式"""
        yaml_path = data_dir / "yolov8/data.yaml"
        notes_path = data_dir / "yolov3/notes.json"
        classes_path = data_dir / "yolov3/classes.txt"

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        names = data.get("names", [])
        categories = [{"id": idx, "name": name} for idx, name in enumerate(names)]
        notes = {
            "categories": categories,
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "contributor": "Label Studio",
            },
        }

        with open(notes_path, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)

        with open(classes_path, "w", encoding="utf-8") as f:
            for name in names:
                f.write(f"{name}\n")

        print("YOLOv8 轉 YOLOv3 完成")

    def _yolo3_to_yolo8(self, data_dir: Path):
        """YOLOv3 轉 YOLOv8 格式"""
        classes_path = data_dir / "classes.txt"
        yaml_path = data_dir / "data.yaml"

        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f.readlines()]

        data_yaml = {
            "train": str(data_dir / "yolov8/train/images"),
            "val": str(data_dir / "yolov8/valid/images"),
            "nc": len(class_names),
            "names": class_names,
        }

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        # 建立 annotations symbolic link
        target = data_dir / "labels"
        symlink = data_dir / "annotations"
        if symlink.exists() and symlink.is_symlink():
            os.remove(symlink)
        os.symlink(target, symlink)

        print("YOLOv3 轉 YOLOv8 完成")

    def convert_yolo_to_labelstudio(self, projname):
        """Tests generated config and json files for yolo imports
        test_import_yolo_data folder assumes only images in the 'images' folder
        with corresponding labels existing in the 'labes' dir and a 'classes.txt' present.
        ex:
        input_dir = "/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/abc/dataset/yolov3"
        image_root_url = "/data/local-files/?d=images"
        """
        proj_dir = Path(os.getcwd()) / "projects" / projname
        dataset_dir_path = proj_dir / "dataset"
        input_data_dir = str(dataset_dir_path / "yolov3")
        # image_root_url = "/data/local-files/?d=images"
        image_root_url = "/data/local-files/?d=yolov3/images"
        # out_json_file = "output_for_labelstudio.json"
        out_json_file = str(dataset_dir_path / "output_for_labelstudio.json")
        image_ext = ".jpg,.jpeg,.png"  # comma seperated string of extns.
        # input_dir: directory with YOLO where images, labels, notes.json are located
        import_yolo.convert_yolo_to_ls(
            input_dir=input_data_dir,
            out_file=out_json_file,
            image_ext=image_ext,
            image_root_url=image_root_url,
        )
        out_config_file = f"{out_json_file[:-5]}.label_config.xml"
        assert os.path.exists(out_config_file) and os.path.exists(out_json_file), (
            "> import failed! files not generated."
        )

    def splitdata(self, projname, ratio=0.8):
        # split_data(output_folder, split_ratio=1.0, record_confidence=record_confidence)
        proj_dir = Path(os.getcwd()) / "projects" / projname
        dataset_dir_path = str(proj_dir / "dataset/reviewed")
        split_data(dataset_dir_path, 0.8)

    # def train(self, projname: str, epochs: int = 50, device: str = "cuda"):
    def train(self, projname: str, epochs: int = 50, device: str = "mps"):
        """訓練 YOLO 模型

        Args:
            projname: 專案名稱
            epochs: 訓練回合數，預設 50
            device: 運算裝置，預設 'mps'
        """
        proj_dir = Path(os.getcwd()) / "projects" / projname
        data_yaml = (
            Path(os.getcwd()) / f"projects/{projname}/dataset/reviewed/data.yaml"
        )
        model = YOLOv8("yolov8n.pt")
        print(f"開始訓練 YOLO 模型，使用裝置: {device}")
        # model.train(str(data_yaml), epochs=epochs, device=device)
        model.train(
            str(data_yaml),
            epochs=epochs,
            device=device,
            project=str(proj_dir / "runs/detect"),
        )
        print("訓練完成！")

    def display_annotation(self, projname):
        proj_dir = Path(os.getcwd()) / "projects" / projname
        dataset_dir_path = str(proj_dir / "dataset")
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


if __name__ == "__main__":
    fire.Fire(YoloCLI)
    """usage
    python app.py create_proj proj1

    python app.py convert_video proj1

    python app-cli.py auto_label_test proj1 "{
    python app-cli.py auto_label proj1 "{
        'normal human hand': 'hand',
        'bottle made by silver stain steel with slightly cone shape': 'mybottle'
    }"

    python app-cli.py display_annotation proj1

    python app-cli.py seg2bbox proj1

    python app-cli.py convert_format proj1 yolo8 yolo3

    python app-cli.py convert_yolo_to_labelstudio proj1

    # actions in lable-studio
    - import
    - review
    - export

    # unzip exported and rename
    project-11-at-2025-03-13-05-50-caf27874.zip -> unzip to reviewed/ folder

    python app-cli.py convert_format proj1 yolo3 yolo8

    # split the data into train/valid/test set
    python app-cli.py splitdata proj1 0.8

    # train yolo model, it will generate model files best.pt under runs/detect/trainxx/weights/
    python app-cli.py train proj1

    """
