#!/usr/bin/env python3
import os
import json
from datetime import datetime
import glob
from pathlib import Path
from tqdm import tqdm

import torch
import fire

from autodistill.helpers import split_data
from autodistill.detection import CaptionOntology, DetectionBaseModel

# from groundingdino.util.inference import Model # not used now
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


class YoloCLI:
    """YOLO 工具的命令列介面"""

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

        for video_path in tqdm(video_files):
            video_path = Path(video_path)
            video_name = video_path.stem
            image_name_pattern = video_name + "-{:05d}.png"
            fps = sv.VideoInfo.from_video_path(str(video_path)).fps
            frame_stride = round(interval / (1000 / fps))

            print(f"處理影片: {video_path.name}")
            print(f"FPS: {fps} frame/s")
            print(f"間隔: {interval} ms")
            print(f"取樣步長: {frame_stride}")

            with sv.ImageSink(
                target_dir_path=frame_dir,
                image_name_pattern=image_name_pattern,
            ) as sink:
                frames = sv.get_video_frames_generator(
                    source_path=str(video_path), stride=frame_stride
                )
                for frame in frames:
                    sink.save_image(frame)
                print(f"總擷取影格數: {len(list(frames))}")

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
        dataset_dir = proj_dir / "dataset"

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

    def convert_format(self, projname: str, source: str, target: str):
        """轉換不同的 YOLO 格式

        Args:
            projname: 專案名稱
            source: 來源格式 ('yolo8' 或 'yolo3')
            target: 目標格式 ('yolo8' 或 'yolo3')
        """
        dataset_dir = Path(os.getcwd()) / "projects" / projname / "dataset"

        if source == "yolo8" and target == "yolo3":
            self._yolo8_to_yolo3(dataset_dir)
        elif source == "yolo3" and target == "yolo8":
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

    def train(self, projname: str, epochs: int = 50, device: str = "cuda"):
        """訓練 YOLO 模型

        Args:
            projname: 專案名稱
            epochs: 訓練回合數，預設 50
            device: 運算裝置，預設 'cuda'
        """
        data_yaml = Path(os.getcwd()) / "projects" / projname / "dataset" / "data.yaml"
        model = YOLOv8("yolov8n.pt")
        print(f"開始訓練 YOLO 模型，使用裝置: {device}")
        model.train(str(data_yaml), epochs=epochs, device=device)
        print("訓練完成！")


if __name__ == "__main__":
    fire.Fire(YoloCLI)
