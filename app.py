#!/usr/bin/env python3
import argparse
import os
import json
import yaml
import shutil
import glob
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


# 1. 標註影像資料：原始的 my_detection_label 函式
def cli_label(args):
    from autodistill.detection import DetectionBaseModel
    import supervision as sv

    # 假設已將 DetectionBaseModel.label 作過 monkey patch（見原始程式碼）
    # 這裡直接使用已經被 patch 的 label 方法
    model = DetectionBaseModel()  # 請依照實際模型初始化
    print("開始標註影像...")
    dataset = model.label(
        input_folder=args.input_folder,
        extension=args.extension,
        output_folder=args.output_folder,
        human_in_the_loop=args.human_in_the_loop,
        roboflow_project=args.roboflow_project,
        roboflow_tags=args.roboflow_tags,
        sahi=args.sahi,
        record_confidence=args.record_confidence,
        nms_settings=args.nms_settings,
    )
    print(
        "標註完成，輸出資料夾：", args.output_folder or (args.input_folder + "_labeled")
    )


# 2. 載入 GroundingDINO 模型
def cli_load_grounding_dino(args):
    from autodistill_grounded_sam.helpers import load_grounding_dino

    model = load_grounding_dino()
    print("GroundingDINO 模型已載入，運行裝置：", model.device)


# 3. YOLO 格式轉 Label Studio（輸出 json 與 label_config.xml）
def cli_convert_yolo_to_labelstudio(args):
    from label_studio_sdk.converter.imports import yolo as import_yolo

    print("轉換 YOLO 格式到 Label Studio 格式...")
    import_yolo.convert_yolo_to_ls(
        input_dir=args.input_data_dir,
        out_file=args.out_json_file,
        image_ext=",.jpg,.jpeg,.png",
        image_root_url=args.image_root_url,
    )
    print("轉換完成！請確認生成的", args.out_json_file, "與相對應的 label_config.xml")


# 4. Segmentation to Bounding Box 轉換
def cli_seg2bbox(args):
    def seg_to_bbox(seg_info):
        parts = seg_info.split()
        class_id, *points = parts
        points = [float(p) for p in points]
        x_min, y_min, x_max, y_max = (
            min(points[0::2]),
            min(points[1::2]),
            max(points[0::2]),
            max(points[1::2]),
        )
        width, height = x_max - x_min, y_max - y_min
        x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
        return f"{int(class_id)} {x_center} {y_center} {width} {height}"

    in_folder = args.in_label_folder
    out_folder = args.out_label_folder or in_folder
    label_files = [f for f in os.listdir(in_folder) if f.endswith(".txt")]
    if not label_files:
        print("找不到任何 .txt 標註檔案。")
        return
    for file in label_files:
        in_file = os.path.join(in_folder, file)
        with open(in_file, "r") as f:
            lines = f.readlines()
        bboxes = [seg_to_bbox(line) for line in lines]
        out_file = os.path.join(out_folder, file)
        with open(out_file, "w") as f:
            f.write("\n".join(bboxes))
        print(f"{file} 轉換完成。")
    print("所有檔案轉換完成。")


# 5. YOLOv8 轉 YOLOv3 格式
def cli_yolo8_to_yolo3(args):
    data_dir = args.data_dir
    yolov8_data_yaml_path = os.path.join(data_dir, "yolov8", "data.yaml")
    yolov3_notes_json_path = os.path.join(data_dir, "yolov3", "notes.json")
    yolov3_classes_txt_path = os.path.join(data_dir, "yolov3", "classes.txt")
    with open(yolov8_data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    categories = [{"id": idx, "name": name} for idx, name in enumerate(names)]
    info = {
        "year": datetime.now().year,
        "version": "1.0",
        "contributor": "Label Studio",
    }
    notes = {"categories": categories, "info": info}
    with open(yolov3_notes_json_path, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2, ensure_ascii=False)
    with open(yolov3_classes_txt_path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(f"{name}\n")
    print(
        "YOLOv8 轉 YOLOv3 完成！生成檔案：",
        yolov3_notes_json_path,
        "與",
        yolov3_classes_txt_path,
    )


# 6. YOLOv3 轉 YOLOv8 格式
def cli_yolo3_to_yolo8(args):
    data_dir = args.data_dir
    yolov3_classes_txt_path = os.path.join(data_dir, "classes.txt")
    yolov8_data_yaml_path = os.path.join(data_dir, "data.yaml")
    with open(yolov3_classes_txt_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    data_yaml_content = {
        "train": os.path.join(data_dir, "yolov8", "train", "images"),
        "val": os.path.join(data_dir, "yolov8", "valid", "images"),
        "nc": len(class_names),
        "names": class_names,
    }
    with open(yolov8_data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False)
    # 建立 annotations 的 symbolic link
    target = f"{data_dir}/labels"
    symlink = f"{data_dir}/annotations"
    if os.path.islink(symlink):
        os.remove(symlink)
    os.symlink(target, symlink)
    print("YOLOv3 轉 YOLOv8 完成！生成檔案：", yolov8_data_yaml_path)


# 7. 建立專案結構（使用 YoloFit 類別中的 create_proj 功能）
def cli_create_proj(args):
    proj_dir = Path(os.getcwd()) / "projects" / args.projname
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
        path = proj_dir / sub
        path.mkdir(parents=True, exist_ok=True)
        print("建立資料夾：", path)
    # 針對 yolov3 也建立 images 的 symbolic link
    yolov3_images_symlink = proj_dir / "dataset/yolov3/images"
    yolov8_train_images = proj_dir / "dataset/yolov8/train/images"
    if yolov3_images_symlink.exists():
        if yolov3_images_symlink.is_symlink():
            os.remove(yolov3_images_symlink)
    os.symlink(yolov8_train_images, yolov3_images_symlink)
    print("專案建立完成：", proj_dir)


# 8. 影片轉圖像（使用 YoloFit 中的 convert_video_to_image）
def cli_convert_video_to_image(args):
    proj_dir = Path(os.getcwd()) / "projects" / args.projname
    video_dir = proj_dir / "videos"
    frame_dir = proj_dir / "frames"
    import cv2
    import supervision as sv

    video_files = glob.glob(str(video_dir / "*.[mM][pP]4")) + glob.glob(
        str(video_dir / "*.[mM][oO][vV]")
    )
    if not video_files:
        print("專案內找不到影片檔案。")
        return
    for video_path in tqdm(video_files, desc="轉換影片為圖像"):
        cap = cv2.VideoCapture(video_path)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_name = f"{Path(video_path).stem}-{count:05d}.png"
            cv2.imwrite(str(frame_dir / img_name), frame)
            count += 1
        cap.release()
        print(f"{video_path} 轉換完成，共提取 {count} 幅影像。")


# 9. YOLO 訓練（使用 YOLOv8 模型訓練）
def cli_train_yolo(args):
    from ultralytics import YOLO

    data_yaml = os.path.join(
        os.getcwd(), "projects", args.projname, "dataset", "data.yaml"
    )
    model = YOLO("yolov8n.pt")
    print("開始訓練 YOLO 模型...")
    model.train(data=data_yaml, epochs=50, device=args.device)
    print("訓練完成！")


# 10. YOLO 預測（單張圖片預測）
def cli_predict_yolo(args):
    from ultralytics import YOLO
    import cv2

    model = YOLO("yolov8n.pt")
    image = cv2.imread(args.image)
    results = model(image)
    for result in results:
        print("預測結果：", result.boxes)


# 11. 資料集分割 (split_data)
def cli_split_data(args):
    from autodistill.helpers import split_data

    print(f"開始分割資料集：{args.data_dir}")
    split_data(
        args.data_dir,
        split_ratio=args.split_ratio,
        record_confidence=args.record_confidence,
    )
    print(f"資料集分割完成，split_ratio: {args.split_ratio}")


def main():
    parser = argparse.ArgumentParser(description="Label CLI 工具")
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="可用的子命令"
    )

    # label 命令：自動標註
    parser_label = subparsers.add_parser("label", help="對指定資料夾內的影像進行標註")
    parser_label.add_argument("input_folder", help="影像資料夾路徑")
    parser_label.add_argument(
        "--extension", default=".jpg", help="影像副檔名，預設 .jpg"
    )
    parser_label.add_argument(
        "--output_folder", help="輸出標註資料夾，預設在 input_folder+'_labeled'"
    )
    parser_label.add_argument(
        "--human_in_the_loop", action="store_true", help="是否啟用 human in the loop"
    )
    parser_label.add_argument("--roboflow_project", help="Roboflow 專案名稱")
    parser_label.add_argument(
        "--roboflow_tags", nargs="+", default=["autodistill"], help="Roboflow 標籤"
    )
    parser_label.add_argument("--sahi", action="store_true", help="是否啟用 SAHI")
    parser_label.add_argument(
        "--record_confidence", action="store_true", help="是否記錄信心分數"
    )
    parser_label.add_argument(
        "--nms_settings",
        default="no_nms",
        choices=["no_nms", "class_specific", "class_agnostic"],
        help="NMS 設定",
    )
    parser_label.set_defaults(func=cli_label)

    # load-grounding-dino 命令
    parser_dino = subparsers.add_parser(
        "load-grounding-dino", help="載入 GroundingDINO 模型"
    )
    parser_dino.set_defaults(func=cli_load_grounding_dino)

    # convert-yolo-to-labelstudio 命令
    parser_yolo_ls = subparsers.add_parser(
        "convert-yolo-to-labelstudio", help="將 YOLO 格式轉換為 Label Studio 格式"
    )
    parser_yolo_ls.add_argument("input_data_dir", help="YOLO 資料夾路徑")
    parser_yolo_ls.add_argument("image_root_url", help="圖片根 URL")
    parser_yolo_ls.add_argument(
        "--out_json_file", default="output_for_labelstudio.json", help="輸出 JSON 檔名"
    )
    parser_yolo_ls.set_defaults(func=cli_convert_yolo_to_labelstudio)

    # seg2bbox 命令
    parser_seg2bbox = subparsers.add_parser(
        "seg2bbox", help="將 segmentation 標註轉為 bounding box 標註"
    )
    parser_seg2bbox.add_argument("in_label_folder", help="輸入標註資料夾")
    parser_seg2bbox.add_argument(
        "out_label_folder",
        nargs="?",
        default=None,
        help="輸出標註資料夾（預設覆寫原檔）",
    )
    parser_seg2bbox.set_defaults(func=cli_seg2bbox)

    # yolo8-to-yolo3 命令
    parser_yolo8_to_yolo3 = subparsers.add_parser(
        "yolo8-to-yolo3", help="將 YOLOv8 格式轉換為 YOLOv3 格式"
    )
    parser_yolo8_to_yolo3.add_argument(
        "data_dir", help="包含 yolov8/data.yaml 的資料夾路徑"
    )
    parser_yolo8_to_yolo3.set_defaults(func=cli_yolo8_to_yolo3)

    # yolo3-to-yolo8 命令
    parser_yolo3_to_yolo8 = subparsers.add_parser(
        "yolo3-to-yolo8", help="將 YOLOv3 格式轉換為 YOLOv8 格式"
    )
    parser_yolo3_to_yolo8.add_argument("data_dir", help="包含 YOLOv3 檔案的資料夾路徑")
    parser_yolo3_to_yolo8.set_defaults(func=cli_yolo3_to_yolo8)

    # create-proj 命令：建立專案結構
    parser_create_proj = subparsers.add_parser(
        "create-proj", help="建立新的專案資料夾結構"
    )
    parser_create_proj.add_argument("projname", help="專案名稱")
    parser_create_proj.set_defaults(func=cli_create_proj)

    # convert-video-to-image 命令
    parser_video2img = subparsers.add_parser(
        "convert-video-to-image", help="將專案中的影片轉換成影像"
    )
    parser_video2img.add_argument("projname", help="專案名稱")
    parser_video2img.set_defaults(func=cli_convert_video_to_image)

    # train-yolo 命令
    parser_train_yolo = subparsers.add_parser(
        "train-yolo", help="使用 YOLOv8 模型進行訓練"
    )
    parser_train_yolo.add_argument("projname", help="專案名稱")
    parser_train_yolo.add_argument(
        "--device", default="cuda", help="運行裝置，如 cuda、mps 或 cpu"
    )
    parser_train_yolo.set_defaults(func=cli_train_yolo)
    """
    python app.py train-yolo abc --device=mps
    /Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/abc/dataset/project-9-at-2025-02-26-01-54-58557314
    """

    # predict-yolo 命令
    parser_predict_yolo = subparsers.add_parser(
        "predict-yolo", help="對單張圖片使用 YOLO 模型進行預測"
    )
    parser_predict_yolo.add_argument("image", help="影像檔案路徑")
    parser_predict_yolo.set_defaults(func=cli_predict_yolo)

    # split-data 命令：分割資料集 (使用 split_data)
    parser_split_data = subparsers.add_parser(
        "split-data", help="分割資料集成 train/valid/test"
    )
    parser_split_data.add_argument("data_dir", help="包含資料集的目錄路徑")
    parser_split_data.add_argument(
        "--split_ratio", type=float, default=0.8, help="資料分割比例，預設 0.8"
    )
    parser_split_data.add_argument(
        "--record_confidence", action="store_true", help="是否記錄信心分數"
    )
    parser_split_data.set_defaults(func=cli_split_data)
    """
    python app.py split-data --data_dir=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/projects/abc/dataset/project-9-at-2025-02-26-01-54-58557314
    """

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

