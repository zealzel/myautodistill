import os
import supervision as sv
from pathlib import Path


HOME = os.getcwd()
ANNOTATIONS_DIRECTORY_PATH = f"{HOME}/dataset/train/labels"
IMAGES_DIRECTORY_PATH = f"{HOME}/dataset/train/images"
DATA_YAML_PATH = f"{HOME}/dataset/data.yaml"

IMAGE_DIR_PATH = f"{HOME}/images"
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 10)

dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH,
)

len(dataset)


def display_annotation():
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


display_annotation()
