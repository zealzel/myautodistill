import os

# from tqdm import tqdm
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

HOME = os.getcwd()
VIDEO_DIR_PATH = f"{HOME}/videos"
IMAGE_DIR_PATH = f"{HOME}/images"
FRAME_STRIDE = 10

# DATASET_NAME = "taco-trash-annotations-in-context"

print(HOME)


# Initiate base model and autolabel
def init_base_model_autolabel():
    ontology = CaptionOntology({"milk bottle": "bottle", "blue cap": "cap"})
    DATASET_DIR_PATH = f"{HOME}/dataset"

    base_model = GroundedSAM(ontology=ontology)
    dataset = base_model.label(
        input_folder=IMAGE_DIR_PATH, extension=".png", output_folder=DATASET_DIR_PATH
    )
    return base_model, dataset


base_model, dataset = init_base_model_autolabel()
