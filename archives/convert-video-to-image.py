import os
import cv2
import supervision as sv
from tqdm.notebook import tqdm

HOME = os.getcwd()
print(HOME)

VIDEO_DIR_PATH = f"{HOME}/videos"
IMAGE_DIR_PATH = f"{HOME}/images"
FRAME_STRIDE = 10

video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH, extensions=["mov", "mp4"]
)

TEST_VIDEO_PATHS, TRAIN_VIDEO_PATHS = video_paths[:2], video_paths[2:]


def convert_video_to_image():
    for video_path in tqdm(TRAIN_VIDEO_PATHS):
        video_name = video_path.stem
        image_name_pattern = video_name + "-{:05d}.png"
        with sv.ImageSink(
            target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern
        ) as sink:
            for image in sv.get_video_frames_generator(
                source_path=str(video_path), stride=FRAME_STRIDE
            ):
                sink.save_image(image=image)


def list_files_with_extensions():
    image_paths = sv.list_files_with_extensions(
        directory=IMAGE_DIR_PATH, extensions=["png", "jpg", "jpg"]
    )
    print("image count:", len(image_paths))
    return image_paths


MAGE_DIR_PATH = f"{HOME}/images"
SAMPLE_SIZE = 16
SAMPLE_GRID_SIZE = (4, 4)
SAMPLE_PLOT_SIZE = (16, 10)


def plot_sample(image_paths):
    titles = [image_path.stem for image_path in image_paths[:SAMPLE_SIZE]]
    images = [cv2.imread(str(image_path)) for image_path in image_paths[:SAMPLE_SIZE]]
    sv.plot_images_grid(
        images=images, titles=titles, grid_size=SAMPLE_GRID_SIZE, size=SAMPLE_PLOT_SIZE
    )


# convert_video_to_image()
image_paths = list_files_with_extensions()
plot_sample(image_paths)
