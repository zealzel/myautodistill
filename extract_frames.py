# First, create a directory to save your video frames
import os
import argparse

# Then, you can use Supervision's get_video_frames_generator function
import supervision as sv
from PIL import Image

parser = argparse.ArgumentParser(description="Extract frames from a video.")
parser.add_argument("video_path", type=str, help="Path to the video file")
args = parser.parse_args()

VIDEO_PATH = args.video_path
clipname = os.path.basename(VIDEO_PATH).split(".")[0]

# os.mkdir(FRAMES_DIR)
FRAMES_DIR = f"extract_frames/{clipname}"
os.makedirs(FRAMES_DIR)
print("FRAMES_DIR:", FRAMES_DIR)
# stride: int = 1,
# start: int = 0,
# end: Optional[int] = None,
fps = sv.VideoInfo.from_video_path(VIDEO_PATH).fps
# mediainfo --Inform="Video;%FrameRate%" /Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/try-autodistill/videos/milk-video-6.mov
interval = 200  # 100ms
stride = int(interval / fps)
print("FPS:", fps)
print("interval:", interval)
print("stride:", stride)
frames_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=stride)

for i, frame in enumerate(frames_generator):
    img = Image.fromarray(frame)
    img.save(f"{FRAMES_DIR}/video_frame{i}.jpg")

print(f"Saved frames to {FRAMES_DIR}")
