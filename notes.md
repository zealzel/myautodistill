# create project

create function create_proj to fullfil function below

under projects folder create file structure based on projname

projects/projname
├── dataset/
| ├── train/
| | ├── images/
| | └── labels/
| ├── valid/
| | ├── images/
| | └── labels/
├── frames/
├── runs/
└── videos/

# typical workflow

raw video -> extract frames -> convert frames to images -> train -> test

## raw video

ex:
video/IMG_2872.MOV

## extract_frames

extract frames from video into folder projects/projname/frames

## autolabel images

under projects/dataset/projname/
train/
├── images/
└── labels/
valid/
├── images/
└── labels/

## split images into train/val/test sets

based on test_val_ratio, split images from frames/ folder into train/val/test sets
where

projects/projname/dataset/train/images

# dataset

The train/val/test data are in the folder under dataset

dataset/project/
