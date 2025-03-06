from label_studio_sdk.converter import Converter

"""
:param config: string or dict: XML string with Label studio labeling config or path to this file or parsed_config
:param project_dir: upload root directory for images, audio and other labeling files
"""

config = "dataset/my-first-customed/data.yaml"
project_dir = ""

cv = Converter()

"""
# official repo from deprecated label-studio-converter
python label-studio-converter export -i exported_tasks.json -c examples/sentiment_analysis/config.xml -o output_dir -f CSV

# tutorials
label-studio-converter import yolo -i /yolo/datasets/one -o output.json --image-root-url "/data/local-files/?d=one/images"


procedures

export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed

http://localhost:8081/data/local-files/?d=train/images/IMG_2872-00000.jpg

label-studio-converter import yolo -i /Users/zealzel/Documents/Codes/Current/ai/machine-vision/yolo-learn/myautodistill/dataset/my-first-customed \
  -o output.json --image-root-url "/data/local-files/?d=train/images"


"""
