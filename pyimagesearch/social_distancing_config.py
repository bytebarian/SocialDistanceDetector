# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# base path to ssd model
SSD_MODEL_PATH = "ssd"
SSD_PROTOTXT = "MobileNetSSD_deploy.prototxt.txt"
SSD_MODEL = "MobileNetSSD_deploy.caffemodel"

# base path to DETR model
DETR_PATH = "detr"
DETR_MODEL = "detr_demo-da2a99e9.pth"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE = 50