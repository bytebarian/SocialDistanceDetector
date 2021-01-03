import os

import cv2

from pyimagesearch import pedestrianTrackerConfig as config
from pyimagesearch.Detector import Detector
from pyimagesearch.detectionYOLO import detect_people


class DetectorYOLO(Detector):
    def __init__(self):
        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
        configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        # check if we are going to use GPU
        if config.USE_GPU:
            # set CUDA as the preferable backend and target
            print("[INFO] setting preferable backend and target to CUDA...")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # determine only the *output* layer names that we need from YOLO
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        boxes, confidences = detect_people(frame, self.net, self.ln, 0)
        return boxes, confidences
