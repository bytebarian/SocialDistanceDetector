import os

import cv2

from pyimagesearch import social_distancing_config as config
from pyimagesearch.Detector import Detector
from pyimagesearch.detectionSSD import detect_people


class DetectorSSD(Detector):
    def __init__(self):
        # derive the paths to the SSD prototxt and model
        prototxt = os.path.sep.join([config.SSD_MODEL_PATH, config.SSD_PROTOTXT])
        model = os.path.sep.join([config.SSD_MODEL_PATH, config.SSD_MODEL])
        self.net = net = cv2.dnn.readNetFromCaffe(prototxt, model)

    def detect(self, frame):
        result = detect_people(frame, self.net)
        return result
