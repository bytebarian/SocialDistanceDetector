import cv2

from pyimagesearch.Detector import Detector
from pyimagesearch.detectionHog import detect_people


class DetectorHog(Detector):
    def __init__(self):
        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        result = detect_people(frame, self.hog)
        return result
