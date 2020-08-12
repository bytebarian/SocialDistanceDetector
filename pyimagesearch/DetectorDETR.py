import os

import torch

torch.set_grad_enabled(False)

from pyimagesearch import social_distancing_config as config
from pyimagesearch.DETR import DETR
from pyimagesearch.Detector import Detector
from pyimagesearch.detectionDetr import detect_people


class DetectorSSD(Detector):
    def __init__(self):
        self.detr = DETR(num_classes=91)
        checkpoint = os.path.join(config.DETR_PATH, config.DETR_MODEL)
        state_dict = torch.load(checkpoint)
        self.detr.load_state_dict(state_dict)
        self.detr.eval()

    def detect(self, frame):
        result = detect_people(frame, self.detr)
        return result
