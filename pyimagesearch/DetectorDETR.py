import os
import torchvision.transforms as T
import torch
from PIL import Image

torch.set_grad_enabled(False)

from pyimagesearch import pedestrianTrackerConfig as config
from pyimagesearch.DETR import DETR
from pyimagesearch.Detector import Detector
from pyimagesearch.detectionDetr import detect, filter_boxes


class DetectorDetr(Detector):
    def __init__(self):
        self.detr = DETR(num_classes=91)
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = os.path.join(config.DETR_PATH, config.DETR_MODEL)
        state_dict = torch.load(checkpoint)
        self.detr.load_state_dict(state_dict)
        self.detr.eval().to(self.DEVICE)
        self.transform = T.Compose([
                            T.Resize(500),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect(self, frame):
        im = Image.fromarray(frame)
        boxes, scores = detect(im, self.detr, self.transform, self.DEVICE)
        return boxes, scores
