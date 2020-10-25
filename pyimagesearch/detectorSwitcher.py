from pyimagesearch.DetectorDETR import DetectorDetr
from pyimagesearch.DetectorHog import DetectorHog
from pyimagesearch.DetectorSSD import DetectorSSD
from pyimagesearch.DetectorYOLO import DetectorYOLO


def hog():
    return DetectorHog()


def ssd():
    return DetectorSSD()


def yolo():
    return DetectorYOLO()


def detr():
    return DetectorDetr()

def get_detector(arg):
    switcher = {
        "hog": hog,
        "ssd": ssd,
        "yolo": yolo,
        "detr": detr
    }
    func = switcher.get(arg, lambda: None)
    detector = func()
    return detector
