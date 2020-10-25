from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2


def detect_people(frame, hog):
    results = []
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        # update our results list to consist of the person
        # prediction probability, bounding box coordinates,
        # and the centroid
        r = ((x, y, x + w, y + h), (int(x + (w / 2)), int(y + (h / 2))))
        results.append(r)

    # return the list of results
    return results
