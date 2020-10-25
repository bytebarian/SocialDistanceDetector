import cv2
import numpy as np

from .social_distancing_config import MIN_CONF


def detect_people(frame, net):

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    results = []

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # extract the index of the class label from the detections
        idx = int(detections[0, 0, i, 1])
        # filter detections by (1) ensuring that the object
        # detected was a person and (2) that the minimum
        # confidence is met
        if idx == 15 and confidence > MIN_CONF:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            width = endX - startX
            height = endY - startY
            centerX = int(startX + (width / 2))
            centerY = int(startY + (height / 2))

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = ((startX, startY, int(startX + width), int(startY + height)), (centerX, centerY))
            results.append(r)

    # return the list of results
    return results