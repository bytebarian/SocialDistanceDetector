# USAGE
# python social_distance_detector.py --input pedestrians.mp4
# python social_distance_detector.py --input pedestrians.mp4 --output output.avi

import argparse

import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist

# import the necessary packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detectorSwitcher import get_detector
from pyimagesearch.perspectiveTransformation import get_perspective_matrix, point_transform, image_transform
from utils.parser import get_config
from deep_sort import DeepSort
import torch

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="pedestrians.mp4",
                help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="output.avi",
                help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
                help="whether or not output frame should be displayed")
ap.add_argument("-a", "--algorithm", type=str, default="detr",
                help="choose what kind of algorithm should be used for people detection")
ap.add_argument("-c", "--coords", type=str, default="[(320, 50), (690, 80), (579, 313), (3, 193)]",
                help="comma seperated list of source points")
ap.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
args = vars(ap.parse_args())

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None

cfg = get_config()
cfg.merge_from_file(args["config_deepsort"])
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

# loop over the frames from the video stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    pts = np.array(eval(args["coords"]), dtype="float32")

    (h, w) = frame.shape[:2]
    pts_dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(pts, pts_dst)
    birdview = cv2.warpPerspective(frame, matrix, (w, h))

    detector = get_detector(args["algorithm"])
    boxes, scores = detector.detect(frame)

    bbox_xcycwh = []
    confs = []
    for (i, bbox) in enumerate(boxes):
        # extract the bounding box and centroid coordinates, then
        # initialize the color of the annotation
        startX = bbox[0]
        startY = bbox[1]
        w = bbox[2]
        h = bbox[3]
        cX = startX + int(w / 2)
        cY = startY + int(h / 2)
        bbox_xcycwh.append([cX, cY, w, h])
        confs.append([scores[i]])

    xywhs = torch.Tensor(bbox_xcycwh)
    confss = torch.Tensor(scores)

    # Pass detections to deepsort
    outputs = deepsort.update(xywhs, confss, frame)

    # initialize the set of indexes that violate the minimum social
    # distance
    violate = set()

    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        # loop over the results
        for (i, bbox) in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(i) for i in bbox]
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
            color = (0, 255, 0)
            id = int(identities[i]) if identities is not None else 0
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            # if the index pair exists within the violation set, then
            # update the color
            # if i in violate:
            #    color = (0, 0, 255)

            # draw the centroid coordinates of the person,
            cv2.circle(frame, (cX, cY), 5, color, 1)
            cv2.circle(frame, (cX, y2), 5, color, 1)
            cv2.line(frame, (cX, cY), (cX, y2), color, 1)
            cv2.putText(frame, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

            cv2.circle(frame, (320, 50), 5, (0, 0, 255), -1)
            cv2.circle(frame, (690, 80), 5, (0, 0, 255), -1)
            cv2.circle(frame, (579, 313), 5, (0, 0, 255), -1)
            cv2.circle(frame, (3, 193), 5, (0, 0, 255), -1)

            # calculate pedestrians position from birdview
            p = (cX, y2)
            px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
                        matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
            py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
                        matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
            p_after = (int(px), int(py))

            cv2.circle(birdview, p_after, 5, color, 1)
            cv2.circle(birdview, p_after, 30, color, 1)

    # ensure there are *at least* two people detections (required in
    # order to compute our pairwise distance maps)
    #if len(boxes) >= 2:
        # extract all centroids from the results and compute the
        # Euclidean distances between all pairs of the centroids
    #    centroids = np.array([r[1] for r in results])
    #    D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
    #    for i in range(0, D.shape[0]):
    #        for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two
                # centroid pairs is less than the configured number
                # of pixels
    #            if D[i, j] < config.MIN_DISTANCE:
                    # update our violation set with the indexes of
                    # the centroid pairs
     #               violate.add(i)
     #               violate.add(j)

    # draw the total number of social distancing violations on the
    # output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # check to see if the output frame should be displayed to our
    # screen
    if args["display"] > 0:
        # show the output frame
        cv2.imshow("Frame", frame)
        cv2.imshow("Birdview", birdview)
        np.concatenate((frame, birdview), axis=1)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                 (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)
