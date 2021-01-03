# USAGE
# python pdestrianTracker.py --input pedestrians.mp4
# python pdestrianTracker.py --input pedestrians.mp4 --output output.avi

import argparse

import cv2
import imutils
import numpy as np
import torch
import time
import pymongo

from deep_sort import DeepSort
# import the necessary packages
from pyimagesearch.detectorSwitcher import get_detector
from tracking import CentroidTracker
from utils.parser import get_config
from datetime import datetime

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

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
ap.add_argument("-c", "--coords", type=str, default="[(427, 0), (700, 0), (530, 318), (1, 198)]",
                help="comma seperated list of source points")
ap.add_argument("-s", "--size", type=str, default="[700,400]",
                help="coma separated tuple describing height and width of transformed birdview image")
ap.add_argument("-t", "--time", type=int, default=0,
                help="set the initial timestamp of video stream start in unix format in nanoseconds")
ap.add_argument("r" "--object", type=str, default="person",
                help="objects to detect and track")
args = vars(ap.parse_args())

# initialize the video stream and pointer to output video file
print("[INFO] accessing video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
initial_time = args["time"] if args["time"] > 0 else time.time_ns()

writer = None

trajectories = {}
ct = CentroidTracker()

mongoclient = pymongo.MongoClient("mongodb://localhost:27017/")
mongodb = mongoclient["massmove"]
mongocol = mongodb["points"]

token = "uw400lj_tKeWjbTdwM4VJz_qZ2MnpsOh5zeBdP3BKS7Au4NaOVSpePcd1Zj47bsNdBtmqCt9Gf5u1UHvWiFYgg=="
org = "ikillforfood@gmail.com"
bucket = "points"
influxclient = InfluxDBClient(url="https://westeurope-1.azure.cloud2.influxdata.com", token=token)
write_api = influxclient.write_api(write_options=SYNCHRONOUS)

cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def draw_markers():
    global i, bbox, cX, cY, id
    color = (0, 255, 0)
    if len(outputs) > 0:
        bbox_xyxy = outputs[:, :4]
        identities = outputs[:, -1]
        # loop over the results
        for (i, bbox) in enumerate(bbox_xyxy):
            x1, y1, x2, y2 = [int(i) for i in bbox]
            cX = int((x1 + x2) / 2)
            cY = int((y1 + y2) / 2)
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

            draw_markers_on_birdview(cX, color, y2, id)


def draw_markers_on_birdview(cX, color, y2, id):
    # calculate pedestrians position from birdview
    p = (cX, y2)
    px = (matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / (
            matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    py = (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / (
            matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2])
    p_after = (int(px), int(py))
    cv2.circle(birdview, p_after, 5, color, 1)
    cv2.circle(birdview, p_after, 30, color, 1)

    add_to_pedestrian_trajectory(id, p_after)


def add_to_pedestrian_trajectory(id, p_after):
    if id in trajectories:
        trajectories[id].add(p_after)
    else:
        trajectory = set()
        trajectory.add(p_after)
        trajectories[id] = trajectory

    (x, y) = p_after

    point = Point("mem").tag("user", id).field("point", f'[{x}, {y}]').time(datetime.utcnow(), WritePrecision.NS)
    write_api.write(bucket, org, point)

    dict = {
       "id": id,
        "point": [x, y],
        "time": timestamp
    }
    mongocol.insert_one(dict)


def calculate_bird_view():
    global h, w, size, matrix, birdview
    pts = np.array(eval(args["coords"]), dtype="float32")
    (h, w) = frame.shape[:2]
    if args["size"] != "":
        size = np.array(eval(args["size"]), dtype="int")
        (h, w) = size[:2]
    pts_dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix = cv2.getPerspectiveTransform(pts, pts_dst)
    # birdview = np.zeros((h, w, 3), np.uint8)
    birdview = cv2.warpPerspective(frame, matrix, (w, h));


def calculate_bboxs_confs():
    global i, bbox, w, h, cX, cY
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


def track(boxes):
    bboxes = []
    for (i, bbox) in enumerate(boxes):
        startX = bbox[0]
        startY = bbox[1]
        endX = startX + bbox[2]
        endY = startY + bbox[3]
        box = np.array([startX, startY, endX, endY])
        bboxes.append(box.astype("int"))

    objects = ct.update(bboxes)
    return objects


def draw_markers_alternate():
    global text
    color = (0, 255, 0)
    # loop over the tracked objects
    for (objectID, centroid) in outputs.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        draw_markers_on_birdview(centroid[0], color, centroid[1], objectID)


# loop over the frames from the video stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()
    timestamp = initial_time + (vs.get(cv2.CAP_PROP_POS_MSEC) * 1000000)

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break
    # resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)

    calculate_bird_view()

    detector = get_detector(args["algorithm"])
    boxes, scores = detector.detect(frame)

    bbox_xcycwh = []
    confs = []
    calculate_bboxs_confs()

    xywhs = torch.Tensor(bbox_xcycwh)
    confss = torch.Tensor(scores)

    # Pass detections to deepsort
    #outputs = deepsort.update(xywhs, confss, frame)
    outputs = track(boxes)

    # initialize the set of indexes that violate the minimum social
    # distance
    violate = set()

    #draw_markers()
    draw_markers_alternate()

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
