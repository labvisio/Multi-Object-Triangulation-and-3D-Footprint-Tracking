import logging

import numpy as np
from ultralytics import YOLO

from bbox_utils import get_centroid
from detection import ObjectDetection


class Tracker:
    def __init__(self, yolo_trackers_list, cam_list, class_list, yolo_confidence):
        self.cam_list = cam_list
        self.trackers = [YOLO(tracker) for tracker in yolo_trackers_list]
        self.results = []
        self.frames = None
        self.classes = class_list
        self.confidence = yolo_confidence

        logging.info(f"--------- {yolo_trackers_list} Tracker initialized with {len(cam_list)} cameras ---------")

    def detect_and_track(self, cameras_frame: list):
        self.frames = cameras_frame
        self.results = []

        for tracker, frame in zip(self.trackers, self.frames):
            result = tracker.track(
                frame,
                persist=True,
                classes=self.classes,
                device="cuda:0",
                conf=self.confidence,
                verbose=False,
                show=False,
                cls=True,
                tracker="source/yolo_conf.yaml",
                )
            self.results.append(result[0])

    def get_detections(self, show=False):
        detections = []
        # print(self.results)

        for i in range(len(self.results)):
            # print(result[i].boxes.xyxy.cpu().numpy())
            if self.results[i] is not None and self.results[i].boxes.id is not None:
                ids = self.results[i].boxes.id.cpu().numpy()
                bbox = self.results[i].boxes.xyxy.cpu().numpy()

                centroid = get_centroid(bbox)

                name = self.results[i].boxes.cls.cpu().numpy()
                for j, b, n, centroid in zip(ids, bbox, name, centroid):
                    detections.append(
                        ObjectDetection(
                            self.cam_list[i], j, b, self.frames[i], None, centroid, n
                        )
                    )

        return detections