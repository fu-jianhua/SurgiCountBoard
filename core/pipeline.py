import time
import cv2
import numpy as np
from ultralytics import YOLO

class Pipeline:
    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, classes=None, use_track: bool = True, tracker: str = "bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.use_track = use_track
        self.tracker = tracker
        self.names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names

    def process(self, frame: np.ndarray, roi_rect):
        if self.use_track:
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                persist=True,
                tracker=self.tracker,
                verbose=False,
            )
        else:
            results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.classes)
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
        clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.cls is not None else np.empty((0,), dtype=int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.id is not None else np.empty((0,), dtype=int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else np.empty((0,), dtype=float)
        ts = time.time()
        counts = {}
        events = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if roi_rect is None or roi_rect.contains(cx, cy):
                name = self.names[clss[i]] if clss[i] < len(self.names) else str(clss[i])
                counts[name] = counts.get(name, 0) + 1
                events.append((ts, int(clss[i]), int(ids[i]) if i < len(ids) else -1, float(confs[i]) if i < len(confs) else 0.0))
        annotated = r.plot()
        if roi_rect is not None:
            if hasattr(roi_rect, "points"):
                pts = np.array(roi_rect.points, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (104, 0, 123), 2)
            else:
                cv2.rectangle(annotated, (roi_rect.x1, roi_rect.y1), (roi_rect.x2, roi_rect.y2), (104, 0, 123), 2)
        return annotated, counts, events