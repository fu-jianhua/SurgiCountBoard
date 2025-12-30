import time
import cv2
import numpy as np
from ultralytics import YOLO
from .counter import InstrumentCounter

class Pipeline:
    def __init__(
        self,
        model_path: str = None,
        model=None,
        conf: float = 0.25,
        iou: float = 0.45,
        classes=None,
        use_track: bool = True,
        tracker: str = "bytetrack.yaml",
        device: str | int = 0,
        half: bool = True,
        imgsz: int = 640,
        max_det: int = 200,
    ):
        self.model = model if model is not None else YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.classes = classes
        self.use_track = use_track
        self.tracker = tracker
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.max_det = max_det
        self.names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
        self.counter = None

    def process(self, frame: np.ndarray, roi_rect):
        if self.use_track:
            results = self.model.track(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                persist=True,
                tracker=self.tracker,
                device=self.device,
                half=self.half,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
        else:
            results = self.model(
                frame,
                conf=self.conf,
                iou=self.iou,
                classes=self.classes,
                device=self.device,
                half=self.half,
                imgsz=self.imgsz,
                max_det=self.max_det,
            )
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
        clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.cls is not None else np.empty((0,), dtype=int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.id is not None else np.empty((0,), dtype=int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else np.empty((0,), dtype=float)
        ts = time.time()
        if roi_rect is not None:
            if hasattr(roi_rect, "points"):
                roi_obj = roi_rect
            else:
                roi_obj = (roi_rect.x1, roi_rect.y1, roi_rect.x2, roi_rect.y2)
        else:
            roi_obj = None
        if self.counter is None:
            self.counter = InstrumentCounter(roi_obj)
        else:
            self.counter.roi = roi_obj
        counts = {}
        events = []
        roi_present = False
        tid_to_conf = {}
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if roi_rect is None or roi_rect.contains(cx, cy):
                roi_present = True
            tid = int(ids[i]) if i < len(ids) else -1
            if tid >= 0:
                tid_to_conf[tid] = float(confs[i]) if i < len(confs) else 0.0
                track = type("Track", (), {})()
                track.track_id = tid
                track.tlbr = [float(x1), float(y1), float(x2), float(y2)]
                track.cls = int(clss[i]) if i < len(clss) else 0
                prev_counted = tid in self.counter.counted_ids
                self.counter.update(track)
                counted, final_cls = self.counter.try_count(track)
                if counted and not prev_counted:
                    name = self.names[final_cls] if final_cls < len(self.names) else str(final_cls)
                    counts[name] = self.counter.class_counter[final_cls]
                    events.append((ts, int(final_cls), tid, tid_to_conf.get(tid, 0.0)))
        if not events:
            for cls_id, v in self.counter.class_counter.items():
                name = self.names[cls_id] if cls_id < len(self.names) else str(cls_id)
                counts[name] = v
        annotated = r.plot()
        if roi_rect is not None:
            if hasattr(roi_rect, "points"):
                pts = np.array(roi_rect.points, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (104, 0, 123), 2)
            else:
                cv2.rectangle(annotated, (roi_rect.x1, roi_rect.y1), (roi_rect.x2, roi_rect.y2), (104, 0, 123), 2)
        return annotated, counts, events, roi_present