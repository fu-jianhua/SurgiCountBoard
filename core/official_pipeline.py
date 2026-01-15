import time
from collections import defaultdict
import numpy as np
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors

class MyObjectCounter(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        _min_track_frames = kwargs.pop("min_track_frames", 3)
        _min_conf = kwargs.pop("min_conf", 0.25)
        _min_area_ratio = kwargs.pop("min_area_ratio", 0.0005)
        _entry_dedup_time = kwargs.pop("entry_dedup_time", 1.0)
        _entry_dedup_dist = kwargs.pop("entry_dedup_dist", 20.0)
        _count_mode = kwargs.pop("count_mode", "roi")
        _line_pos = kwargs.pop("line_pos", 0.5)
        _count_init_inside = kwargs.pop("count_init_inside", True)
        super().__init__(**kwargs)

        self.in_count = 0
        self.out_count = 0
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})
        self.region_initialized = False

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2

        self.track_states = {}
        self._entry_events = []
        self._recent_entries = []
        self.min_track_frames = _min_track_frames
        self.min_conf = _min_conf
        self.min_area_ratio = _min_area_ratio
        self.entry_dedup_time = _entry_dedup_time
        self.entry_dedup_dist = _entry_dedup_dist
        self.count_mode = _count_mode
        self.line_pos = _line_pos
        self.count_init_inside = _count_init_inside
        self._im_w = None
        self._im_h = None

    def _rect_bounds(self):
        xs = [p[0] for p in self.region]
        ys = [p[1] for p in self.region]
        return min(xs), min(ys), max(xs), max(ys)

    def _inside(self, pt):
        x1, y1, x2, y2 = self._rect_bounds()
        return pt[0] >= x1 and pt[0] <= x2 and pt[1] >= y1 and pt[1] <= y2

    def _inside_shrink(self, pt, m):
        x1, y1, x2, y2 = self._rect_bounds()
        return pt[0] >= (x1 + m) and pt[0] <= (x2 - m) and pt[1] >= (y1 + m) and pt[1] <= (y2 - m)

    def _bbox_area_ratio(self, box):
        if self._im_w is None or self._im_h is None:
            return 1.0
        w = max(0.0, float(box[2]) - float(box[0]))
        h = max(0.0, float(box[3]) - float(box[1]))
        area = w * h
        total = float(self._im_w * self._im_h)
        return (area / total) if total > 0 else 0.0

    def _dedup_recent_entry(self, cls_id, cx, cy, ts):
        keep = []
        for e in self._recent_entries:
            if ts - e[0] <= self.entry_dedup_time:
                keep.append(e)
        self._recent_entries = keep
        for e in self._recent_entries:
            if e[1] == cls_id:
                dx = e[2] - cx
                dy = e[3] - cy
                if (dx * dx + dy * dy) ** 0.5 <= self.entry_dedup_dist:
                    return True
        self._recent_entries.append((ts, int(cls_id), float(cx), float(cy)))
        return False

    def count_objects(
        self,
        current_centroid,
        track_id,
        prev_position,
        cls,
        conf,
        box,
    ) -> None:
        prev_inside = self.track_states.get(int(track_id), {}).get("inside", None)
        curr_inside = self._inside(current_centroid)
        curr_inside_s = self._inside_shrink(current_centroid, self.margin)
        frames = len(self.track_history[track_id]) if track_id in self.track_history else 0
        area_ratio = self._bbox_area_ratio(box)
        ts = time.time()
        if frames >= int(self.min_track_frames) and float(conf) >= float(self.min_conf) and float(area_ratio) >= float(self.min_area_ratio):
            if prev_inside is None and self.count_init_inside and curr_inside_s is True:
                if not self._dedup_recent_entry(int(cls), float(current_centroid[0]), float(current_centroid[1]), ts):
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                    self._entry_events.append((ts, int(cls), int(track_id), float(conf), float(current_centroid[0]), float(current_centroid[1])))
            elif prev_inside is False and curr_inside_s is True:
                if not self._dedup_recent_entry(int(cls), float(current_centroid[0]), float(current_centroid[1]), ts):
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                    self._entry_events.append((ts, int(cls), int(track_id), float(conf), float(current_centroid[0]), float(current_centroid[1])))
            elif prev_inside is True and curr_inside is False:
                self.out_count += 1
                self.classwise_count[self.names[cls]]["OUT"] += 1
        self.track_states[int(track_id)] = {"inside": curr_inside}

    def display_counts(self, plot_im) -> None:
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_count.items()
            if value["IN"] != 0 or (value["OUT"] != 0 and (self.show_in or self.show_out))
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0, **kwargs) -> SolutionResults:
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)
        self._entry_events = []
        self._im_h = int(im0.shape[0])
        self._im_w = int(im0.shape[1])

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region

        target_infos = []

        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            target_info = {
                "track_id": int(track_id),
                "class_id": int(cls),
                "class_name": self.names[cls],
                "confidence": float(conf),
                "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                "centroid": self.track_history[track_id][-1] if track_id in self.track_history else None
            }
            target_infos.append(target_info)
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)

            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls, conf, box)

        plot_im = self.annotator.result()
        self.display_counts(plot_im)
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
            target_infos=target_infos
        )
    
class OfficialPipeline:
    def __init__(self, model_path, conf=0.25, iou=0.45, device="cpu", half=False, imgsz=640, max_det=200, use_track=True, line_pos=0.6, count_mode="roi"):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.max_det = max_det
        self.use_track = use_track
        self.line_pos = float(line_pos)
        self.count_mode = str(count_mode)
        self._sol = None
        self._mode = None
        self._seen_tracks = set()
        self._prev_in_by_class = {}

    def _roi_points(self, roi):
        x1, y1, x2, y2 = roi.x1, roi.y1, roi.x2, roi.y2
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def _ensure_solution(self, frame, roi):
        region = self._roi_points(roi)
        if self.use_track:
            self._sol = MyObjectCounter(show=False, region=region, model=self.model_path, tracker="custom_bytetrack.yaml", min_track_frames=2, min_conf=self.conf, min_area_ratio=0.0005, entry_dedup_time=1.0, entry_dedup_dist=20.0, count_mode=self.count_mode, line_pos=self.line_pos, count_init_inside=True)
            print("use_track: True!!!!!!!!!!!!!!!!!")
        else:
            self._sol = MyObjectCounter(show=False, region=region, model=self.model_path, min_track_frames=2, min_conf=self.conf, min_area_ratio=0.0005, entry_dedup_time=1.0, entry_dedup_dist=20.0, count_mode=self.count_mode, line_pos=self.line_pos, count_init_inside=True)
        self._mode = "hybrid"
        

    def process(self, frame, roi, tap):
        if self._sol is None:
            self._ensure_solution(frame, roi)
        results = self._sol.process(frame, tap=tap)
        annotated = getattr(results, "plot_im", frame)
        counts = {}
        
        target_infos = getattr(results, "target_infos", [])
        entry_events = getattr(self._sol, "_entry_events", [])
        
        cw = getattr(results, "classwise_count", None)
        if isinstance(cw, dict) and cw:
            for k, v in cw.items():
                try:
                    cur = int(v.get("IN", 0))
                except Exception:
                    cur = int(v) if isinstance(v, (int, float)) else 0
                prev = int(self._prev_in_by_class.get(str(k), 0))
                delta = cur - prev
                if delta > 0:
                    counts[str(k)] = counts.get(str(k), 0) + delta
                self._prev_in_by_class[str(k)] = cur
        rc = getattr(results, "region_counts", {})
        roi_det = any(isinstance(c, (int, float)) and c > 0 for c in rc.values())
    
        events = entry_events
        return annotated, counts, events, roi_det, target_infos

