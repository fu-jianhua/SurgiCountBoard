import time
from collections import defaultdict
import numpy as np
from typing import Any

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class MyObjectCounter(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})  # Dictionary for counts, categorized by class
        self.region_initialized = False  # Flag indicating whether the region has been initialized

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.margin = self.line_width * 2  # Scales the background rectangle size to display counts properly

    def count_objects(
        self,
        current_centroid: tuple[float, float],
        track_id: int,
        prev_position: tuple[float, float] | None,
        cls: int,
    ) -> None:
        if prev_position is None or track_id in self.counted_ids:
            return

        if self.r_s.contains(self.Point(current_centroid)):
            # Determine motion direction for vertical or horizontal polygons
            region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
            region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)

            if (region_width < region_height and current_centroid[0] > prev_position[0]) or (
                region_width >= region_height and current_centroid[1] > prev_position[1]
            ):  # Moving right or downward
                self.in_count += 1
                self.classwise_count[self.names[cls]]["IN"] += 1
            else:  # Moving left or upward
                self.out_count += 1
                self.classwise_count[self.names[cls]]["OUT"] += 1
            self.counted_ids.append(track_id)

    def display_counts(self, plot_im) -> None:
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_count.items()
            if value["IN"] != 0 or (value["OUT"] != 0 and (self.show_in or self.show_out))
        }
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0, tap) -> SolutionResults:
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)  # Extract tracks
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region

        target_infos = []

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # 收集目标信息
            target_info = {
                "track_id": int(track_id),
                "class_id": int(cls),
                "class_name": self.names[cls],
                "confidence": float(conf),
                "bbox": box.tolist() if hasattr(box, 'tolist') else box,
                "centroid": self.track_history[track_id][-1] if track_id in self.track_history else None
            }
            target_infos.append(target_info)
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # Store track history

            # Store previous position of track for object counting
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # object counting

        plot_im = self.annotator.result()
        # self.display_counts(plot_im)  # Display the counts on the frame
        self.display_output(plot_im)  # Display output with base class function
        print(f"tap: {tap}, in_count: {self.in_count}, out_count: {self.out_count}")

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
            target_infos=target_infos
        )
      
class OfficialPipeline:
    def __init__(self, model_path, conf=0.25, iou=0.45, device="cpu", half=False, use_track=True, line_pos=0.6, count_mode="roi"):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.half = half
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
            self._sol = MyObjectCounter(show=False, region=region, model=self.model_path, tracker="custom_bytetrack.yaml")
            print("use_track: True!!!!!!!!!!!!!!!!!")
        else:
            self._sol = MyObjectCounter(show=False, region=region, model=self.model_path)
        self._mode = "line"
        

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
        print(f"!!!!!entry_events: {entry_events}, counts: {counts}, events: {events}, roi_det: {roi_det}, target_infos: {target_infos}")
        return annotated, counts, events, roi_det, target_infos

