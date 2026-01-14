import time
from typing import Any
from collections import defaultdict
import numpy as np

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

        if len(self.region) == 2:  # Linear region (defined as a line segment)
            if self.r_s.intersects(self.LineString([prev_position, current_centroid])):
                # Determine orientation of the region (vertical or horizontal)
                if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                    # Vertical region: Compare x-coordinates to determine direction
                    if current_centroid[0] > prev_position[0]:  # Moving right
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                    else:  # Moving left
                        self.out_count += 1
                        self.classwise_count[self.names[cls]]["OUT"] += 1
                # Horizontal region: Compare y-coordinates to determine direction
                elif current_centroid[1] > prev_position[1]:  # Moving downward
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                else:  # Moving upward
                    self.out_count += 1
                    self.classwise_count[self.names[cls]]["OUT"] += 1
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # Polygonal region
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

    def process(self, im0) -> SolutionResults:
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
        self.display_counts(plot_im)  # Display the counts on the frame
        self.display_output(plot_im)  # Display output with base class function

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
            target_infos=target_infos
        )
    
        
class MyRegionCounter(BaseSolution):
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the RegionCounter for real-time object counting in user-defined regions."""
        super().__init__(**kwargs)
        self.region_template = {
            "name": "Default Region",
            "polygon": None,
            "counts": 0,
            "region_color": (255, 255, 255),
            "text_color": (0, 0, 0),
        }
        self.region_counts = {}
        self.counting_regions = []
        self.initialize_regions()

    def add_region(
        self,
        name: str,
        polygon_points: list[tuple],
        region_color: tuple[int, int, int],
        text_color: tuple[int, int, int],
    ) -> dict[str, Any]:
        region = self.region_template.copy()
        region.update(
            {
                "name": name,
                "polygon": self.Polygon(polygon_points),
                "region_color": region_color,
                "text_color": text_color,
            }
        )
        self.counting_regions.append(region)
        return region

    def initialize_regions(self):
        """Initialize regions from `self.region` only once."""
        if self.region is None:
            self.initialize_region()
        if not isinstance(self.region, dict):  # Ensure self.region is initialized and structured as a dictionary
            self.region = {"": self.region}
        for i, (name, pts) in enumerate(self.region.items()):
            region = self.add_region(name, pts, colors(i, True), (255, 255, 255))
            region["prepared_polygon"] = self.prep(region["polygon"])

    def process(self, im0: np.ndarray) -> SolutionResults:
        self.extract_tracks(im0)
        annotator = SolutionAnnotator(im0, line_width=self.line_width)

        target_infos = []

        for box, cls, track_id, conf in zip(self.boxes, self.clss, self.track_ids, self.confs):
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
            annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(track_id, True))
            center = self.Point(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
            for region in self.counting_regions:
                if region["prepared_polygon"].contains(center):
                    region["counts"] += 1
                    self.region_counts[region["name"]] = region["counts"]

        # Display region counts
        for region in self.counting_regions:
            poly = region["polygon"]
            pts = list(map(tuple, np.array(poly.exterior.coords, dtype=np.int32)))
            (x1, y1), (x2, y2) = [(int(poly.centroid.x), int(poly.centroid.y))] * 2
            annotator.draw_region(pts, region["region_color"], self.line_width * 2)
            region["counts"] = 0  # Reset for next frame
        plot_im = annotator.result()
        self.display_output(plot_im)

        return SolutionResults(
            plot_im=plot_im, 
            total_tracks=len(self.track_ids), 
            region_counts=self.region_counts,
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

    def _roi_points(self, roi):
        x1, y1, x2, y2 = roi.x1, roi.y1, roi.x2, roi.y2
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def _ensure_solution(self, frame, roi):
        h, w = frame.shape[:2]
        if self.count_mode == "line":
            y = int(self.line_pos * h)
            region = [(0, y), (w - 1, y)]
            if self.use_track:
                self._sol = MyObjectCounter(show=False, region=region, model=self.model_path, tracker="botsort.yaml")
            else:
                self._sol = MyObjectCounter(show=False, region=region, model=self.model_path)
            self._mode = "line"
        else:
            region = self._roi_points(roi)
            try:
                if self.use_track:
                    self._sol = MyRegionCounter(show=False, region=region, model=self.model_path, tracker="botsort.yaml")
                else:
                    self._sol = MyRegionCounter(show=False, region=region, model=self.model_path)
                self._mode = "region"
            except Exception:
                if self.use_track:
                    self._sol = MyObjectCounter(show=False, region=region, model=self.model_path, tracker="botsort.yaml")
                else:
                    self._sol = MyObjectCounter(show=False, region=region, model=self.model_path)
                self._mode = "roi_object"

    def process(self, frame, roi):
        if self._sol is None:
            self._ensure_solution(frame, roi)
        results = self._sol.process(frame)
        annotated = getattr(results, "plot_im", frame)
        counts = {}
        
        # 获取目标详细信息
        target_infos = getattr(results, "target_infos", [])
        
        # 按模式提取或计算按类别计数
        if self._mode == "line":
            cw = getattr(results, "classwise_count", None)
            if isinstance(cw, dict) and cw:
                for k, v in cw.items():
                    try:
                        s = int(v.get("IN", 0)) + int(v.get("OUT", 0))
                    except Exception:
                        s = int(v) if isinstance(v, (int, float)) else 0
                    if s > 0:
                        counts[str(k)] = counts.get(str(k), 0) + s
        else:
            # ROI/区域模式：仅在轨迹首次进入 ROI 时计数一次
            target_infos = getattr(results, "target_infos", [])
            roi_hit = False
            if isinstance(target_infos, list) and target_infos:
                for info in target_infos:
                    name = info.get("class_name") or str(info.get("class_id", 0))
                    track_id = info.get("track_id", 0)
                    centroid = info.get("centroid")
                    if not (isinstance(centroid, (list, tuple, np.ndarray)) and len(centroid) >= 2):
                        bbox = info.get("bbox")
                        if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                            cx = (bbox[0] + bbox[2]) / 2
                            cy = (bbox[1] + bbox[3]) / 2
                            centroid = (cx, cy)
                        else:
                            centroid = (0, 0)
                    inside = roi is not None and roi.contains(float(centroid[0]), float(centroid[1]))
                    roi_hit = roi_hit or inside
                    if inside and track_id not in self._seen_tracks:
                        counts[str(name)] = int(counts.get(str(name), 0)) + 1
                        self._seen_tracks.add(track_id)
            roi_det = roi_hit
        events = []
        for info in target_infos:
            # 转换为 (ts, cls_id, track_id, conf, centroid_x, centroid_y) 格式
            ts = time.time()
            cls_id = info.get("class_id", 0)
            track_id = info.get("track_id", 0)
            conf = info.get("confidence", 0.0)
            centroid = info.get("centroid")
            if not (isinstance(centroid, (list, tuple, np.ndarray)) and len(centroid) >= 2):
                bbox = info.get("bbox")
                if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    centroid = (cx, cy)
                else:
                    centroid = (0, 0)
            events.append((ts, cls_id, track_id, conf, float(centroid[0]), float(centroid[1])))
        # print(f"================annotated：{annotated.shape}, counts: {counts}, events: {events}, roi_det: {roi_det}, target_infos: {target_infos}====================")
        return annotated, counts, events, roi_det, target_infos

