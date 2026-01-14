import time
import cv2
import numpy as np
from collections import Counter
from ultralytics import YOLO
import os
import sys
import re

_BT_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ByteTrack")
if _BT_ROOT not in sys.path:
    sys.path.insert(0, _BT_ROOT)
from yolox.tracker.byte_tracker import BYTETracker   

class Pipeline:
    def __init__(
        self,
        model_path: str = None,
        model=None,
        conf: float = 0.25,
        iou: float = 0.45,
        use_track: bool = True,
        device: str | int = 0,
        half: bool = True,
        imgsz: int = 640,
        max_det: int = 200,
        line_pos: float = 0.6,
        stable_frames: int = 5,
        seg_model: str | object | None = None,
        frame_rate: float = 30.0,
        agnostic_nms: bool = True,
        dedup_iou: float = 0.65,
        count_mode: str = "roi",
        leave_frames: int = 15,
    ):
        # 加载检测模型（Ultralytics YOLO）；可传入已加载的模型或路径
        self.model = model if model is not None else YOLO(model_path)
        self.conf = conf
        self.iou = iou
        # 是否启用跟踪（ByteTrack），用于“一轨一计”和防抖
        self.use_track = use_track
        self.device = device
        self.half = half
        self.imgsz = imgsz
        self.max_det = max_det
        # NMS 是否类别无关（类间也抑制），在密集场景可能导致不同类别近邻互相抑制
        self.agnostic_nms = bool(agnostic_nms)
        # 检测去重 IoU 阈值；过大易合并近邻真目标，过小保留重复框
        self.dedup_iou = float(dedup_iou)
        # 计数模式：ROI 内到达稳态计数；或“计数线”跨线计数（默认 ROI）
        self.count_mode = "line" if str(count_mode).lower() not in ("roi",) else "roi"
        self.names = list(self.model.names.values()) if isinstance(self.model.names, dict) else self.model.names
        n_colors = max(1, len(self.names)) if isinstance(self.names, (list, tuple)) else 1
        self._colors = []
        for i in range(int(n_colors)):
            h = (float(i) * 0.61803398875) % 1.0
            s = 0.8
            v = 0.95
            hsv = np.array([[[int(h * 180.0), int(s * 255.0), int(v * 255.0)]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            self._colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        # 计数线位置（相对宽度 0~1）；稳定帧阈值（轨迹累计帧数达到后才允许计数）
        self.line_pos = float(max(0.0, min(1.0, line_pos)))
        self.stable_frames = int(max(1, stable_frames))
        self.leave_frames = int(max(1, leave_frames))
        self.seg_model = None
        # SEG 辅助判别：用于屏蔽负面类（人/车等），降低误计；为空时不启用
        if seg_model is not None:
            self.seg_model = seg_model if not isinstance(seg_model, str) else YOLO(seg_model)
        self._neg_roots = set([
            "person","car","truck","bus","bicycle","motorcycle",
            "chair","sofa","couch","bed","table",
            "dog","cat","tv","laptop","keyboard","cellphone","phone",
            "book","clock","vase","teddybear","pottedplant"
        ])
        self._tracks = {}
        self._frame_idx = 0
        self._bt = None
        # 初始化跟踪器（ByteTrack），用于轨迹 ID 与状态管理
        if self.use_track:
            class _BTArgs:
                track_thresh = 0.25
                track_buffer = 30
                match_thresh = 0.8
                aspect_ratio_thresh = 3.0
                min_box_area = 1.0
                mot20 = False
            try:
                self._bt = BYTETracker(_BTArgs(), frame_rate=float(frame_rate))
            except Exception:
                self._bt = None

    def _norm_name(self, s):
        try:
            return re.sub(r"[^a-z]+", "", str(s).lower())
        except Exception:
            return str(s).lower()

    def _bbox_iou(self, box: np.ndarray, boxes: np.ndarray):
        if boxes.size == 0:
            return np.zeros((0,), dtype=float)
        xy_max = np.minimum(boxes[:, 2:], box[2:])
        xy_min = np.maximum(boxes[:, :2], box[:2])
        inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
        inter_area = inter[:, 0] * inter[:, 1]
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_box = (box[2] - box[0]) * (box[3] - box[1])
        denom = area_box + area_boxes - inter_area + 1e-9
        return inter_area / denom

    def _dedup_dets(self, boxes: np.ndarray, clss: np.ndarray, confs: np.ndarray):
        if boxes.size == 0:
            return boxes, clss, confs
        cfs = confs if confs.size == boxes.shape[0] else np.ones((boxes.shape[0],), dtype=float) * 0.5
        order = np.argsort(-cfs)
        keep = []
        for i in order:
            b = boxes[i]
            if len(keep) == 0:
                keep.append(i)
                continue
            prev = boxes[keep]
            ious = self._bbox_iou(b, prev)
            if np.all(ious <= float(self.dedup_iou)):
                keep.append(i)
        keep = np.array(keep, dtype=int)
        return boxes[keep], clss[keep] if clss.size == boxes.shape[0] else clss, cfs[keep]

    def process(self, frame: np.ndarray, roi_rect):
        # 单帧处理：检测 -> 去重 -> 跟踪 -> 计数/事件 -> 叠加绘制
        results = self.model(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            half=self.half,
            imgsz=self.imgsz,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
        )
        r = results[0]
        # 提取检测框、类别、ID（若有）、置信度
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
        clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.cls is not None else np.empty((0,), dtype=int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes is not None and r.boxes.id is not None else np.empty((0,), dtype=int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else np.empty((0,), dtype=float)
        # 对检测结果做 IoU 去重（保留高置信框，降低同目标多框重复）
        if boxes is not None and boxes.size > 0:
            boxes, clss, confs = self._dedup_dets(boxes, clss if clss is not None else np.empty((0,), dtype=int), confs if confs is not None else np.empty((0,), dtype=float))
        ts = time.time()
        counts = {}
        events = []
        self._frame_idx += 1
        h, w = frame.shape[:2]
        display_mask = np.ones((len(boxes),), dtype=bool)
        # 计算计数线位置（ROI 矩形内或整幅图宽度）
        if roi_rect is not None and not hasattr(roi_rect, "points"):
            line_x = int(roi_rect.x1 + self.line_pos * (roi_rect.x2 - roi_rect.x1))
        else:
            line_x = int(self.line_pos * w)
        # 若开启跟踪，对检测框做 ByteTrack 跟踪更新
        if self.use_track and self._bt is not None:
            outputs = np.empty((len(boxes), 6), dtype=float)
            if len(boxes) > 0:
                outputs[:, 0:4] = boxes
                outputs[:, 4] = confs if len(confs) == len(boxes) else np.ones((len(boxes),)) * 0.5
                outputs[:, 5] = clss if len(clss) == len(boxes) else np.zeros((len(boxes),))
            det5 = outputs[:, :5] if len(boxes) > 0 else np.empty((0, 5), dtype=float)
            try:
                tracks = self._bt.update(det5, img_info=frame.shape, img_size=frame.shape, image=frame)
            except Exception:
                tracks = []
            for t in tracks:
                tlbr = t.tlbr
                ious = self._bbox_iou(tlbr, outputs[:, :4]) if len(outputs) > 0 else np.zeros((0,), dtype=float)
                det_idx = int(np.argmax(ious)) if ious.size > 0 else -1
                if det_idx >= 0:
                    x1, y1, x2, y2, c, cls_id = outputs[det_idx]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                else:
                    x1, y1, x2, y2 = tlbr
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    c = 0.0
                    cls_id = -1
                tid = int(getattr(t, "track_id", -1))
                if tid < 0:
                    continue
                st = self._tracks.get(tid)
                if st is None:
                    patch_x1 = max(0, int(x1))
                    patch_y1 = max(0, int(y1))
                    patch_x2 = min(w - 1, int(x2))
                    patch_y2 = min(h - 1, int(y2))
                    seg_pass = None
                    # SEG 辅助：对初始目标 patch 做一次分割，如果出现负面类则标记 seg_pass=False
                    if self.seg_model is not None and patch_x2 > patch_x1 and patch_y2 > patch_y1:
                        patch = frame[patch_y1:patch_y2, patch_x1:patch_x2]
                        try:
                            seg_res = self.seg_model(patch, conf=0.2, device=self.device, half=self.half, imgsz=self.imgsz, verbose=False)
                            rr = seg_res[0]
                            seg_names = list(rr.names.values()) if isinstance(rr.names, dict) else rr.names
                            has_any = ((rr.boxes is not None and len(rr.boxes) > 0) or (getattr(rr, "masks", None) is not None and rr.masks is not None and len(rr.masks) > 0))
                            if has_any:
                                has_negative = False
                                if rr.boxes is not None and rr.boxes.cls is not None:
                                    seg_cls = rr.boxes.cls.cpu().numpy().astype(int)
                                    for cid in seg_cls:
                                        seg_name = seg_names[cid] if isinstance(seg_names, (list, tuple)) and cid < len(seg_names) else str(cid)
                                        if self._norm_name(seg_name) in self._neg_roots:
                                            has_negative = True
                                            break
                                seg_pass = (not has_negative)
                            else:
                                seg_pass = False
                        except Exception:
                            seg_pass = None
                    # 初始化轨迹状态：累计帧、最后中心点、是否已计数、类别投票、SEG 判定、最后出现帧
                    st = {"last_cx": None, "frames": 0, "counted": False, "labels": Counter(), "seg_pass": seg_pass, "last_seen": self._frame_idx, "roi_frames": 0, "out_frames": 0}
                    self._tracks[tid] = st
                if det_idx >= 0 and self.seg_model is not None and st.get("seg_pass") is False:
                    if det_idx < display_mask.shape[0]:
                        display_mask[det_idx] = False
                # 更新轨迹状态：累计帧、最后出现帧、类别投票
                st["frames"] += 1
                st["last_seen"] = self._frame_idx
                if cls_id is not None and int(cls_id) >= 0:
                    st["labels"][int(cls_id)] += 1
                prev_cx = st["last_cx"]
                # 左→右跨线（当前实现仅此方向）
                cross = prev_cx is not None and prev_cx < line_x and cx >= line_x
                in_roi = True
                if roi_rect is not None:
                    in_roi = roi_rect.contains(cx, cy) if not hasattr(roi_rect, "points") else roi_rect.contains(cx, cy)
                if in_roi:
                    st["roi_frames"] = int(st.get("roi_frames", 0)) + 1
                    st["out_frames"] = 0
                else:
                    st["out_frames"] = int(st.get("out_frames", 0)) + 1
                    st["roi_frames"] = 0
                if int(st.get("out_frames", 0)) >= self.leave_frames:
                    st["counted"] = False
                if self.count_mode == "line":
                    cond = (not st["counted"]) and cross and in_roi and (st.get("roi_frames", 0) >= self.stable_frames)
                else:
                    cond = (not st["counted"]) and in_roi and (st.get("roi_frames", 0) >= self.stable_frames)
                if cond:
                    # 通过 SEG 判定（未启用/通过/未定），按多数票主类进行“一轨一计”，并记录事件
                    if self.seg_model is None or st["seg_pass"] is True or st["seg_pass"] is None:
                        if len(st["labels"]) > 0:
                            maj_cls = max(st["labels"].items(), key=lambda kv: kv[1])[0]
                            name = self.names[maj_cls] if maj_cls < len(self.names) else str(maj_cls)
                            counts[name] = counts.get(name, 0) + 1
                            events.append((ts, int(maj_cls), int(tid), float(c), float(cx), float(cy)))
                            st["counted"] = True
                st["last_cx"] = cx
            # 清理长时间未出现的轨迹，避免状态膨胀
            remove_tids = []
            for tid, st in self._tracks.items():
                if st["last_seen"] < self._frame_idx - max(self.stable_frames * 5, 30):
                    remove_tids.append(tid)
            for tid in remove_tids:
                self._tracks.pop(tid, None)
        else:
            # 非跟踪回退：逐帧对 ROI 内检测计数（可能过计，仅降级使用）
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if roi_rect is None or roi_rect.contains(cx, cy):
                    name = self.names[clss[i]] if clss[i] < len(self.names) else str(clss[i])
                    counts[name] = counts.get(name, 0) + 1
                    events.append((ts, int(clss[i]), -1, float(confs[i]) if i < len(confs) else 0.0, float(cx), float(cy)))
        annotated = frame.copy()
        # 叠加绘制：边框、类别/置信度标签、ROI 边界与计数线
        for i in range(len(boxes)):
            if not display_mask[i]:
                continue
            x1, y1, x2, y2 = boxes[i]
            if i < len(clss):
                ci = int(clss[i])
                if isinstance(self._colors, list) and ci >= 0 and ci < len(self._colors):
                    color = self._colors[ci]
                else:
                    color = (0, 255, 0)
            else:
                color = (0, 255, 0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if i < len(clss):
                name = self.names[clss[i]] if clss[i] < len(self.names) else str(clss[i])
            else:
                name = ""
            conf_txt = f" {confs[i]:.2f}" if i < len(confs) else ""
            if name or conf_txt:
                lbl = f"{name}{conf_txt}"
                fs = 0.6
                th = 2
                (tw, thh), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
                H, W = annotated.shape[0], annotated.shape[1]
                tx = int(x1)
                ty = int(y1) - 6
                if ty - thh - 4 < 0:
                    ty = int(y1) + thh + 6
                if tx + tw + 4 > W:
                    tx = int(max(0, int(x2) - tw - 4))
                if ty > H - 2:
                    ty = H - 2
                bx1 = max(0, tx)
                by1 = max(0, ty - thh - 4)
                bx2 = min(W - 1, tx + tw + 4)
                by2 = min(H - 1, ty + 2)
                ov = annotated.copy()
                cv2.rectangle(ov, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                annotated = cv2.addWeighted(ov, 0.4, annotated, 0.6, 0)
                cv2.putText(annotated, lbl, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
        if roi_rect is not None:
            if hasattr(roi_rect, "points"):
                pts = np.array(roi_rect.points, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (104, 0, 123), 2)
            else:
                cv2.rectangle(annotated, (roi_rect.x1, roi_rect.y1), (roi_rect.x2, roi_rect.y2), (104, 0, 123), 2)
        if self.count_mode == "line":
            cv2.line(annotated, (int(line_x), 0), (int(line_x), annotated.shape[0] - 1), (0, 192, 255), 2)
        has_roi_det = False
        # 标记当前帧是否检测到落在 ROI 内的目标（用于会话开始/结束判定）
        for i in range(len(boxes)):
            if not display_mask[i]:
                continue
            x1, y1, x2, y2 = boxes[i]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            in_roi = True
            if roi_rect is not None:
                in_roi = roi_rect.contains(cx, cy) if not hasattr(roi_rect, "points") else roi_rect.contains(cx, cy)
            if in_roi:
                has_roi_det = True
                break
        return annotated, counts, events, has_roi_det
