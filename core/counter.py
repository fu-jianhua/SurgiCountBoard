from collections import defaultdict, Counter
from enum import Enum
from .utils import bbox_center, bbox_diag, bbox_fully_in_roi


class TrackState(Enum):
    OUTSIDE = 0
    INSIDE = 1
    REMOVED = 2


class InstrumentCounter:
    def __init__(self, roi):
        self.roi = roi
        self.state = defaultdict(lambda: TrackState.OUTSIDE)
        self.age = defaultdict(int)
        self.roi_frames = defaultdict(int)
        self.cls_votes = defaultdict(list)
        self.last_center = {}
        self.total_move = defaultdict(float)
        self.counted_ids = set()
        self.class_counter = defaultdict(int)
        self.MIN_TRACK_AGE = 12
        self.MIN_ROI_FRAMES = 8
        self.MIN_MOVE_RATIO = 0.5

    def update(self, track):
        tid = track.track_id
        x1, y1, x2, y2 = track.tlbr
        cls = track.cls
        cx, cy = bbox_center(x1, y1, x2, y2)
        diag = bbox_diag(x1, y1, x2, y2)
        self.age[tid] += 1
        self.cls_votes[tid].append(cls)
        prev = self.last_center.get(tid)
        if prev:
            move = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
            self.total_move[tid] += move
        self.last_center[tid] = (cx, cy)
        fully_in = bbox_fully_in_roi(x1, y1, x2, y2, self.roi)
        if self.state[tid] == TrackState.OUTSIDE:
            if fully_in:
                self.state[tid] = TrackState.INSIDE
                self.roi_frames[tid] = 1
        elif self.state[tid] == TrackState.INSIDE:
            if fully_in:
                self.roi_frames[tid] += 1
            else:
                self.state[tid] = TrackState.REMOVED

    def try_count(self, track):
        tid = track.track_id
        x1, y1, x2, y2 = track.tlbr
        diag = bbox_diag(x1, y1, x2, y2)
        if tid in self.counted_ids:
            return False, None
        if self.state[tid] != TrackState.REMOVED:
            return False, None
        if self.age[tid] < self.MIN_TRACK_AGE:
            return False, None
        if self.roi_frames[tid] < self.MIN_ROI_FRAMES:
            return False, None
        if self.total_move[tid] < self.MIN_MOVE_RATIO * diag:
            return False, None
        final_cls = Counter(self.cls_votes[tid]).most_common(1)[0][0]
        self.counted_ids.add(tid)
        self.class_counter[final_cls] += 1
        return True, final_cls