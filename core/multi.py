import time
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Event:
    ts: float
    class_id: int
    track_id: int
    cam_index: int
    x: float
    y: float
    conf: float


class MultiCameraFusion:
    def __init__(self, time_thr: float = 0.3, dist_thr: float = 32.0):
        self.time_thr = float(time_thr)
        self.dist_thr = float(dist_thr)
        self.buffer: List[Event] = []
        self.fused: List[Tuple[float, int, List[Event]]] = []

    def _dist(self, e1: Event, e2: Event) -> float:
        try:
            dx = float(e1.x) - float(e2.x)
            dy = float(e1.y) - float(e2.y)
            return (dx * dx + dy * dy) ** 0.5
        except Exception:
            return 1e9

    def push(self, ev: Event):
        self.buffer.append(ev)
        now = ev.ts
        self.buffer = [e for e in self.buffer if now - e.ts <= max(self.time_thr * 3.0, 2.0)]

    def try_fuse(self) -> Optional[Tuple[float, int, List[Event]]]:
        if len(self.buffer) < 2:
            return None
        cand = []
        for i in range(len(self.buffer)):
            e1 = self.buffer[i]
            for j in range(i + 1, len(self.buffer)):
                e2 = self.buffer[j]
                if e1.cam_index == e2.cam_index:
                    continue
                if abs(e1.ts - e2.ts) > self.time_thr:
                    continue
                if e1.class_id != e2.class_id:
                    continue
                if self._dist(e1, e2) > self.dist_thr:
                    continue
                cand.append((i, j))
        if not cand:
            return None
        used = set()
        groups: List[List[Event]] = []
        for i, j in cand:
            if i in used or j in used:
                continue
            used.add(i); used.add(j)
            groups.append([self.buffer[i], self.buffer[j]])
        if not groups:
            return None
        g = groups[0]
        ts = sum(e.ts for e in g) / float(len(g))
        cls = g[0].class_id
        self.fused.append((ts, cls, g))
        for e in g:
            try:
                self.buffer.remove(e)
            except ValueError:
                pass
        return (ts, cls, g)

    def dump_members_json(self, members: List[Event]) -> str:
        arr = [
            {
                "ts": float(e.ts),
                "class_id": int(e.class_id),
                "track_id": int(e.track_id),
                "cam_index": int(e.cam_index),
                "x": float(e.x),
                "y": float(e.y),
                "conf": float(e.conf),
            }
            for e in members
        ]
        return json.dumps(arr)

