import json

class ROIRect:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = int(min(x1, x2))
        self.y1 = int(min(y1, y2))
        self.x2 = int(max(x1, x2))
        self.y2 = int(max(y1, y2))

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clamp(self, w: int, h: int):
        self.x1 = max(0, min(self.x1, w - 1))
        self.y1 = max(0, min(self.y1, h - 1))
        self.x2 = max(0, min(self.x2, w - 1))
        self.y2 = max(0, min(self.y2, h - 1))

    def to_json(self) -> str:
        return json.dumps({"type": "rect", "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2})