import math

def bbox_center(x1, y1, x2, y2):
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def bbox_diag(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

def bbox_fully_in_roi(x1, y1, x2, y2, roi):
    if roi is None:
        return False
    pts = [(int(x1), int(y1)), (int(x1), int(y2)), (int(x2), int(y1)), (int(x2), int(y2))]
    for px, py in pts:
        if hasattr(roi, "contains"):
            if not roi.contains(px, py):
                return False
        else:
            rx1, ry1, rx2, ry2 = roi
            if not (rx1 <= px <= rx2 and ry1 <= py <= ry2):
                return False
    return True