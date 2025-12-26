import os
import cv2

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def open_writer(path: str, frame_shape: tuple[int, int, int], fps: float = 25.0):
    _ensure_dir(path)
    h, w = frame_shape[0], frame_shape[1]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    return writer

def write_frame(writer, frame):
    writer.write(frame)

def close_writer(writer):
    writer.release()