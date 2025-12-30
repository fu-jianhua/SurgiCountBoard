import os
import cv2
import shutil
import subprocess
import numpy as np

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

class FFmpegWriter:
    def __init__(self, path: str, w: int, h: int, fps: float, ffmpeg_path: str):
        self.w = w
        self.h = h
        self.fps = fps
        self.proc = subprocess.Popen(
            [
                ffmpeg_path,
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{w}x{h}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-pix_fmt",
                "yuv420p",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    def write(self, frame: np.ndarray):
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            frame = cv2.resize(frame, (self.w, self.h))
        self.proc.stdin.write(frame.tobytes())
    def release(self):
        try:
            self.proc.stdin.flush()
        except Exception:
            pass
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass

def open_writer(path: str, frame_shape: tuple[int, int, int], fps: float = 25.0):
    _ensure_dir(path)
    h, w = frame_shape[0], frame_shape[1]
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        try:
            return FFmpegWriter(path, w, h, fps, ffmpeg)
        except Exception:
            pass
    for cc in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
        if writer is not None and writer.isOpened():
            return writer
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

def write_frame(writer, frame):
    writer.write(frame)

def close_writer(writer):
    writer.release()