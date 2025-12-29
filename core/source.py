import cv2
import urllib.request
import numpy as np

class HTTPMJPEGCapture:
    def __init__(self, url: str, timeout: float = 5.0):
        self.url = url
        self.timeout = timeout
        self.response = None
        self.buffer = b""
        try:
            self.response = urllib.request.urlopen(self.url, timeout=self.timeout)
        except Exception:
            self.response = None

    def isOpened(self) -> bool:
        return self.response is not None

    def read(self):
        if self.response is None:
            return False, None
        # Read chunks until we have at least one full JPEG frame. Always decode the latest frame to reduce latency.
        while True:
            start = self.buffer.rfind(b"\xff\xd8")
            end = self.buffer.rfind(b"\xff\xd9")
            if start != -1 and end != -1 and end > start:
                jpg = self.buffer[start : end + 2]
                # Drop everything up to this frame to avoid backlog
                self.buffer = self.buffer[end + 2 :]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return False, None
                return True, frame
            chunk = self.response.read(8192)
            if not chunk:
                return False, None
            # Keep buffer small to avoid accumulating latency
            self.buffer += chunk
            if len(self.buffer) > 2_000_000:
                self.buffer = self.buffer[-1_000_000:]

    def release(self):
        try:
            if self.response is not None:
                self.response.close()
        except Exception:
            pass
        self.response = None

    def set(self, prop_id, value):
        return False

    def get(self, prop_id):
        return 0

def _set_low_buffer(cap):
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

def open_capture(source: str | int):
    if isinstance(source, str):
        s = source.strip()
        if s.isdigit():
            cap = cv2.VideoCapture(int(s))
            _set_low_buffer(cap)
            return cap
        if s.startswith("http://") or s.startswith("https://") or s.startswith("rtsp://"):
            cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
            if cap.isOpened():
                _set_low_buffer(cap)
                return cap
            mjpeg_cap = HTTPMJPEGCapture(s)
            return mjpeg_cap
        cap = cv2.VideoCapture(s)
        _set_low_buffer(cap)
        return cap
    cap = cv2.VideoCapture(source)
    _set_low_buffer(cap)
    return cap