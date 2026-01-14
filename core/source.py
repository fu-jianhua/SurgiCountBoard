import cv2
import urllib.request
import numpy as np
import threading
import time

class HTTPMJPEGCapture:
    # 简易 HTTP MJPEG 拉流：读取字节流，解析 JPEG 帧，减少缓冲避免高延迟
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
        # 读取块直到得到一个完整 JPEG 帧；始终解码最新帧以降低延迟
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

class BackgroundCapture:
    # 后台采集包装：在低延迟模式下开启后台线程持续读帧，主线程 read() 直接取最新帧
    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.running = False
        self.t = None

    def _loop(self):
        while self.running and self.cap.isOpened():
            ok, f = self.cap.read()
            if ok:
                self.frame = f
            else:
                time.sleep(0.01)

    def start(self):
        if self.running:
            return
        self.running = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        if self.frame is not None:
            return True, self.frame
        return self.cap.read()

    def release(self):
        self.running = False
        if self.t is not None:
            try:
                self.t.join(timeout=1.0)
            except Exception:
                pass
        try:
            self.cap.release()
        except Exception:
            pass

    def set(self, prop_id, value):
        try:
            return self.cap.set(prop_id, value)
        except Exception:
            return False

    def get(self, prop_id):
        try:
            return self.cap.get(prop_id)
        except Exception:
            return 0

def _set_low_buffer(cap):
    # 将缓冲区设置为最小，减少延迟与积帧
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

def open_capture(source: str | int, low_latency: bool = False):
    # 打开视频源：本地设备索引/文件/HTTP/RTSP 等
    # 低延迟模式会开启后台采集并尝试 MJPG 编码以提高实时性
    if isinstance(source, str):
        s = source.strip()
        if s.isdigit():
            if low_latency:
                try:
                    cap = cv2.VideoCapture(int(s), cv2.CAP_DSHOW)
                except Exception:
                    cap = cv2.VideoCapture(int(s))
            else:
                cap = cv2.VideoCapture(int(s))
            _set_low_buffer(cap)
            if low_latency:
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:
                    pass
            bg = BackgroundCapture(cap) if low_latency else cap
            if low_latency and isinstance(bg, BackgroundCapture):
                bg.start()
            return bg
        if s.startswith("http://") or s.startswith("https://") or s.startswith("rtsp://"):
            cap = cv2.VideoCapture(s, cv2.CAP_FFMPEG)
            if cap.isOpened():
                _set_low_buffer(cap)
                bg = BackgroundCapture(cap)
                bg.start()
                return bg
            mjpeg_cap = HTTPMJPEGCapture(s)
            bg = BackgroundCapture(mjpeg_cap)
            bg.start()
            return bg
        cap = cv2.VideoCapture(s)
        _set_low_buffer(cap)
        bg = BackgroundCapture(cap)
        bg.start()
        return bg
    if low_latency:
        try:
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        except Exception:
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    _set_low_buffer(cap)
    if low_latency:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass
    bg = BackgroundCapture(cap) if low_latency else BackgroundCapture(cap)
    bg.start()
    return bg