import time
from .store import start_session, end_session

class SessionManager:
    def __init__(self, camera_id: str, idle_seconds: int, roi_json: str, batch_id: str | None = None):
        self.camera_id = camera_id
        self.idle_seconds = idle_seconds
        self.roi_json = roi_json
        self.batch_id = batch_id
        self.current_session_id = None
        self.last_detection_ts = None

    def on_detection(self, now: float = None):
        ts = now or time.time()
        if self.current_session_id is None:
            self.current_session_id = start_session(self.camera_id, self.roi_json, ts, self.batch_id)
        self.last_detection_ts = ts
        return self.current_session_id

    def check_idle_and_end(self, now: float = None, video_path: str = None):
        ts = now or time.time()
        if self.current_session_id is not None and self.last_detection_ts is not None:
            if ts - self.last_detection_ts >= self.idle_seconds:
                sid = self.current_session_id
                end_session(sid, ts, video_path)
                self.current_session_id = None
                self.last_detection_ts = None
                return sid
        return None