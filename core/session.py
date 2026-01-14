import time
from .store import start_session, end_session

class SessionManager:
    # 会话管理：检测出现时开始会话，空窗超过阈值后结束并记录视频路径
    def __init__(self, camera_id: str, idle_seconds: int, roi_json: str):
        self.camera_id = camera_id
        self.idle_seconds = idle_seconds
        self.roi_json = roi_json
        self.current_session_id = None
        self.last_detection_ts = None

    def on_detection(self, now: float = None):
        # 记录一次检测，若尚未开始则创建新会话，并更新最后检测时间
        ts = now or time.time()
        if self.current_session_id is None:
            self.current_session_id = start_session(self.camera_id, self.roi_json, ts)
        self.last_detection_ts = ts
        return self.current_session_id

    def check_idle_and_end(self, now: float = None, video_path: str = None):
        # 若超过空窗阈值则结束会话并清理状态，返回已结束的会话 ID
        ts = now or time.time()
        if self.current_session_id is not None and self.last_detection_ts is not None:
            if ts - self.last_detection_ts >= self.idle_seconds:
                sid = self.current_session_id
                end_session(sid, ts, video_path)
                self.current_session_id = None
                self.last_detection_ts = None
                return sid
        return None