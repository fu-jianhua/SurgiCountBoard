import os
import sqlite3
import time

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "sessions.db")

def _ensure_dir():
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR, exist_ok=True)

def _conn():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

CONN = None

def init_db():
    global CONN
    if CONN is None:
        CONN = _conn()
    c = CONN.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT, start_ts REAL, end_ts REAL, roi TEXT, video_path TEXT)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS detections (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, ts REAL, class_id INTEGER, track_id INTEGER, conf REAL)"
    )
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS uniq_det ON detections(session_id, class_id, track_id)")
    CONN.commit()

def start_session(camera_id: str, roi: str, start_ts: float = None) -> int:
    init_db()
    ts = start_ts or time.time()
    c = CONN.cursor()
    c.execute("INSERT INTO sessions(camera_id, start_ts, roi) VALUES (?, ?, ?)", (camera_id, ts, roi))
    CONN.commit()
    return c.lastrowid

def end_session(session_id: int, end_ts: float = None, video_path: str = None) -> None:
    init_db()
    ts = end_ts or time.time()
    c = CONN.cursor()
    c.execute("UPDATE sessions SET end_ts=?, video_path=? WHERE id=?", (ts, video_path, session_id))
    CONN.commit()

def add_detection(session_id: int, ts: float, class_id: int, track_id: int, conf: float) -> None:
    init_db()
    c = CONN.cursor()
    try:
        c.execute(
            "INSERT OR IGNORE INTO detections(session_id, ts, class_id, track_id, conf) VALUES (?, ?, ?, ?, ?)",
            (session_id, ts, class_id, track_id, conf),
        )
        CONN.commit()
    except sqlite3.Error:
        CONN.rollback()

def list_sessions(limit: int = 50):
    init_db()
    c = CONN.cursor()
    c.execute(
        "SELECT id, camera_id, start_ts, end_ts, video_path FROM sessions ORDER BY start_ts DESC LIMIT ?",
        (limit,),
    )
    return c.fetchall()

def session_stats(session_id: int):
    init_db()
    c = CONN.cursor()
    c.execute(
        "SELECT class_id, COUNT(DISTINCT track_id) FROM detections WHERE session_id=? GROUP BY class_id",
        (session_id,),
    )
    return c.fetchall()

def range_stats(start_ts: float, end_ts: float):
    init_db()
    c = CONN.cursor()
    c.execute(
        "SELECT class_id, COUNT(DISTINCT track_id) FROM detections WHERE ts BETWEEN ? AND ? GROUP BY class_id",
        (start_ts, end_ts),
    )
    return c.fetchall()