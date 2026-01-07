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
    c.execute("CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts)")
    c.execute(
        "CREATE TABLE IF NOT EXISTS multi_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, start_ts REAL, end_ts REAL, meta TEXT)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS cam_sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, multi_id INTEGER, cam_index INTEGER, session_id INTEGER, video_path TEXT)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, multi_id INTEGER, cam_index INTEGER, ts REAL, class_id INTEGER, track_id INTEGER, x REAL, y REAL, conf REAL)"
    )
    c.execute(
        "CREATE TABLE IF NOT EXISTS fused_events (id INTEGER PRIMARY KEY AUTOINCREMENT, multi_id INTEGER, ts REAL, class_id INTEGER, members_json TEXT)"
    )
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

def get_session(session_id: int):
    init_db()
    c = CONN.cursor()
    c.execute(
        "SELECT id, camera_id, start_ts, end_ts, roi, video_path FROM sessions WHERE id=?",
        (session_id,),
    )
    return c.fetchone()

def search_sessions(camera_id: str | None = None, start_ts: float | None = None, end_ts: float | None = None, offset: int = 0, limit: int = 50):
    init_db()
    clauses = []
    params = []
    if camera_id:
        clauses.append("camera_id LIKE ?")
        params.append(f"%{camera_id}%")
    if start_ts is not None:
        clauses.append("start_ts >= ?")
        params.append(start_ts)
    if end_ts is not None:
        clauses.append("start_ts <= ?")
        params.append(end_ts)
    where_sql = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT id, camera_id, start_ts, end_ts, video_path FROM sessions{where_sql} ORDER BY start_ts DESC LIMIT ? OFFSET ?"
    params.extend([limit, max(0, offset)])
    c = CONN.cursor()
    c.execute(sql, tuple(params))
    return c.fetchall()

def start_multi_session(start_ts: float | None = None, meta: str | None = None) -> int:
    init_db()
    ts = start_ts or time.time()
    c = CONN.cursor()
    c.execute("INSERT INTO multi_sessions(start_ts, meta) VALUES (?, ?)", (ts, meta))
    CONN.commit()
    return c.lastrowid

def end_multi_session(multi_id: int, end_ts: float | None = None):
    init_db()
    ts = end_ts or time.time()
    c = CONN.cursor()
    c.execute("UPDATE multi_sessions SET end_ts=? WHERE id=?", (ts, multi_id))
    CONN.commit()

def add_cam_session(multi_id: int, cam_index: int, session_id: int, video_path: str | None = None):
    init_db()
    c = CONN.cursor()
    c.execute(
        "INSERT INTO cam_sessions(multi_id, cam_index, session_id, video_path) VALUES (?, ?, ?, ?)",
        (multi_id, cam_index, session_id, video_path),
    )
    CONN.commit()

def add_event(multi_id: int, cam_index: int, ts: float, class_id: int, track_id: int, x: float, y: float, conf: float):
    init_db()
    c = CONN.cursor()
    try:
        c.execute(
            "INSERT INTO events(multi_id, cam_index, ts, class_id, track_id, x, y, conf) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (multi_id, cam_index, ts, class_id, track_id, x, y, conf),
        )
        CONN.commit()
    except sqlite3.Error:
        CONN.rollback()

def add_fused_event(multi_id: int, ts: float, class_id: int, members_json: str):
    init_db()
    c = CONN.cursor()
    c.execute(
        "INSERT INTO fused_events(multi_id, ts, class_id, members_json) VALUES (?, ?, ?, ?)",
        (multi_id, ts, class_id, members_json),
    )
    CONN.commit()

def get_cam_sessions(multi_id: int):
    init_db()
    c = CONN.cursor()
    c.execute(
        "SELECT cam_index, session_id, video_path FROM cam_sessions WHERE multi_id=? ORDER BY cam_index ASC",
        (multi_id,),
    )
    return c.fetchall()

def events_cam_stats(multi_id: int):
    init_db()
    c = CONN.cursor()
    c.execute(
        "SELECT cam_index, class_id, COUNT(DISTINCT track_id) FROM events WHERE multi_id=? GROUP BY cam_index, class_id",
        (multi_id,),
    )
    return c.fetchall()

def events_final_stats_max(multi_id: int):
    rows = events_cam_stats(int(multi_id))
    best = {}
    for cam_index, class_id, cnt in rows:
        k = int(class_id)
        v = int(cnt)
        prev = best.get(k)
        if prev is None or v > prev:
            best[k] = v
    return [(k, best[k]) for k in sorted(best.keys())]
