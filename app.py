import os
import time
import cv2
import streamlit as st
from core.source import open_capture
from core.roi import ROIRect
from core.pipeline import Pipeline
from core.session import SessionManager
from core.store import init_db, add_detection, list_sessions, session_stats, range_stats
from core.video import open_writer, write_frame, close_writer

st.set_page_config(page_title="SurgiCountBoard", layout="wide")
st.title("SurgiCountBoard")

default_model = os.path.join("e:\\project\\ultralytics\\models\\medical_instruments5\\weights", "best.pt")

with st.sidebar:
    model_path = st.text_input("模型路径", default_model)
    source_str = st.text_input("视频源", "0")
    conf = st.slider("置信度", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("IoU", 0.0, 1.0, 0.45, 0.01)
    idle_seconds = st.number_input("空窗秒数", min_value=1, max_value=120, value=10, step=1)
    roi_x1 = st.number_input("ROI x1", min_value=0, value=0, step=1)
    roi_y1 = st.number_input("ROI y1", min_value=0, value=0, step=1)
    roi_x2 = st.number_input("ROI x2", min_value=0, value=0, step=1)
    roi_y2 = st.number_input("ROI y2", min_value=0, value=0, step=1)
    start_btn = st.button("开始")
    stop_btn = st.button("停止")

col1, col2 = st.columns(2)
org_frame_container = col1.empty()
ann_frame_container = col2.empty()

init_db()

if "running" not in st.session_state:
    st.session_state.running = False
if "writer" not in st.session_state:
    st.session_state.writer = None
if "session" not in st.session_state:
    st.session_state.session = None
if "roi" not in st.session_state:
    st.session_state.roi = None

if start_btn and not st.session_state.running:
    st.session_state.running = True
    cap = open_capture(source_str)
    if not cap.isOpened():
        st.error("无法打开视频源")
        st.session_state.running = False
    else:
        pipeline = Pipeline(model_path=model_path, conf=conf, iou=iou, classes=None, use_track=True)
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取帧")
            st.session_state.running = False
            cap.release()
        else:
            h, w = frame.shape[:2]
            if roi_x2 == 0 and roi_y2 == 0:
                st.session_state.roi = ROIRect(0, 0, w - 1, h - 1)
            else:
                r = ROIRect(int(roi_x1), int(roi_y1), int(roi_x2), int(roi_y2))
                r.clamp(w, h)
                st.session_state.roi = r
            sess = SessionManager(camera_id=str(source_str), idle_seconds=int(idle_seconds), roi_json=st.session_state.roi.to_json())
            st.session_state.session = sess
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while cap.isOpened() and st.session_state.running:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated, counts, events = pipeline.process(frame, st.session_state.roi)
                now = time.time()
                has_roi_det = len(events) > 0
                if has_roi_det:
                    sid = st.session_state.session.on_detection(now)
                    if st.session_state.writer is None:
                        out_dir = os.path.join("runs", "surgicountboard")
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"session_{sid}.mp4")
                        st.session_state.video_path = out_path
                        st.session_state.writer = open_writer(out_path, annotated.shape, fps=cap.get(cv2.CAP_PROP_FPS) or 25)
                    for ts, cls_id, tid, c in events:
                        add_detection(sid, ts, cls_id, tid, c)
                ended = st.session_state.session.check_idle_and_end(now, st.session_state.video_path if "video_path" in st.session_state else None)
                if ended is not None and st.session_state.writer is not None:
                    close_writer(st.session_state.writer)
                    st.session_state.writer = None
                    st.session_state.video_path = None
                if st.session_state.writer is not None:
                    write_frame(st.session_state.writer, annotated)
                org_frame_container.image(frame, channels="BGR")
                ann_frame_container.image(annotated, channels="BGR")
                st.sidebar.write({"当前计数": counts})
                if stop_btn:
                    st.session_state.running = False
            if st.session_state.writer is not None:
                close_writer(st.session_state.writer)
                st.session_state.writer = None
            cap.release()

st.subheader("任务列表")
sessions = list_sessions(50)
st.table({"id": [x[0] for x in sessions], "camera": [x[1] for x in sessions], "start": [x[2] for x in sessions], "end": [x[3] for x in sessions], "video": [x[4] for x in sessions]})

sid_inp = st.number_input("查看任务统计 id", min_value=0, step=1, value=0)
if sid_inp > 0:
    stats = session_stats(int(sid_inp))
    st.table({"class_id": [x[0] for x in stats], "count": [x[1] for x in stats]})

st.subheader("时间段统计")
start_ts = st.number_input("开始时间戳", value=float(time.time() - 3600))
end_ts = st.number_input("结束时间戳", value=float(time.time()))
if end_ts > start_ts:
    rstats = range_stats(start_ts, end_ts)
    st.table({"class_id": [x[0] for x in rstats], "count": [x[1] for x in rstats]})