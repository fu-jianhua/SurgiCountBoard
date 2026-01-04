import os
import time
import cv2
import torch
import streamlit as st
import pandas as pd
import numpy as np
from core.source import open_capture
from core.roi import ROIRect
from core.pipeline import Pipeline
from core.session import SessionManager
from core.store import init_db, add_detection, session_stats, end_session, get_session, search_sessions
from core.video import open_writer, write_frame, close_writer

st.set_page_config(page_title="SurgiCountBoard", layout="wide")
st.title("SurgiCountBoard")
status_banner = st.empty()

def _status_color(s):
    if "运行" in s:
        return "#16a34a"
    if "启动" in s:
        return "#f59e0b"
    if "停止中" in s:
        return "#ef4444"
    return "#6b7280"

def _render_status(text=None):
    t = text or st.session_state.get("status", "已停止")
    c = _status_color(t)
    status_banner.markdown(
        f"<div style=\"padding:8px 12px; border-radius:8px; background:{c}; color:#fff; text-align:center; font-weight:600;\">{t}</div>",
        unsafe_allow_html=True,
    )

default_model = os.path.join("e:\\project\\ultralytics\\models\\medical_instruments5\\weights", "best.pt")

def _fmt_ts(ts: float | None):
    try:
        if ts is None or ts == 0:
            return ""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ts)))
    except Exception:
        return ""

def _parse_dt(s: str | None):
    try:
        if not s:
            return None
        t = time.strptime(s.strip(), "%Y-%m-%d %H:%M:%S")
        return float(time.mktime(t))
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _get_model_names(path: str):
    from ultralytics import YOLO
    m = YOLO(path)
    return list(m.names.values()) if isinstance(m.names, dict) else m.names

with st.sidebar:
    btn_col1, btn_col2 = st.columns(2)
    start_btn = btn_col1.button("开始")
    stop_btn = btn_col2.button("停止")
    with st.expander("视频源与设备", expanded=True):
        model_path = st.text_input("模型路径", default_model)
        source_str = st.text_input("视频源", "0")
        device_inp = st.text_input("设备(0/1 或 cpu)", "0")
        half = st.checkbox("FP16 半精度", True)
        low_latency = st.checkbox("低延迟模式", False)
    with st.expander("推理与跟踪", expanded=False):
        conf = st.slider("置信度", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("IoU", 0.0, 1.0, 0.45, 0.01)
        track_enabled = st.checkbox("启用跟踪", True)
        imgsz = st.number_input("推理分辨率(imgsz)", min_value=256, max_value=1280, value=640, step=64)
        max_det = st.number_input("最大检测数(max_det)", min_value=10, max_value=1000, value=200, step=10)
    with st.expander("会话与ROI", expanded=False):
        idle_seconds = st.number_input("空窗秒数", min_value=1, max_value=120, value=10, step=1)
        if "line_pos_pct" not in st.session_state:
            st.session_state.line_pos_pct = 70
        line_move_step_pct = st.number_input("计数线移动步长(%)", min_value=1, max_value=50, value=5, step=1)
        line_pos_slider = st.slider("计数线位置(%)", 0, 100, int(st.session_state.line_pos_pct), 1)
        st.session_state.line_pos_pct = int(line_pos_slider)
        line_btn_col_l, line_btn_col_r = st.columns(2)
        line_left_btn = line_btn_col_l.button("计数线左移")
        line_right_btn = line_btn_col_r.button("计数线右移")
        if line_left_btn:
            st.session_state.line_pos_pct = max(0, int(st.session_state.line_pos_pct) - int(line_move_step_pct))
        if line_right_btn:
            st.session_state.line_pos_pct = min(100, int(st.session_state.line_pos_pct) + int(line_move_step_pct))
        prev_w = int(imgsz)
        prev_h = max(1, int(prev_w * 9 / 16))
        preview = np.full((prev_h, prev_w, 3), 200, dtype=np.uint8)
        line_x = int(float(st.session_state.line_pos_pct) / 100.0 * prev_w)
        cv2.line(preview, (int(line_x), 0), (int(line_x), prev_h - 1), (0, 192, 255), 2)
        st.image(preview, channels="BGR")
    

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
if "cap" not in st.session_state:
    st.session_state.cap = None
if "status" not in st.session_state:
    st.session_state.status = "已停止"

_render_status()

if stop_btn and st.session_state.running:
    st.session_state.status = "停止中..."
    _render_status()
    st.session_state.running = False
    try:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
    except Exception:
        pass
    st.session_state.cap = None
    if st.session_state.writer is not None:
        close_writer(st.session_state.writer)
        st.session_state.writer = None
    if st.session_state.session is not None and getattr(st.session_state.session, "current_session_id", None) is not None:
        end_session(st.session_state.session.current_session_id, time.time(), st.session_state.video_path if "video_path" in st.session_state else None)
        st.session_state.session.current_session_id = None
        st.session_state.video_path = None
    st.session_state.status = "已停止"
    _render_status()

if start_btn and not st.session_state.running:
    st.session_state.status = "启动中..."
    _render_status()
    st.session_state.running = True
    cap = open_capture(source_str, low_latency=low_latency)
    st.session_state.cap = cap
    if not cap.isOpened():
        st.error("无法打开视频源")
        st.session_state.running = False
        try:
            cap.release()
        except Exception:
            pass
        st.session_state.cap = None
        st.session_state.status = "已停止"
        _render_status()
    else:
        dev = int(device_inp) if device_inp.isdigit() else device_inp
        has_cuda = torch.cuda.is_available()
        if isinstance(dev, int) and not has_cuda:
            st.warning("未检测到CUDA，已自动切换为CPU并关闭FP16")
            dev = "cpu"
            half = False
        if isinstance(dev, str) and dev.lower() == "cpu":
            half = False
        @st.cache_resource(show_spinner=False)
        def _load_model(path):
            from ultralytics import YOLO
            return YOLO(path)
        model_obj = _load_model(model_path)
        try:
            st.session_state.model_names = list(model_obj.names.values()) if isinstance(model_obj.names, dict) else model_obj.names
        except Exception:
            st.session_state.model_names = None
        pipeline = Pipeline(
            model=model_obj,
            conf=conf,
            iou=iou,
            use_track=bool(track_enabled),
            device=dev,
            half=half,
            imgsz=int(imgsz if not low_latency else min(imgsz, 512)),
            max_det=int(max_det),
            frame_rate=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
            seg_model=os.path.join(os.path.dirname(__file__), "yolo11x-seg.pt"),
            line_pos=float(st.session_state.get("line_pos_pct", 70)) / 100.0,
        )
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取帧")
            st.session_state.running = False
            try:
                cap.release()
            except Exception:
                pass
            st.session_state.cap = None
            st.session_state.status = "已停止"
            _render_status()
        else:
            h, w = frame.shape[:2]
            st.session_state.roi = ROIRect(0, 0, w - 1, h - 1)
            sess = SessionManager(camera_id=str(source_str), idle_seconds=int(idle_seconds), roi_json=st.session_state.roi.to_json())
            st.session_state.session = sess
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            st.session_state.status = "运行中"
            _render_status()
            try:
                while cap.isOpened() and st.session_state.running:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    annotated, counts, events = pipeline.process(frame, st.session_state.roi)
                    if len(counts) > 0:
                        overlay = annotated.copy()
                        box_h = 28 * (len(counts) + 1)
                        cv2.rectangle(overlay, (8, 8), (280, 8 + box_h), (0, 0, 0), -1)
                        annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)
                        y = 32
                        for k in sorted(counts.keys()):
                            v = counts[k]
                            cv2.putText(annotated, f"{k}: {v}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            y += 26
                    sf = 960.0 / frame.shape[1] if frame.shape[1] > 960 else 1.0
                    if sf != 1.0:
                        dframe = cv2.resize(frame, (int(frame.shape[1] * sf), int(frame.shape[0] * sf)))
                        dann = cv2.resize(annotated, (int(annotated.shape[1] * sf), int(annotated.shape[0] * sf)))
                    else:
                        dframe = frame
                        dann = annotated
                    now = time.time()
                    has_roi_det = len(events) > 0
                    if has_roi_det:
                        sid = st.session_state.session.on_detection(now)
                        if st.session_state.writer is None:
                            out_dir = os.path.join("runs", "surgicountboard")
                            os.makedirs(out_dir, exist_ok=True)
                            out_path = os.path.join(out_dir, f"session_{sid}.mp4")
                            out_path = os.path.abspath(out_path)
                            st.session_state.video_path = out_path
                            st.session_state.writer = open_writer(out_path, annotated.shape, fps=cap.get(cv2.CAP_PROP_FPS) or 25)
                        for ts, cls_id, tid, c in events:
                            if tid is not None and tid >= 0:
                                add_detection(sid, ts, cls_id, tid, c)
                    ended = st.session_state.session.check_idle_and_end(now, st.session_state.video_path if "video_path" in st.session_state else None)
                    if ended is not None and st.session_state.writer is not None:
                        close_writer(st.session_state.writer)
                        st.session_state.writer = None
                        st.session_state.video_path = None
                    if st.session_state.writer is not None:
                        write_frame(st.session_state.writer, annotated)
                    org_frame_container.image(dframe, channels="BGR", output_format="JPEG")
                    ann_frame_container.image(dann, channels="BGR", output_format="JPEG")
                    time.sleep(0.01 if low_latency else 0.03)
                    if stop_btn:
                        st.session_state.running = False
            except Exception:
                st.session_state.running = False
            finally:
                if st.session_state.writer is not None:
                    close_writer(st.session_state.writer)
                    st.session_state.writer = None
                cap.release()
                st.session_state.cap = None

tab_tasks = st.container()

with tab_tasks:
    st.subheader("任务列表")
    if "page_tasks" not in st.session_state:
        st.session_state.page_tasks = 1
    prev_page_tasks = int(st.session_state.page_tasks)
    col_prev, col_lbl, col_page, col_next = st.columns([1,0.2,1,1])
    if col_prev.button("上一页", use_container_width=True):
        st.session_state.page_tasks = max(1, st.session_state.page_tasks - 1)
    col_lbl.markdown("<div style='line-height:38px; font-weight:600;'>页码：</div>", unsafe_allow_html=True)
    page_val = col_page.number_input("页码", min_value=1, value=int(st.session_state.page_tasks), step=1, label_visibility="collapsed")
    if col_next.button("下一页", use_container_width=True):
        st.session_state.page_tasks = int(page_val) + 1
    else:
        st.session_state.page_tasks = int(page_val)
    offset = int((st.session_state.page_tasks - 1) * 10)
    sessions = search_sessions(None, None, None, offset, int(10))
    if len(sessions) == 0:
        st.session_state.page_tasks = prev_page_tasks
    if len(sessions) == 0:
        st.markdown(
            """
            <div style="border:1px dashed #d1d5db; border-radius:10px; padding:18px; background:#f9fafb;">
              <div style="font-size:18px; font-weight:600; color:#111827;">暂无任务数据</div>
              <ol style="margin:10px 0 0; color:#6b7280; padding-left:22px;">
                <li style="margin:6px 0;">在左侧设置模型与视频源</li>
                <li style="margin:6px 0;">点击“开始”</li>
                <li style="margin:6px 0;">ROI内出现检测后会自动生成任务与录像</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        df = pd.DataFrame([{ "id": x[0], "camera": x[1], "start": _fmt_ts(x[2]), "end": _fmt_ts(x[3]), "video": x[4]} for x in sessions])
        tbl_h = max(180, min(420, 40 + len(df) * 32))
        st.dataframe(df, use_container_width=True, height=tbl_h, hide_index=True)
        sid_options = [int(x[0]) for x in sessions]
        st.subheader("任务详情")
        sid_sel = st.selectbox("选择任务", sid_options, index=0)
        if sid_sel:
            ss = get_session(int(sid_sel))
            stats = session_stats(int(sid_sel))
            names_cn_map = {
                "tray": "托盘",
                "forceps_01": "镊子01",
                "forceps_02": "镊子02",
                "scissors_01": "剪刀01",
                "scissors_02": "剪刀02",
                "scissors_03": "剪刀03",
                "scissors_04": "剪刀04",
                "scissors_05": "剪刀05",
                "scissors_06": "剪刀06",
            }
            try:
                model_names = st.session_state.get("model_names")
                if not model_names:
                    model_names = _get_model_names(model_path)
            except Exception:
                model_names = None
            def _cn_name(cid: int):
                if isinstance(model_names, (list, tuple)) and cid < len(model_names):
                    en = model_names[cid]
                else:
                    en = str(cid)
                return names_cn_map.get(en, en)
            df_stats = pd.DataFrame({
                "class_id": [int(x[0]) for x in stats],
                "name": [_cn_name(int(x[0])) for x in stats],
                "count": [x[1] for x in stats],
            })
            detail_col1, detail_col2 = st.columns(2)
            with detail_col1:
                st.dataframe(
                    df_stats,
                    use_container_width=True,
                    hide_index=True,
                    height=max(140, min(360, 40 + (len(df_stats) + 1) * 32)),
                )
            with detail_col2:
                if ss and ss[5]:
                    st.video(ss[5])