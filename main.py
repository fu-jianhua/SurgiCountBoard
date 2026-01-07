import os
import time
import cv2
import torch
import streamlit as st
import pandas as pd
import numpy as np
from core.source import open_capture
from core.utils import FPSMeter
from core.roi import ROIRect
from core.pipeline import Pipeline
from core.session import SessionManager
from core.video import open_writer, write_frame, close_writer
from core.multi import MultiCameraFusion, Event
from core.store import (
    init_db,
    add_detection,
    session_stats,
    end_session,
    get_session,
    search_sessions,
    start_session,
    add_cam_session,
    add_event,
    add_fused_event,
    get_cam_sessions,
)

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

default_model = os.path.join("e:\\project\\ultralytics\\runs\\train\\medical_yolo11m_ns2\\weights", "best.pt")

with st.sidebar:
    mode = st.selectbox("运行模式", ["单摄像头", "多摄像头"], index=0)
    btn_col1, btn_col2 = st.columns(2)
    start_btn = btn_col1.button("开始")
    stop_btn = btn_col2.button("停止")
    with st.expander("视频源与设备", expanded=True):
        model_path = st.text_input("模型路径", default_model)
        source_str = st.text_input("视频源", "0") if mode == "单摄像头" else st.text_input("多视频源(逗号分隔)", "0,1")
        device_inp = st.text_input("设备(0/1 或 cpu)", "0")
        half = st.checkbox("FP16 半精度", True)
        low_latency = st.checkbox("低延迟模式", False)
    with st.expander("推理与跟踪", expanded=False):
        conf = st.slider("置信度", 0.0, 1.0, 0.25, 0.01)
        iou = st.slider("IoU", 0.0, 1.0, 0.45, 0.01)
        track_enabled = st.checkbox("启用跟踪", True)
        seg_enabled = st.checkbox("启用SEG辅助判别", False)
        imgsz = st.number_input("推理分辨率(imgsz)", min_value=256, max_value=1280, value=640, step=64)
        max_det = st.number_input("最大检测数(max_det)", min_value=10, max_value=1000, value=200, step=10)
    with st.expander("会话与ROI", expanded=False):
        idle_seconds = st.number_input("空窗秒数", min_value=1, max_value=120, value=10, step=1)
        if "line_pos_pct" not in st.session_state:
            st.session_state.line_pos_pct = 60
        if "count_mode" not in st.session_state:
            st.session_state.count_mode = "roi"
        count_mode_label = st.selectbox("计数方式", ["ROI", "计数线"], index=0 if st.session_state.count_mode == "roi" else 1)
        st.session_state.count_mode = "roi" if count_mode_label == "ROI" else "line"
        line_pos_slider = st.slider("计数线位置(%)", 0, 100, int(st.session_state.line_pos_pct), 1)
        st.session_state.line_pos_pct = int(line_pos_slider)
    fps_box = st.empty()

init_db()
if "running" not in st.session_state:
    st.session_state.running = False
if "status" not in st.session_state:
    st.session_state.status = "已停止"
_render_status()

@st.cache_resource(show_spinner=False)
def _load_model(path):
    from ultralytics import YOLO
    return YOLO(path)

def _open_sources(srcs, low_latency):
    caps = []
    for s in srcs:
        caps.append(open_capture(s, low_latency=low_latency))
    return caps

def _single_run():
    col1, col2 = st.columns(2)
    org = col1.empty()
    ann = col2.empty()
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "writer" not in st.session_state:
        st.session_state.writer = None
    if "session" not in st.session_state:
        st.session_state.session = None
    if "running_counts" not in st.session_state:
        st.session_state.running_counts = {}
    if start_btn and not st.session_state.running:
        st.session_state.status = "启动中..."; _render_status(); st.session_state.running = True
        cap = open_capture(source_str, low_latency=low_latency); st.session_state.cap = cap
        if not cap.isOpened():
            st.error("无法打开视频源"); st.session_state.running = False; _render_status(); return
        dev = int(device_inp) if device_inp.isdigit() else device_inp
        use_half = half
        if isinstance(dev, int) and not torch.cuda.is_available():
            dev = "cpu"; use_half = False
        if isinstance(dev, str) and dev.lower() == "cpu":
            use_half = False
        model_obj = _load_model(model_path)
        pipeline = Pipeline(model=model_obj, conf=conf, iou=iou, use_track=bool(track_enabled), device=dev, half=use_half, imgsz=int(imgsz), max_det=int(max_det), frame_rate=float(cap.get(cv2.CAP_PROP_FPS) or 30.0), seg_model=os.path.join(os.path.dirname(__file__), "yolo11x-seg.pt") if bool(seg_enabled) else None, line_pos=float(st.session_state.get("line_pos_pct", 70))/100.0, count_mode=str(st.session_state.get("count_mode","roi")))
        st.session_state.pipeline = pipeline
        ok, frame = cap.read()
        if not ok:
            st.error("无法读取帧"); st.session_state.running = False; _render_status(); return
        h,w = frame.shape[:2]
        roi = ROIRect(0,0,w-1,h-1)
        st.session_state.session = SessionManager(camera_id=str(source_str), idle_seconds=int(idle_seconds), roi_json=roi.to_json())
        st.session_state.roi = roi
        st.session_state.status = "运行中"; _render_status()
    if stop_btn and st.session_state.running:
        st.session_state.running = False
        if st.session_state.writer is not None:
            close_writer(st.session_state.writer); st.session_state.writer = None
        if st.session_state.session is not None and getattr(st.session_state.session,"current_session_id",None) is not None:
            vp = getattr(st.session_state,"video_path",None)
            end_session(st.session_state.session.current_session_id, time.time(), vp)
        if st.session_state.cap is not None:
            st.session_state.cap.release(); st.session_state.cap = None
        st.session_state.status = "已停止"; _render_status()
    if st.session_state.running and st.session_state.cap is not None and st.session_state.pipeline is not None:
        try:
            det_streak = 0; no_det_streak = 0; start_frames = 20; stop_frames = 20
            fps_meter = FPSMeter()
            last_ts = time.time()
            while st.session_state.running and st.session_state.cap.isOpened():
                ok, frame = st.session_state.cap.read()
                if not ok: break
                annotated, counts, events, roi_det = st.session_state.pipeline.process(frame, st.session_state.roi)
                if isinstance(counts, dict) and len(counts) > 0:
                    for k, v in counts.items():
                        st.session_state.running_counts[k] = int(st.session_state.running_counts.get(k, 0)) + int(v)
                if isinstance(st.session_state.running_counts, dict) and len(st.session_state.running_counts) > 0:
                    ov = annotated.copy()
                    box_h = 28 * (len(st.session_state.running_counts) + 1)
                    cv2.rectangle(ov, (8, 8), (280, 8 + box_h), (0, 0, 0), -1)
                    annotated = cv2.addWeighted(ov, 0.35, annotated, 0.65, 0)
                    y = 32
                    for k in sorted(st.session_state.running_counts.keys()):
                        v = st.session_state.running_counts[k]
                        cv2.putText(annotated, f"{k}: {v}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        y += 26
                sf = 960.0 / frame.shape[1] if frame.shape[1] > 960 else 1.0
                dframe = cv2.resize(frame, (int(frame.shape[1]*sf), int(frame.shape[0]*sf))) if sf!=1.0 else frame
                dann = cv2.resize(annotated, (int(annotated.shape[1]*sf), int(annotated.shape[0]*sf))) if sf!=1.0 else annotated
                now = time.time()
                fps_meter.tick(now)
                if roi_det: det_streak += 1; no_det_streak = 0
                else: no_det_streak += 1; det_streak = 0
                if st.session_state.writer is None and det_streak >= start_frames:
                    sid = st.session_state.session.on_detection(now)
                    out_dir = os.path.join("runs","surgicountboard"); os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.abspath(os.path.join(out_dir, f"session_{sid}.mp4"))
                    st.session_state.video_path = out_path
                    est_fps = fps_meter.fps or (st.session_state.cap.get(cv2.CAP_PROP_FPS) or 25)
                    if est_fps is None or est_fps <= 0:
                        est_fps = 10.0
                    est_fps = float(max(4.0, min(30.0, est_fps)))
                    st.session_state.writer = open_writer(out_path, annotated.shape, fps=est_fps)
                elif st.session_state.writer is not None and roi_det:
                    st.session_state.session.on_detection(now)
                sid = st.session_state.session.current_session_id
                for ev in events:
                    ts, cls_id, tid, c, cx, cy = ev
                    if sid is not None and tid is not None and tid >= 0:
                        add_detection(sid, ts, cls_id, tid, c)
                if st.session_state.writer is not None and no_det_streak >= stop_frames:
                    ended = st.session_state.session.check_idle_and_end(now, st.session_state.video_path)
                    if ended is not None:
                        close_writer(st.session_state.writer); st.session_state.writer = None; st.session_state.video_path = None
                if st.session_state.writer is not None:
                    write_frame(st.session_state.writer, annotated)
                org.image(dframe, channels="BGR", output_format="JPEG")
                ann.image(dann, channels="BGR", output_format="JPEG")
                try:
                    cur_fps = fps_meter.fps
                    if cur_fps:
                        fps_box.markdown(f"处理FPS：{cur_fps:.1f}")
                except Exception:
                    pass
                time.sleep(0.01 if low_latency else 0.03)
        finally:
            if st.session_state.cap is not None:
                st.session_state.cap.release(); st.session_state.cap = None

def _multi_run():
    if "multi_id" not in st.session_state:
        st.session_state.multi_id = None
    srcs = [s.strip() for s in source_str.split(",") if s.strip()]
    if start_btn and not st.session_state.running:
        st.session_state.status = "启动中..."; _render_status(); st.session_state.running = True
        caps = _open_sources(srcs, low_latency); st.session_state.captures = caps
        dev = int(device_inp) if device_inp.isdigit() else device_inp
        use_half = half
        if isinstance(dev, int) and not torch.cuda.is_available():
            dev = "cpu"; use_half = False
        if isinstance(dev, str) and dev.lower() == "cpu":
            use_half = False
        model_obj = _load_model(model_path)
        # 立即创建多摄像头会话，camera_id 保存为多源字符串，保证统一的 start/end
        st.session_state.multi_id = start_session(str(source_str), "{}", time.time())
        st.session_state.cam_video_paths = [None] * len(caps)
        pipelines = []; rois = []
        for idx, cap in enumerate(caps):
            ok, frame = cap.read()
            if ok:
                h, w = frame.shape[:2]
                roi = ROIRect(0, 0, w - 1, h - 1)
            else:
                roi = None
            rois.append(roi)
            pl = Pipeline(model=model_obj, conf=conf, iou=iou, use_track=bool(track_enabled), device=dev, half=use_half, imgsz=int(imgsz), max_det=int(max_det), frame_rate=float(cap.get(cv2.CAP_PROP_FPS) or 30.0), seg_model=os.path.join(os.path.dirname(__file__), "yolo11x-seg.pt") if bool(seg_enabled) else None, line_pos=float(st.session_state.get("line_pos_pct", 70))/100.0, count_mode=str(st.session_state.get("count_mode","roi")))
            pipelines.append(pl)
        left_col, right_col = st.columns(2)
        import math
        def _build_grid(parent, n, grid_cols=2):
            items = []
            rows = int(math.ceil(n / float(grid_cols)))
            idx = 0
            for _ in range(rows):
                cols = parent.columns(grid_cols)
                for c in range(grid_cols):
                    if idx < n:
                        items.append(cols[c].empty())
                        idx += 1
            return items
        org_conts = _build_grid(left_col, len(pipelines), grid_cols=2)
        ann_conts = _build_grid(right_col, len(pipelines), grid_cols=2)
        fusion = MultiCameraFusion(time_thr=0.3, dist_thr=32.0)
        st.session_state.status = "运行中"; _render_status()
        writers = [None]*len(pipelines)
        if "multi_counts" not in st.session_state:
            st.session_state.multi_counts = [{} for _ in pipelines]
        meters = [FPSMeter() for _ in pipelines]
        det_streak = [0]*len(pipelines); no_det_streak = [0]*len(pipelines)
        try:
            while st.session_state.running:
                for idx, cap in enumerate(caps):
                    if not cap.isOpened():
                        continue
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    if rois[idx] is None:
                        h, w = frame.shape[:2]
                        rois[idx] = ROIRect(0, 0, w - 1, h - 1)
                    annotated, counts, events, roi_det = pipelines[idx].process(frame, rois[idx])
                    if isinstance(counts, dict) and len(counts) > 0:
                        mc = st.session_state.multi_counts[idx]
                        for k, v in counts.items():
                            mc[k] = int(mc.get(k, 0)) + int(v)
                        st.session_state.multi_counts[idx] = mc
                    curc = st.session_state.multi_counts[idx]
                    if isinstance(curc, dict) and len(curc) > 0:
                        ov = annotated.copy()
                        box_h = 28 * (len(curc) + 1)
                        cv2.rectangle(ov, (8, 8), (280, 8 + box_h), (0, 0, 0), -1)
                        annotated = cv2.addWeighted(ov, 0.35, annotated, 0.65, 0)
                        y = 32
                        for k in sorted(curc.keys()):
                            v = curc[k]
                            cv2.putText(annotated, f"{k}: {v}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            y += 26
                    sf = 480.0 / frame.shape[1] if frame.shape[1] > 480 else 1.0
                    dframe = cv2.resize(frame, (int(frame.shape[1]*sf), int(frame.shape[0]*sf))) if sf!=1.0 else frame
                    dann = cv2.resize(annotated, (int(annotated.shape[1]*sf), int(annotated.shape[0]*sf))) if sf!=1.0 else annotated
                    now = time.time()
                    meters[idx].tick(now)
                    if roi_det: det_streak[idx]+=1; no_det_streak[idx]=0
                    else: no_det_streak[idx]+=1; det_streak[idx]=0
                    if st.session_state.multi_id is None and det_streak[idx] >= 20:
                        st.session_state.multi_id = start_session("multi", "{}", now)
                    # 一旦多摄像头会话创建，立即为每路开启写入器，确保文件存在
                    if writers[idx] is None and st.session_state.multi_id is not None:
                        out_dir = os.path.join("runs","surgicountboard"); os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.abspath(os.path.join(out_dir, f"session_{st.session_state.multi_id}_cam_{idx}.mp4"))
                        est_fps = meters[idx].fps or (cap.get(cv2.CAP_PROP_FPS) or 25)
                        if est_fps is None or est_fps <= 0:
                            est_fps = 10.0
                        est_fps = float(max(4.0, min(30.0, est_fps)))
                        writers[idx] = open_writer(out_path, annotated.shape, fps=est_fps)
                        add_cam_session(int(st.session_state.multi_id), int(idx), int(st.session_state.multi_id), out_path)
                        try:
                            st.session_state.cam_video_paths[idx] = out_path
                        except Exception:
                            pass
                    for ev in events:
                        ts, cls_id, tid, c, cx, cy = ev
                        if st.session_state.multi_id is not None and tid is not None and tid >= 0:
                            add_detection(int(st.session_state.multi_id), ts, int(cls_id), int(idx)*100000 + int(tid), float(c))
                            add_event(int(st.session_state.multi_id), int(idx), ts, int(cls_id), int(tid), float(cx), float(cy), float(c))
                            evobj = Event(ts=ts, class_id=int(cls_id), track_id=int(tid), cam_index=int(idx), x=float(cx), y=float(cy), conf=float(c))
                            fusion.push(evobj)
                            fused = fusion.try_fuse()
                            if fused is not None:
                                fts, fcls, members = fused
                                add_fused_event(int(st.session_state.multi_id), float(fts), int(fcls), fusion.dump_members_json(members))
                    if writers[idx] is not None:
                        write_frame(writers[idx], annotated)
                    org_conts[idx].image(dframe, channels="BGR", output_format="JPEG")
                    ann_conts[idx].image(dann, channels="BGR", output_format="JPEG")
                    try:
                        cur_fps = meters[idx].fps
                        if cur_fps:
                            fps_box.markdown(f"处理FPS（cam{idx}）：{cur_fps:.1f}")
                    except Exception:
                        pass
                if stop_btn:
                    st.session_state.running = False
                time.sleep(0.01 if low_latency else 0.03)
        finally:
            for w in writers:
                if w is not None:
                    close_writer(w)
            if st.session_state.multi_id is not None:
                try:
                    vp = ";".join([p for p in st.session_state.cam_video_paths if p]) if hasattr(st.session_state, "cam_video_paths") else None
                except Exception:
                    vp = None
                end_session(int(st.session_state.multi_id), time.time(), vp)
            for cap in caps:
                try:
                    cap.release()
                except Exception:
                    pass
    if stop_btn and st.session_state.running:
        st.session_state.running = False
        # 统一写入会话结束的视频路径
        try:
            if st.session_state.multi_id is not None:
                vp = ";".join([p for p in st.session_state.cam_video_paths if p]) if hasattr(st.session_state, "cam_video_paths") else None
                end_session(int(st.session_state.multi_id), time.time(), vp)
        except Exception:
            pass
        # 立即释放所有采集器，避免本机摄像头依旧亮着
        try:
            caps = st.session_state.get("captures", [])
            for cap in caps:
                cap.release()
        except Exception:
            pass
        st.session_state.status = "已停止"; _render_status()

if mode == "单摄像头":
    _single_run()
else:
    _multi_run()

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
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    df = pd.DataFrame([{ "id": x[0], "camera": x[1], "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x[2])) if x[2] else "", "end": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(x[3])) if x[3] else "", "video": x[4]} for x in sessions])
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
                model_names = [x for x in _load_model(model_path).names.values()] if isinstance(_load_model(model_path).names, dict) else _load_model(model_path).names
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
        dc1, dc2 = st.columns(2)
        with dc1:
            st.dataframe(df_stats, use_container_width=True, hide_index=True, height=max(140, min(360, 40 + (len(df_stats) + 1) * 32)))
        with dc2:
            cams = get_cam_sessions(int(sid_sel))
            if len(cams) == 0 and ss and ss[5]:
                st.video(ss[5])
            else:
                gcols = st.columns(2)
                for i, (cam_index, _, vp) in enumerate(cams):
                    c = gcols[i % 2]
                    if vp:
                        c.video(vp)
