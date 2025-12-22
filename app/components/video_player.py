"""
Video Player Component for Visual Test

Provides UI for video-based model evaluation with frame-by-frame navigation
and real-time inference display.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import time

import cv2
import numpy as np
import streamlit as st

if TYPE_CHECKING:
    from services.path_coordinator import PathCoordinator


@dataclass
class VideoInfo:
    """Video file metadata."""

    path: Path
    filename: str
    total_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float


def get_video_info(video_path: Path) -> Optional[VideoInfo]:
    """
    Get video file metadata.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo object or None if failed to open video
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        info = VideoInfo(
            path=video_path,
            filename=video_path.name,
            total_frames=total_frames,
            fps=fps,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            duration_sec=total_frames / fps if fps > 0 else 0.0,
        )
        return info
    finally:
        cap.release()


def read_video_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    Read specific frame from video.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to read (0-based)

    Returns:
        Frame as numpy array (BGR) or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        return frame if ret else None
    finally:
        cap.release()


def render_video_player(
    path_coordinator: "PathCoordinator",
    model,
    conf_threshold: float,
) -> None:
    """
    Render video player component with inference.

    Args:
        path_coordinator: PathCoordinator instance for path management
        model: Loaded YOLO model
        conf_threshold: Confidence threshold for detection
    """
    # 1. Video selection
    videos_dir = path_coordinator.get_path("videos_dir")
    video_files = (
        sorted(videos_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
        if videos_dir.exists()
        else []
    )

    if not video_files:
        st.info("No videos found. Use Recording App to capture videos.")
        st.caption(f"Expected location: `{videos_dir}`")
        return

    selected_video = st.selectbox(
        "Select Video",
        video_files,
        format_func=lambda x: x.name,
        key="video_player_select",
    )

    video_info = get_video_info(selected_video)
    if not video_info:
        st.error("Failed to load video")
        return

    # Video info display
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Resolution", f"{video_info.width}x{video_info.height}")
    c2.metric("Frames", video_info.total_frames)
    c3.metric("FPS", f"{video_info.fps:.1f}")
    c4.metric("Duration", f"{video_info.duration_sec:.1f}s")

    st.markdown("---")
    st.subheader("Frame Navigation")

    # 2. Frame navigation with pending state pattern
    pending_key = "video_frame_pending"
    slider_key = "video_frame_slider"
    playing_key = "video_playing"

    # Initialize playing state
    if playing_key not in st.session_state:
        st.session_state[playing_key] = False

    # Calculate max frame
    max_frame = max(0, video_info.total_frames - 1)

    # Apply pending frame if exists (from play action)
    # Must update slider_key directly because Streamlit slider uses session state
    if pending_key in st.session_state:
        pending_frame = st.session_state.pop(pending_key)
        pending_frame = max(0, min(pending_frame, max_frame))
        st.session_state[slider_key] = pending_frame

    # Playback controls and slider on same row: [Play] [----Slider----]
    col_btn, col_slider = st.columns([1, 5])

    with col_btn:
        if st.session_state[playing_key]:
            if st.button("Stop", key="video_stop", use_container_width=True):
                st.session_state[playing_key] = False
                st.rerun()
        else:
            if st.button("Play", key="video_play", use_container_width=True):
                st.session_state[playing_key] = True
                st.rerun()

    with col_slider:
        current_frame = st.slider(
            "Frame",
            min_value=0,
            max_value=max_frame,
            key=slider_key,
            format=f"%d / {video_info.total_frames}",
            label_visibility="collapsed",
        )

    # Frame counter display
    current_time = current_frame / video_info.fps if video_info.fps > 0 else 0
    status = " (Playing...)" if st.session_state[playing_key] else ""
    st.caption(
        f"Frame {current_frame + 1} / {video_info.total_frames}{status} — "
        f"{current_time:.1f}s / {video_info.duration_sec:.1f}s"
    )

    st.markdown("---")

    # 3. Read frame and run inference
    frame = read_video_frame(video_info.path, current_frame)
    if frame is None:
        st.error(
            f"Failed to read frame {current_frame}. "
            "The video file may be corrupted."
        )
        return

    start_time = time.time()
    results = model(frame, conf=conf_threshold, verbose=False)
    inference_time = (time.time() - start_time) * 1000
    result = results[0]
    annotated = result.plot()

    # 4. Display frames
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Original**")
        st.image(frame, channels="BGR", use_container_width=True)
    with col2:
        st.markdown("**Prediction**")
        st.image(annotated, channels="BGR", use_container_width=True)

    # Metrics
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Frame", f"{current_frame + 1}/{video_info.total_frames}")
    m2.metric("Detections", len(result.boxes))
    m3.metric("Inference", f"{inference_time:.1f}ms")

    # Detection list
    if len(result.boxes) > 0:
        st.markdown("**Detections:**")
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            conf = box.conf.item()
            bbox = box.xyxy[0].tolist()
            st.write(
                f"- **{class_name}**: {conf:.2%} "
                f"(bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}])"
            )
    else:
        st.info("No objects detected")

    # Auto-advance if playing
    if st.session_state[playing_key]:
        if current_frame < max_frame:
            time.sleep(0.1)  # ~10fps (considering inference time)
            st.session_state[pending_key] = current_frame + 1
            st.rerun()
        else:
            # Reached end, stop playing
            st.session_state[playing_key] = False
            st.rerun()
