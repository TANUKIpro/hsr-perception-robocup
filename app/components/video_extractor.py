"""
Video Extractor Component

Provides UI for extracting frames from recorded videos.
"""

import streamlit as st
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from services.path_coordinator import PathCoordinator


def render_video_extractor(path_coordinator: "PathCoordinator"):
    """
    Render video frame extractor UI.

    Args:
        path_coordinator: PathCoordinator instance
    """
    st.markdown("---")
    st.subheader("Video Frame Extractor")
    st.caption("Extract frames from recorded videos to create training data.")

    videos_dir = path_coordinator.get_path("videos_dir")
    video_files = list(videos_dir.glob("*.mp4")) if videos_dir.exists() else []

    if not video_files:
        st.info("No videos found. Use the Recording App to capture videos first.")
        return

    # Video selection
    video_options = {f.name: f for f in sorted(video_files, reverse=True)}
    selected_name = st.selectbox("Select Video", list(video_options.keys()), key="video_select")
    selected_video = video_options[selected_name]

    # Extract class name from filename
    class_name = _extract_class_name(selected_name)
    st.write(f"**Class Name:** `{class_name}`")

    # Parameters
    target_frames = st.slider("Target Frames", 10, 200, 50, key="extract_frames")

    # Preview directory
    preview_dir = Path("/tmp/frame_preview")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Extract Preview", key="extract_preview"):
            _extract_frames_preview(selected_video, target_frames, class_name, preview_dir)

    with col2:
        if st.button("Save to Dataset", type="primary", key="save_to_dataset"):
            _save_frames_to_dataset(preview_dir, class_name, path_coordinator)

    # Show preview thumbnails
    _render_preview_thumbnails(preview_dir)


def _extract_class_name(filename: str) -> str:
    """
    Extract class name from video filename.

    Format: classname_yyyymmdd-hh-mm.mp4
    Handle cases where classname contains underscores.
    """
    # Split from right, max 2 splits
    parts = filename.rsplit("_", 2)
    if len(parts) >= 2:
        class_name = "_".join(parts[:-2]) if len(parts) > 2 else parts[0]
    else:
        class_name = filename.replace(".mp4", "")
    return class_name


def _extract_frames_preview(
    video_path: Path,
    target_frames: int,
    class_name: str,
    preview_dir: Path
):
    """Extract frames from video to preview directory."""
    import cv2

    # Create and clear preview directory
    preview_dir.mkdir(exist_ok=True)
    for f in preview_dir.glob("*.jpg"):
        f.unlink()

    # Extract frames
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        st.error("Video has no frames")
        cap.release()
        return

    # Calculate uniform indices
    if total_frames <= target_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / target_frames) for i in range(target_frames)]

    # Extract
    with st.spinner(f"Extracting {len(indices)} frames..."):
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                filename = f"{class_name}_{i+1}.jpg"
                cv2.imwrite(str(preview_dir / filename), frame)

    cap.release()
    st.success(f"Extracted {len(indices)} frames to preview")
    st.rerun()


def _save_frames_to_dataset(
    preview_dir: Path,
    class_name: str,
    path_coordinator: "PathCoordinator"
) -> int:
    """
    Save preview frames to dataset.

    Returns:
        Number of frames saved
    """
    preview_files = list(preview_dir.glob("*.jpg")) if preview_dir.exists() else []
    if not preview_files:
        st.warning("No preview frames. Click 'Extract Preview' first.")
        return 0

    # Get output directory
    output_dir = path_coordinator.get_path("raw_captures_dir") / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find next file number
    pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$")
    max_num = 0
    for f in output_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            num = int(match.group(1))
            max_num = max(max_num, num)
    start_num = max_num + 1

    # Copy files with new numbering
    saved = 0
    for i, src in enumerate(sorted(preview_files)):
        dst = output_dir / f"{class_name}_{start_num + i}.jpg"
        shutil.copy2(src, dst)
        saved += 1

    # Clear preview
    for f in preview_files:
        f.unlink()

    st.success(f"Saved {saved} images to {output_dir} (#{start_num}-{start_num + saved - 1})")
    st.rerun()
    return saved


def _render_preview_thumbnails(preview_dir: Path):
    """Render preview thumbnails grid."""
    preview_files = sorted(preview_dir.glob("*.jpg")) if preview_dir.exists() else []
    if not preview_files:
        return

    st.write(f"**Preview:** ({len(preview_files)} images)")

    # Show first 12 as thumbnails
    display_files = preview_files[:12]
    cols = st.columns(4)
    for i, img_path in enumerate(display_files):
        with cols[i % 4]:
            st.image(str(img_path), use_container_width=True, caption=img_path.name)

    if len(preview_files) > 12:
        st.caption(f"... and {len(preview_files) - 12} more")
