"""
HSR Perception - Object Manager

Streamlit application for managing object registration, data collection,
and training preparation for RoboCup@Home competitions.

Features:
- Object registration with reference images
- Multi-source data collection (ROS2, camera, file upload)
- Auto-annotation pipeline integration
- YOLOv8 fine-tuning management
- Model evaluation and visual verification

Usage:
    streamlit run app/main.py
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="HSR Object Manager",
    page_icon="app/img/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar


def main():
    """Main application entry point - Home page."""
    # Render common sidebar (includes profile selector and stats)
    render_common_sidebar()

    # Home page content - Title with HSR icon (using HTML for inline display)
    import base64
    icon_path = Path(__file__).parent / "img" / "hsr_icon_small.png"
    if icon_path.exists():
        icon_b64 = base64.b64encode(icon_path.read_bytes()).decode()
        st.markdown(
            f'<h1 style="display: flex; align-items: center; gap: 10px;">'
            f'<img src="data:image/png;base64,{icon_b64}" width="40" height="40">'
            f'HSR Object Manager</h1>',
            unsafe_allow_html=True
        )
    else:
        st.title("🤖 HSR Object Manager")
    st.markdown("Object recognition pipeline manager for RoboCup@Home competitions")

    st.subheader("Pipeline Workflow")

    # Row 1: Data preparation (4 columns)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.page_link("pages/1_Dashboard.py", label="📊 Dashboard", use_container_width=True)
            st.caption("Overview & Progress")
            st.markdown("---")
            st.markdown("**In:** Registry, Stats")
            st.markdown("**Out:** Metrics, Status")

    with col2:
        with st.container(border=True):
            st.page_link("pages/2_Registry.py", label="📝 Registry", use_container_width=True)
            st.caption("Object Management")
            st.markdown("---")
            st.markdown("**In:** Object definitions")
            st.markdown("**Out:** `registry.json`")

    with col3:
        with st.container(border=True):
            st.page_link("pages/3_Collection.py", label="📸 Collection", use_container_width=True)
            st.caption("Image Acquisition")
            st.markdown("---")
            st.markdown("**In:** Camera / Upload")
            st.markdown("**Out:** `raw_captures/`")

    with col4:
        with st.container(border=True):
            st.page_link("pages/4_Annotation.py", label="🏷️ Annotation", use_container_width=True)
            st.caption("SAM2 Labeling")
            st.markdown("---")
            st.markdown("**In:** Images, SAM2")
            st.markdown("**Out:** YOLO labels")

    # Row 2: Model pipeline (3 columns)
    col5, col6, col7 = st.columns(3)

    with col5:
        with st.container(border=True):
            st.page_link("pages/5_Training.py", label="🎯 Training", use_container_width=True)
            st.caption("YOLOv8 Fine-tuning")
            st.markdown("---")
            st.markdown("**In:** `data.yaml`, GPU")
            st.markdown("**Out:** `best.pt`")

    with col6:
        with st.container(border=True):
            st.page_link("pages/6_Evaluation.py", label="📈 Evaluation", use_container_width=True)
            st.caption("Model Verification")
            st.markdown("---")
            st.markdown("**In:** Model, Test set")
            st.markdown("**Out:** mAP, Timing")

    with col7:
        with st.container(border=True):
            st.page_link("pages/7_Settings.py", label="⚙️ Settings", use_container_width=True)
            st.caption("Configuration")
            st.markdown("---")
            st.markdown("**In:** System config")
            st.markdown("**Out:** Export files")

    # Pipeline flow indicator
    st.markdown("---")
    st.markdown("**Pipeline:** Registry → Collection → Annotation → Training → Evaluation")
    st.caption("Target: mAP ≥ 85%, Inference ≤ 100ms")


if __name__ == "__main__":
    main()
