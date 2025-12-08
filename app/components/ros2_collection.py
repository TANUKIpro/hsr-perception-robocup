"""
ROS2 Collection Component

Provides UI for ROS2-based image capture.
"""

import streamlit as st
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_registry import ObjectRegistry, RegisteredObject
    from services.path_coordinator import PathCoordinator


def render_ros2_collection(
    obj: "RegisteredObject",
    registry: "ObjectRegistry",
    path_coordinator: "PathCoordinator"
):
    """
    Render ROS2 collection tab content.

    Args:
        obj: Current object being collected
        registry: ObjectRegistry instance
        path_coordinator: PathCoordinator instance
    """
    st.subheader("ROS2 Camera Capture")

    # Try to import ROS2 bridge
    ros2_bridge = _get_ros2_bridge()
    if ros2_bridge is None:
        return

    # Check ROS2 availability
    if not _render_ros2_status(ros2_bridge):
        return

    # Check capture node
    _render_capture_node_status(ros2_bridge)

    # Capture Application
    st.markdown("---")
    _render_capture_app_launcher(ros2_bridge, path_coordinator)

    # Application usage guide
    _render_usage_guide()


def _get_ros2_bridge():
    """Get ROS2Bridge instance or show error."""
    try:
        from services.ros2_bridge import ROS2Bridge
        return ROS2Bridge()
    except ImportError:
        st.error("ROS2 bridge not available")
        return None


def _render_ros2_status(ros2_bridge) -> bool:
    """
    Render ROS2 availability status.

    Returns:
        True if ROS2 is available
    """
    if not ros2_bridge.is_available():
        st.warning(
            "ROS2 is not available. Make sure:\n"
            "1. ROS2 Humble is installed\n"
            "2. The ROS2 environment is sourced\n"
            "3. The capture node is running"
        )

        st.code(
            "# HSR capture node:\n"
            "ros2 launch hsr_perception capture.launch.py\n\n"
            "# Xtion camera (see docs/xtion_setup.md):\n"
            "ros2 launch openni2_camera camera_only.launch.py",
            language="bash"
        )
        return False

    st.success("ROS2 connected")
    return True


def _render_capture_node_status(ros2_bridge):
    """Render capture node status."""
    if ros2_bridge.check_capture_node_running():
        st.success("Capture node is running")
    else:
        st.warning("Capture node not detected. Start it with:")
        st.code(
            "# HSR capture node:\n"
            "ros2 launch hsr_perception capture.launch.py\n\n"
            "# Xtion camera:\n"
            "ros2 launch openni2_camera camera_only.launch.py",
            language="bash"
        )


def _render_capture_app_launcher(ros2_bridge, path_coordinator: "PathCoordinator"):
    """Render capture application launcher section."""
    st.subheader("Image Capture Application")

    st.markdown("""
    Use the standalone GUI application for image capture.
    The application provides all capture controls in one place.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        # Get default output directories
        default_output = str(path_coordinator.get_path("raw_captures_dir"))
        video_dir = str(path_coordinator.get_path("videos_dir"))

        # Mode selection
        use_burst_mode = st.checkbox("Use Burst Capture (Legacy)", value=False)

        if use_burst_mode:
            if st.button("Launch Burst Capture App"):
                if ros2_bridge.open_capture_app(output_dir=default_output):
                    st.success("Burst capture application launched!")
                else:
                    st.error("Failed to launch burst capture application")
        else:
            if st.button("Launch Recording App", type="primary"):
                if ros2_bridge.open_record_app(output_dir=default_output, video_dir=video_dir):
                    st.success("Recording application launched!")
                else:
                    st.error("Failed to launch recording application")

    with col2:
        if use_burst_mode:
            st.markdown("""
            **Burst Capture Features:**
            - Topic selection (auto-detected)
            - Real-time preview with center reticle
            - Single capture / Burst capture
            - Configurable parameters (count, interval)
            """)
        else:
            st.markdown("""
            **Recording App Features:**
            - Topic selection (auto-detected)
            - Real-time preview with center reticle
            - Video recording with countdown
            - Automatic frame extraction (uniform intervals)
            - MP4 video saving
            """)


def _render_usage_guide():
    """Render application usage guide."""
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        ### Basic Operation

        1. **Topic Selection**: Select camera topic from dropdown
        2. **Class Name**: Enter the object class name
        3. **Output Directory**: Set the save directory

        ### Capture Modes

        - **Single Capture**: Capture one image
        - **Start Burst Capture**: Start continuous capture (3-second countdown)

        ### Parameters

        | Parameter | Description | Recommended |
        |-----------|-------------|-------------|
        | Images | Number of images | 50-100 |
        | Interval | Capture interval (sec) | 0.2-0.5 |
        | Overwrite | Start numbering from 1 | ON |
        """)
