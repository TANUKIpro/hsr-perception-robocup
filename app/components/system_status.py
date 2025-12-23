"""
System Status Component

Provides UI for displaying system diagnostics (ROS2, GPU).
"""

import streamlit as st


def render_system_status():
    """Render system status section (ROS2 and GPU)."""
    st.subheader("System Status")

    col1, col2 = st.columns(2)

    with col1:
        render_ros2_diagnostics()

    with col2:
        render_gpu_status()


def render_ros2_diagnostics():
    """Render ROS2 diagnostics and command launcher."""
    st.write("**ROS2 Status:**")

    try:
        from services.ros2_bridge import ROS2Bridge

        ros2_bridge = ROS2Bridge()
        diagnostics = ros2_bridge.get_diagnostics()

        if diagnostics["ros2_available"]:
            st.success("ROS2 Available")
            st.write(f"  Topics: {diagnostics.get('topics_count', 'N/A')}")
            st.write(f"  Image Topics: {diagnostics.get('image_topics_count', 'N/A')}")
            st.write(f"  Nodes: {diagnostics.get('nodes_count', 'N/A')}")
            st.write(
                f"  Capture Node: {'Running' if diagnostics.get('capture_node_running') else 'Not Found'}"
            )

            # ROS2 Command Launcher
            st.markdown("---")
            from components.ros2_command_launcher import render_ros2_command_launcher
            from services.command_presets_manager import CommandPresetsManager
            from services.ros2_process_tracker import ROS2ProcessTracker

            presets_manager = CommandPresetsManager()
            process_tracker = ROS2ProcessTracker(ros2_bridge=ros2_bridge)

            render_ros2_command_launcher(
                presets_manager=presets_manager,
                process_tracker=process_tracker,
            )
        else:
            st.warning("ROS2 Not Available")

    except Exception as e:
        st.error(f"Error checking ROS2 status: {e}")


def render_gpu_status():
    """Render GPU availability and info."""
    st.write("**GPU Status:**")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success("GPU Available")
            st.write(f"  {gpu_name}")
            st.write(f"  Memory: {gpu_memory:.1f} GB")
        else:
            st.warning("GPU Not Available")
    except ImportError:
        st.warning("PyTorch not installed")
    except Exception as e:
        st.error(f"Error checking GPU status: {e}")
