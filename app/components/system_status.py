"""System status component.

Shows GPU availability for training. ROS2 diagnostics were removed
along with the capture pipeline.
"""

import streamlit as st


def render_system_status() -> None:
    st.subheader("System Status")
    render_gpu_status()


def render_gpu_status() -> None:
    st.write("**GPU:**")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success(f"{gpu_name} · {gpu_memory:.1f} GB")
        else:
            st.warning("GPU not available — training will be slow on CPU.")
    except ImportError:
        st.warning("PyTorch not installed")
    except Exception as e:
        st.error(f"Error checking GPU status: {e}")
