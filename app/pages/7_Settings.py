"""Settings page.

Shows pipeline paths, GPU status, and environment overrides. The
profile / registry / data-export UI was removed with the collection
pipeline.
"""

import sys
from pathlib import Path

import streamlit as st

app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar
from components.system_status import render_system_status


def show_settings_page() -> None:
    render_common_sidebar()

    st.title("⚙️ Settings")

    render_system_status()
    st.markdown("---")
    _render_env_overrides()
    st.markdown("---")
    _render_data_paths()
    st.markdown("---")
    _render_about()


def _render_env_overrides() -> None:
    st.subheader("Environment overrides")
    import os
    st.code(
        f"PYBULLET_HSR_ROOT = {os.environ.get('PYBULLET_HSR_ROOT', '/home/roboworks/repos/pybullet_hsr')}",
        language="text",
    )
    st.caption(
        "Set `PYBULLET_HSR_ROOT` before launching the app to point at a different dataset source."
    )


def _render_data_paths() -> None:
    st.subheader("Data paths")
    coordinator = st.session_state.path_coordinator
    summary = coordinator.get_path_summary()
    for key, path in summary.items():
        st.write(f"**{key}:** `{path}`")


def _render_about() -> None:
    st.subheader("About")
    st.write(
        "**HSR Perception** — RoboCup@Home YOLOv8 fine-tuning pipeline "
        "consuming datasets from `pybullet_hsr`."
    )


if __name__ == "__main__":
    show_settings_page()
