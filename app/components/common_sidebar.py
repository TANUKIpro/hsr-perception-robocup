"""Common sidebar rendered on every Streamlit page.

Minimal sidebar for the train + evaluate pipeline: shows the active task
count and the pybullet_hsr dataset source. Profile / Registry concepts
were removed along with the collection pipeline.
"""

import sys
from pathlib import Path

import streamlit as st

app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from services.path_coordinator import (
    PYBULLET_HSR_ANNOTATION_ROOT,
    PYBULLET_HSR_ROOT,
    PathCoordinator,
)
from services.task_manager import TaskManager
from services.ui_settings_manager import UISettingsManager


def _ensure_session_state() -> None:
    """Initialize session-state singletons if missing."""
    if "path_coordinator" not in st.session_state:
        st.session_state.path_coordinator = PathCoordinator()

    if "task_manager" not in st.session_state:
        st.session_state.task_manager = TaskManager(
            path_coordinator=st.session_state.path_coordinator
        )

    if "ui_settings_manager" not in st.session_state:
        st.session_state.ui_settings_manager = UISettingsManager(
            path_coordinator=st.session_state.path_coordinator
        )

    if "_ui_settings_loaded" not in st.session_state:
        st.session_state.ui_settings_manager.load_to_session_state()
        st.session_state._ui_settings_loaded = True


def render_common_sidebar() -> None:
    """Render the minimal sidebar used across pages."""
    _ensure_session_state()

    st.sidebar.title("HSR Perception")
    st.sidebar.caption("Training + Evaluation")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**pybullet_hsr source**")
    st.sidebar.code(str(PYBULLET_HSR_ROOT), language="text")
    coordinator: PathCoordinator = st.session_state.path_coordinator
    if PYBULLET_HSR_ANNOTATION_ROOT.is_dir():
        dumps = coordinator.get_available_pybullet_hsr_dumps()
        if dumps:
            st.sidebar.success(f"{len(dumps)} dump(s) with manifest")
        else:
            st.sidebar.warning(
                "No manifest-bearing dumps found. Run pybullet_hsr's "
                "scripts/write_manifest.py on an existing dump."
            )
    else:
        st.sidebar.warning("annotation_data/ missing — set PYBULLET_HSR_ROOT")

    task_manager = st.session_state.task_manager
    active_tasks = task_manager.get_active_tasks()
    if active_tasks:
        st.sidebar.markdown("---")
        st.sidebar.warning(f"{len(active_tasks)} task(s) running")
