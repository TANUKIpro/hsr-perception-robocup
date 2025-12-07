"""
Common Sidebar Component

Provides consistent sidebar UI across all pages including:
- Profile selector
- Quick stats (Total Objects, Collection Progress)
- Active task indicator
"""

import streamlit as st
from pathlib import Path
import sys

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from services.profile_manager import ProfileManager
from services.path_coordinator import PathCoordinator
from services.task_manager import TaskManager
from object_registry import ObjectRegistry


def _ensure_session_state():
    """Ensure all required session state variables are initialized."""
    if "profile_manager" not in st.session_state:
        st.session_state.profile_manager = ProfileManager()

    if "path_coordinator" not in st.session_state:
        st.session_state.path_coordinator = PathCoordinator(
            profile_manager=st.session_state.profile_manager
        )

    if "registry" not in st.session_state:
        st.session_state.registry = ObjectRegistry(
            path_coordinator=st.session_state.path_coordinator
        )

    if "task_manager" not in st.session_state:
        st.session_state.task_manager = TaskManager(
            path_coordinator=st.session_state.path_coordinator
        )

    if "current_object_id" not in st.session_state:
        st.session_state.current_object_id = None


def _reinitialize_services():
    """Reinitialize all services after profile switch."""
    st.session_state.path_coordinator = PathCoordinator(
        profile_manager=st.session_state.profile_manager
    )
    st.session_state.registry = ObjectRegistry(
        path_coordinator=st.session_state.path_coordinator
    )
    st.session_state.task_manager = TaskManager(
        path_coordinator=st.session_state.path_coordinator
    )


def _render_profile_selector():
    """Render profile selector in sidebar."""
    profile_manager = st.session_state.profile_manager

    profiles = profile_manager.get_all_profiles()
    active_profile = profile_manager.get_active_profile()

    # Create options
    profile_options = {p.display_name: p.id for p in profiles}
    profile_names = list(profile_options.keys())

    # Find current index
    current_idx = 0
    for i, p_id in enumerate(profile_options.values()):
        if p_id == active_profile.id:
            current_idx = i
            break

    selected_name = st.sidebar.selectbox(
        "Profile",
        profile_names,
        index=current_idx,
        key="profile_selector"
    )

    selected_id = profile_options[selected_name]

    # Handle profile switch
    if selected_id != active_profile.id:
        profile_manager.set_active_profile(selected_id)
        _reinitialize_services()
        st.rerun()


def render_common_sidebar():
    """
    Render common sidebar for all pages.

    Includes:
    - Title
    - Profile selector
    - Quick stats
    - Active task indicator
    """
    # Ensure session state is initialized
    _ensure_session_state()

    # Title
    st.sidebar.title("HSR Object Manager")

    # Profile selector
    _render_profile_selector()

    st.sidebar.markdown("---")

    # Quick stats
    registry = st.session_state.registry

    # Update collection counts before displaying
    registry.update_all_collection_counts()

    stats = registry.get_collection_stats()

    st.sidebar.metric("Total Objects", stats["total_objects"])
    st.sidebar.metric(
        "Collection Progress",
        f"{stats['total_collected']}/{stats['total_target']}",
        f"{stats['progress_percent']:.1f}%"
    )

    # Active task indicator
    task_manager = st.session_state.task_manager
    active_tasks = task_manager.get_active_tasks()
    if active_tasks:
        st.sidebar.markdown("---")
        st.sidebar.warning(f"🔄 {len(active_tasks)} task(s) running")
