"""
Settings Page

Provides UI for application settings and configuration.
"""

import streamlit as st
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_registry import ObjectRegistry
    from services.path_coordinator import PathCoordinator

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar
from components.system_status import render_system_status


def show_settings_page() -> None:
    """Settings page."""
    render_common_sidebar()

    st.title("⚙️ Settings")

    registry = st.session_state.registry
    path_coordinator = st.session_state.path_coordinator

    # Data Management
    _render_data_management(registry, path_coordinator)

    st.markdown("---")

    # System Status
    render_system_status()

    st.markdown("---")

    # Data Paths
    _render_data_paths(path_coordinator)

    st.markdown("---")

    # About
    _render_about()


def _render_data_management(registry: "ObjectRegistry", path_coordinator: "PathCoordinator") -> None:
    """Render data management section."""
    st.subheader("Data Management")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Export Options**")

        if st.button("Export to YOLO Config"):
            output_path = registry.export_to_yolo_config("config/object_classes.json")
            st.success(f"Exported to {output_path}")

        if st.button("Update All Collection Counts"):
            registry.update_all_collection_counts()
            st.success("Updated all counts")
            st.rerun()

        if st.button("Sync All to Datasets"):
            objects = registry.get_all_objects()
            synced = path_coordinator.sync_all_objects([obj.name for obj in objects])
            st.success(f"Synced {len(synced)} directories")

    with col2:
        st.write("**Category Management**")

        new_category = st.text_input("New Category")
        if st.button("Add Category") and new_category:
            registry.add_category(new_category)
            st.success(f"Added category: {new_category}")
            st.rerun()

        st.write("Current categories:")
        for cat in registry.categories:
            st.write(f"  • {cat}")


def _render_data_paths(path_coordinator: "PathCoordinator") -> None:
    """Render data paths summary."""
    st.subheader("Data Paths")

    path_summary = path_coordinator.get_path_summary()
    for key, path in path_summary.items():
        st.write(f"**{key}:** `{path}`")


def _render_about() -> None:
    """Render about section."""
    st.subheader("About")

    st.write("""
    **HSR Object Manager** v2.0

    A comprehensive tool for managing object recognition pipelines
    for RoboCup@Home competitions.
    """)

    # Creator info
    st.caption("Created by")
    col1, col2, col3 = st.columns([1, 1, 6])
    img_dir = app_dir / "img"
    with col1:
        if (img_dir / "tid_logo.svg").exists():
            st.image(str(img_dir / "tid_logo.svg"), width=120)
    with col2:
        if (img_dir / "ikeryo.jpg").exists():
            st.image(str(img_dir / "ikeryo.jpg"), width=120)


# For Streamlit native multipage
if __name__ == "__main__":
    show_settings_page()
