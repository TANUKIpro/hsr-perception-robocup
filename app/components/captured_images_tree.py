"""
Captured Images Tree Component

Provides UI for displaying captured images directory structure.
"""

import streamlit as st
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.path_coordinator import PathCoordinator


def render_captured_images_tree(path_coordinator: "PathCoordinator") -> None:
    """
    Render captured images directory tree.

    Args:
        path_coordinator: PathCoordinator instance
    """
    st.markdown("---")
    st.subheader("Collected Images")

    raw_captures_dir = path_coordinator.get_path("raw_captures_dir")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("Refresh", key="refresh_captures"):
            st.rerun()

    with col1:
        if not raw_captures_dir.exists():
            st.info("Capture directory not created yet.")
            return

        subdirs = sorted([d for d in raw_captures_dir.iterdir() if d.is_dir()])

        if not subdirs:
            st.info("No captured images yet. Launch the Capture App to start collecting.")
            return

        # Build tree structure
        tree_lines = [f"📁 {raw_captures_dir.name}/"]

        for i, subdir in enumerate(subdirs):
            # Count image files
            img_count = len(list(subdir.glob("*.jpg"))) + len(list(subdir.glob("*.png")))
            prefix = "└──" if i == len(subdirs) - 1 else "├──"
            tree_lines.append(f"  {prefix} 📂 {subdir.name} ({img_count} images)")

        st.code("\n".join(tree_lines), language=None)
