"""
Dashboard Page

Main dashboard with profile management, collection statistics,
and training readiness overview.
"""

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar, _reinitialize_services
from components.profile_management import render_profile_management
from object_registry import RegisteredObject


def show_dashboard_page() -> None:
    """Render the dashboard page with overview statistics."""
    # Render common sidebar
    render_common_sidebar()

    st.title("📊 Dashboard")

    # Profile Management Section
    render_profile_management(
        profile_manager=st.session_state.profile_manager,
        on_profile_deleted=_reinitialize_services
    )

    st.markdown("---")

    registry = st.session_state.registry

    # Update collection counts from filesystem before displaying stats
    registry.update_all_collection_counts()

    stats = registry.get_collection_stats()
    objects = registry.get_all_objects()

    # Overall progress
    _render_overall_stats(stats)

    st.markdown("---")

    # Pipeline status
    _render_pipeline_status()

    st.markdown("---")

    # Progress by category
    _render_category_progress(stats)

    st.markdown("---")

    # Per-object progress
    _render_object_progress(objects)

    # Training readiness check
    st.markdown("---")
    _render_training_readiness(objects)


def _render_overall_stats(stats: dict[str, Any]) -> None:
    """Render overall statistics metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Objects", stats["total_objects"])
    with col2:
        st.metric("Images Collected", stats["total_collected"])
    with col3:
        st.metric("Target Total", stats["total_target"])
    with col4:
        ready_pct = (stats["ready_objects"] / stats["total_objects"] * 100) if stats["total_objects"] > 0 else 0
        st.metric("Ready for Training", f"{stats['ready_objects']}/{stats['total_objects']}", f"{ready_pct:.0f}%")


def _render_pipeline_status() -> None:
    """Render pipeline status section."""
    st.subheader("Pipeline Status")

    task_manager = st.session_state.task_manager
    path_coordinator = st.session_state.path_coordinator

    col1, col2, col3 = st.columns(3)

    with col1:
        # Annotation status
        annotation_sessions = path_coordinator.get_annotation_sessions()
        ready_datasets = [s for s in annotation_sessions if s["has_data_yaml"]]
        st.metric("Annotated Datasets", len(ready_datasets))

    with col2:
        # Training status
        trained_models = path_coordinator.get_trained_models()
        st.metric("Trained Models", len(trained_models))

    with col3:
        # Active tasks
        active_tasks = task_manager.get_active_tasks()
        st.metric("Active Tasks", len(active_tasks))


def _render_category_progress(stats: dict[str, Any]) -> None:
    """Render progress by category."""
    st.subheader("Progress by Category")

    if stats["by_category"]:
        cols = st.columns(len(stats["by_category"]))
        for i, (cat, cat_stats) in enumerate(stats["by_category"].items()):
            with cols[i]:
                pct = (cat_stats["collected"] / cat_stats["target"] * 100) if cat_stats["target"] > 0 else 0
                st.metric(cat, f"{cat_stats['collected']}/{cat_stats['target']}")
                st.progress(min(pct / 100, 1.0))


def _render_object_progress(objects: list[RegisteredObject]) -> None:
    """Render per-object progress section."""
    st.subheader("Collection Progress by Object")

    if objects:
        for obj in objects:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                pct = (obj.collected_samples / obj.target_samples) if obj.target_samples > 0 else 0
                if pct >= 1.0:
                    status = "🟢"
                elif pct >= 0.5:
                    status = "🟡"
                else:
                    status = "🔴"

                st.write(f"{status} **{obj.display_name}** ({obj.category})")
                st.progress(min(pct, 1.0))

            with col2:
                st.write(f"{obj.collected_samples}/{obj.target_samples}")

            with col3:
                badges = []
                if obj.properties.is_heavy:
                    badges.append("Heavy")
                if obj.properties.is_tiny:
                    badges.append("Tiny")
                if obj.properties.has_liquid:
                    badges.append("Liquid")
                st.write(", ".join(badges) if badges else "-")
    else:
        st.info("No objects registered yet. Go to Registry to add objects.")


def _render_training_readiness(objects: list[RegisteredObject]) -> None:
    """Render training readiness section."""
    st.subheader("Training Readiness")

    registry = st.session_state.registry

    ready_count = sum(1 for obj in objects if obj.collected_samples >= 50)
    total_count = len(objects)

    if total_count == 0:
        st.warning("No objects registered.")
    elif ready_count == total_count:
        st.success(f"All {total_count} objects have sufficient data for training!")
        if st.button("Export to YOLO Config"):
            output_path = registry.export_to_yolo_config("config/object_classes.json")
            st.success(f"Exported to {output_path}")
    else:
        st.warning(f"{total_count - ready_count} objects need more data (minimum 50 images each)")

        need_data = [obj for obj in objects if obj.collected_samples < 50]
        for obj in need_data[:5]:
            st.write(f"  • {obj.display_name}: {obj.collected_samples}/50 minimum")


# For Streamlit native multipage
if __name__ == "__main__":
    show_dashboard_page()
