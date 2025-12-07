"""
Progress Display Component

Reusable UI component for displaying long-running task progress in Streamlit.
"""

import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.task_manager import TaskManager, TaskInfo, TaskStatus


def render_task_progress(
    task_id: str,
    task_manager: Optional[TaskManager] = None,
    show_cancel_button: bool = True,
    show_details: bool = True,
    auto_refresh: bool = True,
    refresh_interval: int = 2,
) -> Optional[TaskInfo]:
    """
    Render progress display for a task.

    Args:
        task_id: Task ID to display
        task_manager: TaskManager instance (creates new one if None)
        show_cancel_button: Show cancel button for running tasks
        show_details: Show detailed extra_data
        auto_refresh: Auto-refresh page when task is running
        refresh_interval: Refresh interval in seconds

    Returns:
        TaskInfo if found, None otherwise
    """
    if task_manager is None:
        task_manager = TaskManager()

    task = task_manager.get_task(task_id)

    if task is None:
        st.error(f"Task not found: {task_id}")
        return None

    # Status indicator
    status_colors = {
        TaskStatus.PENDING: "orange",
        TaskStatus.RUNNING: "blue",
        TaskStatus.COMPLETED: "green",
        TaskStatus.FAILED: "red",
        TaskStatus.CANCELLED: "gray",
    }

    status_icons = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.RUNNING: "🔄",
        TaskStatus.COMPLETED: "✅",
        TaskStatus.FAILED: "❌",
        TaskStatus.CANCELLED: "⏹️",
    }

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"**Task:** `{task.task_id}`")

    with col2:
        st.markdown(f"**Type:** {task.task_type}")

    with col3:
        icon = status_icons.get(task.status, "❓")
        st.markdown(f"**Status:** {icon} {task.status.value}")

    # Progress bar
    st.progress(task.progress, text=task.current_step)

    # Time info
    col1, col2 = st.columns(2)

    with col1:
        if task.started_at:
            st.write(f"**Started:** {task.started_at[:19]}")

    with col2:
        st.write(f"**Elapsed:** {task.elapsed_time_str}")

    # Cancel button
    if show_cancel_button and task.status == TaskStatus.RUNNING:
        if st.button("Cancel Task", key=f"cancel_{task_id}", type="secondary"):
            if task_manager.cancel_task(task_id):
                st.warning("Task cancelled")
                st.rerun()
            else:
                st.error("Failed to cancel task")

    # Error message
    if task.error_message:
        st.error(f"**Error:** {task.error_message}")

    # Result path
    if task.result_path:
        st.success(f"**Result:** `{task.result_path}`")

    # Extra data details
    if show_details and task.extra_data:
        with st.expander("Task Details", expanded=False):
            _render_extra_data(task.extra_data)

    # Auto-refresh
    if auto_refresh and task.status == TaskStatus.RUNNING:
        import time
        time.sleep(refresh_interval)
        st.rerun()

    return task


def render_task_list(
    task_type: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    limit: int = 10,
    show_active_only: bool = False,
) -> None:
    """
    Render list of tasks.

    Args:
        task_type: Filter by task type (annotation, training, evaluation)
        task_manager: TaskManager instance
        limit: Maximum number of tasks to show
        show_active_only: Only show active (pending/running) tasks
    """
    if task_manager is None:
        task_manager = TaskManager()

    if show_active_only:
        tasks = task_manager.get_active_tasks(task_type)
    else:
        tasks = task_manager.get_recent_tasks(limit, task_type)

    if not tasks:
        st.info("No tasks found")
        return

    status_icons = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.RUNNING: "🔄",
        TaskStatus.COMPLETED: "✅",
        TaskStatus.FAILED: "❌",
        TaskStatus.CANCELLED: "⏹️",
    }

    for task in tasks:
        icon = status_icons.get(task.status, "❓")

        with st.expander(
            f"{icon} {task.task_id} - {task.status.value}",
            expanded=task.is_active
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.write(f"**Type:** {task.task_type}")

            with col2:
                st.write(f"**Progress:** {task.progress * 100:.0f}%")

            with col3:
                st.write(f"**Time:** {task.elapsed_time_str}")

            st.progress(task.progress, text=task.current_step)

            if task.error_message:
                st.error(task.error_message)

            if task.result_path:
                st.success(f"Result: `{task.result_path}`")

            # Actions
            col1, col2 = st.columns(2)

            with col1:
                if task.status == TaskStatus.RUNNING:
                    if st.button("Cancel", key=f"cancel_{task.task_id}"):
                        task_manager.cancel_task(task.task_id)
                        st.rerun()

            with col2:
                if task.is_finished:
                    if st.button("Delete", key=f"delete_{task.task_id}"):
                        task_manager.delete_task(task.task_id)
                        st.rerun()


def render_active_task_banner(
    task_type: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
) -> Optional[TaskInfo]:
    """
    Render a banner showing active task if one exists.

    Useful for showing at the top of a page to indicate ongoing work.

    Args:
        task_type: Filter by task type
        task_manager: TaskManager instance

    Returns:
        Active TaskInfo if found, None otherwise
    """
    if task_manager is None:
        task_manager = TaskManager()

    active_tasks = task_manager.get_active_tasks(task_type)

    if not active_tasks:
        return None

    task = active_tasks[0]  # Show most recent active task

    status_icons = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.RUNNING: "🔄",
    }
    icon = status_icons.get(task.status, "❓")

    with st.container():
        st.info(f"{icon} **Active Task:** {task.task_id} - {task.current_step}")
        st.progress(task.progress)

        col1, col2 = st.columns([3, 1])

        with col2:
            if st.button("View Details", key=f"view_{task.task_id}"):
                st.session_state[f"show_task_{task.task_id}"] = True

    return task


def _render_extra_data(data: Dict[str, Any], prefix: str = "") -> None:
    """Recursively render extra data dictionary."""
    for key, value in data.items():
        display_key = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            st.write(f"**{display_key}:**")
            _render_extra_data(value, prefix="  ")
        elif isinstance(value, list):
            st.write(f"**{display_key}:** {len(value)} items")
            if len(value) <= 5:
                for i, item in enumerate(value):
                    st.write(f"  - {item}")
        elif isinstance(value, float):
            st.write(f"**{display_key}:** {value:.4f}")
        else:
            st.write(f"**{display_key}:** {value}")


def render_task_metrics(task: TaskInfo) -> None:
    """
    Render metrics from task extra_data.

    Designed for completed tasks with metrics in extra_data.
    """
    if not task.extra_data:
        return

    metrics = task.extra_data.get("metrics", {})

    if not metrics:
        # Try top-level keys
        metric_keys = ["mAP50", "overall_map50", "success_rate", "inference_time_ms"]
        metrics = {k: task.extra_data[k] for k in metric_keys if k in task.extra_data}

    if not metrics:
        return

    st.subheader("Results")

    # Create columns based on number of metrics
    cols = st.columns(min(len(metrics), 4))

    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % len(cols)]:
            if isinstance(value, float):
                if "map" in key.lower() or "rate" in key.lower():
                    st.metric(key, f"{value:.2%}")
                elif "time" in key.lower():
                    st.metric(key, f"{value:.1f}ms")
                else:
                    st.metric(key, f"{value:.4f}")
            else:
                st.metric(key, str(value))


# =============================================================================
# Enhanced Training Progress Components (Mission Control Style)
# =============================================================================

def render_circular_progress(progress: float, size: int = 140, label: str = "Progress"):
    """
    Render progress indicator using Streamlit components.

    Args:
        progress: Progress value between 0.0 and 1.0
        size: Size of the circle in pixels (unused, kept for compatibility)
        label: Label text below the progress
    """
    percentage = int(progress * 100)
    st.metric(label, f"{percentage}%")
    st.progress(progress)


def render_training_metric_cards(task: TaskInfo):
    """
    Render real-time metric cards during training.

    Shows: Epoch, mAP@50, Loss, ETA

    Args:
        task: TaskInfo with extra_data containing training metrics
    """
    metrics = task.extra_data.get("metrics", {})
    config = task.extra_data.get("config", {})
    current_epoch = task.extra_data.get("epoch", 0)
    total_epochs = config.get("epochs", 50)

    # Extract values
    map50 = metrics.get("mAP50", 0)
    loss = metrics.get("loss", 0)
    target_map50 = 0.85

    # Calculate ETA
    elapsed = task.elapsed_time or 0
    if task.progress > 0.1:
        eta_seconds = (elapsed / task.progress) * (1 - task.progress)
        eta_minutes = eta_seconds / 60
        eta_str = f"{eta_minutes:.0f}m"
    else:
        eta_str = "--"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.metric("EPOCH", f"{current_epoch}/{total_epochs}")

    with col2:
        with st.container(border=True):
            st.metric("MAP@50", f"{map50:.1%}", help=f"Target: {target_map50:.0%}")
            st.caption(f"Target: {target_map50:.0%}")

    with col3:
        with st.container(border=True):
            st.metric("LOSS", f"{loss:.4f}")

    with col4:
        with st.container(border=True):
            st.metric("ETA", eta_str)


def render_training_progress_bar(progress: float, current_step: str, animated: bool = True):
    """
    Render training progress bar using Streamlit components.

    Args:
        progress: Progress value between 0.0 and 1.0
        current_step: Current step description
        animated: Whether to show shimmer animation (unused)
    """
    width_percent = progress * 100
    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(current_step)
    with col2:
        st.caption(f"{width_percent:.0f}%")
    st.progress(progress)


def render_training_active_banner(task: TaskInfo, task_manager: TaskManager):
    """
    Render enhanced active training task banner.

    Features:
    - Circular progress indicator
    - Metric cards
    - Progress bar with step info
    - TensorBoard link (if available)

    Args:
        task: Active TaskInfo
        task_manager: TaskManager instance
    """
    # Header with task info
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{task.task_id}** · :green[● RUNNING]")
        with col2:
            st.caption(f"Elapsed: {task.elapsed_time_str}")

    # Metric cards
    render_training_metric_cards(task)

    # Progress bar
    render_training_progress_bar(task.progress, task.current_step)

    # TensorBoard status
    tensorboard_url = task.extra_data.get("tensorboard_url")
    if tensorboard_url:
        from .tensorboard_embed import render_tensorboard_status
        render_tensorboard_status(tensorboard_url, is_running=True)

    # Cancel button
    if st.button("Cancel Training", key=f"cancel_training_{task.task_id}", type="secondary"):
        if task_manager.cancel_task(task.task_id):
            st.warning("Training cancelled")
            st.rerun()


def render_training_completed_banner(task: TaskInfo):
    """
    Render completion banner for finished training.

    Args:
        task: Completed TaskInfo
    """
    metrics = task.extra_data.get("metrics", {})
    training_time = task.extra_data.get("training_time_minutes", 0)
    epochs_completed = task.extra_data.get("epochs_completed", 0)

    map50 = metrics.get("mAP50", 0)
    map50_95 = metrics.get("mAP50-95", 0)
    target_met = map50 >= 0.85

    status_text = "Target Achieved" if target_met else "Below Target"

    # Status banner
    if target_met:
        st.success(f"✓ Training Complete - {status_text}")
    else:
        st.warning(f"⚠ Training Complete - {status_text}")

    st.caption(f"{epochs_completed} epochs in {training_time:.1f} minutes")

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("mAP@50", f"{map50:.1%}")
    with col2:
        st.metric("mAP@50-95", f"{map50_95:.1%}")
    with col3:
        st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
    with col4:
        st.metric("Recall", f"{metrics.get('recall', 0):.1%}")

    # Show best model path
    best_model = task.extra_data.get("best_model")
    if best_model:
        with st.container(border=True):
            st.caption("Best Model")
            st.code(best_model, language=None)
