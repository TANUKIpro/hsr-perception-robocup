"""
Training Page

Provides UI for running YOLOv8 fine-tuning on annotated datasets.
Integrates with scripts/training/quick_finetune.py via TaskManager.
"""

import streamlit as st
from pathlib import Path
import sys

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from services.task_manager import TaskManager, TaskStatus
from services.path_coordinator import PathCoordinator
from components.progress_display import (
    render_task_progress,
    render_active_task_banner,
    render_task_list,
    render_task_metrics,
)


def show_training_page():
    """Main training page."""
    st.title("Model Training")

    # Get services from session state (profile-aware)
    if "task_manager" not in st.session_state or "path_coordinator" not in st.session_state:
        st.error("Services not initialized. Please reload the page.")
        return

    task_manager = st.session_state.task_manager
    path_coordinator = st.session_state.path_coordinator

    # Check for active training task
    active_task = render_active_task_banner("training", task_manager)

    if active_task:
        st.markdown("---")
        st.subheader("Training in Progress")
        task = render_task_progress(active_task.task_id, task_manager)

        if task and task.status == TaskStatus.COMPLETED:
            st.balloons()
            render_task_metrics(task)

        return

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Start Training", "Models", "History"])

    with tab1:
        _render_start_training(task_manager, path_coordinator)

    with tab2:
        _render_trained_models(path_coordinator)

    with tab3:
        _render_training_history(task_manager)


def _render_start_training(task_manager: TaskManager, path_coordinator: PathCoordinator):
    """Render training configuration and start section."""
    st.subheader("Configure Training")

    # Dataset selection
    st.markdown("### Dataset")

    sessions = path_coordinator.get_annotation_sessions()
    ready_sessions = [s for s in sessions if s["has_data_yaml"]]

    if not ready_sessions:
        st.warning(
            "No annotated datasets found. "
            "Please run annotation first to create a dataset."
        )
        return

    # Check for pre-selected dataset
    default_idx = 0
    if "selected_dataset" in st.session_state:
        for i, s in enumerate(ready_sessions):
            if s["path"] == st.session_state["selected_dataset"]:
                default_idx = i
                break

    selected_session = st.selectbox(
        "Select Dataset",
        ready_sessions,
        index=default_idx,
        format_func=lambda x: f"{x['name']} ({x['created'][:10]})"
    )

    if selected_session:
        session_path = Path(selected_session["path"])
        data_yaml = session_path / "data.yaml"

        # Show dataset info
        try:
            import yaml
            with open(data_yaml) as f:
                config = yaml.safe_load(f)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Classes", len(config.get("names", [])))

            # Count images
            train_dir = session_path / "images" / "train"
            val_dir = session_path / "images" / "val"
            train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
            val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0

            with col2:
                st.metric("Train Images", train_count)

            with col3:
                st.metric("Val Images", val_count)

            # Class names
            with st.expander("View Classes", expanded=False):
                for i, name in enumerate(config.get("names", [])):
                    st.write(f"  {i}: {name}")

        except Exception as e:
            st.error(f"Error reading dataset config: {e}")
            return

    st.markdown("---")

    # Model configuration
    st.markdown("### Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        # Base model selection
        pretrained_models = path_coordinator.get_pretrained_models()
        model_options = ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt"] + [
            str(Path(m).name) for m in pretrained_models if m not in ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt"]
        ]

        base_model = st.selectbox(
            "Base Model",
            model_options,
            index=0,
            help="YOLOv8m is recommended for competition (best accuracy). "
                 "YOLOv8s is faster but less accurate."
        )

        epochs = st.slider(
            "Epochs",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Number of training epochs. More epochs = better accuracy but longer time."
        )

    with col2:
        batch_size = st.slider(
            "Batch Size",
            min_value=4,
            max_value=64,
            value=16,
            step=4,
            help="Larger batch size = faster training but more GPU memory required."
        )

        fast_mode = st.checkbox(
            "Fast Mode (Testing)",
            value=False,
            help="Use smaller model and fewer epochs for quick testing."
        )

    # GPU check
    st.markdown("---")
    st.markdown("### Hardware")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            st.success(f"GPU Available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            st.warning(
                "CUDA not available. Training will be very slow on CPU. "
                "Consider using a machine with GPU."
            )
    except ImportError:
        st.warning("PyTorch not installed. Cannot check GPU availability.")

    # Estimated time
    st.markdown("---")

    # Rough estimate: ~1 minute per epoch for YOLOv8m on GPU
    estimated_minutes = epochs * (2 if base_model == "yolov8m.pt" else 1)
    if fast_mode:
        estimated_minutes = epochs * 0.5

    st.info(
        f"**Estimated training time:** ~{estimated_minutes:.0f} minutes\n\n"
        f"Target metrics: mAP@50 ≥ 85%, Inference ≤ 100ms"
    )

    # Start button
    if st.button("Start Training", type="primary"):
        task_id = task_manager.start_training(
            dataset_yaml=str(data_yaml),
            base_model=base_model,
            output_dir=str(path_coordinator.get_path("finetuned_dir")),
            epochs=epochs,
            batch_size=batch_size,
            fast_mode=fast_mode,
        )

        st.success(f"Training started! Task ID: {task_id}")
        st.rerun()


def _render_trained_models(path_coordinator: PathCoordinator):
    """Render list of trained models."""
    st.subheader("Trained Models")

    models = path_coordinator.get_trained_models()

    if not models:
        st.info("No trained models found. Run training to create one.")
        return

    for model in models:
        with st.expander(f"📦 {model['name']}", expanded=False):
            st.write(f"**Created:** {model['created'][:19]}")

            if model["best_path"]:
                st.write(f"**Best Model:** `{model['best_path']}`")

            if model["last_path"]:
                st.write(f"**Last Model:** `{model['last_path']}`")

            # Check for training result
            model_dir = Path(model["best_path"]).parent.parent if model["best_path"] else None
            if model_dir:
                result_file = model_dir / "training_result.json"
                if result_file.exists():
                    try:
                        import json
                        with open(result_file) as f:
                            result = json.load(f)

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("mAP@50", f"{result.get('metrics', {}).get('mAP50', 0):.2%}")

                        with col2:
                            st.metric("mAP@50-95", f"{result.get('metrics', {}).get('mAP50-95', 0):.2%}")

                        with col3:
                            st.metric("Training Time", f"{result.get('training_time_minutes', 0):.0f}min")

                        with col4:
                            st.metric("Epochs", result.get("epochs_completed", "N/A"))

                    except Exception:
                        pass

            # Action buttons
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Evaluate", key=f"eval_{model['name']}"):
                    st.session_state["selected_model"] = model["best_path"] or model["last_path"]
                    st.info("Go to Evaluation page to evaluate this model")

            with col2:
                if model["best_path"]:
                    # Copy path button
                    st.code(model["best_path"], language=None)


def _render_training_history(task_manager: TaskManager):
    """Render training task history."""
    st.subheader("Training History")

    render_task_list(
        task_type="training",
        task_manager=task_manager,
        limit=10,
        show_active_only=False,
    )


# For Streamlit native multipage
if __name__ == "__main__":
    show_training_page()
