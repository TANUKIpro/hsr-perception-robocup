"""
Evaluation Page

Provides UI for evaluating trained models and verifying competition requirements.
Integrates with scripts/evaluation/evaluate_model.py via TaskManager.
"""

import streamlit as st
import subprocess
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
from components.robustness_test import (
    render_realtime_preview,
    render_batch_test,
    render_similar_object_test,
)


# Competition targets
TARGET_MAP50 = 0.85
TARGET_INFERENCE_MS = 100.0


def show_evaluation_page():
    """Main evaluation page."""
    # Render common sidebar
    from components.common_sidebar import render_common_sidebar
    render_common_sidebar()

    st.title("Model Evaluation")

    # Get services from session state (profile-aware)
    if "task_manager" not in st.session_state or "path_coordinator" not in st.session_state:
        st.error("Services not initialized. Please reload the page.")
        return

    task_manager = st.session_state.task_manager
    path_coordinator = st.session_state.path_coordinator

    # Check for active evaluation task
    active_task = render_active_task_banner("evaluation", task_manager)

    if active_task:
        st.markdown("---")
        st.subheader("Evaluation in Progress")
        task = render_task_progress(active_task.task_id, task_manager)

        if task and task.status == TaskStatus.COMPLETED:
            _render_evaluation_results(task)

        return

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Run Evaluation", "Results", "Visual Test", "Robustness Test", "Xtion Live Test"
    ])

    with tab1:
        _render_run_evaluation(task_manager, path_coordinator)

    with tab2:
        _render_evaluation_results_list(task_manager)

    with tab3:
        _render_visual_test(path_coordinator)

    with tab4:
        _render_robustness_test(path_coordinator)

    with tab5:
        _render_xtion_live_test(path_coordinator)


def _render_run_evaluation(task_manager: TaskManager, path_coordinator: PathCoordinator):
    """Render evaluation configuration and run section."""
    st.subheader("Configure Evaluation")

    # Get available models and datasets
    models = path_coordinator.get_trained_models()
    sessions = path_coordinator.get_annotation_sessions()
    ready_sessions = [s for s in sessions if s["has_data_yaml"]]

    # Check if data is available
    if not models:
        st.warning(
            "No trained models found. "
            "Please run training first to create a model."
        )
        return

    if not ready_sessions:
        st.warning("No annotated datasets found.")
        return

    # Row 1: Model and Dataset selection (2 columns)
    col_model, col_dataset = st.columns(2)

    with col_model:
        st.markdown("### Model")

        # Check for pre-selected model
        default_idx = 0
        if "selected_model" in st.session_state:
            for i, m in enumerate(models):
                if m["best_path"] == st.session_state["selected_model"]:
                    default_idx = i
                    break

        selected_model = st.selectbox(
            "Select Model",
            models,
            index=default_idx,
            format_func=lambda x: f"{x['name']} ({x['created'][:10]})"
        )

        if selected_model:
            model_path = selected_model["best_path"] or selected_model["last_path"]
            st.caption(f"`{model_path}`")

    with col_dataset:
        st.markdown("### Dataset")

        selected_session = st.selectbox(
            "Select Dataset",
            ready_sessions,
            format_func=lambda x: f"{x['name']} ({x['created'][:10]})"
        )

        if selected_session:
            data_yaml = Path(selected_session["path"]) / "data.yaml"
            st.caption(f"`{data_yaml}`")

    st.markdown("---")

    # Row 2: Options and Competition Requirements (2 columns)
    col_options, col_requirements = st.columns(2)

    with col_options:
        st.markdown("### Options")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )

    with col_requirements:
        st.markdown("### Competition Requirements")
        req_col1, req_col2 = st.columns(2)
        with req_col1:
            st.metric("Target mAP@50", f"{TARGET_MAP50:.0%}")
        with req_col2:
            st.metric("Target Inference", f"<{TARGET_INFERENCE_MS:.0f}ms")

    # Run button
    st.markdown("---")

    if st.button("Start Evaluation", type="primary"):
        task_id = task_manager.start_evaluation(
            model_path=model_path,
            dataset_yaml=str(data_yaml),
            conf_threshold=conf_threshold,
        )

        st.success(f"Evaluation started! Task ID: {task_id}")
        st.rerun()


def _render_evaluation_results(task: "TaskInfo"):
    """Render results from a completed evaluation task."""
    if not task.extra_data:
        return

    st.markdown("---")
    st.subheader("Evaluation Results")

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        map50 = task.extra_data.get("overall_map50", 0)
        delta = map50 - TARGET_MAP50
        st.metric(
            "mAP@50",
            f"{map50:.2%}",
            delta=f"{delta:+.1%}",
            delta_color="normal" if map50 >= TARGET_MAP50 else "inverse"
        )

    with col2:
        map50_95 = task.extra_data.get("overall_map50_95", 0)
        st.metric("mAP@50-95", f"{map50_95:.2%}")

    with col3:
        precision = task.extra_data.get("overall_precision", 0)
        st.metric("Precision", f"{precision:.2%}")

    with col4:
        recall = task.extra_data.get("overall_recall", 0)
        st.metric("Recall", f"{recall:.2%}")

    st.markdown("---")

    # Inference time
    col1, col2 = st.columns(2)

    with col1:
        inference_ms = task.extra_data.get("inference_time_ms", 0)
        delta = TARGET_INFERENCE_MS - inference_ms
        st.metric(
            "Inference Time",
            f"{inference_ms:.1f}ms",
            delta=f"{delta:+.1f}ms margin",
            delta_color="normal" if inference_ms <= TARGET_INFERENCE_MS else "inverse"
        )

    with col2:
        inference_std = task.extra_data.get("inference_time_std", 0)
        st.metric("Inference Std", f"±{inference_std:.1f}ms")

    # Requirements check
    st.markdown("---")

    meets_requirements = task.extra_data.get("meets_requirements", False)
    issues = task.extra_data.get("issues", [])

    if meets_requirements:
        st.success("✅ **Model meets all competition requirements!**")
    else:
        st.error("❌ **Model does NOT meet competition requirements**")
        for issue in issues:
            st.write(f"  • {issue}")

    # Per-class metrics
    per_class = task.extra_data.get("per_class_metrics", {})
    if per_class:
        st.markdown("---")
        st.subheader("Per-Class Metrics")

        # Create dataframe for display
        import pandas as pd

        data = []
        for name, metrics in per_class.items():
            data.append({
                "Class": name,
                "AP@50": f"{metrics.get('ap50', 0):.3f}",
                "AP@50-95": f"{metrics.get('ap50_95', 0):.3f}",
                "Precision": f"{metrics.get('precision', 0):.3f}",
                "Recall": f"{metrics.get('recall', 0):.3f}",
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)


def _render_evaluation_results_list(task_manager: TaskManager):
    """Render list of evaluation results."""
    st.subheader("Evaluation History")

    tasks = task_manager.get_recent_tasks(limit=10, task_type="evaluation")
    completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]

    if not completed_tasks:
        st.info("No completed evaluations found.")
        render_task_list(
            task_type="evaluation",
            task_manager=task_manager,
            limit=5,
        )
        return

    for task in completed_tasks:
        meets = task.extra_data.get("meets_requirements", False)
        icon = "✅" if meets else "❌"

        with st.expander(f"{icon} {task.task_id}", expanded=False):
            _render_evaluation_results(task)


def _render_visual_test(path_coordinator: PathCoordinator):
    """Render visual prediction test."""
    st.subheader("Visual Prediction Test")

    # Model selection
    models = path_coordinator.get_trained_models()

    if not models:
        st.warning("No trained models found.")
        return

    selected_model = st.selectbox(
        "Select Model",
        models,
        format_func=lambda x: x['name'],
        key="visual_model"
    )

    if not selected_model:
        return

    model_path = selected_model["best_path"] or selected_model["last_path"]

    # Image source
    st.markdown("---")

    image_source = st.radio(
        "Image Source",
        ["Upload Image", "Select from Dataset"],
        horizontal=True
    )

    test_image = None

    if image_source == "Upload Image":
        uploaded = st.file_uploader(
            "Upload test image",
            type=["jpg", "jpeg", "png"],
            key="test_upload"
        )

        if uploaded:
            # Save temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(uploaded.read())
                test_image = f.name

    else:  # Select from Dataset
        sessions = path_coordinator.get_annotation_sessions()
        ready_sessions = [s for s in sessions if s["has_data_yaml"]]

        if ready_sessions:
            selected_session = st.selectbox(
                "Select Dataset",
                ready_sessions,
                format_func=lambda x: x['name'],
                key="visual_dataset"
            )

            if selected_session:
                val_dir = Path(selected_session["path"]) / "images" / "val"
                if val_dir.exists():
                    images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
                    if images:
                        selected_image = st.selectbox(
                            "Select Image",
                            images[:50],  # Limit to first 50
                            format_func=lambda x: x.name
                        )
                        if selected_image:
                            test_image = str(selected_image)

    # Run prediction
    if test_image:
        st.markdown("---")

        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            key="visual_conf"
        )

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Running prediction..."):
                try:
                    from ultralytics import YOLO

                    model = YOLO(model_path)
                    results = model(test_image, conf=conf_threshold, verbose=False)
                    result = results[0]

                    # Display results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Original")
                        st.image(test_image)

                    with col2:
                        st.subheader("Prediction")
                        annotated = result.plot()
                        st.image(annotated, channels="BGR")

                    # Detection list
                    st.markdown("---")
                    st.subheader("Detections")

                    if len(result.boxes) == 0:
                        st.info("No objects detected")
                    else:
                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            class_name = model.names[class_id]
                            conf = box.conf.item()
                            bbox = box.xyxy[0].tolist()

                            st.write(
                                f"• **{class_name}**: {conf:.2%} "
                                f"(bbox: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}])"
                            )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")


def _render_robustness_test(path_coordinator: PathCoordinator):
    """Render robustness test interface."""
    st.subheader("Robustness Test")
    st.markdown("""
    Test model robustness against real-world conditions:
    - **Brightness**: Lighting variation (dark/bright environments)
    - **Shadow**: Shadows cast on objects
    - **Occlusion**: Partially hidden objects
    - **Hue Rotation**: Similar-looking objects (color variants)
    """)

    # Get available models and datasets
    models = path_coordinator.get_trained_models()
    sessions = path_coordinator.get_annotation_sessions()
    ready_sessions = [s for s in sessions if s["has_data_yaml"]]

    if not models:
        st.warning("No trained models found. Please run training first.")
        return

    if not ready_sessions:
        st.warning("No annotated datasets found.")
        return

    # Row 1: Model, Dataset, and Confidence (3 columns)
    col_model, col_dataset, col_conf = st.columns(3)

    with col_model:
        selected_model = st.selectbox(
            "Model",
            models,
            format_func=lambda x: x['name'],
            key="robustness_model"
        )

    with col_dataset:
        selected_session = st.selectbox(
            "Dataset",
            ready_sessions,
            format_func=lambda x: x['name'],
            key="robustness_dataset"
        )

    with col_conf:
        conf_threshold = st.slider(
            "Confidence",
            min_value=0.1,
            max_value=0.9,
            value=0.25,
            step=0.05,
            key="robustness_conf"
        )

    if not selected_model or not selected_session:
        return

    model_path = selected_model["best_path"] or selected_model["last_path"]
    dataset_path = Path(selected_session["path"])

    st.markdown("---")

    # Row 2: Image selection (full width)
    test_image = None
    test_image_path = None

    image_source = st.radio(
        "Image Source",
        ["Select from Dataset", "Upload Image"],
        horizontal=True,
        key="robustness_image_source"
    )

    if image_source == "Upload Image":
        uploaded = st.file_uploader(
            "Upload test image",
            type=["jpg", "jpeg", "png"],
            key="robustness_upload"
        )

        if uploaded:
            import cv2
            import numpy as np

            # Read uploaded image
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            test_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    else:  # Select from Dataset
        val_dir = dataset_path / "images" / "val"
        if val_dir.exists():
            images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
            if images:
                selected_image = st.selectbox(
                    "Select Image",
                    images[:50],
                    format_func=lambda x: x.name,
                    key="robustness_image"
                )
                if selected_image:
                    import cv2
                    test_image = cv2.imread(str(selected_image))
                    test_image_path = selected_image

    if test_image is None:
        st.info("Please select or upload an image to test.")
        return

    # Load model
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    st.markdown("---")

    # Test mode selection
    test_mode = st.radio(
        "Test Mode",
        ["Real-time Preview", "Batch Test", "Similar Object Test"],
        horizontal=True,
        key="robustness_mode"
    )

    if test_mode == "Real-time Preview":
        render_realtime_preview(model, test_image, conf_threshold)

    elif test_mode == "Batch Test":
        render_batch_test(model, test_image, conf_threshold)

    else:  # Similar Object Test
        render_similar_object_test(model, dataset_path, conf_threshold)


def _render_xtion_live_test(path_coordinator: PathCoordinator):
    """Render Xtion live test section."""
    st.subheader("Xtion Live Test")
    st.write("Test trained models in real-time using the Xtion camera.")

    # Model selection
    st.markdown("### Model Selection")
    models = path_coordinator.get_trained_models()

    if not models:
        st.warning("No trained models found. Please train a model first.")
        return

    model_options = {
        f"{m['name']} ({m['created'][:10]})": m['best_path'] or m['last_path']
        for m in models
    }
    selected_label = st.selectbox(
        "Select Model",
        list(model_options.keys()),
        key="xtion_model_select"
    )
    selected_model = model_options[selected_label]

    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        key="xtion_conf_threshold"
    )

    st.markdown("---")

    # Launch button
    st.markdown("### Launch Application")
    st.info("Click the button to launch the Xtion test app in a new window.")

    if st.button("Launch Xtion Test App", type="primary", key="launch_xtion_app"):
        # Get project root
        project_root = path_coordinator.project_root

        # Path to the xtion test app
        app_path = project_root / "scripts" / "evaluation" / "xtion_test_app.py"

        if not app_path.exists():
            st.error(f"Application not found: {app_path}")
            return

        try:
            # Launch the tkinter app as a subprocess
            subprocess.Popen(
                [
                    sys.executable,
                    str(app_path),
                    "--model", selected_model,
                    "--conf", str(conf_threshold)
                ],
                cwd=str(project_root)
            )
            st.success("Xtion Test App launched! Check for a new window.")

        except Exception as e:
            st.error(f"Failed to launch app: {e}")


# For Streamlit native multipage
if __name__ == "__main__":
    show_evaluation_page()
