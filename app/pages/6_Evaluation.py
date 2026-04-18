"""Evaluation page.

Runs `scripts/evaluation/evaluate_model.py` against a trained model and
displays mAP / inference-time metrics. Robustness / Xtion-live / video
sections were dropped along with the capture and dataset-prep pipelines.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar
from components.progress_display import (
    render_active_task_banner,
    render_task_list,
    render_task_progress,
)
from services.path_coordinator import PathCoordinator
from services.task_manager import TaskManager, TaskStatus


TARGET_MAP50 = 0.85
TARGET_INFERENCE_MS = 100.0


def show_evaluation_page() -> None:
    render_common_sidebar()
    st.title("📈 Model Evaluation")

    task_manager: TaskManager = st.session_state.task_manager
    path_coordinator: PathCoordinator = st.session_state.path_coordinator

    active_task = render_active_task_banner("evaluation", task_manager)
    if active_task:
        st.markdown("---")
        st.subheader("Evaluation in progress")
        task = render_task_progress(active_task.task_id, task_manager)
        if task and task.status == TaskStatus.COMPLETED:
            _render_results(task)
        return

    tab_run, tab_results, tab_visual = st.tabs([
        "Run evaluation", "Results", "Visual test",
    ])

    with tab_run:
        _render_run(task_manager, path_coordinator)
    with tab_results:
        _render_results_list(task_manager)
    with tab_visual:
        _render_visual_test(path_coordinator)


def _render_run(task_manager: TaskManager, path_coordinator: PathCoordinator) -> None:
    st.subheader("Configure evaluation")

    models = path_coordinator.get_trained_models()
    datasets = path_coordinator.get_datasets()

    if not models:
        st.warning("No trained models found. Run training first.")
        return
    if not datasets:
        st.warning("No prepared datasets found. Use the Training page to prepare one.")
        return

    col_m, col_d = st.columns(2)
    with col_m:
        default_idx = 0
        if "selected_model" in st.session_state:
            for i, m in enumerate(models):
                if m.get("best_path") == st.session_state["selected_model"]:
                    default_idx = i
                    break
        selected_model = st.selectbox(
            "Model",
            models,
            index=default_idx,
            format_func=lambda x: f"{x['name']} ({x['created'][:10]})",
        )
        model_path = selected_model["best_path"] or selected_model["last_path"]
        st.caption(f"`{model_path}`")

    with col_d:
        selected_dataset = st.selectbox(
            "Dataset",
            datasets,
            format_func=lambda x: f"{x['name']} ({x['created'][:10]})",
        )
        data_yaml = selected_dataset["data_yaml"]
        st.caption(f"`{data_yaml}`")

    st.markdown("---")
    col_conf, col_targets = st.columns(2)
    with col_conf:
        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.05, max_value=0.9, value=0.25, step=0.05,
        )
    with col_targets:
        st.metric("Target mAP@50", f"{TARGET_MAP50:.0%}")
        st.metric("Target inference", f"<{TARGET_INFERENCE_MS:.0f}ms")

    st.markdown("---")
    if st.button("Start evaluation", type="primary"):
        task_id = task_manager.start_evaluation(
            model_path=model_path,
            dataset_yaml=str(data_yaml),
            conf_threshold=conf_threshold,
        )
        st.success(f"Evaluation started! Task ID: {task_id}")
        st.rerun()


def _render_results(task) -> None:
    if not task.extra_data:
        return

    st.markdown("---")
    st.subheader("Results")

    c1, c2, c3, c4 = st.columns(4)
    map50 = task.extra_data.get("overall_map50", 0)
    map50_95 = task.extra_data.get("overall_map50_95", 0)
    precision = task.extra_data.get("overall_precision", 0)
    recall = task.extra_data.get("overall_recall", 0)
    delta = map50 - TARGET_MAP50
    c1.metric("mAP@50", f"{map50:.2%}", delta=f"{delta:+.1%}",
              delta_color="normal" if map50 >= TARGET_MAP50 else "inverse")
    c2.metric("mAP@50-95", f"{map50_95:.2%}")
    c3.metric("Precision", f"{precision:.2%}")
    c4.metric("Recall", f"{recall:.2%}")

    inference_ms = task.extra_data.get("inference_time_ms", 0)
    inference_std = task.extra_data.get("inference_time_std", 0)
    margin = TARGET_INFERENCE_MS - inference_ms
    c5, c6 = st.columns(2)
    c5.metric("Inference (ms)", f"{inference_ms:.1f}", delta=f"{margin:+.1f} margin",
              delta_color="normal" if inference_ms <= TARGET_INFERENCE_MS else "inverse")
    c6.metric("Inference std", f"±{inference_std:.1f}ms")

    meets = task.extra_data.get("meets_requirements", False)
    if meets:
        st.success("Meets competition requirements.")
    else:
        st.error("Does NOT meet competition requirements.")
        for issue in task.extra_data.get("issues", []):
            st.write(f"- {issue}")

    per_class = task.extra_data.get("per_class_metrics", {})
    if per_class:
        st.markdown("---")
        st.subheader("Per-class metrics")
        import pandas as pd
        rows = [
            {
                "Class": name,
                "AP@50": f"{m.get('ap50', 0):.3f}",
                "AP@50-95": f"{m.get('ap50_95', 0):.3f}",
                "Precision": f"{m.get('precision', 0):.3f}",
                "Recall": f"{m.get('recall', 0):.3f}",
            }
            for name, m in per_class.items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def _render_results_list(task_manager: TaskManager) -> None:
    st.subheader("Evaluation history")
    tasks = task_manager.get_recent_tasks(limit=10, task_type="evaluation")
    completed = [t for t in tasks if t.status == TaskStatus.COMPLETED]
    if not completed:
        st.info("No completed evaluations yet.")
        render_task_list(task_type="evaluation", task_manager=task_manager, limit=5)
        return
    for task in completed:
        meets = task.extra_data.get("meets_requirements", False)
        icon = "✅" if meets else "❌"
        with st.expander(f"{icon} {task.task_id}", expanded=False):
            _render_results(task)


def _render_visual_test(path_coordinator: PathCoordinator) -> None:
    st.subheader("Visual prediction test")
    models = path_coordinator.get_trained_models()
    if not models:
        st.warning("No trained models.")
        return

    selected_model = st.selectbox(
        "Model", models, format_func=lambda x: x["name"], key="visual_model",
    )
    if not selected_model:
        return
    model_path = selected_model["best_path"] or selected_model["last_path"]

    source = st.radio("Image source", ["Upload", "From dataset"], horizontal=True)
    test_image: str | None = None

    if source == "Upload":
        uploaded = st.file_uploader("Test image", type=["jpg", "jpeg", "png"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(uploaded.read())
                test_image = f.name
    else:
        datasets = path_coordinator.get_datasets()
        if not datasets:
            st.info("No prepared datasets.")
            return
        selected_ds = st.selectbox(
            "Dataset", datasets, format_func=lambda x: x["name"], key="visual_ds"
        )
        val_dir = Path(selected_ds["path"]) / "images" / "val"
        if val_dir.exists():
            candidates = sorted(val_dir.glob("*.png")) + sorted(val_dir.glob("*.jpg"))
            if candidates:
                selected_img = st.selectbox(
                    "Image", candidates[:50], format_func=lambda x: x.name
                )
                if selected_img:
                    test_image = str(selected_img)

    if not test_image:
        return

    conf = st.slider("Confidence", 0.05, 0.9, 0.25, step=0.05, key="visual_conf")
    if st.button("Run prediction", type="primary"):
        with st.spinner("Running prediction..."):
            try:
                from ultralytics import YOLO
                model = YOLO(model_path)
                results = model(test_image, conf=conf, verbose=False)
                result = results[0]
                c1, c2 = st.columns(2)
                c1.image(test_image, caption="Input")
                c2.image(result.plot(), channels="BGR", caption="Prediction")
                st.markdown("---")
                if len(result.boxes) == 0:
                    st.info("No detections.")
                else:
                    for box in result.boxes:
                        cid = int(box.cls.item())
                        cname = model.names[cid]
                        score = box.conf.item()
                        bbox = box.xyxy[0].tolist()
                        st.write(
                            f"- **{cname}**: {score:.2%}  "
                            f"bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
                        )
            except Exception as e:
                st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    show_evaluation_page()
