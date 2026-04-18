"""Training page.

Runs YOLOv8 fine-tuning against a prepared dataset (output of
`scripts/data/prepare_dataset.py`). The synthetic / Copy-Paste and
collection flows were removed along with the rest of the annotation
pipeline.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

app_dir = Path(__file__).parent.parent
project_root = app_dir.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
if str(project_root / "scripts") not in sys.path:
    sys.path.insert(0, str(project_root / "scripts"))

from components.common_sidebar import render_common_sidebar
from components.progress_display import (
    render_task_list,
    render_training_active_banner,
    render_training_completed_banner,
    render_training_metric_cards,
)
from components.tensorboard_embed import (
    render_tensorboard_panel,
    render_tensorboard_status,
)
from components.training_charts import render_training_chart
from components.training_styles import ICONS, inject_training_styles
from services.path_coordinator import (
    DATASETS_DIR,
    PYBULLET_HSR_ANNOTATION_ROOT,
    PathCoordinator,
)
from services.task_manager import TaskManager, TaskStatus


PREPARE_SCRIPT = project_root / "scripts" / "data" / "prepare_dataset.py"

_SCRIPTS_DATA_DIR = project_root / "scripts" / "data"
if str(_SCRIPTS_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DATA_DIR))


def show_training_page() -> None:
    render_common_sidebar()
    inject_training_styles()

    st.title("🚀 Model Training")

    task_manager: TaskManager = st.session_state.task_manager
    path_coordinator: PathCoordinator = st.session_state.path_coordinator

    active = next(iter(task_manager.get_active_tasks(task_type="training")), None)
    if active:
        _render_active(active, task_manager)
        return

    tab_start, tab_models, tab_history = st.tabs([
        f"{ICONS['dataset']} Start training",
        f"{ICONS['model']} Models",
        "📜 History",
    ])

    with tab_start:
        _render_start(task_manager, path_coordinator)
    with tab_models:
        _render_trained(path_coordinator)
    with tab_history:
        render_task_list(
            task_type="training",
            task_manager=task_manager,
            limit=10,
            show_active_only=False,
        )


def _render_active(active, task_manager: TaskManager) -> None:
    task = task_manager.get_task(active.task_id)
    if not task:
        return

    render_training_active_banner(task, task_manager)
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)
    render_training_metric_cards(task)
    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    history = task.extra_data.get("training_history", []) if task.extra_data else []
    render_training_chart(history, target_map50=0.85, height=320)

    tb_url = (task.extra_data or {}).get("tensorboard_url")
    if tb_url:
        with st.expander(f"{ICONS['tensorboard']} TensorBoard", expanded=False):
            render_tensorboard_panel(tensorboard_url=tb_url, show_iframe=False)
    else:
        render_tensorboard_status(is_running=False)

    log_file = (task.extra_data or {}).get("log_file")
    if log_file and Path(log_file).exists():
        with st.expander("Training logs", expanded=False):
            try:
                lines = Path(log_file).read_text().splitlines()
                st.code("\n".join(lines[-100:]) or "(empty)", language="text")
            except Exception as e:
                st.info(f"Failed to load logs: {e}")

    if task.status == TaskStatus.COMPLETED:
        st.balloons()
        render_training_completed_banner(task)

    if task.status == TaskStatus.RUNNING:
        time.sleep(2)
        st.rerun()


def _render_start(task_manager: TaskManager, path_coordinator: PathCoordinator) -> None:
    _render_latest_dump_shortcut(task_manager, path_coordinator)
    st.markdown("---")
    _render_dataset_picker(path_coordinator)
    st.markdown("---")
    base_model, epochs, batch_size, fast_mode, auto_scale = _render_model_knobs(path_coordinator)
    st.markdown("---")
    enable_tb, tb_port = _render_monitoring()
    st.markdown("---")
    advanced_params = _render_advanced(auto_scale)
    st.markdown("---")
    _render_launch(
        task_manager=task_manager,
        path_coordinator=path_coordinator,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch_size,
        fast_mode=fast_mode,
        auto_scale=auto_scale,
        enable_tensorboard=enable_tb,
        tensorboard_port=tb_port,
        advanced_params=advanced_params,
    )


def _render_latest_dump_shortcut(
    task_manager: TaskManager, path_coordinator: PathCoordinator
) -> None:
    """Show a one-click 'Prepare & Train on latest dump' flow at the top of the page."""
    st.subheader("Latest dump")
    state = path_coordinator.get_latest_sync_state()
    if state["state"] == "no_dumps":
        st.info(
            f"No pybullet_hsr dump found under `{PYBULLET_HSR_ANNOTATION_ROOT}`."
        )
        return

    dump = state["dump"]
    dump_name = dump["path"].name
    dataset_name = state["local_dataset_name"]

    if state["state"] == "up_to_date":
        st.success(
            f"`datasets/{dataset_name}` already in sync with `{dump_name}`."
        )
    elif state["state"] == "no_local":
        st.warning(
            f"Newest dump `{dump_name}` has not been prepared. "
            f"Click below to prepare and start training in one step."
        )
    else:
        st.warning(
            f"`datasets/{dataset_name}` is stale vs `{dump_name}` "
            f"— {state['reason']}."
        )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Sync latest dump", use_container_width=True):
            if _sync_latest_dump(dump["path"]):
                st.rerun()
    with col_b:
        if st.button("⚡ Prepare & Train on latest dump",
                     type="primary", use_container_width=True):
            data_yaml = _sync_latest_dump(dump["path"])
            if data_yaml:
                st.session_state["_selected_dataset_name"] = dataset_name
                st.session_state["_selected_data_yaml"] = str(data_yaml)
                _kickoff_training_with_current_ui_settings(
                    task_manager=task_manager,
                    path_coordinator=path_coordinator,
                    data_yaml=data_yaml,
                )


def _sync_latest_dump(dump_path) -> Path | None:
    from prepare_dataset import ensure_dataset
    with st.spinner(f"Preparing {dump_path.name}…"):
        try:
            result = ensure_dataset(source=dump_path, symlink=True)
        except SystemExit as exc:
            st.error(f"prepare failed: {exc}")
            return None
        except Exception as exc:  # pragma: no cover - UI safety net
            st.error(f"prepare failed: {exc}")
            return None
    counts = result.get("counts", {})
    if result["action"] == "up-to-date":
        st.info(f"`datasets/{result['dataset_name']}` already up-to-date.")
    else:
        st.success(
            f"Prepared `datasets/{result['dataset_name']}` "
            f"(train={counts.get('train', 0)}, val={counts.get('val', 0)})."
        )
    st.cache_data.clear()
    return Path(result["data_yaml"])


def _kickoff_training_with_current_ui_settings(
    *,
    task_manager: TaskManager,
    path_coordinator: PathCoordinator,
    data_yaml: Path,
) -> None:
    """Start training using the same defaults the user last saw in the UI."""
    ui_settings_manager = st.session_state.ui_settings_manager
    settings = ui_settings_manager.load()
    advanced_params = {
        k: v for k, v in settings.training_params.__dict__.items()
    }
    task_id = task_manager.start_training(
        dataset_yaml=str(data_yaml),
        base_model=None,
        output_dir=str(path_coordinator.get_path("finetuned_dir")),
        epochs=50,
        batch_size=None,
        fast_mode=False,
        auto_scale=True,
        enable_tensorboard=True,
        tensorboard_port=6006,
        advanced_params=advanced_params,
    )
    st.success(f"Training started — task `{task_id}`")
    time.sleep(1)
    st.rerun()


def _render_dataset_picker(path_coordinator: PathCoordinator) -> None:
    st.subheader("Dataset")

    datasets = path_coordinator.get_datasets()
    dumps = path_coordinator.get_available_pybullet_hsr_dumps()

    with st.expander("Prepare from pybullet_hsr", expanded=not datasets):
        if not dumps:
            st.warning(
                f"No manifest-bearing dumps under `{PYBULLET_HSR_ANNOTATION_ROOT}`. "
                "Run `pybullet_hsr/scripts/write_manifest.py --dump-dir <dir>` first."
            )
        else:
            labels = [
                f"{d['path'].name} — {d['manifest'].get('dataset_name', '?')} "
                f"({d['manifest'].get('stats', {}).get('num_images', '?')} imgs, "
                f"{len(d['manifest'].get('classes', []))} cls, "
                f"{d['manifest'].get('label_format', '?')})"
                for d in dumps
            ]
            idx = st.selectbox(
                "pybullet_hsr dump",
                options=list(range(len(dumps))),
                format_func=lambda i: labels[i],
                key="_prepare_dump_idx",
            )
            dump = dumps[idx]
            default_name = dump["manifest"].get("dataset_name", dump["path"].name)
            dataset_name = st.text_input(
                "Dataset name (destination)",
                value=default_name,
                help="Output goes to datasets/<name>",
            )
            val_ratio = st.slider("Val ratio", 0.05, 0.3, 0.1, step=0.05)
            symlink = st.checkbox("Symlink (no copy)", value=True)

            if st.button("Run prepare_dataset.py", type="secondary"):
                dest = DATASETS_DIR / dataset_name
                cmd = [
                    sys.executable, str(PREPARE_SCRIPT),
                    "--source", str(dump["path"]),
                    "--dest", str(dest),
                    "--val-ratio", str(val_ratio),
                ]
                if symlink:
                    cmd.append("--symlink")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=str(project_root)
                )
                if result.returncode == 0:
                    st.success(result.stdout.strip() or "Dataset prepared.")
                    st.rerun()
                else:
                    st.error(result.stderr or "prepare_dataset.py failed")

    if not datasets:
        st.warning(
            f"No dataset found under `{DATASETS_DIR}`. Use the preparer above first."
        )
        st.session_state["_selected_data_yaml"] = None
        return

    options = {d["name"]: d for d in datasets}
    names = list(options.keys())
    default_idx = 0
    prev = st.session_state.get("_selected_dataset_name")
    if prev and prev in names:
        default_idx = names.index(prev)
    selected_name = st.selectbox("Select dataset", names, index=default_idx)
    d = options[selected_name]
    st.session_state["_selected_dataset_name"] = selected_name
    st.session_state["_selected_data_yaml"] = d["data_yaml"]

    try:
        config = yaml.safe_load(Path(d["data_yaml"]).read_text())
        names_map = config.get("names", {})
        if isinstance(names_map, dict):
            class_list = [names_map[k] for k in sorted(names_map.keys())]
        else:
            class_list = list(names_map)
        train_dir = Path(d["path"]) / "images" / "train"
        val_dir = Path(d["path"]) / "images" / "val"
        train_n = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
        val_n = len(list(val_dir.glob("*"))) if val_dir.exists() else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Classes", len(class_list))
        c2.metric("Train", train_n)
        c3.metric("Val", val_n)
        with st.expander("Class names"):
            st.write(", ".join(f"{i}:{n}" for i, n in enumerate(class_list)))
    except Exception as e:
        st.error(f"Could not read data.yaml: {e}")


def _render_model_knobs(path_coordinator: PathCoordinator):
    st.subheader("Model")

    auto_scale = st.checkbox(
        "GPU auto-scaling (pick model + batch automatically)",
        value=False,
    )

    pretrained = path_coordinator.get_pretrained_models()
    col_a, col_b = st.columns(2)
    with col_a:
        if auto_scale:
            base_model = None
            st.caption("Model will be selected by the GPU tier at runtime.")
        else:
            options = sorted({Path(p).name for p in pretrained}
                             | {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"})
            default_idx = options.index("yolov8m.pt") if "yolov8m.pt" in options else 0
            base_model = st.selectbox("Base model", options, index=default_idx)
        epochs = st.slider("Epochs", 5, 200, 50, step=5)
    with col_b:
        if auto_scale:
            batch_size = None
            st.caption("Batch will be selected by GPU tier at runtime.")
        else:
            batch_size = st.slider("Batch size", 4, 64, 16, step=4)
        fast_mode = st.checkbox("Fast mode (smaller model, fewer epochs)", value=False)

    return base_model, epochs, batch_size, fast_mode, auto_scale


def _render_monitoring():
    st.subheader("Monitoring")
    enable_tb = st.checkbox("Enable TensorBoard", value=True)
    port = st.number_input("Port", 1024, 65535, 6006) if enable_tb else 6006
    return enable_tb, int(port)


def _render_advanced(auto_scale: bool) -> dict:
    with st.expander("Advanced parameters", expanded=False):
        from components.training_advanced_params import render_advanced_parameters_section
        return render_advanced_parameters_section(
            auto_scale=auto_scale,
            gpu_tier="unknown",
        )


def _render_launch(
    *,
    task_manager: TaskManager,
    path_coordinator: PathCoordinator,
    base_model: str | None,
    epochs: int,
    batch_size: int | None,
    fast_mode: bool,
    auto_scale: bool,
    enable_tensorboard: bool,
    tensorboard_port: int,
    advanced_params: dict,
) -> None:
    data_yaml = st.session_state.get("_selected_data_yaml")
    if not data_yaml:
        st.info("Select or prepare a dataset first.")
        return

    if st.button("🚀 Start training", type="primary", use_container_width=True):
        task_id = task_manager.start_training(
            dataset_yaml=str(data_yaml),
            base_model=base_model,
            output_dir=str(path_coordinator.get_path("finetuned_dir")),
            epochs=epochs,
            batch_size=batch_size,
            fast_mode=fast_mode,
            auto_scale=auto_scale,
            enable_tensorboard=enable_tensorboard,
            tensorboard_port=tensorboard_port,
            advanced_params=advanced_params,
        )
        st.success(f"Training started — task `{task_id}`")
        if enable_tensorboard:
            st.info(f"TensorBoard will be available on port {tensorboard_port}")
        time.sleep(1)
        st.rerun()


def _render_trained(path_coordinator: PathCoordinator) -> None:
    st.subheader("Trained models")
    models = path_coordinator.get_trained_models()
    if not models:
        st.info("No trained models yet.")
        return
    for model in models:
        with st.expander(f"📦 {model['name']}", expanded=False):
            st.caption(f"Created: {model['created'][:19]}")
            run_dir = Path(model["best_path"] or model["last_path"]).parent.parent
            result_file = run_dir / "training_result.json"
            if result_file.exists():
                try:
                    result = json.loads(result_file.read_text())
                    metrics = result.get("metrics", {})
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("mAP@50", f"{metrics.get('mAP50', 0):.3f}")
                    c2.metric("mAP@50-95", f"{metrics.get('mAP50-95', 0):.3f}")
                    c3.metric("Time (min)", f"{result.get('training_time_minutes', 0):.0f}")
                    c4.metric("Epochs", result.get("epochs_completed", "?"))
                except Exception as e:
                    st.info(f"Could not read training_result.json: {e}")
            if model["best_path"]:
                st.code(f"best: {model['best_path']}", language="text")
            if model["last_path"]:
                st.code(f"last: {model['last_path']}", language="text")
            if st.button("Send to Evaluation", key=f"eval_{model['name']}"):
                st.session_state["selected_model"] = model["best_path"] or model["last_path"]
                st.info("Switch to the Evaluation page to run metrics.")


if __name__ == "__main__":
    show_training_page()
