"""
Training Page

Provides UI for running YOLOv8 fine-tuning on annotated datasets.
Features:
- GPU auto-scaling integration
- TensorBoard monitoring with embedded view + external link
- Competition-optimized training presets
- Real-time training progress with charts
- Mission Control aesthetic UI

Integrates with scripts/training/quick_finetune.py via TaskManager.
"""

import streamlit as st
from pathlib import Path
import sys

# Add app directory to path
app_dir = Path(__file__).parent.parent
project_root = app_dir.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))
if str(project_root / "scripts") not in sys.path:
    sys.path.insert(0, str(project_root / "scripts"))

from services.task_manager import TaskManager, TaskStatus
from services.path_coordinator import PathCoordinator

# Import new Mission Control components
from components.training_styles import inject_training_styles, COLORS, ICONS
from components.training_charts import (
    render_training_chart,
    render_epoch_metrics_chart,
)
from components.tensorboard_embed import (
    render_tensorboard_panel,
    render_tensorboard_status,
)
from components.config_preview import (
    validate_training_config,
    render_validation_messages,
    render_gpu_status_card,
    render_gpu_not_available,
    render_config_summary,
    render_model_recommendation,
    render_target_metrics_info,
)
from components.progress_display import (
    render_task_progress,
    render_active_task_banner,
    render_task_list,
    render_task_metrics,
    render_training_active_banner,
    render_training_completed_banner,
    render_circular_progress,
    render_training_metric_cards,
)


def show_training_page():
    """Main training page with Mission Control aesthetic."""
    # Render common sidebar
    from components.common_sidebar import render_common_sidebar
    render_common_sidebar()

    # Inject custom CSS
    inject_training_styles()

    st.title("🚀 Model Training")

    # Get services from session state (profile-aware)
    if "task_manager" not in st.session_state or "path_coordinator" not in st.session_state:
        st.error("Services not initialized. Please reload the page.")
        return

    task_manager = st.session_state.task_manager
    path_coordinator = st.session_state.path_coordinator

    # Check for active training task
    active_task = _get_active_training_task(task_manager)

    if active_task:
        _render_active_training_view(active_task, task_manager, path_coordinator)
        return

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs([
        f"{ICONS['dataset']} Start Training",
        f"{ICONS['model']} Models",
        "📜 History"
    ])

    with tab1:
        _render_start_training(task_manager, path_coordinator)

    with tab2:
        _render_trained_models(path_coordinator)

    with tab3:
        _render_training_history(task_manager)


def _get_active_training_task(task_manager: TaskManager):
    """Get active training task if exists."""
    tasks = task_manager.get_active_tasks(task_type="training")
    if tasks:
        return tasks[0]
    return None


def _render_active_training_view(active_task, task_manager: TaskManager, path_coordinator):
    """Render the active training view with Mission Control aesthetic."""
    task = task_manager.get_task(active_task.task_id)

    if not task:
        return

    # Active training banner
    render_training_active_banner(task, task_manager)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Training metrics cards
    render_training_metric_cards(task)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Two-column layout: Chart + Config
    col1, col2 = st.columns([2, 1])

    with col1:
        # Training progress chart
        st.html(f"""
        <div style="
            border-radius: 12px;
            padding: 16px;
        ">
            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                opacity: 0.7;
                margin-bottom: 12px;
            ">{ICONS["accuracy"]} Training Progress</div>
        """)

        training_history = task.extra_data.get("training_history", []) if task.extra_data else []
        render_training_chart(training_history, target_map50=0.85, height=320)

        st.html("</div>")

    with col2:
        # GPU and config info
        if task.extra_data:
            gpu_info = task.extra_data.get("gpu_info", "Unknown")
            gpu_tier = task.extra_data.get("gpu_tier", "unknown")
            gpu_memory = 0.0

            # Extract memory from gpu_info if available
            try:
                if "GB" in gpu_info:
                    import re
                    match = re.search(r'(\d+\.?\d*)\s*GB', gpu_info)
                    if match:
                        gpu_memory = float(match.group(1))
            except Exception:
                pass

            render_gpu_status_card(
                gpu_name=gpu_info.split(" (")[0] if " (" in gpu_info else gpu_info,
                gpu_memory=gpu_memory,
                gpu_tier=gpu_tier,
                auto_scale_enabled=True
            )

            # Config summary
            config = task.extra_data.get("config", {})
            if config:
                st.html(f"""
                <div style="
                    border-radius: 12px;
                    padding: 16px;
                    margin-top: 16px;
                ">
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.75rem;
                        opacity: 0.5;
                        text-transform: uppercase;
                        letter-spacing: 0.1em;
                        margin-bottom: 12px;
                    ">Configuration</div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                        <div>
                            <div style="font-size: 0.7rem; opacity: 0.5;">Model</div>
                            <div style="font-family: 'JetBrains Mono', monospace;">{config.get("model", "N/A")}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; opacity: 0.5;">Batch</div>
                            <div style="font-family: 'JetBrains Mono', monospace;">{config.get("batch", "N/A")}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; opacity: 0.5;">Image Size</div>
                            <div style="font-family: 'JetBrains Mono', monospace;">{config.get("imgsz", 640)}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.7rem; opacity: 0.5;">Epochs</div>
                            <div style="font-family: 'JetBrains Mono', monospace;">{config.get("epochs", "N/A")}</div>
                        </div>
                    </div>
                </div>
                """)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # TensorBoard section
    if task.extra_data:
        tensorboard_url = task.extra_data.get("tensorboard_url")
        if tensorboard_url:
            with st.expander(f"{ICONS['tensorboard']} TensorBoard", expanded=False):
                render_tensorboard_panel(
                    tensorboard_url=tensorboard_url,
                    show_iframe=False,
                    iframe_height=500,
                )
        else:
            render_tensorboard_status(is_running=False)

    # CLI output display
    if task.extra_data:
        log_file = task.extra_data.get("log_file")
        if log_file:
            log_path = Path(log_file)
            if log_path.exists():
                with st.expander(f"{ICONS.get('terminal', '>')} Training Logs", expanded=False):
                    try:
                        with open(log_path, "r") as f:
                            lines = f.readlines()
                            # Display the latest 100 lines
                            recent_lines = lines[-100:] if len(lines) > 100 else lines
                            st.code("".join(recent_lines), language="text")
                    except Exception:
                        st.info("Failed to load logs")

    # Check for completion
    if task.status == TaskStatus.COMPLETED:
        st.balloons()
        render_training_completed_banner(task)

    # Auto-refresh every 2 seconds during training
    if task.status == TaskStatus.RUNNING:
        import time
        time.sleep(2)
        st.rerun()


def _render_start_training(task_manager: TaskManager, path_coordinator: PathCoordinator):
    """Render training configuration with Mission Control aesthetic."""

    # Dataset selection section
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>{ICONS["dataset"]}</span>
        <span>Dataset Selection</span>
    </div>
    """)

    sessions = path_coordinator.get_annotation_sessions()
    ready_sessions = [s for s in sessions if s["has_data_yaml"]]

    if not ready_sessions:
        st.html(f"""
        <div class="mc-validation warning mc-animate-fade">
            <span class="icon">⚠</span>
            <div>
                <strong>No annotated datasets found</strong><br>
                <span style="font-size: 0.85rem; opacity: 0.9;">
                    Please run annotation first to create a dataset.
                </span>
            </div>
        </div>
        """)
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
        format_func=lambda x: f"{x['name']} ({x['created'][:10]})",
        label_visibility="collapsed"
    )

    # Dataset info card
    train_count = 0
    val_count = 0
    class_names = []
    data_yaml = None

    if selected_session:
        session_path = Path(selected_session["path"])
        data_yaml = session_path / "data.yaml"

        try:
            import yaml
            with open(data_yaml) as f:
                config = yaml.safe_load(f)
            class_names = config.get("names", [])

            # Count images
            train_dir = session_path / "images" / "train"
            val_dir = session_path / "images" / "val"
            train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
            val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0

            # Dataset info card
            st.html(f"""
            <div style="
                border-radius: 12px;
                padding: 16px;
                margin: 12px 0;
            ">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                    <div style="text-align: center;">
                        <div style="
                            font-family: 'JetBrains Mono', monospace;
                            font-size: 1.5rem;
                            font-weight: 600;
                            color: {COLORS["accent_primary"]};
                        ">{len(class_names)}</div>
                        <div style="
                            font-size: 0.75rem;
                            opacity: 0.5;
                        ">Classes</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="
                            font-family: 'JetBrains Mono', monospace;
                            font-size: 1.5rem;
                            font-weight: 600;
                            color: {COLORS["info"]};
                        ">{train_count}</div>
                        <div style="
                            font-size: 0.75rem;
                            opacity: 0.5;
                        ">Train Images</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="
                            font-family: 'JetBrains Mono', monospace;
                            font-size: 1.5rem;
                            font-weight: 600;
                            color: {COLORS["tier_high"]};
                        ">{val_count}</div>
                        <div style="
                            font-size: 0.75rem;
                            opacity: 0.5;
                        ">Val Images</div>
                    </div>
                </div>
            </div>
            """)

            # Class names expander
            with st.expander("View Classes", expanded=False):
                cols = st.columns(3)
                for i, name in enumerate(class_names):
                    with cols[i % 3]:
                        st.html(f"""
                        <div style="
                            border-radius: 4px;
                            padding: 4px 8px;
                            margin: 2px 0;
                            font-family: 'JetBrains Mono', monospace;
                            font-size: 0.8rem;
                            opacity: 0.85;
                        ">
                            <span style="opacity: 0.5;">{i}:</span> {name}
                        </div>
                        """)

        except Exception as e:
            st.error(f"Error reading dataset config: {e}")
            return

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Hardware detection section
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>⚡</span>
        <span>Hardware & GPU Scaling</span>
    </div>
    """)

    gpu_available = False
    gpu_name = ""
    gpu_memory = 0.0
    gpu_tier = "unknown"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Determine tier
            if gpu_memory >= 24:
                gpu_tier = "workstation"
            elif gpu_memory >= 12:
                gpu_tier = "high"
            elif gpu_memory >= 6:
                gpu_tier = "medium"
            else:
                gpu_tier = "low"

            render_gpu_status_card(
                gpu_name=gpu_name,
                gpu_memory=gpu_memory,
                gpu_tier=gpu_tier,
                auto_scale_enabled=False
            )
        else:
            render_gpu_not_available()
    except ImportError:
        st.warning("PyTorch not installed. Cannot check GPU availability.")

    # Auto-scaling option
    auto_scale = st.checkbox(
        "Enable GPU Auto-Scaling",
        value=True,
        help="Automatically select optimal model, batch size, and settings based on GPU capabilities."
    )

    if auto_scale and gpu_available:
        tier_models = {
            "low": "yolov8s.pt",
            "medium": "yolov8m.pt",
            "high": "yolov8l.pt",
            "workstation": "yolov8x.pt",
        }
        tier_batches = {
            "low": 8,
            "medium": 16,
            "high": 32,
            "workstation": 64,
        }
        recommended_model = tier_models.get(gpu_tier, "yolov8m.pt")
        recommended_batch = tier_batches.get(gpu_tier, 16)

        render_model_recommendation(
            recommended_model=recommended_model,
            recommended_batch=recommended_batch,
            gpu_tier=gpu_tier
        )

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Model configuration section
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>{ICONS["model"]}</span>
        <span>Model Configuration</span>
    </div>
    """)

    col1, col2 = st.columns(2)

    with col1:
        # Base model selection
        if auto_scale and gpu_available:
            base_model = None  # Will be auto-selected
        else:
            pretrained_models = path_coordinator.get_pretrained_models()
            model_options = ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt", "yolov8l.pt", "yolov8x.pt"] + [
                str(Path(m).name) for m in pretrained_models
                if str(Path(m).name) not in ["yolov8m.pt", "yolov8s.pt", "yolov8n.pt", "yolov8l.pt", "yolov8x.pt"]
            ]

            base_model = st.selectbox(
                "Base Model",
                model_options,
                index=0,
                help="YOLOv8m is recommended for competition. YOLOv8l/x for high-end GPUs."
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
        if auto_scale and gpu_available:
            batch_size = None  # Will be auto-selected
        else:
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

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Monitoring section
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>{ICONS["tensorboard"]}</span>
        <span>Monitoring</span>
    </div>
    """)

    enable_tensorboard = st.checkbox(
        "Enable TensorBoard",
        value=True,
        help="View training progress in real-time with loss curves and metrics."
    )

    if enable_tensorboard:
        tensorboard_port = st.number_input(
            "TensorBoard Port",
            min_value=1024,
            max_value=65535,
            value=6006,
            help="Port for TensorBoard server"
        )
    else:
        tensorboard_port = 6006

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Validation
    if data_yaml:
        current_batch = batch_size if batch_size else (recommended_batch if auto_scale and gpu_available else 16)
        current_model = base_model if base_model else (recommended_model if auto_scale and gpu_available else "yolov8m.pt")

        validation = validate_training_config(
            dataset_yaml=str(data_yaml),
            model=current_model,
            batch_size=current_batch,
            epochs=epochs,
            gpu_memory_gb=gpu_memory,
            auto_scale=auto_scale
        )

        render_validation_messages(validation)

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Target metrics
    render_target_metrics_info()

    st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

    # Estimated time and config summary
    if gpu_available:
        tier_multipliers = {
            "low": 2.0,
            "medium": 1.0,
            "high": 0.6,
            "workstation": 0.4,
        }
        multiplier = tier_multipliers.get(gpu_tier, 1.0)
    else:
        multiplier = 5.0

    estimated_minutes = epochs * multiplier
    if fast_mode:
        estimated_minutes = estimated_minutes * 0.5

    if data_yaml and selected_session:
        render_config_summary(
            dataset_name=selected_session["name"],
            train_count=train_count,
            val_count=val_count,
            model=base_model if base_model else (recommended_model if auto_scale and gpu_available else "yolov8m.pt"),
            batch_size=batch_size if batch_size else (recommended_batch if auto_scale and gpu_available else 16),
            epochs=epochs,
            gpu_tier=gpu_tier,
            estimated_time=estimated_minutes,
            auto_scale=auto_scale
        )

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start Training", type="primary", use_container_width=True):
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
            )

            st.html(f"""
            <div class="mc-validation success mc-animate-fade" style="margin-top: 16px;">
                <span class="icon">✓</span>
                <div>
                    <strong>Training started!</strong><br>
                    <span style="font-size: 0.85rem; opacity: 0.9;">
                        Task ID: {task_id}
                    </span>
                </div>
            </div>
            """)

            if enable_tensorboard:
                st.info(f"TensorBoard will be available at http://localhost:{tensorboard_port}")

            import time
            time.sleep(1)
            st.rerun()


def _render_trained_models(path_coordinator: PathCoordinator):
    """Render list of trained models with Mission Control aesthetic."""
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>{ICONS["model"]}</span>
        <span>Trained Models</span>
    </div>
    """)

    models = path_coordinator.get_trained_models()

    if not models:
        st.html(f"""
        <div style="
            border: 1px dashed currentColor;
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            opacity: 0.6;
        ">
            <div style="
                font-size: 2rem;
                margin-bottom: 12px;
            ">{ICONS["model"]}</div>
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.9rem;
                opacity: 0.8;
            ">No trained models found</div>
            <div style="
                font-family: 'Inter', sans-serif;
                font-size: 0.8rem;
                opacity: 0.5;
                margin-top: 4px;
            ">Run training to create one</div>
        </div>
        """)
        return

    for model in models:
        with st.expander(f"📦 {model['name']}", expanded=False):
            # Model info header
            st.html(f"""
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            ">
                <div style="
                    font-family: 'Inter', sans-serif;
                    font-size: 0.8rem;
                    opacity: 0.5;
                ">Created: {model['created'][:19]}</div>
            </div>
            """)

            # Check for training result
            model_dir = Path(model["best_path"]).parent.parent if model["best_path"] else None
            if model_dir:
                result_file = model_dir / "training_result.json"
                if result_file.exists():
                    try:
                        import json
                        with open(result_file) as f:
                            result = json.load(f)

                        metrics = result.get('metrics', {})
                        mAP50 = metrics.get('mAP50', 0)
                        mAP50_95 = metrics.get('mAP50-95', 0)
                        training_time = result.get('training_time_minutes', 0)
                        epochs_done = result.get('epochs_completed', 'N/A')

                        # Target achievement badge
                        target_achieved = mAP50 >= 0.85
                        badge_color = COLORS["success"] if target_achieved else COLORS["warning"]
                        badge_text = "TARGET MET" if target_achieved else "BELOW TARGET"

                        st.html(f"""
                        <div style="
                            border-radius: 8px;
                            padding: 16px;
                            margin-bottom: 12px;
                        ">
                            <div style="
                                display: flex;
                                justify-content: space-between;
                                align-items: center;
                                margin-bottom: 16px;
                            ">
                                <span style="
                                    background: {badge_color}22;
                                    color: {badge_color};
                                    padding: 4px 8px;
                                    border-radius: 4px;
                                    font-family: 'JetBrains Mono', monospace;
                                    font-size: 0.7rem;
                                    font-weight: 600;
                                    letter-spacing: 0.05em;
                                ">{badge_text}</span>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
                                <div style="text-align: center;">
                                    <div style="
                                        font-family: 'JetBrains Mono', monospace;
                                        font-size: 1.25rem;
                                        font-weight: 600;
                                        color: {COLORS["accent_primary"]};
                                    ">{mAP50:.1%}</div>
                                    <div style="font-size: 0.7rem; opacity: 0.5;">mAP@50</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="
                                        font-family: 'JetBrains Mono', monospace;
                                        font-size: 1.25rem;
                                        font-weight: 600;
                                        color: {COLORS["info"]};
                                    ">{mAP50_95:.1%}</div>
                                    <div style="font-size: 0.7rem; opacity: 0.5;">mAP@50-95</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="
                                        font-family: 'JetBrains Mono', monospace;
                                        font-size: 1.25rem;
                                        font-weight: 600;
                                        opacity: 0.7;
                                    ">{training_time:.0f}m</div>
                                    <div style="font-size: 0.7rem; opacity: 0.5;">Time</div>
                                </div>
                                <div style="text-align: center;">
                                    <div style="
                                        font-family: 'JetBrains Mono', monospace;
                                        font-size: 1.25rem;
                                        font-weight: 600;
                                        opacity: 0.7;
                                    ">{epochs_done}</div>
                                    <div style="font-size: 0.7rem; opacity: 0.5;">Epochs</div>
                                </div>
                            </div>
                        </div>
                        """)

                    except Exception:
                        pass

            # Model paths with download buttons
            if model["best_path"]:
                col_best_path, col_best_dl = st.columns([5, 1])
                with col_best_path:
                    st.html(f"""
                    <div style="
                        border-radius: 6px;
                        padding: 8px 12px;
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.75rem;
                    ">
                        <span style="color: {COLORS["success"]};">●</span>
                        <span style="opacity: 0.5;">Best:</span>
                        <span style="opacity: 0.7;">{model['best_path']}</span>
                    </div>
                    """)
                with col_best_dl:
                    best_path = Path(model["best_path"])
                    if best_path.exists():
                        from datetime import datetime
                        profile_name = st.session_state.profile_manager.get_active_profile().display_name
                        download_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                        with open(best_path, "rb") as f:
                            st.download_button(
                                label="⬇",
                                data=f.read(),
                                file_name=f"{profile_name}_{model['name']}_{download_time}_best.pt",
                                mime="application/octet-stream",
                                key=f"dl_best_{model['name']}",
                                use_container_width=True
                            )

            if model["last_path"]:
                col_last_path, col_last_dl = st.columns([5, 1])
                with col_last_path:
                    st.html(f"""
                    <div style="
                        border-radius: 6px;
                        padding: 8px 12px;
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.75rem;
                    ">
                        <span style="opacity: 0.5;">○</span>
                        <span style="opacity: 0.5;">Last:</span>
                        <span style="opacity: 0.7;">{model['last_path']}</span>
                    </div>
                    """)
                with col_last_dl:
                    last_path = Path(model["last_path"])
                    if last_path.exists():
                        from datetime import datetime
                        profile_name = st.session_state.profile_manager.get_active_profile().display_name
                        download_time = datetime.now().strftime("%Y%m%d-%H%M%S")
                        with open(last_path, "rb") as f:
                            st.download_button(
                                label="⬇",
                                data=f.read(),
                                file_name=f"{profile_name}_{model['name']}_{download_time}_last.pt",
                                mime="application/octet-stream",
                                key=f"dl_last_{model['name']}",
                                use_container_width=True
                            )

            # Action button
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            if st.button("📊 Evaluate", key=f"eval_{model['name']}", use_container_width=True):
                st.session_state["selected_model"] = model["best_path"] or model["last_path"]
                st.info("Go to Evaluation page to evaluate this model")


def _render_training_history(task_manager: TaskManager):
    """Render training task history."""
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>📜</span>
        <span>Training History</span>
    </div>
    """)

    render_task_list(
        task_type="training",
        task_manager=task_manager,
        limit=10,
        show_active_only=False,
    )


# For Streamlit native multipage
if __name__ == "__main__":
    show_training_page()
