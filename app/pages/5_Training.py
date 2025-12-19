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
import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.task_manager import Task

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


def show_training_page() -> None:
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


def _get_active_training_task(task_manager: TaskManager) -> "Task | None":
    """Get active training task if exists."""
    tasks = task_manager.get_active_tasks(task_type="training")
    if tasks:
        return tasks[0]
    return None


def _merge_synthetic_images(
    path_coordinator: PathCoordinator,
    dataset_path: Path,
    synthetic_sessions: list[str],
) -> int:
    """
    Merge synthetic images into the training dataset.

    Synthetic images are copied directly to the training set (not validation)
    because they are generated from real training data.

    Args:
        path_coordinator: PathCoordinator instance
        dataset_path: Path to the dataset directory (containing images/train, labels/train)
        synthetic_sessions: List of synthetic session names to include

    Returns:
        Number of synthetic images added
    """
    import shutil

    synthetic_dir = path_coordinator.get_path("synthetic_dir")
    train_images_dir = dataset_path / "images" / "train"
    train_labels_dir = dataset_path / "labels" / "train"

    # Ensure directories exist
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    added_count = 0

    for session_name in synthetic_sessions:
        session_path = synthetic_dir / session_name
        if not session_path.exists():
            continue

        images_dir = session_path / "images"
        labels_dir = session_path / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            continue

        # Find all image-label pairs
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            # Generate unique filename with synthetic prefix
            new_name = f"synthetic_{session_name}_{image_path.stem}"

            # Copy image
            image_dest = train_images_dir / f"{new_name}{image_path.suffix}"
            if not image_dest.exists():  # Avoid duplicates
                shutil.copy2(image_path, image_dest)

                # Copy label
                label_dest = train_labels_dir / f"{new_name}.txt"
                shutil.copy2(label_path, label_dest)

                added_count += 1

    return added_count


def _render_active_training_view(active_task: "Task", task_manager: TaskManager, path_coordinator: PathCoordinator) -> None:
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


def _render_start_training(task_manager: TaskManager, path_coordinator: PathCoordinator) -> None:
    """Render training configuration with Mission Control aesthetic and sub-tabs."""

    # Create sub-tabs for organized configuration
    dataset_tab, synthetic_tab, model_tab, advanced_tab, others_tab = st.tabs([
        "📂 Dataset",
        "🎨 Synthetic",
        "🤖 Model",
        "⚙️ Advanced",
        "📦 Others"
    ])

    # Others tab processed first to get hardware info for other tabs
    with others_tab:
        with st.container(border=True):
            gpu_available, gpu_name, gpu_memory, gpu_tier, auto_scale = _render_hardware_section()
            st.markdown("---")
            enable_tensorboard, tensorboard_port = _render_monitoring_section()

    with dataset_tab:
        with st.container(border=True):
            _render_dataset_section(path_coordinator)

    with synthetic_tab:
        with st.container(border=True):
            _render_synthetic_section(path_coordinator)

    with model_tab:
        with st.container(border=True):
            base_model, epochs, batch_size, fast_mode = _render_model_section(
                auto_scale=auto_scale,
                gpu_available=gpu_available,
                gpu_tier=gpu_tier,
                path_coordinator=path_coordinator
            )

    with advanced_tab:
        with st.container(border=True):
            advanced_params = _render_advanced_section(
                auto_scale=auto_scale,
                gpu_tier=gpu_tier
            )

    # Start button outside tabs
    _render_start_button(
        task_manager=task_manager,
        path_coordinator=path_coordinator,
        base_model=base_model if 'base_model' in locals() else None,
        epochs=epochs if 'epochs' in locals() else 50,
        batch_size=batch_size if 'batch_size' in locals() else None,
        fast_mode=fast_mode if 'fast_mode' in locals() else False,
        auto_scale=auto_scale if 'auto_scale' in locals() else False,
        enable_tensorboard=enable_tensorboard if 'enable_tensorboard' in locals() else True,
        tensorboard_port=tensorboard_port if 'tensorboard_port' in locals() else 6006,
        advanced_params=advanced_params if 'advanced_params' in locals() else {}
    )


def _render_dataset_section(path_coordinator: PathCoordinator) -> None:
    """Render dataset selection section."""
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
        st.session_state["selected_dataset_info"] = None
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
            st.session_state["selected_dataset_info"] = None
            return

    # Store dataset info in session state for use by other sections
    st.session_state["selected_dataset_info"] = {
        "session": selected_session,
        "data_yaml": str(data_yaml) if data_yaml else None,
        "train_count": train_count,
        "val_count": val_count,
        "class_names": class_names
    }


def _render_synthetic_section(path_coordinator: PathCoordinator) -> None:
    """Render synthetic image configuration section with dynamic Copy-Paste settings."""
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>🎨</span>
        <span>Synthetic Image Generation</span>
    </div>
    """)

    st.markdown("""
    Generate synthetic training images dynamically using Copy-Paste augmentation.
    Objects are automatically pasted onto background images with edge blending and white balance.
    """)

    # Enable dynamic synthetic generation
    enable_dynamic_synthetic = st.checkbox(
        "Enable Dynamic Synthetic Generation",
        value=True,
        help="Generate synthetic images before training starts using Copy-Paste augmentation",
        key="enable_dynamic_synthetic"
    )

    if not enable_dynamic_synthetic:
        st.info("Dynamic generation disabled. You can still use pre-generated synthetic sessions below.")
        _render_static_synthetic_selection(path_coordinator)
        return

    # Check prerequisites
    annotated_dir = path_coordinator.get_path("annotated_dir")
    backgrounds_dir = path_coordinator.get_path("backgrounds_dir")

    # Check for annotated masks (uses cached function)
    mask_stats = path_coordinator.get_mask_stats()

    if not mask_stats:
        st.warning("No annotated masks found. Please run annotation first.")
        st.session_state["synthetic_config"] = {"enabled": False}
        return

    # Check for backgrounds
    backgrounds = path_coordinator.get_background_images()
    if not backgrounds:
        st.warning("No background images found. Please add backgrounds to generate synthetic images.")
        st.info(f"Add background images to: `{backgrounds_dir}`")
        st.session_state["synthetic_config"] = {"enabled": False}
        return

    st.success(f"Found {len(mask_stats)} annotated classes and {len(backgrounds)} background images")

    st.markdown("---")

    # Generation Settings
    st.markdown("##### Generation Settings")

    col1, col2 = st.columns(2)

    with col1:
        synthetic_ratio = st.slider(
            "Synthetic:Real Ratio",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.5,
            help="Ratio of synthetic images to real images (e.g., 2.0 = 2:1)",
            key="synthetic_ratio"
        )

    with col2:
        max_objects = st.slider(
            "Max Objects per Image",
            min_value=1,
            max_value=5,
            value=3,
            help="Maximum number of objects to paste per synthetic image",
            key="synthetic_max_objects"
        )

    # Scale and Rotation
    st.markdown("##### Transform Settings")

    col1, col2 = st.columns(2)

    with col1:
        scale_min = st.slider(
            "Min Scale",
            min_value=0.3,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="synthetic_scale_min"
        )

    with col2:
        scale_max = st.slider(
            "Max Scale",
            min_value=0.5,
            max_value=2.0,
            value=1.5,
            step=0.1,
            key="synthetic_scale_max"
        )

    col3, col4 = st.columns(2)

    with col3:
        rotation_min = st.slider(
            "Min Rotation (°)",
            min_value=-180,
            max_value=0,
            value=-15,
            step=5,
            help="Minimum rotation angle in degrees",
            key="synthetic_rotation_min"
        )

    with col4:
        rotation_max = st.slider(
            "Max Rotation (°)",
            min_value=0,
            max_value=180,
            value=15,
            step=5,
            help="Maximum rotation angle in degrees",
            key="synthetic_rotation_max"
        )

    # Appearance Settings
    st.markdown("##### Appearance Settings")

    col1, col2 = st.columns(2)

    with col1:
        enable_white_balance = st.checkbox(
            "Enable White Balance",
            value=True,
            help="Adjust object colors to match background lighting",
            key="synthetic_enable_wb"
        )

        if enable_white_balance:
            white_balance_strength = st.slider(
                "White Balance Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Strength of white balance adjustment (0.0 = none, 1.0 = full)",
                key="synthetic_wb_strength"
            )
        else:
            white_balance_strength = 0.7

    with col2:
        edge_blur_sigma = st.slider(
            "Edge Blur Sigma",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Blur strength for object edges (higher = smoother blending)",
            key="synthetic_edge_blur"
        )

    st.markdown("---")

    # Preview Section
    st.markdown("##### Preview")

    # Initialize preview seed in session state
    if "synth_preview_seed" not in st.session_state:
        st.session_state["synth_preview_seed"] = 42

    # Generate Preview button
    if st.button("🔄 Generate Preview", key="synth_preview_btn"):
        import time
        st.session_state["synth_preview_seed"] = int(time.time() * 1000) % (2**31)
        st.session_state["synth_preview_generated"] = True

    # Generate and display preview images only when button is clicked
    if st.session_state.get("synth_preview_generated", False):
        try:
            # Import annotation page module dynamically (filename starts with number)
            annotation_page = importlib.import_module("pages.4_Annotation")
            previews = annotation_page._generate_preview_images(
                path_coordinator=path_coordinator,
                selected_classes=list(mask_stats.keys()),
                scale_range=(scale_min, scale_max),
                enable_white_balance=enable_white_balance,
                max_objects=max_objects,
                seed=st.session_state["synth_preview_seed"],
                num_samples=3,
                rotation_range=(float(rotation_min), float(rotation_max)),
                enable_horizontal_flip=True,
                enable_vertical_flip=False,
                white_balance_strength=white_balance_strength,
                edge_blur_sigma=edge_blur_sigma,
                min_objects=1,
            )

            if previews:
                preview_cols = st.columns(3)
                for idx, (image, class_names_in_img) in enumerate(previews):
                    with preview_cols[idx]:
                        st.image(image, use_container_width=True)
                        if class_names_in_img:
                            st.caption(f"Classes: {', '.join(class_names_in_img)}")
                        else:
                            st.caption("No objects placed")
            else:
                st.info("Could not generate preview. Check masks and backgrounds.")

        except Exception as e:
            st.warning(f"Preview generation failed: {str(e)}")
    else:
        st.info("Click 'Generate Preview' to see sample synthetic images.")

    # Store configuration in session state
    st.session_state["synthetic_config"] = {
        "enabled": enable_dynamic_synthetic,
        "ratio": synthetic_ratio,
        "scale_range": (scale_min, scale_max),
        "rotation_range": (float(rotation_min), float(rotation_max)),
        "enable_white_balance": enable_white_balance,
        "white_balance_strength": white_balance_strength,
        "edge_blur_sigma": edge_blur_sigma,
        "max_objects": max_objects,
        "backgrounds_dir": str(backgrounds_dir),
        "annotated_dir": str(annotated_dir),
    }

    # Auto-save UI settings when synthetic config changes
    if "ui_settings_manager" in st.session_state:
        st.session_state.ui_settings_manager.save_from_session_state()

    st.markdown("---")

    # Also show static synthetic session selection as fallback
    _render_static_synthetic_selection(path_coordinator)


def _render_static_synthetic_selection(path_coordinator: PathCoordinator) -> None:
    """Render static synthetic session selection (pre-generated sessions)."""
    synthetic_sessions = path_coordinator.get_synthetic_sessions()
    selected_synthetic = []

    if synthetic_sessions:
        st.html(f"""
        <div style="
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span>📁</span>
            <span>Pre-generated Synthetic Sessions</span>
        </div>
        """)

        with st.expander("Select pre-generated synthetic images for training", expanded=False):
            st.info("Synthetic images will be added to training set only (not validation).")

            for session in synthetic_sessions:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.checkbox(
                        f"{session['name']}",
                        value=False,
                        key=f"static_synthetic_{session['name']}",
                        help=f"Created: {session['created'][:10]}"
                    ):
                        selected_synthetic.append(session['name'])
                with col2:
                    st.html(f"""
                    <div style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.85rem;
                        opacity: 0.7;
                        padding-top: 4px;
                    ">{session['image_count']} images</div>
                    """)

            if selected_synthetic:
                total_synthetic = sum(
                    s['image_count'] for s in synthetic_sessions
                    if s['name'] in selected_synthetic
                )
                st.html(f"""
                <div style="
                    border-radius: 8px;
                    padding: 12px;
                    margin-top: 12px;
                    background: {COLORS["accent_primary"]}15;
                ">
                    <span style="
                        font-family: 'JetBrains Mono', monospace;
                        font-size: 0.85rem;
                        color: {COLORS["accent_primary"]};
                    ">+{total_synthetic} pre-generated synthetic images will be added to training</span>
                </div>
                """)

    # Store selected static synthetic sessions in session state
    st.session_state["selected_synthetic_sessions"] = selected_synthetic


@st.cache_resource
def _get_gpu_info() -> dict:
    """Get GPU information (cached to avoid repeated CUDA queries)."""
    try:
        import torch
        if torch.cuda.is_available():
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            # Determine tier
            if memory >= 24:
                tier = "workstation"
            elif memory >= 12:
                tier = "high"
            elif memory >= 6:
                tier = "medium"
            else:
                tier = "low"
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory": memory,
                "tier": tier,
            }
    except ImportError:
        pass
    return {"available": False, "name": "", "memory": 0.0, "tier": "unknown"}


def _render_hardware_section() -> tuple:
    """Render hardware detection and GPU scaling section.

    Returns:
        tuple: (gpu_available, gpu_name, gpu_memory, gpu_tier, auto_scale)
    """

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

    # Get cached GPU info
    gpu_info = _get_gpu_info()
    gpu_available = gpu_info["available"]
    gpu_name = gpu_info["name"]
    gpu_memory = gpu_info["memory"]
    gpu_tier = gpu_info["tier"]

    if gpu_available:
        render_gpu_status_card(
            gpu_name=gpu_name,
            gpu_memory=gpu_memory,
            gpu_tier=gpu_tier,
            auto_scale_enabled=False
        )
    else:
        render_gpu_not_available()

    # Auto-scaling option
    auto_scale = st.checkbox(
        "Enable GPU Auto-Scaling",
        value=False,
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

    return gpu_available, gpu_name, gpu_memory, gpu_tier, auto_scale


def _render_model_section(auto_scale: bool, gpu_available: bool, gpu_tier: str, path_coordinator: PathCoordinator) -> tuple:
    """Render model configuration section.

    Returns:
        tuple: (base_model, epochs, batch_size, fast_mode)
    """
    # Determine recommended values for auto-scale
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
            st.info(f"Auto-scaling will select: **{recommended_model}**")
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
            st.info(f"Auto-scaling will select batch size: **{recommended_batch}**")
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

    return base_model, epochs, batch_size, fast_mode


def _render_advanced_section(auto_scale: bool, gpu_tier: str) -> dict:
    """Render advanced parameters section.

    Returns:
        dict: Advanced training parameters
    """
    st.html(f"""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>⚙️</span>
        <span>Advanced Parameters</span>
    </div>
    """)

    from components.training_advanced_params import render_advanced_parameters_section

    advanced_params = render_advanced_parameters_section(
        auto_scale=auto_scale,
        gpu_tier=gpu_tier,
    )

    return advanced_params


def _render_monitoring_section() -> tuple:
    """Render monitoring configuration section.

    Returns:
        tuple: (enable_tensorboard, tensorboard_port)
    """
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

    return enable_tensorboard, tensorboard_port


def _render_start_button(
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
    """Render validation summary and start training button."""

    # Get dataset info from session state
    dataset_info = st.session_state.get("selected_dataset_info")

    if not dataset_info or not dataset_info.get("data_yaml"):
        st.warning("Please select a dataset in the Dataset tab first.")
        return

    data_yaml = dataset_info["data_yaml"]
    train_count = dataset_info.get("train_count", 0)
    val_count = dataset_info.get("val_count", 0)
    selected_session = dataset_info.get("session")

    # Get synthetic config from session state
    synthetic_config = st.session_state.get("synthetic_config", {"enabled": False})
    selected_synthetic_sessions = st.session_state.get("selected_synthetic_sessions", [])

    # Determine actual model and batch size
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

    # Detect GPU for accurate recommendations
    gpu_available = False
    gpu_memory = 0.0
    gpu_tier = "unknown"

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 24:
                gpu_tier = "workstation"
            elif gpu_memory >= 12:
                gpu_tier = "high"
            elif gpu_memory >= 6:
                gpu_tier = "medium"
            else:
                gpu_tier = "low"
    except ImportError:
        pass

    recommended_model = tier_models.get(gpu_tier, "yolov8m.pt")
    recommended_batch = tier_batches.get(gpu_tier, 16)

    current_model = base_model if base_model else (recommended_model if auto_scale and gpu_available else "yolov8m.pt")
    current_batch = batch_size if batch_size else (recommended_batch if auto_scale and gpu_available else 16)

    st.markdown("---")

    # Validation
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

    # Estimated time
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

    # Config summary
    if selected_session:
        render_config_summary(
            dataset_name=selected_session["name"],
            train_count=train_count,
            val_count=val_count,
            model=current_model,
            batch_size=current_batch,
            epochs=epochs,
            gpu_tier=gpu_tier,
            estimated_time=estimated_minutes,
            auto_scale=auto_scale
        )

    if selected_synthetic_sessions:
        total_static = sum(
            s['image_count'] for s in path_coordinator.get_synthetic_sessions()
            if s['name'] in selected_synthetic_sessions
        )
        st.info(f"📁 +{total_static} pre-generated synthetic images will be added")

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Start button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            # Merge static synthetic images if any are selected
            final_data_yaml = data_yaml
            synthetic_added = 0

            if selected_synthetic_sessions and data_yaml:
                with st.spinner("Merging synthetic images into training dataset..."):
                    synthetic_added = _merge_synthetic_images(
                        path_coordinator=path_coordinator,
                        dataset_path=Path(data_yaml).parent,
                        synthetic_sessions=selected_synthetic_sessions,
                    )
                    if synthetic_added > 0:
                        st.success(f"Added {synthetic_added} static synthetic images to training set")

            # Build synthetic config for dynamic generation
            synth_config_for_training = None
            if synthetic_config.get("enabled"):
                synth_config_for_training = {
                    "dynamic_synthetic_enabled": True,
                    "synthetic_ratio": synthetic_config.get("ratio", 2.0),
                    "synthetic_scale_range": synthetic_config.get("scale_range", (0.5, 1.5)),
                    "synthetic_rotation_range": synthetic_config.get("rotation_range", (-15.0, 15.0)),
                    "synthetic_white_balance": synthetic_config.get("enable_white_balance", True),
                    "synthetic_white_balance_strength": synthetic_config.get("white_balance_strength", 0.7),
                    "synthetic_edge_blur": synthetic_config.get("edge_blur_sigma", 2.0),
                    "synthetic_max_objects": synthetic_config.get("max_objects", 3),
                    "backgrounds_dir": synthetic_config.get("backgrounds_dir"),
                    "annotated_dir": synthetic_config.get("annotated_dir"),
                }

            task_id = task_manager.start_training(
                dataset_yaml=str(final_data_yaml),
                base_model=base_model,
                output_dir=str(path_coordinator.get_path("finetuned_dir")),
                epochs=epochs,
                batch_size=batch_size,
                fast_mode=fast_mode,
                auto_scale=auto_scale,
                enable_tensorboard=enable_tensorboard,
                tensorboard_port=tensorboard_port,
                advanced_params=advanced_params,
                synthetic_config=synth_config_for_training,
            )

            synth_msg = ""
            if synthetic_added > 0:
                synth_msg += f" (+{synthetic_added} static synthetic)"
            if synthetic_config.get("enabled"):
                synth_msg += " (dynamic generation enabled)"

            st.html(f"""
            <div class="mc-validation success mc-animate-fade" style="margin-top: 16px;">
                <span class="icon">✓</span>
                <div>
                    <strong>Training started!</strong><br>
                    <span style="font-size: 0.85rem; opacity: 0.9;">
                        Task ID: {task_id}{synth_msg}
                    </span>
                </div>
            </div>
            """)

            if enable_tensorboard:
                st.info(f"TensorBoard will be available at http://localhost:{tensorboard_port}")

            import time
            time.sleep(1)
            st.rerun()


def _render_trained_models(path_coordinator: PathCoordinator) -> None:
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


def _render_training_history(task_manager: TaskManager) -> None:
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
