"""
Annotation Page

Provides UI for SAM2 interactive annotation application.
Allows users to launch the annotation app for semi-automatic object annotation
using SAM2 segmentation and video tracking.

Also includes dataset preparation functionality to create
training-ready datasets from annotated data.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from services.task_manager import TaskManager, TaskStatus
from services.path_coordinator import PathCoordinator
from services.dataset_preparer import DatasetPreparer
from components.progress_display import (
    render_task_progress,
    render_active_task_banner,
    render_task_list,
    render_task_metrics,
)
from components.dataset_status import (
    render_class_status_grid,
    render_dataset_preparation_panel,
    render_dataset_result,
)
from components.training_styles import inject_training_styles, COLORS, ICONS


def _get_annotation_progress(raw_captures_dir: Path, annotated_dir: Path) -> dict[str, dict[str, Any]]:
    """
    Calculate annotation progress for each class.

    Args:
        raw_captures_dir: Directory containing raw captured images per class
        annotated_dir: Directory containing annotation outputs per class

    Returns:
        Dictionary mapping class_name to progress info:
        {
            "class_name": {
                "total_images": int,
                "annotated_count": int,
                "progress_percent": float,
                "status": "complete" | "in_progress" | "not_started"
            }
        }
    """
    progress = {}

    if not raw_captures_dir.exists():
        return progress

    for class_dir in raw_captures_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        # Count raw images
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        total_images = len(images)

        if total_images == 0:
            continue

        # Count annotated labels (check both labels subdirectory and direct output)
        annotated_count = 0
        class_annotated_dir = annotated_dir / class_name

        if class_annotated_dir.exists():
            # Check labels subdirectory (batch save location)
            labels_dir = class_annotated_dir / "labels"
            if labels_dir.exists():
                annotated_count = len(list(labels_dir.glob("*.txt")))
            else:
                # Fallback to direct output
                annotated_count = len(list(class_annotated_dir.glob("*.txt")))

        # Calculate progress
        progress_percent = (annotated_count / total_images) * 100 if total_images > 0 else 0

        # Determine status
        if annotated_count == 0:
            status = "not_started"
        elif annotated_count >= total_images:
            status = "complete"
        else:
            status = "in_progress"

        progress[class_name] = {
            "total_images": total_images,
            "annotated_count": annotated_count,
            "progress_percent": progress_percent,
            "status": status,
        }

    return progress


def show_annotation_page() -> None:
    """Render the main annotation page."""
    # Render common sidebar
    from components.common_sidebar import render_common_sidebar
    render_common_sidebar()

    # Inject Mission Control styles
    inject_training_styles()

    st.title("🎯 SAM2 Interactive Annotation")

    # Get services from session state (profile-aware)
    if "task_manager" not in st.session_state or "path_coordinator" not in st.session_state:
        st.error("Services not initialized. Please reload the page.")
        return

    task_manager = st.session_state.task_manager
    path_coordinator = st.session_state.path_coordinator

    # Tabs for different sections (ordered by workflow)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Run Annotation",
        "🎨 Generate Synthetic",
        "📦 Prepare Dataset",
        "📁 Sessions",
        "📜 History"
    ])

    with tab1:
        _render_run_annotation(task_manager, path_coordinator)

    with tab2:
        _render_generate_synthetic(path_coordinator)

    with tab3:
        _render_prepare_dataset(path_coordinator)

    with tab4:
        _render_annotation_sessions(path_coordinator)

    with tab5:
        _render_annotation_history(task_manager)


def _render_run_annotation(task_manager: TaskManager, path_coordinator: PathCoordinator) -> None:
    """Render SAM2 Interactive Annotation section."""
    st.subheader("Launch Annotation Application")

    # Check prerequisites
    raw_captures_dir = path_coordinator.get_path("raw_captures_dir")

    # Count available images per class
    image_count = 0
    class_dirs = []
    if raw_captures_dir.exists():
        for class_dir in raw_captures_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if images:
                    class_dirs.append((class_dir.name, len(images), str(class_dir)))
                    image_count += len(images)

    if image_count == 0:
        st.warning(
            "No images found in raw captures directory. "
            "Please collect images first using the Collection page."
        )
        st.info(f"Expected path: `{raw_captures_dir}`")

        # Show sync option if app has collected images
        app_collected_dir = path_coordinator.get_path("app_collected_dir")
        if app_collected_dir.exists():
            app_classes = list(app_collected_dir.iterdir())
            if app_classes:
                st.markdown("---")
                st.subheader("Sync from App Data")
                st.write(f"Found {len(app_classes)} object directories in app data.")

                if st.button("Sync App Data to Datasets"):
                    synced = path_coordinator.sync_all_objects([d.name for d in app_classes])
                    st.success(f"Synced {len(synced)} directories")
                    st.rerun()
        return

    # Show available data with file paths
    st.success(f"Found **{image_count}** images across **{len(class_dirs)}** classes")

    # Get annotation progress for all classes
    annotated_dir = path_coordinator.get_path("annotated_dir")
    annotation_progress = _get_annotation_progress(raw_captures_dir, annotated_dir)

    with st.expander("View Class Distribution", expanded=False):
        for class_name, count, path in sorted(class_dirs):
            st.markdown(f"**{class_name}**: {count} images")
            st.caption(f"`{path}`")

    st.markdown("---")

    # Class selection with progress indicators
    st.markdown("##### Select Class to Annotate")

    # Build class options with progress info for display
    class_options = {}
    display_options = []

    for idx, (name, img_count, path) in enumerate(sorted(class_dirs)):
        class_options[name] = (name, path, idx)

        # Get progress info
        if name in annotation_progress:
            prog = annotation_progress[name]
            annotated = prog["annotated_count"]
            total = prog["total_images"]
            pct = prog["progress_percent"]
            status = prog["status"]

            # Create display label with status indicator
            if status == "complete":
                status_icon = "✅"  # Green checkmark
            elif status == "in_progress":
                status_icon = "🟡"  # Yellow circle
            else:
                status_icon = "⚪"  # White circle

            display_label = f"{status_icon} {name} ({annotated}/{total})"
        else:
            display_label = f"⚪ {name} (0/{img_count})"

        display_options.append((display_label, name))

    # Create selectbox with formatted options
    selected_display = st.selectbox(
        "Class",
        [label for label, _ in display_options],
        help="Choose which class to annotate. Status: ✅ Complete | 🟡 In Progress | ⚪ Not Started",
        label_visibility="collapsed"
    )

    # Map back to actual class name
    selected_class = None
    for label, name in display_options:
        if label == selected_display:
            selected_class = name
            break

    # Show detailed progress for selected class
    if selected_class and selected_class in annotation_progress:
        prog = annotation_progress[selected_class]
        annotated = prog["annotated_count"]
        total = prog["total_images"]
        pct = prog["progress_percent"]
        status = prog["status"]

        # Progress bar with color based on status
        if status == "complete":
            st.progress(1.0, text=f"Annotation complete: {annotated}/{total} images")
        elif status == "in_progress":
            st.progress(pct / 100, text=f"In progress: {annotated}/{total} images ({pct:.0f}%)")
        else:
            st.progress(0.0, text=f"Not started: 0/{total} images")
    elif selected_class:
        # No annotation data yet
        img_count = next((c for n, c, p in class_dirs if n == selected_class), (None, 0, None))[1] if class_dirs else 0
        for n, c, p in class_dirs:
            if n == selected_class:
                img_count = c
                break
        st.progress(0.0, text=f"Not started: 0/{img_count} images")

    class_name, input_dir, class_id = class_options[selected_class]

    # Output directory config (annotated_dir already retrieved above)
    output_dir = str(annotated_dir / class_name)

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Input:** `{input_dir}`")
    with col2:
        st.write(f"**Output:** `{output_dir}`")

    # Device selection
    device = st.radio(
        "Device",
        ["cuda", "cpu"],
        horizontal=True,
        help="CUDA recommended for GPU acceleration. CPU is slower but works without GPU."
    )

    # Model selection
    model_dir = Path("models")
    available_models = []
    if model_dir.exists():
        available_models = sorted([f.name for f in model_dir.glob("sam2*.pt")])
    if not available_models:
        available_models = ["sam2.1_hiera_base_plus.pt"]

    selected_model = st.selectbox(
        "SAM2 Model",
        available_models,
        help="Select the SAM2 model to use. Shows sam2*.pt files in the models/ directory."
    )

    st.markdown("---")

    # Launch button
    if st.button("Launch Annotation App", type="primary"):
        if path_coordinator.open_annotation_app(
            input_dir=input_dir,
            output_dir=output_dir,
            class_id=class_id,
            device=device,
            model_path=f"models/{selected_model}"
        ):
            st.success("Annotation application launched!")
        else:
            st.error("Failed to launch annotation application. Check if the script exists.")

    # Color Legend
    st.markdown("---")
    st.subheader("Color Legend")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **Green**")
        st.caption("Current mask / High confidence")
    with col2:
        st.markdown("🟡 **Yellow**")
        st.caption("Low confidence (review needed)")
    with col3:
        st.markdown("🟠 **Orange**")
        st.caption("Saved annotation")

    st.markdown("---")

    # How to Use section
    with st.expander("How to Use", expanded=False):
        st.markdown("""
### SAM2 Interactive Annotation Application

A Tkinter-based GUI for semi-automatic object annotation using SAM2 segmentation and video tracking.

#### Basic Workflow

1. **Select Image**: Use the image list on the left to select an image
2. **Add Points**:
   - Left-click to add foreground points (include in mask)
   - Right-click to add background points (exclude from mask)
3. **Refine Mask**: Add more points until the mask accurately covers the object
4. **Save Annotation**: Press Enter to save and move to next image

#### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Left-click | Add foreground point |
| Right-click | Add background point |
| Enter | Accept and save annotation |
| Escape | Reset all points |
| Ctrl+Z | Undo last point |
| Arrow keys / N, P | Navigate images |
| Space / M | Toggle mask overlay |
| S | Skip current image |

#### Tracking Mode (Sequential Frame Annotation)

For sequential images (e.g., video frames), use tracking mode for efficient batch annotation:

1. **Enable Tracking Mode**: Check the "Enable Tracking Mode" checkbox
2. **Annotate First Frame**: Add points to create initial mask
3. **Start Tracking**: Click "Start Tracking" to propagate to all frames
4. **Review Results**:
   - Green frames: High confidence
   - Yellow frames: Low confidence (review recommended)
5. **Apply All**: Click "Apply All" to save all annotations

#### VRAM Management

For large image sequences, the application automatically manages GPU memory:

- **Automatic Estimation**: Before tracking, VRAM usage is estimated
- **Batch Splitting**: If estimated usage exceeds 95% of available VRAM:
  - A warning dialog shows estimated usage and available memory
  - Frames are automatically split into smaller batches
  - Each batch is processed sequentially to prevent out-of-memory errors

**Example**: For 200 frames at 1080p on a 12GB GPU:
- Estimated usage: ~14GB
- System splits into 2 batches of 100 frames each
- Processing continues automatically with progress indicator

#### Tips

- Start with objects that have clear boundaries
- Use background points to exclude similar-colored regions
- For tracking mode, choose a representative first frame
- Monitor the confidence indicators for tracking quality
        """)


def _render_prepare_dataset(path_coordinator: PathCoordinator) -> None:
    """Render dataset preparation section with class status."""
    st.html("""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: inherit;
        margin-bottom: 16px;
    ">
        Create training-ready datasets from your annotated data
    </div>
    """)

    # Get dataset preparer
    preparer = DatasetPreparer(path_coordinator)
    classes = preparer.get_available_classes()

    # Class status section
    st.html("""
    <div style="
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: inherit;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    ">
        <span>📊</span>
        <span>Class Status</span>
    </div>
    """)

    render_class_status_grid(classes, columns=2)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # Dataset generation section
    params = render_dataset_preparation_panel(classes)

    if params:
        # Generate dataset
        with st.spinner("Generating dataset..."):
            result = preparer.prepare_dataset(
                class_names=params["classes"],
                output_name=params["dataset_name"],
                val_ratio=params["val_ratio"],
                group_continuous_frames=params.get("group_frames", True),
                group_interval_sec=params.get("group_interval", 2.0),
            )

        render_dataset_result(result, params["dataset_name"])

        if result.success:
            # Add button to go to training
            st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Go to Training", use_container_width=True):
                    st.session_state["selected_dataset"] = str(result.output_dir)
                    st.switch_page("pages/5_Training.py")


def _render_annotation_sessions(path_coordinator: PathCoordinator) -> None:
    """Render list of annotation sessions."""
    st.subheader("Annotation Sessions")

    sessions = path_coordinator.get_annotation_sessions()

    if not sessions:
        st.info("No annotation sessions found. Run annotation to create one.")
        return

    for session in sessions:
        status_icon = "✅" if session["has_data_yaml"] else "⏳"

        with st.expander(f"{status_icon} {session['name']}", expanded=False):
            st.write(f"**Path:** `{session['path']}`")
            st.write(f"**Created:** {session['created'][:19]}")
            st.write(f"**Dataset Ready:** {'Yes' if session['has_data_yaml'] else 'No'}")

            if session["has_data_yaml"]:
                # Show dataset info
                data_yaml = Path(session["path"]) / "data.yaml"
                try:
                    import yaml
                    with open(data_yaml) as f:
                        config = yaml.safe_load(f)

                    st.write(f"**Classes:** {len(config.get('names', []))}")

                    # Count images
                    train_dir = Path(session["path"]) / "images" / "train"
                    val_dir = Path(session["path"]) / "images" / "val"
                    train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
                    val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0

                    st.write(f"**Train Images:** {train_count}")
                    st.write(f"**Val Images:** {val_count}")

                except Exception as e:
                    st.error(f"Error reading config: {e}")

                # Use for training button
                if st.button("Use for Training", key=f"train_{session['name']}"):
                    st.session_state["selected_dataset"] = session["path"]
                    st.info("Go to Training page to start training with this dataset")


def _render_annotation_history(task_manager: TaskManager) -> None:
    """Render annotation task history."""
    st.subheader("Annotation History")

    render_task_list(
        task_type="annotation",
        task_manager=task_manager,
        limit=10,
        show_active_only=False,
    )


# =============================================================================
# Copy-Paste Augmentation Tabs
# =============================================================================


def _get_mask_stats(path_coordinator: PathCoordinator) -> Dict[str, int]:
    """
    Get mask counts per class for synthetic generation.

    Returns:
        Dictionary mapping class_name to mask count
    """
    annotated_dir = path_coordinator.get_path("annotated_dir")
    stats = {}

    if not annotated_dir.exists():
        return stats

    for class_dir in annotated_dir.iterdir():
        if not class_dir.is_dir():
            continue

        masks_dir = class_dir / "masks"
        if masks_dir.exists():
            # Count both naming conventions
            mask_count = len(list(masks_dir.glob("*_mask.png")))
            if mask_count == 0:
                mask_count = len(list(masks_dir.glob("*.png")))
            if mask_count > 0:
                stats[class_dir.name] = mask_count

    return stats


def _generate_preview_images(
    path_coordinator: PathCoordinator,
    selected_classes: List[str],
    scale_range: tuple,
    enable_white_balance: bool,
    max_objects: int,
    seed: int = 42,
    num_samples: int = 3,
    rotation_range: tuple = (-15.0, 15.0),
    enable_horizontal_flip: bool = True,
    enable_vertical_flip: bool = False,
    white_balance_strength: float = 0.7,
    edge_blur_sigma: float = 2.0,
    min_objects: int = 1,
) -> List[tuple]:
    """
    Generate preview synthetic images.

    Args:
        path_coordinator: PathCoordinator instance
        selected_classes: List of class names to include
        scale_range: (min_scale, max_scale) tuple
        enable_white_balance: Whether to enable white balance adjustment
        max_objects: Maximum objects per image
        seed: Random seed for reproducibility
        num_samples: Number of preview images to generate
        rotation_range: (min_rotation, max_rotation) tuple in degrees
        enable_horizontal_flip: Whether to enable horizontal flipping
        enable_vertical_flip: Whether to enable vertical flipping
        white_balance_strength: Strength of white balance adjustment (0.0-1.0)
        edge_blur_sigma: Blur sigma for object edges
        min_objects: Minimum objects per image

    Returns:
        List of (image_rgb, class_names) tuples
    """
    import cv2
    import numpy as np
    from augmentation.copy_paste_augmentor import CopyPasteAugmentor, CopyPasteConfig

    config = CopyPasteConfig(
        scale_range=scale_range,
        rotation_range=rotation_range,
        enable_horizontal_flip=enable_horizontal_flip,
        enable_vertical_flip=enable_vertical_flip,
        enable_white_balance=enable_white_balance,
        white_balance_strength=white_balance_strength,
        edge_blur_sigma=edge_blur_sigma,
        max_objects_per_image=max_objects,
        min_objects_per_image=min_objects,
        seed=seed,
    )

    augmentor = CopyPasteAugmentor(config)
    augmentor.rng = np.random.RandomState(seed)

    # Load backgrounds
    backgrounds_dir = path_coordinator.get_path("backgrounds_dir")
    backgrounds = augmentor._load_images(backgrounds_dir)

    # Load objects from masks
    annotated_dir = path_coordinator.get_path("annotated_dir")
    objects, target_resolution = augmentor._load_objects_from_masks(
        annotated_dir=annotated_dir,
        class_names=selected_classes,
        alpha_blur_sigma=config.edge_blur_sigma,
    )

    if not backgrounds or not objects:
        return []

    previews = []
    for i in range(num_samples):
        # Select random background
        bg_idx = augmentor.rng.randint(0, len(backgrounds))
        bg_path = backgrounds[bg_idx]
        background = cv2.imread(str(bg_path))

        if background is None:
            continue

        # Resize background to match source image resolution
        if target_resolution is not None:
            target_h, target_w = target_resolution
            background = cv2.resize(
                background, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

        # Generate synthetic image
        result = augmentor.generate_synthetic_image(
            background=background,
            objects=objects,
        )

        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
        class_names = [pr.class_name for pr in result.paste_results]

        previews.append((image_rgb, class_names))

    return previews


def _render_generate_synthetic(path_coordinator: PathCoordinator) -> None:
    """Render synthetic image generation section."""
    st.subheader("Generate Synthetic Training Images")

    st.markdown("""
    Generate synthetic training images by pasting annotated objects onto background images.
    Uses masks from SAM2 annotation directly with edge blending and white balance adjustment.
    """)

    # Check for masks from annotation
    mask_stats = _get_mask_stats(path_coordinator)

    if not mask_stats:
        st.warning(
            "No annotated masks found. Please run annotation first using the 'Run Annotation' tab."
        )
        st.info("Masks are automatically saved during SAM2 annotation.")
        return

    # Check for backgrounds
    backgrounds = path_coordinator.get_background_images()

    if not backgrounds:
        st.warning(
            "No background images found. Please add backgrounds to the backgrounds directory."
        )
        backgrounds_dir = path_coordinator.get_path("backgrounds_dir")
        st.info(f"Add background images to: `{backgrounds_dir}`")

        # Show upload option
        _render_background_upload(path_coordinator)
        return

    # Object selection (based on available masks)
    st.markdown("##### Select Classes to Include")

    selected_classes = []
    cols = st.columns(3)
    for idx, (class_name, mask_count) in enumerate(sorted(mask_stats.items())):
        with cols[idx % 3]:
            if st.checkbox(f"{class_name} ({mask_count} masks)", value=True, key=f"synth_obj_{class_name}"):
                selected_classes.append(class_name)

    if not selected_classes:
        st.warning("Please select at least one class.")
        return

    st.markdown("---")

    # Generation settings
    st.markdown("##### Generation Settings")

    col1, col2 = st.columns(2)

    with col1:
        # Count real images
        real_count = 0
        raw_dir = path_coordinator.get_path("raw_captures_dir")
        for class_name in selected_classes:
            class_raw = raw_dir / class_name
            if class_raw.exists():
                real_count += len(list(class_raw.glob("*.jpg"))) + len(list(class_raw.glob("*.png")))

        synthetic_ratio = st.slider(
            "Synthetic:Real Ratio",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.5,
            help="Ratio of synthetic images to real images (e.g., 2.0 = 2:1)",
            key="synth_ratio"
        )

        target_count = int(real_count * synthetic_ratio)
        st.caption(f"Real images: {real_count} → Synthetic to generate: {target_count}")

    with col2:
        enable_white_balance = st.checkbox(
            "Enable White Balance",
            value=True,
            help="Adjust object colors to match background lighting",
            key="synth_wb"
        )

        max_objects = st.slider(
            "Max Objects per Image",
            min_value=1,
            max_value=5,
            value=3,
            help="Maximum number of objects to paste per synthetic image",
            key="synth_max_obj"
        )

    col3, col4 = st.columns(2)

    with col3:
        scale_min = st.slider(
            "Min Scale",
            min_value=0.3,
            max_value=1.0,
            value=0.5,
            step=0.1,
            key="synth_scale_min"
        )

    with col4:
        scale_max = st.slider(
            "Max Scale",
            min_value=0.5,
            max_value=2.0,
            value=1.5,
            step=0.1,
            key="synth_scale_max"
        )

    # Transform Settings
    st.markdown("##### Transform Settings")

    col5, col6 = st.columns(2)

    with col5:
        rotation_min = st.slider(
            "Min Rotation (°)",
            min_value=-180,
            max_value=0,
            value=-15,
            step=5,
            help="Minimum rotation angle in degrees",
            key="synth_rot_min"
        )

    with col6:
        rotation_max = st.slider(
            "Max Rotation (°)",
            min_value=0,
            max_value=180,
            value=15,
            step=5,
            help="Maximum rotation angle in degrees",
            key="synth_rot_max"
        )

    col7, col8 = st.columns(2)

    with col7:
        enable_horizontal_flip = st.checkbox(
            "Enable Horizontal Flip",
            value=True,
            help="Randomly flip objects horizontally",
            key="synth_hflip"
        )

    with col8:
        enable_vertical_flip = st.checkbox(
            "Enable Vertical Flip",
            value=False,
            help="Randomly flip objects vertically",
            key="synth_vflip"
        )

    # Appearance Settings
    st.markdown("##### Appearance Settings")

    col9, col10 = st.columns(2)

    with col9:
        if enable_white_balance:
            white_balance_strength = st.slider(
                "White Balance Strength",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Strength of white balance adjustment (0.0 = none, 1.0 = full)",
                key="synth_wb_strength"
            )
        else:
            white_balance_strength = 0.7  # default when WB disabled

    with col10:
        edge_blur_sigma = st.slider(
            "Edge Blur Sigma",
            min_value=0.0,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Blur strength for object edges (higher = smoother blending)",
            key="synth_edge_blur"
        )

    # Advanced Settings
    with st.expander("⚙️ Advanced Settings"):
        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            min_objects = st.slider(
                "Min Objects per Image",
                min_value=1,
                max_value=max_objects,
                value=1,
                help="Minimum number of objects to paste per synthetic image",
                key="synth_min_obj"
            )

            allow_overlap = st.checkbox(
                "Allow Object Overlap",
                value=False,
                help="Allow objects to overlap each other",
                key="synth_overlap"
            )

        with adv_col2:
            output_quality = st.slider(
                "JPEG Quality",
                min_value=50,
                max_value=100,
                value=95,
                step=5,
                help="Output JPEG quality (higher = better quality, larger files)",
                key="synth_jpeg_quality"
            )

            if allow_overlap:
                overlap_iou_threshold = st.slider(
                    "Overlap IoU Threshold",
                    min_value=0.0,
                    max_value=0.5,
                    value=0.1,
                    step=0.05,
                    help="Maximum allowed overlap between objects (IoU)",
                    key="synth_overlap_iou"
                )
            else:
                overlap_iou_threshold = 0.1  # default when overlap disabled

    st.markdown("---")

    # Background preview
    st.markdown(f"##### Backgrounds ({len(backgrounds)} available)")

    with st.expander("Preview Backgrounds", expanded=False):
        bg_cols = st.columns(4)
        for idx, bg in enumerate(backgrounds[:8]):
            with bg_cols[idx % 4]:
                try:
                    import cv2
                    img = cv2.imread(bg["path"])
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption=bg["name"], use_container_width=True)
                except Exception:
                    st.caption(bg["name"])

    # Add upload option
    _render_background_upload(path_coordinator)

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

    # Generate and display preview images
    try:
        previews = _generate_preview_images(
            path_coordinator=path_coordinator,
            selected_classes=selected_classes,
            scale_range=(scale_min, scale_max),
            enable_white_balance=enable_white_balance,
            max_objects=max_objects,
            seed=st.session_state["synth_preview_seed"],
            num_samples=3,
            rotation_range=(float(rotation_min), float(rotation_max)),
            enable_horizontal_flip=enable_horizontal_flip,
            enable_vertical_flip=enable_vertical_flip,
            white_balance_strength=white_balance_strength,
            edge_blur_sigma=edge_blur_sigma,
            min_objects=min_objects,
        )

        if previews:
            preview_cols = st.columns(3)
            for idx, (image, class_names) in enumerate(previews):
                with preview_cols[idx]:
                    st.image(image, use_container_width=True)
                    if class_names:
                        st.caption(f"Classes: {', '.join(class_names)}")
                    else:
                        st.caption("No objects placed")
        else:
            st.info("Could not generate preview. Check masks and backgrounds.")

    except Exception as e:
        st.warning(f"Preview generation failed: {str(e)}")

    st.markdown("---")

    # Generate button
    if st.button("🎨 Generate Synthetic Dataset", type="primary", use_container_width=True, key="synth_gen_btn"):
        try:
            from augmentation.copy_paste_augmentor import CopyPasteAugmentor, CopyPasteConfig

            # Use the same seed from preview for reproducibility
            batch_seed = st.session_state.get("synth_preview_seed", 42)

            config = CopyPasteConfig(
                synthetic_to_real_ratio=synthetic_ratio,
                scale_range=(scale_min, scale_max),
                rotation_range=(float(rotation_min), float(rotation_max)),
                enable_horizontal_flip=enable_horizontal_flip,
                enable_vertical_flip=enable_vertical_flip,
                enable_white_balance=enable_white_balance,
                white_balance_strength=white_balance_strength,
                edge_blur_sigma=edge_blur_sigma,
                max_objects_per_image=max_objects,
                min_objects_per_image=min_objects,
                allow_overlap=allow_overlap,
                overlap_iou_threshold=overlap_iou_threshold,
                output_quality=output_quality,
                seed=batch_seed,
            )

            augmentor = CopyPasteAugmentor(config)

            # Create output session
            output_dir = path_coordinator.get_synthetic_session_dir()

            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current: int, total: int, message: str):
                progress_bar.progress(current / total)
                status_text.text(message)

            # Generate batch
            annotated_dir = path_coordinator.get_path("annotated_dir")
            backgrounds_dir = path_coordinator.get_path("backgrounds_dir")

            stats = augmentor.generate_batch(
                backgrounds_dir=backgrounds_dir,
                annotated_dir=annotated_dir,
                output_dir=output_dir,
                real_image_count=real_count,
                class_names=selected_classes,
                progress_callback=progress_callback,
            )

            progress_bar.progress(1.0)
            status_text.empty()

            if "error" in stats:
                st.error(stats["error"])
            else:
                # Save generation config for reproducibility
                config_path = augmentor.save_generation_config(
                    output_dir=output_dir,
                    additional_info={
                        "class_names": selected_classes,
                        "real_image_count": real_count,
                        "backgrounds_dir": str(backgrounds_dir),
                        "annotated_dir": str(annotated_dir),
                        "stats": stats,
                    },
                )

                st.success(
                    f"Generated {stats['generated']} synthetic images. "
                    f"Failed: {stats.get('failed', 0)}"
                )
                st.info(f"Output saved to: `{output_dir}`")
                st.caption(f"Config saved: `{config_path.name}`")

                # Per-class stats
                if stats.get("per_class"):
                    st.markdown("**Objects per class:**")
                    for cls, count in stats["per_class"].items():
                        st.caption(f"- {cls}: {count}")

        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def _render_background_upload(path_coordinator: PathCoordinator) -> None:
    """Render background image upload widget."""
    with st.expander("➕ Add Background Image", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload Background Image",
            type=["jpg", "jpeg", "png"],
            help="Upload a background image for synthetic data generation",
            key="bg_upload"
        )

        if uploaded_file is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                saved_path = path_coordinator.add_background_image(tmp_path, uploaded_file.name)
                st.success(f"Background saved: {uploaded_file.name}")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to save background: {str(e)}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)


# For Streamlit native multipage
if __name__ == "__main__":
    show_annotation_page()
