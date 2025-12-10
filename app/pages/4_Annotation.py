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
from typing import Any

import streamlit as st

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

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

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Run Annotation",
        "📦 Prepare Dataset",
        "📁 Sessions",
        "📜 History"
    ])

    with tab1:
        _render_run_annotation(task_manager, path_coordinator)

    with tab2:
        _render_prepare_dataset(path_coordinator)

    with tab3:
        _render_annotation_sessions(path_coordinator)

    with tab4:
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


# For Streamlit native multipage
if __name__ == "__main__":
    show_annotation_page()
