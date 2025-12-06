"""
Annotation Page

Provides UI for running auto-annotation pipeline on collected images.
Integrates with scripts/annotation/auto_annotate.py via TaskManager.
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


def show_annotation_page():
    """Main annotation page."""
    st.title("🏷️ Auto-Annotation")

    # Initialize services
    task_manager = TaskManager()
    path_coordinator = PathCoordinator()

    # Check for active annotation task
    active_task = render_active_task_banner("annotation", task_manager)

    if active_task:
        st.markdown("---")
        st.subheader("Running Task")
        render_task_progress(active_task.task_id, task_manager)
        return

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Run Annotation", "Sessions", "History"])

    with tab1:
        _render_run_annotation(task_manager, path_coordinator)

    with tab2:
        _render_annotation_sessions(path_coordinator)

    with tab3:
        _render_annotation_history(task_manager)


def _render_run_annotation(task_manager: TaskManager, path_coordinator: PathCoordinator):
    """Render annotation configuration and run section."""
    st.subheader("Configure Annotation Pipeline")

    # Check prerequisites
    raw_captures_dir = path_coordinator.get_path("raw_captures_dir")
    class_config_file = path_coordinator.get_path("class_config_file")

    # Count available images
    image_count = 0
    class_dirs = []
    if raw_captures_dir.exists():
        for class_dir in raw_captures_dir.iterdir():
            if class_dir.is_dir():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if images:
                    class_dirs.append((class_dir.name, len(images)))
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

    # Show available data
    st.success(f"Found **{image_count}** images across **{len(class_dirs)}** classes")

    with st.expander("View Class Distribution", expanded=False):
        for class_name, count in sorted(class_dirs):
            st.write(f"  • **{class_name}**: {count} images")

    st.markdown("---")

    # Annotation method selection
    col1, col2 = st.columns(2)

    with col1:
        method = st.radio(
            "Annotation Method",
            ["background", "sam2"],
            format_func=lambda x: {
                "background": "Background Subtraction (Fast)",
                "sam2": "SAM2 Segmentation (Accurate, GPU)"
            }[x],
            help="Background subtraction is faster but requires a background image. "
                 "SAM2 is more accurate but requires GPU."
        )

    with col2:
        train_val_split = st.slider(
            "Train/Val Split",
            min_value=0.5,
            max_value=0.95,
            value=0.85,
            step=0.05,
            help="Ratio of images to use for training (rest for validation)"
        )

    # Background image selection (for background method)
    background_path = None
    if method == "background":
        st.markdown("---")
        st.subheader("Background Image")

        # List available backgrounds
        backgrounds = path_coordinator.get_background_images()

        if backgrounds:
            bg_options = ["Upload new..."] + [bg["name"] for bg in backgrounds]
            selected_bg = st.selectbox("Select Background", bg_options)

            if selected_bg != "Upload new...":
                background_path = next(
                    (bg["path"] for bg in backgrounds if bg["name"] == selected_bg),
                    None
                )
                if background_path:
                    st.image(background_path, caption="Selected background", width=300)

        # Upload option
        if not backgrounds or selected_bg == "Upload new...":
            uploaded_bg = st.file_uploader(
                "Upload Background Image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clean background image (same setup as object captures, but without objects)"
            )

            if uploaded_bg:
                # Save uploaded background
                bg_save_path = path_coordinator.add_background_image(
                    uploaded_bg,
                    name=uploaded_bg.name
                )
                background_path = bg_save_path
                st.success(f"Saved background: {bg_save_path}")

        if not background_path:
            st.warning("Please select or upload a background image")

    # Advanced options
    with st.expander("Advanced Options", expanded=False):
        min_area = st.number_input(
            "Minimum Contour Area",
            min_value=100,
            max_value=5000,
            value=500,
            help="Minimum pixel area for detected contours (background method only)"
        )

    # Session name
    st.markdown("---")
    session_name = st.text_input(
        "Session Name (optional)",
        placeholder="Leave empty for auto-generated name",
        help="Name for this annotation session"
    )

    # Run button
    st.markdown("---")

    can_run = True
    if method == "background" and not background_path:
        can_run = False
        st.error("Please select a background image")

    if not class_config_file.exists():
        can_run = False
        st.error(f"Class config not found: {class_config_file}")

    if can_run:
        # Estimated time
        estimated_time = (image_count / 100) * 5  # ~5 minutes per 100 images
        st.info(f"Estimated time: ~{estimated_time:.0f} minutes for {image_count} images")

        if st.button("Start Annotation", type="primary"):
            # Create session
            session_paths = path_coordinator.create_annotation_session(
                session_name if session_name else None
            )

            # Start task
            task_id = task_manager.start_annotation(
                method=method,
                input_dir=session_paths["input_dir"],
                output_dir=session_paths["output_dir"],
                class_config=session_paths["class_config"],
                background_path=background_path,
                train_val_split=train_val_split,
                min_area=min_area,
            )

            st.success(f"Annotation started! Task ID: {task_id}")
            st.rerun()


def _render_annotation_sessions(path_coordinator: PathCoordinator):
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


def _render_annotation_history(task_manager: TaskManager):
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
