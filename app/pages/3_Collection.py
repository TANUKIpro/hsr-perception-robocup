"""
Collection Page

Provides UI for data collection from multiple sources.
"""

import streamlit as st
from pathlib import Path
import sys

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar
from components.ros2_collection import render_ros2_collection
from components.captured_images_tree import render_captured_images_tree
from components.video_extractor import render_video_extractor


def show_collection_page():
    """Collection page for capturing/importing images."""
    render_common_sidebar()

    st.title("📸 Data Collection")

    registry = st.session_state.registry
    path_coordinator = st.session_state.path_coordinator

    # Update collection counts from filesystem before displaying
    registry.update_all_collection_counts()

    objects = registry.get_all_objects()

    if not objects:
        st.warning("No objects registered. Go to Registry first.")
        return

    # Object selection
    obj, selected_id = _render_object_selector(objects)

    # Show current status
    _render_collection_status(obj)

    # Reference images
    _render_reference_images(registry, selected_id)

    st.markdown("---")

    # Collection methods - with ROS2 tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "🤖 ROS2 Camera",
        "📷 Local Camera",
        "📁 File Upload",
        "📂 Folder Import"
    ])

    with tab1:
        render_ros2_collection(obj, registry, path_coordinator)
        render_captured_images_tree(path_coordinator)
        render_video_extractor(path_coordinator)

    with tab2:
        _render_local_camera_tab(registry, selected_id)

    with tab3:
        _render_file_upload_tab(registry, selected_id)

    with tab4:
        _render_folder_import_tab(registry, selected_id)


def _render_object_selector(objects: list):
    """Render object selector and return selected object."""
    object_options = {
        f"{obj.id}. {obj.display_name} ({obj.collected_samples}/{obj.target_samples})": obj.id
        for obj in objects
    }

    # Pre-select if coming from registry
    default_idx = 0
    if st.session_state.current_object_id:
        for i, obj_id in enumerate(object_options.values()):
            if obj_id == st.session_state.current_object_id:
                default_idx = i
                break

    selected_label = st.selectbox("Select Object", list(object_options.keys()), index=default_idx)
    selected_id = object_options[selected_label]
    obj = st.session_state.registry.get_object(selected_id)

    return obj, selected_id


def _render_collection_status(obj):
    """Render current collection status metrics."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collected", obj.collected_samples)
    with col2:
        st.metric("Target", obj.target_samples)
    with col3:
        pct = (obj.collected_samples / obj.target_samples * 100) if obj.target_samples > 0 else 0
        st.metric("Progress", f"{pct:.1f}%")

    st.progress(min(pct / 100, 1.0))


def _render_reference_images(registry, selected_id: int):
    """Render reference images section."""
    ref_images = registry.get_reference_images(selected_id)
    if ref_images:
        st.subheader("Reference Images")
        cols = st.columns(min(len(ref_images), 4))
        for i, img_path in enumerate(ref_images[:4]):
            with cols[i]:
                st.image(img_path, width=150)


def _render_local_camera_tab(registry, selected_id: int):
    """Render local camera capture tab."""
    st.subheader("Camera Capture")
    st.write("Use your device camera to capture images.")

    # Initialize camera state
    camera_key = f"camera_enabled_{selected_id}"
    if camera_key not in st.session_state:
        st.session_state[camera_key] = False

    # Toggle button to enable/disable camera
    if not st.session_state[camera_key]:
        st.info("Click the button below to start the camera.")
        if st.button("📷 Start Camera", key="start_camera"):
            st.session_state[camera_key] = True
            st.rerun()
    else:
        if st.button("⏹️ Stop Camera", key="stop_camera"):
            st.session_state[camera_key] = False
            st.rerun()

        camera_image = st.camera_input("Take a photo")

        if camera_image:
            image_bytes = camera_image.getvalue()
            saved_path = registry.save_collected_image(selected_id, image_bytes, ".jpg")

            if saved_path:
                st.success(f"Saved: {Path(saved_path).name}")
                st.image(camera_image, width=300)
                new_count = registry.update_collection_count(selected_id)
                st.write(f"Total collected: {new_count}")


def _render_file_upload_tab(registry, selected_id: int):
    """Render file upload tab."""
    st.subheader("File Upload")
    st.write("Upload one or more images.")

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="file_upload"
    )

    if uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, uploaded in enumerate(uploaded_files):
            temp_path = f"/tmp/{uploaded.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded.read())

            registry.add_collected_image(selected_id, temp_path)

            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.write(f"Processing: {uploaded.name}")

        status_text.write(f"Uploaded {len(uploaded_files)} images")
        st.rerun()


def _render_folder_import_tab(registry, selected_id: int):
    """Render folder import tab."""
    st.subheader("Folder Import")
    st.write("Import all images from a folder path.")

    folder_path = st.text_input("Folder Path", placeholder="/path/to/images")

    if st.button("Import from Folder") and folder_path:
        folder = Path(folder_path)

        if not folder.exists():
            st.error("Folder not found")
        else:
            extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

            if not images:
                st.warning("No images found in folder")
            else:
                progress_bar = st.progress(0)

                for i, img_path in enumerate(images):
                    registry.add_collected_image(selected_id, str(img_path))
                    progress_bar.progress((i + 1) / len(images))

                st.success(f"Imported {len(images)} images")
                st.rerun()


# For Streamlit native multipage
if __name__ == "__main__":
    show_collection_page()
