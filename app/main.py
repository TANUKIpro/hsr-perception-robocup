"""
HSR Perception - Object Manager

Streamlit application for managing object registration, data collection,
and training preparation for RoboCup@Home competitions.

Features:
- Object registration with reference images
- Multi-source data collection (ROS2, camera, file upload)
- Auto-annotation pipeline integration
- YOLOv8 fine-tuning management
- Model evaluation and visual verification

Usage:
    streamlit run app/main.py
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="HSR Object Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
import base64
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Optional: clipboard paste support
try:
    from streamlit_paste_button import paste_image_button
    PASTE_BUTTON_AVAILABLE = True
except ImportError:
    PASTE_BUTTON_AVAILABLE = False

from object_registry import ObjectRegistry, RegisteredObject, ObjectProperties
from services.path_coordinator import PathCoordinator
from services.task_manager import TaskManager


# Initialize session state
if "registry" not in st.session_state:
    st.session_state.registry = ObjectRegistry()

if "current_object_id" not in st.session_state:
    st.session_state.current_object_id = None

if "path_coordinator" not in st.session_state:
    st.session_state.path_coordinator = PathCoordinator()

if "task_manager" not in st.session_state:
    st.session_state.task_manager = TaskManager()


def main():
    """Main application entry point."""

    # Sidebar navigation
    st.sidebar.title("🤖 HSR Object Manager")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "📋 Registry",
            "📸 Collection",
            "🏷️ Annotation",
            "🎓 Training",
            "📈 Evaluation",
            "⚙️ Settings"
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Quick stats in sidebar
    registry = st.session_state.registry
    stats = registry.get_collection_stats()

    st.sidebar.metric("Total Objects", stats["total_objects"])
    st.sidebar.metric(
        "Collection Progress",
        f"{stats['total_collected']}/{stats['total_target']}",
        f"{stats['progress_percent']:.1f}%"
    )

    # Active task indicator
    task_manager = st.session_state.task_manager
    active_tasks = task_manager.get_active_tasks()
    if active_tasks:
        st.sidebar.markdown("---")
        st.sidebar.warning(f"🔄 {len(active_tasks)} task(s) running")

    # Route to pages
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "📋 Registry":
        show_registry()
    elif page == "📸 Collection":
        show_collection()
    elif page == "🏷️ Annotation":
        from pages import show_annotation_page
        show_annotation_page()
    elif page == "🎓 Training":
        from pages import show_training_page
        show_training_page()
    elif page == "📈 Evaluation":
        from pages import show_evaluation_page
        show_evaluation_page()
    elif page == "⚙️ Settings":
        show_settings()


def show_dashboard():
    """Dashboard page with overview statistics."""
    st.title("📊 Dashboard")

    registry = st.session_state.registry
    stats = registry.get_collection_stats()
    objects = registry.get_all_objects()

    # Overall progress
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Objects", stats["total_objects"])
    with col2:
        st.metric("Images Collected", stats["total_collected"])
    with col3:
        st.metric("Target Total", stats["total_target"])
    with col4:
        ready_pct = (stats["ready_objects"] / stats["total_objects"] * 100) if stats["total_objects"] > 0 else 0
        st.metric("Ready for Training", f"{stats['ready_objects']}/{stats['total_objects']}", f"{ready_pct:.0f}%")

    st.markdown("---")

    # Pipeline status
    st.subheader("Pipeline Status")

    task_manager = st.session_state.task_manager
    path_coordinator = st.session_state.path_coordinator

    col1, col2, col3 = st.columns(3)

    with col1:
        # Annotation status
        annotation_sessions = path_coordinator.get_annotation_sessions()
        ready_datasets = [s for s in annotation_sessions if s["has_data_yaml"]]
        st.metric("Annotated Datasets", len(ready_datasets))

    with col2:
        # Training status
        trained_models = path_coordinator.get_trained_models()
        st.metric("Trained Models", len(trained_models))

    with col3:
        # Active tasks
        active_tasks = task_manager.get_active_tasks()
        st.metric("Active Tasks", len(active_tasks))

    st.markdown("---")

    # Progress by category
    st.subheader("Progress by Category")

    if stats["by_category"]:
        cols = st.columns(len(stats["by_category"]))
        for i, (cat, cat_stats) in enumerate(stats["by_category"].items()):
            with cols[i]:
                pct = (cat_stats["collected"] / cat_stats["target"] * 100) if cat_stats["target"] > 0 else 0
                st.metric(cat, f"{cat_stats['collected']}/{cat_stats['target']}")
                st.progress(min(pct / 100, 1.0))

    st.markdown("---")

    # Per-object progress
    st.subheader("Collection Progress by Object")

    if objects:
        for obj in objects:
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                pct = (obj.collected_samples / obj.target_samples) if obj.target_samples > 0 else 0
                if pct >= 1.0:
                    status = "🟢"
                elif pct >= 0.5:
                    status = "🟡"
                else:
                    status = "🔴"

                st.write(f"{status} **{obj.display_name}** ({obj.category})")
                st.progress(min(pct, 1.0))

            with col2:
                st.write(f"{obj.collected_samples}/{obj.target_samples}")

            with col3:
                badges = []
                if obj.properties.is_heavy:
                    badges.append("Heavy")
                if obj.properties.is_tiny:
                    badges.append("Tiny")
                if obj.properties.has_liquid:
                    badges.append("Liquid")
                st.write(", ".join(badges) if badges else "-")
    else:
        st.info("No objects registered yet. Go to Registry to add objects.")

    # Training readiness check
    st.markdown("---")
    st.subheader("Training Readiness")

    ready_count = sum(1 for obj in objects if obj.collected_samples >= 50)
    total_count = len(objects)

    if total_count == 0:
        st.warning("No objects registered.")
    elif ready_count == total_count:
        st.success(f"All {total_count} objects have sufficient data for training!")
        if st.button("Export to YOLO Config"):
            output_path = registry.export_to_yolo_config("config/object_classes.json")
            st.success(f"Exported to {output_path}")
    else:
        st.warning(f"{total_count - ready_count} objects need more data (minimum 50 images each)")

        need_data = [obj for obj in objects if obj.collected_samples < 50]
        for obj in need_data[:5]:
            st.write(f"  • {obj.display_name}: {obj.collected_samples}/50 minimum")


def show_registry():
    """Registry page for managing objects."""
    st.title("📋 Object Registry")

    registry = st.session_state.registry

    # Tabs for view/add
    tab1, tab2 = st.tabs(["View Objects", "Add New Object"])

    with tab1:
        objects = registry.get_all_objects()

        if not objects:
            st.info("No objects registered yet. Use 'Add New Object' tab to add objects.")
        else:
            # Filter by category
            categories = ["All"] + registry.categories
            selected_category = st.selectbox("Filter by Category", categories)

            if selected_category != "All":
                objects = [obj for obj in objects if obj.category == selected_category]

            # Display objects
            for obj in objects:
                edit_key = f"edit_mode_{obj.id}"
                if edit_key not in st.session_state:
                    st.session_state[edit_key] = False

                with st.expander(f"**{obj.id}. {obj.display_name}** - {obj.category}", expanded=False):
                    # Check if in edit mode
                    if st.session_state[edit_key]:
                        # === EDIT MODE ===
                        st.markdown("### Edit Object")

                        # Thumbnail upload section - use session state to prevent infinite rerun
                        thumb_processed_key = f"thumb_processed_{obj.id}"
                        paste_processed_key = f"paste_processed_{obj.id}"
                        if thumb_processed_key not in st.session_state:
                            st.session_state[thumb_processed_key] = None
                        if paste_processed_key not in st.session_state:
                            st.session_state[paste_processed_key] = False

                        thumb_col1, thumb_col2, thumb_col3 = st.columns([1, 1, 1])
                        with thumb_col1:
                            thumbnail_path = registry.get_thumbnail_path(obj.id)
                            if thumbnail_path:
                                st.image(thumbnail_path, caption="Current Thumbnail", width=120)
                            else:
                                st.info("No thumbnail")

                        with thumb_col2:
                            thumb_upload = st.file_uploader(
                                "Upload thumbnail",
                                type=["jpg", "jpeg", "png"],
                                key=f"thumb_edit_{obj.id}"
                            )
                            if thumb_upload:
                                # Create unique ID for this file to detect new uploads
                                file_id = f"{thumb_upload.name}_{thumb_upload.size}"
                                if st.session_state[thumb_processed_key] != file_id:
                                    temp_path = f"/tmp/thumb_{thumb_upload.name}"
                                    with open(temp_path, "wb") as f:
                                        f.write(thumb_upload.read())
                                    registry.set_thumbnail(obj.id, temp_path)
                                    st.session_state[thumb_processed_key] = file_id
                                    st.success("Thumbnail updated")
                                    st.rerun()

                        with thumb_col3:
                            if PASTE_BUTTON_AVAILABLE:
                                paste_result = paste_image_button(
                                    label="Paste from clipboard",
                                    key=f"paste_thumb_{obj.id}"
                                )
                                if paste_result.image_data is not None and not st.session_state[paste_processed_key]:
                                    # Save pasted image
                                    registry.save_thumbnail_from_bytes(
                                        obj.id,
                                        paste_result.image_data,
                                        ".png"
                                    )
                                    st.session_state[paste_processed_key] = True
                                    st.success("Thumbnail pasted")
                                    st.rerun()
                            else:
                                st.caption("Install streamlit-paste-button for clipboard paste")

                        st.markdown("---")

                        # Editable fields
                        edit_col1, edit_col2 = st.columns(2)

                        with edit_col1:
                            edit_name = st.text_input(
                                "Name (lowercase, no spaces)",
                                value=obj.name,
                                key=f"edit_name_{obj.id}"
                            )
                            edit_display_name = st.text_input(
                                "Display Name",
                                value=obj.display_name,
                                key=f"edit_display_{obj.id}"
                            )
                            edit_category = st.selectbox(
                                "Category",
                                registry.categories,
                                index=registry.categories.index(obj.category) if obj.category in registry.categories else 0,
                                key=f"edit_cat_{obj.id}"
                            )

                        with edit_col2:
                            edit_target = st.number_input(
                                "Target Samples",
                                min_value=10,
                                value=obj.target_samples,
                                key=f"edit_target_{obj.id}"
                            )
                            edit_remarks = st.text_area(
                                "Remarks",
                                value=obj.remarks or "",
                                key=f"edit_remarks_{obj.id}"
                            )

                        st.write("**Properties:**")
                        prop_col1, prop_col2, prop_col3 = st.columns(3)
                        with prop_col1:
                            edit_heavy = st.checkbox("Heavy Item", value=obj.properties.is_heavy, key=f"edit_heavy_{obj.id}")
                        with prop_col2:
                            edit_tiny = st.checkbox("Tiny Item", value=obj.properties.is_tiny, key=f"edit_tiny_{obj.id}")
                        with prop_col3:
                            edit_liquid = st.checkbox("Has Liquid", value=obj.properties.has_liquid, key=f"edit_liquid_{obj.id}")

                        edit_size = st.text_input(
                            "Size (cm)",
                            value=obj.properties.size_cm or "",
                            key=f"edit_size_{obj.id}"
                        )

                        # Save/Cancel buttons
                        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                        with btn_col1:
                            if st.button("Save", key=f"save_{obj.id}", type="primary"):
                                # Validate name
                                new_name = edit_name.lower().replace(" ", "_")
                                if new_name != obj.name:
                                    existing = registry.get_object_by_name(new_name)
                                    if existing and existing.id != obj.id:
                                        st.error(f"Name '{new_name}' already exists")
                                        st.stop()

                                # Build updates
                                updates = {
                                    'name': new_name,
                                    'display_name': edit_display_name,
                                    'category': edit_category,
                                    'target_samples': edit_target,
                                    'remarks': edit_remarks,
                                    'properties': {
                                        'is_heavy': edit_heavy,
                                        'is_tiny': edit_tiny,
                                        'has_liquid': edit_liquid,
                                        'size_cm': edit_size if edit_size else None,
                                    }
                                }
                                registry.update_object(obj.id, updates)
                                # Clear edit mode and processed states
                                st.session_state[edit_key] = False
                                st.session_state[thumb_processed_key] = None
                                st.session_state[paste_processed_key] = False
                                st.success("Object updated")
                                st.rerun()

                        with btn_col2:
                            if st.button("Cancel", key=f"cancel_{obj.id}"):
                                # Clear edit mode and processed states
                                st.session_state[edit_key] = False
                                st.session_state[thumb_processed_key] = None
                                st.session_state[paste_processed_key] = False
                                st.rerun()

                    else:
                        # === VIEW MODE ===
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            # Display thumbnail
                            thumbnail_path = registry.get_thumbnail_path(obj.id)
                            if thumbnail_path:
                                st.image(thumbnail_path, caption="Thumbnail", width=120)

                            # Reference images
                            ref_images = registry.get_reference_images(obj.id)
                            if ref_images:
                                st.write("**Reference Images:**")
                                img_cols = st.columns(min(len(ref_images), 3))
                                for i, img_path in enumerate(ref_images[:3]):
                                    with img_cols[i]:
                                        st.image(img_path, width=100)
                            else:
                                st.write("No reference images")

                            uploaded = st.file_uploader(
                                "Add reference image",
                                type=["jpg", "jpeg", "png"],
                                key=f"ref_upload_{obj.id}"
                            )
                            if uploaded:
                                temp_path = f"/tmp/{uploaded.name}"
                                with open(temp_path, "wb") as f:
                                    f.write(uploaded.read())
                                version = len(obj.versions) + 1
                                registry.add_reference_image(obj.id, temp_path, version)
                                st.success(f"Added reference image v{version}")
                                st.rerun()

                        with col2:
                            st.write(f"**Name:** {obj.name}")
                            st.write(f"**Category:** {obj.category}")
                            st.write(f"**Target Samples:** {obj.target_samples}")
                            st.write(f"**Collected:** {obj.collected_samples}")

                            if obj.remarks:
                                st.write(f"**Remarks:** {obj.remarks}")

                            props = []
                            if obj.properties.is_heavy:
                                props.append("Heavy Item")
                            if obj.properties.is_tiny:
                                props.append("Tiny Item")
                            if obj.properties.has_liquid:
                                props.append("Has Liquid")
                            if obj.properties.size_cm:
                                props.append(f"Size: {obj.properties.size_cm}")
                            if props:
                                st.write(f"**Properties:** {', '.join(props)}")

                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                if st.button("📸 Collect", key=f"collect_{obj.id}"):
                                    st.session_state.current_object_id = obj.id
                                    st.info("Please navigate to Collection page")
                            with col_b:
                                if st.button("✏️ Edit", key=f"edit_btn_{obj.id}"):
                                    st.session_state[edit_key] = True
                                    st.rerun()
                            with col_c:
                                if st.button("🗑️ Delete", key=f"delete_{obj.id}"):
                                    registry.remove_object(obj.id)
                                    st.rerun()

    with tab2:
        st.subheader("Add New Object")

        # Thumbnail upload (outside form for proper file handling)
        st.write("**Thumbnail (optional):**")

        # Initialize session state for pasted thumbnail
        if "new_obj_pasted_thumb" not in st.session_state:
            st.session_state.new_obj_pasted_thumb = None

        thumb_col1, thumb_col2, thumb_col3 = st.columns([1, 1, 1])
        with thumb_col1:
            new_thumbnail = st.file_uploader(
                "Upload thumbnail",
                type=["jpg", "jpeg", "png"],
                key="new_obj_thumbnail",
                help="Upload a thumbnail image for this object"
            )
        with thumb_col2:
            if PASTE_BUTTON_AVAILABLE:
                paste_result = paste_image_button(
                    label="Paste from clipboard",
                    key="paste_new_thumb"
                )
                if paste_result.image_data is not None:
                    st.session_state.new_obj_pasted_thumb = paste_result.image_data
            else:
                st.caption("Install streamlit-paste-button for clipboard paste")

        with thumb_col3:
            if new_thumbnail:
                st.image(new_thumbnail, caption="Preview (uploaded)", width=120)
            elif st.session_state.new_obj_pasted_thumb:
                st.image(st.session_state.new_obj_pasted_thumb, caption="Preview (pasted)", width=120)
            else:
                st.info("No thumbnail")

        st.markdown("---")

        with st.form("add_object_form"):
            col1, col2 = st.columns(2)

            with col1:
                new_id = st.number_input("ID", min_value=1, value=registry.get_next_id())
                new_name = st.text_input("Name (lowercase, no spaces)", placeholder="redbull")
                new_display_name = st.text_input("Display Name", placeholder="Redbull")
                new_category = st.selectbox("Category", registry.categories)

            with col2:
                new_target = st.number_input("Target Samples", min_value=10, value=100)
                new_remarks = st.text_area("Remarks", placeholder="Additional notes...")

                st.write("**Properties:**")
                is_heavy = st.checkbox("Heavy Item")
                is_tiny = st.checkbox("Tiny Item")
                has_liquid = st.checkbox("Has Liquid")
                size_cm = st.text_input("Size (cm)", placeholder="1.6x1.6x1.6")

            submitted = st.form_submit_button("Add Object")

            if submitted:
                if not new_name:
                    st.error("Name is required")
                elif registry.get_object_by_name(new_name):
                    st.error(f"Object with name '{new_name}' already exists")
                else:
                    obj = RegisteredObject(
                        id=new_id,
                        name=new_name.lower().replace(" ", "_"),
                        display_name=new_display_name or new_name,
                        category=new_category,
                        target_samples=new_target,
                        remarks=new_remarks,
                        properties=ObjectProperties(
                            is_heavy=is_heavy,
                            is_tiny=is_tiny,
                            has_liquid=has_liquid,
                            size_cm=size_cm if size_cm else None,
                        ),
                    )
                    registry.add_object(obj)

                    # Save thumbnail if uploaded or pasted
                    if new_thumbnail:
                        temp_path = f"/tmp/thumb_{new_thumbnail.name}"
                        with open(temp_path, "wb") as f:
                            f.write(new_thumbnail.getvalue())
                        registry.set_thumbnail(obj.id, temp_path)
                    elif st.session_state.new_obj_pasted_thumb:
                        registry.save_thumbnail_from_bytes(
                            obj.id,
                            st.session_state.new_obj_pasted_thumb,
                            ".png"
                        )
                        st.session_state.new_obj_pasted_thumb = None

                    st.success(f"Added object: {new_display_name}")
                    st.rerun()

        # Bulk import from iHR list
        st.markdown("---")
        st.subheader("Quick Import: iHR Object List")

        if st.button("Import iHR Standard Objects"):
            ihr_objects = [
                ("noodles", "Noodles", "Food", False, False, False, None, ""),
                ("tea_bag", "Tea Bag", "Food", False, False, False, None, ""),
                ("potato_chips", "Potato Chips", "Food", False, False, False, None, ""),
                ("gummy", "Gummy", "Food", False, False, False, None, ""),
                ("redbull", "Redbull", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
                ("aquarius", "Aquarius", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
                ("lychee", "Lychee", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
                ("coffee", "Coffee", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
                ("detergent", "Detergent", "Kitchen Item", False, False, False, None, "Without content"),
                ("cup", "Cup", "Kitchen Item", False, False, False, None, ""),
                ("lunch_box", "Lunch Box", "Kitchen Item", False, False, False, None, ""),
                ("bowl", "Bowl", "Kitchen Item", False, False, False, None, ""),
                ("dice", "Dice", "Task Item", False, True, False, "1.6x1.6x1.6", "Tiny Item"),
                ("light_bulb", "Light Bulb", "Task Item", False, False, False, None, "Without content"),
                ("block", "Block", "Task Item", False, False, False, None, ""),
                ("glue_gun", "Glue Gun", "Task Item", False, False, False, None, "Without plastic container"),
                ("shopping_bag", "Shopping Bag", "Bag", False, False, False, None, ""),
            ]

            count = 0
            for i, (name, display, cat, heavy, tiny, liquid, size, remarks) in enumerate(ihr_objects, 1):
                if not registry.get_object_by_name(name):
                    obj = RegisteredObject(
                        id=registry.get_next_id(),
                        name=name,
                        display_name=display,
                        category=cat,
                        target_samples=100,
                        remarks=remarks,
                        properties=ObjectProperties(
                            is_heavy=heavy,
                            is_tiny=tiny,
                            has_liquid=liquid,
                            size_cm=size,
                        ),
                    )
                    registry.add_object(obj)
                    count += 1

            st.success(f"Imported {count} objects from iHR list")
            st.rerun()


def show_collection():
    """Collection page for capturing/importing images."""
    st.title("📸 Data Collection")

    registry = st.session_state.registry
    objects = registry.get_all_objects()
    path_coordinator = st.session_state.path_coordinator

    if not objects:
        st.warning("No objects registered. Go to Registry first.")
        return

    # Object selection
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
    obj = registry.get_object(selected_id)

    # Show current status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collected", obj.collected_samples)
    with col2:
        st.metric("Target", obj.target_samples)
    with col3:
        pct = (obj.collected_samples / obj.target_samples * 100) if obj.target_samples > 0 else 0
        st.metric("Progress", f"{pct:.1f}%")

    st.progress(min(pct / 100, 1.0))

    # Reference images
    ref_images = registry.get_reference_images(selected_id)
    if ref_images:
        st.subheader("Reference Images")
        cols = st.columns(min(len(ref_images), 4))
        for i, img_path in enumerate(ref_images[:4]):
            with cols[i]:
                st.image(img_path, width=150)

    st.markdown("---")

    # Collection methods - with ROS2 tab
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 ROS2 Camera", "📷 Local Camera", "📁 File Upload", "📂 Folder Import"])

    with tab1:
        _show_ros2_collection_tab(obj, registry, path_coordinator)

    with tab2:
        st.subheader("Camera Capture")
        st.write("Use your device camera to capture images.")

        # Initialize camera state - only show camera when explicitly enabled
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

    with tab3:
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

    with tab4:
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


def _show_ros2_collection_tab(obj, registry, path_coordinator):
    """Show ROS2 collection tab content."""
    st.subheader("ROS2 Camera Capture")

    try:
        from services.ros2_bridge import ROS2Bridge
        ros2_bridge = ROS2Bridge()
    except ImportError:
        st.error("ROS2 bridge not available")
        return

    # Check ROS2 availability
    if not ros2_bridge.is_available():
        st.warning(
            "ROS2 is not available. Make sure:\n"
            "1. ROS2 Humble is installed\n"
            "2. The ROS2 environment is sourced\n"
            "3. The capture node is running"
        )

        st.code(
            "# HSR capture node:\n"
            "ros2 launch hsr_perception capture.launch.py\n\n"
            "# Xtion camera (see docs/xtion_setup.md):\n"
            "ros2 launch openni2_camera camera_only.launch.py",
            language="bash"
        )
        return

    st.success("ROS2 connected")

    # Check capture node
    if ros2_bridge.check_capture_node_running():
        st.success("Capture node is running")
    else:
        st.warning("Capture node not detected. Start it with:")
        st.code(
            "# HSR capture node:\n"
            "ros2 launch hsr_perception capture.launch.py\n\n"
            "# Xtion camera:\n"
            "ros2 launch openni2_camera camera_only.launch.py",
            language="bash"
        )

    # Capture Application
    st.markdown("---")
    st.subheader("Image Capture Application")

    st.markdown("""
    Use the standalone GUI application for image capture.
    The application provides all capture controls in one place.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        # Get default output directory
        default_output = str(path_coordinator.get_path("raw_captures_dir"))

        if st.button("Launch Capture App", type="primary"):
            if ros2_bridge.open_capture_app(output_dir=default_output):
                st.success("Capture application launched!")
            else:
                st.error("Failed to launch capture application")

    with col2:
        st.markdown("""
        **Application Features:**
        - Topic selection (auto-detected)
        - Real-time preview with center reticle
        - Single capture / Burst capture
        - Configurable parameters (count, interval, class name, output)
        """)

    # Application usage guide
    with st.expander("How to Use", expanded=False):
        st.markdown("""
        ### Basic Operation

        1. **Topic Selection**: Select camera topic from dropdown
        2. **Class Name**: Enter the object class name
        3. **Output Directory**: Set the save directory

        ### Capture Modes

        - **Single Capture**: Capture one image
        - **Start Burst Capture**: Start continuous capture (3-second countdown)

        ### Parameters

        | Parameter | Description | Recommended |
        |-----------|-------------|-------------|
        | Images | Number of images | 50-100 |
        | Interval | Capture interval (sec) | 0.2-0.5 |
        | Overwrite | Start numbering from 1 | ON |
        """)

    # Collected Images
    st.markdown("---")
    st.subheader("Collected Images")

    raw_captures_dir = path_coordinator.get_path("raw_captures_dir")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("Refresh", key="refresh_captures"):
            st.rerun()

    with col1:
        if raw_captures_dir.exists():
            subdirs = sorted([d for d in raw_captures_dir.iterdir() if d.is_dir()])

            if subdirs:
                # Build tree structure
                tree_lines = [f"📁 {raw_captures_dir.name}/"]

                for i, subdir in enumerate(subdirs):
                    # Count image files
                    img_count = len(list(subdir.glob("*.jpg"))) + len(list(subdir.glob("*.png")))
                    prefix = "└──" if i == len(subdirs) - 1 else "├──"
                    tree_lines.append(f"  {prefix} 📂 {subdir.name} ({img_count} images)")

                st.code("\n".join(tree_lines), language=None)
            else:
                st.info("No captured images yet. Launch the Capture App to start collecting.")
        else:
            st.info("Capture directory not created yet.")


def show_settings():
    """Settings page."""
    st.title("⚙️ Settings")

    registry = st.session_state.registry
    path_coordinator = st.session_state.path_coordinator

    st.subheader("Data Management")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Export Options**")

        if st.button("Export to YOLO Config"):
            output_path = registry.export_to_yolo_config("config/object_classes.json")
            st.success(f"Exported to {output_path}")

        if st.button("Update All Collection Counts"):
            registry.update_all_collection_counts()
            st.success("Updated all counts")
            st.rerun()

        if st.button("Sync All to Datasets"):
            objects = registry.get_all_objects()
            synced = path_coordinator.sync_all_objects([obj.name for obj in objects])
            st.success(f"Synced {len(synced)} directories")

    with col2:
        st.write("**Category Management**")

        new_category = st.text_input("New Category")
        if st.button("Add Category") and new_category:
            registry.add_category(new_category)
            st.success(f"Added category: {new_category}")
            st.rerun()

        st.write("Current categories:")
        for cat in registry.categories:
            st.write(f"  • {cat}")

    st.markdown("---")
    st.subheader("System Status")

    # ROS2 status
    try:
        from services.ros2_bridge import ROS2Bridge
        ros2_bridge = ROS2Bridge()
        diagnostics = ros2_bridge.get_diagnostics()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**ROS2 Status:**")
            if diagnostics["ros2_available"]:
                st.success("ROS2 Available")
                st.write(f"  Topics: {diagnostics.get('topics_count', 'N/A')}")
                st.write(f"  Image Topics: {diagnostics.get('image_topics_count', 'N/A')}")
                st.write(f"  Nodes: {diagnostics.get('nodes_count', 'N/A')}")
                st.write(f"  Capture Node: {'Running' if diagnostics.get('capture_node_running') else 'Not Found'}")
            else:
                st.warning("ROS2 Not Available")

        with col2:
            st.write("**GPU Status:**")
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    st.success(f"GPU Available")
                    st.write(f"  {gpu_name}")
                    st.write(f"  Memory: {gpu_memory:.1f} GB")
                else:
                    st.warning("GPU Not Available")
            except ImportError:
                st.warning("PyTorch not installed")

    except Exception as e:
        st.error(f"Error checking status: {e}")

    st.markdown("---")
    st.subheader("Data Paths")

    path_summary = path_coordinator.get_path_summary()
    for key, path in path_summary.items():
        st.write(f"**{key}:** `{path}`")

    st.markdown("---")
    st.subheader("About")

    st.write("""
    **HSR Object Manager** v2.0

    A comprehensive tool for managing object recognition pipelines
    for RoboCup@Home competitions.

    Features:
    - Register objects with reference images
    - Collect training data via ROS2, camera, or file upload
    - Run auto-annotation pipeline
    - Fine-tune YOLOv8 models
    - Evaluate model performance
    - Export to competition-ready format
    """)


# Import page modules
try:
    from pages import (
        show_annotation_page,
        show_training_page,
        show_evaluation_page,
    )
except ImportError:
    # Fallback - define stub functions
    def show_annotation_page():
        st.title("🏷️ Annotation")
        st.error("Annotation page module not found")

    def show_training_page():
        st.title("🎓 Training")
        st.error("Training page module not found")

    def show_evaluation_page():
        st.title("📈 Evaluation")
        st.error("Evaluation page module not found")


if __name__ == "__main__":
    main()
