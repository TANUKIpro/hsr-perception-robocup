"""
Object Viewer Component

Provides UI for viewing object details in the Registry page (view mode).
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_registry import ObjectRegistry, RegisteredObject


def render_object_viewer(obj: "RegisteredObject", registry: "ObjectRegistry"):
    """
    Render object view mode UI.

    Args:
        obj: RegisteredObject to display
        registry: ObjectRegistry instance
    """
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display thumbnail
        _render_thumbnail(obj, registry)

        # Reference images
        _render_reference_images(obj, registry)

    with col2:
        # Object details
        _render_object_details(obj)

        # Action buttons
        _render_action_buttons(obj, registry)


def _render_thumbnail(obj: "RegisteredObject", registry: "ObjectRegistry"):
    """Render thumbnail image."""
    thumbnail_path = registry.get_thumbnail_path(obj.id)
    if thumbnail_path:
        st.image(thumbnail_path, caption="Thumbnail", width=120)


def _render_reference_images(obj: "RegisteredObject", registry: "ObjectRegistry"):
    """Render reference images section with upload capability."""
    ref_images = registry.get_reference_images(obj.id)
    if ref_images:
        st.write("**Reference Images:**")
        img_cols = st.columns(min(len(ref_images), 3))
        for i, img_path in enumerate(ref_images[:3]):
            with img_cols[i]:
                st.image(img_path, width=100)
    else:
        st.write("No reference images")

    # Upload reference image
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


def _render_object_details(obj: "RegisteredObject"):
    """Render object details text."""
    st.write(f"**Name:** {obj.name}")
    st.write(f"**Category:** {obj.category}")
    st.write(f"**Target Samples:** {obj.target_samples}")
    st.write(f"**Collected:** {obj.collected_samples}")

    if obj.remarks:
        st.write(f"**Remarks:** {obj.remarks}")

    # Properties
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


def _render_action_buttons(obj: "RegisteredObject", registry: "ObjectRegistry"):
    """Render action buttons (Collect, Edit, Delete)."""
    edit_key = f"edit_mode_{obj.id}"

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
