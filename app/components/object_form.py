"""
Object Form Component

Provides UI for adding new objects in the Registry page.
"""

import streamlit as st
from typing import TYPE_CHECKING

from components.thumbnail_upload import render_thumbnail_upload_new, clear_new_thumbnail_state

if TYPE_CHECKING:
    from object_registry import ObjectRegistry

# Avoid circular import
import sys
from pathlib import Path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))


def render_add_object_form(registry: "ObjectRegistry"):
    """
    Render form for adding a new object.

    Args:
        registry: ObjectRegistry instance
    """
    st.subheader("Add New Object")

    # Thumbnail upload (outside form for proper file handling)
    thumbnail_data, thumbnail_ext = render_thumbnail_upload_new()

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
            _handle_form_submit(
                registry=registry,
                new_id=new_id,
                new_name=new_name,
                new_display_name=new_display_name,
                new_category=new_category,
                new_target=new_target,
                new_remarks=new_remarks,
                is_heavy=is_heavy,
                is_tiny=is_tiny,
                has_liquid=has_liquid,
                size_cm=size_cm,
                thumbnail_data=thumbnail_data,
                thumbnail_ext=thumbnail_ext
            )


def _handle_form_submit(
    registry: "ObjectRegistry",
    new_id: int,
    new_name: str,
    new_display_name: str,
    new_category: str,
    new_target: int,
    new_remarks: str,
    is_heavy: bool,
    is_tiny: bool,
    has_liquid: bool,
    size_cm: str,
    thumbnail_data: bytes = None,
    thumbnail_ext: str = ""
):
    """Handle form submission for adding a new object."""
    from object_registry import RegisteredObject, ObjectProperties

    if not new_name:
        st.error("Name is required")
        return

    if registry.get_object_by_name(new_name):
        st.error(f"Object with name '{new_name}' already exists")
        return

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

    # Save thumbnail if provided
    if thumbnail_data:
        if thumbnail_ext in [".jpg", ".jpeg", ".png"]:
            temp_path = f"/tmp/thumb_new{thumbnail_ext}"
            with open(temp_path, "wb") as f:
                f.write(thumbnail_data)
            registry.set_thumbnail(obj.id, temp_path)
        else:
            # Pasted image (bytes directly)
            registry.save_thumbnail_from_bytes(obj.id, thumbnail_data, ".png")

        # Clear pasted thumbnail state
        clear_new_thumbnail_state()

    st.success(f"Added object: {new_display_name or new_name}")
    st.rerun()
