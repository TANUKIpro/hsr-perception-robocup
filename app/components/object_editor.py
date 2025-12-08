"""
Object Editor Component

Provides UI for editing object properties in the Registry page.
"""

import streamlit as st
from typing import TYPE_CHECKING

from components.thumbnail_upload import render_thumbnail_upload, clear_thumbnail_state

if TYPE_CHECKING:
    from object_registry import ObjectRegistry, RegisteredObject


def render_object_editor(obj: "RegisteredObject", registry: "ObjectRegistry") -> bool:
    """
    Render object edit mode UI.

    Args:
        obj: RegisteredObject to edit
        registry: ObjectRegistry instance

    Returns:
        True if edit mode should be exited (save/cancel)
    """
    st.markdown("### Edit Object")

    # Thumbnail upload section
    thumbnail_updated = render_thumbnail_upload(obj.id, registry)
    if thumbnail_updated:
        st.rerun()

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

    # Properties section
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
    exit_edit_mode = False
    edit_key = f"edit_mode_{obj.id}"

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
            clear_thumbnail_state(obj.id)
            st.success("Object updated")
            exit_edit_mode = True

    with btn_col2:
        if st.button("Cancel", key=f"cancel_{obj.id}"):
            # Clear edit mode and processed states
            st.session_state[edit_key] = False
            clear_thumbnail_state(obj.id)
            exit_edit_mode = True

    return exit_edit_mode
