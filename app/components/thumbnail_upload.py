"""
Thumbnail Upload Component

Provides UI for uploading and managing object thumbnails,
including file upload and clipboard paste support.
"""

import streamlit as st
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

# Optional: clipboard paste support
try:
    from streamlit_paste_button import paste_image_button
    PASTE_BUTTON_AVAILABLE = True
except ImportError:
    PASTE_BUTTON_AVAILABLE = False

if TYPE_CHECKING:
    from object_registry import ObjectRegistry


def render_thumbnail_upload(
    obj_id: int,
    registry: "ObjectRegistry",
    key_prefix: str = ""
) -> bool:
    """
    Render thumbnail upload UI for an existing object (edit mode).

    Args:
        obj_id: Object ID to update thumbnail for
        registry: ObjectRegistry instance
        key_prefix: Prefix for session state keys

    Returns:
        True if thumbnail was updated (requires rerun)
    """
    # Session state keys for preventing infinite reruns
    thumb_processed_key = f"{key_prefix}thumb_processed_{obj_id}"
    paste_processed_key = f"{key_prefix}paste_processed_{obj_id}"

    if thumb_processed_key not in st.session_state:
        st.session_state[thumb_processed_key] = None
    if paste_processed_key not in st.session_state:
        st.session_state[paste_processed_key] = False

    thumbnail_updated = False

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Show current thumbnail
        thumbnail_path = registry.get_thumbnail_path(obj_id)
        if thumbnail_path:
            st.image(thumbnail_path, caption="Current Thumbnail", width=120)
        else:
            st.info("No thumbnail")

    with col2:
        # File upload
        thumb_upload = st.file_uploader(
            "Upload thumbnail",
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}thumb_edit_{obj_id}"
        )
        if thumb_upload:
            # Create unique ID for this file to detect new uploads
            file_id = f"{thumb_upload.name}_{thumb_upload.size}"
            if st.session_state[thumb_processed_key] != file_id:
                temp_path = f"/tmp/thumb_{thumb_upload.name}"
                with open(temp_path, "wb") as f:
                    f.write(thumb_upload.read())
                registry.set_thumbnail(obj_id, temp_path)
                st.session_state[thumb_processed_key] = file_id
                st.success("Thumbnail updated")
                thumbnail_updated = True

    with col3:
        # Clipboard paste
        if PASTE_BUTTON_AVAILABLE:
            paste_result = paste_image_button(
                label="Paste from clipboard",
                key=f"{key_prefix}paste_thumb_{obj_id}"
            )
            if paste_result.image_data is not None and not st.session_state[paste_processed_key]:
                registry.save_thumbnail_from_bytes(
                    obj_id,
                    paste_result.image_data,
                    ".png"
                )
                st.session_state[paste_processed_key] = True
                st.success("Thumbnail pasted")
                thumbnail_updated = True
        else:
            st.caption("Install streamlit-paste-button for clipboard paste")

    return thumbnail_updated


def render_thumbnail_upload_new(key_prefix: str = "new_obj") -> Tuple[Optional[bytes], str]:
    """
    Render thumbnail upload UI for a new object (outside form).

    Args:
        key_prefix: Prefix for session state keys

    Returns:
        Tuple of (thumbnail_bytes, extension) or (None, "") if no thumbnail
    """
    st.write("**Thumbnail (optional):**")

    # Initialize session state for pasted thumbnail
    pasted_key = f"{key_prefix}_pasted_thumb"
    if pasted_key not in st.session_state:
        st.session_state[pasted_key] = None

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        new_thumbnail = st.file_uploader(
            "Upload thumbnail",
            type=["jpg", "jpeg", "png"],
            key=f"{key_prefix}_thumbnail",
            help="Upload a thumbnail image for this object"
        )

    with col2:
        if PASTE_BUTTON_AVAILABLE:
            paste_result = paste_image_button(
                label="Paste from clipboard",
                key=f"paste_{key_prefix}_thumb"
            )
            if paste_result.image_data is not None:
                st.session_state[pasted_key] = paste_result.image_data
        else:
            st.caption("Install streamlit-paste-button for clipboard paste")

    with col3:
        if new_thumbnail:
            st.image(new_thumbnail, caption="Preview (uploaded)", width=120)
        elif st.session_state[pasted_key]:
            st.image(st.session_state[pasted_key], caption="Preview (pasted)", width=120)
        else:
            st.info("No thumbnail")

    # Return the thumbnail data
    if new_thumbnail:
        # Get extension from filename
        ext = Path(new_thumbnail.name).suffix
        return new_thumbnail.getvalue(), ext
    elif st.session_state[pasted_key]:
        return st.session_state[pasted_key], ".png"

    return None, ""


def clear_thumbnail_state(obj_id: int, key_prefix: str = ""):
    """Clear thumbnail-related session state for an object."""
    thumb_processed_key = f"{key_prefix}thumb_processed_{obj_id}"
    paste_processed_key = f"{key_prefix}paste_processed_{obj_id}"

    if thumb_processed_key in st.session_state:
        st.session_state[thumb_processed_key] = None
    if paste_processed_key in st.session_state:
        st.session_state[paste_processed_key] = False


def clear_new_thumbnail_state(key_prefix: str = "new_obj"):
    """Clear thumbnail state for new object form."""
    pasted_key = f"{key_prefix}_pasted_thumb"
    if pasted_key in st.session_state:
        st.session_state[pasted_key] = None
