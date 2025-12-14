"""
HSR Perception App Components

Reusable UI components for Streamlit pages.
"""

from components.common_sidebar import render_common_sidebar
from components.profile_management import render_profile_management
from components.thumbnail_upload import (
    render_thumbnail_upload,
    render_thumbnail_upload_new,
    clear_thumbnail_state,
    clear_new_thumbnail_state,
)
from components.object_editor import render_object_editor
from components.object_viewer import render_object_viewer
from components.object_form import render_add_object_form
from components.ros2_collection import render_ros2_collection
from components.captured_images_tree import render_captured_images_tree
from components.video_extractor import render_video_extractor
from components.system_status import (
    render_system_status,
    render_ros2_diagnostics,
    render_gpu_status,
)


__all__ = [
    # Sidebar
    "render_common_sidebar",
    # Profile
    "render_profile_management",
    # Thumbnail
    "render_thumbnail_upload",
    "render_thumbnail_upload_new",
    "clear_thumbnail_state",
    "clear_new_thumbnail_state",
    # Registry
    "render_object_editor",
    "render_object_viewer",
    "render_add_object_form",
    # Collection
    "render_ros2_collection",
    "render_captured_images_tree",
    "render_video_extractor",
    # System
    "render_system_status",
    "render_ros2_diagnostics",
    "render_gpu_status",
]
