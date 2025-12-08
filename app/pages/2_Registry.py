"""
Registry Page

Provides UI for managing object definitions and registration.
"""

import streamlit as st
from pathlib import Path
import sys

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar
from components.object_editor import render_object_editor
from components.object_viewer import render_object_viewer
from components.object_form import render_add_object_form
from components.ihr_import import render_ihr_import


def show_registry_page():
    """Registry page for managing objects."""
    render_common_sidebar()

    st.title("📋 Object Registry")

    registry = st.session_state.registry

    # Tabs for view/add
    tab1, tab2 = st.tabs(["View Objects", "Add New Object"])

    with tab1:
        _render_view_tab(registry)

    with tab2:
        render_add_object_form(registry)
        render_ihr_import(registry)


def _render_view_tab(registry):
    """Render the View Objects tab."""
    objects = registry.get_all_objects()

    if not objects:
        st.info("No objects registered yet. Use 'Add New Object' tab to add objects.")
        return

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
            if st.session_state[edit_key]:
                # Edit mode
                exit_edit = render_object_editor(obj, registry)
                if exit_edit:
                    st.rerun()
            else:
                # View mode
                render_object_viewer(obj, registry)


# For Streamlit native multipage
if __name__ == "__main__":
    show_registry_page()
