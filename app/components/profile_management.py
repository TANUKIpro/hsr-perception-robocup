"""
Profile Management Component

Provides UI for managing profiles (create, update, delete) in the dashboard.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.profile_manager import ProfileManager


def render_profile_management(
    profile_manager: "ProfileManager",
    on_profile_deleted: callable = None
):
    """
    Render profile management section in dashboard.

    Args:
        profile_manager: ProfileManager instance for profile operations
        on_profile_deleted: Callback function to reinitialize services after deletion
    """
    st.subheader("Profile Management")

    active_profile = profile_manager.get_active_profile()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Edit profile name
        with st.form("edit_profile_name_form"):
            new_name = st.text_input(
                "Profile Name",
                value=active_profile.display_name,
                key="edit_profile_name"
            )
            new_desc = st.text_input(
                "Description",
                value=active_profile.description or "",
                key="edit_profile_desc"
            )

            if st.form_submit_button("Update Profile"):
                if new_name != active_profile.display_name or new_desc != (active_profile.description or ""):
                    profile_manager.update_profile(
                        active_profile.id,
                        display_name=new_name if new_name else None,
                        description=new_desc if new_desc else None
                    )
                    st.success("Profile updated")
                    st.rerun()

    with col2:
        st.write("**Profile Actions**")

        # Create new profile
        if st.button("Create New Profile"):
            st.session_state.show_create_profile = True

        # Delete profile (only if more than one exists)
        profiles = profile_manager.get_all_profiles()
        if len(profiles) > 1:
            if st.button("Delete This Profile"):
                st.session_state.show_delete_confirm = True

    # Create profile dialog
    _render_create_profile_dialog(profile_manager)

    # Delete confirmation
    _render_delete_confirm_dialog(profile_manager, on_profile_deleted)


def _render_create_profile_dialog(profile_manager: "ProfileManager"):
    """Render create profile dialog."""
    if not st.session_state.get("show_create_profile", False):
        return

    with st.container():
        st.markdown("---")
        st.markdown("### Create New Profile")

        new_profile_name = st.text_input("New Profile Name", key="new_profile_name_input")
        new_profile_desc = st.text_input("Description (optional)", key="new_profile_desc_input")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Create", type="primary"):
                if new_profile_name:
                    profile_manager.create_profile(new_profile_name, new_profile_desc)
                    st.session_state.show_create_profile = False
                    st.success(f"Created profile: {new_profile_name}")
                    st.rerun()
                else:
                    st.error("Profile name is required")
        with col_b:
            if st.button("Cancel", key="cancel_create_profile"):
                st.session_state.show_create_profile = False
                st.rerun()


def _render_delete_confirm_dialog(
    profile_manager: "ProfileManager",
    on_profile_deleted: callable = None
):
    """Render delete confirmation dialog."""
    if not st.session_state.get("show_delete_confirm", False):
        return

    active_profile = profile_manager.get_active_profile()

    st.markdown("---")
    st.warning(f"Are you sure you want to delete '{active_profile.display_name}'? This cannot be undone.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Yes, Delete", type="primary"):
            # Switch to another profile first
            other_profile = [p for p in profile_manager.get_all_profiles() if p.id != active_profile.id][0]
            profile_manager.set_active_profile(other_profile.id)
            profile_manager.delete_profile(active_profile.id)

            # Call callback to reinitialize services
            if on_profile_deleted:
                on_profile_deleted()

            st.session_state.show_delete_confirm = False
            st.rerun()
    with col_b:
        if st.button("Cancel", key="cancel_delete_profile"):
            st.session_state.show_delete_confirm = False
            st.rerun()
