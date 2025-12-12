"""
Profile Management Component

Provides tabbed UI for managing profiles (list, create, import/export) in Settings page.
"""

import zipfile
import streamlit as st
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.profile_manager import ProfileManager


def render_profile_management_tabs(
    profile_manager: "ProfileManager",
    on_profile_deleted: Callable[[], None] | None = None
) -> None:
    """
    Render tabbed Profile Management UI.

    Args:
        profile_manager: ProfileManager instance for profile operations
        on_profile_deleted: Callback function to reinitialize services after deletion
    """
    st.subheader("Profile Management")

    tab1, tab2, tab3 = st.tabs(["Profile List", "Create New", "Import/Export"])

    with tab1:
        _render_profile_list_tab(profile_manager, on_profile_deleted)

    with tab2:
        _render_create_profile_tab(profile_manager)

    with tab3:
        _render_import_export_tab(profile_manager)


def _render_profile_list_tab(
    profile_manager: "ProfileManager",
    on_profile_deleted: Callable[[], None] | None = None
) -> None:
    """Render profile list tab with cards for each profile."""
    profiles = profile_manager.get_all_profiles()
    active_profile = profile_manager.get_active_profile()

    if not profiles:
        st.info("No profiles found.")
        return

    for profile in profiles:
        is_active = profile.id == active_profile.id

        # Profile card
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                # Profile info
                status = "**Active**" if is_active else ""
                st.markdown(f"### {profile.display_name} {status}")
                if profile.description:
                    st.caption(profile.description)
                st.caption(f"ID: `{profile.id}` | Created: {profile.created_at[:10] if profile.created_at else 'N/A'}")

            with col2:
                if is_active:
                    # Edit button for active profile
                    edit_key = f"edit_profile_{profile.id}"
                    if edit_key not in st.session_state:
                        st.session_state[edit_key] = False

                    if not st.session_state[edit_key]:
                        if st.button("Edit", key=f"edit_btn_{profile.id}"):
                            st.session_state[edit_key] = True
                            st.rerun()
                    else:
                        if st.button("Cancel", key=f"cancel_btn_{profile.id}"):
                            st.session_state[edit_key] = False
                            st.rerun()

        # Edit form (shown when editing)
        if is_active and st.session_state.get(f"edit_profile_{profile.id}", False):
            _render_edit_form(profile_manager, profile, on_profile_deleted)

        st.markdown("---")


def _render_edit_form(
    profile_manager: "ProfileManager",
    profile,
    on_profile_deleted: Callable[[], None] | None = None
) -> None:
    """Render edit form for the active profile."""
    with st.form(f"edit_form_{profile.id}"):
        new_name = st.text_input("Profile Name", value=profile.display_name)
        new_desc = st.text_input("Description", value=profile.description or "")

        col_save, col_delete = st.columns(2)

        with col_save:
            if st.form_submit_button("Save Changes", type="primary"):
                if new_name != profile.display_name or new_desc != (profile.description or ""):
                    profile_manager.update_profile(
                        profile.id,
                        display_name=new_name if new_name else None,
                        description=new_desc if new_desc else None
                    )
                    st.session_state[f"edit_profile_{profile.id}"] = False
                    st.success("Profile updated")
                    st.rerun()

        with col_delete:
            # Delete button (only if more than one profile exists)
            profiles = profile_manager.get_all_profiles()
            if len(profiles) > 1:
                if st.form_submit_button("Delete Profile"):
                    st.session_state.show_delete_confirm = True
                    st.rerun()

    # Delete confirmation dialog
    if st.session_state.get("show_delete_confirm", False):
        _render_delete_confirm(profile_manager, profile, on_profile_deleted)


def _render_delete_confirm(
    profile_manager: "ProfileManager",
    profile,
    on_profile_deleted: Callable[[], None] | None = None
) -> None:
    """Render delete confirmation dialog."""
    st.warning(f"Are you sure you want to delete '{profile.display_name}'? This cannot be undone.")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("Yes, Delete", type="primary", key="confirm_delete"):
            # Switch to another profile first
            other_profile = [p for p in profile_manager.get_all_profiles() if p.id != profile.id][0]
            profile_manager.set_active_profile(other_profile.id)
            profile_manager.delete_profile(profile.id)

            # Call callback to reinitialize services
            if on_profile_deleted:
                on_profile_deleted()

            st.session_state.show_delete_confirm = False
            st.session_state[f"edit_profile_{profile.id}"] = False
            st.rerun()

    with col_no:
        if st.button("Cancel", key="cancel_delete"):
            st.session_state.show_delete_confirm = False
            st.rerun()


def _render_create_profile_tab(profile_manager: "ProfileManager") -> None:
    """Render create new profile tab."""
    st.write("Create a new profile to manage a separate set of objects and datasets.")

    with st.form("create_profile_form"):
        new_name = st.text_input("Profile Name", placeholder="e.g., Competition 2025")
        new_desc = st.text_input("Description (optional)", placeholder="e.g., For RoboCup@Home 2025")

        if st.form_submit_button("Create Profile", type="primary"):
            if new_name:
                profile_manager.create_profile(new_name, new_desc)
                st.success(f"Created profile: {new_name}")
                st.info("Switch to the new profile using the sidebar selector.")
                st.rerun()
            else:
                st.error("Profile name is required")


def _render_import_export_tab(profile_manager: "ProfileManager") -> None:
    """Render import/export tab."""
    active_profile = profile_manager.get_active_profile()

    col1, col2 = st.columns(2)

    # ===== Export Column =====
    with col1:
        st.write("**Export Profile**")
        st.write(f"Current profile: **{active_profile.display_name}**")

        # Initialize session state for export
        export_key = f"export_data_{active_profile.id}"
        filename_key = f"export_filename_{active_profile.id}"

        # Step 1: Prepare Export button
        if st.button("Prepare Export", key="prepare_export_btn"):
            try:
                with st.spinner("Preparing export..."):
                    zip_bytes = profile_manager.export_profile_to_bytes(active_profile.id)

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_name = active_profile.display_name.replace(" ", "_").replace("/", "_")
                filename = f"{safe_name}_{timestamp}.zip"

                # Store in session state
                st.session_state[export_key] = zip_bytes
                st.session_state[filename_key] = filename

                st.success("Export ready! Click 'Download ZIP' below.")
                file_size = len(zip_bytes) / (1024 * 1024)  # MB
                st.info(f"File size: {file_size:.2f} MB")

            except ValueError as e:
                st.error(f"Export failed: {str(e)}")
            except OSError as e:
                if "No space left" in str(e):
                    st.error("Insufficient disk space.")
                else:
                    st.error(f"System error: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

        # Step 2: Show download button if data is ready
        if export_key in st.session_state and st.session_state[export_key] is not None:
            st.download_button(
                label="Download ZIP",
                data=st.session_state[export_key],
                file_name=st.session_state.get(filename_key, "profile_export.zip"),
                mime="application/zip",
                key="download_profile_zip",
                use_container_width=True
            )

    # ===== Import Column =====
    with col2:
        st.write("**Import Profile**")
        st.code("docker cp /path/to/profile.zip hsr-perception:/app/imports/", language="bash")

        # Scan for ZIP files in import directory
        import_dir = Path("/app/imports")
        if not import_dir.exists():
            import_dir.mkdir(parents=True, exist_ok=True)

        zip_files = sorted(import_dir.glob("*.zip"))

        if not zip_files:
            st.caption("No ZIP files found in `/app/imports/`")
        else:
            # File selector
            selected_zip = st.selectbox(
                "Select ZIP file",
                options=zip_files,
                format_func=lambda p: f"{p.name} ({p.stat().st_size / (1024*1024):.1f} MB)",
                key="import_profile_selector"
            )

            # Optional: custom display name
            custom_name = st.text_input(
                "Custom name (optional)",
                key="import_custom_name",
                placeholder="Leave empty to use original name"
            )

            # Import button
            if st.button("Import Profile", key="import_profile_btn"):
                try:
                    with st.spinner("Importing profile..."):
                        imported_profile = profile_manager.import_profile(
                            str(selected_zip),
                            display_name=custom_name if custom_name else None
                        )

                    # Success feedback
                    st.success(f"Imported profile: **{imported_profile.display_name}**")
                    st.info("Switch to the imported profile using the sidebar selector.")

                    # Rerun to refresh UI
                    st.rerun()

                except zipfile.BadZipFile:
                    st.error("Invalid ZIP file. Please select a valid profile export.")
                except ValueError as e:
                    if "path traversal" in str(e).lower():
                        st.error("Security error: Invalid ZIP file structure.")
                    else:
                        st.error(f"Invalid profile: {str(e)}")
                except FileNotFoundError:
                    st.error("ZIP file not found.")
                except OSError as e:
                    if "No space left" in str(e):
                        st.error("Insufficient disk space.")
                    else:
                        st.error(f"System error: {str(e)}")
                except Exception as e:
                    st.error(f"Import failed: {str(e)}")


# Legacy function for backward compatibility (deprecated)
def render_profile_management(
    profile_manager: "ProfileManager",
    on_profile_deleted: Callable[[], None] | None = None
) -> None:
    """
    Deprecated: Use render_profile_management_tabs() instead.
    This function is kept for backward compatibility.
    """
    render_profile_management_tabs(profile_manager, on_profile_deleted)
