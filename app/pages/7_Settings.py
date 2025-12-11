"""
Settings Page

Provides UI for application settings and configuration.
"""

import streamlit as st
import zipfile
from datetime import datetime
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_registry import ObjectRegistry
    from services.path_coordinator import PathCoordinator
    from services.profile_manager import ProfileManager

# Add app directory to path
app_dir = Path(__file__).parent.parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar
from components.system_status import render_system_status


def show_settings_page() -> None:
    """Settings page."""
    render_common_sidebar()

    st.title("⚙️ Settings")

    registry = st.session_state.registry
    path_coordinator = st.session_state.path_coordinator
    profile_manager = st.session_state.profile_manager

    # Data Management
    _render_data_management(registry, path_coordinator)

    st.markdown("---")

    # Profile Import/Export
    _render_profile_import_export(profile_manager)

    st.markdown("---")

    # System Status
    render_system_status()

    st.markdown("---")

    # Data Paths
    _render_data_paths(path_coordinator)

    st.markdown("---")

    # About
    _render_about()


def _render_data_management(registry: "ObjectRegistry", path_coordinator: "PathCoordinator") -> None:
    """Render data management section."""
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


def _render_profile_import_export(profile_manager: "ProfileManager") -> None:
    """Render profile import/export section."""
    st.subheader("Profile Import/Export")

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
        st.info("Place ZIP file in `/app/imports/` directory")

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
                    st.info(
                        "Switch to the imported profile using the sidebar selector."
                    )

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


def _render_data_paths(path_coordinator: "PathCoordinator") -> None:
    """Render data paths summary."""
    st.subheader("Data Paths")

    path_summary = path_coordinator.get_path_summary()
    for key, path in path_summary.items():
        st.write(f"**{key}:** `{path}`")


def _render_about() -> None:
    """Render about section."""
    st.subheader("About")

    st.write("""
    **HSR Object Manager** v2.0

    A comprehensive tool for managing object recognition pipelines
    for RoboCup@Home competitions.
    """)

    # Creator info
    st.caption("Created by")
    col1, col2, col3 = st.columns([1, 1, 6])
    img_dir = app_dir / "img"
    with col1:
        if (img_dir / "tid_logo.svg").exists():
            st.image(str(img_dir / "tid_logo.svg"), width=120)
    with col2:
        if (img_dir / "ikeryo.jpg").exists():
            st.image(str(img_dir / "ikeryo.jpg"), width=120)


# For Streamlit native multipage
if __name__ == "__main__":
    show_settings_page()
