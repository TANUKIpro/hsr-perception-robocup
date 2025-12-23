"""
ROS2 Command Launcher Component

Provides UI for launching and stopping ROS2 commands from presets.
Integrated into the System Status section of the Settings page.
"""

import streamlit as st
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.command_presets_manager import CommandPreset, CommandPresetsManager
    from services.ros2_process_tracker import ROS2ProcessTracker


def render_ros2_command_launcher(
    presets_manager: "CommandPresetsManager",
    process_tracker: "ROS2ProcessTracker",
) -> None:
    """
    Render the ROS2 command launcher UI.

    Displays command presets with launch/stop buttons and
    provides preset management functionality.

    Args:
        presets_manager: CommandPresetsManager instance
        process_tracker: ROS2ProcessTracker instance
    """
    st.write("**ROS2 Command Launcher:**")

    # Cleanup dead processes first
    process_tracker.cleanup_dead_processes()

    # Load presets
    presets = presets_manager.load_presets()

    if not presets:
        st.info("No presets available. Add your first preset below.")
    else:
        # Separate default and custom presets
        default_presets = [p for p in presets if p.is_default]
        custom_presets = [p for p in presets if not p.is_default]

        # Render default presets
        if default_presets:
            st.caption("Quick Launch")
            for preset in default_presets:
                _render_preset_item(preset, presets_manager, process_tracker)

        # Render custom presets
        if custom_presets:
            st.caption("Custom Presets")
            for preset in custom_presets:
                _render_preset_item(preset, presets_manager, process_tracker)

    # Add preset button
    st.markdown("---")
    if st.button("Add New Preset", key="add_preset_btn", use_container_width=True):
        st.session_state.show_preset_editor = True
        st.session_state.editing_preset_id = None

    # Preset editor
    if st.session_state.get("show_preset_editor", False):
        _render_preset_editor(presets_manager)


def _render_preset_item(
    preset: "CommandPreset",
    presets_manager: "CommandPresetsManager",
    process_tracker: "ROS2ProcessTracker",
) -> None:
    """Render a single preset item with launch/stop controls."""
    is_running = process_tracker.is_running(preset.id)
    external_running = (
        not is_running and process_tracker.is_command_running_externally(preset.command)
    )

    with st.container():
        col1, col2 = st.columns([3, 1])

        with col1:
            # Preset name and command
            st.write(f"**{preset.name}**")
            st.caption(f"`{preset.command}`")

        with col2:
            # Action button based on status
            if is_running:
                # Show stop button
                if st.button(
                    "Stop", key=f"stop_{preset.id}", type="secondary", use_container_width=True
                ):
                    success, msg = process_tracker.stop_process(preset.id)
                    if success:
                        st.toast(f"Stopped: {preset.name}")
                        st.rerun()
                    else:
                        st.error(msg)
            elif external_running:
                # Show warning that external process is running
                st.warning("Running (ext)", icon="!")
            else:
                # Show launch button
                if st.button(
                    "Launch", key=f"launch_{preset.id}", type="primary", use_container_width=True
                ):
                    success, msg = process_tracker.start_command(preset)
                    if success:
                        st.toast(f"Launched: {preset.name}")
                        st.rerun()
                    else:
                        st.error(msg)
                        # Offer force launch option
                        if "already running" in msg:
                            if st.button(
                                "Force Launch",
                                key=f"force_{preset.id}",
                                use_container_width=True,
                            ):
                                success, msg = process_tracker.start_command(
                                    preset, force=True
                                )
                                if success:
                                    st.toast(f"Force launched: {preset.name}")
                                    st.rerun()
                                else:
                                    st.error(msg)

        # Status indicator row
        col_status, col_edit, col_delete = st.columns([2, 1, 1])

        with col_status:
            if is_running:
                proc_info = process_tracker.get_process_info(preset.id)
                if proc_info:
                    st.success(f"Running (PID: {proc_info.pid})")
            elif external_running:
                st.info("External process detected")
            else:
                st.info("Stopped")

        # Edit/Delete buttons for custom presets only
        if not preset.is_default:
            with col_edit:
                if st.button("Edit", key=f"edit_{preset.id}", use_container_width=True):
                    st.session_state.show_preset_editor = True
                    st.session_state.editing_preset_id = preset.id
                    st.rerun()

            with col_delete:
                if st.button("Delete", key=f"del_{preset.id}", use_container_width=True):
                    if presets_manager.delete_preset(preset.id):
                        st.toast(f"Deleted: {preset.name}")
                        st.rerun()
                    else:
                        st.error("Cannot delete this preset")

        st.markdown("---")


def _render_preset_editor(presets_manager: "CommandPresetsManager") -> None:
    """Render the preset add/edit form."""
    editing_id = st.session_state.get("editing_preset_id")

    if editing_id:
        st.subheader("Edit Preset")
        preset = presets_manager.get_preset(editing_id)
        if not preset:
            st.error("Preset not found")
            st.session_state.show_preset_editor = False
            st.session_state.editing_preset_id = None
            # Don't rerun after error - let user take next action
            return

        default_name = preset.name
        default_command = preset.command
        default_description = preset.description
    else:
        st.subheader("Add New Preset")
        default_name = ""
        default_command = ""
        default_description = ""

    with st.form("preset_editor_form", clear_on_submit=True):
        name = st.text_input(
            "Preset Name",
            value=default_name,
            placeholder="e.g., Camera Node",
        )

        command = st.text_input(
            "Command",
            value=default_command,
            placeholder="e.g., ros2 launch my_pkg node.launch.py",
        )

        description = st.text_area(
            "Description (optional)",
            value=default_description,
            placeholder="Brief description of what this command does",
            height=80,
        )

        col1, col2 = st.columns(2)

        with col1:
            submit = st.form_submit_button("Save", use_container_width=True, type="primary")

        with col2:
            cancel = st.form_submit_button("Cancel", use_container_width=True)

        if submit:
            if not name or not command:
                st.error("Name and Command are required")
            else:
                if editing_id:
                    # Update existing preset
                    if presets_manager.update_preset(
                        editing_id,
                        name=name,
                        command=command,
                        description=description,
                    ):
                        st.toast(f"Updated: {name}")
                        st.session_state.show_preset_editor = False
                        st.session_state.editing_preset_id = None
                        st.rerun()
                    else:
                        st.error("Failed to update preset")
                else:
                    # Create new preset
                    new_preset = presets_manager.add_preset(
                        name=name,
                        command=command,
                        description=description,
                    )
                    st.toast(f"Added: {new_preset.name}")
                    st.session_state.show_preset_editor = False
                    st.rerun()

        if cancel:
            st.session_state.show_preset_editor = False
            st.session_state.editing_preset_id = None
            st.rerun()
