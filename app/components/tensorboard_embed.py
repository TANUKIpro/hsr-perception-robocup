"""
TensorBoard Integration Panel

Provides embedded iframe display and external link options
using Streamlit standard components.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Optional


def render_tensorboard_panel(
    tensorboard_url: str,
    show_iframe: bool = True,
    iframe_height: int = 600,
):
    """
    Render TensorBoard integration panel.

    Provides:
    - Toggle between embedded iframe and placeholder
    - External link button
    - Quick access links to specific TensorBoard tabs

    Args:
        tensorboard_url: TensorBoard server URL
        show_iframe: Default state for iframe display
        iframe_height: Height of iframe in pixels
    """
    st.subheader("◫ TensorBoard")

    # Controls row
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        show_embed = st.checkbox(
            "Embedded View",
            value=show_iframe,
            key="tensorboard_embed_toggle",
            help="Toggle between embedded view and external link"
        )

    with col2:
        st.link_button("↗ Open External", tensorboard_url)

    with col3:
        st.caption(tensorboard_url)

    if show_embed:
        # Render iframe using Streamlit components
        components.iframe(tensorboard_url, height=iframe_height, scrolling=True)
    else:
        # Render placeholder with quick links
        _render_tensorboard_placeholder(tensorboard_url)


def render_tensorboard_status(
    tensorboard_url: Optional[str] = None,
    is_running: bool = False,
):
    """
    Render compact TensorBoard status indicator.

    Args:
        tensorboard_url: TensorBoard URL if available
        is_running: Whether TensorBoard server is running
    """
    if is_running and tensorboard_url:
        st.success(f"● TensorBoard: {tensorboard_url}")
    else:
        st.info("○ TensorBoard: Not Running")


def render_tensorboard_mini_status(tensorboard_url: Optional[str] = None):
    """
    Render minimal TensorBoard link for compact displays.

    Args:
        tensorboard_url: TensorBoard URL
    """
    if tensorboard_url:
        return f"[◫ TensorBoard ↗]({tensorboard_url})"
    return ""


def _render_tensorboard_placeholder(tensorboard_url: str):
    """Render placeholder with quick links when iframe is hidden."""
    with st.container(border=True):
        st.markdown("◫")
        st.write("TensorBoard is available at the URL above.")
        st.caption("Click 'Open External' or enable embedded view to see training metrics.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.link_button("Scalars", f"{tensorboard_url}#scalars")
        with col2:
            st.link_button("Images", f"{tensorboard_url}#images")
        with col3:
            st.link_button("Graphs", f"{tensorboard_url}#graphs")


def render_tensorboard_section(
    task_extra_data: dict,
    default_show_iframe: bool = False,
):
    """
    Convenience function to render complete TensorBoard section.

    Handles the case where TensorBoard may or may not be available.

    Args:
        task_extra_data: Task's extra_data dict
        default_show_iframe: Whether to show iframe by default
    """
    tensorboard_url = task_extra_data.get("tensorboard_url")

    if tensorboard_url:
        with st.expander("📊 TensorBoard", expanded=True):
            render_tensorboard_panel(
                tensorboard_url=tensorboard_url,
                show_iframe=default_show_iframe,
                iframe_height=500,
            )
    else:
        render_tensorboard_status(is_running=False)
