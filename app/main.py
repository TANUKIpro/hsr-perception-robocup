"""
HSR Perception - Object Manager

Streamlit application for managing object registration, data collection,
and training preparation for RoboCup@Home competitions.

Features:
- Object registration with reference images
- Multi-source data collection (ROS2, camera, file upload)
- Auto-annotation pipeline integration
- YOLOv8 fine-tuning management
- Model evaluation and visual verification

Usage:
    streamlit run app/main.py
"""

import streamlit as st

# Page config must be first Streamlit command
st.set_page_config(
    page_title="HSR Object Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar
from pages import show_dashboard_page


def main():
    """Main application entry point."""
    # Render common sidebar (includes profile selector and stats)
    render_common_sidebar()

    # Show Dashboard as default page
    show_dashboard_page()


# Legacy exports for backward compatibility during transition
# These can be removed once all references are updated
def show_dashboard():
    """Legacy: Redirect to dashboard page."""
    from pages import show_dashboard_page
    show_dashboard_page()


def show_registry():
    """Legacy: Redirect to registry page."""
    from pages import show_registry_page
    show_registry_page()


def show_collection():
    """Legacy: Redirect to collection page."""
    from pages import show_collection_page
    show_collection_page()


def show_settings():
    """Legacy: Redirect to settings page."""
    from pages import show_settings_page
    show_settings_page()


if __name__ == "__main__":
    main()
