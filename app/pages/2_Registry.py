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


def show_page():
    """Main registry page entry point."""
    from components.common_sidebar import render_common_sidebar
    render_common_sidebar()

    # Import and call show_registry from main
    from main import show_registry
    show_registry()


# For Streamlit native multipage
if __name__ == "__main__":
    show_page()
