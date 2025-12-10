"""
GUI Framework for HSR Perception Tkinter Applications.

This package provides common components, utilities, and base classes
for building consistent Tkinter GUI applications.

Modules:
    base_app: Base application class
    ros2_app: ROS2-integrated application base class
    components: Reusable UI components (TopicSelector, PreviewPanel, StatusBar)
    utils: Utility functions (ROS2 image conversion, drawing helpers)
    styles: Theme and style definitions
"""

from gui_framework.base_app import BaseApp
from gui_framework.ros2_app import ROS2App
from gui_framework.styles.theme import AppTheme

__all__ = [
    "BaseApp",
    "ROS2App",
    "AppTheme",
]
