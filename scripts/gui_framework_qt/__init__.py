"""
PyQt6 GUI Framework for HSR Perception Applications.

This package provides a modern PyQt6-based GUI framework for building
desktop applications that integrate with ROS2.

Modules:
    core: Base application classes and ROS2 integration
    styles: Theme management and QSS stylesheets
    components: Reusable UI widgets
    workers: Background task workers
    utils: Utility functions and helpers
"""

from gui_framework_qt.core.base_app import BaseApp
from gui_framework_qt.core.ros2_app import ROS2App
from gui_framework_qt.styles.theme_manager import ThemeManager, Theme

__all__ = [
    "BaseApp",
    "ROS2App",
    "ThemeManager",
    "Theme",
]
