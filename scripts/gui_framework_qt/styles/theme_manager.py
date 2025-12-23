"""
Theme manager for PyQt6 GUI framework.

Provides QSS-based theming with light/dark mode support.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPalette


class Theme(Enum):
    """Available application themes."""

    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


class ThemeManager:
    """
    QSS-based theme management.

    Features:
    - Light/dark theme switching
    - System theme detection
    - Persistent theme settings
    - Dynamic QSS stylesheet loading

    Usage:
        ThemeManager.apply_to_widget(main_window)
        ThemeManager.set_theme(main_window, Theme.DARK)
        ThemeManager.toggle_theme(main_window)
    """

    _current_theme: Theme = Theme.LIGHT
    _styles_dir = Path(__file__).parent

    @classmethod
    def apply_to_widget(cls, widget: QWidget) -> None:
        """
        Apply saved theme to a widget.

        Args:
            widget: Widget to apply theme to
        """
        theme = cls._load_saved_theme()
        cls.set_theme(widget, theme)

    @classmethod
    def set_theme(cls, widget: QWidget, theme: Theme) -> None:
        """
        Set theme for a widget.

        Args:
            widget: Widget to apply theme to
            theme: Theme to apply
        """
        cls._current_theme = theme

        # Handle system theme
        actual_theme = theme
        if theme == Theme.SYSTEM:
            actual_theme = cls._detect_system_theme()

        # Load and apply QSS
        stylesheet = cls._load_stylesheet(actual_theme)
        widget.setStyleSheet(stylesheet)

        # Save preference
        cls._save_theme(theme)

    @classmethod
    def toggle_theme(cls, widget: QWidget) -> Theme:
        """
        Toggle between light and dark themes.

        Args:
            widget: Widget to apply theme to

        Returns:
            New theme
        """
        if cls._current_theme == Theme.LIGHT:
            new_theme = Theme.DARK
        else:
            new_theme = Theme.LIGHT

        cls.set_theme(widget, new_theme)
        return new_theme

    @classmethod
    def get_current_theme(cls) -> Theme:
        """Get the current theme."""
        return cls._current_theme

    @classmethod
    def _load_stylesheet(cls, theme: Theme) -> str:
        """Load QSS stylesheet for the given theme."""
        qss_file = cls._styles_dir / f"{theme.value}.qss"

        if qss_file.exists():
            return qss_file.read_text(encoding="utf-8")

        # Fallback to embedded stylesheet
        return cls._get_embedded_stylesheet(theme)

    @classmethod
    def _get_embedded_stylesheet(cls, theme: Theme) -> str:
        """Get embedded stylesheet (fallback if QSS files not found)."""
        if theme == Theme.DARK:
            return cls._get_dark_stylesheet()
        return cls._get_light_stylesheet()

    @classmethod
    def _get_light_stylesheet(cls) -> str:
        """Get embedded light theme stylesheet."""
        return """
/* Light Theme */
QMainWindow, QWidget {
    background-color: #f0f0f0;
    color: #2c3e50;
    font-family: "Noto Sans JP", "TkDefaultFont", sans-serif;
    font-size: 10pt;
}

QGroupBox {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
}

QPushButton {
    background-color: #ecf0f1;
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 24px;
}

QPushButton:hover {
    background-color: #3498db;
    color: white;
}

QPushButton:pressed {
    background-color: #2980b9;
}

QPushButton:disabled {
    background-color: #d5d8dc;
    color: #95a5a6;
}

QPushButton[class="primary"] {
    background-color: #3498db;
    color: white;
    border: none;
}

QPushButton[class="danger"] {
    background-color: #e74c3c;
    color: white;
    border: none;
}

QPushButton[class="success"] {
    background-color: #27ae60;
    color: white;
    border: none;
}

QComboBox {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
    background-color: white;
}

QComboBox:hover {
    border-color: #3498db;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QSpinBox, QDoubleSpinBox {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
    background-color: white;
}

QLineEdit {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
    background-color: white;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #3498db;
}

QProgressBar {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    text-align: center;
    height: 20px;
    background-color: #ecf0f1;
}

QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 3px;
}

QLabel {
    color: #2c3e50;
}

QCheckBox {
    spacing: 5px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QListWidget {
    border: 1px solid #bdc3c7;
    border-radius: 4px;
    background-color: white;
}

QListWidget::item:selected {
    background-color: #3498db;
    color: white;
}

QSlider::groove:horizontal {
    height: 6px;
    background-color: #bdc3c7;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #3498db;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #2980b9;
}
"""

    @classmethod
    def _get_dark_stylesheet(cls) -> str:
        """Get embedded dark theme stylesheet."""
        return """
/* Dark Theme */
QMainWindow, QWidget {
    background-color: #1e1e1e;
    color: #ecf0f1;
    font-family: "Noto Sans JP", "TkDefaultFont", sans-serif;
    font-size: 10pt;
}

QGroupBox {
    border: 1px solid #404040;
    border-radius: 4px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #ecf0f1;
}

QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 24px;
    color: #ecf0f1;
}

QPushButton:hover {
    background-color: #3498db;
    color: white;
}

QPushButton:pressed {
    background-color: #2980b9;
}

QPushButton:disabled {
    background-color: #353535;
    color: #666666;
}

QPushButton[class="primary"] {
    background-color: #3498db;
    color: white;
    border: none;
}

QPushButton[class="danger"] {
    background-color: #e74c3c;
    color: white;
    border: none;
}

QPushButton[class="success"] {
    background-color: #27ae60;
    color: white;
    border: none;
}

QComboBox {
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
    background-color: #2d2d2d;
    color: #ecf0f1;
}

QComboBox:hover {
    border-color: #3498db;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #ecf0f1;
    selection-background-color: #3498db;
}

QSpinBox, QDoubleSpinBox {
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
    background-color: #2d2d2d;
    color: #ecf0f1;
}

QLineEdit {
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px;
    min-height: 24px;
    background-color: #2d2d2d;
    color: #ecf0f1;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #3498db;
}

QProgressBar {
    border: 1px solid #404040;
    border-radius: 4px;
    text-align: center;
    height: 20px;
    background-color: #2d2d2d;
    color: #ecf0f1;
}

QProgressBar::chunk {
    background-color: #3498db;
    border-radius: 3px;
}

QLabel {
    color: #ecf0f1;
}

QCheckBox {
    spacing: 5px;
    color: #ecf0f1;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
}

QListWidget {
    border: 1px solid #404040;
    border-radius: 4px;
    background-color: #2d2d2d;
    color: #ecf0f1;
}

QListWidget::item:selected {
    background-color: #3498db;
    color: white;
}

QSlider::groove:horizontal {
    height: 6px;
    background-color: #404040;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background-color: #3498db;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #2980b9;
}
"""

    @classmethod
    def _detect_system_theme(cls) -> Theme:
        """Detect system theme preference."""
        app = QApplication.instance()
        if app:
            palette = app.palette()
            bg_color = palette.color(QPalette.ColorRole.Window)
            # If background is dark (lightness < 128), use dark theme
            if bg_color.lightness() < 128:
                return Theme.DARK
        return Theme.LIGHT

    @classmethod
    def _load_saved_theme(cls) -> Theme:
        """Load saved theme preference."""
        settings = QSettings("HSRPerception", "GUIFramework")
        theme_name = settings.value("theme", Theme.LIGHT.value)
        try:
            return Theme(theme_name)
        except ValueError:
            return Theme.LIGHT

    @classmethod
    def _save_theme(cls, theme: Theme) -> None:
        """Save theme preference."""
        settings = QSettings("HSRPerception", "GUIFramework")
        settings.setValue("theme", theme.value)
