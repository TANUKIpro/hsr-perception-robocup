"""
Color palette definitions for PyQt6 GUI framework.

Provides consistent color definitions across themes.
"""


class Colors:
    """
    Unified color palette for the application.

    Usage:
        from gui_framework_qt.styles.colors import Colors
        button_color = Colors.PRIMARY
    """

    # Primary colors
    PRIMARY = "#3498db"  # Blue
    PRIMARY_HOVER = "#2980b9"
    PRIMARY_PRESSED = "#1f5a85"

    # Success colors
    SUCCESS = "#27ae60"  # Green
    SUCCESS_HOVER = "#219a52"
    SUCCESS_PRESSED = "#1a7a41"

    # Warning colors
    WARNING = "#f39c12"  # Orange
    WARNING_HOVER = "#d68910"
    WARNING_PRESSED = "#b9770e"

    # Danger colors
    DANGER = "#e74c3c"  # Red
    DANGER_HOVER = "#c0392b"
    DANGER_PRESSED = "#922b21"

    # Info colors
    INFO = "#17a2b8"  # Cyan
    INFO_HOVER = "#138496"
    INFO_PRESSED = "#0f6674"

    # Neutral colors (Light theme)
    LIGHT_BACKGROUND = "#f0f0f0"
    LIGHT_SURFACE = "#ffffff"
    LIGHT_BORDER = "#bdc3c7"
    LIGHT_TEXT = "#2c3e50"
    LIGHT_TEXT_SECONDARY = "#7f8c8d"

    # Neutral colors (Dark theme)
    DARK_BACKGROUND = "#1e1e1e"
    DARK_SURFACE = "#2d2d2d"
    DARK_BORDER = "#404040"
    DARK_TEXT = "#ecf0f1"
    DARK_TEXT_SECONDARY = "#95a5a6"

    # Common colors
    WHITE = "#ffffff"
    BLACK = "#000000"
    TRANSPARENT = "transparent"

    # Status colors
    RECORDING = "#e74c3c"  # Red for recording indicator
    PREVIEW_PLACEHOLDER = "#2c3e50"  # Dark gray for no image

    @classmethod
    def get_light_palette(cls) -> dict[str, str]:
        """Get light theme color palette."""
        return {
            "background": cls.LIGHT_BACKGROUND,
            "surface": cls.LIGHT_SURFACE,
            "border": cls.LIGHT_BORDER,
            "text": cls.LIGHT_TEXT,
            "text_secondary": cls.LIGHT_TEXT_SECONDARY,
            "primary": cls.PRIMARY,
            "success": cls.SUCCESS,
            "warning": cls.WARNING,
            "danger": cls.DANGER,
            "info": cls.INFO,
        }

    @classmethod
    def get_dark_palette(cls) -> dict[str, str]:
        """Get dark theme color palette."""
        return {
            "background": cls.DARK_BACKGROUND,
            "surface": cls.DARK_SURFACE,
            "border": cls.DARK_BORDER,
            "text": cls.DARK_TEXT,
            "text_secondary": cls.DARK_TEXT_SECONDARY,
            "primary": cls.PRIMARY,
            "success": cls.SUCCESS,
            "warning": cls.WARNING,
            "danger": cls.DANGER,
            "info": cls.INFO,
        }
