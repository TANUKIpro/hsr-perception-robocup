"""
Application Theme and Style Definitions.

Provides consistent styling across all GUI applications.
"""

from tkinter import ttk


class AppTheme:
    """
    Unified theme configuration for Tkinter applications.

    Usage:
        style = ttk.Style()
        AppTheme.apply(style)
    """

    # Theme name
    THEME = "clam"

    # Font definitions
    FONTS = {
        "default": ("TkDefaultFont", 10),
        "bold": ("TkDefaultFont", 10, "bold"),
        "large": ("TkDefaultFont", 12),
        "large_bold": ("TkDefaultFont", 12, "bold"),
        "small": ("TkDefaultFont", 9),
    }

    # Color palette
    COLORS = {
        "primary": "#3498db",      # Blue
        "success": "#27ae60",      # Green
        "warning": "#f39c12",      # Orange
        "danger": "#e74c3c",       # Red
        "info": "#17a2b8",         # Cyan
        "background": "#f0f0f0",   # Light gray
        "text": "#2c3e50",         # Dark gray
        "white": "#ffffff",
        "black": "#000000",
    }

    # Padding values
    PADDING = {
        "frame": 10,
        "widget": 5,
        "button": (10, 5),
        "large_button": (20, 10),
    }

    # Window sizes
    WINDOW_SIZES = {
        "small": "640x480",
        "medium": "800x750",
        "large": "1200x800",
    }

    # Minimum window sizes
    MIN_SIZES = {
        "small": (480, 360),
        "medium": (600, 550),
        "large": (800, 600),
    }

    @classmethod
    def apply(cls, style: ttk.Style) -> None:
        """
        Apply the theme to a ttk.Style instance.

        Args:
            style: The ttk.Style instance to configure
        """
        # Use clam theme if available
        available_themes = style.theme_names()
        if cls.THEME in available_themes:
            style.theme_use(cls.THEME)

        # Configure LabelFrame
        style.configure(
            "TLabelframe",
            padding=cls.PADDING["frame"],
        )
        style.configure(
            "TLabelframe.Label",
            font=cls.FONTS["bold"],
        )

        # Configure standard Button
        style.configure(
            "TButton",
            padding=cls.PADDING["widget"],
        )

        # Configure large button style
        style.configure(
            "Large.TButton",
            font=cls.FONTS["large_bold"],
            padding=cls.PADDING["large_button"],
        )

        # Configure action buttons
        style.configure(
            "Primary.TButton",
            font=cls.FONTS["bold"],
        )
        style.map(
            "Primary.TButton",
            foreground=[("active", cls.COLORS["primary"])],
        )

        # Configure danger button (for stop/cancel actions)
        style.configure(
            "Danger.TButton",
            font=cls.FONTS["bold"],
        )
        style.map(
            "Danger.TButton",
            foreground=[("active", cls.COLORS["danger"])],
        )

        # Configure success button
        style.configure(
            "Success.TButton",
            font=cls.FONTS["bold"],
        )
        style.map(
            "Success.TButton",
            foreground=[("active", cls.COLORS["success"])],
        )

        # Configure ProgressBar
        style.configure(
            "TProgressbar",
            thickness=20,
        )

        # Configure Entry
        style.configure(
            "TEntry",
            padding=cls.PADDING["widget"],
        )

        # Configure Combobox
        style.configure(
            "TCombobox",
            padding=cls.PADDING["widget"],
        )

    @classmethod
    def get_font(cls, name: str) -> tuple:
        """
        Get a font definition by name.

        Args:
            name: Font name key

        Returns:
            Font tuple
        """
        return cls.FONTS.get(name, cls.FONTS["default"])

    @classmethod
    def get_color(cls, name: str) -> str:
        """
        Get a color by name.

        Args:
            name: Color name key

        Returns:
            Color hex string
        """
        return cls.COLORS.get(name, cls.COLORS["text"])

    @classmethod
    def get_window_size(cls, size: str) -> str:
        """
        Get window geometry string.

        Args:
            size: Size key ("small", "medium", "large")

        Returns:
            Geometry string (e.g., "800x600")
        """
        return cls.WINDOW_SIZES.get(size, cls.WINDOW_SIZES["medium"])

    @classmethod
    def get_min_size(cls, size: str) -> tuple[int, int]:
        """
        Get minimum window size.

        Args:
            size: Size key ("small", "medium", "large")

        Returns:
            Tuple of (min_width, min_height)
        """
        return cls.MIN_SIZES.get(size, cls.MIN_SIZES["medium"])
