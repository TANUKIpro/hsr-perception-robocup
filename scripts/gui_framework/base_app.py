"""
Base Application Class.

Provides a foundation for Tkinter-based GUI applications.
"""

import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import ttk

from gui_framework.styles.theme import AppTheme


class BaseApp(ABC):
    """
    Base class for Tkinter GUI applications.

    Provides common window setup, styling, and lifecycle management.

    Subclasses must implement:
        - _build_gui(): Build the GUI layout
        - _on_close(): Handle window close event

    Example:
        class MyApp(BaseApp):
            def _build_gui(self):
                # Build your GUI here
                pass

            def _on_close(self):
                self.root.destroy()

        if __name__ == "__main__":
            root = tk.Tk()
            app = MyApp(root, "My App", "800x600")
            app.run()
    """

    def __init__(
        self,
        root: tk.Tk,
        title: str,
        geometry: str = "800x600",
        min_size: tuple[int, int] | None = None,
        resizable: tuple[bool, bool] = (True, True),
    ) -> None:
        """
        Initialize the base application.

        Args:
            root: Tkinter root window
            title: Window title
            geometry: Window geometry string (e.g., "800x600")
            min_size: Minimum window size as (width, height)
            resizable: Tuple of (width_resizable, height_resizable)
        """
        self.root = root
        self.style = ttk.Style()

        # Setup window
        self._setup_window(title, geometry, min_size, resizable)

        # Apply theme
        self._setup_style()

        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Bind common keyboard shortcuts
        self._setup_keyboard_shortcuts()

        # Build GUI (implemented by subclass)
        self._build_gui()

    def _setup_window(
        self,
        title: str,
        geometry: str,
        min_size: tuple[int, int] | None,
        resizable: tuple[bool, bool],
    ) -> None:
        """Configure the main window."""
        self.root.title(title)
        self.root.geometry(geometry)
        self.root.resizable(*resizable)

        if min_size:
            self.root.minsize(*min_size)

    def _setup_style(self) -> None:
        """Apply the application theme."""
        AppTheme.apply(self.style)

    def _setup_keyboard_shortcuts(self) -> None:
        """Setup common keyboard shortcuts."""
        # Quit on 'q' or 'Q'
        self.root.bind("<q>", lambda e: self._on_close())
        self.root.bind("<Q>", lambda e: self._on_close())

        # Escape to cancel/close
        self.root.bind("<Escape>", self._on_escape)

    def _on_escape(self, event) -> None:
        """
        Handle Escape key press.

        Override this method to customize Escape behavior.
        Default behavior does nothing.
        """
        pass

    @abstractmethod
    def _build_gui(self) -> None:
        """
        Build the GUI layout.

        This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def _on_close(self) -> None:
        """
        Handle window close event.

        This method must be implemented by subclasses.
        Typically should call self.root.destroy() at the end.
        """
        pass

    def run(self) -> None:
        """Start the main event loop."""
        self.root.mainloop()

    def set_status(self, message: str) -> None:
        """
        Set status message (if status bar is available).

        Override this method if your app has a status bar.

        Args:
            message: Status message to display
        """
        pass

    def show_error(self, title: str, message: str) -> None:
        """
        Show an error message dialog.

        Args:
            title: Dialog title
            message: Error message
        """
        from tkinter import messagebox

        messagebox.showerror(title, message)

    def show_warning(self, title: str, message: str) -> None:
        """
        Show a warning message dialog.

        Args:
            title: Dialog title
            message: Warning message
        """
        from tkinter import messagebox

        messagebox.showwarning(title, message)

    def show_info(self, title: str, message: str) -> None:
        """
        Show an info message dialog.

        Args:
            title: Dialog title
            message: Info message
        """
        from tkinter import messagebox

        messagebox.showinfo(title, message)

    def ask_yes_no(self, title: str, message: str) -> bool:
        """
        Show a yes/no confirmation dialog.

        Args:
            title: Dialog title
            message: Question message

        Returns:
            True if user clicked Yes, False otherwise
        """
        from tkinter import messagebox

        return messagebox.askyesno(title, message)
