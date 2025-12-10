"""
Status Bar Component.

Provides a status message display with optional progress bar.
"""

import tkinter as tk
from tkinter import ttk


class StatusBar(ttk.Frame):
    """
    Status bar widget with status message and progress bar.

    Example:
        status_bar = StatusBar(parent)
        status_bar.pack(fill=tk.X)
        status_bar.set_status("Processing...")
        status_bar.show_progress(50, 100)
    """

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        """
        Initialize the status bar.

        Args:
            parent: Parent widget
            **kwargs: Additional arguments passed to ttk.Frame
        """
        super().__init__(parent, **kwargs)

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            self,
            textvariable=self.status_var,
            relief="sunken",
            padding=(5, 2),
        )
        self.status_label.pack(fill=tk.X)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self,
            variable=self.progress_var,
            maximum=100,
        )
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.progress_bar.pack_forget()  # Hidden by default

        self._progress_visible = False

    def set_status(self, message: str) -> None:
        """
        Set the status message.

        Args:
            message: Status message to display
        """
        self.status_var.set(message)

    def get_status(self) -> str:
        """
        Get the current status message.

        Returns:
            Current status message
        """
        return self.status_var.get()

    def show_progress(
        self,
        value: float | None = None,
        maximum: float = 100,
        indeterminate: bool = False,
    ) -> None:
        """
        Show and update the progress bar.

        Args:
            value: Current progress value (None for indeterminate mode)
            maximum: Maximum progress value
            indeterminate: If True, show indeterminate animation
        """
        if not self._progress_visible:
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self._progress_visible = True

        if indeterminate:
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start(10)
        else:
            self.progress_bar.configure(mode="determinate", maximum=maximum)
            self.progress_bar.stop()
            if value is not None:
                self.progress_var.set(value)

    def update_progress(self, value: float) -> None:
        """
        Update the progress bar value.

        Args:
            value: New progress value
        """
        self.progress_var.set(value)

    def hide_progress(self) -> None:
        """Hide the progress bar."""
        if self._progress_visible:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
            self._progress_visible = False
            self.progress_var.set(0)

    def reset(self) -> None:
        """Reset status bar to initial state."""
        self.set_status("Ready")
        self.hide_progress()
