"""
Status bar component for PyQt6 GUI framework.

Provides a status message display with optional progress bar.
"""

from typing import Optional

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt


class StatusBar(QWidget):
    """
    Status bar with message and progress display.

    Features:
    - Status message display
    - Progress bar (determinate and indeterminate modes)
    - Show/hide progress bar

    Example:
        status_bar = StatusBar(parent)
        status_bar.set_status("Processing...")
        status_bar.show_progress(50, 100)
        status_bar.hide_progress()
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """
        Initialize the status bar.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(10)

        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        layout.addWidget(self.status_label, stretch=1)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimumWidth(150)
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

    def set_status(self, message: str) -> None:
        """
        Set the status message.

        Args:
            message: Status message to display
        """
        self.status_label.setText(message)

    def get_status(self) -> str:
        """
        Get the current status message.

        Returns:
            Current status message
        """
        return self.status_label.text()

    def show_progress(
        self,
        value: Optional[float] = None,
        maximum: float = 100,
        indeterminate: bool = False,
    ) -> None:
        """
        Show the progress bar.

        Args:
            value: Current progress value (0-maximum)
            maximum: Maximum progress value
            indeterminate: If True, show indeterminate progress
        """
        self.progress_bar.show()

        if indeterminate:
            # Indeterminate mode (0, 0 range)
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, int(maximum))
            if value is not None:
                self.progress_bar.setValue(int(value))

    def update_progress(self, value: float) -> None:
        """
        Update the progress bar value.

        Args:
            value: Progress value (0-maximum)
        """
        self.progress_bar.setValue(int(value))

    def hide_progress(self) -> None:
        """Hide the progress bar and reset its value."""
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        self.progress_bar.setRange(0, 100)

    def reset(self) -> None:
        """Reset status bar to default state."""
        self.set_status("Ready")
        self.hide_progress()
