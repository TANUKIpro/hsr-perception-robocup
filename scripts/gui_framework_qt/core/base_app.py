"""
Base Application Class for PyQt6.

Provides a foundation for PyQt6-based GUI applications with common
window setup, keyboard shortcuts, and dialog helpers.
"""

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut, QCloseEvent
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QMessageBox,
    QFileDialog,
)


class BaseApp(QMainWindow):
    """
    Base class for PyQt6 GUI applications.

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
                # Cleanup code here
                pass

        if __name__ == "__main__":
            app = QApplication(sys.argv)
            window = MyApp(title="My App", size=(800, 600))
            window.show()
            sys.exit(app.exec())
    """

    # Signal emitted when status needs to be updated
    status_updated = pyqtSignal(str)

    def __init__(
        self,
        title: str = "Application",
        size: tuple[int, int] = (800, 600),
        min_size: Optional[tuple[int, int]] = None,
        resizable: bool = True,
    ) -> None:
        """
        Initialize the base application.

        Args:
            title: Window title
            size: Window size as (width, height)
            min_size: Minimum window size as (width, height)
            resizable: Whether the window is resizable
        """
        super().__init__()

        # Setup window
        self._setup_window(title, size, min_size, resizable)

        # Create central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        # Apply theme
        self._apply_theme()

        # Setup keyboard shortcuts
        self._setup_shortcuts()

        # Build GUI (implemented by subclass)
        self._build_gui()

    def _setup_window(
        self,
        title: str,
        size: tuple[int, int],
        min_size: Optional[tuple[int, int]],
        resizable: bool,
    ) -> None:
        """Configure the main window."""
        self.setWindowTitle(title)
        self.resize(*size)

        if min_size:
            self.setMinimumSize(*min_size)

        if not resizable:
            self.setFixedSize(*size)

    def _apply_theme(self) -> None:
        """Apply the application theme."""
        try:
            from gui_framework_qt.styles import ThemeManager

            ThemeManager.apply_to_widget(self)
        except ImportError:
            # Theme manager not available, use default styling
            pass

    def _setup_shortcuts(self) -> None:
        """Setup common keyboard shortcuts."""
        # Quit on 'q' or 'Q'
        QShortcut(QKeySequence("Q"), self, self.close)

        # Escape to cancel/close
        QShortcut(QKeySequence("Escape"), self, self._on_escape)

    def _on_escape(self) -> None:
        """
        Handle Escape key press.

        Override this method to customize Escape behavior.
        Default behavior does nothing.
        """
        pass

    def _build_gui(self) -> None:
        """
        Build the GUI layout.

        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _build_gui()")

    def _on_close(self) -> None:
        """
        Handle window close event.

        This method should be overridden by subclasses.
        Typically should perform cleanup before the window closes.
        """
        pass  # Default: no cleanup needed

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        self._on_close()
        event.accept()

    def set_status(self, message: str) -> None:
        """
        Set status message.

        Override this method if your app has a status bar.

        Args:
            message: Status message to display
        """
        self.status_updated.emit(message)

    def show_error(self, title: str, message: str) -> None:
        """
        Show an error message dialog.

        Args:
            title: Dialog title
            message: Error message
        """
        QMessageBox.critical(self, title, message)

    def show_warning(self, title: str, message: str) -> None:
        """
        Show a warning message dialog.

        Args:
            title: Dialog title
            message: Warning message
        """
        QMessageBox.warning(self, title, message)

    def show_info(self, title: str, message: str) -> None:
        """
        Show an info message dialog.

        Args:
            title: Dialog title
            message: Info message
        """
        QMessageBox.information(self, title, message)

    def ask_yes_no(self, title: str, message: str) -> bool:
        """
        Show a yes/no confirmation dialog.

        Args:
            title: Dialog title
            message: Question message

        Returns:
            True if user clicked Yes, False otherwise
        """
        result = QMessageBox.question(
            self,
            title,
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return result == QMessageBox.StandardButton.Yes

    def ask_directory(
        self,
        title: str = "Select Directory",
        initial_dir: str = "",
    ) -> Optional[str]:
        """
        Show a directory selection dialog.

        Args:
            title: Dialog title
            initial_dir: Initial directory path

        Returns:
            Selected directory path, or None if cancelled
        """
        directory = QFileDialog.getExistingDirectory(
            self,
            title,
            initial_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        return directory if directory else None

    def ask_open_file(
        self,
        title: str = "Open File",
        initial_dir: str = "",
        filter: str = "All Files (*)",
    ) -> Optional[str]:
        """
        Show a file open dialog.

        Args:
            title: Dialog title
            initial_dir: Initial directory path
            filter: File filter string (e.g., "Images (*.png *.jpg)")

        Returns:
            Selected file path, or None if cancelled
        """
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            title,
            initial_dir,
            filter,
        )
        return filepath if filepath else None
