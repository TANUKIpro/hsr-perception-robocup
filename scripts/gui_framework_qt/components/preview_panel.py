"""
Preview panel component for PyQt6 GUI framework.

Provides a real-time image preview display with overlay support.
"""

from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap


class PreviewPanel(QWidget):
    """
    Real-time image preview panel.

    Features:
    - QTimer-based update loop (configurable FPS)
    - Overlay callback support
    - Aspect ratio preserving resize
    - BGR to RGB conversion for display

    Example:
        preview = PreviewPanel(parent, fps=30)
        preview.set_frame_callback(ros_node.get_frame)
        preview.set_overlay_callback(draw_reticle)
        preview.start()
    """

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        fps: int = 30,
        placeholder_text: str = "No Image",
    ) -> None:
        """
        Initialize the preview panel.

        Args:
            parent: Parent widget
            fps: Target frames per second
            placeholder_text: Text to display when no image is available
        """
        super().__init__(parent)

        self._frame_callback: Optional[Callable[[], Optional[np.ndarray]]] = None
        self._overlay_callback: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None
        self._placeholder_text = placeholder_text
        self._fps = fps
        self._update_interval = int(1000 / fps)

        self._build_ui()

        # Update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_frame)

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel(self._placeholder_text)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1;"
        )
        self.preview_label.setMinimumSize(320, 240)
        layout.addWidget(self.preview_label)

    def start(self) -> None:
        """Start the preview update loop."""
        if not self._timer.isActive():
            self._timer.start(self._update_interval)

    def stop(self) -> None:
        """Stop the preview update loop."""
        self._timer.stop()

    def is_active(self) -> bool:
        """Check if preview is currently active."""
        return self._timer.isActive()

    def set_frame_callback(
        self, callback: Callable[[], Optional[np.ndarray]]
    ) -> None:
        """
        Set the frame callback function.

        Args:
            callback: Function that returns the current frame (BGR format)
        """
        self._frame_callback = callback

    def set_overlay_callback(
        self, callback: Optional[Callable[[np.ndarray], np.ndarray]]
    ) -> None:
        """
        Set an overlay callback for drawing on frames.

        Args:
            callback: Function that takes a frame and returns modified frame
        """
        self._overlay_callback = callback

    @pyqtSlot()
    def _update_frame(self) -> None:
        """Update the preview with the latest frame."""
        if self._frame_callback is None:
            return

        frame = self._frame_callback()

        if frame is None:
            self.preview_label.setText(self._placeholder_text)
            self.preview_label.setPixmap(QPixmap())
            return

        # Apply overlay if set
        if self._overlay_callback is not None:
            frame = self._overlay_callback(frame.copy())

        # Resize for display
        frame = self._resize_for_display(frame)

        # Convert to QPixmap and display
        pixmap = self._numpy_to_pixmap(frame)
        self.preview_label.setPixmap(pixmap)
        self.preview_label.setText("")

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to fit display area while maintaining aspect ratio.

        Args:
            frame: Input BGR image

        Returns:
            Resized frame
        """
        label_w = self.preview_label.width()
        label_h = self.preview_label.height()

        if label_w <= 1 or label_h <= 1:
            return frame

        h, w = frame.shape[:2]
        scale = min(label_w / w, label_h / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != w or new_h != h:
            frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        return frame

    def _numpy_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        """
        Convert numpy array (BGR) to QPixmap.

        Args:
            frame: BGR numpy array

        Returns:
            QPixmap for display
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        # Create QImage from numpy data
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

        # Copy to avoid memory issues when numpy array is modified
        return QPixmap.fromImage(qimg.copy())

    def update_single_frame(self, frame: np.ndarray) -> None:
        """
        Update display with a single frame (for non-continuous use).

        Args:
            frame: BGR numpy array to display
        """
        if self._overlay_callback is not None:
            frame = self._overlay_callback(frame.copy())

        frame = self._resize_for_display(frame)
        pixmap = self._numpy_to_pixmap(frame)
        self.preview_label.setPixmap(pixmap)
        self.preview_label.setText("")

    def get_display_size(self) -> tuple[int, int]:
        """
        Get current display area size.

        Returns:
            Tuple of (width, height)
        """
        return (
            self.preview_label.width(),
            self.preview_label.height(),
        )

    def set_fps(self, fps: int) -> None:
        """
        Set the target FPS.

        Args:
            fps: Target frames per second
        """
        self._fps = fps
        self._update_interval = int(1000 / fps)
        if self._timer.isActive():
            self._timer.setInterval(self._update_interval)
