"""
Image canvas widget for interactive annotation.

Provides a QLabel-based canvas for displaying images and handling
mouse clicks for point-based annotation.
"""

from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPixmap, QImage

from sam2_app_qt.utils.coordinate import canvas_to_image_coords


class ImageCanvas(QLabel):
    """
    Interactive image canvas for annotation.

    Signals:
        left_clicked: Emitted with (x, y) image coordinates on left click
        right_clicked: Emitted with (x, y) image coordinates on right click
        resized: Emitted when canvas is resized
    """

    left_clicked = pyqtSignal(int, int)
    right_clicked = pyqtSignal(int, int)
    resized = pyqtSignal()

    def __init__(self, parent: Optional[object] = None):
        """Initialize image canvas."""
        super().__init__(parent)

        # Display state
        self.current_image: Optional[np.ndarray] = None
        self.scale_factor: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0

        # Setup widget
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: #2c3e50;")
        self.setMinimumSize(400, 300)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setText("No Image - Load images to start")

    def set_image(self, image: np.ndarray) -> None:
        """
        Set the current image for display.

        Args:
            image: RGB image array
        """
        self.current_image = image
        self._update_display()

    def update_display(self, display_image: np.ndarray) -> None:
        """
        Update display with processed image.

        Args:
            display_image: RGB image array with overlays
        """
        if display_image is None:
            return

        canvas_w = self.width()
        canvas_h = self.height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        img_h, img_w = display_image.shape[:2]

        # Calculate scale
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale_factor = min(scale_w, scale_h)

        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        # Calculate offset for centering
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        # Resize and convert to QPixmap
        resized = cv2.resize(
            display_image, (new_w, new_h),
            interpolation=cv2.INTER_LINEAR
        )

        pixmap = self._numpy_to_pixmap(resized)
        self.setPixmap(pixmap)

    def get_image_coords(
        self, canvas_x: int, canvas_y: int
    ) -> Optional[Tuple[int, int]]:
        """
        Convert canvas coordinates to image coordinates.

        Args:
            canvas_x: X coordinate on canvas
            canvas_y: Y coordinate on canvas

        Returns:
            Image coordinates (x, y) or None if outside bounds
        """
        if self.current_image is None:
            return None

        img_h, img_w = self.current_image.shape[:2]
        return canvas_to_image_coords(
            canvas_x, canvas_y,
            self.offset_x, self.offset_y,
            self.scale_factor,
            img_w, img_h,
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press events."""
        coords = self.get_image_coords(
            int(event.position().x()),
            int(event.position().y()),
        )

        if coords is None:
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.left_clicked.emit(coords[0], coords[1])
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(coords[0], coords[1])

    def resizeEvent(self, event) -> None:
        """Handle resize events."""
        super().resizeEvent(event)
        self.resized.emit()

    def _numpy_to_pixmap(self, image: np.ndarray) -> QPixmap:
        """Convert numpy array to QPixmap."""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimg = QImage(
            image.data, w, h, bytes_per_line,
            QImage.Format.Format_RGB888
        )
        return QPixmap.fromImage(qimg.copy())
