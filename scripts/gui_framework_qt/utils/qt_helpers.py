"""
Qt helper functions for PyQt6 GUI framework.

Provides utility functions for common Qt operations.
"""

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt


def numpy_to_pixmap(frame: np.ndarray, is_bgr: bool = True) -> QPixmap:
    """
    Convert numpy array to QPixmap.

    Args:
        frame: Numpy array image
        is_bgr: If True, input is BGR format (OpenCV default)

    Returns:
        QPixmap for display
    """
    # Convert BGR to RGB if needed
    if is_bgr and len(frame.shape) == 3 and frame.shape[2] == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb = frame

    if len(rgb.shape) == 2:
        # Grayscale
        h, w = rgb.shape
        bytes_per_line = w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
        )
    else:
        # Color
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )

    # Copy to avoid memory issues
    return QPixmap.fromImage(qimg.copy())


def pixmap_to_numpy(pixmap: QPixmap) -> Optional[np.ndarray]:
    """
    Convert QPixmap to numpy array (RGB format).

    Args:
        pixmap: QPixmap to convert

    Returns:
        Numpy array in RGB format, or None if conversion fails
    """
    if pixmap.isNull():
        return None

    qimg = pixmap.toImage()
    qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)

    width = qimg.width()
    height = qimg.height()
    ptr = qimg.bits()
    ptr.setsize(height * width * 3)

    arr = np.array(ptr).reshape((height, width, 3))
    return arr.copy()


def scale_pixmap(
    pixmap: QPixmap,
    target_size: tuple[int, int],
    keep_aspect: bool = True,
    smooth: bool = True,
) -> QPixmap:
    """
    Scale a QPixmap to target size.

    Args:
        pixmap: QPixmap to scale
        target_size: Target (width, height)
        keep_aspect: If True, maintain aspect ratio
        smooth: If True, use smooth scaling

    Returns:
        Scaled QPixmap
    """
    if pixmap.isNull():
        return pixmap

    aspect_mode = (
        Qt.AspectRatioMode.KeepAspectRatio
        if keep_aspect
        else Qt.AspectRatioMode.IgnoreAspectRatio
    )
    transform_mode = (
        Qt.TransformationMode.SmoothTransformation
        if smooth
        else Qt.TransformationMode.FastTransformation
    )

    return pixmap.scaled(
        target_size[0], target_size[1], aspect_mode, transform_mode
    )


def create_placeholder_pixmap(
    size: tuple[int, int],
    text: str = "No Image",
    bg_color: str = "#2c3e50",
    text_color: str = "#ecf0f1",
) -> QPixmap:
    """
    Create a placeholder pixmap with text.

    Args:
        size: (width, height) of the pixmap
        text: Text to display
        bg_color: Background color (hex)
        text_color: Text color (hex)

    Returns:
        QPixmap with centered text
    """
    from PyQt6.QtGui import QPainter, QColor, QFont

    pixmap = QPixmap(size[0], size[1])
    pixmap.fill(QColor(bg_color))

    painter = QPainter(pixmap)
    painter.setPen(QColor(text_color))
    font = QFont()
    font.setPointSize(12)
    painter.setFont(font)
    painter.drawText(
        pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text
    )
    painter.end()

    return pixmap
