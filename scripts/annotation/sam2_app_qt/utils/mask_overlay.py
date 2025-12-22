"""
Mask overlay drawing utilities.

Provides functions for drawing masks and points on images.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
    draw_bbox: bool = True,
) -> np.ndarray:
    """
    Draw mask overlay on image.

    Args:
        image: RGB image array
        mask: Boolean mask array
        color: Overlay color (RGB)
        alpha: Overlay transparency
        draw_bbox: Whether to draw bounding box

    Returns:
        Image with mask overlay
    """
    display = image.copy()
    mask_bool = mask.astype(bool)

    # Create overlay
    mask_overlay = np.zeros_like(display)
    mask_overlay[mask_bool] = color
    display = cv2.addWeighted(display, 1.0, mask_overlay, alpha, 0)

    # Draw bounding box
    if draw_bbox:
        bbox = _mask_to_bbox(mask)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    return display


def draw_points(
    image: np.ndarray,
    foreground_points: List[Tuple[int, int]],
    background_points: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Draw annotation points on image.

    Args:
        image: RGB image array
        foreground_points: List of (x, y) foreground points
        background_points: List of (x, y) background points

    Returns:
        Image with points drawn
    """
    display = image.copy()

    # Draw foreground points (green with plus)
    for x, y in foreground_points:
        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        cv2.circle(display, (x, y), 5, (255, 255, 255), 2)
        cv2.line(display, (x - 3, y), (x + 3, y), (255, 255, 255), 2)
        cv2.line(display, (x, y - 3), (x, y + 3), (255, 255, 255), 2)

    # Draw background points (red with minus)
    for x, y in background_points:
        cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(display, (x, y), 5, (255, 255, 255), 2)
        cv2.line(display, (x - 3, y), (x + 3, y), (255, 255, 255), 2)

    return display


def _mask_to_bbox(
    mask: np.ndarray, use_contour: bool = True
) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box from mask.

    Args:
        mask: Boolean mask array
        use_contour: Whether to use contour-based bbox

    Returns:
        Bounding box (x1, y1, x2, y2) or None
    """
    if mask is None or not np.any(mask):
        return None

    mask_uint8 = mask.astype(np.uint8) * 255

    if use_contour:
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        return (x, y, x + w, y + h)
    else:
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(x1), int(y1), int(x2), int(y2))
