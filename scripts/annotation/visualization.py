"""
Annotation Visualization Utilities

Provides shared constants and functions for visualizing annotation results.
Consolidates visualization code from multiple annotator implementations.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class AnnotationColors:
    """BGR color constants for annotation visualization."""

    # Bounding boxes
    BBOX_DEFAULT = (0, 255, 0)  # Green - default/successful
    BBOX_SAVED = (0, 165, 255)  # Orange - saved annotations
    BBOX_LOW_CONF = (0, 200, 255)  # Yellow - low confidence
    BBOX_ERROR = (0, 0, 255)  # Red - error/invalid

    # Mask overlay
    MASK_OVERLAY = (0, 255, 0)  # Green

    # Points
    POINT_FG = (0, 255, 0)  # Green - foreground
    POINT_BG = (0, 0, 255)  # Red - background
    POINT_OUTLINE = (255, 255, 255)  # White - outline

    # Text
    TEXT_SUCCESS = (0, 255, 0)  # Green
    TEXT_ERROR = (0, 0, 255)  # Red
    TEXT_INFO = (255, 255, 255)  # White


class AnnotationStyle:
    """Style constants for annotation visualization."""

    BBOX_THICKNESS = 2
    POINT_RADIUS = 5
    POINT_OUTLINE_THICKNESS = 2
    MASK_ALPHA = 0.3
    MASK_ALPHA_LOW_CONF = 0.4
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.5
    FONT_SCALE_LARGE = 1.0
    FONT_THICKNESS = 2


def draw_bbox_on_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = AnnotationColors.BBOX_DEFAULT,
    thickness: int = AnnotationStyle.BBOX_THICKNESS,
    label: Optional[str] = None,
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: Image array (BGR)
        bbox: Bounding box as (x_min, y_min, x_max, y_max) in pixels
        color: BGR color tuple
        thickness: Line thickness
        label: Optional label text to display above box

    Returns:
        Image with bounding box drawn
    """
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    if label:
        cv2.putText(
            image,
            label,
            (x_min, y_min - 10),
            AnnotationStyle.FONT,
            AnnotationStyle.FONT_SCALE,
            color,
            AnnotationStyle.FONT_THICKNESS,
        )

    return image


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = AnnotationColors.MASK_OVERLAY,
    alpha: float = AnnotationStyle.MASK_ALPHA,
) -> np.ndarray:
    """
    Draw mask overlay on image.

    Args:
        image: Image array (BGR)
        mask: Boolean or binary mask array (H, W)
        color: BGR color tuple for overlay
        alpha: Transparency (0.0 = invisible, 1.0 = opaque)

    Returns:
        Image with mask overlay
    """
    mask_overlay = np.zeros_like(image)
    mask_bool = mask.astype(bool)
    mask_overlay[mask_bool] = color
    return cv2.addWeighted(image, 1.0, mask_overlay, alpha, 0)


def draw_point_marker(
    image: np.ndarray,
    point: Tuple[int, int],
    is_foreground: bool = True,
    radius: int = AnnotationStyle.POINT_RADIUS,
    with_outline: bool = True,
) -> np.ndarray:
    """
    Draw point marker on image.

    Args:
        image: Image array (BGR)
        point: (x, y) coordinates
        is_foreground: If True, use foreground color; else background color
        radius: Point radius
        with_outline: If True, draw white outline

    Returns:
        Image with point drawn
    """
    x, y = point
    color = AnnotationColors.POINT_FG if is_foreground else AnnotationColors.POINT_BG

    cv2.circle(image, (x, y), radius, color, -1)
    if with_outline:
        cv2.circle(
            image, (x, y), radius, AnnotationColors.POINT_OUTLINE,
            AnnotationStyle.POINT_OUTLINE_THICKNESS
        )

    return image


def draw_status_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = AnnotationColors.TEXT_ERROR,
    scale: float = AnnotationStyle.FONT_SCALE_LARGE,
) -> np.ndarray:
    """
    Draw status text on image.

    Args:
        image: Image array (BGR)
        text: Text to display
        position: (x, y) position for text
        color: BGR color tuple
        scale: Font scale

    Returns:
        Image with text drawn
    """
    cv2.putText(
        image,
        text,
        position,
        AnnotationStyle.FONT,
        scale,
        color,
        AnnotationStyle.FONT_THICKNESS,
    )
    return image


def visualize_annotation_result(
    image: np.ndarray,
    bbox_yolo: Optional[Tuple[float, float, float, float]] = None,
    mask: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = AnnotationColors.BBOX_DEFAULT,
    show_yolo_label: bool = True,
) -> np.ndarray:
    """
    Visualize annotation result with bounding box and optional mask.

    Args:
        image: Image array (BGR)
        bbox_yolo: YOLO format bbox (x_center, y_center, width, height) normalized,
                   or None if no detection
        mask: Optional boolean mask array for overlay
        color: BGR color for bbox and mask
        show_yolo_label: If True, show YOLO coordinates as label

    Returns:
        Annotated image
    """
    result = image.copy()
    img_h, img_w = result.shape[:2]

    if bbox_yolo is None:
        return draw_status_text(result, "No object detected")

    # Convert YOLO to pixel coordinates
    from annotation_utils import yolo_to_bbox
    x_min, y_min, x_max, y_max = yolo_to_bbox(*bbox_yolo, img_w, img_h)

    # Draw mask overlay if provided
    if mask is not None:
        result = draw_mask_overlay(result, mask, color)

    # Draw bounding box
    label = None
    if show_yolo_label:
        label = f"({bbox_yolo[0]:.3f}, {bbox_yolo[1]:.3f}, {bbox_yolo[2]:.3f}, {bbox_yolo[3]:.3f})"

    result = draw_bbox_on_image(result, (x_min, y_min, x_max, y_max), color, label=label)

    return result
