"""
HSR Perception - Image Utilities

Common image processing utilities for annotation, training, and evaluation.
Consolidates mask-to-bbox conversion and visualization code from multiple files.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .constants import (
    DEFAULT_BBOX_COLOR,
    DEFAULT_BBOX_MARGIN_RATIO,
    DEFAULT_BBOX_THICKNESS,
    DEFAULT_FONT_SCALE,
    DEFAULT_FONT_THICKNESS,
    DEFAULT_MASK_ALPHA,
    DEFAULT_MASK_COLOR,
    IMAGE_EXTENSIONS,
)


def mask_to_bbox(
    mask: np.ndarray,
    image_shape: Tuple[int, int],
    bbox_margin_ratio: float = DEFAULT_BBOX_MARGIN_RATIO,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert binary mask to bounding box with optional margin.

    Args:
        mask: Binary mask array (2D numpy array)
        image_shape: (height, width) of the original image
        bbox_margin_ratio: Ratio of bbox size to add as margin (default 0.02)

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in pixel coordinates,
        or None if no valid region found.
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        # Fallback: use mask bounds directly
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
    else:
        # Get bounding rect of largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h

    # Add margin
    img_h, img_w = image_shape
    margin_x = int((x_max - x_min) * bbox_margin_ratio)
    margin_y = int((y_max - y_min) * bbox_margin_ratio)

    x_min = max(0, x_min - margin_x)
    y_min = max(0, y_min - margin_y)
    x_max = min(img_w, x_max + margin_x)
    y_max = min(img_h, y_max + margin_y)

    # Validate
    if x_max <= x_min or y_max <= y_min:
        return None

    return (x_min, y_min, x_max, y_max)


def find_object_bbox(
    foreground_mask: np.ndarray,
    image_shape: Tuple[int, int],
    min_contour_area: int = 500,
    max_contour_area_ratio: float = 0.9,
    bbox_margin_ratio: float = DEFAULT_BBOX_MARGIN_RATIO,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find object bounding box from a foreground mask.

    Used by background subtraction annotator to detect objects.

    Args:
        foreground_mask: Binary mask where foreground is non-zero
        image_shape: (height, width) of the original image
        min_contour_area: Minimum contour area in pixels
        max_contour_area_ratio: Maximum ratio of contour area to image area
        bbox_margin_ratio: Ratio of bbox size to add as margin

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in pixel coordinates,
        or None if no valid object found.
    """
    img_h, img_w = image_shape
    max_area = img_h * img_w * max_contour_area_ratio

    # Find contours
    contours, _ = cv2.findContours(
        foreground_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    # Filter by area and find largest valid contour
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area <= area <= max_area:
            valid_contours.append((contour, area))

    if not valid_contours:
        return None

    # Get largest valid contour
    largest_contour = max(valid_contours, key=lambda x: x[1])[0]
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add margin
    margin_x = int(w * bbox_margin_ratio)
    margin_y = int(h * bbox_margin_ratio)

    x_min = max(0, x - margin_x)
    y_min = max(0, y - margin_y)
    x_max = min(img_w, x + w + margin_x)
    y_max = min(img_h, y + h + margin_y)

    return (x_min, y_min, x_max, y_max)


def draw_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = DEFAULT_BBOX_COLOR,
    thickness: int = DEFAULT_BBOX_THICKNESS,
    font_scale: float = DEFAULT_FONT_SCALE,
    font_thickness: int = DEFAULT_FONT_THICKNESS,
) -> np.ndarray:
    """
    Draw bounding box with optional label on image.

    Args:
        image: Image to draw on (modified in place)
        bbox: (x_min, y_min, x_max, y_max) in pixel coordinates
        label: Optional text label to display above box
        color: BGR color tuple
        thickness: Line thickness
        font_scale: Font scale for label
        font_thickness: Font thickness for label

    Returns:
        Image with bounding box drawn (same as input, modified in place)
    """
    x_min, y_min, x_max, y_max = bbox

    # Draw rectangle
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    # Draw label if provided
    if label:
        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # Draw label background
        label_y = max(y_min - 5, text_height + 5)
        cv2.rectangle(
            image,
            (x_min, label_y - text_height - 5),
            (x_min + text_width + 5, label_y + 5),
            color,
            -1,  # Filled
        )

        # Draw label text
        cv2.putText(
            image,
            label,
            (x_min + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness,
        )

    return image


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = DEFAULT_MASK_COLOR,
    alpha: float = DEFAULT_MASK_ALPHA,
) -> np.ndarray:
    """
    Draw semi-transparent mask overlay on image.

    Args:
        image: Image to draw on (modified in place)
        mask: Binary mask array
        color: BGR color tuple for mask
        alpha: Transparency (0.0 = transparent, 1.0 = opaque)

    Returns:
        Image with mask overlay (same as input, modified in place)
    """
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color

    # Blend with original image
    mask_indices = mask > 0
    image[mask_indices] = cv2.addWeighted(
        image, 1 - alpha, colored_mask, alpha, 0
    )[mask_indices]

    return image


def draw_detections(
    image: np.ndarray,
    detections: List[dict],
    color_map: Optional[dict] = None,
    show_confidence: bool = True,
    show_class_name: bool = True,
    thickness: int = DEFAULT_BBOX_THICKNESS,
) -> np.ndarray:
    """
    Draw multiple detection boxes on an image.

    Args:
        image: Image to draw on (modified in place)
        detections: List of detection dicts with keys:
            - bbox: (x_min, y_min, x_max, y_max)
            - class_id: int
            - class_name: str (optional)
            - confidence: float (optional)
        color_map: Dict mapping class_id to BGR color
        show_confidence: Show confidence scores
        show_class_name: Show class names

    Returns:
        Image with detections drawn
    """
    for det in detections:
        bbox = det.get("bbox")
        if bbox is None:
            continue

        class_id = det.get("class_id", 0)
        class_name = det.get("class_name", f"class_{class_id}")
        confidence = det.get("confidence", None)

        # Get color
        if color_map and class_id in color_map:
            color = color_map[class_id]
        else:
            # Generate deterministic color from class_id
            np.random.seed(class_id)
            color = tuple(int(c) for c in np.random.randint(0, 255, 3))

        # Build label
        label_parts = []
        if show_class_name:
            label_parts.append(class_name)
        if show_confidence and confidence is not None:
            label_parts.append(f"{confidence:.2f}")
        label = " ".join(label_parts) if label_parts else ""

        draw_bbox(image, bbox, label=label, color=color, thickness=thickness)

    return image


def list_image_files(
    directory: Union[str, Path],
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    List image files in a directory.

    Args:
        directory: Directory to search
        extensions: List of extensions to include (default: IMAGE_EXTENSIONS)
        recursive: If True, search subdirectories

    Returns:
        List of Path objects for found image files
    """
    directory = Path(directory)
    extensions = extensions or IMAGE_EXTENSIONS

    if not directory.exists():
        return []

    # Normalize extensions
    ext_set = set()
    for ext in extensions:
        ext_lower = ext.lower()
        if not ext_lower.startswith("."):
            ext_lower = "." + ext_lower
        ext_set.add(ext_lower)
        ext_set.add(ext_lower.upper())

    # Find files
    image_files = []
    if recursive:
        for ext in ext_set:
            image_files.extend(directory.rglob(f"*{ext}"))
    else:
        for ext in ext_set:
            image_files.extend(directory.glob(f"*{ext}"))

    return sorted(image_files)


def load_image(
    path: Union[str, Path],
    color_mode: str = "bgr",
) -> Optional[np.ndarray]:
    """
    Load image with error handling.

    Args:
        path: Path to image file
        color_mode: "bgr" (default), "rgb", or "gray"

    Returns:
        Image array or None if loading failed
    """
    image = cv2.imread(str(path))
    if image is None:
        return None

    if color_mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95,
) -> bool:
    """
    Save image with error handling.

    Args:
        image: Image array to save
        path: Output path
        quality: JPEG quality (0-100)

    Returns:
        True if successful, False otherwise
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    params = []
    if path.suffix.lower() in [".jpg", ".jpeg"]:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    elif path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 12)]

    return cv2.imwrite(str(path), image, params)
