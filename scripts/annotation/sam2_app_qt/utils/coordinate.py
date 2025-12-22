"""
Coordinate transformation utilities.

Provides functions for converting between canvas and image coordinates.
"""

from typing import Optional, Tuple


def canvas_to_image_coords(
    canvas_x: int,
    canvas_y: int,
    offset_x: int,
    offset_y: int,
    scale_factor: float,
    img_width: int,
    img_height: int,
) -> Optional[Tuple[int, int]]:
    """
    Convert canvas coordinates to image coordinates.

    Args:
        canvas_x: X coordinate on canvas
        canvas_y: Y coordinate on canvas
        offset_x: X offset of image on canvas
        offset_y: Y offset of image on canvas
        scale_factor: Scale factor between canvas and image
        img_width: Original image width
        img_height: Original image height

    Returns:
        Image coordinates (x, y) or None if outside bounds
    """
    # Remove offset and scale
    img_x = int((canvas_x - offset_x) / scale_factor)
    img_y = int((canvas_y - offset_y) / scale_factor)

    # Check bounds
    if 0 <= img_x < img_width and 0 <= img_y < img_height:
        return (img_x, img_y)

    return None
