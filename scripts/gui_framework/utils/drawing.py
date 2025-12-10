"""
Drawing Utilities for GUI Applications.

Provides common drawing functions for overlays and annotations.
"""

import cv2
import numpy as np


def draw_reticle(
    frame: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw center crosshair reticle on frame.

    Args:
        frame: Input BGR image
        color: BGR color tuple (default: green)
        thickness: Line thickness

    Returns:
        Frame with reticle drawn
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    size = min(w, h) // 20

    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)

    return frame


def draw_countdown(
    frame: np.ndarray,
    count: int,
    message: str = "Get Ready!",
) -> np.ndarray:
    """
    Draw countdown overlay on frame.

    Args:
        frame: Input BGR image
        count: Countdown number to display
        message: Message to display above the number

    Returns:
        Frame with countdown overlay drawn
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

    # Large countdown number in center
    text = str(count)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 8.0
    thickness = 15

    # Get text size for centering
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = cx - text_w // 2
    text_y = cy + text_h // 2

    # Draw text with outline
    cv2.putText(
        frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 5
    )
    cv2.putText(
        frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness
    )

    # Message text above the number
    ready_scale = 1.5
    (ready_w, ready_h), _ = cv2.getTextSize(message, font, ready_scale, 3)
    ready_x = cx - ready_w // 2
    ready_y = cy - text_h // 2 - 40

    cv2.putText(frame, message, (ready_x, ready_y), font, ready_scale, (0, 0, 0), 5)
    cv2.putText(
        frame, message, (ready_x, ready_y), font, ready_scale, (255, 255, 255), 3
    )

    return frame


def draw_recording_indicator(
    frame: np.ndarray,
    elapsed_seconds: float,
    frame_count: int,
    blink: bool = True,
) -> np.ndarray:
    """
    Draw recording status overlay.

    Args:
        frame: Input BGR image
        elapsed_seconds: Recording elapsed time in seconds
        frame_count: Number of frames recorded
        blink: Whether the indicator should blink

    Returns:
        Frame with recording indicator drawn
    """
    import time

    # Red recording indicator (blinking)
    show_dot = not blink or int(time.time() * 2) % 2 == 0
    if show_dot:
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)

    # Recording text
    mins = int(elapsed_seconds // 60)
    secs = int(elapsed_seconds % 60)
    text = f"REC {mins:02d}:{secs:02d} | {frame_count} frames"
    cv2.putText(
        frame, text, (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    return frame


def resize_for_display(
    frame: np.ndarray,
    max_width: int,
    max_height: int,
) -> tuple[np.ndarray, float]:
    """
    Resize frame to fit display area while maintaining aspect ratio.

    Args:
        frame: Input BGR image
        max_width: Maximum display width
        max_height: Maximum display height

    Returns:
        Tuple of (resized frame, scale factor)
    """
    if max_width <= 1 or max_height <= 1:
        return frame, 1.0

    h, w = frame.shape[:2]
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w != w or new_h != h:
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return frame, scale
