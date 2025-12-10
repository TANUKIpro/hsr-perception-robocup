"""
Preview Panel Component.

Provides a real-time image preview display with overlay support.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


class PreviewPanel(ttk.Frame):
    """
    Preview panel for displaying real-time images.

    Handles image conversion, resizing, and optional overlays.
    Updates at approximately 30 FPS.

    Example:
        def get_frame():
            return ros_node.get_frame()

        preview = PreviewPanel(parent, frame_callback=get_frame)
        preview.pack(fill=tk.BOTH, expand=True)
        preview.start()
    """

    def __init__(
        self,
        parent: tk.Widget,
        frame_callback: Optional[Callable[[], Optional[np.ndarray]]] = None,
        fps: int = 30,
        placeholder_text: str = "No Image",
        **kwargs,
    ) -> None:
        """
        Initialize the preview panel.

        Args:
            parent: Parent widget
            frame_callback: Callback function that returns the current frame
            fps: Target frames per second
            placeholder_text: Text to display when no image is available
            **kwargs: Additional arguments passed to ttk.Frame
        """
        super().__init__(parent, **kwargs)

        self._frame_callback = frame_callback
        self._overlay_callback: Optional[
            Callable[[np.ndarray], np.ndarray]
        ] = None
        self._fps = fps
        self._update_interval = int(1000 / fps)
        self._active = False
        self._placeholder_text = placeholder_text

        # Preview label
        self.preview_label = ttk.Label(
            self,
            text=placeholder_text,
            anchor="center",
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Store current photo reference to prevent garbage collection
        self._current_photo: Optional[ImageTk.PhotoImage] = None

    def set_frame_callback(
        self, callback: Callable[[], Optional[np.ndarray]]
    ) -> None:
        """
        Set the frame callback function.

        Args:
            callback: Function that returns the current frame
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

    def start(self) -> None:
        """Start the preview update loop."""
        if not self._active:
            self._active = True
            self._update()

    def stop(self) -> None:
        """Stop the preview update loop."""
        self._active = False

    def is_active(self) -> bool:
        """Check if preview is currently active."""
        return self._active

    def _update(self) -> None:
        """Update the preview image."""
        if not self._active:
            return

        frame = None
        if self._frame_callback is not None:
            frame = self._frame_callback()

        if frame is not None:
            # Apply overlay if set
            if self._overlay_callback is not None:
                frame = self._overlay_callback(frame.copy())

            # Resize to fit display area
            frame = self._resize_for_display(frame)

            # Convert to PhotoImage
            self._current_photo = self._numpy_to_photoimage(frame)
            self.preview_label.configure(
                image=self._current_photo,
                text="",
            )
        else:
            self.preview_label.configure(
                image="",
                text=self._placeholder_text,
            )

        # Schedule next update
        self.after(self._update_interval, self._update)

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame to fit display area while maintaining aspect ratio.

        Args:
            frame: Input BGR image

        Returns:
            Resized frame
        """
        # Get current label size
        label_width = self.preview_label.winfo_width()
        label_height = self.preview_label.winfo_height()

        # Skip if size is too small
        if label_width <= 1 or label_height <= 1:
            return frame

        h, w = frame.shape[:2]
        scale_w = label_width / w
        scale_h = label_height / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != w or new_h != h:
            frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        return frame

    def _numpy_to_photoimage(self, frame: np.ndarray) -> ImageTk.PhotoImage:
        """
        Convert numpy array to PhotoImage.

        Args:
            frame: BGR numpy array

        Returns:
            PhotoImage for Tkinter display
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        return ImageTk.PhotoImage(image)

    def get_display_size(self) -> tuple[int, int]:
        """
        Get current display area size.

        Returns:
            Tuple of (width, height)
        """
        return (
            self.preview_label.winfo_width(),
            self.preview_label.winfo_height(),
        )

    def update_single_frame(self, frame: np.ndarray) -> None:
        """
        Update display with a single frame (for non-continuous use).

        Args:
            frame: BGR numpy array to display
        """
        if self._overlay_callback is not None:
            frame = self._overlay_callback(frame.copy())

        frame = self._resize_for_display(frame)
        self._current_photo = self._numpy_to_photoimage(frame)
        self.preview_label.configure(
            image=self._current_photo,
            text="",
        )
