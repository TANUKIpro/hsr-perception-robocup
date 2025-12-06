#!/usr/bin/env python3
"""
SAM3 Tracker Annotator

Tracking-based auto-annotation using SAM3 Video API.
Select an object in the first frame, and track it through all subsequent frames.
"""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from annotation_utils import AnnotationResult, bbox_to_yolo, write_yolo_label


# =============================================================================
# Environment Detection
# =============================================================================


def is_colab_environment() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def is_notebook_environment() -> bool:
    """Detect if running in any Jupyter notebook environment."""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is None:
            return False
        if "IPKernelApp" not in ipython.config:
            return False
        return True
    except ImportError:
        return False


# =============================================================================
# Colab Object Selector
# =============================================================================


class ColabObjectSelector:
    """
    Interactive object selector for Google Colab using ipywidgets.

    Provides X/Y slider controls to position a cursor on the image,
    then confirms selection to return normalized coordinates.
    """

    def __init__(self, image_path: str):
        """
        Initialize selector with an image.

        Args:
            image_path: Path to the image file
        """
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.height, self.width = self.image.shape[:2]
        self.selected_point: Optional[Tuple[float, float]] = None
        self._selection_complete = False
        self._cancelled = False
        self._widgets_initialized = False

    def _setup_widgets(self):
        """Create ipywidgets UI components."""
        if self._widgets_initialized:
            return

        import ipywidgets as widgets

        # X/Y position sliders (normalized 0-1)
        self.x_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.005,
            description="X position:",
            continuous_update=True,
            readout_format=".3f",
            layout=widgets.Layout(width="400px"),
        )

        self.y_slider = widgets.FloatSlider(
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.005,
            description="Y position:",
            continuous_update=True,
            readout_format=".3f",
            layout=widgets.Layout(width="400px"),
        )

        # Confirm button
        self.confirm_button = widgets.Button(
            description="Confirm Selection",
            button_style="success",
            icon="check",
        )
        self.confirm_button.on_click(self._on_confirm)

        # Cancel button
        self.cancel_button = widgets.Button(
            description="Cancel",
            button_style="danger",
            icon="times",
        )
        self.cancel_button.on_click(self._on_cancel)

        # Status label
        self.status_label = widgets.Label(
            value="Move sliders to position cursor on object center, then click Confirm"
        )

        # Output widget for image display
        self.output = widgets.Output()

        # Observe slider changes
        self.x_slider.observe(self._update_display, names="value")
        self.y_slider.observe(self._update_display, names="value")

        self._widgets_initialized = True

    def _update_display(self, change=None):
        """Update image display with cursor at current slider position."""
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        with self.output:
            clear_output(wait=True)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(self.image_rgb)

            # Draw crosshair cursor at current position
            x_pixel = self.x_slider.value * self.width
            y_pixel = self.y_slider.value * self.height

            # Draw crosshair lines
            ax.axhline(y=y_pixel, color="lime", linewidth=1, alpha=0.7)
            ax.axvline(x=x_pixel, color="lime", linewidth=1, alpha=0.7)

            # Draw center marker
            ax.scatter(
                [x_pixel], [y_pixel], c="lime", s=200, marker="+", linewidths=3
            )
            ax.scatter([x_pixel], [y_pixel], c="red", s=50, marker="o", alpha=0.7)

            # Add coordinate text
            ax.set_title(
                f"Position: ({self.x_slider.value:.3f}, {self.y_slider.value:.3f})\n"
                f"Pixel: ({int(x_pixel)}, {int(y_pixel)})",
                fontsize=12,
            )
            ax.axis("off")

            plt.tight_layout()
            plt.show()

    def _on_confirm(self, button):
        """Handle confirm button click."""
        self.selected_point = (self.x_slider.value, self.y_slider.value)
        self._selection_complete = True
        self.status_label.value = (
            f"Selected: ({self.selected_point[0]:.3f}, {self.selected_point[1]:.3f})"
        )

    def _on_cancel(self, button):
        """Handle cancel button click."""
        self._cancelled = True
        self._selection_complete = True
        self.status_label.value = "Selection cancelled"

    def display(self):
        """Display the interactive selector UI."""
        import ipywidgets as widgets
        from IPython.display import display

        self._setup_widgets()

        # Initial display
        self._update_display()

        # Layout
        controls = widgets.VBox(
            [
                self.status_label,
                self.x_slider,
                self.y_slider,
                widgets.HBox([self.confirm_button, self.cancel_button]),
            ]
        )

        ui = widgets.VBox([self.output, controls])
        display(ui)

    def get_selection(self) -> Optional[Tuple[float, float]]:
        """
        Get the current selection.

        Returns:
            Tuple (x, y) normalized coordinates, or None if cancelled/not selected

        Raises:
            ValueError: If selection was cancelled
        """
        if self._cancelled:
            raise ValueError("Selection cancelled by user")
        if not self._selection_complete:
            return None
        return self.selected_point

    def is_complete(self) -> bool:
        """Check if selection is complete."""
        return self._selection_complete

    def is_cancelled(self) -> bool:
        """Check if selection was cancelled."""
        return self._cancelled


@dataclass
class SAM3TrackerConfig:
    """Configuration for SAM3 Tracker."""

    gpu_id: int = 0
    max_long_side: int = 1024  # Resize for 8GB VRAM optimization
    min_mask_area: int = 500
    mask_threshold: float = 0.5
    box_margin: float = 0.02


class SAM3Tracker:
    """
    Tracking-based annotator using SAM3 Video API.

    Workflow:
    1. Load image sequence as video
    2. Select object in first frame (interactive click or coordinates)
    3. Track object through all frames
    4. Generate YOLO annotations and masks
    """

    def __init__(self, config: Optional[SAM3TrackerConfig] = None):
        """
        Initialize SAM3 Tracker.

        Args:
            config: Tracker configuration (default: SAM3TrackerConfig())
        """
        self.config = config or SAM3TrackerConfig()
        self.predictor = None
        self._frame_paths: List[Path] = []
        self._original_sizes: Dict[int, Tuple[int, int]] = {}

    def _load_model(self) -> None:
        """Lazy load SAM3 video predictor."""
        if self.predictor is not None:
            return

        try:
            from sam3.model_builder import build_sam3_video_predictor
        except ImportError:
            raise ImportError(
                "SAM3 not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/sam3.git\n"
                "pip install einops decord pycocotools numba python-rapidjson"
            )

        print(f"Loading SAM3 video predictor on GPU {self.config.gpu_id}...")
        gpus_to_use = [self.config.gpu_id]
        self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
        print("SAM3 video predictor loaded successfully")

    def _resize_image(
        self,
        image: np.ndarray,
        max_long_side: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Resize image for VRAM optimization.

        Args:
            image: Input image (H, W, C)
            max_long_side: Maximum size of longer edge

        Returns:
            (resized_image, scale_factor)
        """
        h, w = image.shape[:2]
        scale = min(max_long_side / max(h, w), 1.0)
        if scale < 1.0:
            new_h, new_w = int(h * scale), int(w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return resized, scale
        return image, 1.0

    def _mask_to_bbox(
        self,
        mask: np.ndarray,
        original_shape: Tuple[int, int],
        scale_factor: float = 1.0,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert binary mask to bounding box.

        Args:
            mask: Binary mask array
            original_shape: (height, width) of original image
            scale_factor: Scale factor if image was resized

        Returns:
            (x_min, y_min, x_max, y_max) in original image coordinates, or None
        """
        # Ensure mask is binary uint8
        if mask.dtype == bool:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = (mask > self.config.mask_threshold).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            # Fallback: use mask bounds directly
            rows = np.any(mask_uint8 > 0, axis=1)
            cols = np.any(mask_uint8 > 0, axis=0)
            if not np.any(rows) or not np.any(cols):
                return None

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            y_min, y_max = int(y_indices[0]), int(y_indices[-1])
            x_min, x_max = int(x_indices[0]), int(x_indices[-1])
        else:
            # Get bounding rect of largest contour
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) < self.config.min_mask_area:
                return None
            x, y, w, h = cv2.boundingRect(largest)
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h

        # Scale back to original size if needed
        if scale_factor < 1.0:
            x_min = int(x_min / scale_factor)
            y_min = int(y_min / scale_factor)
            x_max = int(x_max / scale_factor)
            y_max = int(y_max / scale_factor)

        # Add margin
        orig_h, orig_w = original_shape
        margin_x = int((x_max - x_min) * self.config.box_margin)
        margin_y = int((y_max - y_min) * self.config.box_margin)

        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(orig_w, x_max + margin_x)
        y_max = min(orig_h, y_max + margin_y)

        # Validate
        if x_max <= x_min or y_max <= y_min:
            return None

        return (x_min, y_min, x_max, y_max)

    def _extract_cutout(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Extract object cutout with transparent background.

        Args:
            image: Original image (H, W, C) BGR
            mask: Binary mask
            bbox: (x_min, y_min, x_max, y_max)

        Returns:
            RGBA image with transparent background
        """
        x_min, y_min, x_max, y_max = bbox

        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # Create RGBA image
        bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Set alpha from mask
        if mask.dtype == bool:
            bgra[:, :, 3] = mask.astype(np.uint8) * 255
        else:
            bgra[:, :, 3] = (mask > 0).astype(np.uint8) * 255

        # Crop to bbox
        cutout = bgra[y_min:y_max, x_min:x_max]

        return cutout

    def start_session(self, input_path: str) -> str:
        """
        Start SAM3 video session.

        Args:
            input_path: Path to JPEG folder or MP4 video file

        Returns:
            session_id
        """
        self._load_model()

        response = self.predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=input_path,
            )
        )
        return response["session_id"]

    def select_object_by_click(
        self,
        session_id: str,
        frame_index: int,
        x: float,
        y: float,
        obj_id: int = 1,
    ) -> Dict:
        """
        Select object at click position.

        Args:
            session_id: Session ID from start_session
            frame_index: Frame index (0-based)
            x: Click X position (normalized 0-1)
            y: Click Y position (normalized 0-1)
            obj_id: Object ID for tracking

        Returns:
            Response with initial mask
        """
        points = torch.tensor([[x, y]], dtype=torch.float32)
        point_labels = torch.tensor([1], dtype=torch.int32)  # 1 = positive

        response = self.predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                points=points,
                point_labels=point_labels,
                obj_id=obj_id,
            )
        )
        return response

    def select_object_interactive(
        self,
        session_id: str,
        first_frame_path: str,
        obj_id: int = 1,
    ) -> Dict:
        """
        Interactively select object with mouse click or Colab sliders.

        In Colab environment, returns a ColabObjectSelector for async selection.
        In local environment, uses OpenCV GUI for synchronous selection.

        Args:
            session_id: Session ID
            first_frame_path: Path to first frame image
            obj_id: Object ID

        Returns:
            Response with initial mask (local) or selector info (Colab)
        """
        if is_colab_environment():
            return self._select_object_colab(session_id, first_frame_path, obj_id)
        else:
            return self._select_object_opencv(session_id, first_frame_path, obj_id)

    def _select_object_opencv(
        self,
        session_id: str,
        first_frame_path: str,
        obj_id: int = 1,
    ) -> Dict:
        """
        Select object using OpenCV GUI (for local use).

        Args:
            session_id: Session ID
            first_frame_path: Path to first frame image
            obj_id: Object ID

        Returns:
            Response with initial mask
        """
        click_point = [None]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                click_point[0] = (x, y)

        # Load and display image
        image = cv2.imread(first_frame_path)
        if image is None:
            raise ValueError(f"Failed to load image: {first_frame_path}")

        window_name = "Click on the object to track (ESC to cancel)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)

        print("\n[SAM3 Tracker] Click on the object you want to track")
        print("              Press ESC to cancel")

        while True:
            display_img = image.copy()

            # Show crosshair on click
            if click_point[0]:
                cv2.drawMarker(
                    display_img,
                    click_point[0],
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    markerSize=30,
                    thickness=2,
                )
                cv2.putText(
                    display_img,
                    "Click detected! Press ENTER to confirm or click elsewhere",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(window_name, display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                cv2.destroyAllWindows()
                raise ValueError("Object selection cancelled by user")

            if key == 13 and click_point[0]:  # ENTER
                break

        cv2.destroyAllWindows()

        # Normalize coordinates
        h, w = image.shape[:2]
        x_norm = click_point[0][0] / w
        y_norm = click_point[0][1] / h

        print(f"[SAM3 Tracker] Selected point: ({x_norm:.3f}, {y_norm:.3f})")

        return self.select_object_by_click(session_id, 0, x_norm, y_norm, obj_id)

    def _select_object_colab(
        self,
        session_id: str,
        first_frame_path: str,
        obj_id: int = 1,
    ) -> Dict:
        """
        Select object using Colab ipywidgets UI.

        Returns a selector object for async selection in notebook.
        The notebook should call selector.get_selection() after user confirms.

        Args:
            session_id: Session ID
            first_frame_path: Path to first frame image
            obj_id: Object ID

        Returns:
            Dict with selector and session info for notebook handling
        """
        selector = ColabObjectSelector(first_frame_path)
        selector.display()

        print("\n[SAM3 Tracker] Use sliders to position cursor on object center")
        print("              Click 'Confirm Selection' when ready")

        # Return selector for async handling in notebook
        return {
            "selector": selector,
            "session_id": session_id,
            "obj_id": obj_id,
            "first_frame_path": first_frame_path,
        }

    def propagate_tracking(
        self,
        session_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Generator[Dict, None, None]:
        """
        Propagate tracking through all frames.

        Args:
            session_id: Session ID
            progress_callback: Optional callback(current, total)

        Yields:
            Dict with frame_index and outputs (masks)
        """
        for response in self.predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            yield response

    def close_session(self, session_id: str) -> None:
        """Close SAM3 session and free resources."""
        if self.predictor:
            self.predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            torch.cuda.empty_cache()

    def shutdown(self) -> None:
        """Shutdown SAM3 predictor completely."""
        if self.predictor:
            self.predictor.shutdown()
            self.predictor = None
            torch.cuda.empty_cache()

    def annotate_sequence(
        self,
        input_dir: str,
        class_id: int,
        output_dir: str,
        click_point: Optional[Tuple[float, float]] = None,
        save_masks: bool = False,
        save_cutouts: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AnnotationResult:
        """
        Annotate entire image sequence using tracking.

        Args:
            input_dir: Directory containing images
            class_id: YOLO class ID
            output_dir: Output directory
            click_point: Optional (x, y) normalized coordinates for first frame.
                        If None, interactive selection is used.
            save_masks: Save mask images
            save_cutouts: Save cutout images
            progress_callback: Progress callback(current, total)

        Returns:
            AnnotationResult
        """
        self._load_model()

        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Create output directories
        labels_dir = output_path / "labels"
        images_dir = output_path / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        if save_masks:
            masks_dir = output_path / "masks"
            masks_dir.mkdir(exist_ok=True)

        if save_cutouts:
            cutouts_dir = output_path / "cutouts"
            cutouts_dir.mkdir(exist_ok=True)

        # Find images (sorted by name for consistent ordering)
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        images = sorted(
            [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions],
            key=lambda x: x.name,
        )

        if not images:
            raise ValueError(f"No images found in {input_dir}")

        self._frame_paths = images
        result = AnnotationResult(total_images=len(images))

        # Store original sizes
        for i, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is not None:
                self._original_sizes[i] = (img.shape[0], img.shape[1])

        session_id = None
        try:
            # Start session with image directory
            session_id = self.start_session(str(input_path))
            print(f"[SAM3 Tracker] Session started: {session_id}")

            # Select object in first frame
            if click_point:
                x, y = click_point
                print(f"[SAM3 Tracker] Using provided click point: ({x:.3f}, {y:.3f})")
                self.select_object_by_click(session_id, 0, x, y, obj_id=1)
            else:
                self.select_object_interactive(session_id, str(images[0]), obj_id=1)

            # Track through all frames
            print(f"[SAM3 Tracker] Tracking {len(images)} frames...")

            frame_results = {}
            for response in tqdm(
                self.propagate_tracking(session_id),
                total=len(images),
                desc="Tracking",
            ):
                frame_idx = response["frame_index"]
                outputs = response.get("outputs", {})
                frame_results[frame_idx] = outputs

            # Process results
            for frame_idx, outputs in frame_results.items():
                if frame_idx >= len(images):
                    continue

                img_file = images[frame_idx]
                image = cv2.imread(str(img_file))
                if image is None:
                    result.failed += 1
                    result.failed_paths.append(str(img_file))
                    continue

                img_h, img_w = image.shape[:2]

                # Get mask for obj_id=1
                mask = outputs.get(1)  # obj_id=1
                if mask is None:
                    result.failed += 1
                    result.failed_paths.append(str(img_file))
                    continue

                # Convert to numpy if tensor
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()

                # Ensure correct shape
                if mask.ndim == 3:
                    mask = mask[0]  # Remove batch dimension

                # Get bounding box
                bbox_pixels = self._mask_to_bbox(mask, (img_h, img_w))
                if bbox_pixels is None:
                    result.failed += 1
                    result.failed_paths.append(str(img_file))
                    continue

                x_min, y_min, x_max, y_max = bbox_pixels

                # Convert to YOLO format
                yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

                # Save label
                label_path = labels_dir / f"{img_file.stem}.txt"
                write_yolo_label(str(label_path), class_id, yolo_bbox)

                # Copy image
                shutil.copy2(img_file, images_dir / img_file.name)

                # Save mask
                if save_masks:
                    # Resize mask to original image size if needed
                    if mask.shape[:2] != (img_h, img_w):
                        mask_resized = cv2.resize(
                            mask.astype(np.uint8),
                            (img_w, img_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        mask_resized = mask

                    mask_path = masks_dir / f"{img_file.stem}_mask.png"
                    cv2.imwrite(
                        str(mask_path),
                        (mask_resized > 0).astype(np.uint8) * 255,
                    )

                # Save cutout
                if save_cutouts:
                    cutout = self._extract_cutout(image, mask, bbox_pixels)
                    cutout_path = cutouts_dir / f"{img_file.stem}_cutout.png"
                    cv2.imwrite(str(cutout_path), cutout)

                result.successful += 1

                if progress_callback:
                    progress_callback(frame_idx + 1, len(images))

        finally:
            if session_id:
                self.close_session(session_id)

        return result

    def visualize_result(
        self,
        image_path: str,
        mask: np.ndarray,
        bbox: Tuple[float, float, float, float],
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> np.ndarray:
        """
        Visualize annotation result.

        Args:
            image_path: Path to image
            mask: Segmentation mask
            bbox: YOLO format bounding box
            output_path: Optional save path
            show: Show in window

        Returns:
            Annotated image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_h, img_w = image.shape[:2]

        # Resize mask if needed
        if mask.shape[:2] != (img_h, img_w):
            mask = cv2.resize(
                mask.astype(np.uint8),
                (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )

        # Draw mask overlay
        mask_overlay = np.zeros_like(image)
        mask_overlay[mask > 0] = [0, 255, 0]
        image = cv2.addWeighted(image, 1, mask_overlay, 0.3, 0)

        # Draw bounding box
        from annotation_utils import yolo_to_bbox

        x_min, y_min, x_max, y_max = yolo_to_bbox(*bbox, img_w, img_h)
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Add label
        label = f"YOLO: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})"
        cv2.putText(
            image,
            label,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        if output_path:
            cv2.imwrite(output_path, image)

        if show:
            if is_colab_environment() or is_notebook_environment():
                # Use matplotlib for notebook/Colab display
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title("SAM3 Tracking Result")
                plt.show()
            else:
                # Use OpenCV for local display
                cv2.imshow("SAM3 Tracking Result", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return image


def main():
    """Command-line interface for SAM3 tracker annotation."""
    parser = argparse.ArgumentParser(
        description="Tracking-based annotation using SAM3 Video API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (click to select object)
  python sam3_tracker.py \\
      --input-dir datasets/raw_captures/cup \\
      --output-dir datasets/annotated/cup \\
      --class-id 0

  # Coordinate mode (for automation)
  python sam3_tracker.py \\
      --input-dir datasets/raw_captures/cup \\
      --output-dir datasets/annotated/cup \\
      --class-id 0 \\
      --click 0.5 0.6

  # Save masks and cutouts
  python sam3_tracker.py \\
      --input-dir datasets/raw_captures/cup \\
      --output-dir datasets/annotated/cup \\
      --class-id 0 \\
      --save-masks \\
      --save-cutouts
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Input image directory",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory (labels/ and images/ will be created)",
    )
    parser.add_argument(
        "--class-id",
        "-c",
        type=int,
        required=True,
        help="YOLO class ID",
    )
    parser.add_argument(
        "--click",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="First frame click position (normalized 0-1)",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU ID to use (default: 0)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1024,
        help="Max image long side size (default: 1024, for VRAM saving)",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save mask images",
    )
    parser.add_argument(
        "--save-cutouts",
        action="store_true",
        help="Save cutout images (RGBA with transparent background)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum mask area in pixels (default: 500)",
    )

    args = parser.parse_args()

    config = SAM3TrackerConfig(
        gpu_id=args.gpu_id,
        max_long_side=args.max_size,
        min_mask_area=args.min_area,
    )

    tracker = SAM3Tracker(config)

    try:
        click_point = tuple(args.click) if args.click else None

        result = tracker.annotate_sequence(
            input_dir=args.input_dir,
            class_id=args.class_id,
            output_dir=args.output_dir,
            click_point=click_point,
            save_masks=args.save_masks,
            save_cutouts=args.save_cutouts,
        )

        print("\n" + result.summary())

        if result.failed > 0:
            print("\nFailed images (first 10):")
            for path in result.failed_paths[:10]:
                print(f"  - {path}")

    finally:
        tracker.shutdown()


if __name__ == "__main__":
    main()
