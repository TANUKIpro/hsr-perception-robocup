#!/usr/bin/env python3
"""
SAM2 Video Tracking Predictor

A wrapper class for SAM2 Video Predictor that enables tracking objects
across sequential image frames with VRAM management and batch processing.

Features:
- Automatic image sequence preparation (renaming to sequential format)
- VRAM usage estimation and batch splitting
- Object tracking propagation across frames
- Confidence scoring for tracking quality assessment
"""

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


@dataclass
class TrackingResult:
    """Result of tracking for a single frame."""

    mask: np.ndarray  # Boolean mask (H, W)
    confidence: float  # Confidence score [0, 1]
    is_low_confidence: bool  # True if confidence below threshold
    mask_area: int  # Number of pixels in mask

    @classmethod
    def from_mask(
        cls, mask: np.ndarray, reference_area: Optional[int] = None
    ) -> "TrackingResult":
        """
        Create TrackingResult from mask with automatic confidence calculation.

        Args:
            mask: Boolean mask array
            reference_area: Reference mask area for confidence calculation
        """
        mask_area = int(np.sum(mask))

        # Calculate confidence based on mask area stability
        if reference_area is not None and reference_area > 0:
            area_ratio = mask_area / reference_area
            # Confidence drops if area changes significantly
            if 0.5 <= area_ratio <= 2.0:
                confidence = 1.0 - abs(1.0 - area_ratio) * 0.5
            else:
                confidence = max(0.0, 0.5 - abs(1.0 - area_ratio) * 0.1)
        else:
            # No reference, use mask area as basic confidence
            confidence = min(1.0, mask_area / 1000.0) if mask_area > 0 else 0.0

        is_low_confidence = confidence < 0.5 or mask_area < 100

        return cls(
            mask=mask,
            confidence=confidence,
            is_low_confidence=is_low_confidence,
            mask_area=mask_area,
        )


@dataclass
class VRAMEstimate:
    """VRAM usage estimation result."""

    estimated_usage_gb: float
    available_gb: float
    total_gb: float
    needs_split: bool
    recommended_batch_size: int
    num_batches: int


class VideoTrackingPredictor:
    """
    SAM2 Video Predictor wrapper for sequential image tracking.

    Provides VRAM management, batch processing, and tracking propagation
    for annotation of sequential images.
    """

    # Constants for VRAM estimation
    BYTES_PER_FRAME_BASE = 50 * 1024 * 1024  # 50MB base per frame
    BYTES_PER_PIXEL = 12  # Approximate bytes per pixel for processing
    VRAM_SAFETY_THRESHOLD = 0.95  # 95% VRAM usage threshold

    def __init__(
        self,
        model_path: str = "sam2_b.pt",
        device: str = "cuda",
        low_confidence_threshold: float = 0.5,
    ):
        """
        Initialize VideoTrackingPredictor.

        Args:
            model_path: Path to SAM2 model checkpoint
            device: Device to run model on ("cuda" or "cpu")
            low_confidence_threshold: Threshold for marking low confidence results
        """
        self.model_path = model_path
        self.device = device
        self.low_confidence_threshold = low_confidence_threshold

        self.predictor = None
        self.inference_state: Optional[dict] = None
        self.temp_dir: Optional[str] = None
        self.frame_map: Dict[int, Path] = {}
        self.tracking_results: Dict[int, TrackingResult] = {}
        self.reference_mask_area: Optional[int] = None

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None):
        """
        Load SAM2 Video Predictor model.

        Args:
            progress_callback: Optional callback for progress updates
        """
        if self.predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        if progress_callback:
            progress_callback("Loading SAM2 Video Predictor...")

        # Determine model config based on model path
        model_cfg = self._get_model_config()

        # Handle CPU fallback
        actual_device = self.device
        if self.device == "cuda" and not torch.cuda.is_available():
            actual_device = "cpu"
            if progress_callback:
                progress_callback("CUDA not available, using CPU...")

        self.predictor = build_sam2_video_predictor(
            model_cfg, self.model_path, device=actual_device
        )

        if progress_callback:
            progress_callback("SAM2 Video Predictor loaded successfully")

    def _get_model_config(self) -> str:
        """Determine model config file based on model path."""
        model_path_lower = self.model_path.lower()

        if "sam2.1" in model_path_lower or "sam2_1" in model_path_lower:
            if "base" in model_path_lower or "_b" in model_path_lower:
                return "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "large" in model_path_lower or "_l" in model_path_lower:
                return "configs/sam2.1/sam2.1_hiera_l.yaml"
            elif "small" in model_path_lower or "_s" in model_path_lower:
                return "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "tiny" in model_path_lower or "_t" in model_path_lower:
                return "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                return "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            if "sam2_b" in model_path_lower or "base" in model_path_lower:
                return "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "sam2_l" in model_path_lower or "large" in model_path_lower:
                return "configs/sam2.1/sam2.1_hiera_l.yaml"
            elif "sam2_t" in model_path_lower or "tiny" in model_path_lower:
                return "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                return "configs/sam2.1/sam2.1_hiera_b+.yaml"

    @staticmethod
    def get_available_vram() -> Tuple[float, float]:
        """
        Get available and total VRAM in GB.

        Returns:
            Tuple of (available_gb, total_gb)
        """
        if not torch.cuda.is_available():
            return (0.0, 0.0)

        torch.cuda.synchronize()
        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)

        # Available = Total - Reserved (more conservative estimate)
        available = total - reserved
        return (available / (1024**3), total / (1024**3))

    def estimate_vram_usage(
        self, num_frames: int, image_size: Tuple[int, int]
    ) -> VRAMEstimate:
        """
        Estimate VRAM usage for processing given number of frames.

        Args:
            num_frames: Number of frames to process
            image_size: Image dimensions (height, width)

        Returns:
            VRAMEstimate with usage details and recommendations
        """
        height, width = image_size
        pixels_per_frame = height * width

        # Estimate VRAM per frame
        per_frame_bytes = (
            self.BYTES_PER_FRAME_BASE + pixels_per_frame * self.BYTES_PER_PIXEL
        )
        total_estimated_bytes = per_frame_bytes * num_frames

        # Add model memory overhead (approximately 2GB for base model)
        model_overhead_bytes = 2 * 1024**3
        total_with_overhead = total_estimated_bytes + model_overhead_bytes

        estimated_gb = total_with_overhead / (1024**3)
        available_gb, total_gb = self.get_available_vram()

        # Check if split is needed
        threshold_gb = total_gb * self.VRAM_SAFETY_THRESHOLD
        needs_split = estimated_gb > threshold_gb

        # Calculate recommended batch size
        if needs_split and available_gb > 0:
            safe_frames = int(
                (threshold_gb - model_overhead_bytes / (1024**3))
                / (per_frame_bytes / (1024**3))
            )
            recommended_batch_size = max(10, min(safe_frames, num_frames))
            num_batches = (num_frames + recommended_batch_size - 1) // recommended_batch_size
        else:
            recommended_batch_size = num_frames
            num_batches = 1

        return VRAMEstimate(
            estimated_usage_gb=estimated_gb,
            available_gb=available_gb,
            total_gb=total_gb,
            needs_split=needs_split,
            recommended_batch_size=recommended_batch_size,
            num_batches=num_batches,
        )

    def init_sequence(
        self,
        image_dir: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[int, List[Path]]:
        """
        Initialize image sequence for tracking.

        Creates a temporary directory with symlinks to original images
        renamed to sequential format required by SAM2.

        Args:
            image_dir: Directory containing images
            progress_callback: Optional callback for progress updates

        Returns:
            Tuple of (num_frames, list of original paths in order)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Clean up previous temp directory
        self._cleanup_temp_dir()

        if progress_callback:
            progress_callback("Preparing image sequence...")

        # Find and sort images using natural sort (same as sam2_interactive_app.py)
        def natural_sort_key(path):
            """Sort key for natural/alphanumeric ordering."""
            return [
                int(c) if c.isdigit() else c.lower()
                for c in re.split(r"(\d+)", path.name)
            ]

        image_dir_path = Path(image_dir)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = sorted(
            [
                p
                for p in image_dir_path.iterdir()
                if p.suffix.lower() in image_extensions
            ],
            key=natural_sort_key,
        )

        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")

        num_frames = len(image_paths)

        # Create temp directory with sequential symlinks
        self.temp_dir = tempfile.mkdtemp(prefix="sam2_tracking_")
        self.frame_map = {}

        for idx, original_path in enumerate(image_paths):
            # SAM2 expects files named as integers: 0.jpg, 1.jpg, etc.
            link_name = f"{idx}.jpg"
            link_path = Path(self.temp_dir) / link_name

            # Create symlink to original file
            os.symlink(original_path.absolute(), link_path)
            self.frame_map[idx] = original_path

        if progress_callback:
            progress_callback(f"Initialized {num_frames} frames")

        # Initialize SAM2 inference state
        self.inference_state = self.predictor.init_state(
            video_path=self.temp_dir,
            offload_video_to_cpu=True,  # Save GPU memory
        )

        # Clear previous results
        self.tracking_results = {}
        self.reference_mask_area = None

        return num_frames, image_paths

    def set_initial_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        foreground_points: List[Tuple[int, int]],
        background_points: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Set initial prompt on a frame for tracking.

        Args:
            frame_idx: Frame index to add prompt
            obj_id: Object ID for tracking
            foreground_points: List of (x, y) foreground points
            background_points: List of (x, y) background points

        Returns:
            Initial mask for the frame
        """
        if self.inference_state is None:
            raise RuntimeError("Sequence not initialized. Call init_sequence() first.")

        # Prepare points and labels
        all_points = list(foreground_points) + list(background_points)
        all_labels = [1] * len(foreground_points) + [0] * len(background_points)

        if not all_points:
            raise ValueError("At least one point is required")

        points = np.array(all_points, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        # Add points to the frame
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )

        # Convert mask logits to boolean mask
        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()

        # Store reference mask area for confidence calculation
        self.reference_mask_area = int(np.sum(mask))

        # Store result
        self.tracking_results[frame_idx] = TrackingResult.from_mask(
            mask, reference_area=None  # Initial frame has high confidence
        )
        self.tracking_results[frame_idx].confidence = 1.0
        self.tracking_results[frame_idx].is_low_confidence = False

        return mask

    def propagate_tracking(
        self,
        progress_callback: Optional[Callable[[int, int, int, np.ndarray], None]] = None,
    ) -> Dict[int, TrackingResult]:
        """
        Propagate tracking to all frames.

        Args:
            progress_callback: Callback with (current_frame, total_frames, frame_idx, mask)
                - current_frame: Number of frames processed so far
                - total_frames: Total number of frames
                - frame_idx: Current frame index being processed
                - mask: Boolean mask array for the current frame

        Returns:
            Dictionary mapping frame index to TrackingResult
        """
        if self.inference_state is None:
            raise RuntimeError("Sequence not initialized. Call init_sequence() first.")

        num_frames = self.inference_state["num_frames"]
        processed = 0

        for frame_idx, obj_ids, video_res_masks in self.predictor.propagate_in_video(
            self.inference_state
        ):
            # Convert mask to boolean numpy array
            mask = (video_res_masks[0, 0] > 0.0).cpu().numpy()

            # Calculate tracking result with confidence
            result = TrackingResult.from_mask(
                mask, reference_area=self.reference_mask_area
            )
            self.tracking_results[frame_idx] = result

            processed += 1
            if progress_callback:
                progress_callback(processed, num_frames, frame_idx, mask)

        return self.tracking_results

    def add_correction_points(
        self,
        frame_idx: int,
        obj_id: int,
        foreground_points: List[Tuple[int, int]],
        background_points: List[Tuple[int, int]],
    ) -> np.ndarray:
        """
        Add correction points to a frame and update tracking.

        Args:
            frame_idx: Frame index to correct
            obj_id: Object ID
            foreground_points: List of (x, y) foreground points
            background_points: List of (x, y) background points

        Returns:
            Updated mask for the frame
        """
        if self.inference_state is None:
            raise RuntimeError("Sequence not initialized. Call init_sequence() first.")

        # Prepare points and labels
        all_points = list(foreground_points) + list(background_points)
        all_labels = [1] * len(foreground_points) + [0] * len(background_points)

        if not all_points:
            raise ValueError("At least one point is required")

        points = np.array(all_points, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.int32)

        # Add correction points
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
            clear_old_points=False,  # Keep existing points
        )

        # Convert mask logits to boolean mask
        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()

        # Update result
        self.tracking_results[frame_idx] = TrackingResult.from_mask(
            mask, reference_area=self.reference_mask_area
        )

        return mask

    def get_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get mask for specific frame."""
        result = self.tracking_results.get(frame_idx)
        return result.mask if result else None

    def get_result(self, frame_idx: int) -> Optional[TrackingResult]:
        """Get TrackingResult for specific frame."""
        return self.tracking_results.get(frame_idx)

    def get_frame_path(self, frame_idx: int) -> Optional[Path]:
        """Get original image path for specific frame."""
        return self.frame_map.get(frame_idx)

    def get_low_confidence_frames(self) -> List[int]:
        """Get list of frame indices with low confidence tracking."""
        return [
            idx
            for idx, result in self.tracking_results.items()
            if result.is_low_confidence
        ]

    def get_frame_image(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Load and return the image for a specific frame.

        Args:
            frame_idx: Frame index

        Returns:
            RGB image array or None if not found
        """
        path = self.get_frame_path(frame_idx)
        if path is None or not path.exists():
            return None

        image = cv2.imread(str(path))
        if image is None:
            return None

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def reset(self):
        """Reset tracking state."""
        if self.inference_state is not None and self.predictor is not None:
            self.predictor.reset_state(self.inference_state)

        self.inference_state = None
        self.tracking_results = {}
        self.reference_mask_area = None
        self._cleanup_temp_dir()

    def _cleanup_temp_dir(self):
        """Clean up temporary directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
            self.frame_map = {}

    def __del__(self):
        """Cleanup on deletion."""
        self._cleanup_temp_dir()


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert boolean mask to bounding box.

    Args:
        mask: Boolean mask array (H, W)

    Returns:
        Bounding box as (x_min, y_min, x_max, y_max) or None if empty mask
    """
    if not np.any(mask):
        return None

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (int(x_min), int(y_min), int(x_max), int(y_max))
