"""
SAM2 Interactive Image Predictor.

Wrapper for SAM2 Image Predictor with point-based segmentation.
Provides interactive segmentation using foreground/background points.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np


class SAM2InteractivePredictor:
    """
    Wrapper for SAM2 Image Predictor with point-based segmentation.

    Provides interactive segmentation using foreground/background points.
    Supports iterative refinement using previous mask as input.
    """

    def __init__(self, model_path: str = "sam2_b.pt", device: str = "cuda"):
        """
        Initialize SAM2 predictor.

        Args:
            model_path: Path to SAM2 model checkpoint
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.predictor = None
        self.model = None
        self.current_image: Optional[np.ndarray] = None
        self.low_res_mask: Optional[np.ndarray] = None

    def load_model(self, progress_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Load SAM2 model.

        Args:
            progress_callback: Optional callback for progress updates
        """
        if self.predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        if progress_callback:
            progress_callback("Loading SAM2 model...")

        # Determine model config based on model path
        model_path_lower = self.model_path.lower()
        if "sam2.1" in model_path_lower or "sam2_1" in model_path_lower:
            if "base" in model_path_lower or "_b" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "large" in model_path_lower or "_l" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            elif "small" in model_path_lower or "_s" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "tiny" in model_path_lower or "_t" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            if "sam2_b" in model_path_lower or "base" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "sam2_l" in model_path_lower or "large" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            elif "sam2_t" in model_path_lower or "tiny" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        self.model = build_sam2(model_cfg, self.model_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

        if progress_callback:
            progress_callback("SAM2 model loaded successfully")

    def set_image(self, image_rgb: np.ndarray) -> None:
        """
        Set image for segmentation.

        Args:
            image_rgb: RGB image array (H, W, 3)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.predictor.set_image(image_rgb)
        self.current_image = image_rgb
        self.low_res_mask = None

    def predict(
        self,
        foreground_points: List[Tuple[int, int]],
        background_points: List[Tuple[int, int]],
        use_previous_mask: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Predict mask from points.

        Args:
            foreground_points: List of (x, y) foreground points
            background_points: List of (x, y) background points
            use_previous_mask: Whether to use previous mask for refinement

        Returns:
            Tuple of (best_mask, iou_score, low_res_mask)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not foreground_points and not background_points:
            raise ValueError("At least one point is required")

        # Prepare points and labels
        all_points = list(foreground_points) + list(background_points)
        all_labels = [1] * len(foreground_points) + [0] * len(background_points)

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)

        # Use previous mask for refinement if available
        mask_input = self.low_res_mask if use_previous_mask else None

        masks, iou_predictions, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=True,
        )

        # Select best mask (highest IoU)
        best_idx = int(np.argmax(iou_predictions))
        best_mask = masks[best_idx].astype(bool)
        best_iou = float(iou_predictions[best_idx])

        # Store low_res_mask for future refinement
        self.low_res_mask = low_res_masks[best_idx : best_idx + 1]

        return best_mask, best_iou, self.low_res_mask

    def reset_mask_state(self) -> None:
        """Reset the stored low_res_mask for new annotation."""
        self.low_res_mask = None
