"""
HSR Perception - Configuration Utilities

Unified configuration management with dataclasses.
Consolidates configuration patterns from annotation, training, and evaluation modules.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .constants import (
    DEFAULT_BBOX_MARGIN_RATIO,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MIN_CONTOUR_AREA,
    DEFAULT_MAX_CONTOUR_AREA_RATIO,
    DEFAULT_PATIENCE,
    DEFAULT_YOLO_MODEL,
    SAM2_DEFAULT_MIN_MASK_REGION_AREA,
    SAM2_DEFAULT_POINTS_PER_SIDE,
    SAM2_DEFAULT_PRED_IOU_THRESH,
    SAM2_DEFAULT_STABILITY_SCORE_THRESH,
)


# =============================================================================
# Annotator Configurations
# =============================================================================


@dataclass
class AnnotatorConfig:
    """Base configuration for all annotators."""

    bbox_margin_ratio: float = DEFAULT_BBOX_MARGIN_RATIO
    min_contour_area: int = DEFAULT_MIN_CONTOUR_AREA
    max_contour_area_ratio: float = DEFAULT_MAX_CONTOUR_AREA_RATIO


@dataclass
class BackgroundSubtractionConfig(AnnotatorConfig):
    """
    Configuration for background subtraction annotator.

    Attributes:
        blur_kernel_size: Gaussian blur kernel size (must be odd)
        threshold_method: Method for thresholding ("otsu", "adaptive", "fixed")
        fixed_threshold: Threshold value when using "fixed" method
        morph_kernel_size: Morphological operations kernel size
        erosion_iterations: Number of erosion iterations
        dilation_iterations: Number of dilation iterations
    """

    blur_kernel_size: int = 5
    threshold_method: str = "otsu"  # "otsu", "adaptive", "fixed"
    fixed_threshold: int = 30
    morph_kernel_size: int = 5
    erosion_iterations: int = 2
    dilation_iterations: int = 3

    def __post_init__(self):
        """Validate configuration."""
        if self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        if self.threshold_method not in ("otsu", "adaptive", "fixed"):
            raise ValueError(f"Invalid threshold_method: {self.threshold_method}")


@dataclass
class SAM2Config(AnnotatorConfig):
    """
    Configuration for SAM2 annotator.

    Attributes:
        device: Device to use ("cuda", "cpu", or specific GPU)
        points_per_side: Points per side for automatic mask generation
        pred_iou_thresh: Predicted IoU threshold for mask filtering
        stability_score_thresh: Stability score threshold
        min_mask_region_area: Minimum mask region area in pixels
    """

    device: str = "cuda"
    points_per_side: int = SAM2_DEFAULT_POINTS_PER_SIDE
    pred_iou_thresh: float = SAM2_DEFAULT_PRED_IOU_THRESH
    stability_score_thresh: float = SAM2_DEFAULT_STABILITY_SCORE_THRESH
    min_mask_region_area: int = SAM2_DEFAULT_MIN_MASK_REGION_AREA


def get_sam2_model_config(model_path: str) -> str:
    """
    Determine SAM2 model config path from model checkpoint path.

    Analyzes the model path to determine which SAM2 configuration
    file should be used for loading the model.

    Args:
        model_path: Path to SAM2 model checkpoint file

    Returns:
        Config path string (e.g., "configs/sam2.1/sam2.1_hiera_b+.yaml")
    """
    model_path_lower = model_path.lower()

    if "sam2.1" in model_path_lower or "sam2_1" in model_path_lower:
        # SAM2.1 models
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
        # SAM2 models - use SAM2.1 config for compatibility with newer checkpoints
        if "sam2_b" in model_path_lower or "base" in model_path_lower:
            return "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "sam2_l" in model_path_lower or "large" in model_path_lower:
            return "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "sam2_t" in model_path_lower or "tiny" in model_path_lower:
            return "configs/sam2.1/sam2.1_hiera_t.yaml"
        else:
            return "configs/sam2.1/sam2.1_hiera_b+.yaml"  # Default to base


# =============================================================================
# Training Configurations
# =============================================================================


@dataclass
class TrainingConfig:
    """
    Configuration for YOLO training.

    Provides two preset configurations:
    - competition(): Standard competition-day settings
    - fast(): Faster training for testing or limited resources
    """

    # Model settings
    model: str = DEFAULT_YOLO_MODEL
    imgsz: int = DEFAULT_IMAGE_SIZE

    # Training settings
    epochs: int = DEFAULT_EPOCHS
    batch: int = DEFAULT_BATCH_SIZE
    patience: int = DEFAULT_PATIENCE

    # Optimizer settings
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Augmentation settings
    augment: bool = True
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.1

    # Performance settings
    workers: int = 8
    cache: bool = True
    amp: bool = True
    close_mosaic: int = 10

    # Checkpointing
    save: bool = True
    save_period: int = 5
    exist_ok: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for YOLO training."""
        return {
            "model": self.model,
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch,
            "patience": self.patience,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "augment": self.augment,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "workers": self.workers,
            "cache": self.cache,
            "amp": self.amp,
            "close_mosaic": self.close_mosaic,
            "save": self.save,
            "save_period": self.save_period,
            "exist_ok": self.exist_ok,
        }

    @classmethod
    def competition(cls) -> "TrainingConfig":
        """Get competition-optimized configuration."""
        return cls()

    @classmethod
    def fast(cls) -> "TrainingConfig":
        """Get fast training configuration for testing."""
        return cls(
            model="yolov8s.pt",
            epochs=30,
            batch=32,
            patience=5,
            imgsz=480,
        )


# =============================================================================
# Evaluation Configurations
# =============================================================================


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    iou_threshold: float = DEFAULT_IOU_THRESHOLD
    device: str = "cuda"
    verbose: bool = True


# =============================================================================
# Class Configuration Loading
# =============================================================================


def load_class_config(config_path: str) -> Dict:
    """
    Load class configuration from JSON file.

    Args:
        config_path: Path to object_classes.json or similar

    Returns:
        Dictionary with class configuration
    """
    import json

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_class_names(config: Dict) -> List[str]:
    """
    Extract class names from configuration.

    Args:
        config: Configuration dictionary from load_class_config

    Returns:
        List of class names in order
    """
    objects = config.get("objects", [])
    return [obj.get("class_name", f"class_{i}") for i, obj in enumerate(objects)]


def get_class_id_map(config: Dict) -> Dict[str, int]:
    """
    Create mapping from class name to class ID.

    Args:
        config: Configuration dictionary from load_class_config

    Returns:
        Dictionary mapping class names to IDs
    """
    objects = config.get("objects", [])
    return {
        obj.get("class_name", f"class_{i}"): obj.get("class_id", i)
        for i, obj in enumerate(objects)
    }
