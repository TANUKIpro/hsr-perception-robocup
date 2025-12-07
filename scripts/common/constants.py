"""
HSR Perception - Common Constants

Shared constants used across the annotation, training, and evaluation modules.
Centralizes values that were previously duplicated in multiple files.
"""

from typing import List, Tuple

# =============================================================================
# Image File Extensions
# =============================================================================
# Supported image file extensions for processing
IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp"]

# With uppercase variants for case-insensitive matching
IMAGE_EXTENSIONS_ALL: List[str] = IMAGE_EXTENSIONS + [e.upper() for e in IMAGE_EXTENSIONS]


# =============================================================================
# Competition Defaults
# =============================================================================
# Data collection
DEFAULT_TARGET_SAMPLES: int = 100
DEFAULT_BURST_INTERVAL: float = 0.2  # seconds (5 FPS)
MIN_SAMPLES_FOR_TRAINING: int = 50

# Dataset splitting
DEFAULT_TRAIN_RATIO: float = 0.85


# =============================================================================
# Model/Inference Defaults
# =============================================================================
# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25
DEFAULT_IOU_THRESHOLD: float = 0.5

# Competition performance targets
TARGET_MAP50: float = 0.85  # 85% mAP at IoU 0.5
TARGET_INFERENCE_MS: float = 100.0  # Maximum 100ms inference time


# =============================================================================
# Annotation Defaults
# =============================================================================
# Bounding box margin ratio for annotation
DEFAULT_BBOX_MARGIN_RATIO: float = 0.02

# Contour area thresholds
DEFAULT_MIN_CONTOUR_AREA: int = 500
DEFAULT_MAX_CONTOUR_AREA_RATIO: float = 0.9


# =============================================================================
# Visualization Defaults
# =============================================================================
# Default colors (BGR format for OpenCV)
DEFAULT_BBOX_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green
DEFAULT_MASK_COLOR: Tuple[int, int, int] = (0, 255, 0)  # Green
DEFAULT_MASK_ALPHA: float = 0.3
DEFAULT_BBOX_THICKNESS: int = 2
DEFAULT_FONT_SCALE: float = 0.6
DEFAULT_FONT_THICKNESS: int = 2


# =============================================================================
# SAM2 Model Defaults
# =============================================================================
# SAM2 model configurations
SAM2_DEFAULT_POINTS_PER_SIDE: int = 32
SAM2_DEFAULT_PRED_IOU_THRESH: float = 0.88
SAM2_DEFAULT_STABILITY_SCORE_THRESH: float = 0.92
SAM2_DEFAULT_MIN_MASK_REGION_AREA: int = 100


# =============================================================================
# Training Defaults
# =============================================================================
# YOLO training hyperparameters
DEFAULT_YOLO_MODEL: str = "yolov8m.pt"
DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 16
DEFAULT_IMAGE_SIZE: int = 640
DEFAULT_PATIENCE: int = 10
