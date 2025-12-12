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
DEFAULT_TRAIN_RATIO: float = 0.80  # 80%/20% split for better validation
DEFAULT_GROUP_INTERVAL_SEC: float = 2.0  # Group frames within 2 seconds


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


# =============================================================================
# GPU Scaling Constants
# =============================================================================
# VRAM thresholds for GPU classification (GB)
GPU_VRAM_THRESHOLDS: dict = {
    "low": (0, 6),
    "medium": (6, 12),
    "high": (12, 24),
    "workstation": (24, float("inf")),
}

# Model VRAM overhead (empirical values in GB)
MODEL_VRAM_OVERHEAD: dict = {
    "yolov8n.pt": 1.2,
    "yolov8s.pt": 1.8,
    "yolov8m.pt": 2.5,
    "yolov8l.pt": 3.5,
    "yolov8x.pt": 5.0,
}

# Per-sample memory at 640x640 (empirical, in MB)
MODEL_PER_SAMPLE_MEMORY_MB: dict = {
    "yolov8n.pt": 180,
    "yolov8s.pt": 250,
    "yolov8m.pt": 380,
    "yolov8l.pt": 550,
    "yolov8x.pt": 750,
}

# Standard image sizes for training
STANDARD_IMAGE_SIZES: List[int] = [320, 480, 640, 800, 1024, 1280]

# GPU scaling safety margins
DEFAULT_VRAM_SAFETY_MARGIN: float = 0.15  # 15% safety margin
DEFAULT_MAX_VRAM_UTILIZATION: float = 0.85  # Max 85% utilization

# OOM recovery settings
OOM_BATCH_REDUCTION_FACTOR: float = 0.5  # Reduce to 50% on OOM
MAX_OOM_RETRIES: int = 3


# =============================================================================
# TensorBoard Settings
# =============================================================================
TENSORBOARD_DEFAULT_PORT: int = 6006
TENSORBOARD_FLUSH_SECS: int = 30
TENSORBOARD_LOG_FREQUENCY: int = 1  # Log every epoch


# =============================================================================
# Copy-Paste Augmentation Defaults
# =============================================================================
# Object extraction settings
DEFAULT_ALPHA_BLUR_SIGMA: float = 2.0  # Gaussian blur sigma for edge softening
DEFAULT_EXTRACTION_PADDING: int = 5  # Padding around object when cropping
DEFAULT_MIN_OBJECT_SIZE: int = 32  # Minimum object size in pixels

# Synthesis ratio settings
DEFAULT_SYNTHETIC_RATIO_MIN: float = 1.0  # Minimum synthetic:real ratio (1:1)
DEFAULT_SYNTHETIC_RATIO_MAX: float = 3.0  # Maximum synthetic:real ratio (3:1)
DEFAULT_SYNTHETIC_RATIO: float = 2.0  # Default synthetic:real ratio (2:1)

# Object transformation settings
DEFAULT_SCALE_RANGE: Tuple[float, float] = (0.5, 1.5)  # Scale factor range
DEFAULT_ROTATION_RANGE: Tuple[float, float] = (-15.0, 15.0)  # Rotation in degrees

# Blending settings
DEFAULT_WHITE_BALANCE_STRENGTH: float = 0.7  # LAB color adjustment strength
DEFAULT_MAX_OBJECTS_PER_IMAGE: int = 3  # Maximum objects per synthetic image
DEFAULT_MIN_OBJECTS_PER_IMAGE: int = 1  # Minimum objects per synthetic image
DEFAULT_OVERLAP_IOU_THRESHOLD: float = 0.1  # Maximum overlap allowed
