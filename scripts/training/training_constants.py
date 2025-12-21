"""
Training module constants.

Centralizes magic numbers and configuration values used across training scripts.
"""
from enum import IntEnum
from typing import Dict, Tuple

# =============================================================================
# VRAM Thresholds (GB)
# =============================================================================


class VRAMThreshold(IntEnum):
    """VRAM thresholds in GB for GPU tier classification."""

    LOW_MAX = 6
    MEDIUM_MAX = 12
    HIGH_MAX = 24


# =============================================================================
# Model VRAM Overhead (GB) - Empirically measured
# =============================================================================
MODEL_VRAM_OVERHEAD_GB: Dict[str, float] = {
    "yolov8n.pt": 1.2,
    "yolov8s.pt": 1.8,
    "yolov8m.pt": 2.5,
    "yolov8l.pt": 3.5,
    "yolov8x.pt": 5.0,
}

# =============================================================================
# Per-sample Memory at 640x640 (MB) - Empirically measured
# =============================================================================
MODEL_PER_SAMPLE_MEMORY_MB: Dict[str, int] = {
    "yolov8n.pt": 180,
    "yolov8s.pt": 250,
    "yolov8m.pt": 380,
    "yolov8l.pt": 550,
    "yolov8x.pt": 750,
}

# =============================================================================
# OOM Recovery Settings
# =============================================================================
MAX_OOM_RETRIES: int = 3
OOM_BATCH_REDUCTION_FACTOR: float = 0.5  # Reduce to 50%
MIN_BATCH_SIZE: int = 4

# =============================================================================
# GPU Scaling Settings
# =============================================================================
DEFAULT_VRAM_SAFETY_MARGIN: float = 0.15  # 15%
DEFAULT_MAX_VRAM_UTILIZATION: float = 0.85  # 85%
CPU_FALLBACK_BATCH_SIZE: int = 8
MAX_BATCH_SIZE: int = 128

# =============================================================================
# Training Time Estimates (seconds per iteration)
# =============================================================================
SECONDS_PER_ITERATION: Dict[str, float] = {
    "low": 0.5,
    "medium": 0.25,
    "high": 0.15,
    "workstation": 0.08,
    "cpu": 5.0,
}
TRAINING_OVERHEAD_FACTOR: float = 1.1  # 10% overhead for validation, checkpointing

# =============================================================================
# Standard Image Sizes
# =============================================================================
STANDARD_IMAGE_SIZES: Tuple[int, ...] = (320, 480, 640, 800, 1024, 1280)

# =============================================================================
# Garbage Collection Settings
# =============================================================================
DEFAULT_GC_PASSES: int = 2
