"""
HSR Perception - Training Module

Competition day fine-tuning scripts for YOLOv8 models.

Features:
- GPU hardware auto-scaling
- TensorBoard integration with competition metrics
- OOM recovery with automatic parameter adjustment
"""

from .quick_finetune import (
    CompetitionTrainer,
    COMPETITION_CONFIG,
    FAST_CONFIG,
    TrainingResult,
)
from .gpu_scaler import (
    GPUScaler,
    GPUScalingConfig,
    GPUProfile,
    GPUTier,
    OOMRecoveryStrategy,
)
from .training_config import (
    TrainingConfig,
    AugmentationConfig,
    OptimizerConfig,
    PerformanceConfig,
)
from .tensorboard_monitor import (
    TensorBoardServer,
    TensorBoardManager,
    CompetitionTensorBoardCallback,
    enable_ultralytics_tensorboard,
    check_tensorboard_available,
)

__all__ = [
    # Core training
    "CompetitionTrainer",
    "TrainingResult",
    "COMPETITION_CONFIG",
    "FAST_CONFIG",
    # GPU scaling
    "GPUScaler",
    "GPUScalingConfig",
    "GPUProfile",
    "GPUTier",
    "OOMRecoveryStrategy",
    # Configuration
    "TrainingConfig",
    "AugmentationConfig",
    "OptimizerConfig",
    "PerformanceConfig",
    # TensorBoard
    "TensorBoardServer",
    "TensorBoardManager",
    "CompetitionTensorBoardCallback",
    "enable_ultralytics_tensorboard",
    "check_tensorboard_available",
]
