"""
HSR Perception - Training Module

Competition day fine-tuning scripts for YOLOv8 models.

Features:
- GPU hardware auto-scaling
- TensorBoard integration with competition metrics
- OOM recovery with automatic parameter adjustment
"""

# Core training (facade)
from .quick_finetune import CompetitionTrainer

# Extracted modules (new structure)
from .config_manager import (
    COMPETITION_CONFIG,
    FAST_CONFIG,
    ConfigBuilder,
    TrainerConfig,
)
from .dataset_validator import (
    DatasetValidator,
    DatasetValidationResult,
    validate_dataset,
)
from .model_operations import (
    ModelValidator,
    ModelExporter,
    validate_model,
    export_model,
)
from .synthetic_data_manager import (
    SyntheticConfig,
    SyntheticDataManager,
    SyntheticGenerationResult,
    SYNTHETIC_CONFIG_KEYS,
    filter_synthetic_keys,
)
from .training_executor import (
    TrainingExecutor,
    TrainingResult,
)

# GPU scaling
from .gpu_scaler import (
    GPUScaler,
    GPUScalingConfig,
    GPUProfile,
    GPUTier,
    OOMRecoveryStrategy,
)

# Configuration dataclasses
from .training_config import (
    TrainingConfig,
    AugmentationConfig,
    OptimizerConfig,
    PerformanceConfig,
)

# TensorBoard
from .tensorboard_monitor import (
    TensorBoardServer,
    TensorBoardManager,
    CompetitionTensorBoardCallback,
    enable_ultralytics_tensorboard,
    check_tensorboard_available,
)

# Logging
from .training_logger import (
    setup_training_logger,
    get_logger,
    ColoredFormatter,
)

# Constants
from .training_constants import (
    VRAMThreshold,
    MODEL_VRAM_OVERHEAD_GB,
    MODEL_PER_SAMPLE_MEMORY_MB,
    MAX_OOM_RETRIES,
    OOM_BATCH_REDUCTION_FACTOR,
    MIN_BATCH_SIZE,
    DEFAULT_VRAM_SAFETY_MARGIN,
    DEFAULT_MAX_VRAM_UTILIZATION,
    STANDARD_IMAGE_SIZES,
)

__all__ = [
    # Core training (facade)
    "CompetitionTrainer",
    # Extracted modules
    "COMPETITION_CONFIG",
    "FAST_CONFIG",
    "ConfigBuilder",
    "TrainerConfig",
    "DatasetValidator",
    "DatasetValidationResult",
    "validate_dataset",
    "ModelValidator",
    "ModelExporter",
    "validate_model",
    "export_model",
    "SyntheticConfig",
    "SyntheticDataManager",
    "SyntheticGenerationResult",
    "SYNTHETIC_CONFIG_KEYS",
    "filter_synthetic_keys",
    "TrainingExecutor",
    "TrainingResult",
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
    # Logging
    "setup_training_logger",
    "get_logger",
    "ColoredFormatter",
    # Constants
    "VRAMThreshold",
    "MODEL_VRAM_OVERHEAD_GB",
    "MODEL_PER_SAMPLE_MEMORY_MB",
    "MAX_OOM_RETRIES",
    "OOM_BATCH_REDUCTION_FACTOR",
    "MIN_BATCH_SIZE",
    "DEFAULT_VRAM_SAFETY_MARGIN",
    "DEFAULT_MAX_VRAM_UTILIZATION",
    "STANDARD_IMAGE_SIZES",
]
