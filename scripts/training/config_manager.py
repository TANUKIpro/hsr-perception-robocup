"""
Configuration management for competition training.

Provides training configuration constants and builder utilities.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


# Competition-optimized training configuration
# Updated for better generalization based on Tier 1 recommendations
COMPETITION_CONFIG = {
    # Model settings
    "model": "yolov8m.pt",
    "imgsz": 640,
    # Training settings
    "epochs": 50,
    "batch": 16,
    "patience": 10,  # Early stopping patience
    # Optimizer settings
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.001,  # Increased from 0.0005 for better regularization
    # Warmup settings (for stable fine-tuning)
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    # Layer freeze (prevent overfitting on small datasets)
    "freeze": 10,  # Freeze first 10 backbone layers
    # Augmentation settings
    "augment": True,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 10.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 0.7,  # Reduced from 0.8 for better generalization on small datasets
    "mixup": 0.1,
    # Performance settings
    "workers": 8,
    "cache": True,  # Cache images in RAM for speed
    "amp": True,  # Automatic mixed precision
    "close_mosaic": 20,  # Increased from 15 for longer pure-image training
    # Overfitting prevention (Tier 1) - enabled for better generalization
    "label_smoothing": 0.05,  # Enabled for overfitting prevention
    "cos_lr": True,  # Enabled for smoother LR decay
    "multi_scale": False,  # Keep False due to VRAM consumption
    # Checkpointing
    "save": True,
    "save_period": 5,
    "exist_ok": True,
    # LLRD (Layer-wise Learning Rate Decay)
    # Disabled by default for backward compatibility
    "llrd_enabled": False,
    "llrd_decay_rate": 0.9,  # LR decay factor per layer (0.0-1.0]
    # Dynamic Copy-Paste settings
    "dynamic_synthetic_enabled": True,  # Enable by default (user requirement)
    "synthetic_ratio": 2.0,
    "synthetic_scale_range": (0.5, 1.5),
    "synthetic_rotation_range": (-15.0, 15.0),
    "synthetic_white_balance": True,
    "synthetic_white_balance_strength": 0.7,
    "synthetic_edge_blur": 2.0,
    "synthetic_max_objects": 3,
    "synthetic_num_workers": 0,  # 0 = auto (cpu_count // 2), >0 = explicit worker count
    "backgrounds_dir": None,  # Passed from UI
    "annotated_dir": None,  # Passed from UI
}

# Fast training configuration (for testing or limited GPU)
FAST_CONFIG = {
    **COMPETITION_CONFIG,
    "model": "yolov8s.pt",  # Smaller model
    "epochs": 30,
    "batch": 32,
    "patience": 5,
    "imgsz": 480,
}


@dataclass
class TrainerConfig:
    """Configuration for CompetitionTrainer."""

    base_model: Optional[str] = None
    output_dir: str = "models/finetuned"
    auto_scale: bool = True
    tensorboard: bool = True
    tensorboard_port: int = 6006
    training_config: Optional[Dict] = field(default=None)

    def generate_run_name(self) -> str:
        """Generate unique run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"competition_{timestamp}"


class ConfigBuilder:
    """Builder for training configurations."""

    def __init__(self, base_config: Optional[Dict] = None):
        """
        Initialize builder with base configuration.

        Args:
            base_config: Base configuration to build upon (defaults to COMPETITION_CONFIG)
        """
        self.config = (base_config or COMPETITION_CONFIG).copy()

    def with_epochs(self, epochs: int) -> "ConfigBuilder":
        """Set number of training epochs."""
        self.config["epochs"] = epochs
        return self

    def with_batch_size(self, batch: int) -> "ConfigBuilder":
        """Set batch size."""
        self.config["batch"] = batch
        return self

    def with_image_size(self, imgsz: int) -> "ConfigBuilder":
        """Set image size."""
        self.config["imgsz"] = imgsz
        return self

    def with_model(self, model: str) -> "ConfigBuilder":
        """Set base model."""
        self.config["model"] = model
        return self

    def with_llrd(
        self,
        enabled: bool = True,
        decay_rate: float = 0.9,
    ) -> "ConfigBuilder":
        """
        Configure Layer-wise Learning Rate Decay.

        Args:
            enabled: Whether to enable LLRD
            decay_rate: LR decay factor per layer (0.0-1.0]
        """
        self.config["llrd_enabled"] = enabled
        self.config["llrd_decay_rate"] = decay_rate
        return self

    def with_swa(
        self,
        enabled: bool = True,
        start_epoch: int = 10,
        lr: float = 0.0005,
    ) -> "ConfigBuilder":
        """
        Configure Stochastic Weight Averaging.

        Args:
            enabled: Whether to enable SWA
            start_epoch: Number of final epochs to apply SWA
            lr: Learning rate during SWA phase
        """
        self.config["swa_enabled"] = enabled
        self.config["swa_start_epoch"] = start_epoch
        self.config["swa_lr"] = lr
        return self

    def with_synthetic(
        self,
        enabled: bool = True,
        backgrounds_dir: Optional[str] = None,
        annotated_dir: Optional[str] = None,
        ratio: float = 2.0,
    ) -> "ConfigBuilder":
        """
        Configure dynamic Copy-Paste synthetic data generation.

        Args:
            enabled: Whether to enable synthetic generation
            backgrounds_dir: Directory containing background images
            annotated_dir: Directory containing annotated images
            ratio: Synthetic to real image ratio
        """
        self.config["dynamic_synthetic_enabled"] = enabled
        if backgrounds_dir:
            self.config["backgrounds_dir"] = backgrounds_dir
        if annotated_dir:
            self.config["annotated_dir"] = annotated_dir
        self.config["synthetic_ratio"] = ratio
        return self

    def with_freeze(self, layers: int) -> "ConfigBuilder":
        """Set number of backbone layers to freeze."""
        self.config["freeze"] = layers
        return self

    def build(self) -> Dict:
        """Build and return the configuration."""
        return self.config.copy()
