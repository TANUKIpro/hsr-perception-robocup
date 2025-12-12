#!/usr/bin/env python3
"""
Training Configuration Management

Centralized configuration classes for YOLOv8 training with type safety
and GPU-aware presets.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .gpu_scaler import GPUProfile, GPUScaler, GPUTier


@dataclass
class AugmentationConfig:
    """Data augmentation settings for training."""

    # HSV color space augmentation (lighting condition adaptation)
    hsv_h: float = 0.015  # Hue shift (+-1.5%)
    hsv_s: float = 0.7  # Saturation shift (+-70%)
    hsv_v: float = 0.4  # Value shift (+-40%)

    # Geometric transformations
    degrees: float = 10.0  # Rotation (+-10 degrees)
    translate: float = 0.1  # Translation (+-10%)
    scale: float = 0.5  # Scale (0.5x - 1.5x)
    shear: float = 2.0  # Shear (+-2 degrees)

    # Flip augmentation
    flipud: float = 0.0  # Vertical flip probability (disabled for objects)
    fliplr: float = 0.5  # Horizontal flip probability

    # Advanced augmentation
    mosaic: float = 1.0  # Mosaic augmentation probability
    mixup: float = 0.1  # MixUp augmentation probability

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for YOLO training."""
        return {
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
        }


@dataclass
class OptimizerConfig:
    """Optimizer settings for training."""

    optimizer: str = "AdamW"
    lr0: float = 0.001  # Initial learning rate
    lrf: float = 0.01  # Final learning rate scale (lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005

    # Layer-wise Learning Rate Decay (LLRD)
    llrd_enabled: bool = False  # Enable LLRD for fine-tuning
    llrd_decay_rate: float = 0.9  # LR decay factor per layer depth

    # Stochastic Weight Averaging (SWA)
    swa_enabled: bool = False  # Enable SWA for fine-tuning
    swa_start_epoch: int = 10  # Start SWA at (epochs - N)
    swa_lr: float = 0.0005  # SWA learning rate (~1/2 of base LR)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "llrd_enabled": self.llrd_enabled,
            "llrd_decay_rate": self.llrd_decay_rate,
            "swa_enabled": self.swa_enabled,
            "swa_start_epoch": self.swa_start_epoch,
            "swa_lr": self.swa_lr,
        }


@dataclass
class PerformanceConfig:
    """Performance and resource settings."""

    workers: int = 8  # Data loader workers
    cache: bool = True  # Cache images in RAM
    amp: bool = True  # Automatic mixed precision

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            "workers": self.workers,
            "cache": self.cache,
            "amp": self.amp,
        }


@dataclass
class CheckpointConfig:
    """Checkpoint and saving settings."""

    save: bool = True
    save_period: int = 5  # Save every N epochs
    exist_ok: bool = True  # Overwrite existing run

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YOLO training."""
        return {
            "save": self.save,
            "save_period": self.save_period,
            "exist_ok": self.exist_ok,
        }


@dataclass
class TrainingConfig:
    """
    Complete training configuration for YOLOv8.

    Provides type-safe configuration with GPU-aware presets.
    """

    # Model settings
    model: str = "yolov8m.pt"
    imgsz: int = 640

    # Training settings
    epochs: int = 50
    batch: int = 16
    patience: int = 10  # Early stopping patience

    # Close mosaic for final epochs (improves final accuracy)
    close_mosaic: int = 10

    # Sub-configurations
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for YOLO model.train().

        Returns:
            Complete configuration dictionary
        """
        config = {
            "model": self.model,
            "imgsz": self.imgsz,
            "epochs": self.epochs,
            "batch": self.batch,
            "patience": self.patience,
            "close_mosaic": self.close_mosaic,
            "augment": True,
        }

        # Merge sub-configurations
        config.update(self.augmentation.to_dict())
        config.update(self.optimizer.to_dict())
        config.update(self.performance.to_dict())
        config.update(self.checkpoint.to_dict())

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """
        Create TrainingConfig from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            TrainingConfig instance
        """
        augmentation = AugmentationConfig(
            hsv_h=config_dict.get("hsv_h", 0.015),
            hsv_s=config_dict.get("hsv_s", 0.7),
            hsv_v=config_dict.get("hsv_v", 0.4),
            degrees=config_dict.get("degrees", 10.0),
            translate=config_dict.get("translate", 0.1),
            scale=config_dict.get("scale", 0.5),
            shear=config_dict.get("shear", 2.0),
            flipud=config_dict.get("flipud", 0.0),
            fliplr=config_dict.get("fliplr", 0.5),
            mosaic=config_dict.get("mosaic", 1.0),
            mixup=config_dict.get("mixup", 0.1),
        )

        optimizer = OptimizerConfig(
            optimizer=config_dict.get("optimizer", "AdamW"),
            lr0=config_dict.get("lr0", 0.001),
            lrf=config_dict.get("lrf", 0.01),
            momentum=config_dict.get("momentum", 0.937),
            weight_decay=config_dict.get("weight_decay", 0.0005),
            llrd_enabled=config_dict.get("llrd_enabled", False),
            llrd_decay_rate=config_dict.get("llrd_decay_rate", 0.9),
            swa_enabled=config_dict.get("swa_enabled", False),
            swa_start_epoch=config_dict.get("swa_start_epoch", 10),
            swa_lr=config_dict.get("swa_lr", 0.0005),
        )

        performance = PerformanceConfig(
            workers=config_dict.get("workers", 8),
            cache=config_dict.get("cache", True),
            amp=config_dict.get("amp", True),
        )

        checkpoint = CheckpointConfig(
            save=config_dict.get("save", True),
            save_period=config_dict.get("save_period", 5),
            exist_ok=config_dict.get("exist_ok", True),
        )

        return cls(
            model=config_dict.get("model", "yolov8m.pt"),
            imgsz=config_dict.get("imgsz", 640),
            epochs=config_dict.get("epochs", 50),
            batch=config_dict.get("batch", 16),
            patience=config_dict.get("patience", 10),
            close_mosaic=config_dict.get("close_mosaic", 10),
            augmentation=augmentation,
            optimizer=optimizer,
            performance=performance,
            checkpoint=checkpoint,
        )

    @classmethod
    def from_gpu_profile(
        cls,
        profile: GPUProfile,
        fast_mode: bool = False,
    ) -> "TrainingConfig":
        """
        Create TrainingConfig from GPU profile.

        Args:
            profile: GPU hardware profile
            fast_mode: Use faster settings for testing

        Returns:
            GPU-optimized TrainingConfig
        """
        scaler = GPUScaler()
        scaler.profile = profile  # Use provided profile
        config_dict = scaler.get_optimal_config(fast_mode=fast_mode)
        return cls.from_dict(config_dict)

    @classmethod
    def auto_detect(cls, fast_mode: bool = False) -> "TrainingConfig":
        """
        Create TrainingConfig with auto-detected GPU settings.

        Args:
            fast_mode: Use faster settings for testing

        Returns:
            GPU-optimized TrainingConfig
        """
        scaler = GPUScaler()
        config_dict = scaler.get_optimal_config(fast_mode=fast_mode)
        return cls.from_dict(config_dict)

    @classmethod
    def competition_default(cls) -> "TrainingConfig":
        """
        Create competition-optimized configuration.

        Balanced settings for competition day (~45 min training).
        """
        return cls(
            model="yolov8m.pt",
            imgsz=640,
            epochs=50,
            batch=16,
            patience=10,
        )

    @classmethod
    def fast_test(cls) -> "TrainingConfig":
        """
        Create fast testing configuration.

        Smaller model and fewer epochs for quick validation.
        """
        return cls(
            model="yolov8s.pt",
            imgsz=480,
            epochs=30,
            batch=32,
            patience=5,
            performance=PerformanceConfig(
                workers=4,
                cache=True,
                amp=True,
            ),
        )

    @classmethod
    def for_tier(cls, tier: GPUTier) -> "TrainingConfig":
        """
        Create configuration for specific GPU tier.

        Args:
            tier: GPU performance tier

        Returns:
            Tier-optimized TrainingConfig
        """
        from .gpu_scaler import TIER_CONFIGS

        config_dict = TIER_CONFIGS.get(tier, TIER_CONFIGS[GPUTier.MEDIUM])
        return cls.from_dict(config_dict)

    def with_overrides(self, **kwargs) -> "TrainingConfig":
        """
        Create new config with overridden values.

        Args:
            **kwargs: Values to override

        Returns:
            New TrainingConfig with overrides applied
        """
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return TrainingConfig.from_dict(config_dict)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Training Configuration:\n"
            f"  Model: {self.model}\n"
            f"  Image Size: {self.imgsz}\n"
            f"  Epochs: {self.epochs}\n"
            f"  Batch Size: {self.batch}\n"
            f"  Patience: {self.patience}\n"
            f"  Workers: {self.performance.workers}\n"
            f"  Cache: {self.performance.cache}\n"
            f"  AMP: {self.performance.amp}"
        )


# Legacy compatibility: COMPETITION_CONFIG and FAST_CONFIG as dictionaries
def get_competition_config() -> Dict[str, Any]:
    """Get competition-optimized configuration as dictionary."""
    return TrainingConfig.competition_default().to_dict()


def get_fast_config() -> Dict[str, Any]:
    """Get fast testing configuration as dictionary."""
    return TrainingConfig.fast_test().to_dict()


# For backwards compatibility
COMPETITION_CONFIG = get_competition_config()
FAST_CONFIG = get_fast_config()
