#!/usr/bin/env python3
"""
GPU Hardware Auto-Scaling for YOLOv8 Training

Automatically detects GPU capabilities and optimizes training parameters
for various hardware configurations (RTX 2080 to A100/H100).
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from colorama import Fore, Style


class GPUTier(Enum):
    """GPU performance classification based on VRAM."""

    LOW = "low"  # < 6GB VRAM (GTX 1060, GTX 1650)
    MEDIUM = "medium"  # 6-12GB VRAM (RTX 2080, RTX 3070)
    HIGH = "high"  # 12-24GB VRAM (RTX 3090, RTX 4090)
    WORKSTATION = "workstation"  # > 24GB VRAM (A100, H100)
    CPU_ONLY = "cpu"


@dataclass
class GPUProfile:
    """Complete GPU hardware profile."""

    name: str
    device_index: int
    total_memory_gb: float
    available_memory_gb: float
    tier: GPUTier
    compute_capability: Tuple[int, int]
    multi_processor_count: int
    supports_bf16: bool
    supports_tf32: bool

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"GPU: {self.name}\n"
            f"  Memory: {self.total_memory_gb:.1f} GB total, "
            f"{self.available_memory_gb:.1f} GB available\n"
            f"  Tier: {self.tier.value}\n"
            f"  Compute: SM {self.compute_capability[0]}.{self.compute_capability[1]}\n"
            f"  Features: BF16={self.supports_bf16}, TF32={self.supports_tf32}"
        )


@dataclass
class GPUScalingConfig:
    """Configuration for GPU scaling behavior."""

    auto_detect: bool = True
    force_gpu_tier: Optional[GPUTier] = None
    vram_safety_margin: float = 0.15  # 15% safety margin
    max_vram_utilization: float = 0.85  # Max 85% utilization
    enable_oom_recovery: bool = True
    oom_batch_reduction_factor: float = 0.5
    max_oom_retries: int = 3


# VRAM thresholds for GPU classification (in GB)
VRAM_THRESHOLDS = {
    GPUTier.LOW: (0, 6),
    GPUTier.MEDIUM: (6, 12),
    GPUTier.HIGH: (12, 24),
    GPUTier.WORKSTATION: (24, float("inf")),
}

# Model VRAM overhead (empirical values in GB)
MODEL_VRAM_OVERHEAD = {
    "yolov8n.pt": 1.2,
    "yolov8s.pt": 1.8,
    "yolov8m.pt": 2.5,
    "yolov8l.pt": 3.5,
    "yolov8x.pt": 5.0,
}

# Per-sample memory at 640x640 (empirical, in MB)
MODEL_PER_SAMPLE_MEMORY_MB = {
    "yolov8n.pt": 180,
    "yolov8s.pt": 250,
    "yolov8m.pt": 380,
    "yolov8l.pt": 550,
    "yolov8x.pt": 750,
}

# Standard image sizes
STANDARD_IMAGE_SIZES = [320, 480, 640, 800, 1024, 1280]

# Tier-based recommended configurations
TIER_CONFIGS = {
    GPUTier.LOW: {
        "model": "yolov8s.pt",
        "batch": 8,
        "imgsz": 480,
        "workers": 4,
        "cache": False,
        "epochs": 40,
        "patience": 8,
    },
    GPUTier.MEDIUM: {
        "model": "yolov8m.pt",
        "batch": 16,
        "imgsz": 640,
        "workers": 8,
        "cache": True,
        "epochs": 50,
        "patience": 10,
    },
    GPUTier.HIGH: {
        "model": "yolov8l.pt",
        "batch": 32,
        "imgsz": 640,
        "workers": 8,
        "cache": True,
        "epochs": 50,
        "patience": 10,
    },
    GPUTier.WORKSTATION: {
        "model": "yolov8x.pt",
        "batch": 64,
        "imgsz": 800,
        "workers": 8,
        "cache": True,
        "epochs": 50,
        "patience": 10,
    },
    GPUTier.CPU_ONLY: {
        "model": "yolov8n.pt",
        "batch": 4,
        "imgsz": 320,
        "workers": 2,
        "cache": False,
        "epochs": 30,
        "patience": 5,
    },
}

# Model downgrade path for OOM recovery
MODEL_DOWNGRADE_PATH = {
    "yolov8x.pt": "yolov8l.pt",
    "yolov8l.pt": "yolov8m.pt",
    "yolov8m.pt": "yolov8s.pt",
    "yolov8s.pt": "yolov8n.pt",
    "yolov8n.pt": "yolov8n.pt",  # No further downgrade
}


class GPUScaler:
    """
    GPU hardware auto-detection and training parameter optimization.

    Automatically detects GPU capabilities and provides optimal training
    configurations for various hardware configurations.
    """

    def __init__(self, config: Optional[GPUScalingConfig] = None):
        """
        Initialize GPU scaler.

        Args:
            config: Scaling configuration (uses defaults if None)
        """
        self.config = config or GPUScalingConfig()
        self.profile: Optional[GPUProfile] = None

        if self.config.auto_detect:
            self._detect_gpu()

    def _detect_gpu(self) -> None:
        """Detect and profile GPU hardware."""
        try:
            import torch

            if not torch.cuda.is_available():
                self.profile = None
                return

            # Get device properties
            props = torch.cuda.get_device_properties(0)

            # Get available memory
            torch.cuda.synchronize()
            total_memory = props.total_memory / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            available = total_memory - reserved

            # Determine tier
            tier = self._classify_tier(total_memory)

            # Check feature support
            compute_cap = (props.major, props.minor)
            supports_bf16 = compute_cap >= (8, 0)  # Ampere and later
            supports_tf32 = compute_cap >= (8, 0)

            self.profile = GPUProfile(
                name=props.name,
                device_index=0,
                total_memory_gb=total_memory,
                available_memory_gb=available,
                tier=tier,
                compute_capability=compute_cap,
                multi_processor_count=props.multi_processor_count,
                supports_bf16=supports_bf16,
                supports_tf32=supports_tf32,
            )

        except ImportError:
            self.profile = None
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: GPU detection failed: {e}{Style.RESET_ALL}")
            self.profile = None

    def _classify_tier(self, vram_gb: float) -> GPUTier:
        """Classify GPU into performance tier based on VRAM."""
        if self.config.force_gpu_tier:
            return self.config.force_gpu_tier

        for tier, (min_vram, max_vram) in VRAM_THRESHOLDS.items():
            if min_vram <= vram_gb < max_vram:
                return tier

        return GPUTier.MEDIUM  # Default fallback

    def get_profile(self) -> Optional[GPUProfile]:
        """Get detected GPU profile."""
        return self.profile

    def get_tier(self) -> GPUTier:
        """Get GPU tier."""
        if self.profile:
            return self.profile.tier
        return GPUTier.CPU_ONLY

    def get_optimal_config(self, fast_mode: bool = False) -> Dict:
        """
        Get optimal training configuration for current GPU.

        Args:
            fast_mode: Use faster/smaller settings for testing

        Returns:
            Training configuration dictionary
        """
        tier = self.get_tier()
        base_config = TIER_CONFIGS[tier].copy()

        if fast_mode:
            base_config = self._apply_fast_mode(base_config)

        # Calculate dynamic batch size if GPU available
        if self.profile:
            base_config["batch"] = self.calculate_batch_size(
                base_config["model"],
                base_config["imgsz"],
            )

        # Common settings
        base_config.update(
            {
                "optimizer": "AdamW",
                "lr0": 0.001,
                "lrf": 0.01,
                "momentum": 0.937,
                "weight_decay": 0.0005,
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
                "mosaic": 1.0,
                "mixup": 0.1,
                "amp": True,
                "close_mosaic": 10,
                "save": True,
                "save_period": 5,
                "exist_ok": True,
            }
        )

        return base_config

    def _apply_fast_mode(self, config: Dict) -> Dict:
        """Apply fast mode settings for testing."""
        fast_config = config.copy()

        # Use smaller model if not already smallest
        model = config.get("model", "yolov8m.pt")
        if model in MODEL_DOWNGRADE_PATH:
            fast_config["model"] = MODEL_DOWNGRADE_PATH.get(model, model)

        # Reduce epochs and patience
        fast_config["epochs"] = min(30, config.get("epochs", 50))
        fast_config["patience"] = min(5, config.get("patience", 10))

        # Reduce image size if possible
        imgsz = config.get("imgsz", 640)
        if imgsz > 480:
            fast_config["imgsz"] = 480

        return fast_config

    def calculate_batch_size(
        self,
        model: str,
        imgsz: int,
        target_utilization: Optional[float] = None,
    ) -> int:
        """
        Calculate optimal batch size based on GPU profile.

        Args:
            model: Model name (e.g., 'yolov8m.pt')
            imgsz: Image size
            target_utilization: Target VRAM utilization (0.0-1.0)

        Returns:
            Optimal batch size
        """
        if not self.profile:
            return 8  # Conservative default for CPU

        target_util = target_utilization or self.config.max_vram_utilization
        available_gb = self.profile.available_memory_gb * target_util

        # Get model overhead and per-sample memory
        overhead_gb = MODEL_VRAM_OVERHEAD.get(model, 2.5)
        per_sample_mb = MODEL_PER_SAMPLE_MEMORY_MB.get(model, 380)

        # Scale per-sample memory by image size ratio
        size_ratio = (imgsz / 640) ** 2
        per_sample_mb *= size_ratio

        # Calculate available memory for samples
        available_for_samples_mb = (available_gb - overhead_gb) * 1024

        if available_for_samples_mb <= 0:
            return 4  # Minimum batch size

        # Calculate batch size
        batch_size = int(available_for_samples_mb / per_sample_mb)

        # Clamp to reasonable range
        batch_size = max(4, min(batch_size, 128))

        # Round down to power of 2 for CUDA efficiency
        batch_size = 2 ** int(math.log2(batch_size))

        return batch_size

    def estimate_training_time(
        self,
        epochs: int,
        dataset_size: int,
        batch_size: int,
    ) -> float:
        """
        Estimate training time in minutes.

        Args:
            epochs: Number of training epochs
            dataset_size: Number of training images
            batch_size: Batch size

        Returns:
            Estimated training time in minutes
        """
        if not self.profile:
            # Conservative CPU estimate
            iterations_per_epoch = math.ceil(dataset_size / batch_size)
            seconds_per_iteration = 5.0  # Very slow on CPU
            return (epochs * iterations_per_epoch * seconds_per_iteration) / 60

        # GPU estimates based on tier
        seconds_per_iteration = {
            GPUTier.LOW: 0.5,
            GPUTier.MEDIUM: 0.25,
            GPUTier.HIGH: 0.15,
            GPUTier.WORKSTATION: 0.08,
            GPUTier.CPU_ONLY: 5.0,
        }.get(self.profile.tier, 0.25)

        iterations_per_epoch = math.ceil(dataset_size / batch_size)
        total_seconds = epochs * iterations_per_epoch * seconds_per_iteration

        # Add overhead for validation, checkpointing, etc. (10%)
        total_seconds *= 1.1

        return total_seconds / 60

    def get_summary(self) -> str:
        """Get human-readable summary of GPU and recommended settings."""
        if not self.profile:
            return "No GPU detected. Using CPU mode (training will be slow)."

        config = self.get_optimal_config()
        return (
            f"{self.profile.summary()}\n\n"
            f"Recommended Configuration:\n"
            f"  Model: {config['model']}\n"
            f"  Batch Size: {config['batch']}\n"
            f"  Image Size: {config['imgsz']}\n"
            f"  Workers: {config['workers']}\n"
            f"  Cache: {config['cache']}\n"
            f"  Epochs: {config['epochs']}"
        )


class OOMRecoveryStrategy:
    """
    Handles Out-of-Memory recovery with progressive parameter reduction.

    Recovery steps:
    1. Reduce batch size by 50%
    2. Reduce image size
    3. Downgrade to smaller model
    """

    def __init__(
        self,
        initial_config: Dict,
        max_retries: int = 3,
    ):
        """
        Initialize OOM recovery strategy.

        Args:
            initial_config: Initial training configuration
            max_retries: Maximum number of recovery attempts
        """
        self.initial_config = initial_config.copy()
        self.current_config = initial_config.copy()
        self.max_retries = max_retries
        self.retry_count = 0
        self.changes: List[str] = []

    def get_recovery_config(self) -> Optional[Dict]:
        """
        Get configuration for OOM recovery attempt.

        Returns:
            New config dict or None if max retries exceeded.
        """
        if self.retry_count >= self.max_retries:
            return None

        self.retry_count += 1
        new_config = self.current_config.copy()

        if self.retry_count == 1:
            # First retry: Reduce batch size by 50%
            old_batch = new_config.get("batch", 16)
            new_batch = max(4, old_batch // 2)
            new_config["batch"] = new_batch
            self.changes.append(f"Batch: {old_batch} -> {new_batch}")

        elif self.retry_count == 2:
            # Second retry: Also reduce image size
            current_imgsz = new_config.get("imgsz", 640)
            if current_imgsz in STANDARD_IMAGE_SIZES:
                current_idx = STANDARD_IMAGE_SIZES.index(current_imgsz)
                # Move to smaller image size (lower index)
                new_idx = max(0, current_idx - 1)
                new_imgsz = STANDARD_IMAGE_SIZES[new_idx]
            else:
                # Find the next smaller standard size
                new_imgsz = max(s for s in STANDARD_IMAGE_SIZES if s < current_imgsz)

            if new_imgsz < current_imgsz:
                new_config["imgsz"] = new_imgsz
                self.changes.append(f"Image size: {current_imgsz} -> {new_imgsz}")

        elif self.retry_count == 3:
            # Third retry: Use smaller model
            model = new_config.get("model", "yolov8m.pt")
            new_model = MODEL_DOWNGRADE_PATH.get(model, model)
            if new_model != model:
                new_config["model"] = new_model
                self.changes.append(f"Model: {model} -> {new_model}")

            # Also reduce batch size again
            old_batch = new_config.get("batch", 16)
            new_batch = max(4, old_batch // 2)
            new_config["batch"] = new_batch
            self.changes.append(f"Batch: {old_batch} -> {new_batch}")

        # Clear CUDA cache before retry
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        self.current_config = new_config
        return new_config

    def get_changes_summary(self) -> str:
        """Get summary of changes made during recovery."""
        if not self.changes:
            return "No changes made"
        return "; ".join(self.changes)

    def reset(self) -> None:
        """Reset recovery state."""
        self.current_config = self.initial_config.copy()
        self.retry_count = 0
        self.changes = []


def get_gpu_scaler(config: Optional[GPUScalingConfig] = None) -> GPUScaler:
    """
    Factory function to create GPUScaler instance.

    Args:
        config: Optional scaling configuration

    Returns:
        Configured GPUScaler instance
    """
    return GPUScaler(config)


def print_gpu_info() -> None:
    """Print GPU information to console."""
    scaler = GPUScaler()
    print(scaler.get_summary())


if __name__ == "__main__":
    # Print GPU information when run directly
    print_gpu_info()
