#!/usr/bin/env python3
"""
SWA-enabled YOLOv8 Trainer

Implements Stochastic Weight Averaging (SWA) for improved fine-tuning performance.
SWA averages model weights during the final training epochs, finding flatter minima
that generalize better to unseen data.

Expected improvement: +1-2% mAP on few-shot datasets.

Reference:
- Averaging Weights Leads to Wider Optima and Better Generalization (arXiv:1803.05407)
- PyTorch SWA: https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from ultralytics.utils import LOGGER, colorstr


@dataclass
class SWAConfig:
    """
    Configuration for Stochastic Weight Averaging.

    Attributes:
        enabled: Whether SWA is enabled
        swa_start_epoch: Number of final epochs to apply SWA
                        SWA starts at (total_epochs - swa_start_epoch)
        swa_lr: Learning rate during SWA phase (typically ~1/2 of base LR)
        update_bn: Whether to update batch normalization statistics after SWA
    """

    enabled: bool = False
    swa_start_epoch: int = 10  # Last N epochs for SWA
    swa_lr: float = 0.0005
    update_bn: bool = True  # Auto update BN statistics

    def __post_init__(self):
        """Validate configuration values."""
        if self.swa_start_epoch < 1:
            raise ValueError(
                f"swa_start_epoch must be >= 1, got {self.swa_start_epoch}"
            )
        if self.swa_lr <= 0:
            raise ValueError(f"swa_lr must be > 0, got {self.swa_lr}")


class SWACallback:
    """
    Callback for Stochastic Weight Averaging with Ultralytics YOLO.

    This callback integrates SWA into the YOLO training loop by:
    1. Creating an AveragedModel to track weight averages
    2. Updating averaged weights during the final N epochs
    3. Applying averaged weights and updating BN statistics after training

    Usage:
        swa_config = SWAConfig(enabled=True, swa_start_epoch=10)
        swa_callback = SWACallback(swa_config)

        model = YOLO("yolov8m.pt")
        model.add_callback("on_train_epoch_end", swa_callback.on_train_epoch_end)
        model.add_callback("on_train_end", swa_callback.on_train_end)
        model.train(data=dataset_yaml, epochs=50, ...)
    """

    def __init__(self, config: SWAConfig):
        """
        Initialize SWA callback.

        Args:
            config: SWA configuration
        """
        self.config = config
        self.swa_model: Optional[AveragedModel] = None
        self.swa_scheduler: Optional[SWALR] = None
        self.swa_start: int = 0
        self.swa_n_averaged: int = 0
        self.initialized: bool = False
        self.save_dir: Optional[Path] = None

        if self.config.enabled:
            LOGGER.info(
                f"{colorstr('SWA:')} Callback created - will average last "
                f"{self.config.swa_start_epoch} epochs, swa_lr={self.config.swa_lr}"
            )

    def _initialize(self, trainer) -> None:
        """
        Initialize SWA model and scheduler.

        Called on first epoch end to ensure model and optimizer exist.

        Args:
            trainer: Ultralytics trainer instance
        """
        if self.initialized or not self.config.enabled:
            return

        # Calculate SWA start epoch
        total_epochs = trainer.epochs

        # Validate swa_start_epoch against total epochs
        if self.config.swa_start_epoch >= total_epochs:
            LOGGER.warning(
                f"{colorstr('SWA:')} swa_start_epoch ({self.config.swa_start_epoch}) "
                f">= total_epochs ({total_epochs}). SWA will not be applied."
            )
            self.config.enabled = False
            return

        self.swa_start = total_epochs - self.config.swa_start_epoch
        self.save_dir = trainer.save_dir

        LOGGER.info(
            f"{colorstr('SWA:')} Initializing - will start averaging at epoch {self.swa_start} "
            f"(total epochs: {total_epochs})"
        )

        # Create AveragedModel for weight averaging
        device = next(trainer.model.parameters()).device
        self.swa_model = AveragedModel(trainer.model, device=device)

        # Note: We don't use SWALR scheduler to avoid conflicts with YOLO's built-in
        # scheduler. Instead, we manually set the LR when SWA starts.
        self.swa_scheduler = None

        self.initialized = True

    def on_train_epoch_end(self, trainer) -> None:
        """
        Called at the end of each training epoch.

        Updates SWA model weights during the final N epochs.

        Args:
            trainer: Ultralytics trainer instance
        """
        if not self.config.enabled:
            return

        # Initialize on first call
        self._initialize(trainer)

        current_epoch = trainer.epoch
        if current_epoch >= self.swa_start:
            self._update_swa_weights(trainer)

    def _update_swa_weights(self, trainer) -> None:
        """
        Update SWA model with current model weights.

        Args:
            trainer: Ultralytics trainer instance
        """
        if self.swa_model is None:
            return

        # On first SWA update, set optimizer LR to SWA LR
        # This avoids conflicts with YOLO's built-in LR scheduler
        if self.swa_n_averaged == 0:
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = self.config.swa_lr
            LOGGER.info(
                f"{colorstr('SWA:')} Set learning rate to {self.config.swa_lr} for averaging phase"
            )

        # Update averaged model parameters
        self.swa_model.update_parameters(trainer.model)
        self.swa_n_averaged += 1

        LOGGER.info(
            f"{colorstr('SWA:')} Updated weights at epoch {trainer.epoch + 1} "
            f"(averaged {self.swa_n_averaged} models)"
        )

    def on_train_end(self, trainer) -> None:
        """
        Called when training ends.

        Applies SWA averaged weights and updates batch normalization statistics.

        Args:
            trainer: Ultralytics trainer instance
        """
        if not self.config.enabled or self.swa_n_averaged == 0:
            return

        LOGGER.info(
            f"{colorstr('SWA:')} Training complete - applying averaged weights "
            f"from {self.swa_n_averaged} models"
        )

        # Apply averaged weights to model
        self._apply_swa_weights(trainer)

        # Update batch normalization statistics
        if self.config.update_bn:
            self._update_bn_statistics(trainer)

        # Save SWA model
        self._save_swa_model(trainer)

    def _apply_swa_weights(self, trainer) -> None:
        """
        Apply SWA averaged weights to the model.

        Args:
            trainer: Ultralytics trainer instance
        """
        if self.swa_model is None:
            return

        # Copy averaged weights to model
        trainer.model.load_state_dict(self.swa_model.module.state_dict())
        LOGGER.info(f"{colorstr('SWA:')} Applied averaged weights to model")

    def _update_bn_statistics(self, trainer) -> None:
        """
        Update batch normalization statistics using training data.

        This is necessary because SWA averages weights but not BN statistics.

        Args:
            trainer: Ultralytics trainer instance
        """
        LOGGER.info(f"{colorstr('SWA:')} Updating BatchNorm statistics...")

        try:
            # Set model to train mode for BN update
            self.swa_model.train()

            # Use PyTorch's update_bn utility on the underlying module
            # update_bn expects the actual model, not the AveragedModel wrapper
            update_bn(trainer.train_loader, self.swa_model)

            # Copy updated weights and BN stats back to trainer model
            trainer.model.load_state_dict(self.swa_model.module.state_dict())
            LOGGER.info(f"{colorstr('SWA:')} BatchNorm statistics updated successfully")
        except Exception as e:
            LOGGER.warning(
                f"{colorstr('SWA:')} Failed to update BatchNorm statistics: {e}"
            )
            LOGGER.warning(
                f"{colorstr('SWA:')} Continuing with existing BN statistics"
            )

    def _save_swa_model(self, trainer) -> None:
        """
        Save the SWA model weights.

        Saves to 'weights/swa.pt' in the run directory.

        Args:
            trainer: Ultralytics trainer instance
        """
        if self.save_dir is None:
            self.save_dir = trainer.save_dir

        swa_path = Path(self.save_dir) / "weights" / "swa.pt"
        swa_path.parent.mkdir(parents=True, exist_ok=True)

        # Create checkpoint dict similar to Ultralytics format
        ckpt = {
            "epoch": trainer.epoch,
            "model": copy.deepcopy(trainer.model).half(),
            "swa_n_averaged": self.swa_n_averaged,
            "swa_config": {
                "enabled": self.config.enabled,
                "swa_start_epoch": self.config.swa_start_epoch,
                "swa_lr": self.config.swa_lr,
                "update_bn": self.config.update_bn,
            },
        }

        torch.save(ckpt, swa_path)
        LOGGER.info(f"{colorstr('SWA:')} Saved SWA model to {swa_path}")


def create_swa_callback(
    enabled: bool = False,
    swa_start_epoch: int = 10,
    swa_lr: float = 0.0005,
    update_bn: bool = True,
) -> Optional[SWACallback]:
    """
    Factory function to create SWA callback.

    Args:
        enabled: Whether SWA is enabled
        swa_start_epoch: Last N epochs for SWA
        swa_lr: Learning rate during SWA phase
        update_bn: Whether to update BN statistics after SWA

    Returns:
        SWACallback instance if enabled, None otherwise
    """
    if not enabled:
        return None

    config = SWAConfig(
        enabled=enabled,
        swa_start_epoch=swa_start_epoch,
        swa_lr=swa_lr,
        update_bn=update_bn,
    )
    return SWACallback(config)


def register_swa_callbacks(model, swa_callback: SWACallback) -> None:
    """
    Register SWA callbacks with a YOLO model.

    Args:
        model: Ultralytics YOLO model
        swa_callback: SWACallback instance
    """
    if swa_callback is None or not swa_callback.config.enabled:
        return

    model.add_callback("on_train_epoch_end", swa_callback.on_train_epoch_end)
    model.add_callback("on_train_end", swa_callback.on_train_end)

    LOGGER.info(f"{colorstr('SWA:')} Callbacks registered with model")
