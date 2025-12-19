#!/usr/bin/env python3
"""
Quick Fine-Tuning Script for Competition Day

Optimized YOLOv8 fine-tuning for rapid model adaptation during competitions.
Features:
- GPU hardware auto-scaling for various GPU configurations
- TensorBoard integration with competition-specific metrics
- OOM recovery with automatic parameter adjustment

Designed to complete training within ~45-60 minutes on a capable GPU.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from colorama import Fore, Style, init as colorama_init

colorama_init()

# Add scripts directory to path for common module imports
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from common.device_utils import log_gpu_status
from common.constants import TARGET_MAP50

# Import new modules
from .gpu_scaler import GPUScaler, GPUScalingConfig, OOMRecoveryStrategy
from .tensorboard_monitor import (
    CompetitionTensorBoardCallback,
    TensorBoardServer,
    TensorBoardManager,
    enable_ultralytics_tensorboard,
    check_tensorboard_available,
)
from .training_config import TrainingConfig
from .swa_trainer import SWAConfig, SWACallback, create_swa_callback, register_swa_callbacks
from .llrd_trainer import LLRDDetectionTrainer, LLRDConfig
from .memory_utils import full_training_cleanup, cleanup_cuda_memory, log_memory_snapshot

# Import augmentation modules
sys.path.insert(0, str(_scripts_dir / "augmentation"))
from augmentation.copy_paste_augmentor import CopyPasteAugmentor, CopyPasteConfig

# Keys that should NOT be passed to YOLO's train() method
SYNTHETIC_CONFIG_KEYS = {
    "dynamic_synthetic_enabled",
    "backgrounds_dir",
    "annotated_dir",
    "synthetic_ratio",
    "synthetic_scale_range",
    "synthetic_rotation_range",
    "synthetic_white_balance",
    "synthetic_white_balance_strength",
    "synthetic_edge_blur",
    "synthetic_max_objects",
}

# Competition-optimized training configuration (legacy compatibility)
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
    "multi_scale": False,  # Keep False due to VRAM consumption (enabled in GPU scaler for HIGH tier)
    # Checkpointing
    "save": True,
    "save_period": 5,
    "exist_ok": True,
    # LLRD (Layer-wise Learning Rate Decay)
    # Disabled by default for backward compatibility
    # Enable for potential +1-3% mAP improvement on small datasets
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
    "backgrounds_dir": None,  # Passed from UI
    "annotated_dir": None,    # Passed from UI
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
class TrainingResult:
    """Result of a training run."""

    best_model_path: str
    last_model_path: str
    metrics: Dict
    training_time_minutes: float
    epochs_completed: int
    config: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        """Generate summary string."""
        return f"""
{'='*60}
  Training Complete
{'='*60}
  Timestamp: {self.timestamp}
  Training Time: {self.training_time_minutes:.1f} minutes
  Epochs Completed: {self.epochs_completed}

  Best Model: {self.best_model_path}
  Last Model: {self.last_model_path}

  Metrics:
    mAP50: {self.metrics.get('mAP50', 'N/A'):.4f}
    mAP50-95: {self.metrics.get('mAP50-95', 'N/A'):.4f}
    Precision: {self.metrics.get('precision', 'N/A'):.4f}
    Recall: {self.metrics.get('recall', 'N/A'):.4f}
{'='*60}
"""

    def meets_target(self, target_map: float = 0.85) -> bool:
        """Check if training met target mAP."""
        return self.metrics.get("mAP50", 0) >= target_map

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "best_model_path": self.best_model_path,
            "last_model_path": self.last_model_path,
            "metrics": self.metrics,
            "training_time_minutes": self.training_time_minutes,
            "epochs_completed": self.epochs_completed,
            "config": self.config,
            "timestamp": self.timestamp,
        }


class CompetitionTrainer:
    """
    Competition-day YOLOv8 fine-tuning handler.

    Features:
    - GPU hardware auto-scaling
    - TensorBoard integration with competition metrics
    - OOM recovery with automatic parameter adjustment
    - Optimized for rapid training with early stopping
    """

    def __init__(
        self,
        base_model: Optional[str] = None,
        output_dir: str = "models/finetuned",
        config: Optional[Dict] = None,
        auto_scale: bool = True,
        tensorboard: bool = True,
        tensorboard_port: int = 6006,
    ):
        """
        Initialize trainer.

        Args:
            base_model: Path to pretrained model (auto-detected if None)
            output_dir: Directory for training outputs
            config: Training configuration (auto-scaled if None)
            auto_scale: Enable GPU auto-scaling
            tensorboard: Enable TensorBoard monitoring
            tensorboard_port: TensorBoard server port
        """
        self.output_dir = Path(output_dir)
        self.auto_scale = auto_scale
        self.tensorboard_enabled = tensorboard
        self.tensorboard_port = tensorboard_port

        # GPU scaling
        self.gpu_scaler: Optional[GPUScaler] = None
        if auto_scale:
            self.gpu_scaler = GPUScaler()
            if config is None:
                config = self.gpu_scaler.get_optimal_config()
                print(f"{Fore.CYAN}GPU Auto-scaling enabled:{Style.RESET_ALL}")
                print(self.gpu_scaler.get_summary())

        self.config = config or COMPETITION_CONFIG.copy()

        # Set base model from config or default
        if base_model is None:
            self.base_model = self.config.get("model", "yolov8m.pt")
        else:
            self.base_model = base_model

        self.run_name = self._generate_run_name()

        # TensorBoard setup
        self.tensorboard_server: Optional[TensorBoardServer] = None
        self.tensorboard_callback: Optional[CompetitionTensorBoardCallback] = None
        self.tensorboard_url: str = ""

        # SWA setup
        self.swa_callback: Optional[SWACallback] = None

        # Verify CUDA availability
        self._check_cuda()

    def _check_cuda(self) -> None:
        """Check CUDA availability and abort if not available for training."""
        has_gpu = log_gpu_status(verbose=True)
        if not has_gpu and not self.config.get("allow_cpu", False):
            raise RuntimeError(
                "GPU not available. Training on CPU would exceed time limit (2-3 hours). "
                "Check CUDA drivers or use --allow-cpu to override."
            )

    def _generate_run_name(self) -> str:
        """Generate unique run name with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"competition_{timestamp}"

    def _validate_dataset(self, dataset_yaml: str) -> bool:
        """Validate dataset configuration."""
        dataset_path = Path(dataset_yaml)

        if not dataset_path.exists():
            print(f"{Fore.RED}Error: Dataset config not found: {dataset_yaml}{Style.RESET_ALL}")
            return False

        with open(dataset_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required fields
        required = ["train", "val", "names"]
        for field in required:
            if field not in config:
                print(f"{Fore.RED}Error: Missing field '{field}' in dataset config{Style.RESET_ALL}")
                return False

        # Check paths exist
        base_path = dataset_path.parent
        train_path = base_path / config["train"]
        val_path = base_path / config["val"]

        if not train_path.exists():
            print(f"{Fore.RED}Error: Train path not found: {train_path}{Style.RESET_ALL}")
            return False

        if not val_path.exists():
            print(f"{Fore.RED}Error: Val path not found: {val_path}{Style.RESET_ALL}")
            return False

        # Count images
        train_count = len(list(train_path.glob("*")))
        val_count = len(list(val_path.glob("*")))

        print(f"Dataset validated:")
        print(f"  Train images: {train_count}")
        print(f"  Val images: {val_count}")
        print(f"  Classes: {len(config['names'])}")

        return True

    def _setup_tensorboard(self, model: Any) -> None:
        """Setup TensorBoard monitoring."""
        if not self.tensorboard_enabled:
            return

        if not check_tensorboard_available():
            print(
                f"{Fore.YELLOW}Warning: TensorBoard not available. "
                f"Install with: pip install tensorboard{Style.RESET_ALL}"
            )
            return

        # Enable Ultralytics built-in TensorBoard
        enable_ultralytics_tensorboard()

        # Create log directory
        log_dir = self.output_dir / self.run_name / "tensorboard"

        # Create custom callback
        self.tensorboard_callback = CompetitionTensorBoardCallback(
            log_dir=str(log_dir),
            target_map50=TARGET_MAP50,
        )

        # Register callbacks
        model.add_callback(
            "on_pretrain_routine_start",
            self.tensorboard_callback.on_pretrain_routine_start,
        )
        model.add_callback(
            "on_train_epoch_start",
            self.tensorboard_callback.on_train_epoch_start,
        )
        model.add_callback(
            "on_train_epoch_end",
            self.tensorboard_callback.on_train_epoch_end,
        )
        model.add_callback(
            "on_fit_epoch_end",
            self.tensorboard_callback.on_fit_epoch_end,
        )
        model.add_callback(
            "on_train_end",
            self.tensorboard_callback.on_train_end,
        )

        # Start TensorBoard server
        self.tensorboard_server = TensorBoardServer(
            str(log_dir),
            port=self.tensorboard_port,
        )
        self.tensorboard_url = self.tensorboard_server.start()

    def _cleanup_tensorboard(self) -> None:
        """Cleanup TensorBoard resources (server remains running for user access)."""
        # Note: We don't stop the server here so users can view results after training
        pass

    def _cleanup_callbacks(self, model: Any) -> None:
        """
        Clear all registered callbacks from the model.

        Args:
            model: YOLO model instance
        """
        try:
            if hasattr(model, 'callbacks'):
                # Clear all callback dictionaries
                for callback_name in model.callbacks:
                    if isinstance(model.callbacks[callback_name], dict):
                        model.callbacks[callback_name].clear()
                print(f"{Fore.GREEN}Cleared model callbacks{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Error clearing callbacks: {e}{Style.RESET_ALL}")

    def _cleanup_all_resources(
        self,
        model: Any,
        trainer: Any = None,
        stop_tensorboard: bool = False
    ) -> None:
        """
        Comprehensive cleanup of all training resources.

        Args:
            model: YOLO model to clean up
            trainer: Optional trainer object to clean up
            stop_tensorboard: Whether to stop TensorBoard server
        """
        print(f"\n{Fore.CYAN}Cleaning up training resources...{Style.RESET_ALL}")

        # Clear model callbacks first
        self._cleanup_callbacks(model)

        # Use comprehensive cleanup from memory_utils
        stats = full_training_cleanup(
            model=model.model if hasattr(model, 'model') else model,
            optimizer=trainer.optimizer if trainer and hasattr(trainer, 'optimizer') else None,
            swa_callback=self.swa_callback,
            tensorboard_callback=self.tensorboard_callback,
            tensorboard_server=self.tensorboard_server if stop_tensorboard else None,
            synchronize_cuda=True,
            num_gc_passes=2
        )

        print(f"{Fore.GREEN}Cleanup complete. "
              f"Freed {stats.get('cuda_freed_mb', 0):.1f}MB CUDA memory{Style.RESET_ALL}")

    def _setup_swa(self, model: Any) -> None:
        """Setup SWA (Stochastic Weight Averaging) if enabled."""
        swa_enabled = self.config.get("swa_enabled", False)
        if not swa_enabled:
            return

        swa_start_epoch = self.config.get("swa_start_epoch", 10)
        swa_lr = self.config.get("swa_lr", 0.0005)

        print(f"{Fore.CYAN}SWA enabled:{Style.RESET_ALL}")
        print(f"  Start epoch: epochs - {swa_start_epoch}")
        print(f"  SWA LR: {swa_lr}")

        # Create SWA callback
        self.swa_callback = create_swa_callback(
            enabled=True,
            swa_start_epoch=swa_start_epoch,
            swa_lr=swa_lr,
            update_bn=True,  # Always update BN statistics
        )

        # Register callbacks with model
        if self.swa_callback is not None:
            register_swa_callbacks(model, self.swa_callback)

    def _generate_dynamic_synthetic(self, dataset_path: Path) -> int:
        """Generate dynamic Copy-Paste synthetic images before training."""
        if not self.config.get('dynamic_synthetic_enabled', False):
            return 0

        backgrounds_dir = self.config.get('backgrounds_dir')
        annotated_dir = self.config.get('annotated_dir')

        if not backgrounds_dir or not annotated_dir:
            print(f"{Fore.YELLOW}Dynamic synthetic: backgrounds_dir or annotated_dir not specified{Style.RESET_ALL}")
            return 0

        backgrounds_dir = Path(backgrounds_dir)
        annotated_dir = Path(annotated_dir)

        if not backgrounds_dir.exists() or not annotated_dir.exists():
            print(f"{Fore.YELLOW}Dynamic synthetic: directories do not exist{Style.RESET_ALL}")
            return 0

        print(f"{Fore.CYAN}Generating dynamic synthetic images...{Style.RESET_ALL}")

        # CopyPasteConfig construction
        cp_config = CopyPasteConfig(
            synthetic_to_real_ratio=self.config.get('synthetic_ratio', 2.0),
            scale_range=self.config.get('synthetic_scale_range', (0.5, 1.5)),
            rotation_range=self.config.get('synthetic_rotation_range', (-15.0, 15.0)),
            enable_white_balance=self.config.get('synthetic_white_balance', True),
            white_balance_strength=self.config.get('synthetic_white_balance_strength', 0.7),
            edge_blur_sigma=self.config.get('synthetic_edge_blur', 2.0),
            max_objects_per_image=self.config.get('synthetic_max_objects', 3),
            seed=int(time.time()),
        )

        augmentor = CopyPasteAugmentor(cp_config)

        # Count training images
        train_images_dir = dataset_path / "images" / "train"
        real_count = len(list(train_images_dir.glob("*"))) if train_images_dir.exists() else 0

        if real_count == 0:
            print(f"{Fore.YELLOW}No training images found for synthetic generation{Style.RESET_ALL}")
            return 0

        # Generate synthetic images
        output_dir = dataset_path / "synthetic_dynamic"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load class names from dataset YAML
            import yaml
            yaml_path = dataset_path / "data.yaml"
            if not yaml_path.exists():
                print(f"{Fore.YELLOW}Dataset YAML not found: {yaml_path}{Style.RESET_ALL}")
                return 0

            with open(yaml_path, "r") as f:
                data_config = yaml.safe_load(f)

            class_names = list(data_config.get("names", {}).values())
            if not class_names:
                print(f"{Fore.YELLOW}No class names found in dataset YAML{Style.RESET_ALL}")
                return 0

            stats = augmentor.generate_batch(
                backgrounds_dir=backgrounds_dir,
                annotated_dir=annotated_dir,
                output_dir=output_dir,
                real_image_count=real_count,
                class_names=class_names,
            )

            if "error" in stats:
                print(f"{Fore.RED}Synthetic generation error: {stats['error']}{Style.RESET_ALL}")
                return 0

            # Merge generated images into training set
            added = self._merge_synthetic_to_train(output_dir, dataset_path)
            print(f"{Fore.GREEN}Added {added} dynamic synthetic images to training set{Style.RESET_ALL}")
            return added

        except Exception as e:
            print(f"{Fore.RED}Dynamic synthetic generation failed: {e}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return 0

    def _merge_synthetic_to_train(self, synthetic_dir: Path, dataset_path: Path) -> int:
        """Merge generated synthetic images into training set."""
        import shutil

        train_images_dir = dataset_path / "images" / "train"
        train_labels_dir = dataset_path / "labels" / "train"

        train_images_dir.mkdir(parents=True, exist_ok=True)
        train_labels_dir.mkdir(parents=True, exist_ok=True)

        images_dir = synthetic_dir / "images"
        labels_dir = synthetic_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            return 0

        added = 0
        images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            # Unique filename for saving
            new_name = f"dynamic_synth_{image_path.stem}"
            image_dest = train_images_dir / f"{new_name}{image_path.suffix}"
            label_dest = train_labels_dir / f"{new_name}.txt"

            if not image_dest.exists():
                shutil.copy2(image_path, image_dest)
                shutil.copy2(label_path, label_dest)
                added += 1

        return added

    def get_tensorboard_url(self) -> str:
        """Get TensorBoard URL if available."""
        return self.tensorboard_url

    def _train_with_llrd(
        self,
        dataset_yaml: str,
        resume: bool = False,
        verbose: bool = True,
    ) -> Any:
        """
        Execute training using LLRD (Layer-wise Learning Rate Decay).

        Args:
            dataset_yaml: Path to dataset configuration YAML
            resume: Resume from checkpoint if available
            verbose: Enable verbose output

        Returns:
            Training results from LLRDDetectionTrainer
        """
        from ultralytics.cfg import DEFAULT_CFG_DICT

        print(f"\n{Fore.CYAN}LLRD Training Mode{Style.RESET_ALL}")
        print(f"  Decay rate: {self.config.get('llrd_decay_rate', 0.9)}")

        # Prepare LLRD config
        llrd_config = LLRDConfig(
            enabled=True,
            decay_rate=self.config.get("llrd_decay_rate", 0.9),
        )

        # Build overrides for LLRDDetectionTrainer
        # Remove LLRD-specific keys and synthetic keys from config
        excluded_keys = {"llrd_enabled", "llrd_decay_rate"} | SYNTHETIC_CONFIG_KEYS
        trainer_config = {k: v for k, v in self.config.items()
                         if k not in excluded_keys}

        overrides = {
            "data": dataset_yaml,
            "project": str(self.output_dir),
            "name": self.run_name,
            "model": self.base_model,
            "resume": resume,
            "verbose": verbose,
            **trainer_config,
        }

        # Create and run LLRD trainer
        trainer = LLRDDetectionTrainer(
            cfg=DEFAULT_CFG_DICT,
            overrides=overrides,
            llrd_config=llrd_config,
        )

        # Note: TensorBoard/SWA callbacks need to be registered differently for custom trainer
        # The built-in Ultralytics logging will still work

        results = trainer.train()
        return results

    def train(
        self,
        dataset_yaml: str,
        resume: bool = False,
        verbose: bool = True,
        enable_oom_recovery: bool = True,
    ) -> TrainingResult:
        """
        Execute fine-tuning.

        Args:
            dataset_yaml: Path to dataset configuration YAML
            resume: Resume from checkpoint if available
            verbose: Enable verbose output
            enable_oom_recovery: Enable automatic OOM recovery

        Returns:
            TrainingResult with metrics and paths
        """
        from ultralytics import YOLO

        # Validate dataset
        if not self._validate_dataset(dataset_yaml):
            raise ValueError("Dataset validation failed")

        # Prepare output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate dynamic Copy-Paste synthetic images before training
        dataset_path = Path(dataset_yaml).parent
        synthetic_added = self._generate_dynamic_synthetic(dataset_path)
        if synthetic_added > 0:
            print(f"{Fore.GREEN}Generated {synthetic_added} dynamic synthetic images{Style.RESET_ALL}")

        # Check if LLRD is enabled
        use_llrd = self.config.get("llrd_enabled", False)

        # Use try-finally to ensure cleanup happens
        try:
            # Start training
            print(f"\n{Fore.GREEN}Starting training...{Style.RESET_ALL}")
            print(f"Run name: {self.run_name}")
            print(f"Output: {self.output_dir / self.run_name}")

            # Print configuration summary
            print(f"\nConfiguration:")
            print(f"  Model: {self.config.get('model', self.base_model)}")
            print(f"  Batch Size: {self.config.get('batch', 16)}")
            print(f"  Image Size: {self.config.get('imgsz', 640)}")
            print(f"  Epochs: {self.config.get('epochs', 50)}")
            print(f"  LLRD: {'Enabled' if use_llrd else 'Disabled'}")

            start_time = time.time()
            results = None

            if use_llrd:
                # LLRD training path - use custom LLRDDetectionTrainer
                results = self._train_with_llrd(
                    dataset_yaml=dataset_yaml,
                    resume=resume,
                    verbose=verbose,
                )
            else:
                # Standard training path
                print(f"\nLoading base model: {self.base_model}")
                model = YOLO(self.base_model)

                # Setup TensorBoard
                self._setup_tensorboard(model)

                # Setup SWA (Stochastic Weight Averaging)
                self._setup_swa(model)

                if self.tensorboard_url:
                    print(f"TensorBoard: {self.tensorboard_url}")

                # Training with OOM recovery
                if enable_oom_recovery:
                    oom_recovery = OOMRecoveryStrategy(self.config)
                    current_config = self.config.copy()

                    while True:
                        try:
                            # Filter out synthetic keys before passing to YOLO
                            yolo_config = {k: v for k, v in current_config.items() if k not in SYNTHETIC_CONFIG_KEYS}
                            results = model.train(
                                data=dataset_yaml,
                                project=str(self.output_dir),
                                name=self.run_name,
                                resume=resume,
                                verbose=verbose,
                                **yolo_config,
                            )
                            break  # Success

                        except RuntimeError as e:
                            error_msg = str(e).lower()
                            if "out of memory" in error_msg or "oom" in error_msg:
                                print(f"\n{Fore.YELLOW}OOM detected. Attempting recovery...{Style.RESET_ALL}")

                                recovery_config = oom_recovery.get_recovery_config()
                                if recovery_config is None:
                                    print(f"{Fore.RED}Max OOM retries exceeded.{Style.RESET_ALL}")
                                    raise

                                print(f"Recovery attempt {oom_recovery.retry_count}: "
                                      f"{oom_recovery.get_changes_summary()}")
                                current_config = recovery_config

                                # Comprehensive cleanup before retry to prevent memory leak
                                self._cleanup_all_resources(
                                    model=model,
                                    trainer=None,
                                    stop_tensorboard=False  # Keep TensorBoard running
                                )
                                del model

                                # Reload model for retry
                                model = YOLO(self.base_model)
                                self._setup_tensorboard(model)
                                self._setup_swa(model)
                            elif "cuda" in error_msg:
                                # CUDA general error (not OOM) - fail immediately
                                print(f"{Fore.RED}CUDA error detected (not OOM): {e}{Style.RESET_ALL}")
                                raise
                            else:
                                raise
                else:
                    # Filter out synthetic keys before passing to YOLO
                    yolo_config = {k: v for k, v in self.config.items() if k not in SYNTHETIC_CONFIG_KEYS}
                    results = model.train(
                        data=dataset_yaml,
                        project=str(self.output_dir),
                        name=self.run_name,
                        resume=resume,
                        verbose=verbose,
                        **yolo_config,
                    )

            training_time = (time.time() - start_time) / 60  # minutes

            # Extract metrics
            metrics = self._extract_metrics(results)

            # Get model paths
            run_dir = self.output_dir / self.run_name
            best_path = run_dir / "weights" / "best.pt"
            last_path = run_dir / "weights" / "last.pt"

            result = TrainingResult(
                best_model_path=str(best_path),
                last_model_path=str(last_path),
                metrics=metrics,
                training_time_minutes=training_time,
                epochs_completed=self.config.get("epochs", 0),
                config=self.config.copy(),
            )

            # Save result
            result_path = run_dir / "training_result.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Add TensorBoard URL to result
            if self.tensorboard_url:
                result_data = result.to_dict()
                result_data["tensorboard_url"] = self.tensorboard_url
                with open(result_path, "w") as f:
                    json.dump(result_data, f, indent=2)

            # Cleanup
            self._cleanup_tensorboard()

            return result

        finally:
            # Guaranteed cleanup on success or failure
            # This ensures resources are freed even if an exception occurs
            log_memory_snapshot("Before final cleanup")

    def _extract_metrics(self, results) -> Dict:
        """Extract relevant metrics from training results."""
        try:
            # Access results box metrics
            box = results.results_dict
            return {
                "mAP50": box.get("metrics/mAP50(B)", 0),
                "mAP50-95": box.get("metrics/mAP50-95(B)", 0),
                "precision": box.get("metrics/precision(B)", 0),
                "recall": box.get("metrics/recall(B)", 0),
            }
        except Exception as e:
            print(f"Warning: Could not extract metrics: {e}")
            return {}

    def validate(self, model_path: str, dataset_yaml: str) -> Dict:
        """
        Run validation on a trained model.

        Args:
            model_path: Path to model weights
            dataset_yaml: Path to dataset configuration

        Returns:
            Dictionary of validation metrics
        """
        from ultralytics import YOLO

        print(f"\nValidating model: {model_path}")
        model = YOLO(model_path)
        results = model.val(data=dataset_yaml)

        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

    def export(
        self,
        model_path: str,
        format: str = "onnx",
        simplify: bool = True,
    ) -> str:
        """
        Export model for deployment.

        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, etc.)
            simplify: Simplify ONNX model

        Returns:
            Path to exported model
        """
        from ultralytics import YOLO

        print(f"\nExporting model: {model_path}")
        print(f"Format: {format}")

        model = YOLO(model_path)
        export_path = model.export(format=format, simplify=simplify)

        print(f"Exported to: {export_path}")
        return export_path


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description="Competition-day YOLOv8 fine-tuning with GPU auto-scaling and TensorBoard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard competition training (auto-scales to GPU)
  python quick_finetune.py --dataset datasets/competition_day/data.yaml

  # Fast training (smaller model, fewer epochs)
  python quick_finetune.py --dataset datasets/competition_day/data.yaml --fast

  # Disable auto-scaling and use specific model
  python quick_finetune.py --dataset data.yaml --model yolov8l.pt --no-auto-scale

  # Disable TensorBoard
  python quick_finetune.py --dataset data.yaml --no-tensorboard

  # Resume interrupted training
  python quick_finetune.py --dataset datasets/competition_day/data.yaml --resume

  # Force specific GPU tier settings
  python quick_finetune.py --dataset data.yaml --gpu-tier medium

  # Enable LLRD for better fine-tuning (+1-3% mAP on small datasets)
  python quick_finetune.py --dataset data.yaml --llrd

  # LLRD with custom decay rate
  python quick_finetune.py --dataset data.yaml --llrd --llrd-decay-rate 0.85
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to dataset YAML configuration",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Base model path or name (auto-detected by default)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="models/finetuned",
        help="Output directory (default: models/finetuned)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast training config (smaller model, fewer epochs)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        help="Override batch size",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing model",
    )
    parser.add_argument(
        "--export",
        choices=["onnx", "torchscript", "tflite"],
        help="Export model after training",
    )

    # GPU scaling options
    parser.add_argument(
        "--no-auto-scale",
        action="store_true",
        help="Disable GPU auto-scaling",
    )
    parser.add_argument(
        "--gpu-tier",
        choices=["low", "medium", "high", "workstation"],
        help="Force specific GPU tier settings",
    )

    # TensorBoard options
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard monitoring",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="TensorBoard server port (default: 6006)",
    )

    # OOM recovery
    parser.add_argument(
        "--no-oom-recovery",
        action="store_true",
        help="Disable automatic OOM recovery",
    )

    # LLRD options
    parser.add_argument(
        "--llrd",
        action="store_true",
        help="Enable Layer-wise Learning Rate Decay (+1-3%% mAP on small datasets)",
    )
    parser.add_argument(
        "--llrd-decay-rate",
        type=float,
        default=0.9,
        help="LLRD decay rate per layer (default: 0.9)",
    )

    # Dynamic Copy-Paste options
    parser.add_argument(
        "--dynamic-synthetic",
        action="store_true",
        help="Enable dynamic Copy-Paste synthetic image generation (default: enabled)",
    )
    parser.add_argument(
        "--no-dynamic-synthetic",
        action="store_true",
        help="Disable dynamic Copy-Paste synthetic image generation",
    )
    parser.add_argument(
        "--backgrounds-dir",
        type=str,
        help="Directory containing background images for Copy-Paste",
    )
    parser.add_argument(
        "--annotated-dir",
        type=str,
        help="Directory containing annotated images for Copy-Paste",
    )
    parser.add_argument(
        "--synthetic-ratio",
        type=float,
        default=2.0,
        help="Synthetic to real image ratio (default: 2.0)",
    )

    args = parser.parse_args()

    # Determine configuration
    auto_scale = not args.no_auto_scale
    config = None

    if args.fast:
        config = FAST_CONFIG.copy()
        auto_scale = False  # Fast mode uses fixed config
    elif args.gpu_tier:
        # Use specific tier config
        from .gpu_scaler import GPUTier, TIER_CONFIGS
        tier = GPUTier(args.gpu_tier)
        config = TIER_CONFIGS[tier].copy()
        auto_scale = False

    # Apply overrides
    if config is None and not auto_scale:
        config = COMPETITION_CONFIG.copy()

    if config and args.epochs:
        config["epochs"] = args.epochs
    if config and args.batch:
        config["batch"] = args.batch

    # Apply LLRD settings
    if args.llrd:
        if config is None:
            config = COMPETITION_CONFIG.copy()
        config["llrd_enabled"] = True
        config["llrd_decay_rate"] = args.llrd_decay_rate

    # Apply Copy-Paste settings
    if config is None:
        config = COMPETITION_CONFIG.copy()

    # Dynamic synthetic is enabled by default, can be disabled with --no-dynamic-synthetic
    if args.no_dynamic_synthetic:
        config["dynamic_synthetic_enabled"] = False
    elif args.dynamic_synthetic or args.backgrounds_dir or args.annotated_dir:
        config["dynamic_synthetic_enabled"] = True

    if args.backgrounds_dir:
        config["backgrounds_dir"] = args.backgrounds_dir
    if args.annotated_dir:
        config["annotated_dir"] = args.annotated_dir
    if args.synthetic_ratio:
        config["synthetic_ratio"] = args.synthetic_ratio

    # Create trainer
    trainer = CompetitionTrainer(
        base_model=args.model,
        output_dir=args.output,
        config=config,
        auto_scale=auto_scale,
        tensorboard=not args.no_tensorboard,
        tensorboard_port=args.tensorboard_port,
    )

    try:
        if args.validate_only:
            # Validation only mode
            if args.model is None:
                print(f"{Fore.RED}Error: --model is required for --validate-only{Style.RESET_ALL}")
                sys.exit(1)
            metrics = trainer.validate(args.model, args.dataset)
            print("\nValidation Results:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        else:
            # Full training
            result = trainer.train(
                dataset_yaml=args.dataset,
                resume=args.resume,
                enable_oom_recovery=not args.no_oom_recovery,
            )

            print(result.summary())

            # Show TensorBoard URL
            if trainer.tensorboard_url:
                print(f"\n{Fore.CYAN}TensorBoard: {trainer.tensorboard_url}{Style.RESET_ALL}")

            # Check target
            if result.meets_target():
                print(f"{Fore.GREEN}Target mAP (85%) achieved!{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Warning: Target mAP not reached. "
                      f"Consider adding more data or training longer.{Style.RESET_ALL}")

            # Export if requested
            if args.export:
                trainer.export(result.best_model_path, format=args.export)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Training interrupted by user.{Style.RESET_ALL}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Fore.RED}Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
