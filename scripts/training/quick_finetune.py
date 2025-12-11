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

    def get_tensorboard_url(self) -> str:
        """Get TensorBoard URL if available."""
        return self.tensorboard_url

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

        # Load model
        print(f"\nLoading base model: {self.base_model}")
        model = YOLO(self.base_model)

        # Prepare output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup TensorBoard
        self._setup_tensorboard(model)

        # Start training
        print(f"\n{Fore.GREEN}Starting training...{Style.RESET_ALL}")
        print(f"Run name: {self.run_name}")
        print(f"Output: {self.output_dir / self.run_name}")
        if self.tensorboard_url:
            print(f"TensorBoard: {self.tensorboard_url}")

        # Print configuration summary
        print(f"\nConfiguration:")
        print(f"  Model: {self.config.get('model', self.base_model)}")
        print(f"  Batch Size: {self.config.get('batch', 16)}")
        print(f"  Image Size: {self.config.get('imgsz', 640)}")
        print(f"  Epochs: {self.config.get('epochs', 50)}")

        start_time = time.time()
        results = None

        # Training with OOM recovery
        if enable_oom_recovery:
            oom_recovery = OOMRecoveryStrategy(self.config)
            current_config = self.config.copy()

            while True:
                try:
                    results = model.train(
                        data=dataset_yaml,
                        project=str(self.output_dir),
                        name=self.run_name,
                        resume=resume,
                        verbose=verbose,
                        **current_config,
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

                        # Cleanup old model before retry to prevent memory leak
                        import gc
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()

                        # Reload model for retry
                        model = YOLO(self.base_model)
                        self._setup_tensorboard(model)
                    elif "cuda" in error_msg:
                        # CUDA general error (not OOM) - fail immediately
                        print(f"{Fore.RED}CUDA error detected (not OOM): {e}{Style.RESET_ALL}")
                        raise
                    else:
                        raise
        else:
            results = model.train(
                data=dataset_yaml,
                project=str(self.output_dir),
                name=self.run_name,
                resume=resume,
                verbose=verbose,
                **self.config,
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
