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
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from colorama import Fore, Style, init as colorama_init

colorama_init()

# Add scripts directory to path for common module imports
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from common.device_utils import log_gpu_status
from common.constants import TARGET_MAP50

# Import extracted modules - use try/except for both direct execution and module import
try:
    # When run as module: python -m scripts.training.quick_finetune
    from .config_manager import COMPETITION_CONFIG, FAST_CONFIG
    from .dataset_validator import DatasetValidator
    from .gpu_scaler import GPUScaler, GPUTier, OOMRecoveryStrategy, TIER_CONFIGS
    from .memory_utils import full_training_cleanup, log_memory_snapshot
    from .model_operations import ModelExporter, ModelValidator
    from .swa_trainer import SWACallback, create_swa_callback, register_swa_callbacks
    from .synthetic_data_manager import (
        SYNTHETIC_CONFIG_KEYS,
        SyntheticConfig,
        SyntheticDataManager,
    )
    from .tensorboard_monitor import (
        CompetitionTensorBoardCallback,
        TensorBoardServer,
        check_tensorboard_available,
        enable_ultralytics_tensorboard,
    )
    from .training_executor import TrainingExecutor, TrainingResult
except ImportError:
    # When run directly: python scripts/training/quick_finetune.py
    from config_manager import COMPETITION_CONFIG, FAST_CONFIG
    from dataset_validator import DatasetValidator
    from gpu_scaler import GPUScaler, GPUTier, OOMRecoveryStrategy, TIER_CONFIGS
    from memory_utils import full_training_cleanup, log_memory_snapshot
    from model_operations import ModelExporter, ModelValidator
    from swa_trainer import SWACallback, create_swa_callback, register_swa_callbacks
    from synthetic_data_manager import (
        SYNTHETIC_CONFIG_KEYS,
        SyntheticConfig,
        SyntheticDataManager,
    )
    from tensorboard_monitor import (
        CompetitionTensorBoardCallback,
        TensorBoardServer,
        check_tensorboard_available,
        enable_ultralytics_tensorboard,
    )
    from training_executor import TrainingExecutor, TrainingResult


class CompetitionTrainer:
    """
    Competition-day YOLOv8 fine-tuning handler.

    This is a facade class that orchestrates the training pipeline using
    extracted modules:
    - DatasetValidator: Dataset validation
    - SyntheticDataManager: Copy-Paste augmentation
    - TrainingExecutor: Core training logic
    - ModelValidator/ModelExporter: Model operations
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

        # Initialize extracted module instances
        self._dataset_validator = DatasetValidator()
        self._model_validator = ModelValidator()
        self._model_exporter = ModelExporter()

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
        """Validate dataset configuration using DatasetValidator."""
        result = self._dataset_validator.validate(dataset_yaml)
        return result.is_valid

    def _setup_tensorboard(self, model: Any) -> None:
        """Setup TensorBoard monitoring with comprehensive error handling."""
        if not self.tensorboard_enabled:
            return

        if not check_tensorboard_available():
            print(
                f"{Fore.YELLOW}Warning: TensorBoard not available. "
                f"Install with: pip install tensorboard{Style.RESET_ALL}"
            )
            self.tensorboard_enabled = False
            return

        try:
            # Enable Ultralytics built-in TensorBoard
            enable_ultralytics_tensorboard()

            # Create log directory
            log_dir = self.output_dir / self.run_name / "tensorboard"

            # Create custom callback
            self.tensorboard_callback = CompetitionTensorBoardCallback(
                log_dir=str(log_dir),
                target_map50=TARGET_MAP50,
            )

            # Register callbacks with individual error handling
            callback_mappings = [
                ("on_pretrain_routine_start", self.tensorboard_callback.on_pretrain_routine_start),
                ("on_train_epoch_start", self.tensorboard_callback.on_train_epoch_start),
                ("on_train_epoch_end", self.tensorboard_callback.on_train_epoch_end),
                ("on_fit_epoch_end", self.tensorboard_callback.on_fit_epoch_end),
                ("on_train_end", self.tensorboard_callback.on_train_end),
            ]

            for event_name, callback_fn in callback_mappings:
                try:
                    model.add_callback(event_name, callback_fn)
                except Exception as cb_error:
                    print(f"{Fore.YELLOW}Warning: Failed to register callback {event_name}: {cb_error}{Style.RESET_ALL}")

            # Start TensorBoard server
            self.tensorboard_server = TensorBoardServer(
                str(log_dir),
                port=self.tensorboard_port,
            )
            self.tensorboard_url = self.tensorboard_server.start()

        except ImportError as e:
            print(f"{Fore.YELLOW}Warning: TensorBoard import failed: {e}{Style.RESET_ALL}")
            self.tensorboard_enabled = False
        except OSError as e:
            print(f"{Fore.YELLOW}Warning: TensorBoard server startup failed: {e}{Style.RESET_ALL}")
            self.tensorboard_enabled = False
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Unexpected error setting up TensorBoard: {e}{Style.RESET_ALL}")
            self.tensorboard_enabled = False

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
        """Generate dynamic Copy-Paste synthetic images using SyntheticDataManager."""
        synthetic_config = SyntheticConfig.from_dict(self.config)
        manager = SyntheticDataManager(synthetic_config, verbose=True)
        result = manager.generate(dataset_path)
        return result.images_added

    def get_tensorboard_url(self) -> str:
        """Get TensorBoard URL if available."""
        return self.tensorboard_url

    def _train_with_llrd(
        self,
        dataset_yaml: str,
        resume: bool = False,
        verbose: bool = True,
    ) -> Any:
        """Execute training using LLRD via TrainingExecutor."""
        executor = TrainingExecutor(
            base_model=self.base_model,
            output_dir=self.output_dir,
            run_name=self.run_name,
            config=self.config,
            verbose=verbose,
        )
        return executor.execute_llrd(dataset_yaml, resume=resume)

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
                    current_config = copy.deepcopy(self.config)

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
                                try:
                                    model = YOLO(self.base_model)
                                    self._setup_tensorboard(model)
                                    self._setup_swa(model)
                                except Exception as reload_error:
                                    print(f"{Fore.RED}Failed to reload model: {reload_error}{Style.RESET_ALL}")
                                    raise RuntimeError(
                                        f"Model reload failed during OOM recovery: {reload_error}"
                                    ) from reload_error
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

            # Save result (single write operation)
            result_data = result.to_dict()
            if self.tensorboard_url:
                result_data["tensorboard_url"] = self.tensorboard_url
            result_path = run_dir / "training_result.json"
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)

            # Cleanup
            self._cleanup_tensorboard()

            return result

        finally:
            # Guaranteed cleanup on success or failure
            # This ensures resources are freed even if an exception occurs
            log_memory_snapshot("Before final cleanup")
            try:
                # Only cleanup if model exists (standard training path, not LLRD)
                if 'model' in dir() and model is not None:
                    self._cleanup_all_resources(
                        model=model,
                        trainer=None,
                        stop_tensorboard=False  # Keep TensorBoard for result viewing
                    )
            except Exception as cleanup_error:
                print(f"{Fore.YELLOW}Warning: Error during final cleanup: {cleanup_error}{Style.RESET_ALL}")
            log_memory_snapshot("After final cleanup")

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
        """Run validation using ModelValidator."""
        return self._model_validator.validate(model_path, dataset_yaml)

    def export(
        self,
        model_path: str,
        format: str = "onnx",
        simplify: bool = True,
    ) -> str:
        """Export model using ModelExporter."""
        return self._model_exporter.export(model_path, format, simplify)


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
