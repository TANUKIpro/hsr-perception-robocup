"""
Training execution logic for YOLOv8 fine-tuning.

Provides the core training execution with OOM recovery, LLRD support,
metrics extraction, and result management.
"""
import copy
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from colorama import Fore, Style

from .gpu_scaler import OOMRecoveryStrategy
from .memory_utils import log_memory_snapshot
from .synthetic_data_manager import SYNTHETIC_CONFIG_KEYS


# Maximum OOM retry attempts
MAX_OOM_RETRIES = 3


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


class TrainingExecutor:
    """
    Handles core training execution logic.

    Supports:
    - Standard YOLOv8 training
    - LLRD (Layer-wise Learning Rate Decay) training
    - OOM recovery with automatic parameter adjustment
    - Result extraction and persistence
    """

    def __init__(
        self,
        base_model: str,
        output_dir: Path,
        run_name: str,
        config: Dict,
        verbose: bool = True,
    ):
        """
        Initialize executor.

        Args:
            base_model: Path to base model weights
            output_dir: Directory for training outputs
            run_name: Unique run name for this training session
            config: Training configuration dictionary
            verbose: Whether to print progress messages
        """
        self.base_model = base_model
        self.output_dir = output_dir
        self.run_name = run_name
        self.config = config
        self.verbose = verbose

    def execute(
        self,
        dataset_yaml: str,
        model: Any,
        resume: bool = False,
        enable_oom_recovery: bool = True,
        setup_callbacks: Optional[Callable[[Any], None]] = None,
        cleanup_resources: Optional[Callable[[Any], None]] = None,
    ) -> TrainingResult:
        """
        Execute training with standard YOLO model.

        Args:
            dataset_yaml: Path to dataset configuration YAML
            model: YOLO model instance
            resume: Resume from checkpoint if available
            enable_oom_recovery: Enable automatic OOM recovery
            setup_callbacks: Optional callback setup function
            cleanup_resources: Optional resource cleanup function

        Returns:
            TrainingResult with metrics and paths
        """
        start_time = time.time()

        if self.verbose:
            print(f"\n{Fore.GREEN}Starting training...{Style.RESET_ALL}")
            print(f"Run name: {self.run_name}")
            print(f"Output: {self.output_dir / self.run_name}")
            self._print_config_summary()

        results = None

        try:
            if enable_oom_recovery:
                results = self._run_with_oom_recovery(
                    model=model,
                    dataset_yaml=dataset_yaml,
                    resume=resume,
                    setup_callbacks=setup_callbacks,
                    cleanup_resources=cleanup_resources,
                )
            else:
                # Direct training without OOM recovery
                yolo_config = self._filter_yolo_config(self.config)
                results = model.train(
                    data=dataset_yaml,
                    project=str(self.output_dir),
                    name=self.run_name,
                    resume=resume,
                    verbose=self.verbose,
                    **yolo_config,
                )

            training_time = (time.time() - start_time) / 60  # minutes

            # Build and return result
            return self._build_result(results, training_time)

        finally:
            log_memory_snapshot("After training execution")

    def execute_llrd(
        self,
        dataset_yaml: str,
        resume: bool = False,
    ) -> Any:
        """
        Execute training using LLRD (Layer-wise Learning Rate Decay).

        Args:
            dataset_yaml: Path to dataset configuration YAML
            resume: Resume from checkpoint if available

        Returns:
            Training results from LLRDDetectionTrainer
        """
        from ultralytics.cfg import DEFAULT_CFG_DICT

        from .llrd_trainer import LLRDConfig, LLRDDetectionTrainer

        if self.verbose:
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
        trainer_config = {
            k: v for k, v in self.config.items() if k not in excluded_keys
        }

        overrides = {
            "data": dataset_yaml,
            "project": str(self.output_dir),
            "name": self.run_name,
            "model": self.base_model,
            "resume": resume,
            "verbose": self.verbose,
            **trainer_config,
        }

        # Create and run LLRD trainer
        trainer = LLRDDetectionTrainer(
            cfg=DEFAULT_CFG_DICT,
            overrides=overrides,
            llrd_config=llrd_config,
        )

        results = trainer.train()
        return results

    def _run_with_oom_recovery(
        self,
        model: Any,
        dataset_yaml: str,
        resume: bool,
        setup_callbacks: Optional[Callable[[Any], None]] = None,
        cleanup_resources: Optional[Callable[[Any], None]] = None,
    ) -> Any:
        """
        Run training with OOM recovery.

        Args:
            model: YOLO model instance
            dataset_yaml: Path to dataset configuration YAML
            resume: Resume from checkpoint
            setup_callbacks: Optional callback setup function
            cleanup_resources: Optional resource cleanup function

        Returns:
            Training results
        """
        from ultralytics import YOLO

        oom_recovery = OOMRecoveryStrategy(self.config)
        current_config = copy.deepcopy(self.config)
        current_model = model

        while True:
            try:
                yolo_config = self._filter_yolo_config(current_config)
                results = current_model.train(
                    data=dataset_yaml,
                    project=str(self.output_dir),
                    name=self.run_name,
                    resume=resume,
                    verbose=self.verbose,
                    **yolo_config,
                )
                return results  # Success

            except RuntimeError as e:
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "oom" in error_msg:
                    if self.verbose:
                        print(
                            f"\n{Fore.YELLOW}OOM detected. "
                            f"Attempting recovery...{Style.RESET_ALL}"
                        )

                    recovery_config = oom_recovery.get_recovery_config()
                    if recovery_config is None:
                        if self.verbose:
                            print(
                                f"{Fore.RED}Max OOM retries exceeded.{Style.RESET_ALL}"
                            )
                        raise

                    if self.verbose:
                        print(
                            f"Recovery attempt {oom_recovery.retry_count}: "
                            f"{oom_recovery.get_changes_summary()}"
                        )
                    current_config = recovery_config

                    # Cleanup before retry
                    if cleanup_resources:
                        cleanup_resources(current_model)
                    del current_model

                    # Reload model for retry
                    try:
                        current_model = YOLO(self.base_model)
                        if setup_callbacks:
                            setup_callbacks(current_model)
                    except Exception as reload_error:
                        if self.verbose:
                            print(
                                f"{Fore.RED}Failed to reload model: "
                                f"{reload_error}{Style.RESET_ALL}"
                            )
                        raise RuntimeError(
                            f"Model reload failed during OOM recovery: {reload_error}"
                        ) from reload_error

                elif "cuda" in error_msg:
                    # CUDA general error (not OOM) - fail immediately
                    if self.verbose:
                        print(
                            f"{Fore.RED}CUDA error detected (not OOM): "
                            f"{e}{Style.RESET_ALL}"
                        )
                    raise
                else:
                    raise

    def _filter_yolo_config(self, config: Dict) -> Dict:
        """Filter out non-YOLO keys from config."""
        excluded_keys = {"llrd_enabled", "llrd_decay_rate"} | SYNTHETIC_CONFIG_KEYS
        return {k: v for k, v in config.items() if k not in excluded_keys}

    def _print_config_summary(self) -> None:
        """Print training configuration summary."""
        use_llrd = self.config.get("llrd_enabled", False)
        print(f"\nConfiguration:")
        print(f"  Model: {self.config.get('model', self.base_model)}")
        print(f"  Batch Size: {self.config.get('batch', 16)}")
        print(f"  Image Size: {self.config.get('imgsz', 640)}")
        print(f"  Epochs: {self.config.get('epochs', 50)}")
        print(f"  LLRD: {'Enabled' if use_llrd else 'Disabled'}")

    def _build_result(self, results: Any, training_time: float) -> TrainingResult:
        """
        Build TrainingResult from training results.

        Args:
            results: Training results from YOLO
            training_time: Training time in minutes

        Returns:
            TrainingResult instance
        """
        metrics = self._extract_metrics(results)

        run_dir = self.output_dir / self.run_name
        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"

        return TrainingResult(
            best_model_path=str(best_path),
            last_model_path=str(last_path),
            metrics=metrics,
            training_time_minutes=training_time,
            epochs_completed=self.config.get("epochs", 0),
            config=self.config.copy(),
        )

    def _extract_metrics(self, results: Any) -> Dict:
        """Extract relevant metrics from training results."""
        try:
            box = results.results_dict
            return {
                "mAP50": box.get("metrics/mAP50(B)", 0),
                "mAP50-95": box.get("metrics/mAP50-95(B)", 0),
                "precision": box.get("metrics/precision(B)", 0),
                "recall": box.get("metrics/recall(B)", 0),
            }
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not extract metrics: {e}")
            return {}

    def save_result(
        self,
        result: TrainingResult,
        tensorboard_url: str = "",
    ) -> Path:
        """
        Save training result to JSON file.

        Args:
            result: TrainingResult to save
            tensorboard_url: Optional TensorBoard URL to include

        Returns:
            Path to saved result file
        """
        result_data = result.to_dict()
        if tensorboard_url:
            result_data["tensorboard_url"] = tensorboard_url

        run_dir = self.output_dir / self.run_name
        result_path = run_dir / "training_result.json"

        with open(result_path, "w") as f:
            json.dump(result_data, f, indent=2)

        return result_path
