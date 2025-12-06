#!/usr/bin/env python3
"""
Quick Fine-Tuning Script for Competition Day

Optimized YOLOv8m fine-tuning for rapid model adaptation during competitions.
Designed to complete training within ~45-60 minutes on a capable GPU.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from colorama import Fore, Style, init as colorama_init

colorama_init()

# Competition-optimized training configuration
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
    "weight_decay": 0.0005,
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
    "mosaic": 1.0,
    "mixup": 0.1,
    # Performance settings
    "workers": 8,
    "cache": True,  # Cache images in RAM for speed
    "amp": True,  # Automatic mixed precision
    "close_mosaic": 10,  # Disable mosaic for final N epochs
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

    Optimized for rapid training with early stopping and checkpoint management.
    """

    def __init__(
        self,
        base_model: str = "models/pretrained/yolov8m.pt",
        output_dir: str = "models/finetuned",
        config: Optional[Dict] = None,
    ):
        """
        Initialize trainer.

        Args:
            base_model: Path to pretrained model (or model name like 'yolov8m.pt')
            output_dir: Directory for training outputs
            config: Training configuration (uses COMPETITION_CONFIG if None)
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config or COMPETITION_CONFIG.copy()
        self.run_name = self._generate_run_name()

        # Verify CUDA availability
        self._check_cuda()

    def _check_cuda(self) -> None:
        """Check CUDA availability and log GPU info."""
        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"{Fore.GREEN}GPU Available: {gpu_name} ({gpu_memory:.1f} GB){Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Warning: CUDA not available. Training will be slow.{Style.RESET_ALL}")
        except ImportError:
            print(f"{Fore.RED}Warning: PyTorch not installed.{Style.RESET_ALL}")

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

    def train(
        self,
        dataset_yaml: str,
        resume: bool = False,
        verbose: bool = True,
    ) -> TrainingResult:
        """
        Execute fine-tuning.

        Args:
            dataset_yaml: Path to dataset configuration YAML
            resume: Resume from checkpoint if available
            verbose: Enable verbose output

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

        # Start training
        print(f"\n{Fore.GREEN}Starting training...{Style.RESET_ALL}")
        print(f"Run name: {self.run_name}")
        print(f"Output: {self.output_dir / self.run_name}")

        start_time = time.time()

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
        description="Competition-day YOLOv8 fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard competition training
  python quick_finetune.py --dataset datasets/competition_day/data.yaml

  # Fast training (smaller model, fewer epochs)
  python quick_finetune.py --dataset datasets/competition_day/data.yaml --fast

  # Resume interrupted training
  python quick_finetune.py --dataset datasets/competition_day/data.yaml --resume

  # Custom base model
  python quick_finetune.py --dataset data.yaml --model models/pretrained/yolov8m.pt
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
        default="yolov8m.pt",
        help="Base model path or name (default: yolov8m.pt)",
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

    args = parser.parse_args()

    # Select configuration
    config = FAST_CONFIG.copy() if args.fast else COMPETITION_CONFIG.copy()

    # Apply overrides
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch:
        config["batch"] = args.batch

    # Create trainer
    trainer = CompetitionTrainer(
        base_model=args.model,
        output_dir=args.output,
        config=config,
    )

    try:
        if args.validate_only:
            # Validation only mode
            metrics = trainer.validate(args.model, args.dataset)
            print("\nValidation Results:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")
        else:
            # Full training
            result = trainer.train(
                dataset_yaml=args.dataset,
                resume=args.resume,
            )

            print(result.summary())

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
        sys.exit(1)


if __name__ == "__main__":
    main()
