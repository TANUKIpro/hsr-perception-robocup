#!/usr/bin/env python3
"""
Training Task Runner

Subprocess wrapper for running YOLOv8 fine-tuning.
Updates task status file for progress tracking with epoch-by-epoch updates.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "training"))

from app.services.task_manager import update_task_status


class TrainingProgressCallback:
    """Callback for updating task progress during training."""

    def __init__(self, task_id: str, total_epochs: int):
        self.task_id = task_id
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        self.current_epoch = trainer.epoch + 1
        progress = 0.1 + (self.current_epoch / self.total_epochs) * 0.8  # 10%-90%

        # Extract metrics if available
        metrics = {}
        try:
            if hasattr(trainer, "metrics"):
                metrics["mAP50"] = trainer.metrics.get("metrics/mAP50(B)", 0)
                metrics["mAP50-95"] = trainer.metrics.get("metrics/mAP50-95(B)", 0)
            if hasattr(trainer, "loss"):
                metrics["loss"] = float(trainer.loss) if trainer.loss else None
        except Exception:
            pass

        update_task_status(
            self.task_id,
            progress=progress,
            current_step=f"Training epoch {self.current_epoch}/{self.total_epochs}",
            extra_data={"epoch": self.current_epoch, "metrics": metrics}
        )

    def on_train_start(self, trainer):
        """Called when training starts."""
        update_task_status(
            self.task_id,
            progress=0.1,
            current_step="Training started..."
        )

    def on_train_end(self, trainer):
        """Called when training ends."""
        update_task_status(
            self.task_id,
            progress=0.9,
            current_step="Saving model..."
        )


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 training")
    parser.add_argument("--task-id", required=True, help="Task ID for status updates")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML")
    parser.add_argument("--model", default="yolov8m.pt", help="Base model path or name")
    parser.add_argument("--output", default="models/finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--fast", action="store_true", help="Use fast training config")
    args = parser.parse_args()

    task_id = args.task_id

    try:
        update_task_status(
            task_id,
            progress=0.02,
            current_step="Checking GPU availability..."
        )

        # Check GPU
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_info = ""
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_info = f"{gpu_name} ({gpu_memory:.1f} GB)"
            print(f"GPU Available: {gpu_info}")
        else:
            print("Warning: CUDA not available. Training will be slow.")

        update_task_status(
            task_id,
            progress=0.05,
            current_step="Loading training modules...",
            extra_data={"gpu_available": gpu_available, "gpu_info": gpu_info}
        )

        # Import training modules
        from quick_finetune import CompetitionTrainer, COMPETITION_CONFIG, FAST_CONFIG

        # Select config
        config = FAST_CONFIG.copy() if args.fast else COMPETITION_CONFIG.copy()
        config["epochs"] = args.epochs
        config["batch"] = args.batch

        update_task_status(
            task_id,
            progress=0.08,
            current_step="Initializing trainer..."
        )

        # Create trainer
        trainer = CompetitionTrainer(
            base_model=args.model,
            output_dir=args.output,
            config=config,
        )

        update_task_status(
            task_id,
            progress=0.1,
            current_step="Loading model and starting training..."
        )

        # Load YOLO model and add callbacks
        from ultralytics import YOLO
        model = YOLO(args.model)

        # Create and register progress callback
        progress_callback = TrainingProgressCallback(task_id, args.epochs)
        model.add_callback("on_train_start", progress_callback.on_train_start)
        model.add_callback("on_train_epoch_end", progress_callback.on_train_epoch_end)
        model.add_callback("on_train_end", progress_callback.on_train_end)

        # Validate dataset
        if not trainer._validate_dataset(args.dataset):
            raise ValueError("Dataset validation failed")

        # Start training
        import time
        start_time = time.time()

        results = model.train(
            data=args.dataset,
            project=str(trainer.output_dir),
            name=trainer.run_name,
            verbose=True,
            **config,
        )

        training_time = (time.time() - start_time) / 60  # minutes

        # Extract final metrics
        try:
            metrics = {
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
            }
        except Exception:
            metrics = {}

        # Get model paths
        run_dir = trainer.output_dir / trainer.run_name
        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"

        result_path = str(best_path) if best_path.exists() else str(last_path)

        update_task_status(
            task_id,
            progress=1.0,
            current_step="Completed",
            status="completed",
            result_path=result_path,
            extra_data={
                "training_time_minutes": training_time,
                "epochs_completed": args.epochs,
                "metrics": metrics,
                "best_model": str(best_path) if best_path.exists() else None,
                "last_model": str(last_path) if last_path.exists() else None,
                "run_dir": str(run_dir),
            }
        )

        print(f"\nTraining completed!")
        print(f"Training time: {training_time:.1f} minutes")
        print(f"Best model: {best_path}")
        if metrics:
            print(f"Final mAP50: {metrics.get('mAP50', 'N/A'):.4f}")

    except KeyboardInterrupt:
        update_task_status(
            task_id,
            progress=0.0,
            current_step="Cancelled",
            status="cancelled",
            error_message="Training cancelled by user"
        )
        print("Training cancelled by user")
        sys.exit(1)

    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback_str = traceback.format_exc()

        update_task_status(
            task_id,
            progress=0.0,
            current_step="Failed",
            status="failed",
            error_message=error_msg,
            extra_data={"traceback": traceback_str}
        )

        print(f"Error: {error_msg}", file=sys.stderr)
        print(traceback_str, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
