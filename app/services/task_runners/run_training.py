#!/usr/bin/env python3
"""
Training Task Runner

Subprocess wrapper for running YOLOv8 fine-tuning.
Features:
- GPU auto-scaling integration
- TensorBoard automatic launch
- Epoch-by-epoch progress tracking
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts" / "training"))
sys.path.insert(0, str(project_root / "scripts"))

from app.services.task_manager import update_task_status
from training.synthetic_data_manager import SYNTHETIC_CONFIG_KEYS


class TrainingProgressCallback:
    """Callback for updating task progress during training."""

    def __init__(self, task_id: str, total_epochs: int, tasks_dir: str = None):
        self.task_id = task_id
        self.total_epochs = total_epochs
        self.tasks_dir = tasks_dir
        self.current_epoch = 0
        self.training_history = []  # Accumulate history for charts

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
                metrics["precision"] = trainer.metrics.get("metrics/precision(B)", 0)
                metrics["recall"] = trainer.metrics.get("metrics/recall(B)", 0)
            if hasattr(trainer, "loss"):
                metrics["loss"] = float(trainer.loss) if trainer.loss else None
            # Extract loss components if available
            if hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                try:
                    loss_items = trainer.loss_items
                    if len(loss_items) >= 3:
                        metrics["box_loss"] = float(loss_items[0])
                        metrics["cls_loss"] = float(loss_items[1])
                        metrics["dfl_loss"] = float(loss_items[2])
                except Exception:
                    pass
        except Exception:
            pass

        # Accumulate history for charts
        history_entry = {
            "epoch": self.current_epoch,
            "mAP50": metrics.get("mAP50", 0),
            "mAP50-95": metrics.get("mAP50-95", 0),
            "loss": metrics.get("loss", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
        }
        self.training_history.append(history_entry)

        # Limit history size (to prevent JSON file bloat)
        MAX_HISTORY_SIZE = 50
        if len(self.training_history) > MAX_HISTORY_SIZE:
            # Keep the first entry + recent entries (for chart display)
            self.training_history = [self.training_history[0]] + self.training_history[-(MAX_HISTORY_SIZE-1):]

        update_task_status(
            self.task_id,
            progress=progress,
            current_step=f"Training epoch {self.current_epoch}/{self.total_epochs}",
            extra_data={
                "epoch": self.current_epoch,
                "metrics": metrics,
                "training_history": self.training_history,
            },
            tasks_dir=self.tasks_dir
        )

    def on_train_start(self, trainer):
        """Called when training starts."""
        self.training_history = []  # Reset history on start
        update_task_status(
            self.task_id,
            progress=0.1,
            current_step="Training started...",
            extra_data={"training_history": []},
            tasks_dir=self.tasks_dir
        )

    def on_train_end(self, trainer):
        """Called when training ends."""
        update_task_status(
            self.task_id,
            progress=0.9,
            current_step="Saving model...",
            extra_data={"training_history": self.training_history},
            tasks_dir=self.tasks_dir
        )


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 training with GPU auto-scaling")
    parser.add_argument("--task-id", required=True, help="Task ID for status updates")
    parser.add_argument("--tasks-dir", help="Tasks directory for status files")
    parser.add_argument("--dataset", required=True, help="Path to dataset YAML")
    parser.add_argument("--model", default=None, help="Base model path or name (auto-detected if not specified)")
    parser.add_argument("--output", default="models/finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (auto-scaled if not specified)")
    parser.add_argument("--fast", action="store_true", help="Use fast training config")
    # New options
    parser.add_argument("--auto-scale", action="store_true", default=True, help="Enable GPU auto-scaling")
    parser.add_argument("--no-auto-scale", action="store_true", help="Disable GPU auto-scaling")
    parser.add_argument("--tensorboard", action="store_true", default=True, help="Enable TensorBoard")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard")
    parser.add_argument("--tensorboard-port", type=int, default=6006, help="TensorBoard port")
    parser.add_argument("--advanced-params", type=str, default=None, help="Advanced parameters as JSON string")
    parser.add_argument("--synthetic-config", type=str, default=None, help="Synthetic generation config as JSON string")
    args = parser.parse_args()

    task_id = args.task_id
    tasks_dir = args.tasks_dir

    # Determine settings
    auto_scale = args.auto_scale and not args.no_auto_scale
    tensorboard_enabled = args.tensorboard and not args.no_tensorboard

    try:
        update_task_status(
            task_id,
            progress=0.02,
            current_step="Checking GPU availability...",
            tasks_dir=tasks_dir
        )

        # Check GPU
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_info = ""
        gpu_tier = "unknown"
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
            extra_data={"gpu_available": gpu_available, "gpu_info": gpu_info},
            tasks_dir=tasks_dir
        )

        # Import training modules
        from training.quick_finetune import CompetitionTrainer, COMPETITION_CONFIG, FAST_CONFIG
        from training.gpu_scaler import GPUScaler
        from training.tensorboard_monitor import (
            CompetitionTensorBoardCallback,
            TensorBoardServer,
            enable_ultralytics_tensorboard,
        )

        # GPU scaling
        config = None
        if auto_scale and not args.fast:
            update_task_status(
                task_id,
                progress=0.06,
                current_step="Auto-scaling for GPU...",
                tasks_dir=tasks_dir
            )
            gpu_scaler = GPUScaler()
            config = gpu_scaler.get_optimal_config()
            gpu_tier = gpu_scaler.get_tier().value
            print(f"GPU Tier: {gpu_tier}")
            print(f"Auto-scaled config: model={config['model']}, batch={config['batch']}")
        else:
            config = FAST_CONFIG.copy() if args.fast else COMPETITION_CONFIG.copy()

        # Apply overrides
        config["epochs"] = args.epochs
        if args.batch:
            config["batch"] = args.batch
        if args.model:
            config["model"] = args.model

        # Apply advanced parameters (highest priority - overrides GPU scaling)
        if args.advanced_params:
            try:
                import json
                advanced = json.loads(args.advanced_params)

                # Whitelist validation for security
                allowed_keys = {
                    # Augmentation
                    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
                    "flipud", "fliplr", "mosaic", "mixup",
                    # Optimizer
                    "optimizer", "lr0", "lrf", "momentum", "weight_decay",
                    # LLRD (Layer-wise Learning Rate Decay)
                    "llrd_enabled", "llrd_decay_rate",
                    # SWA (Stochastic Weight Averaging)
                    "swa_enabled", "swa_start_epoch", "swa_lr",
                    # Performance
                    "workers", "cache", "amp", "imgsz", "patience", "close_mosaic",
                    # Layer freeze (prevents overfitting)
                    "freeze", "warmup_epochs",
                    # Overfitting prevention (Tier 1)
                    "label_smoothing", "cos_lr", "multi_scale",
                    # Checkpoint
                    "save", "save_period", "exist_ok",
                }

                # Filter to only allowed keys
                filtered_params = {k: v for k, v in advanced.items() if k in allowed_keys}
                config.update(filtered_params)

                print(f"Applied {len(filtered_params)} advanced parameters:")
                for key, value in filtered_params.items():
                    print(f"  {key}: {value}")

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse advanced parameters: {e}")
            except Exception as e:
                print(f"Warning: Error applying advanced parameters: {e}")

        # Apply synthetic config (stored separately, not added to YOLO config)
        # SYNTHETIC_CONFIG_KEYS is imported from training.synthetic_data_manager
        synthetic_config = {}
        if args.synthetic_config:
            try:
                import json
                synthetic = json.loads(args.synthetic_config)

                # Filter to only allowed keys
                synthetic_config = {k: v for k, v in synthetic.items() if k in SYNTHETIC_CONFIG_KEYS}

                print(f"Parsed {len(synthetic_config)} synthetic generation parameters:")
                for key, value in synthetic_config.items():
                    print(f"  {key}: {value}")

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse synthetic config: {e}")
            except Exception as e:
                print(f"Warning: Error applying synthetic config: {e}")

        update_task_status(
            task_id,
            progress=0.08,
            current_step="Initializing trainer...",
            extra_data={
                "gpu_available": gpu_available,
                "gpu_info": gpu_info,
                "gpu_tier": gpu_tier,
                "config": {
                    "model": config.get("model"),
                    "batch": config.get("batch"),
                    "epochs": config.get("epochs"),
                    "imgsz": config.get("imgsz"),
                },
            },
            tasks_dir=tasks_dir
        )

        # Create trainer with synthetic config merged for _generate_dynamic_synthetic
        trainer_config = config.copy()
        trainer_config.update(synthetic_config)

        trainer = CompetitionTrainer(
            base_model=args.model or config.get("model", "yolov8m.pt"),
            output_dir=args.output,
            config=trainer_config,
            auto_scale=False,  # Already scaled above
            tensorboard=tensorboard_enabled,
            tensorboard_port=args.tensorboard_port,
        )

        update_task_status(
            task_id,
            progress=0.1,
            current_step="Loading model and starting training...",
            tasks_dir=tasks_dir
        )

        # Load YOLO model and add callbacks
        from ultralytics import YOLO
        from scripts.common.model_utils import resolve_model_path
        base_model_name = args.model or config.get("model", "yolov8m.pt")
        base_model = resolve_model_path(base_model_name, verbose=True)
        model = YOLO(base_model)

        # Create and register progress callback
        progress_callback = TrainingProgressCallback(task_id, args.epochs, tasks_dir)
        model.add_callback("on_train_start", progress_callback.on_train_start)
        model.add_callback("on_train_epoch_end", progress_callback.on_train_epoch_end)
        model.add_callback("on_train_end", progress_callback.on_train_end)

        # Setup TensorBoard
        tensorboard_url = ""
        tensorboard_server = None
        if tensorboard_enabled:
            enable_ultralytics_tensorboard()

            # Create custom callback
            log_dir = Path(args.output) / trainer.run_name / "tensorboard"
            tb_callback = CompetitionTensorBoardCallback(
                log_dir=str(log_dir),
                target_map50=0.85,
            )
            model.add_callback("on_pretrain_routine_start", tb_callback.on_pretrain_routine_start)
            model.add_callback("on_train_epoch_start", tb_callback.on_train_epoch_start)
            model.add_callback("on_train_epoch_end", tb_callback.on_train_epoch_end)
            model.add_callback("on_fit_epoch_end", tb_callback.on_fit_epoch_end)
            model.add_callback("on_train_end", tb_callback.on_train_end)

            # Start TensorBoard server
            tensorboard_server = TensorBoardServer(str(log_dir), port=args.tensorboard_port)
            tensorboard_url = tensorboard_server.start()

            update_task_status(
                task_id,
                progress=0.1,
                current_step="Training started...",
                extra_data={"tensorboard_url": tensorboard_url},
                tasks_dir=tasks_dir
            )
            print(f"TensorBoard: {tensorboard_url}")

        # Validate dataset
        if not trainer._validate_dataset(args.dataset):
            raise ValueError("Dataset validation failed")

        # Generate dynamic synthetic images if enabled
        dataset_dir_for_synthetic = Path(args.dataset).resolve().parent
        synthetic_added = trainer._generate_dynamic_synthetic(dataset_dir_for_synthetic)
        if synthetic_added > 0:
            print(f"Added {synthetic_added} dynamic synthetic images to training set")

        # Change working directory to dataset directory for relative path resolution
        # This allows data.yaml to use "path: ." which works in both host and container
        import os
        dataset_path = Path(args.dataset).resolve()  # Resolve to absolute path
        dataset_dir = dataset_path.parent
        original_cwd = os.getcwd()

        # Extract LLRD parameters from config
        llrd_enabled = config.pop("llrd_enabled", False)
        llrd_decay_rate = config.pop("llrd_decay_rate", 0.9)

        # Extract SWA parameters from config (custom implementation, not YOLO native)
        swa_enabled = config.pop("swa_enabled", False)
        swa_start_epoch = config.pop("swa_start_epoch", 10)
        swa_lr = config.pop("swa_lr", 0.0005)

        try:
            os.chdir(dataset_dir)
            print(f"Changed working directory to: {dataset_dir}")

            # Start training
            import time
            start_time = time.time()

            if llrd_enabled:
                # Use LLRD-enabled custom trainer
                print(f"LLRD enabled with decay_rate={llrd_decay_rate}")
                from training.llrd_trainer import LLRDDetectionTrainer, LLRDConfig

                llrd_config = LLRDConfig(enabled=True, decay_rate=llrd_decay_rate)

                # Build overrides for LLRD trainer (filter out synthetic keys)
                yolo_config = {k: v for k, v in config.items() if k not in SYNTHETIC_CONFIG_KEYS}
                llrd_overrides = {
                    "data": str(dataset_path),
                    "model": base_model,
                    "project": str(trainer.output_dir),
                    "name": trainer.run_name,
                    "verbose": True,
                    **yolo_config,
                }

                llrd_trainer = LLRDDetectionTrainer(
                    overrides=llrd_overrides,
                    llrd_config=llrd_config,
                )

                # Add progress callbacks to LLRD trainer
                llrd_trainer.add_callback("on_train_start", progress_callback.on_train_start)
                llrd_trainer.add_callback("on_train_epoch_end", progress_callback.on_train_epoch_end)
                llrd_trainer.add_callback("on_train_end", progress_callback.on_train_end)

                # Add TensorBoard callbacks if enabled
                if tensorboard_enabled:
                    llrd_trainer.add_callback("on_pretrain_routine_start", tb_callback.on_pretrain_routine_start)
                    llrd_trainer.add_callback("on_train_epoch_start", tb_callback.on_train_epoch_start)
                    llrd_trainer.add_callback("on_train_epoch_end", tb_callback.on_train_epoch_end)
                    llrd_trainer.add_callback("on_fit_epoch_end", tb_callback.on_fit_epoch_end)
                    llrd_trainer.add_callback("on_train_end", tb_callback.on_train_end)

                # Train with LLRD
                results = llrd_trainer.train()
            else:
                # Use standard YOLO training (filter out synthetic keys)
                yolo_config = {k: v for k, v in config.items() if k not in SYNTHETIC_CONFIG_KEYS}
                results = model.train(
                    data=str(dataset_path),  # Use resolved absolute path
                    project=str(trainer.output_dir),
                    name=trainer.run_name,
                    verbose=True,
                    **yolo_config,
                )

            training_time = (time.time() - start_time) / 60  # minutes
        finally:
            # Always restore original working directory
            os.chdir(original_cwd)
            print(f"Restored working directory to: {original_cwd}")

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
                "tensorboard_url": tensorboard_url,
                "gpu_tier": gpu_tier,
                "llrd_enabled": llrd_enabled,
                "llrd_decay_rate": llrd_decay_rate if llrd_enabled else None,
                "config": {
                    "model": config.get("model"),
                    "batch": config.get("batch"),
                    "epochs": config.get("epochs"),
                    "imgsz": config.get("imgsz"),
                },
            },
            tasks_dir=tasks_dir
        )

        print(f"\nTraining completed!")
        print(f"Training time: {training_time:.1f} minutes")
        print(f"Best model: {best_path}")
        if tensorboard_url:
            print(f"TensorBoard: {tensorboard_url}")
        if metrics:
            print(f"Final mAP50: {metrics.get('mAP50', 'N/A'):.4f}")

    except KeyboardInterrupt:
        update_task_status(
            task_id,
            progress=0.0,
            current_step="Cancelled",
            status="cancelled",
            error_message="Training cancelled by user",
            tasks_dir=tasks_dir
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
            extra_data={"traceback": traceback_str},
            tasks_dir=tasks_dir
        )

        print(f"Error: {error_msg}", file=sys.stderr)
        print(traceback_str, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
