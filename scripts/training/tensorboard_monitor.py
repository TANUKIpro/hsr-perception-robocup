#!/usr/bin/env python3
"""
TensorBoard Integration for YOLOv8 Training

Provides TensorBoard monitoring with custom competition-specific metrics.
Designed for minimal impact on training speed.
"""

import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from colorama import Fore, Style

# Check TensorBoard availability
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


@dataclass
class TensorBoardConfig:
    """Configuration for TensorBoard integration."""

    log_dir: Path
    run_name: str
    port: int = 6006
    host: str = "0.0.0.0"
    auto_launch: bool = True
    flush_secs: int = 30
    log_frequency: int = 1  # Log every N epochs


class TensorBoardServer:
    """
    Manages TensorBoard server as a subprocess.

    Handles automatic port selection, background launch, and cleanup.
    """

    DEFAULT_PORT = 6006

    def __init__(self, log_dir: str, port: Optional[int] = None):
        """
        Initialize TensorBoard server manager.

        Args:
            log_dir: Directory containing TensorBoard logs
            port: Port to use (auto-selects if None or unavailable)
        """
        self.log_dir = log_dir
        self.port = port or self._find_available_port()
        self.process: Optional[subprocess.Popen] = None
        self.url: str = ""

    def _find_available_port(self, start_port: int = 6006) -> int:
        """
        Find an available port for TensorBoard.

        Args:
            start_port: Port to start searching from

        Returns:
            Available port number
        """
        for port in range(start_port, start_port + 100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("localhost", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("No available port found for TensorBoard")

    def start(self, background: bool = True, wait_seconds: float = 2.0) -> str:
        """
        Start TensorBoard server.

        Args:
            background: Run in background (non-blocking)
            wait_seconds: Time to wait for server startup

        Returns:
            TensorBoard URL
        """
        if self.process is not None and self.is_running():
            return self.url

        cmd = [
            "tensorboard",
            "--logdir",
            str(self.log_dir),
            "--port",
            str(self.port),
            "--bind_all",
        ]

        try:
            if background:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                # Wait for server to start
                time.sleep(wait_seconds)
            else:
                # Blocking mode (for debugging)
                subprocess.run(cmd, check=True)

            self.url = f"http://localhost:{self.port}"
            print(f"{Fore.GREEN}TensorBoard started: {self.url}{Style.RESET_ALL}")
            return self.url

        except FileNotFoundError:
            print(
                f"{Fore.YELLOW}Warning: TensorBoard not found. "
                f"Install with: pip install tensorboard{Style.RESET_ALL}"
            )
            return ""
        except Exception as e:
            print(f"{Fore.YELLOW}Warning: Failed to start TensorBoard: {e}{Style.RESET_ALL}")
            return ""

    def stop(self) -> None:
        """Stop TensorBoard server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
            finally:
                self.process = None

    def is_running(self) -> bool:
        """Check if TensorBoard server is running."""
        if self.process is None:
            return False
        return self.process.poll() is None

    def get_url(self) -> str:
        """Get TensorBoard URL."""
        return self.url

    @classmethod
    def find_running_instance(cls, port: int = 6006) -> Optional[str]:
        """
        Check if TensorBoard is already running on a port.

        Args:
            port: Port to check

        Returns:
            URL if running, None otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            result = s.connect_ex(("localhost", port))
            if result == 0:
                return f"http://localhost:{port}"
        return None


class CompetitionTensorBoardCallback:
    """
    Custom TensorBoard callback for competition-day training.

    Logs additional competition-specific metrics:
    - Epoch execution time
    - Estimated time remaining (ETA)
    - Training progress percentage
    - Target achievement rate (mAP vs target)
    - Gap to target mAP

    Designed for minimal performance impact.
    """

    def __init__(
        self,
        log_dir: str,
        target_map50: float = 0.85,
        target_inference_ms: float = 100.0,
        log_frequency: int = 1,
        flush_secs: int = 30,
    ):
        """
        Initialize competition callback.

        Args:
            log_dir: TensorBoard log directory
            target_map50: Competition mAP target (default 85%)
            target_inference_ms: Target inference time in ms
            log_frequency: Log every N epochs
            flush_secs: Seconds between buffer flushes
        """
        self.log_dir = Path(log_dir)
        self.target_map50 = target_map50
        self.target_inference_ms = target_inference_ms
        self.log_frequency = log_frequency
        self.flush_secs = flush_secs

        self.writer: Optional[SummaryWriter] = None
        self.start_time: Optional[float] = None
        self.epoch_times: List[float] = []
        self.total_epochs: int = 0
        self._epoch_start: float = 0

    def on_pretrain_routine_start(self, trainer: Any) -> None:
        """Called before training starts. Initialize TensorBoard writer."""
        if not TENSORBOARD_AVAILABLE:
            print(
                f"{Fore.YELLOW}Warning: TensorBoard not available. "
                f"Install with: pip install tensorboard{Style.RESET_ALL}"
            )
            return

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize writer
        self.writer = SummaryWriter(str(self.log_dir), flush_secs=self.flush_secs)
        self.total_epochs = trainer.epochs
        self.start_time = time.time()

        # Log initial configuration
        config_text = (
            f"**Training Configuration**\n\n"
            f"- Model: {getattr(trainer.model, 'yaml_file', 'unknown')}\n"
            f"- Epochs: {self.total_epochs}\n"
            f"- Batch Size: {trainer.batch_size}\n"
            f"- Image Size: {trainer.args.imgsz}\n"
        )
        self.writer.add_text("config/training", config_text, 0)

        target_text = (
            f"**Competition Targets**\n\n"
            f"- Target mAP50: {self.target_map50:.0%}\n"
            f"- Target Inference: {self.target_inference_ms}ms\n"
        )
        self.writer.add_text("config/targets", target_text, 0)

    def on_train_epoch_start(self, trainer: Any) -> None:
        """Called at the start of each training epoch."""
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Called at the end of each training epoch."""
        if not self.writer:
            return

        epoch = trainer.epoch + 1
        epoch_time = time.time() - self._epoch_start
        self.epoch_times.append(epoch_time)

        # Check log frequency
        if epoch % self.log_frequency != 0:
            return

        # Calculate metrics
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.total_epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_minutes = eta_seconds / 60
        elapsed_minutes = (time.time() - self.start_time) / 60
        progress = epoch / self.total_epochs

        # Log competition metrics
        self.writer.add_scalar("competition/epoch_time_sec", epoch_time, epoch)
        self.writer.add_scalar("competition/avg_epoch_time_sec", avg_epoch_time, epoch)
        self.writer.add_scalar("competition/eta_minutes", eta_minutes, epoch)
        self.writer.add_scalar("competition/elapsed_minutes", elapsed_minutes, epoch)
        self.writer.add_scalar("competition/progress", progress, epoch)

    def on_fit_epoch_end(self, trainer: Any) -> None:
        """Called after validation at the end of each epoch."""
        if not self.writer:
            return

        epoch = trainer.epoch + 1

        # Get current mAP50 from trainer metrics
        current_map50 = 0.0
        try:
            if hasattr(trainer, "metrics"):
                current_map50 = trainer.metrics.get("metrics/mAP50(B)", 0)
        except Exception:
            pass

        # Calculate target achievement
        if self.target_map50 > 0:
            target_achievement = min(current_map50 / self.target_map50, 1.0)
        else:
            target_achievement = 0.0

        map50_gap = self.target_map50 - current_map50
        target_met = 1.0 if current_map50 >= self.target_map50 else 0.0

        # Log target metrics
        self.writer.add_scalar("competition/current_map50", current_map50, epoch)
        self.writer.add_scalar("competition/target_achievement", target_achievement, epoch)
        self.writer.add_scalar("competition/map50_gap", map50_gap, epoch)
        self.writer.add_scalar("competition/target_met", target_met, epoch)

    def on_train_end(self, trainer: Any) -> None:
        """Called when training ends."""
        if not self.writer:
            return

        # Calculate final statistics
        total_time_minutes = (time.time() - self.start_time) / 60

        # Get final metrics
        final_map50 = 0.0
        try:
            if hasattr(trainer, "metrics"):
                final_map50 = trainer.metrics.get("metrics/mAP50(B)", 0)
        except Exception:
            pass

        target_achieved = final_map50 >= self.target_map50

        # Log final summary
        summary_text = (
            f"**Training Complete**\n\n"
            f"- Total Time: {total_time_minutes:.1f} minutes\n"
            f"- Final mAP50: {final_map50:.4f}\n"
            f"- Target ({self.target_map50:.0%}): "
            f"{'Achieved' if target_achieved else 'Not Achieved'}\n"
        )
        self.writer.add_text("summary/final", summary_text, 0)

        # Close writer
        self.writer.close()


class GPUMonitorCallback:
    """
    Optional GPU monitoring callback for TensorBoard.

    Logs GPU memory usage at specified intervals.
    Has higher overhead, so uses lower frequency by default.
    """

    def __init__(
        self,
        writer: SummaryWriter,
        log_interval: int = 5,  # Every 5 epochs
    ):
        """
        Initialize GPU monitor.

        Args:
            writer: TensorBoard SummaryWriter
            log_interval: Log every N epochs
        """
        self.writer = writer
        self.log_interval = log_interval

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Log GPU stats at end of epoch."""
        epoch = trainer.epoch + 1

        if epoch % self.log_interval != 0:
            return

        try:
            import torch

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                self.writer.add_scalar("gpu/memory_allocated_gb", allocated, epoch)
                self.writer.add_scalar("gpu/memory_reserved_gb", reserved, epoch)
        except Exception:
            pass


class TensorBoardManager:
    """
    High-level manager for TensorBoard integration.

    Handles log directory management, server lifecycle, and cleanup.
    """

    def __init__(self, base_dir: Path, max_runs: int = 10):
        """
        Initialize TensorBoard manager.

        Args:
            base_dir: Base directory for training runs
            max_runs: Maximum number of runs to keep
        """
        self.base_dir = Path(base_dir)
        self.max_runs = max_runs
        self.server: Optional[TensorBoardServer] = None
        self.callback: Optional[CompetitionTensorBoardCallback] = None

    def get_log_dir(self, run_name: str) -> Path:
        """
        Get TensorBoard log directory for a run.

        Args:
            run_name: Training run name

        Returns:
            Path to log directory
        """
        return self.base_dir / run_name / "tensorboard"

    def create_callback(
        self,
        run_name: str,
        target_map50: float = 0.85,
    ) -> CompetitionTensorBoardCallback:
        """
        Create competition callback for a training run.

        Args:
            run_name: Training run name
            target_map50: Target mAP for competition

        Returns:
            Configured callback
        """
        log_dir = self.get_log_dir(run_name)
        self.callback = CompetitionTensorBoardCallback(
            log_dir=str(log_dir),
            target_map50=target_map50,
        )
        return self.callback

    def start_server(self, run_name: str, port: int = 6006) -> str:
        """
        Start TensorBoard server for a run.

        Args:
            run_name: Training run name
            port: Server port

        Returns:
            TensorBoard URL
        """
        log_dir = self.get_log_dir(run_name)
        self.server = TensorBoardServer(str(log_dir), port)
        return self.server.start()

    def stop_server(self) -> None:
        """Stop TensorBoard server."""
        if self.server:
            self.server.stop()
            self.server = None

    def cleanup_old_logs(self) -> int:
        """
        Clean up old TensorBoard logs.

        Keeps only the most recent runs up to max_runs.

        Returns:
            Number of runs cleaned up
        """
        import shutil

        if not self.base_dir.exists():
            return 0

        # Get all run directories
        runs = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        cleaned = 0
        for old_run in runs[self.max_runs :]:
            tb_dir = old_run / "tensorboard"
            if tb_dir.exists():
                try:
                    shutil.rmtree(tb_dir)
                    cleaned += 1
                except Exception:
                    pass

        return cleaned

    def get_all_log_dirs(self) -> List[str]:
        """
        Get all TensorBoard log directories.

        Returns:
            List of log directory paths
        """
        log_dirs = []
        if self.base_dir.exists():
            for run_dir in self.base_dir.iterdir():
                if run_dir.is_dir():
                    tb_dir = run_dir / "tensorboard"
                    if tb_dir.exists():
                        log_dirs.append(str(tb_dir))
        return log_dirs


def enable_ultralytics_tensorboard() -> None:
    """
    Enable TensorBoard in Ultralytics settings.

    This enables the built-in TensorBoard logging in YOLO training.
    """
    try:
        from ultralytics import settings

        settings.update({"tensorboard": True})
        print(f"{Fore.GREEN}Ultralytics TensorBoard enabled{Style.RESET_ALL}")
    except ImportError:
        print(
            f"{Fore.YELLOW}Warning: Ultralytics not available. "
            f"TensorBoard will use custom callbacks only.{Style.RESET_ALL}"
        )
    except Exception as e:
        print(f"{Fore.YELLOW}Warning: Failed to enable Ultralytics TensorBoard: {e}{Style.RESET_ALL}")


def check_tensorboard_available() -> bool:
    """Check if TensorBoard is available."""
    return TENSORBOARD_AVAILABLE
