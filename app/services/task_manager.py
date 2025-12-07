"""
Task Manager

Manages long-running ML pipeline tasks using subprocesses.
Provides progress tracking via JSON status files that survive Streamlit reruns.
Supports profile-based data isolation via PathCoordinator.
"""

import json
import os
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .path_coordinator import PathCoordinator


class TaskStatus(Enum):
    """Status of a task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """
    Information about a task.

    Stored as JSON in the tasks directory for persistence across Streamlit reruns.
    """

    task_id: str
    task_type: str  # "annotation", "training", "evaluation"
    status: TaskStatus
    progress: float  # 0.0 to 1.0
    current_step: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    pid: Optional[int] = None
    command: Optional[List[str]] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "TaskInfo":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = TaskStatus(data["status"])
        return cls(**data)

    @property
    def is_active(self) -> bool:
        """Check if task is still active (pending or running)."""
        return self.status in [TaskStatus.PENDING, TaskStatus.RUNNING]

    @property
    def is_finished(self) -> bool:
        """Check if task has finished (completed, failed, or cancelled)."""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]

    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if not self.started_at:
            return None

        start = datetime.fromisoformat(self.started_at)
        if self.completed_at:
            end = datetime.fromisoformat(self.completed_at)
        else:
            end = datetime.now()

        return (end - start).total_seconds()

    @property
    def elapsed_time_str(self) -> str:
        """Get elapsed time as formatted string."""
        elapsed = self.elapsed_time
        if elapsed is None:
            return "N/A"

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"


class TaskManager:
    """
    Manages long-running ML pipeline tasks.

    Task execution flow:
    1. Create task with unique ID
    2. Launch subprocess running task runner script
    3. Task runner updates JSON status file with progress
    4. Streamlit polls status file on each rerun
    5. On completion, results are stored at result_path

    Usage:
        manager = TaskManager()

        # Start annotation task
        task_id = manager.start_annotation(
            method="background",
            input_dir="datasets/raw_captures",
            output_dir="datasets/annotated/session_001",
            class_config="config/object_classes.json",
            background_path="datasets/backgrounds/white.jpg"
        )

        # Poll status
        task = manager.get_task(task_id)
        print(f"Progress: {task.progress * 100:.1f}%")

        # Cancel if needed
        manager.cancel_task(task_id)
    """

    def __init__(self, path_coordinator: Optional["PathCoordinator"] = None):
        """
        Initialize task manager.

        Args:
            path_coordinator: PathCoordinator instance for profile-aware paths.
                            If None, creates a new one (uses active profile).
        """
        # Use path coordinator for profile-aware paths
        if path_coordinator is None:
            from .path_coordinator import PathCoordinator
            self._path_coordinator = PathCoordinator()
        else:
            self._path_coordinator = path_coordinator

        # Get tasks directory from coordinator (profile-aware)
        self.tasks_dir = self._path_coordinator.get_path("app_tasks_dir")
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

        # Project root for finding scripts
        self.project_root = self._path_coordinator.project_root

    def _status_file(self, task_id: str) -> Path:
        """Get path to task status file."""
        return self.tasks_dir / f"{task_id}.json"

    def _save_status(self, task: TaskInfo) -> None:
        """Save task status to file."""
        with open(self._status_file(task.task_id), "w") as f:
            json.dump(task.to_dict(), f, indent=2)

    def _load_status(self, task_id: str) -> Optional[TaskInfo]:
        """Load task status from file."""
        status_file = self._status_file(task_id)
        if not status_file.exists():
            return None
        try:
            with open(status_file) as f:
                return TaskInfo.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get task information.

        Args:
            task_id: Task ID

        Returns:
            TaskInfo or None if not found
        """
        task = self._load_status(task_id)

        # Check if running process is still alive
        if task and task.status == TaskStatus.RUNNING and task.pid:
            if not self._is_process_running(task.pid):
                # Process died without updating status
                task.status = TaskStatus.FAILED
                task.error_message = "Process terminated unexpectedly"
                task.completed_at = datetime.now().isoformat()
                self._save_status(task)

        return task

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def get_all_tasks(self, task_type: Optional[str] = None) -> List[TaskInfo]:
        """
        Get all tasks, optionally filtered by type.

        Args:
            task_type: Optional filter by task type

        Returns:
            List of TaskInfo sorted by start time (newest first)
        """
        tasks = []
        for f in self.tasks_dir.glob("*.json"):
            task = self.get_task(f.stem)
            if task:
                if task_type is None or task.task_type == task_type:
                    tasks.append(task)

        return sorted(tasks, key=lambda t: t.started_at or "", reverse=True)

    def get_active_tasks(self, task_type: Optional[str] = None) -> List[TaskInfo]:
        """
        Get active (pending or running) tasks.

        Args:
            task_type: Optional filter by task type

        Returns:
            List of active TaskInfo
        """
        return [t for t in self.get_all_tasks(task_type) if t.is_active]

    def get_recent_tasks(self, limit: int = 10, task_type: Optional[str] = None) -> List[TaskInfo]:
        """
        Get most recent tasks.

        Args:
            limit: Maximum number of tasks to return
            task_type: Optional filter by task type

        Returns:
            List of TaskInfo limited to specified count
        """
        return self.get_all_tasks(task_type)[:limit]

    def _generate_task_id(self, task_type: str) -> str:
        """Generate unique task ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{task_type}_{timestamp}"

    def _launch_subprocess(self, cmd: List[str], task: TaskInfo) -> subprocess.Popen:
        """
        Launch subprocess for task.

        Args:
            cmd: Command to run
            task: Task info (will be updated with PID)

        Returns:
            Popen object
        """
        # Use start_new_session to detach from parent process group
        # This allows the subprocess to survive if parent is terminated
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            cwd=str(self.project_root),
        )

        task.pid = process.pid
        task.status = TaskStatus.RUNNING
        task.command = cmd
        self._save_status(task)

        return process

    # ========== Annotation ==========

    def start_annotation(
        self,
        method: str,
        input_dir: str,
        output_dir: str,
        class_config: str,
        background_path: Optional[str] = None,
        train_val_split: float = 0.85,
        min_area: int = 500,
    ) -> str:
        """
        Start annotation task.

        Args:
            method: Annotation method ("background" or "sam2")
            input_dir: Input directory with raw captures
            output_dir: Output directory for annotated dataset
            class_config: Path to class configuration JSON
            background_path: Background image path (required for background method)
            train_val_split: Train/val split ratio
            min_area: Minimum contour area for background method

        Returns:
            Task ID

        Raises:
            ValueError: If method is "background" and background_path is not provided
        """
        if method == "background" and not background_path:
            raise ValueError("Background path required for background method")

        task_id = self._generate_task_id("annotation")

        task = TaskInfo(
            task_id=task_id,
            task_type="annotation",
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step="Initializing...",
            started_at=datetime.now().isoformat(),
            extra_data={
                "method": method,
                "input_dir": input_dir,
                "output_dir": output_dir,
            }
        )
        self._save_status(task)

        # Build command
        runner_script = self.project_root / "app" / "services" / "task_runners" / "run_annotation.py"
        cmd = [
            sys.executable,
            str(runner_script),
            "--task-id", task_id,
            "--tasks-dir", str(self.tasks_dir),
            "--method", method,
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--class-config", class_config,
            "--split", str(train_val_split),
            "--min-area", str(min_area),
        ]

        if background_path:
            cmd.extend(["--background", background_path])

        self._launch_subprocess(cmd, task)
        return task_id

    # ========== Training ==========

    def start_training(
        self,
        dataset_yaml: str,
        base_model: str = "yolov8m.pt",
        output_dir: str = "models/finetuned",
        epochs: int = 50,
        batch_size: int = 16,
        fast_mode: bool = False,
    ) -> str:
        """
        Start training task.

        Args:
            dataset_yaml: Path to dataset YAML configuration
            base_model: Base model path or name
            output_dir: Output directory for trained model
            epochs: Number of training epochs
            batch_size: Batch size
            fast_mode: Use fast training configuration

        Returns:
            Task ID
        """
        task_id = self._generate_task_id("training")

        task = TaskInfo(
            task_id=task_id,
            task_type="training",
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step="Initializing...",
            started_at=datetime.now().isoformat(),
            extra_data={
                "dataset_yaml": dataset_yaml,
                "base_model": base_model,
                "epochs": epochs,
                "fast_mode": fast_mode,
            }
        )
        self._save_status(task)

        # Build command
        runner_script = self.project_root / "app" / "services" / "task_runners" / "run_training.py"
        cmd = [
            sys.executable,
            str(runner_script),
            "--task-id", task_id,
            "--tasks-dir", str(self.tasks_dir),
            "--dataset", dataset_yaml,
            "--model", base_model,
            "--output", output_dir,
            "--epochs", str(epochs),
            "--batch", str(batch_size),
        ]

        if fast_mode:
            cmd.append("--fast")

        self._launch_subprocess(cmd, task)
        return task_id

    # ========== Evaluation ==========

    def start_evaluation(
        self,
        model_path: str,
        dataset_yaml: str,
        conf_threshold: float = 0.25,
    ) -> str:
        """
        Start evaluation task.

        Args:
            model_path: Path to trained model
            dataset_yaml: Path to dataset YAML configuration
            conf_threshold: Confidence threshold for detections

        Returns:
            Task ID
        """
        task_id = self._generate_task_id("evaluation")

        task = TaskInfo(
            task_id=task_id,
            task_type="evaluation",
            status=TaskStatus.PENDING,
            progress=0.0,
            current_step="Initializing...",
            started_at=datetime.now().isoformat(),
            extra_data={
                "model_path": model_path,
                "dataset_yaml": dataset_yaml,
            }
        )
        self._save_status(task)

        # Build command
        runner_script = self.project_root / "app" / "services" / "task_runners" / "run_evaluation.py"
        cmd = [
            sys.executable,
            str(runner_script),
            "--task-id", task_id,
            "--tasks-dir", str(self.tasks_dir),
            "--model", model_path,
            "--dataset", dataset_yaml,
            "--conf", str(conf_threshold),
        ]

        self._launch_subprocess(cmd, task)
        return task_id

    # ========== Task Control ==========

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.

        Args:
            task_id: Task ID

        Returns:
            True if task was cancelled, False otherwise
        """
        task = self.get_task(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False

        if task.pid:
            try:
                # Send SIGTERM to process group for clean termination
                os.killpg(os.getpgid(task.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass  # Process already terminated

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now().isoformat()
        task.current_step = "Cancelled by user"
        self._save_status(task)
        return True

    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task record.

        Args:
            task_id: Task ID

        Returns:
            True if deleted, False if not found
        """
        # Cancel if running
        task = self.get_task(task_id)
        if task and task.status == TaskStatus.RUNNING:
            self.cancel_task(task_id)

        status_file = self._status_file(task_id)
        if status_file.exists():
            status_file.unlink()
            return True
        return False

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """
        Delete old completed/failed task records.

        Args:
            max_age_hours: Maximum age in hours for task records

        Returns:
            Number of tasks deleted
        """
        deleted = 0
        cutoff = datetime.now().timestamp() - (max_age_hours * 3600)

        for f in self.tasks_dir.glob("*.json"):
            if f.stat().st_mtime < cutoff:
                task = self.get_task(f.stem)
                if task and task.is_finished:
                    f.unlink()
                    deleted += 1

        return deleted


# Helper function for updating task status from runner scripts
def update_task_status(
    task_id: str,
    progress: Optional[float] = None,
    current_step: Optional[str] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
    result_path: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    tasks_dir: Optional[str] = None,
) -> None:
    """
    Update task status from a runner script.

    This function is meant to be called from task runner scripts to update
    the JSON status file.

    Args:
        task_id: Task ID
        progress: Progress value (0.0 to 1.0)
        current_step: Current step description
        status: New status (pending, running, completed, failed, cancelled)
        error_message: Error message if failed
        result_path: Path to result file/directory
        extra_data: Additional data to merge into extra_data
        tasks_dir: Tasks directory path (defaults to app/data/tasks/)
    """
    if tasks_dir is None:
        tasks_dir = Path(__file__).parent.parent / "data" / "tasks"
    else:
        tasks_dir = Path(tasks_dir)

    status_file = tasks_dir / f"{task_id}.json"

    if not status_file.exists():
        raise FileNotFoundError(f"Task status file not found: {status_file}")

    with open(status_file) as f:
        data = json.load(f)

    if progress is not None:
        data["progress"] = progress

    if current_step is not None:
        data["current_step"] = current_step

    if status is not None:
        data["status"] = status
        if status in ["completed", "failed", "cancelled"]:
            data["completed_at"] = datetime.now().isoformat()

    if error_message is not None:
        data["error_message"] = error_message

    if result_path is not None:
        data["result_path"] = result_path

    if extra_data is not None:
        if "extra_data" not in data:
            data["extra_data"] = {}
        data["extra_data"].update(extra_data)

    with open(status_file, "w") as f:
        json.dump(data, f, indent=2)
