"""
ROS2 Process Tracker

Tracks and manages running ROS2 processes launched from command presets.
Provides duplicate detection and process lifecycle management.
"""

import json
import os
import signal
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .command_presets_manager import CommandPreset
    from .ros2_bridge import ROS2Bridge


@dataclass
class RunningProcess:
    """Information about a running ROS2 process."""

    preset_id: str
    pid: int
    command: str
    started_at: str
    log_file: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunningProcess":
        """Create instance from dictionary."""
        return cls(**data)


class ROS2ProcessTracker:
    """
    Tracks running ROS2 processes.

    Features:
    - Start/stop commands with ROS2 environment
    - Track running processes across Streamlit re-runs
    - Detect duplicate processes (internal and external)
    - Auto-cleanup dead processes
    """

    def __init__(
        self,
        status_file: Optional[Path] = None,
        ros2_bridge: Optional["ROS2Bridge"] = None,
    ):
        """
        Initialize ROS2ProcessTracker.

        Args:
            status_file: JSON file for tracking running processes.
                        Defaults to: app/data/shared/ros2_processes.json
            ros2_bridge: ROS2Bridge instance for command execution.
        """
        if status_file is None:
            app_root = Path(__file__).parent.parent
            status_file = app_root / "data" / "shared" / "ros2_processes.json"

        self.status_file = status_file
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log directory
        self.log_dir = self.status_file.parent / "ros2_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ROS2Bridge
        if ros2_bridge is None:
            from .ros2_bridge import ROS2Bridge

            ros2_bridge = ROS2Bridge()

        self.ros2_bridge = ros2_bridge

        # Cleanup dead processes on initialization
        self.cleanup_dead_processes()

    def _load_processes(self) -> Dict[str, RunningProcess]:
        """Load running processes from JSON file."""
        if not self.status_file.exists():
            return {}

        try:
            with open(self.status_file, encoding="utf-8") as f:
                data = json.load(f)

            return {
                preset_id: RunningProcess.from_dict(proc_data)
                for preset_id, proc_data in data.get("processes", {}).items()
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return {}

    def _save_processes(self, processes: Dict[str, RunningProcess]) -> None:
        """Save running processes to JSON file."""
        data = {
            "processes": {
                preset_id: proc.to_dict() for preset_id, proc in processes.items()
            }
        }

        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process is still running."""
        try:
            os.kill(pid, 0)  # Signal 0 checks existence without killing
            return True
        except OSError:
            return False

    def _extract_command_pattern(self, command: str) -> str:
        """
        Extract searchable pattern from command for duplicate detection.

        Examples:
        - "ros2 launch hsr_perception capture.launch.py" -> "hsr_perception.*capture.launch.py"
        - "ros2 run my_pkg my_node" -> "my_pkg.*my_node"
        """
        parts = command.split()
        if len(parts) >= 4 and parts[0] == "ros2" and parts[1] == "launch":
            # ros2 launch <pkg> <launch_file>
            # Include both package and launch file for more specific matching
            pkg = parts[2]
            launch_file = parts[3]
            return f"{pkg}.*{launch_file}"
        elif len(parts) >= 4 and parts[0] == "ros2" and parts[1] == "run":
            # ros2 run <pkg> <node>
            pkg = parts[2]
            node = parts[3]
            return f"{pkg}.*{node}"
        else:
            # Generic: use last part of command
            return parts[-1] if parts else command

    def is_command_running_externally(self, command: str) -> bool:
        """
        Check if a similar command is already running externally.

        Uses pgrep to search for matching process command patterns.
        This detects processes started outside this application.

        Args:
            command: Full command string to check

        Returns:
            True if similar process found, False otherwise
        """
        pattern = self._extract_command_pattern(command)

        try:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            # pgrep returns 0 if match found
            return result.returncode == 0
        except Exception:
            return False  # Fail open - allow launch if check fails

    def is_running(self, preset_id: str) -> bool:
        """
        Check if a preset is currently running (tracked internally).

        Args:
            preset_id: Preset ID to check

        Returns:
            True if running, False otherwise
        """
        processes = self._load_processes()

        if preset_id not in processes:
            return False

        return self._is_process_alive(processes[preset_id].pid)

    def start_command(
        self,
        preset: "CommandPreset",
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Start a command from preset with duplicate detection.

        Args:
            preset: CommandPreset to launch
            force: If True, skip duplicate checks

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not force:
            # Check 1: Already tracked as running
            if self.is_running(preset.id):
                return (False, "This preset is already running (tracked)")

            # Check 2: External process check
            if self.is_command_running_externally(preset.command):
                return (False, "Similar process is already running (external)")

        # Create log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{preset.id}_{timestamp}.log"

        # Launch command
        process = self.ros2_bridge.launch_command_background(
            command=preset.command,
            log_file=str(log_file),
        )

        if process is None:
            return (False, "Failed to launch command")

        # Track process
        processes = self._load_processes()
        processes[preset.id] = RunningProcess(
            preset_id=preset.id,
            pid=process.pid,
            command=preset.command,
            started_at=datetime.now().isoformat(),
            log_file=str(log_file),
        )
        self._save_processes(processes)

        return (True, f"Launched successfully (PID: {process.pid})")

    def stop_process(self, preset_id: str) -> Tuple[bool, str]:
        """
        Stop a running process by preset ID.

        Args:
            preset_id: Preset ID of process to stop

        Returns:
            Tuple of (success: bool, message: str)
        """
        processes = self._load_processes()

        if preset_id not in processes:
            return (False, "Process not found in tracking")

        proc = processes[preset_id]

        # Try to terminate gracefully
        try:
            # Try to kill process group first (for child processes)
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            # Fallback: kill the process directly if killpg fails
            # (may happen with start_new_session=True)
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass  # Already dead

        # Remove from tracking
        del processes[preset_id]
        self._save_processes(processes)

        return (True, f"Process stopped (PID: {proc.pid})")

    def get_running_processes(self) -> List[RunningProcess]:
        """
        Get list of currently running processes.

        Automatically cleans up dead processes.

        Returns:
            List of RunningProcess objects
        """
        processes = self._load_processes()

        # Filter out dead processes
        alive_processes = {
            preset_id: proc
            for preset_id, proc in processes.items()
            if self._is_process_alive(proc.pid)
        }

        # Update if any were removed
        if len(alive_processes) != len(processes):
            self._save_processes(alive_processes)

        return list(alive_processes.values())

    def cleanup_dead_processes(self) -> int:
        """
        Remove dead processes from tracking.

        Returns:
            Number of processes removed
        """
        processes = self._load_processes()

        alive_processes = {
            preset_id: proc
            for preset_id, proc in processes.items()
            if self._is_process_alive(proc.pid)
        }

        removed_count = len(processes) - len(alive_processes)

        if removed_count > 0:
            self._save_processes(alive_processes)

        return removed_count

    def get_process_info(self, preset_id: str) -> Optional[RunningProcess]:
        """
        Get running process info by preset ID.

        Args:
            preset_id: Preset ID to look up

        Returns:
            RunningProcess if found and alive, None otherwise
        """
        processes = self._load_processes()
        proc = processes.get(preset_id)

        if proc and self._is_process_alive(proc.pid):
            return proc

        return None
