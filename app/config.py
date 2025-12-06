"""
HSR Perception App Configuration

Centralized configuration management for the Streamlit application.
Handles environment detection, path resolution, and feature flags.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import json


@dataclass
class AppConfig:
    """
    Application configuration with environment-aware defaults.

    Supports:
    - Local development
    - Docker deployment
    - ROS2 integration (optional)
    """

    # Environment detection
    environment: str = field(default_factory=lambda: os.getenv("HSR_ENV", "local"))

    # Project root (auto-detected or from env)
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Feature flags
    ros2_enabled: bool = field(default_factory=lambda: os.getenv("HSR_ROS2_ENABLED", "true").lower() == "true")
    gpu_enabled: bool = field(default_factory=lambda: os.getenv("HSR_GPU_ENABLED", "true").lower() == "true")

    # ROS2 settings
    ros2_source_script: str = field(default_factory=lambda: os.getenv(
        "ROS2_SOURCE_SCRIPT",
        "/opt/ros/humble/setup.bash"
    ))

    # Default HSR camera topics
    default_image_topics: list = field(default_factory=lambda: [
        "/hsrb/head_rgbd_sensor/rgb/image_rect_color",
        "/hsrb/head_rgbd_sensor/rgb/image_raw",
        "/hsrb/hand_camera/image_raw",
        "/camera/color/image_raw",
        "/camera/rgb/image_raw",
    ])

    # Capture node service names
    capture_services: Dict[str, str] = field(default_factory=lambda: {
        "set_class": "/continuous_capture/set_class",
        "start_burst": "/continuous_capture/start_burst",
        "get_status": "/continuous_capture/get_status",
    })

    def __post_init__(self):
        """Initialize paths based on environment."""
        if self.environment == "docker":
            self.project_root = Path("/app")

        # Ensure project_root is a Path
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

    @property
    def app_dir(self) -> Path:
        """App directory path."""
        return self.project_root / "app"

    @property
    def app_data_dir(self) -> Path:
        """App data directory path."""
        return self.app_dir / "data"

    @property
    def datasets_dir(self) -> Path:
        """Datasets directory path."""
        return self.project_root / "datasets"

    @property
    def models_dir(self) -> Path:
        """Models directory path."""
        return self.project_root / "models"

    @property
    def config_dir(self) -> Path:
        """Config directory path."""
        return self.project_root / "config"

    @property
    def scripts_dir(self) -> Path:
        """Scripts directory path."""
        return self.project_root / "scripts"

    def check_ros2_available(self) -> bool:
        """
        Check if ROS2 is available in the current environment.

        Returns:
            True if ROS2 commands can be executed
        """
        if not self.ros2_enabled:
            return False

        import subprocess
        try:
            result = subprocess.run(
                ["bash", "-c", f"source {self.ros2_source_script} && ros2 topic list"],
                capture_output=True,
                timeout=5.0
            )
            return result.returncode == 0
        except Exception:
            return False

    def check_gpu_available(self) -> bool:
        """
        Check if GPU (CUDA) is available.

        Returns:
            True if CUDA is available
        """
        if not self.gpu_enabled:
            return False

        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def to_dict(self) -> Dict:
        """Convert to dictionary for display/debugging."""
        return {
            "environment": self.environment,
            "project_root": str(self.project_root),
            "ros2_enabled": self.ros2_enabled,
            "gpu_enabled": self.gpu_enabled,
            "ros2_source_script": self.ros2_source_script,
            "app_dir": str(self.app_dir),
            "datasets_dir": str(self.datasets_dir),
            "models_dir": str(self.models_dir),
            "config_dir": str(self.config_dir),
        }


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """
    Get the global configuration instance.

    Creates a new instance if one doesn't exist.

    Returns:
        AppConfig instance
    """
    global _config
    if _config is None:
        _config = AppConfig()
    return _config


def reload_config() -> AppConfig:
    """
    Reload configuration from environment.

    Use when environment variables may have changed.

    Returns:
        New AppConfig instance
    """
    global _config
    _config = AppConfig()
    return _config
