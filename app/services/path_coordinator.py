"""
Path Coordinator

Standardizes paths between the Streamlit app and ML pipeline scripts.
Handles path translation, symlink creation, and directory management.
Supports profile-based data isolation.
"""

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from .profile_manager import ProfileManager


@dataclass
class PathConfig:
    """Path configuration with profile-relative defaults."""

    # App data paths (relative to profile root)
    app_data_dir: str = "app_data"
    app_collected_dir: str = "app_data/collected_images"
    app_reference_dir: str = "app_data/reference_images"
    app_tasks_dir: str = "app_data/tasks"
    app_registry_file: str = "app_data/object_registry.json"
    app_thumbnails_dir: str = "app_data/thumbnails"

    # Dataset paths (relative to profile root)
    datasets_dir: str = "datasets"
    raw_captures_dir: str = "datasets/raw_captures"
    annotated_dir: str = "datasets/annotated"
    backgrounds_dir: str = "datasets/backgrounds"
    videos_dir: str = "datasets/videos"

    # Model paths - finetuned is profile-specific
    finetuned_dir: str = "models/finetuned"

    # Shared paths (relative to project root, not profile)
    models_dir: str = "models"
    pretrained_dir: str = "models/pretrained"
    config_dir: str = "config"
    class_config_file: str = "config/object_classes.json"


class PathCoordinator:
    """
    Coordinates paths between app and ML pipeline with profile support.

    Provides:
    - Profile-aware path resolution
    - Path resolution relative to project root for shared resources
    - Symlink management for data migration
    - Session-based output directory creation
    - Path validation utilities

    Usage:
        coordinator = PathCoordinator()

        # Get absolute path (profile-aware)
        raw_dir = coordinator.get_path("raw_captures_dir")

        # Get shared path (project-level)
        pretrained = coordinator.get_path("pretrained_dir")

        # Prepare paths for annotation
        paths = coordinator.prepare_annotation_paths()

        # Sync app data to datasets directory
        coordinator.sync_app_to_datasets("apple")
    """

    # Paths that are shared across all profiles (resolved from project root)
    SHARED_PATHS = {
        "models_dir",
        "pretrained_dir",
        "config_dir",
        "class_config_file",
    }

    def __init__(
        self,
        project_root: Optional[Union[str, Path]] = None,
        profile_manager: Optional["ProfileManager"] = None
    ):
        """
        Initialize path coordinator.

        Args:
            project_root: Project root directory. If None, auto-detected from file location.
            profile_manager: ProfileManager instance. If None, creates a new one.
        """
        if project_root is None:
            # Auto-detect: app/services/path_coordinator.py -> project root
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.config = PathConfig()

        # Initialize profile manager
        if profile_manager is None:
            from .profile_manager import ProfileManager
            self._profile_manager = ProfileManager(self.project_root)
        else:
            self._profile_manager = profile_manager

        # Ensure critical directories exist
        self._ensure_directories()

    def _get_profile_root(self, profile_id: Optional[str] = None) -> Path:
        """Get the root directory for a profile."""
        if profile_id is None:
            profile_id = self._profile_manager.get_active_profile_id()
        return self._profile_manager.get_profile_path(profile_id)

    def _ensure_directories(self, profile_id: Optional[str] = None) -> None:
        """Create required directories if they don't exist."""
        profile_root = self._get_profile_root(profile_id)

        # Profile-specific directories
        profile_directories = [
            self.config.app_data_dir,
            self.config.app_collected_dir,
            self.config.app_reference_dir,
            self.config.app_tasks_dir,
            self.config.app_thumbnails_dir,
            self.config.datasets_dir,
            self.config.raw_captures_dir,
            self.config.annotated_dir,
            self.config.backgrounds_dir,
            self.config.videos_dir,
            self.config.finetuned_dir,
        ]

        for dir_path in profile_directories:
            full_path = profile_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        # Shared directories (project-level)
        shared_directories = [
            self.config.models_dir,
            self.config.pretrained_dir,
            self.config.config_dir,
        ]

        for dir_path in shared_directories:
            full_path = self.project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str, profile_id: Optional[str] = None) -> Path:
        """
        Get absolute path for a configured path key.

        For shared paths (pretrained, config), resolves relative to project root.
        For profile-specific paths, resolves relative to the active profile directory.

        Args:
            key: Path configuration key (e.g., "raw_captures_dir", "class_config_file")
            profile_id: Optional profile ID. If None, uses active profile.

        Returns:
            Absolute Path object

        Raises:
            KeyError: If key is not found in configuration
        """
        if not hasattr(self.config, key):
            raise KeyError(f"Unknown path key: {key}. Available: {list(vars(self.config).keys())}")

        rel_path = getattr(self.config, key)

        if key in self.SHARED_PATHS:
            # Shared paths are relative to project root
            return self.project_root / rel_path
        else:
            # Profile-specific paths are relative to profile root
            profile_root = self._get_profile_root(profile_id)
            return profile_root / rel_path

    def get_relative_path(self, key: str) -> str:
        """
        Get relative path string for a configured path key.

        Args:
            key: Path configuration key

        Returns:
            Relative path string
        """
        if not hasattr(self.config, key):
            raise KeyError(f"Unknown path key: {key}")
        return getattr(self.config, key)

    def resolve_path(self, path: Union[str, Path], profile_id: Optional[str] = None) -> Path:
        """
        Resolve a path relative to project root or profile root.

        Args:
            path: Relative or absolute path
            profile_id: Optional profile ID for profile-relative paths

        Returns:
            Absolute Path object
        """
        path = Path(path)
        if path.is_absolute():
            return path
        # Default to profile root for relative paths
        profile_root = self._get_profile_root(profile_id)
        return profile_root / path

    def get_profile_manager(self) -> "ProfileManager":
        """Get the profile manager instance."""
        return self._profile_manager

    # ========== Data Synchronization ==========

    def sync_app_to_datasets(self, object_name: str, force: bool = False) -> Path:
        """
        Sync collected images from app directory to datasets directory.

        Creates a symlink from datasets/raw_captures/{object_name} to
        app_data/collected_images/{object_name}.

        Args:
            object_name: Name of the object (used as directory name)
            force: If True, overwrite existing link/directory

        Returns:
            Path to the synced directory in datasets/

        Raises:
            FileNotFoundError: If source directory doesn't exist
        """
        app_dir = self.get_path("app_collected_dir") / object_name
        dataset_dir = self.get_path("raw_captures_dir") / object_name

        if not app_dir.exists():
            raise FileNotFoundError(f"No collected images for: {object_name} at {app_dir}")

        # Handle existing target
        if dataset_dir.exists() or dataset_dir.is_symlink():
            if not force:
                # If it's already a symlink to the right place, return
                if dataset_dir.is_symlink() and dataset_dir.resolve() == app_dir.resolve():
                    return dataset_dir
                # If real directory exists with data, return it
                if dataset_dir.is_dir() and not dataset_dir.is_symlink():
                    return dataset_dir

            # Remove existing link or empty directory
            if dataset_dir.is_symlink():
                dataset_dir.unlink()
            elif dataset_dir.is_dir() and not any(dataset_dir.iterdir()):
                dataset_dir.rmdir()
            else:
                # Non-empty directory, don't overwrite
                return dataset_dir

        # Create symlink
        if os.name == 'nt':
            # Windows: use directory junction
            import subprocess
            subprocess.run(
                ['cmd', '/c', 'mklink', '/J', str(dataset_dir), str(app_dir)],
                check=True, capture_output=True
            )
        else:
            # Unix: create symlink
            dataset_dir.symlink_to(app_dir.absolute())

        return dataset_dir

    def sync_all_objects(self, object_names: List[str]) -> Dict[str, Path]:
        """
        Sync all registered objects to datasets directory.

        Args:
            object_names: List of object names to sync

        Returns:
            Dictionary mapping object name to synced path
        """
        synced = {}
        for name in object_names:
            try:
                synced[name] = self.sync_app_to_datasets(name)
            except FileNotFoundError:
                pass  # Skip objects without collected images
        return synced

    # ========== Session Management ==========

    def create_annotation_session(self, session_name: Optional[str] = None) -> Dict[str, str]:
        """
        Create a new annotation session with prepared paths.

        Args:
            session_name: Optional session name. If None, generates timestamp-based name.

        Returns:
            Dictionary with keys:
            - session_name: Name of the session
            - input_dir: Path to raw captures
            - output_dir: Path for annotation output
            - class_config: Path to class configuration
            - data_yaml: Expected path for output data.yaml
        """
        if session_name is None:
            session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

        output_dir = self.get_path("annotated_dir") / session_name
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "session_name": session_name,
            "input_dir": str(self.get_path("raw_captures_dir")),
            "output_dir": str(output_dir),
            "class_config": str(self.get_path("class_config_file")),
            "data_yaml": str(output_dir / "data.yaml"),
        }

    def get_annotation_sessions(self) -> List[Dict[str, str]]:
        """
        Get list of existing annotation sessions.

        Returns:
            List of session info dictionaries with keys:
            - name: Session name
            - path: Session directory path
            - has_data_yaml: Whether data.yaml exists
            - created: Creation timestamp
        """
        annotated_dir = self.get_path("annotated_dir")
        sessions = []

        if not annotated_dir.exists():
            return sessions

        for session_dir in sorted(annotated_dir.iterdir(), reverse=True):
            if session_dir.is_dir():
                data_yaml = session_dir / "data.yaml"
                sessions.append({
                    "name": session_dir.name,
                    "path": str(session_dir),
                    "has_data_yaml": data_yaml.exists(),
                    "created": datetime.fromtimestamp(session_dir.stat().st_ctime).isoformat(),
                })

        return sessions

    def get_training_paths(self, annotation_session: str) -> Dict[str, str]:
        """
        Get paths for training based on an annotation session.

        Args:
            annotation_session: Name of the annotation session

        Returns:
            Dictionary with keys:
            - dataset_yaml: Path to data.yaml
            - output_dir: Path for model output

        Raises:
            FileNotFoundError: If dataset not found
        """
        session_dir = self.get_path("annotated_dir") / annotation_session
        dataset_yaml = session_dir / "data.yaml"

        if not dataset_yaml.exists():
            raise FileNotFoundError(
                f"Dataset not found: {dataset_yaml}. "
                f"Run annotation first to create data.yaml"
            )

        return {
            "dataset_yaml": str(dataset_yaml),
            "output_dir": str(self.get_path("finetuned_dir")),
        }

    # ========== Model Management ==========

    def get_trained_models(self) -> List[Dict[str, str]]:
        """
        Get list of trained models.

        Returns:
            List of model info dictionaries with keys:
            - name: Model run name
            - best_path: Path to best.pt
            - last_path: Path to last.pt
            - created: Creation timestamp
        """
        finetuned_dir = self.get_path("finetuned_dir")
        models = []

        if not finetuned_dir.exists():
            return models

        for model_dir in sorted(finetuned_dir.iterdir(), reverse=True):
            if model_dir.is_dir():
                weights_dir = model_dir / "weights"
                best_pt = weights_dir / "best.pt"
                last_pt = weights_dir / "last.pt"

                if best_pt.exists() or last_pt.exists():
                    models.append({
                        "name": model_dir.name,
                        "best_path": str(best_pt) if best_pt.exists() else None,
                        "last_path": str(last_pt) if last_pt.exists() else None,
                        "created": datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat(),
                    })

        return models

    def get_pretrained_models(self) -> List[str]:
        """
        Get list of available pretrained models.

        Returns:
            List of model file paths
        """
        pretrained_dir = self.get_path("pretrained_dir")
        models = []

        if pretrained_dir.exists():
            for model_file in pretrained_dir.glob("*.pt"):
                models.append(str(model_file))

        # Also include standard YOLO model names that will be auto-downloaded
        standard_models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
        for model in standard_models:
            if model not in [Path(m).name for m in models]:
                models.append(model)

        return models

    # ========== Background Images ==========

    def get_background_images(self) -> List[Dict[str, str]]:
        """
        Get list of available background images for annotation.

        Returns:
            List of background info dictionaries with keys:
            - name: File name
            - path: Full path
        """
        backgrounds_dir = self.get_path("backgrounds_dir")
        images = []

        if not backgrounds_dir.exists():
            return images

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        for img_file in backgrounds_dir.iterdir():
            if img_file.suffix.lower() in extensions:
                images.append({
                    "name": img_file.name,
                    "path": str(img_file),
                })

        return images

    def add_background_image(self, source_path: Union[str, Path], name: Optional[str] = None) -> str:
        """
        Add a background image to the backgrounds directory.

        Args:
            source_path: Path to source image
            name: Optional name for the saved file

        Returns:
            Path to saved background image
        """
        source = Path(source_path)
        if name is None:
            name = source.name

        dest = self.get_path("backgrounds_dir") / name
        shutil.copy2(source, dest)
        return str(dest)

    # ========== Validation ==========

    def validate_paths(self) -> Dict[str, bool]:
        """
        Validate that all required paths exist.

        Returns:
            Dictionary mapping path key to existence status
        """
        results = {}
        for key in vars(self.config):
            if key.endswith("_dir") or key.endswith("_file"):
                path = self.get_path(key)
                if key.endswith("_file"):
                    results[key] = path.exists()
                else:
                    results[key] = path.is_dir()
        return results

    def get_path_summary(self) -> Dict[str, str]:
        """
        Get summary of all configured paths.

        Returns:
            Dictionary mapping path key to absolute path string
        """
        summary = {}
        for key in vars(self.config):
            path = self.get_path(key)
            summary[key] = str(path)
        return summary

    # ========== Application Launchers ==========

    def open_annotation_app(
        self,
        input_dir: str,
        output_dir: str,
        class_id: int = 0,
        device: str = "cuda"
    ) -> bool:
        """
        Launch SAM2 Interactive Annotation Application.

        Opens a Tkinter-based GUI for semi-automatic object annotation
        using SAM2 segmentation and video tracking.

        Args:
            input_dir: Directory containing images to annotate
            output_dir: Directory for YOLO label output
            class_id: YOLO class ID for annotations
            device: Device for inference ("cuda" or "cpu")

        Returns:
            True if application launched successfully, False otherwise
        """
        import subprocess
        import sys

        script_path = self.project_root / "scripts/annotation/sam2_interactive_app.py"

        if not script_path.exists():
            return False

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script_path),
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--class-id", str(class_id),
            "--device", device
        ]

        try:
            subprocess.Popen(cmd, start_new_session=True)
            return True
        except Exception:
            return False
