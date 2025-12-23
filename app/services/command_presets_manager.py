"""
Command Presets Manager

Manages ROS2 command presets stored in a JSON configuration file.
Presets are application-wide (not per-profile).
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class CommandPreset:
    """Single command preset configuration."""

    id: str
    name: str
    command: str
    description: str
    is_default: bool
    created_at: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CommandPreset":
        """Create instance from dictionary."""
        return cls(**data)


class CommandPresetsManager:
    """
    Manages command presets stored in JSON.

    Presets are stored in config/command_presets.json at project level,
    shared across all profiles.
    """

    DEFAULT_PRESETS = [
        {
            "name": "Capture Node",
            "command": "ros2 launch hsr_perception capture.launch.py",
            "description": "Start continuous capture node for HSR perception",
            "is_default": True,
        },
        {
            "name": "OpenNI2 Camera",
            "command": "ros2 launch openni2_camera camera_only.launch.py",
            "description": "Launch OpenNI2 camera driver",
            "is_default": True,
        },
    ]

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize CommandPresetsManager.

        Args:
            config_path: Path to presets JSON file.
                        Defaults to: project_root/config/command_presets.json
        """
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "command_presets.json"

        self.config_path = config_path
        self._ensure_config_exists()

    def _ensure_config_exists(self) -> None:
        """Create config file with defaults if it doesn't exist."""
        if not self.config_path.exists():
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Create default presets
            presets = [
                CommandPreset(
                    id=str(uuid.uuid4()),
                    created_at=datetime.now().isoformat(),
                    **preset_data,
                )
                for preset_data in self.DEFAULT_PRESETS
            ]

            self._save_presets(presets)

    def load_presets(self) -> List[CommandPreset]:
        """
        Load all presets from JSON file.

        Returns:
            List of CommandPreset objects
        """
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = json.load(f)
            return [CommandPreset.from_dict(p) for p in data.get("presets", [])]
        except (json.JSONDecodeError, FileNotFoundError, KeyError, TypeError):
            # Return defaults on error
            return [
                CommandPreset(
                    id=str(uuid.uuid4()),
                    created_at=datetime.now().isoformat(),
                    **preset_data,
                )
                for preset_data in self.DEFAULT_PRESETS
            ]

    def _save_presets(self, presets: List[CommandPreset]) -> None:
        """
        Save presets to JSON file.

        Args:
            presets: List of CommandPreset objects to save
        """
        data = {"version": "1.0.0", "presets": [p.to_dict() for p in presets]}

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_preset(
        self,
        name: str,
        command: str,
        description: str = "",
    ) -> CommandPreset:
        """
        Add a new custom preset.

        Args:
            name: Display name for the preset
            command: Full command to execute
            description: Optional description

        Returns:
            Newly created CommandPreset
        """
        presets = self.load_presets()

        new_preset = CommandPreset(
            id=str(uuid.uuid4()),
            name=name,
            command=command,
            description=description,
            is_default=False,
            created_at=datetime.now().isoformat(),
        )

        presets.append(new_preset)
        self._save_presets(presets)
        return new_preset

    def update_preset(self, preset_id: str, **kwargs) -> bool:
        """
        Update an existing preset.

        Args:
            preset_id: ID of preset to update
            **kwargs: Fields to update (name, command, description)

        Returns:
            True if updated, False if not found
        """
        presets = self.load_presets()

        for preset in presets:
            if preset.id == preset_id:
                # Update allowed fields only
                for key in ["name", "command", "description"]:
                    if key in kwargs:
                        setattr(preset, key, kwargs[key])

                self._save_presets(presets)
                return True

        return False

    def delete_preset(self, preset_id: str) -> bool:
        """
        Delete a preset.

        Default presets cannot be deleted.

        Args:
            preset_id: ID of preset to delete

        Returns:
            True if deleted, False if not found or is default
        """
        presets = self.load_presets()

        # Find preset and check if it's default
        for preset in presets:
            if preset.id == preset_id:
                if preset.is_default:
                    return False  # Cannot delete default presets
                break
        else:
            return False  # Not found

        # Remove preset
        presets = [p for p in presets if p.id != preset_id]
        self._save_presets(presets)
        return True

    def get_preset(self, preset_id: str) -> Optional[CommandPreset]:
        """
        Get a preset by ID.

        Args:
            preset_id: ID of preset to find

        Returns:
            CommandPreset if found, None otherwise
        """
        for preset in self.load_presets():
            if preset.id == preset_id:
                return preset
        return None

    def get_default_presets(self) -> List[CommandPreset]:
        """
        Get default presets only.

        Returns:
            List of default CommandPreset objects
        """
        return [p for p in self.load_presets() if p.is_default]

    def get_custom_presets(self) -> List[CommandPreset]:
        """
        Get custom (non-default) presets only.

        Returns:
            List of custom CommandPreset objects
        """
        return [p for p in self.load_presets() if not p.is_default]
