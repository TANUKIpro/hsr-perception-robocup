"""
Profile Manager

Manages profile CRUD operations and profile switching.
Each profile contains isolated data for objects, datasets, and trained models.
"""

import json
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ProfileMetadata:
    """Metadata for a single profile."""
    id: str
    display_name: str
    created_at: str
    last_accessed: Optional[str] = None
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileMetadata":
        return cls(**data)


class ProfileManager:
    """
    Manages profiles for the HSR Perception application.

    Responsibilities:
    - Create, read, update, delete profiles
    - Switch active profile
    - Migrate existing data to prof_1
    - Provide profile paths to other services
    """

    PROFILES_DIR = "profiles"
    PROFILES_REGISTRY_FILE = "profiles.json"
    PROFILE_METADATA_FILE = "profile.json"

    # Subdirectories within each profile
    PROFILE_SUBDIRS = [
        "app_data",
        "app_data/thumbnails",
        "app_data/collected_images",
        "app_data/reference_images",
        "app_data/tasks",
        "datasets",
        "datasets/raw_captures",
        "datasets/annotated",
        "datasets/backgrounds",
        "models",
        "models/finetuned",
    ]

    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.profiles_dir = self.project_root / self.PROFILES_DIR
        self.registry_file = self.profiles_dir / self.PROFILES_REGISTRY_FILE

        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure profiles directory and registry exist."""
        if not self.registry_file.exists():
            self._initialize_profiles()

    def _initialize_profiles(self) -> None:
        """Initialize profiles directory with default profile."""
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

        # Check if migration is needed
        old_app_data = self.project_root / "app" / "data"
        old_registry = old_app_data / "object_registry.json"

        needs_migration = old_registry.exists()

        # Create default profile
        default_profile = ProfileMetadata(
            id="prof_1",
            display_name="Default Profile",
            created_at=datetime.now().isoformat(),
            description="Migrated from original data" if needs_migration else "Default profile"
        )

        registry_data = {
            "version": "1.0.0",
            "active_profile_id": "prof_1",
            "profiles": [default_profile.to_dict()]
        }

        # Create profile directory structure
        self._create_profile_directories("prof_1")

        if needs_migration:
            self._migrate_existing_data("prof_1")
        else:
            # Create empty registry
            self._create_empty_registry("prof_1")

        # Save profile metadata
        self._save_profile_metadata("prof_1", default_profile)

        # Save profiles registry
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)

    def _create_profile_directories(self, profile_id: str) -> None:
        """Create all subdirectories for a profile."""
        profile_dir = self.profiles_dir / profile_id
        for subdir in self.PROFILE_SUBDIRS:
            (profile_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _migrate_existing_data(self, profile_id: str) -> None:
        """Migrate existing data to the specified profile."""
        profile_dir = self.profiles_dir / profile_id

        # Migration mappings: (source relative, destination relative to profile)
        migrations = [
            ("app/data/object_registry.json", "app_data/object_registry.json"),
            ("app/data/thumbnails", "app_data/thumbnails"),
            ("app/data/collected_images", "app_data/collected_images"),
            ("app/data/reference_images", "app_data/reference_images"),
            ("app/data/tasks", "app_data/tasks"),
            ("datasets/raw_captures", "datasets/raw_captures"),
            ("datasets/annotated", "datasets/annotated"),
            ("datasets/backgrounds", "datasets/backgrounds"),
            ("models/finetuned", "models/finetuned"),
        ]

        for src_rel, dst_rel in migrations:
            src = self.project_root / src_rel
            dst = profile_dir / dst_rel

            if src.exists():
                if src.is_file():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                else:
                    # For directories, copy contents
                    if dst.exists():
                        shutil.rmtree(dst)
                    if any(src.iterdir()):  # Only copy if not empty
                        shutil.copytree(src, dst)
                    else:
                        dst.mkdir(parents=True, exist_ok=True)

    def _create_empty_registry(self, profile_id: str) -> None:
        """Create empty object registry for a profile."""
        profile_dir = self.profiles_dir / profile_id
        registry_path = profile_dir / "app_data" / "object_registry.json"

        empty_registry = {
            "version": "1.0.0",
            "updated_at": datetime.now().isoformat(),
            "categories": ["Food", "Drink", "Kitchen Item", "Task Item", "Bag", "Other"],
            "objects": []
        }

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(empty_registry, f, indent=2, ensure_ascii=False)

    def _load_registry(self) -> dict:
        """Load profiles registry."""
        with open(self.registry_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_registry(self, data: dict) -> None:
        """Save profiles registry."""
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _save_profile_metadata(self, profile_id: str, metadata: ProfileMetadata) -> None:
        """Save profile metadata file."""
        profile_dir = self.profiles_dir / profile_id
        metadata_path = profile_dir / self.PROFILE_METADATA_FILE

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

    # ===== Public API =====

    def get_active_profile_id(self) -> str:
        """Get the currently active profile ID."""
        data = self._load_registry()
        return data.get("active_profile_id", "prof_1")

    def get_active_profile(self) -> ProfileMetadata:
        """Get the currently active profile metadata."""
        data = self._load_registry()
        active_id = data.get("active_profile_id", "prof_1")
        for profile_data in data.get("profiles", []):
            if profile_data["id"] == active_id:
                return ProfileMetadata.from_dict(profile_data)
        raise ValueError(f"Active profile not found: {active_id}")

    def get_all_profiles(self) -> List[ProfileMetadata]:
        """Get all profiles."""
        data = self._load_registry()
        return [ProfileMetadata.from_dict(p) for p in data.get("profiles", [])]

    def get_profile(self, profile_id: str) -> Optional[ProfileMetadata]:
        """Get a specific profile by ID."""
        data = self._load_registry()
        for profile_data in data.get("profiles", []):
            if profile_data["id"] == profile_id:
                return ProfileMetadata.from_dict(profile_data)
        return None

    def set_active_profile(self, profile_id: str) -> None:
        """Switch to a different profile."""
        data = self._load_registry()

        # Verify profile exists
        profile_ids = [p["id"] for p in data.get("profiles", [])]
        if profile_id not in profile_ids:
            raise ValueError(f"Profile not found: {profile_id}")

        # Update last_accessed on previous profile
        old_active_id = data.get("active_profile_id")
        for profile_data in data.get("profiles", []):
            if profile_data["id"] == old_active_id:
                profile_data["last_accessed"] = datetime.now().isoformat()
                self._save_profile_metadata(
                    old_active_id,
                    ProfileMetadata.from_dict(profile_data)
                )

        data["active_profile_id"] = profile_id
        self._save_registry(data)

    def create_profile(self, display_name: str, description: str = "") -> ProfileMetadata:
        """Create a new empty profile."""
        data = self._load_registry()

        # Generate new profile ID
        existing_ids = [p["id"] for p in data.get("profiles", [])]
        new_id = self._generate_profile_id(existing_ids)

        # Create profile metadata
        new_profile = ProfileMetadata(
            id=new_id,
            display_name=display_name,
            created_at=datetime.now().isoformat(),
            description=description
        )

        # Create directory structure
        self._create_profile_directories(new_id)
        self._create_empty_registry(new_id)
        self._save_profile_metadata(new_id, new_profile)

        # Add to registry
        data["profiles"].append(new_profile.to_dict())
        self._save_registry(data)

        return new_profile

    def _generate_profile_id(self, existing_ids: List[str]) -> str:
        """Generate a unique profile ID."""
        i = 1
        while True:
            new_id = f"prof_{i}"
            if new_id not in existing_ids:
                return new_id
            i += 1

    def update_profile(
        self,
        profile_id: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None
    ) -> ProfileMetadata:
        """Update profile metadata (name, description)."""
        data = self._load_registry()

        for profile_data in data.get("profiles", []):
            if profile_data["id"] == profile_id:
                if display_name is not None:
                    profile_data["display_name"] = display_name
                if description is not None:
                    profile_data["description"] = description

                updated_profile = ProfileMetadata.from_dict(profile_data)
                self._save_profile_metadata(profile_id, updated_profile)
                self._save_registry(data)
                return updated_profile

        raise ValueError(f"Profile not found: {profile_id}")

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile and all its data."""
        data = self._load_registry()

        # Cannot delete the last profile
        if len(data.get("profiles", [])) <= 1:
            raise ValueError("Cannot delete the last profile")

        # Cannot delete active profile
        if profile_id == data.get("active_profile_id"):
            raise ValueError("Cannot delete the active profile. Switch to another profile first.")

        # Remove from registry
        data["profiles"] = [p for p in data["profiles"] if p["id"] != profile_id]
        self._save_registry(data)

        # Delete profile directory
        profile_dir = self.profiles_dir / profile_id
        if profile_dir.exists():
            shutil.rmtree(profile_dir)

        return True

    def get_profile_path(self, profile_id: Optional[str] = None) -> Path:
        """Get the root path for a profile."""
        if profile_id is None:
            profile_id = self.get_active_profile_id()
        return self.profiles_dir / profile_id

    def get_active_profile_path(self) -> Path:
        """Get the root path for the active profile."""
        return self.get_profile_path(self.get_active_profile_id())

    def duplicate_profile(self, source_profile_id: str, new_display_name: str) -> ProfileMetadata:
        """Duplicate an existing profile with all its data."""
        source_path = self.get_profile_path(source_profile_id)
        if not source_path.exists():
            raise ValueError(f"Source profile not found: {source_profile_id}")

        data = self._load_registry()
        new_id = self._generate_profile_id([p["id"] for p in data["profiles"]])
        new_path = self.profiles_dir / new_id

        # Copy entire profile directory
        shutil.copytree(source_path, new_path)

        # Create new metadata
        new_profile = ProfileMetadata(
            id=new_id,
            display_name=new_display_name,
            created_at=datetime.now().isoformat(),
            description=f"Duplicated from {source_profile_id}"
        )

        self._save_profile_metadata(new_id, new_profile)
        data["profiles"].append(new_profile.to_dict())
        self._save_registry(data)

        return new_profile
