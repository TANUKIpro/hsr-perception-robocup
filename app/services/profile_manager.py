"""
Profile Manager

Manages profile CRUD operations and profile switching.
Each profile contains isolated data for objects, datasets, and trained models.
"""

import json
import shutil
import tempfile
import zipfile
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

    # ===== Import/Export =====

    def _safe_extract_zip(self, zip_file: zipfile.ZipFile, extract_to: Path) -> None:
        """
        Safely extract ZIP file with path traversal protection.

        Prevents ZipSlip vulnerability by validating all extracted paths.

        Args:
            zip_file: Open ZipFile object
            extract_to: Extraction target directory

        Raises:
            ValueError: If ZIP contains path traversal attempt (e.g., ../)
        """
        extract_to = extract_to.resolve()

        for member in zip_file.namelist():
            # Check for absolute paths in ZIP
            if Path(member).is_absolute():
                raise ValueError(
                    f"Security error: ZIP contains absolute path: {member}"
                )

            # Resolve member path
            member_path = (extract_to / member).resolve()

            # Security check: use relative_to for robust path validation
            try:
                member_path.relative_to(extract_to)
            except ValueError:
                raise ValueError(
                    f"Security error: ZIP contains path traversal attempt: {member}"
                )

        # Safe to extract
        zip_file.extractall(extract_to)

    def _resolve_duplicate_name(self, base_name: str) -> str:
        """
        Resolve duplicate profile display name by appending (2), (3), etc.

        Args:
            base_name: Original display name

        Returns:
            Unique display name

        Example:
            "Default Profile" -> "Default Profile (2)" (if duplicate)
        """
        data = self._load_registry()
        existing_names = [p["display_name"] for p in data.get("profiles", [])]

        if base_name not in existing_names:
            return base_name

        # Try "Name (2)", "Name (3)", etc.
        counter = 2
        while True:
            candidate = f"{base_name} ({counter})"
            if candidate not in existing_names:
                return candidate
            counter += 1

    def export_profile(self, profile_id: str, output_path: str) -> str:
        """
        Export a profile to a ZIP file.

        Args:
            profile_id: Profile ID to export
            output_path: Output ZIP file path (including .zip extension)

        Returns:
            str: Absolute path to the generated ZIP file

        Raises:
            ValueError: Profile not found or invalid output path
            PermissionError: No write permission
            OSError: Disk full or other OS error
        """
        # 1. Validation
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile not found: {profile_id}")

        output_path = Path(output_path)
        if not output_path.parent.exists():
            raise ValueError(f"Output directory does not exist: {output_path.parent}")

        # 2. Get profile directory
        profile_dir = self.get_profile_path(profile_id)

        # 3. Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="profile_export_"))

        try:
            # 4. Copy profile data
            export_root = temp_dir / profile_id
            export_root.mkdir()

            # Copy subdirectories
            for subdir in ["app_data", "datasets", "models"]:
                src = profile_dir / subdir
                if src.exists():
                    shutil.copytree(src, export_root / subdir)

            # Copy profile metadata
            profile_meta_file = profile_dir / self.PROFILE_METADATA_FILE
            if profile_meta_file.exists():
                shutil.copy2(profile_meta_file, export_root / self.PROFILE_METADATA_FILE)

            # 5. Create export metadata
            export_meta = {
                "version": "1.0.0",
                "export_date": datetime.now().isoformat(),
                "original_profile_id": profile_id,
                "display_name": profile.display_name,
                "description": profile.description or "",
                "created_at": profile.created_at
            }

            with open(export_root / "export_metadata.json", "w", encoding="utf-8") as f:
                json.dump(export_meta, f, indent=2, ensure_ascii=False)

            # 6. Create ZIP archive
            base_name = str(output_path).replace(".zip", "")
            archive_path = shutil.make_archive(base_name, 'zip', temp_dir)

            return archive_path

        finally:
            # 7. Cleanup
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass  # Cleanup failure shouldn't block main operation

    def export_profile_to_bytes(self, profile_id: str) -> bytes:
        """
        Export a profile to a ZIP file and return as bytes.

        Suitable for browser download via st.download_button().

        Args:
            profile_id: Profile ID to export

        Returns:
            bytes: ZIP file content as bytes

        Raises:
            ValueError: Profile not found
            OSError: Disk full or other OS error
        """
        # 1. Validation
        profile = self.get_profile(profile_id)
        if not profile:
            raise ValueError(f"Profile not found: {profile_id}")

        # 2. Get profile directory
        profile_dir = self.get_profile_path(profile_id)

        # 3. Create temporary directory
        temp_dir = Path(tempfile.mkdtemp(prefix="profile_export_"))

        try:
            # 4. Copy profile data
            export_root = temp_dir / profile_id
            export_root.mkdir()

            # Copy subdirectories
            for subdir in ["app_data", "datasets", "models"]:
                src = profile_dir / subdir
                if src.exists():
                    shutil.copytree(src, export_root / subdir)

            # Copy profile metadata
            profile_meta_file = profile_dir / self.PROFILE_METADATA_FILE
            if profile_meta_file.exists():
                shutil.copy2(profile_meta_file, export_root / self.PROFILE_METADATA_FILE)

            # 5. Create export metadata
            export_meta = {
                "version": "1.0.0",
                "export_date": datetime.now().isoformat(),
                "original_profile_id": profile_id,
                "display_name": profile.display_name,
                "description": profile.description or "",
                "created_at": profile.created_at
            }

            with open(export_root / "export_metadata.json", "w", encoding="utf-8") as f:
                json.dump(export_meta, f, indent=2, ensure_ascii=False)

            # 6. Create ZIP archive in temp directory
            zip_path = temp_dir / "export.zip"
            base_name = str(zip_path).replace(".zip", "")
            archive_path = shutil.make_archive(base_name, 'zip', temp_dir, profile_id)

            # 7. Read ZIP file as bytes
            with open(archive_path, "rb") as f:
                zip_bytes = f.read()

            return zip_bytes

        finally:
            # 8. Cleanup
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass  # Cleanup failure shouldn't block main operation

    def import_profile(self, zip_path: str, display_name: Optional[str] = None) -> ProfileMetadata:
        """
        Import a profile from a ZIP file.

        Args:
            zip_path: Path to the ZIP file to import
            display_name: Custom display name (uses original name if None)

        Returns:
            ProfileMetadata: Metadata for the newly created profile

        Raises:
            FileNotFoundError: ZIP file not found
            zipfile.BadZipFile: Invalid ZIP format
            ValueError: Invalid profile structure or security violation
            OSError: Disk full or other OS error
        """
        zip_path = Path(zip_path)

        # 1. Validate ZIP file
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")

        if not zipfile.is_zipfile(zip_path):
            raise zipfile.BadZipFile(f"Invalid ZIP file: {zip_path}")

        # 2. Check uncompressed size (Zip Bomb protection)
        MAX_UNCOMPRESSED_SIZE = 2 * 1024 * 1024 * 1024  # 2GB limit
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            total_size = sum(info.file_size for info in zip_ref.infolist())
            if total_size > MAX_UNCOMPRESSED_SIZE:
                raise ValueError(
                    f"ZIP file too large: {total_size / (1024**3):.2f} GB "
                    f"(max: {MAX_UNCOMPRESSED_SIZE / (1024**3):.2f} GB)"
                )

        # 3. Extract to temporary directory with security check
        temp_dir = Path(tempfile.mkdtemp(prefix="profile_import_"))

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                self._safe_extract_zip(zip_ref, temp_dir)

            # 3. Validate profile structure
            profile_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            if len(profile_dirs) != 1:
                raise ValueError(
                    f"Invalid profile structure: Expected single profile directory, "
                    f"found {len(profile_dirs)}"
                )

            extracted_profile = profile_dirs[0]

            # Check required files
            export_meta_path = extracted_profile / "export_metadata.json"
            profile_meta_path = extracted_profile / self.PROFILE_METADATA_FILE

            if not export_meta_path.exists():
                raise ValueError("Invalid profile: export_metadata.json not found")

            if not profile_meta_path.exists():
                raise ValueError("Invalid profile: profile.json not found")

            # Load metadata
            with open(export_meta_path, 'r', encoding='utf-8') as f:
                export_meta = json.load(f)

            # 4. Generate new profile ID
            data = self._load_registry()
            new_id = self._generate_profile_id([p["id"] for p in data["profiles"]])

            # 5. Resolve duplicate display name
            if display_name is None:
                display_name = export_meta.get("display_name", "Imported Profile")

            final_display_name = self._resolve_duplicate_name(display_name)

            # 6. Create new profile directory
            new_profile_dir = self.profiles_dir / new_id
            new_profile_dir.mkdir(parents=True, exist_ok=True)

            # 7. Copy data
            for item in extracted_profile.iterdir():
                if item.name == "export_metadata.json":
                    continue  # Skip export metadata

                dest = new_profile_dir / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)

            # 8. Create metadata
            new_profile = ProfileMetadata(
                id=new_id,
                display_name=final_display_name,
                created_at=datetime.now().isoformat(),
                description=export_meta.get("description", "Imported from ZIP")
            )

            # Save profile metadata
            self._save_profile_metadata(new_id, new_profile)

            # Add to registry
            data["profiles"].append(new_profile.to_dict())
            self._save_registry(data)

            return new_profile

        finally:
            # 9. Cleanup
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass  # Cleanup failure shouldn't block main operation
