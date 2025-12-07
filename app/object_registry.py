"""
Object Registry - Data Model and Operations

Manages object definitions, reference images, and collection status.
Supports profile-based data isolation via PathCoordinator.
"""

import json
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from services.path_coordinator import PathCoordinator


@dataclass
class ObjectVersion:
    """A version/variant of an object."""
    version: int
    image_path: Optional[str] = None
    source_link: Optional[str] = None


@dataclass
class ObjectProperties:
    """Physical and handling properties of an object."""
    is_heavy: bool = False
    is_tiny: bool = False
    has_liquid: bool = False
    size_cm: Optional[str] = None
    grasp_strategy: Optional[str] = None


@dataclass
class RegisteredObject:
    """A registered object in the system."""
    id: int
    name: str
    display_name: str
    category: str
    versions: List[ObjectVersion] = field(default_factory=list)
    properties: ObjectProperties = field(default_factory=ObjectProperties)
    remarks: str = ""
    target_samples: int = 100
    collected_samples: int = 0
    last_updated: Optional[str] = None
    thumbnail_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "versions": [{"version": v.version, "image_path": v.image_path, "source_link": v.source_link} for v in self.versions],
            "properties": asdict(self.properties),
            "remarks": self.remarks,
            "target_samples": self.target_samples,
            "collected_samples": self.collected_samples,
            "last_updated": self.last_updated,
            "thumbnail_path": self.thumbnail_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegisteredObject":
        versions = [ObjectVersion(**v) for v in data.get("versions", [])]
        properties = ObjectProperties(**data.get("properties", {}))
        return cls(
            id=data["id"],
            name=data["name"],
            display_name=data.get("display_name", data["name"]),
            category=data["category"],
            versions=versions,
            properties=properties,
            remarks=data.get("remarks", ""),
            target_samples=data.get("target_samples", 100),
            collected_samples=data.get("collected_samples", 0),
            last_updated=data.get("last_updated"),
            thumbnail_path=data.get("thumbnail_path"),
        )


class ObjectRegistry:
    """
    Central registry for managing objects, reference images, and collection status.

    Supports profile-based data isolation via PathCoordinator.
    """

    def __init__(self, path_coordinator: Optional["PathCoordinator"] = None):
        """
        Initialize object registry.

        Args:
            path_coordinator: PathCoordinator instance for profile-aware paths.
                            If None, creates a new one (uses active profile).
        """
        # Use path coordinator for profile-aware paths
        if path_coordinator is None:
            from services.path_coordinator import PathCoordinator
            self._path_coordinator = PathCoordinator()
        else:
            self._path_coordinator = path_coordinator

        # Get paths from coordinator (profile-aware)
        self.data_dir = self._path_coordinator.get_path("app_data_dir")
        self.registry_file = self._path_coordinator.get_path("app_registry_file")
        self.reference_images_dir = self._path_coordinator.get_path("app_reference_dir")
        self.collected_images_dir = self._path_coordinator.get_path("app_collected_dir")
        self.thumbnails_dir = self._path_coordinator.get_path("app_thumbnails_dir")

        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reference_images_dir.mkdir(exist_ok=True)
        self.collected_images_dir.mkdir(exist_ok=True)
        self.thumbnails_dir.mkdir(exist_ok=True)

        # Load or initialize registry
        self.objects: Dict[int, RegisteredObject] = {}
        self.categories: List[str] = []
        self._load()

    def _load(self) -> None:
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.categories = data.get("categories", [])
                for obj_data in data.get("objects", []):
                    obj = RegisteredObject.from_dict(obj_data)
                    self.objects[obj.id] = obj
        else:
            # Initialize with default categories
            self.categories = ["Food", "Drink", "Kitchen Item", "Task Item", "Bag", "Other"]
            self._save()

    def _save(self) -> None:
        """Save registry to file."""
        data = {
            "version": "1.0.0",
            "updated_at": datetime.now().isoformat(),
            "categories": self.categories,
            "objects": [obj.to_dict() for obj in self.objects.values()],
        }
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_object(self, obj: RegisteredObject) -> None:
        """Add or update an object."""
        obj.last_updated = datetime.now().isoformat()
        self.objects[obj.id] = obj

        # Create directory for collected images
        obj_dir = self.collected_images_dir / obj.name
        obj_dir.mkdir(exist_ok=True)

        self._save()

    def remove_object(self, obj_id: int) -> bool:
        """Remove an object."""
        if obj_id in self.objects:
            del self.objects[obj_id]
            self._save()
            return True
        return False

    def get_object(self, obj_id: int) -> Optional[RegisteredObject]:
        """Get an object by ID."""
        return self.objects.get(obj_id)

    def get_object_by_name(self, name: str) -> Optional[RegisteredObject]:
        """Get an object by name."""
        for obj in self.objects.values():
            if obj.name == name:
                return obj
        return None

    def get_all_objects(self) -> List[RegisteredObject]:
        """Get all objects sorted by ID."""
        return sorted(self.objects.values(), key=lambda x: x.id)

    def get_objects_by_category(self, category: str) -> List[RegisteredObject]:
        """Get objects filtered by category."""
        return [obj for obj in self.objects.values() if obj.category == category]

    def get_next_id(self) -> int:
        """Get next available ID."""
        if not self.objects:
            return 1
        return max(self.objects.keys()) + 1

    def add_category(self, category: str) -> None:
        """Add a new category."""
        if category not in self.categories:
            self.categories.append(category)
            self._save()

    def update_object(self, obj_id: int, updates: dict) -> bool:
        """Update object fields."""
        obj = self.get_object(obj_id)
        if not obj:
            return False

        old_name = obj.name

        # Apply updates to simple fields
        for key in ['display_name', 'category', 'remarks', 'target_samples']:
            if key in updates:
                setattr(obj, key, updates[key])

        # Handle name change (requires renaming directories)
        if 'name' in updates and updates['name'] != old_name:
            new_name = updates['name']
            # Rename collected images directory
            old_collected_dir = self.collected_images_dir / old_name
            new_collected_dir = self.collected_images_dir / new_name
            if old_collected_dir.exists():
                old_collected_dir.rename(new_collected_dir)
            # Rename reference images directory
            old_ref_dir = self.reference_images_dir / old_name
            new_ref_dir = self.reference_images_dir / new_name
            if old_ref_dir.exists():
                old_ref_dir.rename(new_ref_dir)
            obj.name = new_name

        # Handle properties update
        if 'properties' in updates:
            props = updates['properties']
            if isinstance(props, dict):
                obj.properties = ObjectProperties(**props)
            elif isinstance(props, ObjectProperties):
                obj.properties = props

        obj.last_updated = datetime.now().isoformat()
        self._save()
        return True

    # Thumbnail Management
    def set_thumbnail(self, obj_id: int, image_path: str) -> Optional[str]:
        """Set thumbnail image for an object."""
        obj = self.get_object(obj_id)
        if not obj:
            return None

        # Copy image to thumbnails directory
        src = Path(image_path)
        dst = self.thumbnails_dir / f"{obj.name}{src.suffix}"
        shutil.copy2(src, dst)

        # Update object with relative path from data_dir
        obj.thumbnail_path = str(dst.relative_to(self.data_dir))
        obj.last_updated = datetime.now().isoformat()
        self._save()
        return str(dst)

    def save_thumbnail_from_bytes(self, obj_id: int, image_data: bytes, extension: str = ".jpg") -> Optional[str]:
        """Save thumbnail from bytes (for clipboard paste)."""
        obj = self.get_object(obj_id)
        if not obj:
            return None

        dst = self.thumbnails_dir / f"{obj.name}{extension}"
        with open(dst, "wb") as f:
            f.write(image_data)

        obj.thumbnail_path = str(dst.relative_to(self.data_dir))
        obj.last_updated = datetime.now().isoformat()
        self._save()
        return str(dst)

    def get_thumbnail_path(self, obj_id: int) -> Optional[str]:
        """Get full path to thumbnail image."""
        obj = self.get_object(obj_id)
        if not obj or not obj.thumbnail_path:
            return None
        full_path = self.data_dir / obj.thumbnail_path
        if full_path.exists():
            return str(full_path)
        return None

    # Reference Image Management
    def add_reference_image(self, obj_id: int, image_path: str, version: int = 1) -> Optional[str]:
        """Add a reference image for an object."""
        obj = self.get_object(obj_id)
        if not obj:
            return None

        # Create object reference directory
        obj_ref_dir = self.reference_images_dir / obj.name
        obj_ref_dir.mkdir(exist_ok=True)

        # Copy image
        src = Path(image_path)
        dst = obj_ref_dir / f"v{version}{src.suffix}"
        shutil.copy2(src, dst)

        # Update object versions
        existing_version = next((v for v in obj.versions if v.version == version), None)
        if existing_version:
            existing_version.image_path = str(dst.relative_to(self.data_dir))
        else:
            obj.versions.append(ObjectVersion(
                version=version,
                image_path=str(dst.relative_to(self.data_dir))
            ))

        obj.last_updated = datetime.now().isoformat()
        self._save()
        return str(dst)

    def get_reference_images(self, obj_id: int) -> List[str]:
        """Get all reference image paths for an object."""
        obj = self.get_object(obj_id)
        if not obj:
            return []

        images = []
        for v in obj.versions:
            if v.image_path:
                full_path = self.data_dir / v.image_path
                if full_path.exists():
                    images.append(str(full_path))
        return images

    # Collection Management
    def get_collected_images_dir(self, obj_name: str) -> Path:
        """Get directory for collected images of an object."""
        obj_dir = self.collected_images_dir / obj_name
        obj_dir.mkdir(exist_ok=True)
        return obj_dir

    def add_collected_image(self, obj_id: int, image_path: str) -> Optional[str]:
        """Add a collected image for an object."""
        obj = self.get_object(obj_id)
        if not obj:
            return None

        obj_dir = self.get_collected_images_dir(obj.name)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        src = Path(image_path)
        dst = obj_dir / f"{obj.name}_{timestamp}{src.suffix}"

        shutil.copy2(src, dst)
        self.update_collection_count(obj_id)
        return str(dst)

    def save_collected_image(self, obj_id: int, image_data: bytes, extension: str = ".jpg") -> Optional[str]:
        """Save collected image from bytes."""
        obj = self.get_object(obj_id)
        if not obj:
            return None

        obj_dir = self.get_collected_images_dir(obj.name)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dst = obj_dir / f"{obj.name}_{timestamp}{extension}"

        with open(dst, "wb") as f:
            f.write(image_data)

        self.update_collection_count(obj_id)
        return str(dst)

    def get_collected_images(self, obj_id: int) -> List[str]:
        """Get all collected image paths for an object."""
        obj = self.get_object(obj_id)
        if not obj:
            return []

        obj_dir = self.collected_images_dir / obj.name
        if not obj_dir.exists():
            return []

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        images = []
        for f in sorted(obj_dir.iterdir()):
            if f.suffix.lower() in extensions:
                images.append(str(f))
        return images

    def update_collection_count(self, obj_id: int) -> int:
        """Update and return collection count for an object."""
        obj = self.get_object(obj_id)
        if not obj:
            return 0

        images = self.get_collected_images(obj_id)
        obj.collected_samples = len(images)
        obj.last_updated = datetime.now().isoformat()
        self._save()
        return obj.collected_samples

    def update_all_collection_counts(self) -> None:
        """Update collection counts for all objects."""
        for obj_id in self.objects.keys():
            self.update_collection_count(obj_id)

    # Statistics
    def get_collection_stats(self) -> dict:
        """Get overall collection statistics."""
        total_target = sum(obj.target_samples for obj in self.objects.values())
        total_collected = sum(obj.collected_samples for obj in self.objects.values())

        by_category = {}
        for obj in self.objects.values():
            if obj.category not in by_category:
                by_category[obj.category] = {"target": 0, "collected": 0, "objects": 0}
            by_category[obj.category]["target"] += obj.target_samples
            by_category[obj.category]["collected"] += obj.collected_samples
            by_category[obj.category]["objects"] += 1

        ready_objects = sum(1 for obj in self.objects.values()
                          if obj.collected_samples >= obj.target_samples * 0.5)

        return {
            "total_objects": len(self.objects),
            "total_target": total_target,
            "total_collected": total_collected,
            "progress_percent": (total_collected / total_target * 100) if total_target > 0 else 0,
            "by_category": by_category,
            "ready_objects": ready_objects,
        }

    # Export
    def export_to_yolo_config(self, output_path: str) -> str:
        """Export to YOLO-compatible object_classes.json format."""
        objects = self.get_all_objects()

        # Build categories from unique category names
        unique_categories = list(set(obj.category for obj in objects))
        categories = [{"id": i, "name": cat.lower().replace(" ", "_"), "display_name": cat}
                     for i, cat in enumerate(unique_categories)]
        category_map = {cat["display_name"]: cat["id"] for cat in categories}

        yolo_config = {
            "version": "1.0.0",
            "categories": categories,
            "objects": [
                {
                    "class_id": obj.id - 1,  # YOLO uses 0-indexed
                    "class_name": obj.name,
                    "category_id": category_map.get(obj.category, 0),
                    "object_type": "consistent",
                    "target_samples": obj.target_samples,
                    "collected_samples": obj.collected_samples,
                }
                for obj in objects
            ],
            "settings": {
                "default_target_samples": 100,
                "min_samples_for_training": 50,
                "train_val_split": 0.85,
            }
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(yolo_config, f, indent=2, ensure_ascii=False)

        return output_path
