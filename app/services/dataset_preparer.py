"""
Dataset Preparer Service

Provides functionality to prepare YOLO training datasets from
raw captures and annotation labels within the Streamlit app.
"""

import re
import shutil
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.path_coordinator import PathCoordinator
import yaml


def _extract_timestamp(filename: str) -> datetime | None:
    """
    Extract timestamp from filename.

    Expected format: {class_name}_{YYYYMMDD}_{HHMMSS}_{milliseconds}.{ext}
    Example: apple_20251211_123456_123.jpg

    Args:
        filename: Image filename

    Returns:
        datetime object if timestamp found, None otherwise
    """
    # Pattern: something_YYYYMMDD_HHMMSS_mmm.ext
    pattern = r".*_(\d{8})_(\d{6})_(\d{3})\.\w+$"
    match = re.match(pattern, filename)
    if match:
        date_str, time_str, ms_str = match.groups()
        try:
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            return None
    return None


def _group_by_timestamp(
    pairs: list[tuple[Path, Path, str]], interval_sec: float
) -> list[list[tuple[Path, Path, str]]]:
    """
    Group image-label pairs by timestamp proximity.

    Groups images that were captured within interval_sec of each other,
    ensuring continuous frames from burst captures stay together.

    Args:
        pairs: List of (image_path, label_path, class_name) tuples
        interval_sec: Maximum seconds between frames in same group

    Returns:
        List of groups, each group is a list of pairs
    """
    if not pairs:
        return []

    # Try to extract timestamps
    pairs_with_ts = []
    pairs_without_ts = []

    for pair in pairs:
        ts = _extract_timestamp(pair[0].name)
        if ts:
            pairs_with_ts.append((ts, pair))
        else:
            pairs_without_ts.append(pair)

    groups = []

    # Group pairs with timestamps
    if pairs_with_ts:
        # Sort by timestamp
        pairs_with_ts.sort(key=lambda x: x[0])

        current_group = []
        last_ts = None

        for ts, pair in pairs_with_ts:
            if last_ts and (ts - last_ts).total_seconds() > interval_sec:
                if current_group:
                    groups.append(current_group)
                current_group = []

            current_group.append(pair)
            last_ts = ts

        if current_group:
            groups.append(current_group)

    # Add pairs without timestamps as individual groups
    for pair in pairs_without_ts:
        groups.append([pair])

    return groups


@dataclass
class ClassInfo:
    """Information about a single class."""
    name: str
    image_count: int
    label_count: int
    matched_count: int
    images_dir: Path
    labels_dir: Path | None

    @property
    def match_ratio(self) -> float:
        """Ratio of images that have matching labels."""
        if self.image_count == 0:
            return 0.0
        return self.matched_count / self.image_count

    @property
    def is_ready(self) -> bool:
        """Check if class has sufficient data for training."""
        return self.matched_count >= 10  # Minimum 10 samples

    @property
    def status(self) -> str:
        """Get status string."""
        if self.matched_count == 0:
            return "no_data"
        elif self.matched_count < 10:
            return "insufficient"
        elif self.match_ratio < 1.0:
            return "partial"
        else:
            return "ready"


@dataclass
class DatasetResult:
    """Result of dataset preparation."""
    success: bool
    output_dir: Path | None
    train_count: int
    val_count: int
    class_names: list[str]
    error_message: str | None = None


class DatasetPreparer:
    """
    Service for preparing YOLO training datasets.

    Combines raw capture images with annotation labels,
    creates train/val splits, and generates data.yaml.
    """

    def __init__(self, path_coordinator: "PathCoordinator") -> None:
        """
        Initialize DatasetPreparer.

        Args:
            path_coordinator: PathCoordinator instance for profile-aware paths
        """
        self.path_coordinator = path_coordinator

    def get_available_classes(self) -> list[ClassInfo]:
        """
        Get list of available classes with their statistics.

        Returns:
            List of ClassInfo objects
        """
        classes = []

        raw_captures_dir = self.path_coordinator.get_path("raw_captures_dir")
        annotated_dir = self.path_coordinator.get_path("annotated_dir")

        if not raw_captures_dir.exists():
            return classes

        # Find all class directories in raw_captures
        for class_dir in sorted(raw_captures_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            # Count images
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            image_count = len(images)

            # Count labels
            labels_dir = annotated_dir / class_name / "labels"
            if labels_dir.exists():
                labels = list(labels_dir.glob("*.txt"))
                label_count = len(labels)
            else:
                labels_dir = annotated_dir / class_name
                label_count = 0

            # Count matching pairs
            matched_count = 0
            for img in images:
                label_path = labels_dir / f"{img.stem}.txt"
                if label_path.exists():
                    matched_count += 1

            classes.append(ClassInfo(
                name=class_name,
                image_count=image_count,
                label_count=label_count,
                matched_count=matched_count,
                images_dir=class_dir,
                labels_dir=labels_dir,
            ))

        return classes

    def get_ready_classes(self) -> list[ClassInfo]:
        """Get only classes that are ready for training."""
        return [c for c in self.get_available_classes() if c.is_ready]

    def prepare_dataset(
        self,
        class_names: list[str],
        output_name: str,
        val_ratio: float = 0.2,
        seed: int = 42,
        group_continuous_frames: bool = True,
        group_interval_sec: float = 2.0,
    ) -> DatasetResult:
        """
        Prepare a YOLO training dataset from selected classes.

        Args:
            class_names: List of class names to include
            output_name: Name for the output dataset directory
            val_ratio: Ratio for validation split (0.0-1.0)
            seed: Random seed for reproducibility
            group_continuous_frames: If True, group frames by timestamp to prevent
                data leakage from burst captures (default: True)
            group_interval_sec: Maximum seconds between frames in same group
                (default: 2.0 seconds)

        Returns:
            DatasetResult with success status and statistics
        """
        random.seed(seed)

        try:
            # Get class info for selected classes
            all_classes = {c.name: c for c in self.get_available_classes()}

            selected_classes = []
            for name in class_names:
                if name not in all_classes:
                    return DatasetResult(
                        success=False,
                        output_dir=None,
                        train_count=0,
                        val_count=0,
                        class_names=[],
                        error_message=f"Class not found: {name}"
                    )
                selected_classes.append(all_classes[name])

            # Collect all matching pairs
            all_pairs = []
            for cls in selected_classes:
                pairs = self._find_matching_pairs(cls)
                all_pairs.extend(pairs)

            if not all_pairs:
                return DatasetResult(
                    success=False,
                    output_dir=None,
                    train_count=0,
                    val_count=0,
                    class_names=[],
                    error_message="No image-label pairs found"
                )

            # Create output directory
            annotated_dir = self.path_coordinator.get_path("annotated_dir")
            output_dir = annotated_dir / output_name

            if output_dir.exists():
                shutil.rmtree(output_dir)

            # Create directory structure
            (output_dir / "images" / "train").mkdir(parents=True)
            (output_dir / "images" / "val").mkdir(parents=True)
            (output_dir / "labels" / "train").mkdir(parents=True)
            (output_dir / "labels" / "val").mkdir(parents=True)

            # Create class mapping (sorted for consistency)
            sorted_names = sorted(class_names)
            class_mapping = {name: idx for idx, name in enumerate(sorted_names)}

            # Split dataset (with or without grouping)
            if group_continuous_frames:
                # Group frames by timestamp
                groups = _group_by_timestamp(all_pairs, group_interval_sec)
                random.shuffle(groups)

                # Split at group level
                val_group_count = max(1, int(len(groups) * val_ratio))
                train_groups = groups[:-val_group_count] if val_group_count < len(groups) else []
                val_groups = groups[-val_group_count:]

                # Flatten back to pairs
                train_pairs = [pair for group in train_groups for pair in group]
                val_pairs = [pair for group in val_groups for pair in group]
            else:
                # Traditional random split
                random.shuffle(all_pairs)
                val_count = int(len(all_pairs) * val_ratio)

                train_pairs = all_pairs[:-val_count] if val_count < len(all_pairs) else []
                val_pairs = all_pairs[-val_count:] if val_count > 0 else []

            # Process pairs
            self._process_pairs(train_pairs, output_dir, "train", class_mapping)
            self._process_pairs(val_pairs, output_dir, "val", class_mapping)

            # Generate data.yaml
            data_yaml = {
                'path': str(output_dir.resolve()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': len(sorted_names),
                'names': sorted_names,
            }

            yaml_path = output_dir / "data.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

            return DatasetResult(
                success=True,
                output_dir=output_dir,
                train_count=len(train_pairs),
                val_count=len(val_pairs),
                class_names=sorted_names,
            )

        except Exception as e:
            return DatasetResult(
                success=False,
                output_dir=None,
                train_count=0,
                val_count=0,
                class_names=[],
                error_message=str(e)
            )

    def _find_matching_pairs(
        self,
        cls: ClassInfo,
    ) -> list[tuple[Path, Path, str]]:
        """Find matching image-label pairs for a class."""
        pairs = []

        images = list(cls.images_dir.glob("*.jpg")) + list(cls.images_dir.glob("*.png"))

        for image_path in images:
            label_path = cls.labels_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path, cls.name))

        return pairs

    def _process_pairs(
        self,
        pairs: list[tuple[Path, Path, str]],
        output_dir: Path,
        split: str,
        class_mapping: dict[str, int],
    ) -> None:
        """Process pairs and copy to output directory."""
        for image_path, label_path, class_name in pairs:
            # Generate unique filename
            new_name = f"{class_name}_{image_path.stem}"

            # Copy image
            image_dest = output_dir / "images" / split / f"{new_name}{image_path.suffix}"
            shutil.copy2(image_path, image_dest)

            # Remap and write label
            new_class_id = class_mapping[class_name]
            remapped_lines = self._remap_labels(label_path, new_class_id)

            label_dest = output_dir / "labels" / split / f"{new_name}.txt"
            with open(label_dest, 'w') as f:
                f.write('\n'.join(remapped_lines))

    def _remap_labels(self, label_path: Path, new_class_id: int) -> list[str]:
        """Read label file and remap class IDs."""
        lines = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    parts[0] = str(new_class_id)
                    lines.append(' '.join(parts))
        return lines
