"""
Annotation Utilities for YOLO Format

Provides conversion functions, validation, and dataset management utilities
for YOLO format annotations.
"""

import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class AnnotationResult:
    """Result of an annotation operation."""

    total_images: int = 0
    successful: int = 0
    failed: int = 0
    failed_paths: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_images == 0:
            return 0.0
        return (self.successful / self.total_images) * 100

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"Annotation Result:\n"
            f"  Total: {self.total_images}\n"
            f"  Successful: {self.successful}\n"
            f"  Failed: {self.failed}\n"
            f"  Success Rate: {self.success_rate:.1f}%"
        )


def bbox_to_yolo(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert absolute bounding box coordinates to YOLO format.

    Args:
        x_min: Left edge of bounding box (pixels)
        y_min: Top edge of bounding box (pixels)
        x_max: Right edge of bounding box (pixels)
        y_max: Bottom edge of bounding box (pixels)
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    # Clamp values to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return (x_center, y_center, width, height)


def yolo_to_bbox(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO format to absolute bounding box coordinates.

    Args:
        x_center: Center X normalized [0, 1]
        y_center: Center Y normalized [0, 1]
        width: Width normalized [0, 1]
        height: Height normalized [0, 1]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Tuple of (x_min, y_min, x_max, y_max) in pixels
    """
    w = width * img_width
    h = height * img_height
    x_min = int((x_center * img_width) - (w / 2))
    y_min = int((y_center * img_height) - (h / 2))
    x_max = int((x_center * img_width) + (w / 2))
    y_max = int((y_center * img_height) + (h / 2))

    # Clamp to image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    return (x_min, y_min, x_max, y_max)


def write_yolo_label(
    label_path: str,
    class_id: int,
    bbox: Tuple[float, float, float, float],
    append: bool = False,
) -> None:
    """
    Write annotation to YOLO label file.

    Args:
        label_path: Path to output .txt file
        class_id: Class index (0-indexed)
        bbox: Tuple of (x_center, y_center, width, height) normalized
        append: If True, append to existing file; otherwise overwrite
    """
    mode = "a" if append else "w"
    x_c, y_c, w, h = bbox

    with open(label_path, mode) as f:
        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")


def read_yolo_label(
    label_path: str,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Read YOLO label file.

    Args:
        label_path: Path to .txt label file

    Returns:
        List of tuples (class_id, x_center, y_center, width, height)
    """
    labels = []
    path = Path(label_path)

    if not path.exists():
        return labels

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                labels.append((class_id, x_c, y_c, w, h))

    return labels


def validate_yolo_annotation(label_path: str) -> Tuple[bool, List[str]]:
    """
    Validate YOLO annotation file format.

    Args:
        label_path: Path to .txt label file

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    path = Path(label_path)

    if not path.exists():
        return (False, [f"File not found: {label_path}"])

    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            # Check number of fields
            if len(parts) < 5:
                errors.append(f"Line {line_num}: Expected 5 fields, got {len(parts)}")
                continue

            # Check class_id is integer
            try:
                class_id = int(parts[0])
                if class_id < 0:
                    errors.append(f"Line {line_num}: Negative class_id: {class_id}")
            except ValueError:
                errors.append(f"Line {line_num}: Invalid class_id: {parts[0]}")
                continue

            # Check bbox values are valid floats in [0, 1]
            for i, name in enumerate(["x_center", "y_center", "width", "height"], 1):
                try:
                    val = float(parts[i])
                    if not (0.0 <= val <= 1.0):
                        errors.append(
                            f"Line {line_num}: {name}={val} out of range [0, 1]"
                        )
                except ValueError:
                    errors.append(f"Line {line_num}: Invalid {name}: {parts[i]}")

    return (len(errors) == 0, errors)


def create_dataset_yaml(
    output_path: str,
    train_path: str,
    val_path: str,
    class_names: List[str],
    test_path: Optional[str] = None,
) -> None:
    """
    Generate YOLO dataset.yaml configuration file.

    Args:
        output_path: Path for output data.yaml file
        train_path: Path to training images directory
        val_path: Path to validation images directory
        class_names: List of class names (index = class_id)
        test_path: Optional path to test images directory
    """
    config = {
        "path": str(Path(output_path).parent.absolute()),
        "train": train_path,
        "val": val_path,
        "names": {i: name for i, name in enumerate(class_names)},
    }

    if test_path:
        config["test"] = test_path

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.85,
    seed: int = 42,
    copy_files: bool = True,
) -> Dict[str, int]:
    """
    Split dataset into train and validation sets.

    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO label files
        output_dir: Output directory for split dataset
        train_ratio: Ratio of training data (default: 0.85)
        seed: Random seed for reproducibility
        copy_files: If True, copy files; if False, create symlinks

    Returns:
        Dictionary with counts: {"train": n, "val": m}
    """
    random.seed(seed)

    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    output_path = Path(output_dir)

    # Create output directories
    train_images = output_path / "images" / "train"
    train_labels = output_path / "labels" / "train"
    val_images = output_path / "images" / "val"
    val_labels = output_path / "labels" / "val"

    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Find all images with corresponding labels
    valid_pairs = []
    for img_path in images_path.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        label_path = labels_path / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))

    # Shuffle and split
    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * train_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    # Copy/link files
    def transfer_files(pairs, img_dest, lbl_dest):
        for img_src, lbl_src in pairs:
            img_dst = img_dest / img_src.name
            lbl_dst = lbl_dest / lbl_src.name

            if copy_files:
                shutil.copy2(img_src, img_dst)
                shutil.copy2(lbl_src, lbl_dst)
            else:
                if not img_dst.exists():
                    img_dst.symlink_to(img_src.absolute())
                if not lbl_dst.exists():
                    lbl_dst.symlink_to(lbl_src.absolute())

    transfer_files(train_pairs, train_images, train_labels)
    transfer_files(val_pairs, val_images, val_labels)

    return {"train": len(train_pairs), "val": len(val_pairs)}


def load_class_config(config_path: str) -> Dict:
    """
    Load object classes configuration from JSON file.

    Args:
        config_path: Path to object_classes.json

    Returns:
        Dictionary containing class configuration
    """
    with open(config_path, "r") as f:
        return json.load(f)


def get_class_names(config: Dict) -> List[str]:
    """
    Extract ordered list of class names from config.

    Args:
        config: Loaded class configuration dictionary

    Returns:
        List of class names ordered by class_id
    """
    objects = config.get("objects", [])
    # Sort by class_id and extract names
    sorted_objects = sorted(objects, key=lambda x: x["class_id"])
    return [obj["class_name"] for obj in sorted_objects]


def get_class_id_map(config: Dict) -> Dict[str, int]:
    """
    Create mapping from class name to class ID.

    Args:
        config: Loaded class configuration dictionary

    Returns:
        Dictionary mapping class_name -> class_id
    """
    objects = config.get("objects", [])
    return {obj["class_name"]: obj["class_id"] for obj in objects}


def update_collected_samples(
    config_path: str,
    class_name: str,
    count: int,
) -> None:
    """
    Update collected_samples count in config file.

    Args:
        config_path: Path to object_classes.json
        class_name: Name of the class to update
        count: New sample count
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    for obj in config.get("objects", []):
        if obj["class_name"] == class_name:
            obj["collected_samples"] = count
            from datetime import datetime
            obj["last_updated"] = datetime.now().isoformat()
            break

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Test functions
    print("Testing bbox_to_yolo...")
    yolo = bbox_to_yolo(100, 100, 300, 400, 640, 480)
    print(f"  Input: (100, 100, 300, 400) on 640x480")
    print(f"  Output: {yolo}")

    print("\nTesting yolo_to_bbox...")
    bbox = yolo_to_bbox(*yolo, 640, 480)
    print(f"  Input: {yolo} on 640x480")
    print(f"  Output: {bbox}")

    print("\nTests complete.")
