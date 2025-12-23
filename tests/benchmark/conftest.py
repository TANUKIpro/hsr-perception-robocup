"""
Shared fixtures for benchmark tests.

This module provides common test data and utilities for performance benchmarking.
"""

import pytest
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple


@pytest.fixture
def benchmark_temp_dir(tmp_path: Path) -> Path:
    """
    Provide temporary directory for benchmark tests.

    Returns:
        Path to temporary directory
    """
    return tmp_path


@pytest.fixture
def dummy_background_images(benchmark_temp_dir: Path) -> Path:
    """
    Create minimal dummy background images for benchmarking.

    Creates 5 small background images (100x100 px) to avoid slow I/O.

    Returns:
        Path to backgrounds directory
    """
    backgrounds_dir = benchmark_temp_dir / "backgrounds"
    backgrounds_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 simple background images
    for i in range(5):
        # Create varied backgrounds (different colors/patterns)
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        # Different background colors
        if i == 0:
            img[:] = (180, 180, 180)  # Light gray
        elif i == 1:
            img[:] = (100, 120, 140)  # Bluish gray
        elif i == 2:
            img[:] = (160, 140, 120)  # Brownish
        elif i == 3:
            img[:50, :] = (200, 200, 200)  # Half and half
            img[50:, :] = (100, 100, 100)
        else:
            # Gradient
            for y in range(100):
                img[y, :] = (100 + y, 150, 180 - y)

        cv2.imwrite(str(backgrounds_dir / f"bg_{i:03d}.jpg"), img)

    return backgrounds_dir


@pytest.fixture
def dummy_annotated_objects(benchmark_temp_dir: Path) -> Tuple[Path, list]:
    """
    Create minimal dummy annotated objects for benchmarking.

    Creates object images with corresponding mask files.
    Structure:
        annotated/
            apple/
                images/
                    obj_001.jpg
                masks/
                    obj_001_mask.png
            cup/
                images/
                    obj_001.jpg
                masks/
                    obj_001_mask.png

    Returns:
        Tuple of (annotated_dir, class_names)
    """
    annotated_dir = benchmark_temp_dir / "annotated"
    class_names = ["apple", "cup"]

    for class_name in class_names:
        class_dir = annotated_dir / class_name
        images_dir = class_dir / "images"
        masks_dir = class_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Create 3 objects per class
        for i in range(3):
            # Create object image (100x100 with colored object in center)
            img = np.zeros((100, 100, 3), dtype=np.uint8)

            # Different colors per class
            if class_name == "apple":
                color = (50, 50, 200)  # Reddish (BGR)
            else:
                color = (200, 200, 200)  # White-ish

            # Draw object (circle or rectangle)
            if i % 2 == 0:
                cv2.circle(img, (50, 50), 20, color, -1)
            else:
                cv2.rectangle(img, (30, 30), (70, 70), color, -1)

            # Save image
            img_path = images_dir / f"obj_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)

            # Create corresponding mask (binary mask with same shape as object)
            mask = np.zeros((100, 100), dtype=np.uint8)
            if i % 2 == 0:
                cv2.circle(mask, (50, 50), 20, 255, -1)
            else:
                cv2.rectangle(mask, (30, 30), (70, 70), 255, -1)

            # Save mask
            mask_path = masks_dir / f"obj_{i:03d}_mask.png"
            cv2.imwrite(str(mask_path), mask)

    return annotated_dir, class_names


@pytest.fixture
def dummy_dataset_yaml(benchmark_temp_dir: Path) -> Path:
    """
    Create a dummy data.yaml for YOLO dataset.

    Returns:
        Path to data.yaml file
    """
    import yaml

    dataset_dir = benchmark_temp_dir / "dataset"
    train_images_dir = dataset_dir / "images" / "train"
    train_labels_dir = dataset_dir / "labels" / "train"
    train_images_dir.mkdir(parents=True, exist_ok=True)
    train_labels_dir.mkdir(parents=True, exist_ok=True)

    # Create a few dummy training images
    for i in range(3):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (128, 128, 128)
        img_path = train_images_dir / f"train_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img)

        # Create corresponding label
        label_path = train_labels_dir / f"train_{i:03d}.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    # Create data.yaml
    config = {
        "path": str(dataset_dir.absolute()),
        "train": "images/train",
        "val": "images/train",  # Use same for simplicity in benchmark
        "names": {0: "apple", 1: "cup"},
    }

    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f)

    return dataset_dir
