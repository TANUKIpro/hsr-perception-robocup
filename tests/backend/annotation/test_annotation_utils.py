"""
Tests for annotation utilities.

Tests YOLO format conversion, dataset operations, and annotation utilities.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from annotation.annotation_utils import (
    AnnotationResult,
    bbox_to_yolo,
    create_dataset_yaml,
    get_class_id_map,
    get_class_names,
    load_class_config,
    mask_to_bbox,
    read_yolo_label,
    split_dataset,
    validate_yolo_annotation,
    write_yolo_label,
    yolo_to_bbox,
    _extract_timestamp,
    _group_by_timestamp,
)


class TestAnnotationResult:
    """Test AnnotationResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = AnnotationResult()

        assert result.total_images == 0
        assert result.successful == 0
        assert result.failed == 0
        assert result.failed_paths == []

    def test_custom_values(self):
        """Test custom initialization."""
        result = AnnotationResult(
            total_images=100,
            successful=95,
            failed=5,
            failed_paths=["img1.jpg", "img2.jpg"],
        )

        assert result.total_images == 100
        assert result.successful == 95
        assert result.failed == 5
        assert len(result.failed_paths) == 2

    def test_success_rate_all_successful(self):
        """Test success rate with all images successful."""
        result = AnnotationResult(total_images=100, successful=100, failed=0)

        assert result.success_rate == 100.0

    def test_success_rate_partial(self):
        """Test success rate with partial success."""
        result = AnnotationResult(total_images=100, successful=75, failed=25)

        assert result.success_rate == 75.0

    def test_success_rate_zero_images(self):
        """Test success rate with no images."""
        result = AnnotationResult()

        assert result.success_rate == 0.0

    def test_summary_format(self):
        """Test summary string format."""
        result = AnnotationResult(
            total_images=10,
            successful=8,
            failed=2,
        )

        summary = result.summary()

        assert "Total: 10" in summary
        assert "Successful: 8" in summary
        assert "Failed: 2" in summary
        assert "80.0%" in summary


class TestBboxToYolo:
    """Test bbox_to_yolo conversion function."""

    def test_basic_conversion(self):
        """Test basic coordinate conversion."""
        # Bbox at center of 640x480 image
        x_c, y_c, w, h = bbox_to_yolo(270, 190, 370, 290, 640, 480)

        # Center should be at (0.5, 0.5), size 100x100
        assert abs(x_c - 0.5) < 0.01
        assert abs(y_c - 0.5) < 0.01
        assert abs(w - 100/640) < 0.01
        assert abs(h - 100/480) < 0.01

    def test_top_left_corner(self):
        """Test bbox at top-left corner."""
        x_c, y_c, w, h = bbox_to_yolo(0, 0, 64, 48, 640, 480)

        # Center at (32, 24) normalized
        assert abs(x_c - 0.05) < 0.01
        assert abs(y_c - 0.05) < 0.01
        assert abs(w - 0.1) < 0.01
        assert abs(h - 0.1) < 0.01

    def test_full_image(self):
        """Test bbox covering entire image."""
        x_c, y_c, w, h = bbox_to_yolo(0, 0, 640, 480, 640, 480)

        assert abs(x_c - 0.5) < 0.01
        assert abs(y_c - 0.5) < 0.01
        assert abs(w - 1.0) < 0.01
        assert abs(h - 1.0) < 0.01

    def test_clamps_to_valid_range(self):
        """Test that values are clamped to [0, 1]."""
        # Bbox extending beyond image boundaries
        x_c, y_c, w, h = bbox_to_yolo(-50, -50, 700, 500, 640, 480)

        assert 0.0 <= x_c <= 1.0
        assert 0.0 <= y_c <= 1.0
        assert 0.0 <= w <= 1.0
        assert 0.0 <= h <= 1.0


class TestYoloToBbox:
    """Test yolo_to_bbox conversion function."""

    def test_basic_conversion(self):
        """Test basic YOLO to bbox conversion."""
        x_min, y_min, x_max, y_max = yolo_to_bbox(0.5, 0.5, 0.2, 0.2, 640, 480)

        # Center at (320, 240), size 128x96
        assert abs(x_min - (320 - 64)) < 2
        assert abs(y_min - (240 - 48)) < 2
        assert abs(x_max - (320 + 64)) < 2
        assert abs(y_max - (240 + 48)) < 2

    def test_top_left_corner(self):
        """Test YOLO bbox at top-left corner."""
        x_min, y_min, x_max, y_max = yolo_to_bbox(0.05, 0.05, 0.1, 0.1, 640, 480)

        assert x_min == 0  # Clamped to 0
        assert y_min == 0  # Clamped to 0

    def test_clamps_to_image_boundaries(self):
        """Test that coordinates are clamped to image boundaries."""
        # Large bbox that would extend beyond image
        x_min, y_min, x_max, y_max = yolo_to_bbox(0.9, 0.9, 0.5, 0.5, 640, 480)

        assert x_min >= 0
        assert y_min >= 0
        assert x_max <= 640
        assert y_max <= 480

    def test_roundtrip_conversion(self):
        """Test that bbox -> yolo -> bbox preserves values (approximately)."""
        original = (100, 100, 300, 400)
        img_w, img_h = 640, 480

        yolo = bbox_to_yolo(*original, img_w, img_h)
        restored = yolo_to_bbox(*yolo, img_w, img_h)

        # Allow small rounding differences
        assert abs(restored[0] - original[0]) <= 1
        assert abs(restored[1] - original[1]) <= 1
        assert abs(restored[2] - original[2]) <= 1
        assert abs(restored[3] - original[3]) <= 1


class TestWriteYoloLabel:
    """Test write_yolo_label function."""

    def test_write_single_label(self, tmp_path):
        """Test writing a single label."""
        label_path = tmp_path / "test.txt"
        bbox = (0.5, 0.5, 0.2, 0.2)

        write_yolo_label(str(label_path), 0, bbox)

        content = label_path.read_text().strip()
        parts = content.split()

        assert parts[0] == "0"
        assert float(parts[1]) == pytest.approx(0.5, abs=0.0001)
        assert float(parts[2]) == pytest.approx(0.5, abs=0.0001)
        assert float(parts[3]) == pytest.approx(0.2, abs=0.0001)
        assert float(parts[4]) == pytest.approx(0.2, abs=0.0001)

    def test_write_overwrites_by_default(self, tmp_path):
        """Test that write overwrites existing file by default."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("old content")

        write_yolo_label(str(label_path), 1, (0.5, 0.5, 0.1, 0.1))

        content = label_path.read_text()
        assert "old content" not in content
        assert content.startswith("1 ")

    def test_append_mode(self, tmp_path):
        """Test appending to existing file."""
        label_path = tmp_path / "test.txt"

        write_yolo_label(str(label_path), 0, (0.3, 0.3, 0.1, 0.1), append=False)
        write_yolo_label(str(label_path), 1, (0.7, 0.7, 0.1, 0.1), append=True)

        lines = label_path.read_text().strip().split("\n")

        assert len(lines) == 2
        assert lines[0].startswith("0 ")
        assert lines[1].startswith("1 ")


class TestReadYoloLabel:
    """Test read_yolo_label function."""

    def test_read_single_label(self, tmp_path):
        """Test reading a single label."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n")

        labels = read_yolo_label(str(label_path))

        assert len(labels) == 1
        assert labels[0] == (0, 0.5, 0.5, 0.2, 0.2)

    def test_read_multiple_labels(self, tmp_path):
        """Test reading multiple labels."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 0.3 0.3 0.1 0.1\n1 0.7 0.7 0.2 0.2\n")

        labels = read_yolo_label(str(label_path))

        assert len(labels) == 2
        assert labels[0][0] == 0
        assert labels[1][0] == 1

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading non-existent file returns empty list."""
        labels = read_yolo_label(str(tmp_path / "nonexistent.txt"))

        assert labels == []

    def test_skip_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n\n\n1 0.3 0.3 0.1 0.1\n")

        labels = read_yolo_label(str(label_path))

        assert len(labels) == 2

    def test_skip_incomplete_lines(self, tmp_path):
        """Test that lines with fewer than 5 fields are skipped."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 0.5 0.5\n0 0.5 0.5 0.2 0.2\n")

        labels = read_yolo_label(str(label_path))

        assert len(labels) == 1


class TestValidateYoloAnnotation:
    """Test validate_yolo_annotation function."""

    def test_valid_annotation(self, tmp_path):
        """Test valid annotation file."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")

        is_valid, errors = validate_yolo_annotation(str(label_path))

        assert is_valid is True
        assert errors == []

    def test_file_not_found(self, tmp_path):
        """Test validation of non-existent file."""
        is_valid, errors = validate_yolo_annotation(str(tmp_path / "missing.txt"))

        assert is_valid is False
        assert "not found" in errors[0].lower()

    def test_insufficient_fields(self, tmp_path):
        """Test detection of insufficient fields."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 0.5 0.5\n")  # Only 3 fields

        is_valid, errors = validate_yolo_annotation(str(label_path))

        assert is_valid is False
        assert any("Expected 5 fields" in e for e in errors)

    def test_invalid_class_id(self, tmp_path):
        """Test detection of invalid class ID."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("abc 0.5 0.5 0.2 0.2\n")

        is_valid, errors = validate_yolo_annotation(str(label_path))

        assert is_valid is False
        assert any("Invalid class_id" in e for e in errors)

    def test_negative_class_id(self, tmp_path):
        """Test detection of negative class ID."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("-1 0.5 0.5 0.2 0.2\n")

        is_valid, errors = validate_yolo_annotation(str(label_path))

        assert is_valid is False
        assert any("Negative class_id" in e for e in errors)

    def test_out_of_range_values(self, tmp_path):
        """Test detection of values outside [0, 1]."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("0 1.5 0.5 0.2 0.2\n")  # x_center > 1

        is_valid, errors = validate_yolo_annotation(str(label_path))

        assert is_valid is False
        assert any("out of range" in e for e in errors)

    def test_multiple_errors(self, tmp_path):
        """Test that multiple errors are collected."""
        label_path = tmp_path / "test.txt"
        label_path.write_text("abc 0.5 0.5 0.2 0.2\n0 1.5 -0.1 0.2 0.2\n")

        is_valid, errors = validate_yolo_annotation(str(label_path))

        assert is_valid is False
        assert len(errors) >= 2


class TestCreateDatasetYaml:
    """Test create_dataset_yaml function."""

    def test_basic_yaml_creation(self, tmp_path):
        """Test basic dataset YAML creation."""
        yaml_path = tmp_path / "data.yaml"
        class_names = ["apple", "banana", "orange"]

        create_dataset_yaml(
            str(yaml_path),
            "train/images",
            "val/images",
            class_names,
        )

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["path"] == "."
        assert config["train"] == "train/images"
        assert config["val"] == "val/images"
        assert config["names"] == {0: "apple", 1: "banana", 2: "orange"}

    def test_with_test_path(self, tmp_path):
        """Test YAML creation with test path."""
        yaml_path = tmp_path / "data.yaml"

        create_dataset_yaml(
            str(yaml_path),
            "train/images",
            "val/images",
            ["class1"],
            test_path="test/images",
        )

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["test"] == "test/images"

    def test_empty_classes(self, tmp_path):
        """Test YAML creation with empty class list."""
        yaml_path = tmp_path / "data.yaml"

        create_dataset_yaml(
            str(yaml_path),
            "train/images",
            "val/images",
            [],
        )

        with open(yaml_path) as f:
            config = yaml.safe_load(f)

        assert config["names"] == {}


class TestExtractTimestamp:
    """Test _extract_timestamp function."""

    def test_valid_timestamp(self):
        """Test extraction from valid filename."""
        filename = "apple_20251211_123456_789.jpg"

        ts = _extract_timestamp(filename)

        assert ts is not None
        assert ts.year == 2025
        assert ts.month == 12
        assert ts.day == 11
        assert ts.hour == 12
        assert ts.minute == 34
        assert ts.second == 56

    def test_invalid_format(self):
        """Test extraction from invalid format."""
        ts = _extract_timestamp("random_image.jpg")

        assert ts is None

    def test_partial_format(self):
        """Test extraction from partially matching format."""
        ts = _extract_timestamp("image_20251211.jpg")  # Missing time

        assert ts is None


class TestGroupByTimestamp:
    """Test _group_by_timestamp function."""

    def test_empty_pairs(self):
        """Test with empty input."""
        groups = _group_by_timestamp([], 2.0)

        assert groups == []

    def test_single_pair(self):
        """Test with single pair."""
        pairs = [(Path("apple_20251211_120000_000.jpg"), Path("label.txt"))]

        groups = _group_by_timestamp(pairs, 2.0)

        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_pairs_within_interval(self):
        """Test pairs within interval are grouped together."""
        pairs = [
            (Path("apple_20251211_120000_000.jpg"), Path("l1.txt")),
            (Path("apple_20251211_120001_000.jpg"), Path("l2.txt")),  # 1 sec later
            (Path("apple_20251211_120002_000.jpg"), Path("l3.txt")),  # 2 sec later
        ]

        groups = _group_by_timestamp(pairs, 2.0)

        # All should be in same group (each consecutive pair is within 2 sec)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_pairs_beyond_interval(self):
        """Test pairs beyond interval are in separate groups."""
        pairs = [
            (Path("apple_20251211_120000_000.jpg"), Path("l1.txt")),
            (Path("apple_20251211_120010_000.jpg"), Path("l2.txt")),  # 10 sec later
        ]

        groups = _group_by_timestamp(pairs, 2.0)

        assert len(groups) == 2


class TestSplitDataset:
    """Test split_dataset function."""

    def test_basic_split(self, tmp_path):
        """Test basic dataset splitting."""
        # Create source directories
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        output_dir = tmp_path / "output"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create test files
        for i in range(10):
            (images_dir / f"img_{i:03d}.jpg").touch()
            (labels_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        result = split_dataset(
            str(images_dir),
            str(labels_dir),
            str(output_dir),
            train_ratio=0.8,
            seed=42,
            group_continuous_frames=False,
        )

        assert result["train"] == 8
        assert result["val"] == 2
        assert (output_dir / "images" / "train").exists()
        assert (output_dir / "images" / "val").exists()

    def test_handles_missing_labels(self, tmp_path):
        """Test that images without labels are skipped."""
        images_dir = tmp_path / "images"
        labels_dir = tmp_path / "labels"
        output_dir = tmp_path / "output"
        images_dir.mkdir()
        labels_dir.mkdir()

        # Create 5 images but only 3 labels
        for i in range(5):
            (images_dir / f"img_{i:03d}.jpg").touch()
        for i in range(3):
            (labels_dir / f"img_{i:03d}.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        result = split_dataset(
            str(images_dir),
            str(labels_dir),
            str(output_dir),
            train_ratio=0.8,
            group_continuous_frames=False,
        )

        assert result["train"] + result["val"] == 3


class TestLoadClassConfig:
    """Test load_class_config function."""

    def test_load_valid_config(self, tmp_path):
        """Test loading valid config."""
        config_path = tmp_path / "classes.json"
        config_data = {
            "categories": ["food"],
            "objects": [{"class_id": 0, "class_name": "apple"}],
        }
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        result = load_class_config(str(config_path))

        assert result == config_data

    def test_file_not_found(self, tmp_path):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_class_config(str(tmp_path / "missing.json"))


class TestGetClassNames:
    """Test get_class_names function."""

    def test_extract_names(self):
        """Test extracting class names from config."""
        config = {
            "objects": [
                {"class_id": 0, "class_name": "apple"},
                {"class_id": 1, "class_name": "banana"},
            ]
        }

        names = get_class_names(config)

        assert names == ["apple", "banana"]

    def test_sorts_by_class_id(self):
        """Test that names are sorted by class_id."""
        config = {
            "objects": [
                {"class_id": 2, "class_name": "cherry"},
                {"class_id": 0, "class_name": "apple"},
                {"class_id": 1, "class_name": "banana"},
            ]
        }

        names = get_class_names(config)

        assert names == ["apple", "banana", "cherry"]

    def test_empty_objects(self):
        """Test with empty objects list."""
        config = {"objects": []}

        names = get_class_names(config)

        assert names == []


class TestGetClassIdMap:
    """Test get_class_id_map function."""

    def test_create_map(self):
        """Test creating class ID map."""
        config = {
            "objects": [
                {"class_id": 0, "class_name": "apple"},
                {"class_id": 1, "class_name": "banana"},
            ]
        }

        id_map = get_class_id_map(config)

        assert id_map == {"apple": 0, "banana": 1}

    def test_empty_objects(self):
        """Test with empty objects list."""
        config = {"objects": []}

        id_map = get_class_id_map(config)

        assert id_map == {}


class TestMaskToBbox:
    """Test mask_to_bbox function."""

    def test_simple_mask(self):
        """Test conversion of simple rectangular mask."""
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:200, 150:350] = True

        bbox = mask_to_bbox(mask)

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        assert x_min == 150
        assert y_min == 100
        assert x_max == 349  # Last True column
        assert y_max == 199  # Last True row

    def test_empty_mask(self):
        """Test with empty (all False) mask."""
        mask = np.zeros((480, 640), dtype=bool)

        bbox = mask_to_bbox(mask)

        assert bbox is None

    def test_with_margin(self):
        """Test bbox expansion with margin."""
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:300, 200:400] = True  # 100x200 box

        bbox = mask_to_bbox(mask, margin_ratio=0.1, image_shape=(480, 640))

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # Original bbox would be (200, 200, 399, 299)
        # Margin of 10% expands by 20 and 10 pixels
        assert x_min < 200
        assert y_min < 200
        assert x_max > 399
        assert y_max > 299

    def test_margin_clamped_to_image(self):
        """Test that margin is clamped to image boundaries."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[0:10, 0:10] = True  # Near corner

        bbox = mask_to_bbox(mask, margin_ratio=0.5, image_shape=(100, 100))

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        assert x_min >= 0
        assert y_min >= 0

    def test_contour_mode(self):
        """Test contour-based detection mode."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 150:350] = 255

        bbox = mask_to_bbox(mask, use_contour=True)

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # Contour mode may have slightly different bounds
        assert 145 <= x_min <= 155
        assert 95 <= y_min <= 105


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
