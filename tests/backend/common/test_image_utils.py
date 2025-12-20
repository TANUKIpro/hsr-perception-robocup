"""
Tests for image utilities.

Tests image processing utilities including mask-to-bbox conversion,
visualization functions, and file I/O operations.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from common.image_utils import (
    mask_to_bbox,
    find_object_bbox,
    draw_bbox,
    draw_mask_overlay,
    draw_detections,
    list_image_files,
    load_image,
    save_image,
)


class TestMaskToBbox:
    """Test mask_to_bbox function."""

    def test_basic_mask_to_bbox(self):
        """Test basic mask to bounding box conversion."""
        # Create a simple rectangular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 255  # Rectangle from (30,20) to (60,40)

        bbox = mask_to_bbox(mask, (100, 100), bbox_margin_ratio=0.0)

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # Should be approximately at the mask bounds
        assert x_min <= 30
        assert y_min <= 20
        assert x_max >= 59
        assert y_max >= 39

    def test_empty_mask_returns_none(self):
        """Test empty mask returns None."""
        mask = np.zeros((100, 100), dtype=np.uint8)

        bbox = mask_to_bbox(mask, (100, 100))

        assert bbox is None

    def test_with_margin(self):
        """Test margin is applied correctly."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:50, 30:50] = 255  # 20x20 square

        bbox = mask_to_bbox(mask, (100, 100), bbox_margin_ratio=0.1)

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # With 10% margin on 20px width, margin should be 2px
        # So bounds should extend by 2 from original
        assert x_min < 30
        assert y_min < 30
        assert x_max > 49
        assert y_max > 49

    def test_clamp_to_image_bounds(self):
        """Test bbox is clamped to image boundaries."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:20, 0:20] = 255  # Corner mask

        bbox = mask_to_bbox(mask, (100, 100), bbox_margin_ratio=0.5)

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # Should be clamped to [0, 100)
        assert x_min >= 0
        assert y_min >= 0
        assert x_max <= 100
        assert y_max <= 100


class TestFindObjectBbox:
    """Test find_object_bbox function."""

    def test_find_object(self):
        """Test finding object in foreground mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # 50x50 square in center

        bbox = find_object_bbox(mask, (100, 100), min_contour_area=100)

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        assert x_min <= 25
        assert y_min <= 25
        assert x_max >= 74
        assert y_max >= 74

    def test_filter_by_min_area(self):
        """Test filtering by minimum contour area."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[45:55, 45:55] = 255  # 10x10 = 100 pixels

        # Should find with low threshold
        bbox = find_object_bbox(mask, (100, 100), min_contour_area=50)
        assert bbox is not None

        # Should not find with high threshold
        bbox = find_object_bbox(mask, (100, 100), min_contour_area=500)
        assert bbox is None

    def test_filter_by_max_area_ratio(self):
        """Test filtering by maximum area ratio."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:100, 0:100] = 255  # Full image = 10000 pixels

        # Should not find when max_area_ratio is low
        bbox = find_object_bbox(
            mask, (100, 100), min_contour_area=100, max_contour_area_ratio=0.5
        )
        assert bbox is None

        # Should find when max_area_ratio is high
        bbox = find_object_bbox(
            mask, (100, 100), min_contour_area=100, max_contour_area_ratio=1.0
        )
        assert bbox is not None

    def test_no_valid_contours(self):
        """Test when no valid contours are found."""
        mask = np.zeros((100, 100), dtype=np.uint8)

        bbox = find_object_bbox(mask, (100, 100))

        assert bbox is None


class TestDrawBbox:
    """Test draw_bbox function."""

    def test_draw_rectangle(self):
        """Test drawing rectangle on image."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (20, 20, 80, 80)

        result = draw_bbox(image, bbox, color=(0, 255, 0))

        # Check that some green pixels are in the result
        green_mask = (result[:, :, 1] == 255) & (result[:, :, 0] == 0) & (result[:, :, 2] == 0)
        assert np.any(green_mask)

    def test_draw_with_label(self):
        """Test drawing bbox with label."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (20, 30, 80, 80)

        result = draw_bbox(image, bbox, label="test", color=(0, 255, 0))

        # Image should be modified
        assert not np.array_equal(result, np.zeros((100, 100, 3), dtype=np.uint8))

    def test_draw_modifies_in_place(self):
        """Test that draw_bbox modifies image in place."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (20, 20, 80, 80)

        result = draw_bbox(image, bbox)

        # Result should be same object as input
        assert result is image


class TestDrawMaskOverlay:
    """Test draw_mask_overlay function."""

    def test_overlay_application(self):
        """Test mask overlay is applied correctly."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        result = draw_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5)

        # Check that masked region is different from background
        bg_pixel = result[10, 10]
        fg_pixel = result[50, 50]
        assert not np.array_equal(bg_pixel, fg_pixel)

    def test_alpha_blending(self):
        """Test alpha blending works correctly."""
        image = np.ones((100, 100, 3), dtype=np.uint8) * 100
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        result = draw_mask_overlay(image, mask, color=(0, 200, 0), alpha=0.5)

        # Check that the masked region has some green added
        center_pixel = result[50, 50]
        # Green channel should be higher than original (100)
        assert center_pixel[1] > 100


class TestDrawDetections:
    """Test draw_detections function."""

    def test_multiple_detections(self):
        """Test drawing multiple detections."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"bbox": (10, 10, 30, 30), "class_id": 0},
            {"bbox": (50, 50, 80, 80), "class_id": 1},
        ]

        result = draw_detections(image, detections)

        # Image should be modified
        assert not np.array_equal(result, np.zeros((100, 100, 3), dtype=np.uint8))

    def test_color_map_usage(self):
        """Test color map is used correctly."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [{"bbox": (10, 10, 30, 30), "class_id": 0}]
        color_map = {0: (255, 0, 0)}  # Blue (BGR)

        result = draw_detections(image, detections, color_map=color_map)

        # Check that blue pixels are present
        blue_mask = (result[:, :, 0] == 255)
        assert np.any(blue_mask)

    def test_confidence_display(self):
        """Test confidence is displayed in label."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = [
            {"bbox": (10, 20, 60, 60), "class_id": 0, "confidence": 0.95, "class_name": "test"}
        ]

        result = draw_detections(
            image, detections, show_confidence=True, show_class_name=True
        )

        # Image should be modified (label drawn)
        assert not np.array_equal(result, np.zeros((100, 100, 3), dtype=np.uint8))

    def test_skip_detection_without_bbox(self):
        """Test detections without bbox are skipped."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original = image.copy()
        detections = [{"class_id": 0}]  # No bbox

        result = draw_detections(image, detections)

        # Image should not be modified
        assert np.array_equal(result, original)


class TestListImageFiles:
    """Test list_image_files function."""

    def test_find_jpg_png(self, tmp_path):
        """Test finding JPG and PNG files."""
        # Create test files
        (tmp_path / "image1.jpg").touch()
        (tmp_path / "image2.png").touch()
        (tmp_path / "other.txt").touch()

        files = list_image_files(tmp_path)

        assert len(files) == 2
        extensions = [f.suffix.lower() for f in files]
        assert ".jpg" in extensions
        assert ".png" in extensions

    def test_recursive_search(self, tmp_path):
        """Test recursive file search."""
        # Create subdirectory with images
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "image1.jpg").touch()
        (subdir / "image2.jpg").touch()

        # Non-recursive
        files = list_image_files(tmp_path, recursive=False)
        assert len(files) == 1

        # Recursive
        files = list_image_files(tmp_path, recursive=True)
        assert len(files) == 2

    def test_custom_extensions(self, tmp_path):
        """Test custom file extensions."""
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image.bmp").touch()

        files = list_image_files(tmp_path, extensions=[".bmp"])

        assert len(files) == 1
        assert files[0].suffix.lower() == ".bmp"

    def test_nonexistent_directory(self, tmp_path):
        """Test with non-existent directory."""
        nonexistent = tmp_path / "nonexistent"

        files = list_image_files(nonexistent)

        assert files == []


class TestLoadImage:
    """Test load_image function."""

    def test_load_bgr(self, tmp_path):
        """Test loading image in BGR mode."""
        # Create a test image (use PNG to avoid JPEG compression artifacts)
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[:, :, 2] = 255  # Red in BGR
        image_path = tmp_path / "test.png"
        cv2.imwrite(str(image_path), test_image)

        result = load_image(image_path, color_mode="bgr")

        assert result is not None
        assert result.shape == (50, 50, 3)
        # Red should be in channel 2 (BGR format)
        assert result[25, 25, 2] == 255

    def test_load_rgb(self, tmp_path):
        """Test loading image in RGB mode."""
        # Use PNG to avoid JPEG compression artifacts
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[:, :, 2] = 255  # Red in BGR
        image_path = tmp_path / "test.png"
        cv2.imwrite(str(image_path), test_image)

        result = load_image(image_path, color_mode="rgb")

        assert result is not None
        # Red should be in channel 0 (RGB format after conversion)
        assert result[25, 25, 0] == 255

    def test_load_gray(self, tmp_path):
        """Test loading image in grayscale mode."""
        test_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        image_path = tmp_path / "test.jpg"
        cv2.imwrite(str(image_path), test_image)

        result = load_image(image_path, color_mode="gray")

        assert result is not None
        assert len(result.shape) == 2  # Grayscale is 2D

    def test_load_nonexistent(self):
        """Test loading non-existent file."""
        result = load_image("/nonexistent/path/image.jpg")

        assert result is None


class TestSaveImage:
    """Test save_image function."""

    def test_save_jpg(self, tmp_path):
        """Test saving JPEG image."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        output_path = tmp_path / "output.jpg"

        result = save_image(image, output_path)

        assert result is True
        assert output_path.exists()

    def test_save_png(self, tmp_path):
        """Test saving PNG image."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        output_path = tmp_path / "output.png"

        result = save_image(image, output_path)

        assert result is True
        assert output_path.exists()

    def test_quality_parameter(self, tmp_path):
        """Test quality parameter affects file size."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        high_quality_path = tmp_path / "high.jpg"
        low_quality_path = tmp_path / "low.jpg"

        save_image(image, high_quality_path, quality=95)
        save_image(image, low_quality_path, quality=10)

        # High quality should be larger than low quality
        assert high_quality_path.stat().st_size > low_quality_path.stat().st_size

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created."""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        output_path = tmp_path / "subdir" / "nested" / "output.jpg"

        result = save_image(image, output_path)

        assert result is True
        assert output_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
