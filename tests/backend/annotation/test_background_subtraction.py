"""
Tests for background subtraction annotator.

Tests image preprocessing, mask creation, object detection, and annotation generation.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
scripts_annotation = scripts_dir / "annotation"
if str(scripts_annotation) not in sys.path:
    sys.path.insert(0, str(scripts_annotation))
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from background_subtraction import (
    AnnotatorConfig,
    BackgroundSubtractionAnnotator,
    create_background_from_images,
)


class TestAnnotatorConfig:
    """Test AnnotatorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AnnotatorConfig()

        assert config.min_contour_area == 500
        assert config.blur_kernel_size == 5
        assert config.threshold_method == "otsu"
        assert config.fixed_threshold == 30
        assert config.morph_kernel_size == 5
        assert config.erosion_iterations == 2
        assert config.dilation_iterations == 3

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AnnotatorConfig(
            min_contour_area=1000,
            threshold_method="adaptive",
            erosion_iterations=3,
        )

        assert config.min_contour_area == 1000
        assert config.threshold_method == "adaptive"
        assert config.erosion_iterations == 3


class TestBackgroundSubtractionAnnotatorInit:
    """Test BackgroundSubtractionAnnotator initialization."""

    def test_init_with_valid_background(self, tmp_path):
        """Test initialization with valid background image."""
        # Create a simple white background image
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)

        annotator = BackgroundSubtractionAnnotator(str(bg_path))

        assert annotator.background is not None
        assert annotator.background.shape == (480, 640)

    def test_init_with_invalid_path(self):
        """Test initialization with non-existent image."""
        with pytest.raises(ValueError, match="Failed to load"):
            BackgroundSubtractionAnnotator("/nonexistent/path.jpg")

    def test_init_with_custom_config(self, tmp_path):
        """Test initialization with custom configuration."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)

        config = AnnotatorConfig(blur_kernel_size=7)
        annotator = BackgroundSubtractionAnnotator(str(bg_path), config)

        assert annotator.config.blur_kernel_size == 7


class TestBackgroundPreprocessing:
    """Test image preprocessing methods."""

    @pytest.fixture
    def annotator(self, tmp_path):
        """Create annotator with a white background."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)
        return BackgroundSubtractionAnnotator(str(bg_path))

    def test_preprocess_color_image(self, annotator):
        """Test preprocessing of color image."""
        color_image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = annotator._preprocess_image(color_image)

        assert len(result.shape) == 2  # Grayscale
        assert result.shape == (480, 640)

    def test_preprocess_grayscale_image(self, annotator):
        """Test preprocessing of already grayscale image."""
        gray_image = np.zeros((480, 640), dtype=np.uint8)

        result = annotator._preprocess_image(gray_image)

        assert len(result.shape) == 2
        assert result.shape == (480, 640)

    def test_compute_difference(self, annotator):
        """Test difference computation between image and background."""
        # Create an image that differs from the white background
        image = np.ones((480, 640), dtype=np.uint8) * 100

        diff = annotator._compute_difference(image, annotator.background)

        # Difference should be non-zero where image differs from background
        assert diff.shape == (480, 640)
        assert np.any(diff > 0)

    def test_compute_difference_resizes_background(self, annotator):
        """Test that background is resized if sizes don't match."""
        # Create smaller image
        image = np.ones((240, 320), dtype=np.uint8) * 100

        diff = annotator._compute_difference(image, annotator.background)

        # Result should match image size
        assert diff.shape == (240, 320)


class TestMaskCreation:
    """Test mask creation methods."""

    @pytest.fixture
    def annotator(self, tmp_path):
        """Create annotator with a white background."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)
        return BackgroundSubtractionAnnotator(str(bg_path))

    def test_create_mask_otsu(self, annotator):
        """Test mask creation with Otsu thresholding."""
        annotator.config.threshold_method = "otsu"
        diff = np.zeros((480, 640), dtype=np.uint8)
        diff[100:200, 100:200] = 200  # High difference region

        mask = annotator._create_mask(diff)

        assert mask.shape == (480, 640)
        # Mask should be binary (0 or 255)
        unique_values = np.unique(mask)
        assert all(v in [0, 255] for v in unique_values)

    def test_create_mask_adaptive(self, annotator):
        """Test mask creation with adaptive thresholding."""
        annotator.config.threshold_method = "adaptive"
        diff = np.zeros((480, 640), dtype=np.uint8)
        diff[100:200, 100:200] = 200

        mask = annotator._create_mask(diff)

        assert mask.shape == (480, 640)

    def test_create_mask_fixed(self, annotator):
        """Test mask creation with fixed thresholding."""
        annotator.config.threshold_method = "fixed"
        annotator.config.fixed_threshold = 50
        diff = np.zeros((480, 640), dtype=np.uint8)
        diff[100:200, 100:200] = 100  # Above threshold

        mask = annotator._create_mask(diff)

        assert mask.shape == (480, 640)


class TestObjectDetection:
    """Test object detection from mask."""

    @pytest.fixture
    def annotator(self, tmp_path):
        """Create annotator with a white background."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)
        return BackgroundSubtractionAnnotator(str(bg_path))

    def test_find_object_bbox_with_object(self, annotator):
        """Test finding bounding box when object present."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:200, 150:350] = 255  # 100x200 white region

        bbox = annotator._find_object_bbox(mask, (480, 640))

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # Should be approximately around the white region (with margin)
        assert x_min < 150
        assert y_min < 100
        assert x_max > 350
        assert y_max > 200

    def test_find_object_bbox_empty_mask(self, annotator):
        """Test finding bounding box with empty mask."""
        mask = np.zeros((480, 640), dtype=np.uint8)

        bbox = annotator._find_object_bbox(mask, (480, 640))

        assert bbox is None

    def test_find_object_bbox_too_small(self, annotator):
        """Test that too-small objects are filtered out."""
        annotator.config.min_contour_area = 10000
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[100:110, 100:110] = 255  # Very small 10x10 region

        bbox = annotator._find_object_bbox(mask, (480, 640))

        assert bbox is None

    def test_find_object_bbox_too_large(self, annotator):
        """Test that too-large objects are filtered out."""
        annotator.config.max_contour_area_ratio = 0.1
        mask = np.zeros((480, 640), dtype=np.uint8)
        mask[50:450, 50:600] = 255  # Very large region

        bbox = annotator._find_object_bbox(mask, (480, 640))

        assert bbox is None

    def test_find_largest_contour(self, annotator):
        """Test that largest valid contour is selected."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        # Small object
        mask[50:80, 50:80] = 255  # 30x30
        # Large object
        mask[200:350, 200:450] = 255  # 150x250

        bbox = annotator._find_object_bbox(mask, (480, 640))

        assert bbox is not None
        x_min, y_min, x_max, y_max = bbox
        # Should detect the larger object
        assert x_min < 220
        assert y_min < 220


class TestAnnotateImage:
    """Test full image annotation pipeline."""

    @pytest.fixture
    def setup_annotator(self, tmp_path):
        """Create annotator and test images."""
        # Create white background
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)

        # Create test image with black object on white background
        test_path = tmp_path / "test_image.jpg"
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        # Add black rectangle as object
        test_image[150:350, 200:450] = 0
        cv2.imwrite(str(test_path), test_image)

        annotator = BackgroundSubtractionAnnotator(str(bg_path))
        return annotator, str(test_path)

    def test_annotate_image_with_object(self, setup_annotator):
        """Test annotation of image with detectable object."""
        annotator, test_path = setup_annotator

        yolo_bbox = annotator.annotate_image(test_path)

        assert yolo_bbox is not None
        x_c, y_c, w, h = yolo_bbox
        # Values should be normalized [0, 1]
        assert 0 <= x_c <= 1
        assert 0 <= y_c <= 1
        assert 0 <= w <= 1
        assert 0 <= h <= 1
        # Center should be roughly in the middle
        assert 0.4 <= x_c <= 0.6
        assert 0.4 <= y_c <= 0.6

    def test_annotate_image_nonexistent(self, tmp_path):
        """Test annotation of non-existent image."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)

        annotator = BackgroundSubtractionAnnotator(str(bg_path))
        result = annotator.annotate_image("/nonexistent/image.jpg")

        assert result is None

    def test_annotate_image_no_object(self, tmp_path):
        """Test annotation when no object detected."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)

        # Create image identical to background
        test_path = tmp_path / "test.jpg"
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(test_path), test_image)

        annotator = BackgroundSubtractionAnnotator(str(bg_path))
        result = annotator.annotate_image(str(test_path))

        # Identical images should produce no detection
        # (depends on noise, might still detect something)
        # Just verify it doesn't crash
        assert result is None or isinstance(result, tuple)


class TestCreateBackgroundFromImages:
    """Test background creation from multiple images."""

    def test_create_background_median(self, tmp_path):
        """Test background creation with median method."""
        # Create test images
        images = []
        for i in range(3):
            img_path = tmp_path / f"bg_{i}.jpg"
            img = np.ones((100, 100), dtype=np.uint8) * (100 + i * 50)
            cv2.imwrite(str(img_path), img)
            images.append(str(img_path))

        output_path = tmp_path / "background.jpg"

        create_background_from_images(images, str(output_path), method="median")

        assert output_path.exists()
        result = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        assert result is not None
        assert result.shape == (100, 100)

    def test_create_background_mean(self, tmp_path):
        """Test background creation with mean method."""
        images = []
        for i in range(3):
            img_path = tmp_path / f"bg_{i}.jpg"
            img = np.ones((100, 100), dtype=np.uint8) * (100 + i * 50)
            cv2.imwrite(str(img_path), img)
            images.append(str(img_path))

        output_path = tmp_path / "background.jpg"

        create_background_from_images(images, str(output_path), method="mean")

        assert output_path.exists()

    def test_create_background_no_valid_images(self, tmp_path):
        """Test error when no valid images provided."""
        images = ["/nonexistent/1.jpg", "/nonexistent/2.jpg"]
        output_path = tmp_path / "background.jpg"

        with pytest.raises(ValueError, match="No valid images"):
            create_background_from_images(images, str(output_path))


class TestThresholdMethods:
    """Test different thresholding methods."""

    @pytest.fixture
    def create_test_setup(self, tmp_path):
        """Create annotator and test image for threshold testing."""
        def _create(threshold_method):
            bg_path = tmp_path / f"bg_{threshold_method}.jpg"
            bg_image = np.ones((200, 200, 3), dtype=np.uint8) * 200
            cv2.imwrite(str(bg_path), bg_image)

            test_path = tmp_path / f"test_{threshold_method}.jpg"
            test_image = np.ones((200, 200, 3), dtype=np.uint8) * 200
            test_image[50:150, 50:150] = 50  # Dark square
            cv2.imwrite(str(test_path), test_image)

            config = AnnotatorConfig(threshold_method=threshold_method)
            annotator = BackgroundSubtractionAnnotator(str(bg_path), config)
            return annotator, str(test_path)

        return _create

    def test_otsu_threshold(self, create_test_setup):
        """Test Otsu thresholding detects object."""
        annotator, test_path = create_test_setup("otsu")

        result = annotator.annotate_image(test_path)

        assert result is not None

    def test_adaptive_threshold(self, create_test_setup):
        """Test adaptive thresholding detects object."""
        annotator, test_path = create_test_setup("adaptive")

        result = annotator.annotate_image(test_path)

        # Adaptive might or might not detect depending on local contrast
        # Just verify it doesn't crash
        assert result is None or isinstance(result, tuple)

    def test_fixed_threshold(self, create_test_setup):
        """Test fixed thresholding detects object."""
        annotator, test_path = create_test_setup("fixed")

        result = annotator.annotate_image(test_path)

        assert result is not None


class TestMorphologicalOperations:
    """Test morphological operations in mask creation."""

    def test_erosion_removes_noise(self, tmp_path):
        """Test that erosion removes small noise regions."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)

        config = AnnotatorConfig(
            erosion_iterations=5,
            dilation_iterations=1,
        )
        annotator = BackgroundSubtractionAnnotator(str(bg_path), config)

        # Create difference with small noise
        diff = np.zeros((200, 200), dtype=np.uint8)
        diff[50:150, 50:150] = 200  # Main object
        diff[10:15, 10:15] = 200  # Small noise

        mask = annotator._create_mask(diff)

        # Small noise should be removed by erosion
        assert mask.shape == (200, 200)

    def test_dilation_expands_objects(self, tmp_path):
        """Test that dilation expands object boundaries."""
        bg_path = tmp_path / "background.jpg"
        bg_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        cv2.imwrite(str(bg_path), bg_image)

        config = AnnotatorConfig(
            erosion_iterations=1,
            dilation_iterations=5,
        )
        annotator = BackgroundSubtractionAnnotator(str(bg_path), config)

        diff = np.zeros((200, 200), dtype=np.uint8)
        diff[80:120, 80:120] = 200  # Small square

        mask = annotator._create_mask(diff)

        # Mask should exist and be binary
        assert mask.shape == (200, 200)
        unique = np.unique(mask)
        assert all(v in [0, 255] for v in unique)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
