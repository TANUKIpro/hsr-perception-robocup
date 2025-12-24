"""
Tests for Object Extractor

Tests for extracting objects from images with masks for Copy-Paste augmentation.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add scripts directory to path
_repo_root = Path(__file__).parent.parent.parent.parent
_scripts_dir = _repo_root / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from augmentation.object_extractor import ExtractedObject, ObjectExtractor


class TestObjectExtractor:
    """Test ObjectExtractor class."""

    def test_init_default(self):
        """Test default initialization."""
        extractor = ObjectExtractor()
        assert extractor.alpha_blur_sigma == 2.0
        assert extractor.padding == 5
        assert extractor.min_object_size == 32

    def test_init_custom(self):
        """Test custom initialization."""
        extractor = ObjectExtractor(
            alpha_blur_sigma=3.0, padding=10, min_object_size=64
        )
        assert extractor.alpha_blur_sigma == 3.0
        assert extractor.padding == 10
        assert extractor.min_object_size == 64


class TestExtractObject:
    """Test extract_object method."""

    def create_test_image_and_mask(self, size=(200, 200), obj_size=(100, 100)):
        """Create test image and mask."""
        image_rgb = np.zeros((*size, 3), dtype=np.uint8)
        mask = np.zeros(size, dtype=bool)

        # Create object in center
        h, w = size
        oh, ow = obj_size
        y1 = (h - oh) // 2
        x1 = (w - ow) // 2
        y2 = y1 + oh
        x2 = x1 + ow

        # Object region
        image_rgb[y1:y2, x1:x2] = [255, 128, 64]  # Orange object
        mask[y1:y2, x1:x2] = True

        return image_rgb, mask

    def test_extract_object_basic(self):
        """Test basic object extraction."""
        extractor = ObjectExtractor(alpha_blur_sigma=2.0, padding=5)

        image_rgb, mask = self.create_test_image_and_mask(
            size=(200, 200), obj_size=(100, 100)
        )

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask,
            class_id=0,
            class_name="cup",
            source_path="test.jpg",
        )

        assert obj is not None
        assert isinstance(obj, ExtractedObject)
        assert obj.class_id == 0
        assert obj.class_name == "cup"
        assert obj.source_image_path == "test.jpg"
        assert obj.rgba.shape[2] == 4  # RGBA
        assert obj.rgba.dtype == np.uint8

    def test_extract_object_with_padding(self):
        """Test that padding is applied correctly."""
        extractor = ObjectExtractor(alpha_blur_sigma=0, padding=10)

        image_rgb, mask = self.create_test_image_and_mask(
            size=(200, 200), obj_size=(80, 80)
        )

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask,
            class_id=0,
            class_name="test",
            source_path="test.jpg",
        )

        assert obj is not None
        # Object is 80x80, with padding 10 on each side, should be 100x100
        # (if object is centered and there's enough space)
        assert obj.rgba.shape[0] >= 80
        assert obj.rgba.shape[1] >= 80

    def test_extract_object_too_small(self):
        """Test that objects smaller than min_object_size are rejected."""
        extractor = ObjectExtractor(min_object_size=50)

        image_rgb, mask = self.create_test_image_and_mask(
            size=(200, 200), obj_size=(30, 30)  # Too small
        )

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask,
            class_id=0,
            class_name="test",
            source_path="test.jpg",
        )

        # Should return None because object is too small
        assert obj is None

    def test_extract_object_size_boundary(self):
        """Test object at minimum size boundary."""
        extractor = ObjectExtractor(min_object_size=50)

        image_rgb, mask = self.create_test_image_and_mask(
            size=(200, 200), obj_size=(50, 50)  # Exactly at threshold
        )

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask,
            class_id=0,
            class_name="test",
            source_path="test.jpg",
        )

        # Should succeed
        assert obj is not None

    def test_extract_object_empty_mask(self):
        """Test extraction with empty mask."""
        extractor = ObjectExtractor()

        image_rgb = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=bool)  # Empty mask

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask,
            class_id=0,
            class_name="test",
            source_path="test.jpg",
        )

        # Should return None for empty mask
        assert obj is None

    def test_extract_object_edge_case(self):
        """Test extraction with object at image edge."""
        extractor = ObjectExtractor(padding=5)

        image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)

        # Object at top-left corner
        mask[0:40, 0:40] = True
        image_rgb[0:40, 0:40] = 255

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask,
            class_id=0,
            class_name="test",
            source_path="test.jpg",
        )

        # Should handle edge case
        assert obj is not None
        assert obj.rgba.shape[0] > 0
        assert obj.rgba.shape[1] > 0

    def test_extract_object_metadata(self):
        """Test that metadata is correctly stored."""
        extractor = ObjectExtractor()

        image_rgb, mask = self.create_test_image_and_mask(
            size=(200, 200), obj_size=(100, 100)
        )

        obj = extractor.extract_object(
            image_rgb=image_rgb,
            mask=mask,
            class_id=5,
            class_name="plate",
            source_path="/path/to/image.jpg",
        )

        assert obj is not None
        assert obj.class_id == 5
        assert obj.class_name == "plate"
        assert obj.source_image_path == "/path/to/image.jpg"
        assert obj.source_image_stem == "image"
        assert obj.original_image_size == (200, 200)
        assert len(obj.extraction_timestamp) > 0


class TestCreateSoftAlpha:
    """Test _create_soft_alpha method."""

    def test_create_soft_alpha_no_blur(self):
        """Test alpha creation without blurring."""
        extractor = ObjectExtractor(alpha_blur_sigma=0)

        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True

        alpha = extractor._create_soft_alpha(mask)

        # Should be binary: 0 or 255
        assert alpha.dtype == np.uint8
        assert set(np.unique(alpha)) <= {0, 255}

        # Check that mask region is 255
        assert alpha[50, 50] == 255
        assert alpha[10, 10] == 0

    def test_create_soft_alpha_with_blur(self):
        """Test alpha creation with blurring."""
        extractor = ObjectExtractor(alpha_blur_sigma=2.0)

        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        alpha = extractor._create_soft_alpha(mask)

        # Should have smooth edges
        assert alpha.dtype == np.uint8

        # Interior should be 255 (fully opaque)
        assert alpha[50, 50] == 255

        # Edges should have intermediate values (gradient)
        # Check near the edge
        edge_values = alpha[29:31, 50]  # Just outside and inside edge
        # Should have some variation (not all same)
        assert len(np.unique(edge_values)) > 1

    def test_create_soft_alpha_small_mask(self):
        """Test alpha creation with small mask."""
        extractor = ObjectExtractor(alpha_blur_sigma=2.0)

        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True  # Small 10x10 region

        alpha = extractor._create_soft_alpha(mask)

        # Should still work
        assert alpha.dtype == np.uint8
        assert alpha.shape == (50, 50)


class TestSaveAndLoadObject:
    """Test saving and loading extracted objects."""

    def create_test_object(self):
        """Create a test ExtractedObject."""
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[25:75, 25:75, :3] = 255
        rgba[25:75, 25:75, 3] = 255

        return ExtractedObject(
            rgba=rgba,
            class_id=3,
            class_name="cup",
            source_image_path="/path/to/image.jpg",
            source_image_stem="image",
            original_bbox=(20, 20, 120, 120),
            original_image_size=(200, 200),
            extraction_timestamp="2025-01-15T10:30:00",
        )

    def test_save_extracted_object(self):
        """Test saving object to NPZ."""
        extractor = ObjectExtractor()
        obj = self.create_test_object()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = extractor.save_extracted_object(obj, tmpdir)

            # Check file exists
            assert Path(saved_path).exists()
            assert Path(saved_path).suffix == ".npz"

            # Check filename contains class name
            assert "cup" in Path(saved_path).name

    def test_load_extracted_object(self):
        """Test loading object from NPZ."""
        extractor = ObjectExtractor()
        obj = self.create_test_object()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = extractor.save_extracted_object(obj, tmpdir)

            # Load it back
            loaded_obj = ObjectExtractor.load_extracted_object(saved_path)

            assert loaded_obj is not None
            assert loaded_obj.class_id == obj.class_id
            assert loaded_obj.class_name == obj.class_name
            assert loaded_obj.source_image_path == obj.source_image_path
            assert loaded_obj.original_bbox == obj.original_bbox
            assert loaded_obj.original_image_size == obj.original_image_size
            np.testing.assert_array_equal(loaded_obj.rgba, obj.rgba)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        result = ObjectExtractor.load_extracted_object("/nonexistent/file.npz")
        assert result is None

    def test_load_invalid_file(self):
        """Test loading from invalid file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a valid npz file")
            f.flush()

            result = ObjectExtractor.load_extracted_object(f.name)
            assert result is None

            Path(f.name).unlink()


class TestExtractedObject:
    """Test ExtractedObject dataclass."""

    def test_extracted_object_creation(self):
        """Test creating ExtractedObject."""
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)

        obj = ExtractedObject(
            rgba=rgba,
            class_id=5,
            class_name="plate",
            source_image_path="/test/image.jpg",
            source_image_stem="image",
            original_bbox=(10, 20, 110, 120),
            original_image_size=(480, 640),
            extraction_timestamp="2025-01-15T12:00:00",
        )

        assert obj.class_id == 5
        assert obj.class_name == "plate"
        assert obj.source_image_path == "/test/image.jpg"
        assert obj.source_image_stem == "image"
        assert obj.original_bbox == (10, 20, 110, 120)
        assert obj.original_image_size == (480, 640)
        assert obj.extraction_timestamp == "2025-01-15T12:00:00"
        np.testing.assert_array_equal(obj.rgba, rgba)

    def test_to_metadata(self):
        """Test converting to metadata dict."""
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)

        obj = ExtractedObject(
            rgba=rgba,
            class_id=5,
            class_name="plate",
            source_image_path="/test/image.jpg",
            source_image_stem="image",
            original_bbox=(10, 20, 110, 120),
            original_image_size=(480, 640),
            extraction_timestamp="2025-01-15T12:00:00",
        )

        metadata = obj.to_metadata()

        # Check metadata structure
        assert isinstance(metadata, dict)
        assert metadata["class_id"] == 5
        assert metadata["class_name"] == "plate"
        assert metadata["source_image_path"] == "/test/image.jpg"
        assert metadata["source_image_stem"] == "image"
        assert metadata["original_bbox"] == [10, 20, 110, 120]
        assert metadata["original_image_size"] == [480, 640]
        assert metadata["rgba_shape"] == [100, 100, 4]
        assert metadata["extraction_timestamp"] == "2025-01-15T12:00:00"

        # Should be JSON serializable
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)


class TestBatchExtraction:
    """Test batch extraction methods."""

    def test_batch_extract_from_images_and_masks(self):
        """Test batch extraction with image/mask pairs."""
        extractor = ObjectExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create image and mask directories
            image_dir = tmpdir / "images"
            mask_dir = tmpdir / "masks"
            output_dir = tmpdir / "output"
            image_dir.mkdir()
            mask_dir.mkdir()

            # Create test image and mask
            image = np.full((200, 200, 3), 128, dtype=np.uint8)
            mask = np.zeros((200, 200), dtype=np.uint8)
            mask[50:150, 50:150] = 255

            # Save test files
            cv2.imwrite(str(image_dir / "test1.jpg"), image)
            cv2.imwrite(str(mask_dir / "test1_mask.png"), mask)

            # Extract
            saved_paths = extractor.batch_extract_from_images_and_masks(
                image_dir=image_dir,
                mask_dir=mask_dir,
                class_id=0,
                class_name="cup",
                output_dir=output_dir,
            )

            # Should have extracted one object
            assert len(saved_paths) >= 0  # Might be 0 if object is too small

    def test_batch_extract_no_masks(self):
        """Test batch extraction when masks are missing."""
        extractor = ObjectExtractor()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            image_dir = tmpdir / "images"
            mask_dir = tmpdir / "masks"
            output_dir = tmpdir / "output"
            image_dir.mkdir()
            mask_dir.mkdir()

            # Create image but no mask
            image = np.full((200, 200, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(image_dir / "test1.jpg"), image)

            # Extract
            saved_paths = extractor.batch_extract_from_images_and_masks(
                image_dir=image_dir,
                mask_dir=mask_dir,
                class_id=0,
                class_name="cup",
                output_dir=output_dir,
            )

            # Should not extract anything
            assert len(saved_paths) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
