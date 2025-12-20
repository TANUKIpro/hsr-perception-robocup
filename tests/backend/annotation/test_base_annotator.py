"""
Tests for base annotator.

Tests the abstract base class for annotators and shared implementations.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from annotation.base_annotator import BaseAnnotator


class ConcreteAnnotator(BaseAnnotator):
    """Concrete implementation of BaseAnnotator for testing."""

    def __init__(self, return_bbox: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize with optional fixed return value.

        Args:
            return_bbox: Fixed bbox to return, or None to simulate no detection
        """
        self.return_bbox = return_bbox
        self.annotate_image_calls = []

    def annotate_image(
        self,
        image_path: str,
        **kwargs,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return the configured bbox value."""
        self.annotate_image_calls.append(image_path)
        return self.return_bbox


class FailingAnnotator(BaseAnnotator):
    """Annotator that always returns None (no detection)."""

    def annotate_image(
        self,
        image_path: str,
        **kwargs,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Always return None."""
        return None


class TestBaseAnnotator:
    """Test BaseAnnotator abstract class."""

    def test_abstract_method_enforcement(self):
        """Test that annotate_image is enforced as abstract method."""
        # Attempting to instantiate BaseAnnotator directly should fail
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAnnotator()

    def test_cannot_instantiate(self):
        """Test that BaseAnnotator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAnnotator()

    def test_concrete_implementation_works(self):
        """Test that concrete implementation can be instantiated."""
        annotator = ConcreteAnnotator()
        assert annotator is not None


class TestConcreteAnnotator:
    """Test BaseAnnotator using a concrete implementation."""

    def test_annotate_batch(self, tmp_path):
        """Test batch annotation."""
        # Create test images
        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"image_{i}.jpg"), img)

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create annotator that returns a fixed bbox
        annotator = ConcreteAnnotator(return_bbox=(0.5, 0.5, 0.2, 0.2))

        result = annotator.annotate_batch(
            image_dir=str(tmp_path),
            class_id=0,
            output_dir=str(output_dir),
        )

        # Check results
        assert result.total_images == 3
        assert result.successful == 3
        assert result.failed == 0
        assert len(annotator.annotate_image_calls) == 3

        # Check that label files were created
        label_files = list(output_dir.glob("*.txt"))
        assert len(label_files) == 3

    def test_annotate_batch_with_failures(self, tmp_path):
        """Test batch annotation with some failures."""
        # Create test images
        for i in range(3):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"image_{i}.jpg"), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create annotator that returns None (simulating no detection)
        annotator = FailingAnnotator()

        result = annotator.annotate_batch(
            image_dir=str(tmp_path),
            class_id=0,
            output_dir=str(output_dir),
        )

        # Check results
        assert result.total_images == 3
        assert result.successful == 0
        assert result.failed == 3
        assert len(result.failed_paths) == 3

    def test_progress_callback(self, tmp_path):
        """Test progress callback is called correctly."""
        # Create test images
        for i in range(5):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(tmp_path / f"image_{i}.jpg"), img)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        annotator = ConcreteAnnotator(return_bbox=(0.5, 0.5, 0.2, 0.2))

        # Track callback calls
        callback_calls = []

        def progress_callback(current, total):
            callback_calls.append((current, total))

        result = annotator.annotate_batch(
            image_dir=str(tmp_path),
            class_id=0,
            output_dir=str(output_dir),
            progress_callback=progress_callback,
        )

        # Check that callback was called for each image
        assert len(callback_calls) == 5
        # Check the progression
        for i, (current, total) in enumerate(callback_calls):
            assert current == i + 1
            assert total == 5

    @patch("annotation.base_annotator.cv2")
    @patch("annotation.base_annotator.visualize_annotation_result")
    def test_visualize_annotation(self, mock_visualize, mock_cv2, tmp_path):
        """Test annotation visualization."""
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = test_image

        mock_visualize.return_value = test_image

        annotator = ConcreteAnnotator(return_bbox=(0.5, 0.5, 0.2, 0.2))

        result = annotator.visualize_annotation(
            image_path=str(tmp_path / "test.jpg"),
            show=False,
        )

        assert result is not None
        mock_cv2.imread.assert_called_once()
        mock_visualize.assert_called_once()

    @patch("annotation.base_annotator.cv2")
    def test_visualize_annotation_saves_output(self, mock_cv2, tmp_path):
        """Test that visualization saves output when path is provided."""
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cv2.imread.return_value = test_image

        annotator = ConcreteAnnotator(return_bbox=(0.5, 0.5, 0.2, 0.2))

        with patch("annotation.base_annotator.visualize_annotation_result") as mock_viz:
            mock_viz.return_value = test_image
            output_path = str(tmp_path / "output.jpg")

            result = annotator.visualize_annotation(
                image_path=str(tmp_path / "test.jpg"),
                output_path=output_path,
                show=False,
            )

            mock_cv2.imwrite.assert_called_once()

    @patch("annotation.base_annotator.cv2")
    def test_visualize_annotation_file_not_found(self, mock_cv2, tmp_path):
        """Test visualization raises error for missing file."""
        mock_cv2.imread.return_value = None

        annotator = ConcreteAnnotator()

        with pytest.raises(ValueError, match="Failed to load image"):
            annotator.visualize_annotation(
                image_path=str(tmp_path / "nonexistent.jpg"),
                show=False,
            )


class TestAnnotationResult:
    """Test AnnotationResult dataclass used by annotators."""

    def test_annotation_result_creation(self):
        """Test creating AnnotationResult."""
        from annotation.annotation_utils import AnnotationResult

        result = AnnotationResult(total_images=10)
        result.successful = 8
        result.failed = 2

        assert result.total_images == 10
        assert result.successful == 8
        assert result.failed == 2

    def test_annotation_result_summary(self):
        """Test AnnotationResult summary."""
        from annotation.annotation_utils import AnnotationResult

        result = AnnotationResult(total_images=10)
        result.successful = 8
        result.failed = 2
        result.failed_paths = ["/path/to/failed1.jpg", "/path/to/failed2.jpg"]

        summary = result.summary()

        assert "10" in summary
        assert "8" in summary
        assert "2" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
