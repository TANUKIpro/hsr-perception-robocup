"""
Tests for auto-annotation pipeline.

Tests the main orchestration script for auto-annotation workflow.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


class TestAnnotationReport:
    """Test AnnotationReport dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from annotation.auto_annotate import AnnotationReport
        from annotation.annotation_utils import AnnotationResult

        report = AnnotationReport(
            timestamp="2024-01-01 00:00:00",
            method="background",
            total_classes=2,
            total_images=100,
            successful=80,
            failed=20,
            class_results={},
            dataset_path="/path/to/dataset",
            train_count=70,
            val_count=30,
        )

        assert report.success_rate == 80.0

    def test_success_rate_zero_images(self):
        """Test success rate with zero images."""
        from annotation.auto_annotate import AnnotationReport

        report = AnnotationReport(
            timestamp="2024-01-01 00:00:00",
            method="background",
            total_classes=0,
            total_images=0,
            successful=0,
            failed=0,
            class_results={},
            dataset_path="/path/to/dataset",
            train_count=0,
            val_count=0,
        )

        assert report.success_rate == 0.0

    def test_summary_generation(self):
        """Test summary report generation."""
        from annotation.auto_annotate import AnnotationReport
        from annotation.annotation_utils import AnnotationResult

        result = AnnotationResult(total_images=50)
        result.successful = 45
        result.failed = 5

        report = AnnotationReport(
            timestamp="2024-01-01 12:00:00",
            method="background",
            total_classes=1,
            total_images=50,
            successful=45,
            failed=5,
            class_results={"test_class": result},
            dataset_path="/path/to/dataset",
            train_count=40,
            val_count=10,
        )

        summary = report.summary()

        # Check summary contains expected information
        assert "Auto-Annotation Report" in summary
        assert "background" in summary
        assert "50" in summary  # total images
        assert "45" in summary  # successful
        assert "test_class" in summary


class TestAutoAnnotator:
    """Test AutoAnnotator class."""

    def test_init_background_method(self, tmp_path):
        """Test initialization with background method."""
        from annotation.auto_annotate import AutoAnnotator

        bg_path = tmp_path / "background.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_path), img)

        annotator = AutoAnnotator(
            method="background",
            background_path=str(bg_path),
        )

        assert annotator.method == "background"
        assert annotator.background_path == str(bg_path)
        assert annotator._annotator is None  # Lazy initialization

    def test_init_sam2_method(self):
        """Test initialization with SAM2 method."""
        from annotation.auto_annotate import AutoAnnotator

        annotator = AutoAnnotator(
            method="sam2",
            sam2_model_path="models/sam2_b.pt",
        )

        assert annotator.method == "sam2"
        assert annotator.sam2_model_path == "models/sam2_b.pt"

    def test_init_invalid_method(self):
        """Test initialization with invalid method raises error."""
        from annotation.auto_annotate import AutoAnnotator

        annotator = AutoAnnotator(method="invalid")

        with pytest.raises(ValueError, match="Unknown method"):
            annotator._get_annotator()

    def test_get_annotator_background_requires_path(self):
        """Test that background method requires background path."""
        from annotation.auto_annotate import AutoAnnotator

        annotator = AutoAnnotator(method="background", background_path=None)

        with pytest.raises(ValueError, match="Background path required"):
            annotator._get_annotator()

    def test_get_annotator_background_success(self, tmp_path):
        """Test successful annotator initialization for background method."""
        from annotation.auto_annotate import AutoAnnotator

        # Create background image
        bg_path = tmp_path / "background.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_path), img)

        annotator = AutoAnnotator(
            method="background",
            background_path=str(bg_path),
        )

        # Get annotator (triggers lazy initialization)
        bg_annotator = annotator._get_annotator()

        assert bg_annotator is not None
        # Second call should return the same instance
        assert annotator._get_annotator() is bg_annotator

    def test_get_annotator_sam2_import_error(self):
        """Test that SAM2 method handles import error gracefully."""
        from annotation.auto_annotate import AutoAnnotator

        annotator = AutoAnnotator(method="sam2")

        # Mock the import to fail
        with patch.dict("sys.modules", {"sam2_annotator": None}):
            # Need to force the import to fail by removing the module
            # and simulating ImportError
            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "sam2_annotator":
                    raise ImportError("No module named 'sam2_annotator'")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                with pytest.raises(ImportError, match="SAM2 annotator not available"):
                    annotator._get_annotator()

    @patch("annotation.auto_annotate.BackgroundSubtractionAnnotator")
    @patch("annotation.auto_annotate.load_class_config")
    @patch("annotation.auto_annotate.get_class_names")
    @patch("annotation.auto_annotate.split_dataset")
    @patch("annotation.auto_annotate.create_dataset_yaml")
    def test_run_pipeline(
        self,
        mock_create_yaml,
        mock_split,
        mock_get_names,
        mock_load_config,
        mock_annotator_class,
        tmp_path,
    ):
        """Test complete annotation pipeline run."""
        from annotation.auto_annotate import AutoAnnotator

        # Setup mocks
        mock_load_config.return_value = {
            "objects": [{"class_name": "test_class", "class_id": 0}]
        }
        mock_get_names.return_value = ["test_class"]
        mock_split.return_value = {"train": 8, "val": 2}

        # Create mock annotator
        mock_annotator = MagicMock()
        mock_annotator.annotate_image.return_value = (0.5, 0.5, 0.2, 0.2)
        mock_annotator_class.return_value = mock_annotator

        # Create input directory structure
        input_dir = tmp_path / "input"
        class_dir = input_dir / "test_class"
        class_dir.mkdir(parents=True)

        # Create test images
        for i in range(10):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"image_{i}.jpg"), img)

        # Create config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"objects": [{"class_name": "test_class", "class_id": 0}]}, f)

        # Create output directory
        output_dir = tmp_path / "output"

        # Create background image
        bg_path = tmp_path / "background.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_path), img)

        # Run pipeline
        annotator = AutoAnnotator(
            method="background",
            background_path=str(bg_path),
        )

        report = annotator.run(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            class_config_path=str(config_path),
            verify=False,
            update_config=False,
        )

        # Check report
        assert report.method == "background"
        assert report.total_classes == 1
        assert report.total_images == 10
        assert report.successful == 10
        assert report.failed == 0
        assert report.train_count == 8
        assert report.val_count == 2

    @patch("annotation.auto_annotate.BackgroundSubtractionAnnotator")
    @patch("annotation.auto_annotate.load_class_config")
    @patch("annotation.auto_annotate.get_class_names")
    @patch("annotation.auto_annotate.split_dataset")
    @patch("annotation.auto_annotate.create_dataset_yaml")
    def test_run_with_failures(
        self,
        mock_create_yaml,
        mock_split,
        mock_get_names,
        mock_load_config,
        mock_annotator_class,
        tmp_path,
    ):
        """Test pipeline handles annotation failures."""
        from annotation.auto_annotate import AutoAnnotator

        # Setup mocks
        mock_load_config.return_value = {
            "objects": [{"class_name": "test_class", "class_id": 0}]
        }
        mock_get_names.return_value = ["test_class"]
        mock_split.return_value = {"train": 4, "val": 1}

        # Create mock annotator that fails half the time
        mock_annotator = MagicMock()
        # Alternates between success and failure
        mock_annotator.annotate_image.side_effect = [
            (0.5, 0.5, 0.2, 0.2),  # Success
            None,  # Fail
            (0.5, 0.5, 0.2, 0.2),  # Success
            None,  # Fail
            (0.5, 0.5, 0.2, 0.2),  # Success
        ]
        mock_annotator_class.return_value = mock_annotator

        # Create input directory structure
        input_dir = tmp_path / "input"
        class_dir = input_dir / "test_class"
        class_dir.mkdir(parents=True)

        # Create test images
        for i in range(5):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(class_dir / f"image_{i}.jpg"), img)

        # Create config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"objects": [{"class_name": "test_class", "class_id": 0}]}, f)

        output_dir = tmp_path / "output"

        bg_path = tmp_path / "background.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_path), img)

        annotator = AutoAnnotator(
            method="background",
            background_path=str(bg_path),
        )

        report = annotator.run(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            class_config_path=str(config_path),
            verify=False,
            update_config=False,
        )

        # Check report reflects failures
        assert report.total_images == 5
        assert report.successful == 3
        assert report.failed == 2
        assert report.success_rate == 60.0

    def test_run_empty_class_directory(self, tmp_path):
        """Test pipeline handles empty class directories."""
        from annotation.auto_annotate import AutoAnnotator

        # Create input directory structure with empty class dir
        input_dir = tmp_path / "input"
        class_dir = input_dir / "empty_class"
        class_dir.mkdir(parents=True)

        # Create config file
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"objects": [{"class_name": "empty_class", "class_id": 0}]}, f)

        output_dir = tmp_path / "output"

        bg_path = tmp_path / "background.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_path), img)

        annotator = AutoAnnotator(
            method="background",
            background_path=str(bg_path),
        )

        report = annotator.run(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            class_config_path=str(config_path),
            verify=False,
            update_config=False,
        )

        # Report should show 0 images
        assert report.total_images == 0
        assert report.total_classes == 0

    def test_run_missing_class_directory(self, tmp_path):
        """Test pipeline handles missing class directories."""
        from annotation.auto_annotate import AutoAnnotator

        # Create input directory structure without the class directory
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True)

        # Create config file with a class that doesn't have a directory
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"objects": [{"class_name": "missing_class", "class_id": 0}]}, f)

        output_dir = tmp_path / "output"

        bg_path = tmp_path / "background.jpg"
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(bg_path), img)

        annotator = AutoAnnotator(
            method="background",
            background_path=str(bg_path),
        )

        # Should not raise error, just skip the missing class
        report = annotator.run(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            class_config_path=str(config_path),
            verify=False,
            update_config=False,
        )

        assert report.total_images == 0
        assert report.total_classes == 0


class TestProgressReporting:
    """Test progress reporting during annotation."""

    @patch("annotation.auto_annotate.BackgroundSubtractionAnnotator")
    @patch("annotation.auto_annotate.load_class_config")
    @patch("annotation.auto_annotate.get_class_names")
    @patch("annotation.auto_annotate.split_dataset")
    @patch("annotation.auto_annotate.create_dataset_yaml")
    def test_progress_output(
        self,
        mock_create_yaml,
        mock_split,
        mock_get_names,
        mock_load_config,
        mock_annotator_class,
        tmp_path,
        capsys,
    ):
        """Test that progress output is generated."""
        from annotation.auto_annotate import AutoAnnotator

        mock_load_config.return_value = {
            "objects": [{"class_name": "test_class", "class_id": 0}]
        }
        mock_get_names.return_value = ["test_class"]
        mock_split.return_value = {"train": 1, "val": 1}

        mock_annotator = MagicMock()
        mock_annotator.annotate_image.return_value = (0.5, 0.5, 0.2, 0.2)
        mock_annotator_class.return_value = mock_annotator

        input_dir = tmp_path / "input"
        class_dir = input_dir / "test_class"
        class_dir.mkdir(parents=True)

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(class_dir / "image.jpg"), img)

        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump({"objects": [{"class_name": "test_class", "class_id": 0}]}, f)

        output_dir = tmp_path / "output"

        bg_path = tmp_path / "background.jpg"
        cv2.imwrite(str(bg_path), img)

        annotator = AutoAnnotator(method="background", background_path=str(bg_path))

        annotator.run(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            class_config_path=str(config_path),
            verify=False,
            update_config=False,
        )

        captured = capsys.readouterr()
        # Check that class processing was logged
        assert "test_class" in captured.out
        assert "annotated" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
