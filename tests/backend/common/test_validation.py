"""
Tests for validation utilities.

Tests the unified validation and error handling for pipeline operations.
"""

import sys
from pathlib import Path

import pytest
import yaml

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent.parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from common.validation import (
    ErrorSeverity,
    PipelineError,
    ValidationResult,
    validate_dataset_yaml,
    validate_model_path,
    validate_yolo_annotation,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_enum_values(self):
        """Test that all expected enum values exist."""
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_enum_count(self):
        """Test that there are exactly 4 severity levels."""
        assert len(ErrorSeverity) == 4


class TestPipelineError:
    """Test PipelineError dataclass."""

    def test_creation_with_defaults(self):
        """Test creating PipelineError with default values."""
        error = PipelineError(message="Test error")

        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.ERROR
        assert error.source == ""
        assert error.details is None

    def test_creation_with_all_fields(self):
        """Test creating PipelineError with all fields specified."""
        error = PipelineError(
            message="Test error",
            severity=ErrorSeverity.WARNING,
            source="test_module",
            details="Additional info",
        )

        assert error.message == "Test error"
        assert error.severity == ErrorSeverity.WARNING
        assert error.source == "test_module"
        assert error.details == "Additional info"

    def test_format_without_color(self):
        """Test format() without color codes."""
        error = PipelineError(
            message="Something went wrong",
            severity=ErrorSeverity.ERROR,
            source="validator",
        )

        formatted = error.format(use_color=False)

        assert "[ERROR]" in formatted
        assert "(validator)" in formatted
        assert "Something went wrong" in formatted

    def test_format_with_details(self):
        """Test format() includes details when present."""
        error = PipelineError(
            message="File error",
            severity=ErrorSeverity.CRITICAL,
            details="File not found at /path/to/file",
        )

        formatted = error.format(use_color=False)

        assert "[CRITICAL]" in formatted
        assert "File error" in formatted
        assert "Details: File not found at /path/to/file" in formatted

    def test_format_with_color(self):
        """Test format() with color codes."""
        error = PipelineError(message="Warning", severity=ErrorSeverity.WARNING)

        formatted = error.format(use_color=True)

        # Should contain ANSI color codes
        assert "\x1b[" in formatted or "[WARNING]" in formatted

    def test_format_info_severity(self):
        """Test format() for INFO severity."""
        error = PipelineError(message="Info message", severity=ErrorSeverity.INFO)

        formatted = error.format(use_color=False)

        assert "[INFO]" in formatted


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_default_is_valid(self):
        """Test that default result is valid."""
        result = ValidationResult()

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_add_error_invalidates(self):
        """Test that adding an error marks result as invalid."""
        result = ValidationResult()
        result.add_error("Something failed")

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].message == "Something failed"
        assert result.errors[0].severity == ErrorSeverity.ERROR

    def test_add_warning_keeps_valid(self):
        """Test that adding a warning does not invalidate result."""
        result = ValidationResult()
        result.add_warning("Minor issue")

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].message == "Minor issue"
        assert result.warnings[0].severity == ErrorSeverity.WARNING

    def test_add_info(self):
        """Test adding info message."""
        result = ValidationResult()
        result.add_info("Note: something happened", source="test")

        assert result.is_valid is True
        assert len(result.warnings) == 1
        assert result.warnings[0].severity == ErrorSeverity.INFO

    def test_add_error_with_source_and_details(self):
        """Test adding error with source and details."""
        result = ValidationResult()
        result.add_error(
            message="Parse error",
            source="yaml_parser",
            details="Unexpected token at line 5",
        )

        assert result.errors[0].source == "yaml_parser"
        assert result.errors[0].details == "Unexpected token at line 5"

    def test_merge_results(self):
        """Test merging two ValidationResults."""
        result1 = ValidationResult()
        result1.add_warning("Warning 1")

        result2 = ValidationResult()
        result2.add_error("Error 1")
        result2.add_warning("Warning 2")

        result1.merge(result2)

        assert result1.is_valid is False  # Because result2 had an error
        assert len(result1.errors) == 1
        assert len(result1.warnings) == 2

    def test_merge_valid_results(self):
        """Test merging two valid results stays valid."""
        result1 = ValidationResult()
        result1.add_warning("Warning 1")

        result2 = ValidationResult()
        result2.add_warning("Warning 2")

        result1.merge(result2)

        assert result1.is_valid is True
        assert len(result1.warnings) == 2

    def test_format_all(self):
        """Test format_all() includes all errors and warnings."""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_warning("Warning 1")

        formatted = result.format_all(use_color=False)

        assert "Error 1" in formatted
        assert "Warning 1" in formatted

    def test_format_all_empty(self):
        """Test format_all() with no errors or warnings."""
        result = ValidationResult()

        formatted = result.format_all(use_color=False)

        assert formatted == ""


class TestValidateDatasetYaml:
    """Test validate_dataset_yaml function."""

    def test_file_not_found(self, tmp_path):
        """Test validation when file does not exist."""
        result = validate_dataset_yaml(str(tmp_path / "nonexistent.yaml"))

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "not found" in result.errors[0].message.lower()

    def test_invalid_yaml_format(self, tmp_path):
        """Test validation with malformed YAML."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: content:")

        result = validate_dataset_yaml(str(yaml_file))

        assert result.is_valid is False
        assert any("parse" in e.message.lower() or "yaml" in e.message.lower()
                  for e in result.errors)

    def test_empty_yaml(self, tmp_path):
        """Test validation with empty YAML file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        result = validate_dataset_yaml(str(yaml_file))

        assert result.is_valid is False
        assert any("empty" in e.message.lower() for e in result.errors)

    def test_missing_required_fields(self, tmp_path):
        """Test validation with missing required fields."""
        yaml_file = tmp_path / "partial.yaml"
        yaml_file.write_text("path: .\n")

        result = validate_dataset_yaml(str(yaml_file))

        assert result.is_valid is False
        # Should report missing train, val, names
        missing_fields = [e.message for e in result.errors if "missing" in e.message.lower()]
        assert len(missing_fields) >= 1

    def test_train_path_not_found(self, tmp_path):
        """Test validation when train path does not exist."""
        yaml_file = tmp_path / "data.yaml"
        config = {
            "path": ".",
            "train": "nonexistent/train",
            "val": "nonexistent/val",
            "names": {0: "class0"},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        result = validate_dataset_yaml(str(yaml_file))

        assert result.is_valid is False
        assert any("train" in e.message.lower() and "not found" in e.message.lower()
                  for e in result.errors)

    def test_no_classes_defined(self, tmp_path):
        """Test validation with empty names field."""
        # Create directory structure
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        yaml_file = tmp_path / "data.yaml"
        config = {
            "path": ".",
            "train": "train",
            "val": "val",
            "names": {},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        result = validate_dataset_yaml(str(yaml_file))

        assert result.is_valid is False
        assert any("class" in e.message.lower() and "no" in e.message.lower()
                  for e in result.errors)

    def test_warning_single_class(self, tmp_path):
        """Test warning when only one class is defined."""
        # Create directory structure with images
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # Create dummy images
        for i in range(60):
            (train_dir / f"img{i:03d}.jpg").touch()

        yaml_file = tmp_path / "data.yaml"
        config = {
            "path": ".",
            "train": "train",
            "val": "val",
            "names": {0: "only_class"},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        result = validate_dataset_yaml(str(yaml_file))

        # Should have a warning about single class
        assert any("1" in w.message and "class" in w.message.lower()
                  for w in result.warnings)

    def test_warning_few_images(self, tmp_path):
        """Test warning when too few training images."""
        # Create directory structure
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # Create only a few images
        for i in range(10):
            (train_dir / f"img{i:03d}.jpg").touch()

        yaml_file = tmp_path / "data.yaml"
        config = {
            "path": ".",
            "train": "train",
            "val": "val",
            "names": {0: "class0", 1: "class1"},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        result = validate_dataset_yaml(str(yaml_file))

        # Should have a warning about few images
        assert any("10" in w.message or "image" in w.message.lower()
                  for w in result.warnings)

    def test_valid_dataset(self, tmp_path):
        """Test validation with valid dataset configuration."""
        # Create directory structure with sufficient images
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        # Create enough images
        for i in range(60):
            (train_dir / f"img{i:03d}.jpg").touch()
        for i in range(20):
            (val_dir / f"val{i:03d}.jpg").touch()

        yaml_file = tmp_path / "data.yaml"
        config = {
            "path": ".",
            "train": "train",
            "val": "val",
            "names": {0: "apple", 1: "banana"},
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        result = validate_dataset_yaml(str(yaml_file))

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_names_as_list(self, tmp_path):
        """Test validation with names as list instead of dict."""
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        train_dir.mkdir()
        val_dir.mkdir()

        for i in range(60):
            (train_dir / f"img{i:03d}.jpg").touch()

        yaml_file = tmp_path / "data.yaml"
        config = {
            "path": ".",
            "train": "train",
            "val": "val",
            "names": ["apple", "banana"],  # List format
        }
        with open(yaml_file, "w") as f:
            yaml.dump(config, f)

        result = validate_dataset_yaml(str(yaml_file))

        assert result.is_valid is True


class TestValidateYoloAnnotation:
    """Test validate_yolo_annotation function."""

    def test_file_not_found(self, tmp_path):
        """Test validation when label file does not exist."""
        is_valid, errors = validate_yolo_annotation(
            str(tmp_path / "nonexistent.txt"), 640, 480
        )

        assert is_valid is False
        assert any("not found" in e.lower() for e in errors)

    def test_empty_file_is_valid(self, tmp_path):
        """Test that empty label file is valid (no objects)."""
        label_file = tmp_path / "empty.txt"
        label_file.write_text("")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is True
        assert len(errors) == 0

    def test_wrong_number_of_values(self, tmp_path):
        """Test validation with wrong number of values per line."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("0 0.5 0.5 0.2\n")  # Only 4 values instead of 5

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert any("5" in e for e in errors)

    def test_invalid_number_format(self, tmp_path):
        """Test validation with non-numeric values."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("0 abc 0.5 0.2 0.3\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert any("invalid" in e.lower() or "format" in e.lower() for e in errors)

    def test_negative_class_id(self, tmp_path):
        """Test validation with negative class ID."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("-1 0.5 0.5 0.2 0.3\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert any("negative" in e.lower() or "-1" in e for e in errors)

    def test_out_of_range_x_center(self, tmp_path):
        """Test validation with x_center out of range."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("0 1.5 0.5 0.2 0.3\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert any("x_center" in e for e in errors)

    def test_out_of_range_y_center(self, tmp_path):
        """Test validation with y_center out of range."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("0 0.5 -0.1 0.2 0.3\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert any("y_center" in e for e in errors)

    def test_zero_width(self, tmp_path):
        """Test validation with zero width (invalid)."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("0 0.5 0.5 0.0 0.3\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert any("width" in e for e in errors)

    def test_zero_height(self, tmp_path):
        """Test validation with zero height (invalid)."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.0\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert any("height" in e for e in errors)

    def test_valid_annotation(self, tmp_path):
        """Test validation with valid annotation."""
        label_file = tmp_path / "valid.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.15 0.25\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is True
        assert len(errors) == 0

    def test_multiple_errors(self, tmp_path):
        """Test validation reports multiple errors."""
        label_file = tmp_path / "invalid.txt"
        label_file.write_text("0 1.5 0.5 0.2 0.3\n0 0.5 -0.1 0.2 0.3\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is False
        assert len(errors) >= 2

    def test_blank_lines_ignored(self, tmp_path):
        """Test that blank lines are ignored."""
        label_file = tmp_path / "valid.txt"
        label_file.write_text("0 0.5 0.5 0.2 0.3\n\n1 0.3 0.7 0.15 0.25\n")

        is_valid, errors = validate_yolo_annotation(str(label_file), 640, 480)

        assert is_valid is True
        assert len(errors) == 0


class TestValidateModelPath:
    """Test validate_model_path function."""

    def test_model_not_found(self, tmp_path):
        """Test validation when model file does not exist."""
        result = validate_model_path(str(tmp_path / "nonexistent_model.pt"))

        assert result.is_valid is False
        assert any("not found" in e.message.lower() for e in result.errors)

    def test_yolo_model_auto_download_info(self, tmp_path):
        """Test that standard YOLO models show auto-download info when not present."""
        # Use a non-existent but valid YOLO model name to force the info message
        # Note: In Docker, yolov8m.pt is pre-downloaded, so we use a different approach
        from pathlib import Path

        # Create a temp directory context where yolov8m.pt doesn't exist
        result = validate_model_path(str(tmp_path / "yolov8m.pt"))

        # If file doesn't exist but name is a known YOLO model, should be valid
        assert result.is_valid is True
        # Should have an info message about auto-download
        assert any("download" in w.message.lower() for w in result.warnings)

    def test_yolo_model_variants(self):
        """Test various YOLO model names are recognized."""
        models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]

        for model in models:
            result = validate_model_path(model)
            assert result.is_valid is True

    def test_unexpected_extension_warning(self, tmp_path):
        """Test warning for unexpected file extension."""
        model_file = tmp_path / "model.xyz"
        model_file.touch()

        result = validate_model_path(str(model_file))

        # Should be valid but with warning
        assert result.is_valid is True
        assert any("extension" in w.message.lower() for w in result.warnings)

    def test_valid_model_pt(self, tmp_path):
        """Test validation with valid .pt file."""
        model_file = tmp_path / "custom_model.pt"
        model_file.touch()

        result = validate_model_path(str(model_file))

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_valid_model_onnx(self, tmp_path):
        """Test validation with valid .onnx file."""
        model_file = tmp_path / "model.onnx"
        model_file.touch()

        result = validate_model_path(str(model_file))

        assert result.is_valid is True
        assert len(result.warnings) == 0  # No warning for valid extension

    def test_valid_model_engine(self, tmp_path):
        """Test validation with valid .engine file (TensorRT)."""
        model_file = tmp_path / "model.engine"
        model_file.touch()

        result = validate_model_path(str(model_file))

        assert result.is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
