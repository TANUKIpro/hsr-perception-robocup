"""
HSR Perception - Validation Utilities

Unified validation and error handling for pipeline operations.
Provides structured error reporting and validation result objects.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from colorama import Fore, Style


class ErrorSeverity(Enum):
    """Severity levels for pipeline errors."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PipelineError:
    """
    Structured error for pipeline operations.

    Attributes:
        message: Error description
        severity: Error severity level
        source: Source file or component that raised the error
        details: Additional details or context
    """

    message: str
    severity: ErrorSeverity = ErrorSeverity.ERROR
    source: str = ""
    details: Optional[str] = None

    def format(self, use_color: bool = True) -> str:
        """
        Format error message with optional color.

        Args:
            use_color: If True, add ANSI color codes

        Returns:
            Formatted error string
        """
        prefix_map = {
            ErrorSeverity.INFO: (Fore.BLUE, "[INFO]"),
            ErrorSeverity.WARNING: (Fore.YELLOW, "[WARNING]"),
            ErrorSeverity.ERROR: (Fore.RED, "[ERROR]"),
            ErrorSeverity.CRITICAL: (Fore.RED + Style.BRIGHT, "[CRITICAL]"),
        }

        color, prefix = prefix_map.get(self.severity, (Fore.WHITE, "[UNKNOWN]"))

        source_str = f" ({self.source})" if self.source else ""
        details_str = f"\n  Details: {self.details}" if self.details else ""

        if use_color:
            return f"{color}{prefix}{Style.RESET_ALL}{source_str}: {self.message}{details_str}"
        else:
            return f"{prefix}{source_str}: {self.message}{details_str}"


@dataclass
class ValidationResult:
    """
    Result of a validation operation.

    Provides a consistent interface for reporting validation outcomes
    with both errors and warnings.
    """

    is_valid: bool = True
    errors: List[PipelineError] = field(default_factory=list)
    warnings: List[PipelineError] = field(default_factory=list)

    def add_error(
        self,
        message: str,
        source: str = "",
        details: Optional[str] = None,
    ) -> None:
        """Add an error and mark result as invalid."""
        self.is_valid = False
        self.errors.append(
            PipelineError(message, ErrorSeverity.ERROR, source, details)
        )

    def add_warning(
        self,
        message: str,
        source: str = "",
        details: Optional[str] = None,
    ) -> None:
        """Add a warning (does not affect validity)."""
        self.warnings.append(
            PipelineError(message, ErrorSeverity.WARNING, source, details)
        )

    def add_info(
        self,
        message: str,
        source: str = "",
        details: Optional[str] = None,
    ) -> None:
        """Add an info message."""
        self.warnings.append(
            PipelineError(message, ErrorSeverity.INFO, source, details)
        )

    def merge(self, other: "ValidationResult") -> None:
        """Merge another ValidationResult into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

    def format_all(self, use_color: bool = True) -> str:
        """
        Format all errors and warnings.

        Args:
            use_color: If True, add ANSI color codes

        Returns:
            Formatted string with all messages
        """
        lines = []
        for error in self.errors:
            lines.append(error.format(use_color))
        for warning in self.warnings:
            lines.append(warning.format(use_color))
        return "\n".join(lines)

    def print_all(self, use_color: bool = True) -> None:
        """Print all errors and warnings to stdout."""
        formatted = self.format_all(use_color)
        if formatted:
            print(formatted)


def validate_dataset_yaml(dataset_yaml: str) -> ValidationResult:
    """
    Validate a YOLO dataset configuration file.

    Checks:
    - File exists
    - Required fields present (train, val, names)
    - Train and val paths exist
    - At least one class defined

    Args:
        dataset_yaml: Path to dataset YAML file

    Returns:
        ValidationResult with any errors or warnings
    """
    result = ValidationResult()
    dataset_path = Path(dataset_yaml)

    # Check file exists
    if not dataset_path.exists():
        result.add_error(
            f"Dataset config not found: {dataset_yaml}",
            source="validate_dataset_yaml",
        )
        return result

    # Load and parse YAML
    try:
        with open(dataset_path, "r") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        result.add_error(
            f"Failed to parse YAML: {e}",
            source="validate_dataset_yaml",
        )
        return result

    if config is None:
        result.add_error(
            "Empty dataset configuration",
            source="validate_dataset_yaml",
        )
        return result

    # Check required fields
    required_fields = ["train", "val", "names"]
    for field_name in required_fields:
        if field_name not in config:
            result.add_error(
                f"Missing required field: '{field_name}'",
                source="validate_dataset_yaml",
            )

    if not result.is_valid:
        return result

    # Check paths exist
    base_path = dataset_path.parent
    train_path = base_path / config["train"]
    val_path = base_path / config["val"]

    if not train_path.exists():
        result.add_error(
            f"Train path not found: {train_path}",
            source="validate_dataset_yaml",
        )

    if not val_path.exists():
        result.add_error(
            f"Val path not found: {val_path}",
            source="validate_dataset_yaml",
        )

    # Check classes
    names = config.get("names", {})
    if isinstance(names, dict):
        num_classes = len(names)
    elif isinstance(names, list):
        num_classes = len(names)
    else:
        num_classes = 0

    if num_classes == 0:
        result.add_error(
            "No classes defined in 'names' field",
            source="validate_dataset_yaml",
        )
    elif num_classes < 2:
        result.add_warning(
            f"Only {num_classes} class defined",
            source="validate_dataset_yaml",
            details="Consider adding more classes for robust training",
        )

    # Count images if paths exist
    if train_path.exists():
        train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
        if len(train_images) == 0:
            result.add_error(
                "No images found in train directory",
                source="validate_dataset_yaml",
                details=str(train_path),
            )
        elif len(train_images) < 50:
            result.add_warning(
                f"Only {len(train_images)} training images found",
                source="validate_dataset_yaml",
                details="Recommend at least 50 images per class for good results",
            )

    return result


def validate_yolo_annotation(
    label_path: str,
    image_width: int,
    image_height: int,
) -> Tuple[bool, List[str]]:
    """
    Validate a YOLO format annotation file.

    Args:
        label_path: Path to .txt label file
        image_width: Width of corresponding image
        image_height: Height of corresponding image

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    label_path = Path(label_path)

    if not label_path.exists():
        return False, ["Label file not found"]

    try:
        with open(label_path, "r") as f:
            lines = f.readlines()
    except Exception as e:
        return False, [f"Failed to read label file: {e}"]

    if not lines:
        return True, []  # Empty file is valid (no objects)

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {i}: Expected 5 values, got {len(parts)}")
            continue

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
        except ValueError as e:
            errors.append(f"Line {i}: Invalid number format: {e}")
            continue

        if class_id < 0:
            errors.append(f"Line {i}: Negative class ID: {class_id}")

        if not (0 <= x_center <= 1):
            errors.append(f"Line {i}: x_center out of range: {x_center}")
        if not (0 <= y_center <= 1):
            errors.append(f"Line {i}: y_center out of range: {y_center}")
        if not (0 < width <= 1):
            errors.append(f"Line {i}: width out of range: {width}")
        if not (0 < height <= 1):
            errors.append(f"Line {i}: height out of range: {height}")

    return len(errors) == 0, errors


def validate_model_path(model_path: str) -> ValidationResult:
    """
    Validate a model file path.

    Args:
        model_path: Path to model file

    Returns:
        ValidationResult
    """
    result = ValidationResult()
    path = Path(model_path)

    if not path.exists():
        # Check if it's a YOLO model name that will be downloaded
        yolo_models = [
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt",
        ]
        if path.name in yolo_models:
            result.add_info(
                f"Model '{path.name}' will be downloaded automatically",
                source="validate_model_path",
            )
        else:
            result.add_error(
                f"Model file not found: {model_path}",
                source="validate_model_path",
            )
    else:
        # Check file extension
        valid_extensions = [".pt", ".pth", ".onnx", ".engine"]
        if path.suffix.lower() not in valid_extensions:
            result.add_warning(
                f"Unexpected model file extension: {path.suffix}",
                source="validate_model_path",
                details=f"Expected one of: {valid_extensions}",
            )

    return result
