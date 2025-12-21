"""
Dataset validation for YOLOv8 training.

Validates dataset YAML configuration and directory structure.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml
from colorama import Fore, Style


@dataclass
class DatasetValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    train_count: int = 0
    val_count: int = 0
    class_count: int = 0
    error_message: Optional[str] = None


class DatasetValidator:
    """Validates YOLO dataset configuration."""

    def __init__(self, verbose: bool = True):
        """
        Initialize validator.

        Args:
            verbose: Whether to print validation messages
        """
        self.verbose = verbose

    def validate(self, dataset_yaml: str) -> DatasetValidationResult:
        """
        Validate dataset configuration.

        Args:
            dataset_yaml: Path to dataset YAML file

        Returns:
            DatasetValidationResult with validation status and statistics
        """
        dataset_path = Path(dataset_yaml)

        # Check file exists
        if not dataset_path.exists():
            error_msg = f"Dataset config not found: {dataset_yaml}"
            if self.verbose:
                print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            return DatasetValidationResult(is_valid=False, error_message=error_msg)

        # Load and parse YAML
        try:
            with open(dataset_path, "r") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse YAML: {e}"
            if self.verbose:
                print(f"{Fore.RED}Error: {error_msg}{Style.RESET_ALL}")
            return DatasetValidationResult(is_valid=False, error_message=error_msg)

        # Check required fields
        error = self._check_required_fields(config)
        if error:
            if self.verbose:
                print(f"{Fore.RED}Error: {error}{Style.RESET_ALL}")
            return DatasetValidationResult(is_valid=False, error_message=error)

        # Check paths exist
        base_path = dataset_path.parent
        error = self._check_paths_exist(base_path, config)
        if error:
            if self.verbose:
                print(f"{Fore.RED}Error: {error}{Style.RESET_ALL}")
            return DatasetValidationResult(is_valid=False, error_message=error)

        # Count images
        train_path = base_path / config["train"]
        val_path = base_path / config["val"]
        train_count = sum(1 for _ in train_path.iterdir())
        val_count = sum(1 for _ in val_path.iterdir())
        class_count = len(config["names"])

        if self.verbose:
            print(f"Dataset validated:")
            print(f"  Train images: {train_count}")
            print(f"  Val images: {val_count}")
            print(f"  Classes: {class_count}")

        return DatasetValidationResult(
            is_valid=True,
            train_count=train_count,
            val_count=val_count,
            class_count=class_count,
        )

    def _check_required_fields(self, config: Dict) -> Optional[str]:
        """
        Check for required YAML fields.

        Args:
            config: Parsed YAML configuration

        Returns:
            Error message if validation fails, None otherwise
        """
        required = ["train", "val", "names"]
        for field in required:
            if field not in config:
                return f"Missing field '{field}' in dataset config"
        return None

    def _check_paths_exist(self, base_path: Path, config: Dict) -> Optional[str]:
        """
        Verify train/val paths exist.

        Args:
            base_path: Base directory for relative paths
            config: Parsed YAML configuration

        Returns:
            Error message if validation fails, None otherwise
        """
        train_path = base_path / config["train"]
        val_path = base_path / config["val"]

        if not train_path.exists():
            return f"Train path not found: {train_path}"

        if not val_path.exists():
            return f"Val path not found: {val_path}"

        return None


def validate_dataset(dataset_yaml: str, verbose: bool = True) -> bool:
    """
    Validate dataset and return boolean result.

    Convenience function for backward compatibility.

    Args:
        dataset_yaml: Path to dataset YAML file
        verbose: Whether to print validation messages

    Returns:
        True if dataset is valid, False otherwise
    """
    validator = DatasetValidator(verbose=verbose)
    result = validator.validate(dataset_yaml)
    return result.is_valid
