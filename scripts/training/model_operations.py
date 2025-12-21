"""
Model validation and export operations.

Provides utilities for validating trained models and exporting them
for deployment in various formats.
"""
from typing import Dict

from colorama import Fore, Style


class ModelValidator:
    """Validates trained models."""

    def __init__(self, verbose: bool = True):
        """
        Initialize validator.

        Args:
            verbose: Whether to print validation messages
        """
        self.verbose = verbose

    def validate(self, model_path: str, dataset_yaml: str) -> Dict:
        """
        Run validation on a trained model.

        Args:
            model_path: Path to model weights
            dataset_yaml: Path to dataset configuration

        Returns:
            Dictionary of validation metrics (mAP50, mAP50-95, precision, recall)
        """
        from ultralytics import YOLO

        if self.verbose:
            print(f"\nValidating model: {model_path}")

        model = YOLO(model_path)
        results = model.val(data=dataset_yaml)

        return {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }


class ModelExporter:
    """Exports models for deployment."""

    def __init__(self, verbose: bool = True):
        """
        Initialize exporter.

        Args:
            verbose: Whether to print export messages
        """
        self.verbose = verbose

    def export(
        self,
        model_path: str,
        format: str = "onnx",
        simplify: bool = True,
    ) -> str:
        """
        Export model for deployment.

        Args:
            model_path: Path to model weights
            format: Export format (onnx, torchscript, tflite, etc.)
            simplify: Simplify ONNX model (only applies to ONNX format)

        Returns:
            Path to exported model
        """
        from ultralytics import YOLO

        if self.verbose:
            print(f"\nExporting model: {model_path}")
            print(f"Format: {format}")

        model = YOLO(model_path)
        export_path = model.export(format=format, simplify=simplify)

        if self.verbose:
            print(f"Exported to: {export_path}")

        return export_path


def validate_model(model_path: str, dataset_yaml: str, verbose: bool = True) -> Dict:
    """
    Validate a trained model.

    Convenience function for backward compatibility.

    Args:
        model_path: Path to model weights
        dataset_yaml: Path to dataset configuration
        verbose: Whether to print validation messages

    Returns:
        Dictionary of validation metrics
    """
    return ModelValidator(verbose=verbose).validate(model_path, dataset_yaml)


def export_model(
    model_path: str,
    format: str = "onnx",
    simplify: bool = True,
    verbose: bool = True,
) -> str:
    """
    Export a model for deployment.

    Convenience function for backward compatibility.

    Args:
        model_path: Path to model weights
        format: Export format
        simplify: Simplify ONNX model
        verbose: Whether to print export messages

    Returns:
        Path to exported model
    """
    return ModelExporter(verbose=verbose).export(model_path, format, simplify)
