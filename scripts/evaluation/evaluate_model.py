#!/usr/bin/env python3
"""
Model Evaluation Script

Comprehensive evaluation of trained YOLO models including:
- mAP calculation at various IoU thresholds
- Per-class performance metrics
- Inference time measurement
- Competition requirements verification
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate

colorama_init()

# Add scripts directory to path for common module imports
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from common.constants import TARGET_MAP50, TARGET_INFERENCE_MS


@dataclass
class ClassMetrics:
    """Per-class evaluation metrics."""

    class_name: str
    precision: float
    recall: float
    f1_score: float
    ap50: float
    ap50_95: float
    num_samples: int

    def to_dict(self) -> Dict:
        return {
            "class_name": self.class_name,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "ap50": self.ap50,
            "ap50_95": self.ap50_95,
            "num_samples": self.num_samples,
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report."""

    model_path: str
    dataset_path: str
    overall_map50: float
    overall_map50_95: float
    overall_precision: float
    overall_recall: float
    per_class_metrics: Dict[str, ClassMetrics]
    inference_time_ms: float
    inference_time_std: float
    num_test_images: int
    timestamp: str = field(default_factory=lambda: __import__("datetime").datetime.now().isoformat())

    # Competition targets (imported from common.constants)
    _TARGET_MAP50 = TARGET_MAP50
    _TARGET_INFERENCE_MS = TARGET_INFERENCE_MS

    def meets_requirements(self) -> Tuple[bool, List[str]]:
        """Check if model meets competition requirements."""
        issues = []

        if self.overall_map50 < self._TARGET_MAP50:
            issues.append(
                f"mAP50 {self.overall_map50:.2%} < {self._TARGET_MAP50:.0%} target"
            )

        if self.inference_time_ms > self._TARGET_INFERENCE_MS:
            issues.append(
                f"Inference {self.inference_time_ms:.1f}ms > {self._TARGET_INFERENCE_MS:.0f}ms target"
            )

        return (len(issues) == 0, issues)

    def summary(self) -> str:
        """Generate formatted summary."""
        meets, issues = self.meets_requirements()
        status = f"{Fore.GREEN}PASS{Style.RESET_ALL}" if meets else f"{Fore.RED}FAIL{Style.RESET_ALL}"

        lines = [
            f"\n{'='*60}",
            f"  Model Evaluation Report",
            f"{'='*60}",
            f"  Model: {self.model_path}",
            f"  Dataset: {self.dataset_path}",
            f"  Test Images: {self.num_test_images}",
            f"  Status: {status}",
            f"{'='*60}",
            "",
            "  Overall Metrics:",
            f"    mAP@50: {self.overall_map50:.4f} (target: {self._TARGET_MAP50:.0%})",
            f"    mAP@50-95: {self.overall_map50_95:.4f}",
            f"    Precision: {self.overall_precision:.4f}",
            f"    Recall: {self.overall_recall:.4f}",
            "",
            "  Inference Time:",
            f"    Mean: {self.inference_time_ms:.1f} ms (target: <{self._TARGET_INFERENCE_MS:.0f}ms)",
            f"    Std: {self.inference_time_std:.1f} ms",
            "",
        ]

        if issues:
            lines.append("  Issues:")
            for issue in issues:
                lines.append(f"    {Fore.RED}- {issue}{Style.RESET_ALL}")
            lines.append("")

        # Per-class table
        if self.per_class_metrics:
            lines.append("  Per-Class Results:")
            table_data = []
            for name, m in self.per_class_metrics.items():
                status_icon = (
                    f"{Fore.GREEN}OK{Style.RESET_ALL}"
                    if m.ap50 >= 0.8
                    else f"{Fore.YELLOW}WARN{Style.RESET_ALL}"
                    if m.ap50 >= 0.6
                    else f"{Fore.RED}LOW{Style.RESET_ALL}"
                )
                table_data.append([
                    name,
                    m.num_samples,
                    f"{m.precision:.3f}",
                    f"{m.recall:.3f}",
                    f"{m.ap50:.3f}",
                    f"{m.ap50_95:.3f}",
                    status_icon,
                ])

            lines.append(
                tabulate(
                    table_data,
                    headers=["Class", "Samples", "Prec", "Recall", "AP50", "AP50-95", "Status"],
                    tablefmt="simple",
                )
            )

        lines.append(f"\n{'='*60}\n")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_path": self.model_path,
            "dataset_path": self.dataset_path,
            "overall_map50": self.overall_map50,
            "overall_map50_95": self.overall_map50_95,
            "overall_precision": self.overall_precision,
            "overall_recall": self.overall_recall,
            "per_class_metrics": {
                name: m.to_dict() for name, m in self.per_class_metrics.items()
            },
            "inference_time_ms": self.inference_time_ms,
            "inference_time_std": self.inference_time_std,
            "num_test_images": self.num_test_images,
            "timestamp": self.timestamp,
            "meets_requirements": self.meets_requirements()[0],
        }


class ModelEvaluator:
    """
    Evaluate trained YOLO model performance.

    Provides comprehensive metrics for competition readiness verification.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained YOLO model
            device: Device to run inference on
        """
        from ultralytics import YOLO

        self.model_path = model_path
        self.device = device

        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names

        print(f"Model loaded. Classes: {len(self.class_names)}")

    def evaluate(
        self,
        dataset_yaml: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> EvaluationReport:
        """
        Run comprehensive evaluation.

        Args:
            dataset_yaml: Path to dataset configuration
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for mAP calculation

        Returns:
            EvaluationReport with all metrics
        """
        print(f"\nRunning evaluation on: {dataset_yaml}")

        # Run YOLO validation
        results = self.model.val(
            data=dataset_yaml,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        # Extract overall metrics
        overall_map50 = results.box.map50
        overall_map50_95 = results.box.map
        overall_precision = results.box.mp
        overall_recall = results.box.mr

        # Extract per-class metrics
        per_class_metrics = {}
        ap50_per_class = results.box.ap50
        ap_per_class = results.box.ap

        for i, name in self.class_names.items():
            # Get class-specific metrics
            class_ap50 = ap50_per_class[i] if i < len(ap50_per_class) else 0
            class_ap = ap_per_class[i] if i < len(ap_per_class) else 0

            per_class_metrics[name] = ClassMetrics(
                class_name=name,
                precision=overall_precision,  # Per-class precision not directly available
                recall=overall_recall,  # Per-class recall not directly available
                f1_score=2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-6),
                ap50=class_ap50,
                ap50_95=class_ap,
                num_samples=0,  # Would need to count from dataset
            )

        # Measure inference time
        inference_ms, inference_std = self.measure_inference_time()

        # Count test images
        import yaml
        with open(dataset_yaml, "r") as f:
            ds_config = yaml.safe_load(f)
        val_path = Path(dataset_yaml).parent / ds_config.get("val", "")
        num_images = len(list(val_path.glob("*"))) if val_path.exists() else 0

        return EvaluationReport(
            model_path=self.model_path,
            dataset_path=dataset_yaml,
            overall_map50=overall_map50,
            overall_map50_95=overall_map50_95,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            per_class_metrics=per_class_metrics,
            inference_time_ms=inference_ms,
            inference_time_std=inference_std,
            num_test_images=num_images,
        )

    def measure_inference_time(
        self,
        num_iterations: int = 100,
        warmup: int = 10,
        image_size: Tuple[int, int] = (640, 480),
    ) -> Tuple[float, float]:
        """
        Measure inference time on synthetic images.

        Args:
            num_iterations: Number of inference iterations
            warmup: Warmup iterations (not counted)
            image_size: Image size (width, height)

        Returns:
            Tuple of (mean_ms, std_ms)
        """
        print(f"Measuring inference time ({num_iterations} iterations)...")

        # Create dummy image
        dummy_image = np.random.randint(0, 255, (image_size[1], image_size[0], 3), dtype=np.uint8)

        # Warmup
        for _ in range(warmup):
            self.model(dummy_image, verbose=False)

        # Measure
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.model(dummy_image, verbose=False)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        mean_ms = np.mean(times)
        std_ms = np.std(times)

        print(f"Inference time: {mean_ms:.1f} +/- {std_ms:.1f} ms")

        return mean_ms, std_ms

    def predict_single(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        save_path: Optional[str] = None,
    ) -> List[Dict]:
        """
        Run prediction on single image.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            save_path: Optional path to save annotated image

        Returns:
            List of detection dictionaries
        """
        results = self.model(image_path, conf=conf_threshold, verbose=False)
        result = results[0]

        detections = []
        for box in result.boxes:
            det = {
                "class_id": int(box.cls.item()),
                "class_name": self.class_names[int(box.cls.item())],
                "confidence": float(box.conf.item()),
                "bbox": box.xyxy[0].tolist(),
            }
            detections.append(det)

        if save_path:
            annotated = result.plot()
            cv2.imwrite(save_path, annotated)

        return detections


def main():
    """Command-line interface for model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation
  python evaluate_model.py --model models/finetuned/best.pt \\
      --dataset datasets/competition_day/data.yaml

  # Quick inference time test
  python evaluate_model.py --model models/finetuned/best.pt --time-only

  # Single image prediction
  python evaluate_model.py --model models/finetuned/best.pt \\
      --image test.jpg --output result.jpg
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        help="Path to dataset YAML for full evaluation",
    )
    parser.add_argument(
        "--image",
        "-i",
        help="Single image path for prediction",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path for annotated image",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--time-only",
        action="store_true",
        help="Only measure inference time",
    )
    parser.add_argument(
        "--save-report",
        help="Save evaluation report to JSON file",
    )

    args = parser.parse_args()

    evaluator = ModelEvaluator(args.model)

    if args.time_only:
        # Just measure inference time
        mean_ms, std_ms = evaluator.measure_inference_time()
        target = TARGET_INFERENCE_MS
        status = (
            f"{Fore.GREEN}PASS{Style.RESET_ALL}"
            if mean_ms <= target
            else f"{Fore.RED}FAIL{Style.RESET_ALL}"
        )
        print(f"\nInference Time: {mean_ms:.1f} ms (target: <{target:.0f}ms) [{status}]")

    elif args.image:
        # Single image prediction
        detections = evaluator.predict_single(
            args.image,
            conf_threshold=args.conf,
            save_path=args.output,
        )
        print(f"\nDetections in {args.image}:")
        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.2%}")
        if args.output:
            print(f"\nSaved to: {args.output}")

    elif args.dataset:
        # Full evaluation
        report = evaluator.evaluate(args.dataset, conf_threshold=args.conf)
        print(report.summary())

        if args.save_report:
            with open(args.save_report, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Report saved to: {args.save_report}")

        # Exit with error code if requirements not met
        meets, _ = report.meets_requirements()
        if not meets:
            exit(1)

    else:
        parser.print_help()
        print(f"\n{Fore.YELLOW}Specify --dataset for evaluation or --image for prediction{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
