#!/usr/bin/env python3
"""
Auto-Annotation Pipeline

Main orchestration script for competition day auto-annotation workflow.
Supports both background subtraction (primary) and SAM2 (fallback) methods.
"""

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from colorama import Fore, Style, init as colorama_init
from tabulate import tabulate

from annotation_utils import (
    AnnotationResult,
    create_dataset_yaml,
    get_class_names,
    load_class_config,
    split_dataset,
    update_collected_samples,
    validate_yolo_annotation,
)
from background_subtraction import (
    AnnotatorConfig,
    BackgroundSubtractionAnnotator,
)

# Initialize colorama for cross-platform colored output
colorama_init()


@dataclass
class AnnotationReport:
    """Complete report for annotation pipeline execution."""

    timestamp: str
    method: str
    total_classes: int
    total_images: int
    successful: int
    failed: int
    class_results: Dict[str, AnnotationResult]
    dataset_path: str
    train_count: int
    val_count: int

    @property
    def success_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return (self.successful / self.total_images) * 100

    def summary(self) -> str:
        """Generate formatted summary report."""
        lines = [
            f"\n{'='*60}",
            f"  Auto-Annotation Report",
            f"{'='*60}",
            f"  Timestamp: {self.timestamp}",
            f"  Method: {self.method}",
            f"  Dataset: {self.dataset_path}",
            f"{'='*60}",
            "",
            "  Overall Statistics:",
            f"    Total Classes: {self.total_classes}",
            f"    Total Images: {self.total_images}",
            f"    Successful: {Fore.GREEN}{self.successful}{Style.RESET_ALL}",
            f"    Failed: {Fore.RED}{self.failed}{Style.RESET_ALL}",
            f"    Success Rate: {self.success_rate:.1f}%",
            "",
            f"  Dataset Split:",
            f"    Train: {self.train_count}",
            f"    Val: {self.val_count}",
            "",
        ]

        # Per-class results table
        table_data = []
        for class_name, result in self.class_results.items():
            status = (
                f"{Fore.GREEN}OK{Style.RESET_ALL}"
                if result.success_rate >= 90
                else f"{Fore.YELLOW}WARN{Style.RESET_ALL}"
                if result.success_rate >= 70
                else f"{Fore.RED}FAIL{Style.RESET_ALL}"
            )
            table_data.append(
                [
                    class_name,
                    result.total_images,
                    result.successful,
                    result.failed,
                    f"{result.success_rate:.1f}%",
                    status,
                ]
            )

        if table_data:
            lines.append("  Per-Class Results:")
            lines.append(
                tabulate(
                    table_data,
                    headers=["Class", "Total", "Success", "Failed", "Rate", "Status"],
                    tablefmt="simple",
                )
            )

        lines.append(f"\n{'='*60}\n")

        return "\n".join(lines)


class AutoAnnotator:
    """
    Main auto-annotation orchestrator.

    Handles the complete annotation pipeline including:
    - Loading class configuration
    - Running annotation method (background subtraction or SAM2)
    - Splitting dataset into train/val
    - Generating YOLO dataset configuration
    - Reporting results
    """

    def __init__(
        self,
        method: str = "background",
        background_path: Optional[str] = None,
        sam2_model_path: Optional[str] = None,
        annotator_config: Optional[AnnotatorConfig] = None,
    ):
        """
        Initialize auto-annotator.

        Args:
            method: Annotation method ("background" or "sam2")
            background_path: Path to background image (required for background method)
            sam2_model_path: Path to SAM2 model (required for sam2 method)
            annotator_config: Configuration for background subtraction
        """
        self.method = method
        self.background_path = background_path
        self.sam2_model_path = sam2_model_path
        self.annotator_config = annotator_config or AnnotatorConfig()

        self._annotator = None

    def _get_annotator(self):
        """Lazy initialization of annotator."""
        if self._annotator is not None:
            return self._annotator

        if self.method == "background":
            if not self.background_path:
                raise ValueError("Background path required for background method")
            self._annotator = BackgroundSubtractionAnnotator(
                self.background_path, self.annotator_config
            )
        elif self.method == "sam2":
            # Import SAM2 annotator only when needed
            try:
                from sam2_annotator import SAM2Annotator

                self._annotator = SAM2Annotator(
                    model_path=self.sam2_model_path or "models/pretrained/sam2_b.pt"
                )
            except ImportError:
                raise ImportError(
                    "SAM2 annotator not available. "
                    "Install segment-anything-2 package."
                )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self._annotator

    def run(
        self,
        input_dir: str,
        output_dir: str,
        class_config_path: str,
        train_val_split: float = 0.85,
        verify: bool = True,
        update_config: bool = True,
    ) -> AnnotationReport:
        """
        Run complete annotation pipeline.

        Directory structure expected:
        input_dir/
            class_name_1/
                image_001.jpg
                image_002.jpg
                ...
            class_name_2/
                ...

        Args:
            input_dir: Directory containing per-class image subdirectories
            output_dir: Output directory for YOLO dataset
            class_config_path: Path to object_classes.json
            train_val_split: Ratio of training data (default: 0.85)
            verify: If True, validate generated annotations
            update_config: If True, update collected_samples in config

        Returns:
            AnnotationReport with complete statistics
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        # Load class configuration
        config = load_class_config(class_config_path)
        class_names = get_class_names(config)
        class_id_map = {obj["class_name"]: obj["class_id"] for obj in config["objects"]}

        # Create output directories
        images_dir = output_path / "images" / "all"
        labels_dir = output_path / "labels" / "all"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Get annotator
        annotator = self._get_annotator()

        # Process each class
        class_results: Dict[str, AnnotationResult] = {}
        total_images = 0
        total_successful = 0
        total_failed = 0

        for class_name in class_names:
            class_dir = input_path / class_name

            if not class_dir.exists():
                print(f"{Fore.YELLOW}Warning: No directory for class '{class_name}'{Style.RESET_ALL}")
                continue

            class_id = class_id_map[class_name]
            print(f"\nProcessing class: {class_name} (ID: {class_id})")

            # Find images
            images = list(class_dir.glob("*"))
            images = [f for f in images if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

            if not images:
                print(f"  No images found")
                continue

            # Annotate each image
            result = AnnotationResult(total_images=len(images))

            for img_path in images:
                bbox = annotator.annotate_image(str(img_path))

                if bbox is not None:
                    # Copy image to output
                    new_img_name = f"{class_name}_{img_path.name}"
                    new_img_path = images_dir / new_img_name
                    shutil.copy2(img_path, new_img_path)

                    # Write label
                    from annotation_utils import write_yolo_label

                    label_path = labels_dir / f"{new_img_path.stem}.txt"
                    write_yolo_label(str(label_path), class_id, bbox)

                    result.successful += 1
                else:
                    result.failed += 1
                    result.failed_paths.append(str(img_path))

            class_results[class_name] = result
            total_images += result.total_images
            total_successful += result.successful
            total_failed += result.failed

            print(f"  {result.successful}/{result.total_images} annotated ({result.success_rate:.1f}%)")

            # Update config if requested
            if update_config:
                update_collected_samples(class_config_path, class_name, result.successful)

        # Verify annotations if requested
        if verify:
            print("\nVerifying annotations...")
            for label_file in labels_dir.glob("*.txt"):
                is_valid, errors = validate_yolo_annotation(str(label_file))
                if not is_valid:
                    print(f"{Fore.RED}Invalid: {label_file.name}{Style.RESET_ALL}")
                    for err in errors[:3]:
                        print(f"  {err}")

        # Split dataset
        print("\nSplitting dataset...")
        split_result = split_dataset(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            output_dir=str(output_path),
            train_ratio=train_val_split,
        )
        print(f"  Train: {split_result['train']}, Val: {split_result['val']}")

        # Create dataset.yaml
        dataset_yaml_path = output_path / "data.yaml"
        create_dataset_yaml(
            output_path=str(dataset_yaml_path),
            train_path="images/train",
            val_path="images/val",
            class_names=class_names,
        )
        print(f"Dataset config saved: {dataset_yaml_path}")

        # Generate report
        report = AnnotationReport(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            method=self.method,
            total_classes=len(class_results),
            total_images=total_images,
            successful=total_successful,
            failed=total_failed,
            class_results=class_results,
            dataset_path=str(output_path),
            train_count=split_result["train"],
            val_count=split_result["val"],
        )

        return report


def main():
    """Command-line interface for auto-annotation pipeline."""
    parser = argparse.ArgumentParser(
        description="Auto-annotate images for YOLO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using background subtraction (recommended)
  python auto_annotate.py --method background \\
      --background datasets/backgrounds/white_sheet.jpg \\
      --input-dir datasets/raw_captures \\
      --output-dir datasets/competition_day \\
      --class-config config/object_classes.json

  # Using SAM2 (requires GPU)
  python auto_annotate.py --method sam2 \\
      --input-dir datasets/raw_captures \\
      --output-dir datasets/competition_day \\
      --class-config config/object_classes.json
        """,
    )

    parser.add_argument(
        "--method",
        "-m",
        choices=["background", "sam2"],
        default="background",
        help="Annotation method (default: background)",
    )
    parser.add_argument(
        "--background",
        "-b",
        help="Path to background reference image (required for background method)",
    )
    parser.add_argument(
        "--sam2-model",
        help="Path to SAM2 model checkpoint (for sam2 method)",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing per-class image subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--class-config",
        "-c",
        default="config/object_classes.json",
        help="Path to object_classes.json (default: config/object_classes.json)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.85,
        help="Train/val split ratio (default: 0.85)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip annotation verification",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Don't update collected_samples in config",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum contour area for background method (default: 500)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.method == "background" and not args.background:
        parser.error("--background is required for background method")

    # Create annotator config
    annotator_config = AnnotatorConfig(min_contour_area=args.min_area)

    # Run pipeline
    annotator = AutoAnnotator(
        method=args.method,
        background_path=args.background,
        sam2_model_path=args.sam2_model,
        annotator_config=annotator_config,
    )

    try:
        report = annotator.run(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            class_config_path=args.class_config,
            train_val_split=args.split,
            verify=not args.no_verify,
            update_config=not args.no_update_config,
        )

        print(report.summary())

        # Save report to file
        report_path = Path(args.output_dir) / "annotation_report.json"
        with open(report_path, "w") as f:
            report_dict = {
                "timestamp": report.timestamp,
                "method": report.method,
                "total_classes": report.total_classes,
                "total_images": report.total_images,
                "successful": report.successful,
                "failed": report.failed,
                "success_rate": report.success_rate,
                "train_count": report.train_count,
                "val_count": report.val_count,
                "class_results": {
                    name: {
                        "total": r.total_images,
                        "successful": r.successful,
                        "failed": r.failed,
                        "success_rate": r.success_rate,
                    }
                    for name, r in report.class_results.items()
                },
            }
            json.dump(report_dict, f, indent=2)
        print(f"Report saved: {report_path}")

        # Exit with error if success rate is too low
        if report.success_rate < 70:
            print(f"\n{Fore.RED}Warning: Low success rate ({report.success_rate:.1f}%)")
            print(f"Consider using SAM2 method or checking background image.{Style.RESET_ALL}")
            sys.exit(1)

    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
