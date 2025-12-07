#!/usr/bin/env python3
"""
Visual Verification Tool

Interactive visual verification of model predictions for quality assurance
before competition deployment.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from colorama import Fore, Style, init as colorama_init

colorama_init()

# Add scripts directory to path for common module imports
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from common.constants import IMAGE_EXTENSIONS


class VisualVerifier:
    """
    Interactive visual verification of YOLO model predictions.

    Provides tools for:
    - Viewing predictions with confidence scores
    - Side-by-side comparison with ground truth
    - Batch verification with navigation
    - Generating sample images for reports
    """

    def __init__(
        self,
        model_path: str,
        class_config_path: Optional[str] = None,
    ):
        """
        Initialize verifier.

        Args:
            model_path: Path to trained YOLO model
            class_config_path: Optional path to class configuration JSON
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.class_colors = self._generate_colors()

        # Load class config if provided
        self.class_config = None
        if class_config_path and Path(class_config_path).exists():
            with open(class_config_path, "r") as f:
                self.class_config = json.load(f)

    def _generate_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Generate distinct colors for each class."""
        np.random.seed(42)
        colors = {}
        for i in range(len(self.class_names)):
            colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
        return colors

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        color_override: Optional[Tuple[int, int, int]] = None,
        thickness: int = 2,
        show_conf: bool = True,
    ) -> np.ndarray:
        """
        Draw detection boxes on image.

        Args:
            image: Input image
            detections: List of detection dicts with bbox, class_id, confidence
            color_override: Optional color to use for all boxes
            thickness: Line thickness
            show_conf: Show confidence scores

        Returns:
            Annotated image
        """
        annotated = image.copy()

        for det in detections:
            bbox = det["bbox"]
            class_id = det["class_id"]
            conf = det.get("confidence", 1.0)
            class_name = self.class_names.get(class_id, f"class_{class_id}")

            x1, y1, x2, y2 = map(int, bbox)
            color = color_override or self.class_colors.get(class_id, (0, 255, 0))

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{class_name}"
            if show_conf:
                label += f" {conf:.2f}"

            font_scale = 0.5
            font_thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_h - baseline - 5),
                (x1 + label_w, y1),
                color,
                -1,
            )

            # Label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )

        return annotated

    def predict_image(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run prediction and return annotated image with detections.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold

        Returns:
            Tuple of (annotated_image, detections_list)
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

        annotated = result.plot()

        return annotated, detections

    def verify_single(
        self,
        image_path: str,
        conf_threshold: float = 0.25,
        window_name: str = "Verification",
    ) -> None:
        """
        Verify single image interactively.

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold
            window_name: OpenCV window name
        """
        annotated, detections = self.predict_image(image_path, conf_threshold)

        # Add info overlay
        h, w = annotated.shape[:2]
        info = f"Detections: {len(detections)} | Press any key to continue"
        cv2.putText(
            annotated,
            info,
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def verify_batch(
        self,
        image_dir: str,
        conf_threshold: float = 0.25,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Verify batch of images with interactive navigation.

        Controls:
        - n/Right: Next image
        - p/Left: Previous image
        - s: Save current image
        - q/Esc: Quit

        Args:
            image_dir: Directory containing images
            conf_threshold: Confidence threshold
            output_dir: Optional directory to save annotated images

        Returns:
            Statistics dictionary
        """
        image_path = Path(image_dir)
        images = sorted([
            f for f in image_path.glob("*")
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

        if not images:
            print(f"{Fore.RED}No images found in {image_dir}{Style.RESET_ALL}")
            return {}

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        print(f"\nFound {len(images)} images")
        print("Controls: n=next, p=prev, s=save, q=quit\n")

        stats = {
            "total": len(images),
            "viewed": 0,
            "saved": 0,
            "detections_per_image": [],
        }

        current_idx = 0
        window_name = "Batch Verification"

        while True:
            img_path = images[current_idx]
            annotated, detections = self.predict_image(str(img_path), conf_threshold)

            # Add navigation info
            h, w = annotated.shape[:2]
            nav_info = f"[{current_idx + 1}/{len(images)}] {img_path.name} | Detections: {len(detections)}"
            cv2.putText(
                annotated,
                nav_info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                "n:next p:prev s:save q:quit",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q') or key == 27:  # q or Esc
                break
            elif key == ord('n') or key == 83:  # n or Right arrow
                current_idx = min(current_idx + 1, len(images) - 1)
                stats["viewed"] += 1
            elif key == ord('p') or key == 81:  # p or Left arrow
                current_idx = max(current_idx - 1, 0)
            elif key == ord('s') and output_dir:  # s
                save_path = Path(output_dir) / f"verified_{img_path.name}"
                cv2.imwrite(str(save_path), annotated)
                print(f"Saved: {save_path}")
                stats["saved"] += 1

            stats["detections_per_image"].append(len(detections))

        cv2.destroyAllWindows()

        return stats

    def create_comparison_grid(
        self,
        images: List[str],
        conf_threshold: float = 0.25,
        grid_cols: int = 3,
        cell_size: Tuple[int, int] = (320, 240),
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Create comparison grid of predictions.

        Args:
            images: List of image paths
            conf_threshold: Confidence threshold
            grid_cols: Number of columns in grid
            cell_size: Size of each cell (width, height)
            output_path: Optional path to save grid

        Returns:
            Grid image
        """
        cells = []

        for img_path in images:
            annotated, detections = self.predict_image(img_path, conf_threshold)

            # Resize to cell size
            cell = cv2.resize(annotated, cell_size)

            # Add filename
            filename = Path(img_path).name[:20]
            cv2.putText(
                cell,
                filename,
                (5, cell_size[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

            cells.append(cell)

        # Pad to complete grid
        grid_rows = (len(cells) + grid_cols - 1) // grid_cols
        while len(cells) < grid_rows * grid_cols:
            cells.append(np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8))

        # Arrange in grid
        rows = []
        for i in range(0, len(cells), grid_cols):
            row = np.hstack(cells[i:i + grid_cols])
            rows.append(row)

        grid = np.vstack(rows)

        if output_path:
            cv2.imwrite(output_path, grid)
            print(f"Grid saved: {output_path}")

        return grid

    def generate_report_samples(
        self,
        test_dir: str,
        output_dir: str,
        samples_per_class: int = 3,
        conf_threshold: float = 0.25,
    ) -> List[str]:
        """
        Generate sample prediction images for documentation/reports.

        Args:
            test_dir: Directory with test images
            output_dir: Output directory for samples
            samples_per_class: Number of samples per class
            conf_threshold: Confidence threshold

        Returns:
            List of generated image paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        test_path = Path(test_dir)
        images = list(test_path.glob("*"))
        images = [f for f in images if f.suffix.lower() in IMAGE_EXTENSIONS]

        # Collect samples per class
        class_samples: Dict[str, List[str]] = {name: [] for name in self.class_names.values()}

        for img_path in images:
            _, detections = self.predict_image(str(img_path), conf_threshold)

            for det in detections:
                class_name = det["class_name"]
                if len(class_samples[class_name]) < samples_per_class:
                    class_samples[class_name].append(str(img_path))

            # Check if we have enough samples
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break

        # Generate annotated samples
        generated = []
        for class_name, sample_paths in class_samples.items():
            for i, img_path in enumerate(sample_paths[:samples_per_class]):
                annotated, _ = self.predict_image(img_path, conf_threshold)
                output_file = output_path / f"{class_name}_sample_{i+1}.jpg"
                cv2.imwrite(str(output_file), annotated)
                generated.append(str(output_file))
                print(f"Generated: {output_file}")

        return generated


def main():
    """Command-line interface for visual verification."""
    parser = argparse.ArgumentParser(
        description="Visual verification of YOLO model predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify single image
  python visual_verification.py --model best.pt --image test.jpg

  # Batch verification with navigation
  python visual_verification.py --model best.pt --batch-dir test_images/

  # Create comparison grid
  python visual_verification.py --model best.pt --batch-dir test_images/ \\
      --grid --output grid.jpg

  # Generate report samples
  python visual_verification.py --model best.pt --batch-dir test_images/ \\
      --report-samples --output samples/
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained YOLO model",
    )
    parser.add_argument(
        "--image",
        "-i",
        help="Single image path for verification",
    )
    parser.add_argument(
        "--batch-dir",
        "-b",
        help="Directory for batch verification",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output path/directory",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create comparison grid instead of interactive view",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="Number of columns in grid (default: 3)",
    )
    parser.add_argument(
        "--report-samples",
        action="store_true",
        help="Generate sample images for reports",
    )
    parser.add_argument(
        "--class-config",
        help="Path to class configuration JSON",
    )

    args = parser.parse_args()

    verifier = VisualVerifier(args.model, args.class_config)

    if args.image:
        # Single image verification
        verifier.verify_single(args.image, args.conf)

    elif args.batch_dir:
        if args.report_samples:
            # Generate report samples
            if not args.output:
                print(f"{Fore.RED}--output required for report samples{Style.RESET_ALL}")
                return
            verifier.generate_report_samples(
                args.batch_dir,
                args.output,
                conf_threshold=args.conf,
            )

        elif args.grid:
            # Create grid
            images = sorted(Path(args.batch_dir).glob("*"))
            images = [str(f) for f in images if f.suffix.lower() in IMAGE_EXTENSIONS]
            images = images[:9]  # Limit to 9 for grid

            output = args.output or "comparison_grid.jpg"
            verifier.create_comparison_grid(
                images,
                conf_threshold=args.conf,
                grid_cols=args.grid_cols,
                output_path=output,
            )

        else:
            # Interactive batch verification
            stats = verifier.verify_batch(
                args.batch_dir,
                conf_threshold=args.conf,
                output_dir=args.output,
            )
            if stats:
                print(f"\nVerification complete:")
                print(f"  Images viewed: {stats.get('viewed', 0)}")
                print(f"  Images saved: {stats.get('saved', 0)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
