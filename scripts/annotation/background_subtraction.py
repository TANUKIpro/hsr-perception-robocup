"""
Background Subtraction Annotator

Auto-annotation using background subtraction for objects on a uniform background
(e.g., white sheet). This is the primary annotation method for competition day
due to its speed and simplicity.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from annotation_utils import AnnotationResult, bbox_to_yolo, write_yolo_label


@dataclass
class AnnotatorConfig:
    """Configuration for background subtraction annotator."""

    min_contour_area: int = 500
    blur_kernel_size: int = 5
    threshold_method: str = "otsu"  # "otsu" or "adaptive" or "fixed"
    fixed_threshold: int = 30
    morph_kernel_size: int = 5
    erosion_iterations: int = 2
    dilation_iterations: int = 3
    margin_ratio: float = 0.02
    max_contour_area_ratio: float = 0.9  # Max object area as ratio of image


class BackgroundSubtractionAnnotator:
    """
    Auto-annotation using background subtraction.

    Works best with:
    - Uniform background (white sheet recommended)
    - Good lighting conditions
    - Single object per image
    """

    def __init__(
        self,
        background_path: str,
        config: Optional[AnnotatorConfig] = None,
    ):
        """
        Initialize annotator with background reference image.

        Args:
            background_path: Path to background reference image
            config: Annotator configuration (uses defaults if None)
        """
        self.config = config or AnnotatorConfig()
        self.background = self._load_background(background_path)

    def _load_background(self, path: str) -> np.ndarray:
        """Load and preprocess background image."""
        bg = cv2.imread(path)
        if bg is None:
            raise ValueError(f"Failed to load background image: {path}")

        # Convert to grayscale
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

        # Apply blur to reduce noise
        bg_blur = cv2.GaussianBlur(
            bg_gray,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            0,
        )

        return bg_blur

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and apply blur."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        blurred = cv2.GaussianBlur(
            gray,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            0,
        )

        return blurred

    def _compute_difference(
        self, image: np.ndarray, background: np.ndarray
    ) -> np.ndarray:
        """Compute absolute difference between image and background."""
        # Resize background if needed
        if image.shape != background.shape:
            background = cv2.resize(background, (image.shape[1], image.shape[0]))

        diff = cv2.absdiff(image, background)
        return diff

    def _create_mask(self, diff: np.ndarray) -> np.ndarray:
        """Create binary mask from difference image."""
        if self.config.threshold_method == "otsu":
            _, mask = cv2.threshold(
                diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        elif self.config.threshold_method == "adaptive":
            mask = cv2.adaptiveThreshold(
                diff,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2,
            )
        else:  # fixed
            _, mask = cv2.threshold(
                diff, self.config.fixed_threshold, 255, cv2.THRESH_BINARY
            )

        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size),
        )

        # Erode to remove noise
        mask = cv2.erode(mask, kernel, iterations=self.config.erosion_iterations)

        # Dilate to restore object size
        mask = cv2.dilate(mask, kernel, iterations=self.config.dilation_iterations)

        return mask

    def _find_object_bbox(
        self, mask: np.ndarray, img_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find bounding box of the largest object in mask.

        Args:
            mask: Binary mask image
            img_shape: (height, width) of original image

        Returns:
            Tuple (x_min, y_min, x_max, y_max) or None if no object found
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Filter contours by area
        img_area = img_shape[0] * img_shape[1]
        max_area = img_area * self.config.max_contour_area_ratio
        min_area = self.config.min_contour_area

        valid_contours = [
            c for c in contours if min_area <= cv2.contourArea(c) <= max_area
        ]

        if not valid_contours:
            return None

        # Get largest contour
        largest = max(valid_contours, key=cv2.contourArea)

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest)

        # Add margin
        margin_x = int(w * self.config.margin_ratio)
        margin_y = int(h * self.config.margin_ratio)

        x_min = max(0, x - margin_x)
        y_min = max(0, y - margin_y)
        x_max = min(img_shape[1], x + w + margin_x)
        y_max = min(img_shape[0], y + h + margin_y)

        return (x_min, y_min, x_max, y_max)

    def annotate_image(
        self,
        image_path: str,
        return_debug: bool = False,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Generate YOLO format annotation for a single image.

        Args:
            image_path: Path to input image
            return_debug: If True, returns debug info instead of bbox

        Returns:
            Tuple (x_center, y_center, width, height) normalized to [0, 1],
            or None if no object detected
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None

        img_h, img_w = image.shape[:2]

        # Preprocess
        preprocessed = self._preprocess_image(image)

        # Compute difference
        diff = self._compute_difference(preprocessed, self.background)

        # Create mask
        mask = self._create_mask(diff)

        # Find object bbox
        bbox = self._find_object_bbox(mask, (img_h, img_w))

        if bbox is None:
            return None

        # Convert to YOLO format
        x_min, y_min, x_max, y_max = bbox
        yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

        return yolo_bbox

    def annotate_batch(
        self,
        image_dir: str,
        class_id: int,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AnnotationResult:
        """
        Annotate all images in a directory.

        Args:
            image_dir: Directory containing images to annotate
            class_id: YOLO class ID for all images
            output_dir: Directory for output label files
            progress_callback: Optional callback(current, total) for progress

        Returns:
            AnnotationResult with statistics
        """
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        images = [
            f
            for f in image_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        result = AnnotationResult(total_images=len(images))

        for i, img_file in enumerate(tqdm(images, desc="Annotating")):
            bbox = self.annotate_image(str(img_file))

            if bbox is not None:
                # Write label file
                label_path = output_path / f"{img_file.stem}.txt"
                write_yolo_label(str(label_path), class_id, bbox)
                result.successful += 1
            else:
                result.failed += 1
                result.failed_paths.append(str(img_file))

            if progress_callback:
                progress_callback(i + 1, len(images))

        return result

    def visualize_annotation(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        show: bool = True,
    ) -> np.ndarray:
        """
        Visualize annotation result on image.

        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            show: If True, display image in window

        Returns:
            Annotated image with bounding box drawn
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        img_h, img_w = image.shape[:2]

        # Get annotation
        bbox = self.annotate_image(image_path)

        if bbox is not None:
            # Convert back to pixel coordinates
            from annotation_utils import yolo_to_bbox

            x_min, y_min, x_max, y_max = yolo_to_bbox(*bbox, img_w, img_h)

            # Draw bounding box
            cv2.rectangle(
                image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2
            )

            # Add label
            label = f"({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})"
            cv2.putText(
                image,
                label,
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
        else:
            # No detection
            cv2.putText(
                image,
                "No object detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if output_path:
            cv2.imwrite(output_path, image)

        if show:
            cv2.imshow("Annotation", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image


def create_background_from_images(
    image_paths: list,
    output_path: str,
    method: str = "median",
) -> None:
    """
    Create background reference image from multiple images.

    Useful when you don't have a clean background shot - take multiple
    images of the empty background and combine them.

    Args:
        image_paths: List of paths to background images
        output_path: Path for output background image
        method: Combination method ("median" or "mean")
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)

    if not images:
        raise ValueError("No valid images found")

    # Stack images
    stack = np.stack(images, axis=0)

    if method == "median":
        background = np.median(stack, axis=0).astype(np.uint8)
    else:  # mean
        background = np.mean(stack, axis=0).astype(np.uint8)

    cv2.imwrite(output_path, background)
    print(f"Background saved to: {output_path}")


def main():
    """Command-line interface for background subtraction annotation."""
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using background subtraction"
    )
    parser.add_argument(
        "--background",
        "-b",
        required=True,
        help="Path to background reference image",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing images to annotate",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory for output label files",
    )
    parser.add_argument(
        "--class-id",
        "-c",
        type=int,
        required=True,
        help="YOLO class ID for annotations",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=500,
        help="Minimum contour area in pixels (default: 500)",
    )
    parser.add_argument(
        "--threshold",
        choices=["otsu", "adaptive", "fixed"],
        default="otsu",
        help="Thresholding method (default: otsu)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize first annotation result",
    )

    args = parser.parse_args()

    config = AnnotatorConfig(
        min_contour_area=args.min_area,
        threshold_method=args.threshold,
    )

    annotator = BackgroundSubtractionAnnotator(args.background, config)

    result = annotator.annotate_batch(
        image_dir=args.input_dir,
        class_id=args.class_id,
        output_dir=args.output_dir,
    )

    print("\n" + result.summary())

    if result.failed > 0:
        print("\nFailed images:")
        for path in result.failed_paths[:10]:  # Show first 10
            print(f"  - {path}")
        if len(result.failed_paths) > 10:
            print(f"  ... and {len(result.failed_paths) - 10} more")

    if args.visualize and result.successful > 0:
        # Visualize first successful annotation
        input_path = Path(args.input_dir)
        images = list(input_path.glob("*"))
        for img in images:
            if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                label_path = Path(args.output_dir) / f"{img.stem}.txt"
                if label_path.exists():
                    print(f"\nVisualizing: {img}")
                    annotator.visualize_annotation(str(img))
                    break


if __name__ == "__main__":
    main()
