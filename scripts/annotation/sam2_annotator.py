#!/usr/bin/env python3
"""
SAM2 Annotator

Auto-annotation using Segment Anything 2 (SAM2) for flexible object segmentation.
Use as fallback when background subtraction fails or for complex backgrounds.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from annotation_utils import AnnotationResult, bbox_to_yolo, write_yolo_label


class SAM2Annotator:
    """
    Auto-annotation using Segment Anything 2.

    Works with any background and provides high-quality segmentation masks.
    Requires GPU for reasonable performance.
    """

    def __init__(
        self,
        model_path: str = "sam2_b.pt",
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        box_margin: float = 0.02,
    ):
        """
        Initialize SAM2 annotator.

        Args:
            model_path: Path to SAM2 model checkpoint
            device: Device to run model on ("cuda" or "cpu")
            points_per_side: Points per side for automatic mask generation
            pred_iou_thresh: Predicted IoU threshold for filtering masks
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area in pixels
            box_margin: Margin to add to bounding boxes (ratio)
        """
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.min_mask_region_area = min_mask_region_area
        self.box_margin = box_margin

        # Lazy load SAM2 to avoid import errors when not using this method
        self.model = None
        self.mask_generator = None
        self.model_path = model_path

    def _load_model(self) -> None:
        """Lazy load SAM2 model."""
        if self.model is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        print(f"Loading SAM2 model: {self.model_path}")

        # Determine model config based on model path
        if "sam2_b" in self.model_path.lower() or "base" in self.model_path.lower():
            model_cfg = "sam2_hiera_b+.yaml"
        elif "sam2_l" in self.model_path.lower() or "large" in self.model_path.lower():
            model_cfg = "sam2_hiera_l.yaml"
        elif "sam2_t" in self.model_path.lower() or "tiny" in self.model_path.lower():
            model_cfg = "sam2_hiera_t.yaml"
        else:
            model_cfg = "sam2_hiera_b+.yaml"  # Default to base

        self.model = build_sam2(model_cfg, self.model_path, device=self.device)

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

        print("SAM2 model loaded successfully")

    def _select_best_mask(
        self,
        masks: List[Dict],
        image_shape: Tuple[int, int],
        min_area_ratio: float = 0.01,
        max_area_ratio: float = 0.9,
    ) -> Optional[Dict]:
        """
        Select the best mask from SAM2 output.

        Selection criteria:
        1. Filter by area ratio (not too small, not too large)
        2. Prefer masks with higher predicted IoU
        3. Prefer masks with higher stability score

        Args:
            masks: List of mask dictionaries from SAM2
            image_shape: (height, width) of image
            min_area_ratio: Minimum mask area as ratio of image
            max_area_ratio: Maximum mask area as ratio of image

        Returns:
            Best mask dictionary or None
        """
        if not masks:
            return None

        img_area = image_shape[0] * image_shape[1]
        min_area = img_area * min_area_ratio
        max_area = img_area * max_area_ratio

        # Filter by area
        valid_masks = [
            m for m in masks
            if min_area <= m["area"] <= max_area
        ]

        if not valid_masks:
            return None

        # Sort by combined score (IoU + stability)
        def mask_score(m):
            return m["predicted_iou"] + m["stability_score"]

        valid_masks.sort(key=mask_score, reverse=True)

        return valid_masks[0]

    def _mask_to_bbox(
        self,
        mask: np.ndarray,
        image_shape: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """
        Convert binary mask to bounding box.

        Args:
            mask: Binary mask array
            image_shape: (height, width) of image

        Returns:
            Tuple (x_min, y_min, x_max, y_max)
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            # Fallback: use mask bounds directly
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                return (0, 0, 0, 0)

            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
        else:
            # Get bounding rect of largest contour
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            x_min, y_min = x, y
            x_max, y_max = x + w, y + h

        # Add margin
        h, w = image_shape
        margin_x = int((x_max - x_min) * self.box_margin)
        margin_y = int((y_max - y_min) * self.box_margin)

        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)

        return (x_min, y_min, x_max, y_max)

    def annotate_image(
        self,
        image_path: str,
        return_mask: bool = False,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Generate YOLO format annotation for a single image.

        Args:
            image_path: Path to input image
            return_mask: If True, also return the segmentation mask

        Returns:
            Tuple (x_center, y_center, width, height) normalized to [0, 1],
            or None if no object detected.
            If return_mask=True, returns (bbox, mask) tuple.
        """
        self._load_model()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None

        # Convert BGR to RGB for SAM2
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]

        # Generate masks
        masks = self.mask_generator.generate(image_rgb)

        if not masks:
            return None

        # Select best mask
        best_mask = self._select_best_mask(masks, (img_h, img_w))

        if best_mask is None:
            return None

        # Convert mask to bbox
        mask_array = best_mask["segmentation"]
        x_min, y_min, x_max, y_max = self._mask_to_bbox(mask_array, (img_h, img_w))

        # Skip if bbox is invalid
        if x_max <= x_min or y_max <= y_min:
            return None

        # Convert to YOLO format
        yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

        if return_mask:
            return (yolo_bbox, mask_array)

        return yolo_bbox

    def annotate_batch(
        self,
        image_dir: str,
        class_id: int,
        output_dir: str,
        save_masks: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> AnnotationResult:
        """
        Annotate all images in a directory.

        Args:
            image_dir: Directory containing images
            class_id: YOLO class ID for all images
            output_dir: Directory for output label files
            save_masks: If True, also save segmentation masks
            progress_callback: Optional callback(current, total)

        Returns:
            AnnotationResult with statistics
        """
        self._load_model()

        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if save_masks:
            masks_path = output_path / "masks"
            masks_path.mkdir(exist_ok=True)

        # Find images
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        images = [
            f for f in image_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        result = AnnotationResult(total_images=len(images))

        for i, img_file in enumerate(tqdm(images, desc="SAM2 Annotating")):
            annotation = self.annotate_image(str(img_file), return_mask=save_masks)

            if annotation is not None:
                if save_masks:
                    bbox, mask = annotation
                    # Save mask
                    mask_file = masks_path / f"{img_file.stem}_mask.png"
                    cv2.imwrite(str(mask_file), mask.astype(np.uint8) * 255)
                else:
                    bbox = annotation

                # Write label
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
        Visualize SAM2 annotation with mask overlay.

        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            show: If True, display in window

        Returns:
            Annotated image
        """
        self._load_model()

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        annotation = self.annotate_image(image_path, return_mask=True)

        if annotation is not None:
            bbox, mask = annotation
            img_h, img_w = image.shape[:2]

            # Draw mask overlay
            mask_overlay = np.zeros_like(image)
            mask_overlay[mask] = [0, 255, 0]  # Green overlay
            image = cv2.addWeighted(image, 1, mask_overlay, 0.3, 0)

            # Draw bounding box
            from annotation_utils import yolo_to_bbox
            x_min, y_min, x_max, y_max = yolo_to_bbox(*bbox, img_w, img_h)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Add label
            label = f"YOLO: ({bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f})"
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
            cv2.imshow("SAM2 Annotation", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return image


def main():
    """Command-line interface for SAM2 annotation."""
    parser = argparse.ArgumentParser(
        description="Auto-annotate images using SAM2"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="sam2_b.pt",
        help="Path to SAM2 model checkpoint",
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing images",
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
        help="YOLO class ID",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save segmentation masks",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize first result",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.88,
        help="Predicted IoU threshold",
    )
    parser.add_argument(
        "--stability-thresh",
        type=float,
        default=0.92,
        help="Stability score threshold",
    )

    args = parser.parse_args()

    annotator = SAM2Annotator(
        model_path=args.model,
        device=args.device,
        pred_iou_thresh=args.iou_thresh,
        stability_score_thresh=args.stability_thresh,
    )

    result = annotator.annotate_batch(
        image_dir=args.input_dir,
        class_id=args.class_id,
        output_dir=args.output_dir,
        save_masks=args.save_masks,
    )

    print("\n" + result.summary())

    if result.failed > 0:
        print("\nFailed images (first 10):")
        for path in result.failed_paths[:10]:
            print(f"  - {path}")

    if args.visualize and result.successful > 0:
        input_path = Path(args.input_dir)
        for img in input_path.glob("*"):
            if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                label_path = Path(args.output_dir) / f"{img.stem}.txt"
                if label_path.exists():
                    print(f"\nVisualizing: {img}")
                    annotator.visualize_annotation(str(img))
                    break


if __name__ == "__main__":
    main()
