"""
Copy-Paste Augmentor

Generates synthetic training images by pasting extracted objects onto background images.
Features:
- Gaussian edge blending for natural object boundaries
- White balance adjustment (LAB color space) to match background lighting
- Random position, scale, and flip transformations
- Automatic YOLO annotation generation
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add scripts directory to path for imports
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from annotation.annotation_utils import bbox_to_yolo, write_yolo_label
from common.constants import IMAGE_EXTENSIONS

from .object_extractor import ExtractedObject, ObjectExtractor


@dataclass
class CopyPasteConfig:
    """Configuration for Copy-Paste augmentation."""

    # Synthesis ratio (synthetic : real)
    synthetic_to_real_ratio: float = 2.0

    # Scaling settings
    scale_range: Tuple[float, float] = (0.5, 1.5)

    # Rotation and flip
    rotation_range: Tuple[float, float] = (-15.0, 15.0)
    enable_horizontal_flip: bool = True
    enable_vertical_flip: bool = False

    # Blending settings
    edge_blur_sigma: float = 2.0
    enable_white_balance: bool = True
    white_balance_strength: float = 0.7

    # Multiple objects settings
    max_objects_per_image: int = 3
    min_objects_per_image: int = 1
    allow_overlap: bool = False
    overlap_iou_threshold: float = 0.1

    # Output settings
    output_image_format: str = "jpg"
    output_quality: int = 95

    # Random seed
    seed: int = 42


@dataclass
class PasteResult:
    """Result of a single paste operation."""

    class_id: int
    class_name: str
    bbox_pixel: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    scale_applied: float
    flipped: bool


@dataclass
class SyntheticImageResult:
    """Result of generating a synthetic image."""

    image: np.ndarray  # BGR image
    paste_results: List[PasteResult]
    background_path: str
    source_objects: List[str]  # Paths to source NPZ files


class CopyPasteAugmentor:
    """
    Copy-Paste Data Augmentation.

    Generates synthetic training images by pasting extracted objects
    onto background images with blending for natural appearance.
    """

    def __init__(self, config: Optional[CopyPasteConfig] = None):
        """
        Initialize CopyPasteAugmentor.

        Args:
            config: Configuration settings (uses defaults if None)
        """
        self.config = config or CopyPasteConfig()
        self.rng = np.random.RandomState(self.config.seed)

    def blend_object(
        self,
        background: np.ndarray,
        object_rgba: np.ndarray,
        position: Tuple[int, int],
        scale: float = 1.0,
        flip_horizontal: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Blend an object onto a background image.

        Args:
            background: BGR background image
            object_rgba: RGBA object image
            position: (x, y) paste position (top-left corner)
            scale: Scale factor for object
            flip_horizontal: Whether to flip horizontally

        Returns:
            Tuple of (blended_image, actual_bbox)
        """
        result = background.copy()
        x, y = position

        # Apply transformations to object
        obj_rgba = object_rgba.copy()

        # Flip if requested
        if flip_horizontal:
            obj_rgba = cv2.flip(obj_rgba, 1)

        # Scale if needed
        if scale != 1.0:
            new_h = int(obj_rgba.shape[0] * scale)
            new_w = int(obj_rgba.shape[1] * scale)
            if new_h > 0 and new_w > 0:
                obj_rgba = cv2.resize(
                    obj_rgba, (new_w, new_h), interpolation=cv2.INTER_LINEAR
                )

        obj_h, obj_w = obj_rgba.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Calculate clipping boundaries
        src_x1, src_y1 = 0, 0
        src_x2, src_y2 = obj_w, obj_h
        dst_x1, dst_y1 = x, y
        dst_x2, dst_y2 = x + obj_w, y + obj_h

        # Adjust for image boundaries
        if dst_x1 < 0:
            src_x1 = -dst_x1
            dst_x1 = 0
        if dst_y1 < 0:
            src_y1 = -dst_y1
            dst_y1 = 0
        if dst_x2 > bg_w:
            src_x2 -= dst_x2 - bg_w
            dst_x2 = bg_w
        if dst_y2 > bg_h:
            src_y2 -= dst_y2 - bg_h
            dst_y2 = bg_h

        # Check if object is completely outside
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return result, (0, 0, 0, 0)

        # Extract object region
        obj_rgb = obj_rgba[src_y1:src_y2, src_x1:src_x2, :3]
        obj_alpha = obj_rgba[src_y1:src_y2, src_x1:src_x2, 3]

        # Convert RGB to BGR (assuming object is stored as RGB)
        obj_bgr = cv2.cvtColor(obj_rgb, cv2.COLOR_RGB2BGR)

        # Apply white balance adjustment
        if self.config.enable_white_balance:
            target_region = result[dst_y1:dst_y2, dst_x1:dst_x2]
            obj_bgr = self._adjust_white_balance_lab(
                obj_bgr, target_region, obj_alpha, self.config.white_balance_strength
            )

        # Alpha blending
        alpha_normalized = obj_alpha.astype(np.float32) / 255.0
        alpha_3ch = np.stack([alpha_normalized] * 3, axis=-1)

        bg_region = result[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        obj_float = obj_bgr.astype(np.float32)

        blended = alpha_3ch * obj_float + (1 - alpha_3ch) * bg_region
        result[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)

        # Return actual bounding box
        actual_bbox = (dst_x1, dst_y1, dst_x2, dst_y2)

        return result, actual_bbox

    def _adjust_white_balance_lab(
        self,
        obj_bgr: np.ndarray,
        target_region: np.ndarray,
        mask: np.ndarray,
        strength: float = 0.7,
    ) -> np.ndarray:
        """
        Adjust white balance using LAB color space.

        Args:
            obj_bgr: Object image in BGR
            target_region: Background region in BGR
            mask: Alpha mask
            strength: Adjustment strength (0.0-1.0)

        Returns:
            Color-adjusted BGR image
        """
        # Convert to LAB
        obj_lab = cv2.cvtColor(obj_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Create mask for object pixels
        mask_bool = mask > 128
        if not np.any(mask_bool):
            return obj_bgr

        # Calculate mean colors
        obj_mean = [np.mean(obj_lab[:, :, i][mask_bool]) for i in range(3)]
        target_mean = [np.mean(target_lab[:, :, i]) for i in range(3)]

        # Apply adjustment
        for i in range(3):
            diff = (target_mean[i] - obj_mean[i]) * strength
            # Reduce adjustment for L channel (luminance)
            if i == 0:
                diff *= 0.5
            obj_lab[:, :, i] = np.clip(obj_lab[:, :, i] + diff, 0, 255)

        return cv2.cvtColor(obj_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def find_paste_position(
        self,
        bg_shape: Tuple[int, int],
        obj_shape: Tuple[int, int],
        existing_boxes: List[Tuple[int, int, int, int]],
        margin: int = 10,
        max_attempts: int = 50,
    ) -> Optional[Tuple[int, int]]:
        """
        Find a valid paste position avoiding overlaps.

        Args:
            bg_shape: Background (height, width)
            obj_shape: Object (height, width)
            existing_boxes: List of existing bounding boxes
            margin: Margin from image edges
            max_attempts: Maximum attempts to find valid position

        Returns:
            (x, y) position or None if no valid position found
        """
        bg_h, bg_w = bg_shape
        obj_h, obj_w = obj_shape

        # Calculate valid range
        x_min = margin
        x_max = bg_w - obj_w - margin
        y_min = margin
        y_max = bg_h - obj_h - margin

        if x_max <= x_min or y_max <= y_min:
            return None

        for _ in range(max_attempts):
            x = self.rng.randint(x_min, max(x_min + 1, x_max))
            y = self.rng.randint(y_min, max(y_min + 1, y_max))

            new_bbox = (x, y, x + obj_w, y + obj_h)

            # Check overlap with existing boxes
            if self.config.allow_overlap:
                return (x, y)

            overlap_found = False
            for existing in existing_boxes:
                iou = self._compute_iou(new_bbox, existing)
                if iou > self.config.overlap_iou_threshold:
                    overlap_found = True
                    break

            if not overlap_found:
                return (x, y)

        return None

    def _compute_iou(
        self,
        bbox1: Tuple[int, int, int, int],
        bbox2: Tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def generate_synthetic_image(
        self,
        background: np.ndarray,
        objects: List[Tuple[ExtractedObject, str]],  # (object, npz_path)
        background_path: str = "",
    ) -> SyntheticImageResult:
        """
        Generate a single synthetic image.

        Args:
            background: BGR background image
            objects: List of (ExtractedObject, npz_path) tuples
            background_path: Path to background image

        Returns:
            SyntheticImageResult
        """
        result_image = background.copy()
        paste_results = []
        existing_boxes = []
        source_objects = []

        # Randomly select number of objects
        num_objects = self.rng.randint(
            self.config.min_objects_per_image,
            self.config.max_objects_per_image + 1,
        )
        num_objects = min(num_objects, len(objects))

        # Randomly select objects
        selected_indices = self.rng.choice(
            len(objects), size=num_objects, replace=False
        )

        for idx in selected_indices:
            obj, npz_path = objects[idx]

            # Random transformations
            scale = self.rng.uniform(*self.config.scale_range)
            flip = (
                self.config.enable_horizontal_flip
                and self.rng.random() > 0.5
            )

            # Calculate scaled object size
            scaled_h = int(obj.rgba.shape[0] * scale)
            scaled_w = int(obj.rgba.shape[1] * scale)

            # Find paste position
            position = self.find_paste_position(
                bg_shape=result_image.shape[:2],
                obj_shape=(scaled_h, scaled_w),
                existing_boxes=existing_boxes,
            )

            if position is None:
                continue

            # Paste object
            result_image, actual_bbox = self.blend_object(
                background=result_image,
                object_rgba=obj.rgba,
                position=position,
                scale=scale,
                flip_horizontal=flip,
            )

            # Skip if bbox is invalid
            if actual_bbox[2] <= actual_bbox[0] or actual_bbox[3] <= actual_bbox[1]:
                continue

            existing_boxes.append(actual_bbox)
            source_objects.append(npz_path)

            paste_results.append(
                PasteResult(
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    bbox_pixel=actual_bbox,
                    scale_applied=scale,
                    flipped=flip,
                )
            )

        return SyntheticImageResult(
            image=result_image,
            paste_results=paste_results,
            background_path=background_path,
            source_objects=source_objects,
        )

    def generate_yolo_labels(
        self,
        paste_results: List[PasteResult],
        image_shape: Tuple[int, int],
    ) -> List[str]:
        """
        Generate YOLO format labels from paste results.

        Args:
            paste_results: List of paste results
            image_shape: (height, width) of the image

        Returns:
            List of YOLO format label lines
        """
        img_h, img_w = image_shape
        lines = []

        for result in paste_results:
            x_min, y_min, x_max, y_max = result.bbox_pixel

            # Clip to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_w, x_max)
            y_max = min(img_h, y_max)

            # Skip invalid boxes
            if x_max <= x_min or y_max <= y_min:
                continue

            # Convert to YOLO format
            yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
            x_c, y_c, w, h = yolo_bbox

            lines.append(
                f"{result.class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"
            )

        return lines

    def generate_batch(
        self,
        backgrounds_dir: Path,
        annotated_dir: Path,
        output_dir: Path,
        real_image_count: int,
        class_names: List[str],
        progress_callback: Optional[callable] = None,
    ) -> Dict:
        """
        Generate a batch of synthetic images using masks directly.

        Args:
            backgrounds_dir: Directory containing background images
            annotated_dir: Directory containing annotated class subdirectories
                           (each with masks/ and images/ subdirectories)
            output_dir: Output directory for synthetic images
            real_image_count: Number of real images (for ratio calculation)
            class_names: List of class names
            progress_callback: Optional callback(current, total, message)

        Returns:
            Statistics dictionary
        """
        # Calculate number of synthetic images to generate
        num_synthetic = int(real_image_count * self.config.synthetic_to_real_ratio)

        # Create output directories
        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        # Load backgrounds
        backgrounds = self._load_images(backgrounds_dir)
        if not backgrounds:
            return {"error": "No background images found", "generated": 0}

        # Load objects directly from masks (on-the-fly extraction)
        if progress_callback:
            progress_callback(0, num_synthetic, "Loading objects from masks...")

        all_objects, target_resolution = self._load_objects_from_masks(
            annotated_dir=annotated_dir,
            class_names=class_names,
            alpha_blur_sigma=self.config.edge_blur_sigma,
            padding=5,
        )
        if not all_objects:
            return {"error": "No mask files found in annotated directory", "generated": 0}

        stats = {
            "generated": 0,
            "failed": 0,
            "per_class": {name: 0 for name in class_names},
        }

        for i in range(num_synthetic):
            if progress_callback:
                progress_callback(i + 1, num_synthetic, f"Generating image {i + 1}")

            try:
                # Select random background
                bg_path = backgrounds[self.rng.randint(0, len(backgrounds))]
                background = cv2.imread(str(bg_path))
                if background is None:
                    stats["failed"] += 1
                    continue

                # Resize background to match source image resolution
                if target_resolution is not None:
                    target_h, target_w = target_resolution
                    background = cv2.resize(
                        background, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )

                # Generate synthetic image
                result = self.generate_synthetic_image(
                    background=background,
                    objects=all_objects,
                    background_path=str(bg_path),
                )

                if not result.paste_results:
                    stats["failed"] += 1
                    continue

                # Generate filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18]
                filename = f"synthetic_{timestamp}_{i:05d}"

                # Save image
                img_path = images_out / f"{filename}.{self.config.output_image_format}"
                if self.config.output_image_format == "jpg":
                    cv2.imwrite(
                        str(img_path),
                        result.image,
                        [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality],
                    )
                else:
                    cv2.imwrite(str(img_path), result.image)

                # Save labels
                labels = self.generate_yolo_labels(
                    result.paste_results, result.image.shape[:2]
                )
                label_path = labels_out / f"{filename}.txt"
                with open(label_path, "w") as f:
                    f.write("\n".join(labels))

                stats["generated"] += 1

                # Update per-class stats
                for pr in result.paste_results:
                    if pr.class_name in stats["per_class"]:
                        stats["per_class"][pr.class_name] += 1

            except Exception as e:
                stats["failed"] += 1
                if progress_callback:
                    progress_callback(
                        i + 1, num_synthetic, f"Error: {str(e)}"
                    )

        return stats

    def _load_images(self, directory: Path) -> List[Path]:
        """Load all image paths from a directory."""
        if not directory.exists():
            return []

        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(directory.glob(f"*{ext}"))
            images.extend(directory.glob(f"*{ext.upper()}"))

        return images

    def _load_objects_from_masks(
        self,
        annotated_dir: Path,
        class_names: List[str],
        alpha_blur_sigma: float = 2.0,
        padding: int = 5,
    ) -> Tuple[List[Tuple[ExtractedObject, str]], Optional[Tuple[int, int]]]:
        """
        Load objects directly from mask files (no NPZ required).

        Args:
            annotated_dir: Directory containing annotated class subdirectories
            class_names: List of class names to load
            alpha_blur_sigma: Gaussian blur sigma for edge softening
            padding: Padding around object when cropping

        Returns:
            Tuple of (objects_list, target_resolution)
            - objects_list: List of (ExtractedObject, mask_path) tuples
            - target_resolution: (height, width) of source images, or None if no images
        """
        all_objects = []
        target_resolution = None  # (height, width)
        extractor = ObjectExtractor(
            alpha_blur_sigma=alpha_blur_sigma,
            padding=padding,
        )

        for class_id, class_name in enumerate(class_names):
            class_dir = annotated_dir / class_name
            masks_dir = class_dir / "masks"
            images_dir = class_dir / "images"

            if not masks_dir.exists():
                continue

            # Try both naming conventions: *_mask.png and *.png
            mask_patterns = ["*_mask.png", "*.png"]
            mask_files = set()
            for pattern in mask_patterns:
                mask_files.update(masks_dir.glob(pattern))

            for mask_path in mask_files:
                # Strip "_mask" suffix if present to find corresponding image
                img_stem = mask_path.stem
                if img_stem.endswith("_mask"):
                    img_stem = img_stem[:-5]

                img_path = None

                # Check images directory
                if images_dir.exists():
                    for ext in [".jpg", ".jpeg", ".png"]:
                        candidate = images_dir / f"{img_stem}{ext}"
                        if candidate.exists():
                            img_path = candidate
                            break

                if img_path is None:
                    continue

                # Load image and mask
                image_bgr = cv2.imread(str(img_path))
                mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

                if image_bgr is None or mask_gray is None:
                    continue

                # Get target resolution from first valid image
                if target_resolution is None:
                    target_resolution = image_bgr.shape[:2]  # (height, width)

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                mask_bool = mask_gray > 127

                # Extract object on-the-fly
                obj = extractor.extract_object(
                    image_rgb=image_rgb,
                    mask=mask_bool,
                    class_id=class_id,
                    class_name=class_name,
                    source_path=str(img_path),
                )

                if obj is not None:
                    all_objects.append((obj, str(mask_path)))

        return all_objects, target_resolution


def generate_data_yaml(
    output_dir: Path,
    class_names: List[str],
    train_path: str = "images/train",
    val_path: str = "images/val",
) -> None:
    """
    Generate data.yaml for YOLO training.

    Args:
        output_dir: Output directory
        class_names: List of class names
        train_path: Relative path to training images
        val_path: Relative path to validation images
    """
    config = {
        "path": ".",
        "train": train_path,
        "val": val_path,
        "names": {i: name for i, name in enumerate(class_names)},
    }

    yaml_path = output_dir / "data.yaml"
    import yaml
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


if __name__ == "__main__":
    # Test code
    print("CopyPasteAugmentor module loaded successfully")

    config = CopyPasteConfig(
        synthetic_to_real_ratio=2.0,
        scale_range=(0.5, 1.5),
        enable_white_balance=True,
    )

    augmentor = CopyPasteAugmentor(config)
    print(f"Synthetic ratio: {config.synthetic_to_real_ratio}")
    print(f"Scale range: {config.scale_range}")
    print(f"White balance enabled: {config.enable_white_balance}")
