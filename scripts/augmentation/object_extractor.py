"""
Object Extractor for Copy-Paste Augmentation

Extracts objects from images using SAM2 masks and saves them as RGBA NPZ files.
The alpha channel includes edge blurring for natural blending during paste operations.
"""

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add scripts directory to path for imports
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from annotation.annotation_utils import mask_to_bbox


@dataclass
class ExtractedObject:
    """Data structure for an extracted object."""

    rgba: np.ndarray  # (H, W, 4) Cropped RGBA image
    class_id: int
    class_name: str
    source_image_path: str
    source_image_stem: str
    original_bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max) in source image
    original_image_size: Tuple[int, int]  # (height, width) of source image
    extraction_timestamp: str

    def to_metadata(self) -> Dict:
        """Return JSON-serializable metadata."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "source_image_path": self.source_image_path,
            "source_image_stem": self.source_image_stem,
            "original_bbox": list(self.original_bbox),
            "original_image_size": list(self.original_image_size),
            "rgba_shape": list(self.rgba.shape),
            "extraction_timestamp": self.extraction_timestamp,
        }


class ObjectExtractor:
    """
    Extract objects from images using masks.

    Uses SAM2-generated masks to cut out objects with soft alpha edges
    for natural blending during Copy-Paste augmentation.
    """

    def __init__(
        self,
        alpha_blur_sigma: float = 2.0,
        padding: int = 5,
        min_object_size: int = 32,
    ):
        """
        Initialize ObjectExtractor.

        Args:
            alpha_blur_sigma: Gaussian blur sigma for edge softening (default: 2.0)
            padding: Padding around the object when cropping (default: 5 pixels)
            min_object_size: Minimum object size in pixels (default: 32)
        """
        self.alpha_blur_sigma = alpha_blur_sigma
        self.padding = padding
        self.min_object_size = min_object_size

    def extract_object(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        class_id: int,
        class_name: str,
        source_path: str,
    ) -> Optional[ExtractedObject]:
        """
        Extract an object from an image using a mask.

        Args:
            image_rgb: RGB image (H, W, 3)
            mask: Boolean mask (H, W)
            class_id: YOLO class ID
            class_name: Class name
            source_path: Path to source image

        Returns:
            ExtractedObject or None if extraction fails
        """
        # 1. Get bounding box from mask
        bbox = mask_to_bbox(mask, use_contour=True)
        if bbox is None:
            return None

        x_min, y_min, x_max, y_max = bbox

        # 2. Size check
        obj_width = x_max - x_min
        obj_height = y_max - y_min
        if obj_width < self.min_object_size or obj_height < self.min_object_size:
            return None

        # 3. Calculate crop region with padding
        h, w = image_rgb.shape[:2]
        x1_pad = max(0, x_min - self.padding)
        y1_pad = max(0, y_min - self.padding)
        x2_pad = min(w, x_max + self.padding)
        y2_pad = min(h, y_max + self.padding)

        # 4. Crop image and mask
        cropped_rgb = image_rgb[y1_pad:y2_pad, x1_pad:x2_pad].copy()
        cropped_mask = mask[y1_pad:y2_pad, x1_pad:x2_pad].copy()

        # 5. Create soft alpha channel with edge blur
        alpha = self._create_soft_alpha(cropped_mask)

        # 6. Build RGBA image
        rgba = np.dstack([cropped_rgb, alpha])

        return ExtractedObject(
            rgba=rgba,
            class_id=class_id,
            class_name=class_name,
            source_image_path=source_path,
            source_image_stem=Path(source_path).stem,
            original_bbox=(x1_pad, y1_pad, x2_pad, y2_pad),
            original_image_size=(h, w),
            extraction_timestamp=datetime.now().isoformat(),
        )

    def _create_soft_alpha(self, mask: np.ndarray) -> np.ndarray:
        """
        Create a soft alpha channel from a mask with blurred edges.

        Args:
            mask: Boolean mask (H, W)

        Returns:
            Alpha channel (H, W) as uint8 [0-255]
        """
        # Convert mask to 0-255
        alpha = mask.astype(np.uint8) * 255

        if self.alpha_blur_sigma > 0:
            # Calculate kernel size from sigma (6-sigma rule, must be odd)
            ksize = int(np.ceil(self.alpha_blur_sigma * 6)) | 1

            # Apply Gaussian blur to soften edges
            alpha_blurred = cv2.GaussianBlur(
                alpha, (ksize, ksize), self.alpha_blur_sigma
            )

            # Keep interior fully opaque by restoring original mask interior
            # Erode the mask to get the safe interior region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=2)
            alpha_blurred[eroded > 0] = 255

            return alpha_blurred

        return alpha

    def save_extracted_object(
        self,
        obj: ExtractedObject,
        output_dir: str,
    ) -> str:
        """
        Save an extracted object to NPZ file.

        Args:
            obj: ExtractedObject to save
            output_dir: Output directory

        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename: {class_name}_{source_stem}_{hash}.npz
        timestamp_hash = hashlib.md5(
            obj.extraction_timestamp.encode()
        ).hexdigest()[:8]
        base_name = f"{obj.class_name}_{obj.source_image_stem}_{timestamp_hash}"

        file_path = output_path / f"{base_name}.npz"

        # Save RGBA + metadata as NPZ
        np.savez_compressed(
            file_path,
            rgba=obj.rgba,
            metadata=json.dumps(obj.to_metadata()),
        )

        return str(file_path)

    @staticmethod
    def load_extracted_object(file_path: str) -> Optional[ExtractedObject]:
        """
        Load an extracted object from NPZ file.

        Args:
            file_path: Path to NPZ file

        Returns:
            ExtractedObject or None if loading fails
        """
        path = Path(file_path)

        if not path.exists() or path.suffix != ".npz":
            return None

        try:
            data = np.load(file_path, allow_pickle=True)
            rgba = data["rgba"]
            metadata = json.loads(str(data["metadata"]))

            return ExtractedObject(
                rgba=rgba,
                class_id=metadata["class_id"],
                class_name=metadata["class_name"],
                source_image_path=metadata["source_image_path"],
                source_image_stem=metadata["source_image_stem"],
                original_bbox=tuple(metadata["original_bbox"]),
                original_image_size=tuple(metadata["original_image_size"]),
                extraction_timestamp=metadata["extraction_timestamp"],
            )
        except Exception:
            return None

    def batch_extract_from_images_and_masks(
        self,
        image_dir: Path,
        mask_dir: Path,
        class_id: int,
        class_name: str,
        output_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> List[str]:
        """
        Extract objects from a batch of images using corresponding masks.

        Args:
            image_dir: Directory containing source images
            mask_dir: Directory containing mask images (same names as source)
            class_id: YOLO class ID
            class_name: Class name
            output_dir: Output directory for NPZ files
            progress_callback: Optional callback for progress updates

        Returns:
            List of saved file paths
        """
        saved_paths = []

        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]

        total = len(images)
        for idx, img_path in enumerate(images):
            if progress_callback:
                progress_callback(idx + 1, total, f"Processing {img_path.name}")

            # Find corresponding mask
            mask_path = mask_dir / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue

            # Load image and mask
            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Convert mask to boolean
            mask_bool = mask > 127

            # Extract object
            obj = self.extract_object(
                image_rgb=image_rgb,
                mask=mask_bool,
                class_id=class_id,
                class_name=class_name,
                source_path=str(img_path),
            )

            if obj is not None:
                saved_path = self.save_extracted_object(obj, str(output_dir))
                saved_paths.append(saved_path)

        return saved_paths

    def extract_from_image_and_yolo_label(
        self,
        image_path: Path,
        label_path: Path,
        class_names: List[str],
        output_dir: Path,
        sam_predictor: Optional[object] = None,
    ) -> List[str]:
        """
        Extract objects from an image using YOLO labels (requires SAM2 for mask generation).

        This method uses YOLO bounding boxes as prompts for SAM2 to generate masks,
        then extracts the objects.

        Args:
            image_path: Path to source image
            label_path: Path to YOLO label file
            class_names: List of class names (index = class_id)
            output_dir: Output directory for NPZ files
            sam_predictor: SAM2 predictor instance (optional)

        Returns:
            List of saved file paths
        """
        # This requires SAM2 to generate masks from bounding boxes
        # For now, this is a placeholder for future implementation
        raise NotImplementedError(
            "extract_from_image_and_yolo_label requires SAM2 predictor. "
            "Use batch_extract_from_images_and_masks with pre-generated masks instead."
        )


if __name__ == "__main__":
    # Test code
    print("ObjectExtractor module loaded successfully")

    # Create test extractor
    extractor = ObjectExtractor(
        alpha_blur_sigma=2.0,
        padding=5,
        min_object_size=32,
    )

    print(f"Alpha blur sigma: {extractor.alpha_blur_sigma}")
    print(f"Padding: {extractor.padding}")
    print(f"Min object size: {extractor.min_object_size}")
