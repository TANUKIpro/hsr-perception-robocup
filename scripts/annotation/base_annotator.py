"""
Base Annotator

Abstract base class for image annotators providing common interfaces
and shared implementations for batch processing and visualization.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Add scripts directory to path for common module imports
_scripts_dir = Path(__file__).parent.parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

from common.constants import IMAGE_EXTENSIONS
from annotation_utils import AnnotationResult, write_yolo_label
from visualization import visualize_annotation_result


class BaseAnnotator(ABC):
    """
    Abstract base class for image annotators.

    Provides common interfaces and shared implementations for:
    - Single image annotation (abstract, must be implemented)
    - Batch annotation with progress tracking
    - Annotation visualization

    Subclasses must implement:
    - annotate_image(): Core annotation logic returning YOLO bbox
    """

    @abstractmethod
    def annotate_image(
        self,
        image_path: str,
        **kwargs,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Generate YOLO format annotation for a single image.

        Args:
            image_path: Path to input image
            **kwargs: Subclass-specific options

        Returns:
            Tuple (x_center, y_center, width, height) normalized to [0, 1],
            or None if no object detected
        """
        pass

    def annotate_batch(
        self,
        image_dir: str,
        class_id: int,
        output_dir: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        desc: str = "Annotating",
    ) -> AnnotationResult:
        """
        Annotate all images in a directory.

        This default implementation can be overridden for specific needs
        (e.g., SAM2 mask saving).

        Args:
            image_dir: Directory containing images to annotate
            class_id: YOLO class ID for all images
            output_dir: Directory for output label files
            progress_callback: Optional callback(current, total) for progress
            desc: Description for progress bar

        Returns:
            AnnotationResult with statistics
        """
        image_path = Path(image_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find all images
        images = [
            f
            for f in image_path.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        result = AnnotationResult(total_images=len(images))

        for i, img_file in enumerate(tqdm(images, desc=desc)):
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
        window_title: str = "Annotation",
    ) -> np.ndarray:
        """
        Visualize annotation result on image.

        Default implementation using shared visualization module.
        Can be overridden for custom visualization (e.g., mask overlay).

        Args:
            image_path: Path to input image
            output_path: Optional path to save visualization
            show: If True, display image in window
            window_title: Window title for display

        Returns:
            Annotated image with bounding box drawn
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Get annotation
        bbox = self.annotate_image(image_path)

        # Visualize
        result = visualize_annotation_result(image, bbox)

        if output_path:
            cv2.imwrite(output_path, result)

        if show:
            cv2.imshow(window_title, result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return result
