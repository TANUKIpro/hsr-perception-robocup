"""
Image Navigator Module

Handles image list loading, navigation, and annotation status tracking
for the SAM2 interactive annotation tool.
"""

import re
from pathlib import Path
from typing import Callable, List, Optional, Set


class ImageNavigator:
    """
    Manages image list and navigation for annotation workflow.

    Provides:
    - Natural sort ordering for image files
    - Annotation status tracking
    - Next/previous navigation with bounds checking
    """

    SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        on_changed: Optional[Callable[[int], None]] = None,
    ):
        """
        Initialize image navigator.

        Args:
            input_dir: Directory containing images to annotate
            output_dir: Directory for output labels (for checking existing annotations)
            on_changed: Callback when current image changes, receives new index
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.on_changed = on_changed

        self.image_list: List[Path] = []
        self.current_index: int = 0
        self.annotated_images: Set[Path] = set()

    def load_images(self) -> int:
        """
        Load list of images from input directory.

        Returns:
            Number of images found
        """
        self.image_list = sorted(
            [
                f
                for f in self.input_dir.iterdir()
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ],
            key=self._natural_sort_key,
        )

        # Check for existing annotations
        self._scan_existing_annotations()

        return len(self.image_list)

    def _natural_sort_key(self, path: Path):
        """Sort key for natural/alphanumeric ordering."""
        return [
            int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", path.name)
        ]

    def _scan_existing_annotations(self) -> None:
        """Scan for existing annotation files and update annotated set."""
        labels_dir = self.output_dir / "labels"

        for img_path in self.image_list:
            # Check in labels subdirectory first (batch save location)
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                # Fallback to output_dir directly (legacy location)
                label_path = self.output_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.annotated_images.add(img_path)

    def navigate_next(self) -> bool:
        """
        Navigate to next image.

        Returns:
            True if navigation was successful, False if at end
        """
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            if self.on_changed:
                self.on_changed(self.current_index)
            return True
        return False

    def navigate_prev(self) -> bool:
        """
        Navigate to previous image.

        Returns:
            True if navigation was successful, False if at start
        """
        if self.current_index > 0:
            self.current_index -= 1
            if self.on_changed:
                self.on_changed(self.current_index)
            return True
        return False

    def navigate_to(self, index: int) -> bool:
        """
        Navigate to specific image index.

        Args:
            index: Target image index

        Returns:
            True if navigation was successful, False if index out of bounds
        """
        if 0 <= index < len(self.image_list):
            self.current_index = index
            if self.on_changed:
                self.on_changed(self.current_index)
            return True
        return False

    def mark_annotated(self, path: Path) -> None:
        """Mark an image as annotated."""
        self.annotated_images.add(path)

    def is_annotated(self, path: Path) -> bool:
        """Check if an image is annotated."""
        return path in self.annotated_images

    @property
    def current_path(self) -> Optional[Path]:
        """Get current image path."""
        if self.image_list and 0 <= self.current_index < len(self.image_list):
            return self.image_list[self.current_index]
        return None

    @property
    def total_count(self) -> int:
        """Get total number of images."""
        return len(self.image_list)

    @property
    def annotated_count(self) -> int:
        """Get number of annotated images."""
        return len(self.annotated_images)

    def get_progress_text(self) -> str:
        """Get progress text for display."""
        current = self.current_index + 1 if self.image_list else 0
        total = len(self.image_list)
        annotated = len(self.annotated_images)
        return f"Image {current}/{total} | Annotated: {annotated}/{total}"
