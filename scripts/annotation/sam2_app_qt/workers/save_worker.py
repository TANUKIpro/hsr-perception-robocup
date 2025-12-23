"""
Save worker for async annotation saving.

Provides QThread-based worker for saving annotations in the background.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

from PyQt6.QtCore import QThread, pyqtSignal

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_tracking_predictor import TrackingResult
from annotation_utils import batch_save_yolo_labels


class SaveWorker(QThread):
    """
    Worker thread for saving annotations.

    Signals:
        progress: Emitted with progress percentage
        finished: Emitted with save result
        error: Emitted when save fails
    """

    progress = pyqtSignal(int)  # percentage
    finished = pyqtSignal(object)  # SaveResult
    error = pyqtSignal(str)

    def __init__(
        self,
        tracking_results: Dict[int, TrackingResult],
        frame_map: Dict[int, Path],
        output_dir: str,
        class_id: int,
        copy_images: bool = True,
        parent: Optional[object] = None,
    ):
        """
        Initialize save worker.

        Args:
            tracking_results: Dict of frame_idx -> TrackingResult
            frame_map: Dict of frame_idx -> image path
            output_dir: Output directory for labels
            class_id: YOLO class ID
            copy_images: Whether to copy images
            parent: Parent QObject
        """
        super().__init__(parent)
        self.tracking_results = tracking_results
        self.frame_map = frame_map
        self.output_dir = output_dir
        self.class_id = class_id
        self.copy_images = copy_images

    def run(self) -> None:
        """Save annotations in background thread."""
        try:
            result = batch_save_yolo_labels(
                tracking_results=self.tracking_results,
                frame_map=self.frame_map,
                output_dir=self.output_dir,
                class_id=self.class_id,
                copy_images=self.copy_images,
            )
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
