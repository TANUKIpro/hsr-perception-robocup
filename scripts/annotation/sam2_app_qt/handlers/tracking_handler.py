"""
Tracking handler for SAM2 annotation application.

Handles all tracking-related operations including start, stop, apply, and exclusion.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, TYPE_CHECKING

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QMessageBox

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_tracking_predictor import VideoTrackingPredictor, TrackingResult

if TYPE_CHECKING:
    from sam2_app_qt.state import AnnotationState, TrackingState
    from sam2_app_qt.workers import TrackingWorker, SaveWorker


class TrackingHandler(QObject):
    """
    Handler for tracking operations.

    Signals:
        status_updated: Emitted when status message changes
        progress_updated: Emitted with progress percentage
        display_update_requested: Emitted when display should update
        image_list_update_requested: Emitted when image list should update
    """

    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    display_update_requested = pyqtSignal()
    image_list_update_requested = pyqtSignal()
    tracking_complete = pyqtSignal(dict)
    save_complete = pyqtSignal(object, int)  # result, excluded_count

    def __init__(
        self,
        model_path: str,
        device: str,
        tracking_state: "TrackingState",
        annotation_state: "AnnotationState",
        parent: Optional[QObject] = None,
    ):
        """
        Initialize tracking handler.

        Args:
            model_path: Path to SAM2 model
            device: Device to run model on
            tracking_state: TrackingState instance
            annotation_state: AnnotationState instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self.model_path = model_path
        self.device = device
        self.tracking_state = tracking_state
        self.annotation_state = annotation_state

        self.video_predictor: Optional[VideoTrackingPredictor] = None
        self._tracking_worker: Optional["TrackingWorker"] = None
        self._save_worker: Optional["SaveWorker"] = None

        self.frame_map: Dict[int, Path] = {}
        self.image_list = []
        self.current_index = 0

    def set_image_list(self, image_list, current_index: int) -> None:
        """Set the current image list and index."""
        self.image_list = image_list
        self.current_index = current_index

    def start_tracking(
        self,
        update_frame_callback: Callable[[int, np.ndarray], None],
    ) -> bool:
        """
        Start video tracking from current mask.

        Args:
            update_frame_callback: Callback to update displayed frame

        Returns:
            True if tracking started, False otherwise
        """
        from sam2_app_qt.workers import TrackingWorker

        if not self.tracking_state.is_tracking_mode:
            return False

        if self.annotation_state.current_mask is None:
            return False

        self.status_updated.emit("Initializing...")
        self.tracking_state.set_processing(True)
        self.tracking_state.clear_stop_request()

        # Initialize video predictor
        if self.video_predictor is None:
            self.video_predictor = VideoTrackingPredictor(
                model_path=self.model_path,
                device=self.device,
            )

        self._update_frame_callback = update_frame_callback

        # Start tracking worker
        self._tracking_worker = TrackingWorker(
            video_predictor=self.video_predictor,
            image_list=self.image_list,
            start_frame=self.current_index,
            foreground_points=self.annotation_state.foreground_points,
            background_points=self.annotation_state.background_points,
            obj_id=self.tracking_state.current_obj_id,
            parent=self,
        )
        self._tracking_worker.progress.connect(self._on_progress)
        self._tracking_worker.finished.connect(self._on_complete)
        self._tracking_worker.error.connect(self._on_error)
        self._tracking_worker.status.connect(self.status_updated.emit)
        self._tracking_worker.start()

        return True

    def stop_tracking(self) -> None:
        """Request tracking to stop."""
        if self._tracking_worker:
            self._tracking_worker.request_stop()

    def cancel_tracking(self) -> None:
        """Cancel tracking and reset state."""
        self.tracking_state.disable_tracking()

        if self.video_predictor:
            self.video_predictor.reset()

        self.frame_map.clear()

    def apply_results(self, output_dir: str, class_id: int) -> None:
        """
        Apply tracking results and save annotations.

        Args:
            output_dir: Output directory for annotations
            class_id: YOLO class ID
        """
        from sam2_app_qt.workers import SaveWorker

        if not self.tracking_state.is_tracking_initialized:
            return

        # Filter excluded frames
        included_results = {
            idx: res for idx, res in self.tracking_state.tracking_results.items()
            if not self.tracking_state.is_frame_excluded(idx)
        }
        included_map = {
            idx: path for idx, path in self.frame_map.items()
            if not self.tracking_state.is_frame_excluded(idx)
        }

        excluded_count = len(self.tracking_state.excluded_frames)

        self.status_updated.emit("Saving annotations...")

        self._save_worker = SaveWorker(
            tracking_results=included_results,
            frame_map=included_map,
            output_dir=output_dir,
            class_id=class_id,
            copy_images=True,
            parent=self,
        )
        self._save_worker.finished.connect(
            lambda r: self.save_complete.emit(r, excluded_count)
        )
        self._save_worker.error.connect(self._on_save_error)
        self._save_worker.start()

    def toggle_exclusion(self, frame_idx: int) -> bool:
        """
        Toggle exclusion of a frame.

        Args:
            frame_idx: Frame index to toggle

        Returns:
            True if frame is now excluded
        """
        if frame_idx not in self.tracking_state.tracking_results:
            return False

        return self.tracking_state.toggle_frame_exclusion(frame_idx)

    def exclude_all_low_confidence(self) -> int:
        """Exclude all low confidence frames."""
        return self.tracking_state.exclude_all_low_confidence()

    def include_all(self) -> int:
        """Include all frames."""
        return self.tracking_state.include_all()

    @pyqtSlot(int, int, int, object)
    def _on_progress(
        self, current: int, total: int, frame_idx: int, mask: np.ndarray
    ) -> None:
        """Handle tracking progress."""
        progress = int((current / total) * 100)
        self.progress_updated.emit(progress)
        self.status_updated.emit(f"{current}/{total} frames")

        # Store result
        if self.video_predictor:
            temp_result = TrackingResult.from_mask(
                mask,
                reference_area=self.video_predictor.reference_mask_area,
            )
            self.tracking_state.tracking_results[frame_idx] = temp_result
            self.tracking_state.is_tracking_initialized = True

            if frame_idx < len(self.image_list):
                self.frame_map[frame_idx] = self.image_list[frame_idx]

        # Callback to update display
        if self._update_frame_callback:
            self._update_frame_callback(frame_idx, mask)

    @pyqtSlot(dict)
    def _on_complete(self, results: Dict[int, TrackingResult]) -> None:
        """Handle tracking completion."""
        self.tracking_state.set_processing(False)
        self.tracking_state.set_tracking_results(results)

        low_conf = len(self.tracking_state.low_confidence_frames)
        total = len(results)

        status = f"Complete ({total} frames"
        if low_conf > 0:
            status += f", {low_conf} warnings)"
        else:
            status += ")"

        self.status_updated.emit(status)
        self.tracking_complete.emit(results)

    @pyqtSlot(str)
    def _on_error(self, error: str) -> None:
        """Handle tracking error."""
        self.tracking_state.set_processing(False)
        self.status_updated.emit("Failed")

    @pyqtSlot(str)
    def _on_save_error(self, error: str) -> None:
        """Handle save error."""
        self.status_updated.emit("Save failed")
