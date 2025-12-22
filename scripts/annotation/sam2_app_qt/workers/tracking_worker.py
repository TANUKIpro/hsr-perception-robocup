"""
Tracking worker for async video tracking.

Provides QThread-based worker for SAM2 video tracking operations.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from video_tracking_predictor import (
    VideoTrackingPredictor,
    TrackingResult,
    BatchInfo,
)


class TrackingWorker(QThread):
    """
    Worker thread for video tracking.

    Signals:
        progress: Emitted with (current, total, frame_idx, mask)
        batch_complete: Emitted when a batch is complete
        finished: Emitted with final results dict
        error: Emitted when tracking fails
        status: Emitted with status messages
    """

    progress = pyqtSignal(int, int, int, object)  # current, total, frame_idx, mask
    batch_complete = pyqtSignal(int, int, dict)  # batch_idx, num_batches, results
    finished = pyqtSignal(dict)  # results dict
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(
        self,
        video_predictor: VideoTrackingPredictor,
        image_list: List[Path],
        start_frame: int,
        foreground_points: List[Tuple[int, int]],
        background_points: List[Tuple[int, int]],
        obj_id: int = 1,
        batch_size: Optional[int] = None,
        parent: Optional[object] = None,
    ):
        """
        Initialize tracking worker.

        Args:
            video_predictor: VideoTrackingPredictor instance
            image_list: List of image paths
            start_frame: Frame index to start tracking from
            foreground_points: Initial foreground points
            background_points: Initial background points
            obj_id: Object ID for tracking
            batch_size: Batch size for processing (None for auto)
            parent: Parent QObject
        """
        super().__init__(parent)
        self.video_predictor = video_predictor
        self.image_list = image_list
        self.start_frame = start_frame
        self.foreground_points = foreground_points
        self.background_points = background_points
        self.obj_id = obj_id
        self.batch_size = batch_size

        self._stop_requested = False
        self._pause_requested = False

    def request_stop(self) -> None:
        """Request tracking to stop."""
        self._stop_requested = True

    def clear_stop_request(self) -> None:
        """Clear stop request."""
        self._stop_requested = False

    def request_pause(self) -> None:
        """Request tracking to pause."""
        self._pause_requested = True

    def resume(self) -> None:
        """Resume from pause."""
        self._pause_requested = False

    def run(self) -> None:
        """Run tracking in background thread."""
        try:
            end_frame = len(self.image_list) - 1
            num_frames = end_frame - self.start_frame + 1

            # Create batch info
            batch_info = BatchInfo(
                batch_index=0,
                start_frame=self.start_frame,
                end_frame=end_frame,
                frame_count=num_frames,
                is_first_batch=True,
                is_last_batch=True,
            )

            self.status.emit(f"Preparing frames {self.start_frame}-{end_frame}...")

            # Initialize sequence
            self.video_predictor.init_batch_sequence(
                image_paths=self.image_list,
                batch_info=batch_info,
                progress_callback=lambda msg: self.status.emit(msg),
            )

            # Set initial prompt
            self.status.emit("Setting initial prompt...")
            self.video_predictor.set_initial_prompt(
                frame_idx=0,
                obj_id=self.obj_id,
                foreground_points=self.foreground_points,
                background_points=self.background_points,
            )

            # Propagate tracking
            self.status.emit("Propagating...")

            def progress_callback(current: int, total: int, local_idx: int, mask: np.ndarray) -> None:
                global_idx = self.start_frame + local_idx
                self.progress.emit(current, total, global_idx, mask)

            def stop_check() -> bool:
                return self._stop_requested

            local_results = self.video_predictor.propagate_tracking(
                progress_callback=progress_callback,
                stop_check=stop_check,
            )

            # Convert to global indices
            global_results = {
                self.start_frame + local_idx: result
                for local_idx, result in local_results.items()
            }

            self.finished.emit(global_results)

        except Exception as e:
            self.error.emit(str(e))
