"""
Tracking state management for video annotation.

Manages video tracking mode status, results, low confidence frames,
and excluded frames for training data filtering.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from video_tracking_predictor import TrackingResult


@dataclass
class TrackingState:
    """
    Manages video tracking state for sequential annotation.

    Tracks tracking mode status, results, low confidence frames,
    and excluded frames for training data filtering.
    """

    is_tracking_mode: bool = False
    is_tracking_initialized: bool = False
    tracking_results: Dict[int, "TrackingResult"] = field(default_factory=dict)
    low_confidence_frames: List[int] = field(default_factory=list)
    confirmed_frames: Set[int] = field(default_factory=set)
    excluded_frames: Set[int] = field(default_factory=set)
    current_obj_id: int = 1

    # Stop/pause control fields
    is_processing: bool = False
    stop_requested: bool = False
    is_paused_between_batches: bool = False
    current_batch_index: int = 0
    total_batches: int = 0

    def enable_tracking(self) -> None:
        """Enable tracking mode."""
        self.is_tracking_mode = True

    def disable_tracking(self) -> None:
        """Disable tracking mode and clear results."""
        self.is_tracking_mode = False
        self.is_tracking_initialized = False
        self.tracking_results.clear()
        self.low_confidence_frames.clear()
        self.confirmed_frames.clear()
        self.excluded_frames.clear()

    def set_tracking_results(
        self, results: Dict[int, "TrackingResult"]
    ) -> None:
        """Set tracking results and identify low confidence frames."""
        self.tracking_results = results
        self.is_tracking_initialized = True
        self.low_confidence_frames = [
            idx for idx, result in results.items()
            if result.is_low_confidence
        ]
        self.excluded_frames.clear()

    def get_frame_status(self, frame_idx: int) -> str:
        """
        Get status indicator for a frame.

        Returns:
            Status string: 'confirmed', 'excluded', 'tracked',
            'low_confidence', or 'pending'
        """
        if frame_idx in self.confirmed_frames:
            return "confirmed"
        if frame_idx in self.excluded_frames:
            return "excluded"
        if frame_idx in self.low_confidence_frames:
            return "low_confidence"
        if frame_idx in self.tracking_results:
            return "tracked"
        return "pending"

    def toggle_frame_exclusion(self, frame_idx: int) -> bool:
        """
        Toggle exclusion state of a frame.

        Args:
            frame_idx: Frame index to toggle

        Returns:
            True if frame is now excluded, False if now included
        """
        if frame_idx in self.excluded_frames:
            self.excluded_frames.discard(frame_idx)
            return False
        else:
            self.excluded_frames.add(frame_idx)
            return True

    def is_frame_excluded(self, frame_idx: int) -> bool:
        """Check if frame is excluded."""
        return frame_idx in self.excluded_frames

    def exclude_all_low_confidence(self) -> int:
        """
        Exclude all low confidence frames.

        Returns:
            Number of frames excluded
        """
        count = 0
        for frame_idx in self.low_confidence_frames:
            if frame_idx not in self.excluded_frames:
                self.excluded_frames.add(frame_idx)
                count += 1
        return count

    def include_all(self) -> int:
        """
        Include all frames (clear exclusions).

        Returns:
            Number of frames that were excluded
        """
        count = len(self.excluded_frames)
        self.excluded_frames.clear()
        return count

    def get_included_frames(self) -> List[int]:
        """Get list of frame indices that are not excluded."""
        return [
            idx for idx in self.tracking_results.keys()
            if idx not in self.excluded_frames
        ]

    def get_exclusion_stats(self) -> Tuple[int, int, int]:
        """
        Get exclusion statistics.

        Returns:
            Tuple of (included_count, excluded_count, low_confidence_included_count)
        """
        total = len(self.tracking_results)
        excluded = len(self.excluded_frames)
        included = total - excluded

        low_conf_included = sum(
            1 for idx in self.low_confidence_frames
            if idx not in self.excluded_frames
        )

        return (included, excluded, low_conf_included)

    # Stop/pause control methods
    def request_stop(self) -> None:
        """Request processing to stop."""
        self.stop_requested = True

    def clear_stop_request(self) -> None:
        """Clear stop request flag."""
        self.stop_requested = False

    def is_stop_requested(self) -> bool:
        """Check if stop has been requested."""
        return self.stop_requested

    def set_processing(self, processing: bool) -> None:
        """Set processing state."""
        self.is_processing = processing
        if not processing:
            self.stop_requested = False
            self.is_paused_between_batches = False

    def pause_for_batch_review(self, batch_idx: int, total_batches: int) -> None:
        """Set state for batch review pause."""
        self.is_paused_between_batches = True
        self.current_batch_index = batch_idx
        self.total_batches = total_batches

    def resume_from_pause(self) -> None:
        """Resume from batch review pause."""
        self.is_paused_between_batches = False
