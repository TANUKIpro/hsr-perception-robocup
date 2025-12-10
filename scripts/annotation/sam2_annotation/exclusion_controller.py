"""
Exclusion Controller Module

Manages frame exclusion logic for filtering training data
in the SAM2 interactive annotation tool.
"""

from typing import Callable, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from sam2_interactive_app import TrackingState


class ExclusionController:
    """
    Controls frame exclusion for training data filtering.

    Provides:
    - Toggle individual frame exclusion
    - Bulk exclusion of low confidence frames
    - Exclusion statistics
    """

    def __init__(
        self,
        tracking_state: "TrackingState",
        on_update: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize exclusion controller.

        Args:
            tracking_state: Reference to tracking state for accessing results
            on_update: Callback when exclusion state changes
        """
        self.tracking_state = tracking_state
        self.on_update = on_update

    def toggle_frame(self, frame_idx: int) -> Tuple[bool, str]:
        """
        Toggle exclusion state of a frame.

        Args:
            frame_idx: Frame index to toggle

        Returns:
            Tuple of (is_now_excluded, status_message)
        """
        if not self.tracking_state.is_tracking_initialized:
            return False, "Tracking not initialized"

        if frame_idx not in self.tracking_state.tracking_results:
            return False, "Frame has no tracking results"

        is_excluded = self.tracking_state.toggle_frame_exclusion(frame_idx)

        if self.on_update:
            self.on_update()

        if is_excluded:
            return True, f"Frame {frame_idx} excluded from training"
        else:
            return False, f"Frame {frame_idx} included in training"

    def exclude_all_low_confidence(self) -> Tuple[int, str]:
        """
        Exclude all low confidence frames.

        Returns:
            Tuple of (count_excluded, status_message)
        """
        if not self.tracking_state.is_tracking_initialized:
            return 0, "Tracking not initialized"

        count = self.tracking_state.exclude_all_low_confidence()

        if self.on_update:
            self.on_update()

        if count > 0:
            return count, f"Excluded {count} low confidence frames"
        else:
            return 0, "No additional frames to exclude"

    def include_all(self) -> Tuple[int, str]:
        """
        Include all frames (clear exclusions).

        Returns:
            Tuple of (count_included, status_message)
        """
        if not self.tracking_state.is_tracking_initialized:
            return 0, "Tracking not initialized"

        count = self.tracking_state.include_all()

        if self.on_update:
            self.on_update()

        if count > 0:
            return count, f"Included {count} previously excluded frames"
        else:
            return 0, "All frames already included"

    def get_stats_text(self) -> str:
        """
        Get exclusion statistics text for display.

        Returns:
            Formatted statistics string
        """
        if not self.tracking_state.is_tracking_initialized:
            return ""

        included, excluded, low_conf_included = self.tracking_state.get_exclusion_stats()
        total = included + excluded

        if excluded > 0:
            stats_text = f"Included: {included}/{total} ({excluded} excluded)"
        else:
            stats_text = f"Included: {included}/{total}"

        if low_conf_included > 0:
            stats_text += f" | {low_conf_included} low-conf included"

        return stats_text

    def get_low_confidence_warning(self) -> str:
        """
        Get low confidence warning text.

        Returns:
            Warning text or empty string if no low confidence frames
        """
        if not self.tracking_state.is_tracking_initialized:
            return ""

        low_conf_count = len(self.tracking_state.low_confidence_frames)
        if low_conf_count > 0:
            return f"Warning: {low_conf_count} low confidence frames"
        return ""

    def get_confirmation_message(self) -> str:
        """
        Get confirmation message for save dialog.

        Returns:
            Formatted confirmation message
        """
        included, excluded, low_conf_included = self.tracking_state.get_exclusion_stats()

        msg = f"Save annotations for {included} frames?"
        if excluded > 0:
            msg += f"\n\n({excluded} frames excluded)"
        if low_conf_included > 0:
            msg += f"\nWarning: {low_conf_included} low confidence frames included"

        return msg
