"""
Batch Manager Module

Manages VRAM estimation and batch splitting for large video sequences
in the SAM2 interactive annotation tool.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from video_tracking_predictor import VRAMEstimate


@dataclass
class BatchDecision:
    """
    Result of batch split evaluation.

    Attributes:
        should_split: Whether batching is needed
        vram_estimate: VRAM usage estimate
        batch_count: Number of batches needed
        message: Formatted message for user display
    """

    should_split: bool
    vram_estimate: "VRAMEstimate"
    batch_count: int
    message: str


class BatchManager:
    """
    Manages VRAM-aware batch processing for video tracking.

    Provides:
    - Batch split evaluation based on VRAM estimates
    - Warning message formatting
    - Batch info generation for UI
    """

    def __init__(self, video_predictor=None):
        """
        Initialize batch manager.

        Args:
            video_predictor: VideoTrackingPredictor instance for VRAM estimation
        """
        self.video_predictor = video_predictor

    def set_video_predictor(self, predictor) -> None:
        """Set the video predictor for VRAM estimation."""
        self.video_predictor = predictor

    def evaluate_batch_split(
        self,
        num_frames: int,
        image_size: Tuple[int, int],
        start_frame: int = 0,
    ) -> BatchDecision:
        """
        Evaluate whether batch splitting is needed.

        Args:
            num_frames: Total number of frames to process
            image_size: (height, width) of images
            start_frame: Starting frame index

        Returns:
            BatchDecision with split recommendation
        """
        if self.video_predictor is None:
            # No predictor available, assume no split needed
            return BatchDecision(
                should_split=False,
                vram_estimate=None,
                batch_count=1,
                message="",
            )

        vram_estimate = self.video_predictor.estimate_vram_usage(num_frames, image_size)

        if vram_estimate.needs_split:
            message = self.format_vram_warning(vram_estimate, num_frames, start_frame)
            return BatchDecision(
                should_split=True,
                vram_estimate=vram_estimate,
                batch_count=vram_estimate.num_batches,
                message=message,
            )
        else:
            return BatchDecision(
                should_split=False,
                vram_estimate=vram_estimate,
                batch_count=1,
                message="",
            )

    def format_vram_warning(
        self,
        vram_estimate: "VRAMEstimate",
        num_frames: int,
        start_frame: int = 0,
    ) -> str:
        """
        Format VRAM warning message for display.

        Args:
            vram_estimate: VRAM usage estimate
            num_frames: Total number of frames
            start_frame: Starting frame index

        Returns:
            Formatted warning message
        """
        end_frame = start_frame + num_frames - 1
        message = (
            f"VRAM Capacity Warning\n\n"
            f"Target: frames {start_frame}-{end_frame} ({num_frames} frames)\n"
            f"Estimated VRAM usage: {vram_estimate.estimated_usage_gb:.1f}GB\n"
            f"Available VRAM: {vram_estimate.available_gb:.1f}GB\n\n"
            f"Processing with batch splitting:\n"
        )

        batch_size = vram_estimate.recommended_batch_size
        for i in range(vram_estimate.num_batches):
            batch_start = start_frame + i * batch_size
            batch_end = min(start_frame + (i + 1) * batch_size - 1, end_frame)
            count = batch_end - batch_start + 1
            message += f"- Batch {i + 1}: frames {batch_start}-{batch_end} ({count} images)\n"

        return message

    def create_batch_info_text(
        self,
        batch_idx: int,
        num_batches: int,
        total_frames: int,
        low_conf_count: int,
    ) -> Tuple[str, str]:
        """
        Create batch info text for UI display.

        Args:
            batch_idx: Current batch index (0-based)
            num_batches: Total number of batches
            total_frames: Frames processed in this batch
            low_conf_count: Number of low confidence frames

        Returns:
            Tuple of (batch_info_text, summary_text)
        """
        batch_info = f"Batch {batch_idx + 1}/{num_batches} completed"
        summary = f"({total_frames} frames, {low_conf_count} low confidence)"

        return batch_info, summary

    def get_batch_progress_text(
        self,
        batch_idx: int,
        num_batches: int,
        current_frame: int,
        total_frames: int,
    ) -> str:
        """
        Get progress text for tracking UI.

        Args:
            batch_idx: Current batch index (0-based)
            num_batches: Total number of batches
            current_frame: Current frame within batch
            total_frames: Total frames in batch

        Returns:
            Progress text string
        """
        return f"Tracking: Batch {batch_idx + 1}/{num_batches} ({current_frame}/{total_frames} frames)"

    def get_completion_status(
        self,
        total_frames: int,
        low_conf_count: int,
        was_stopped: bool = False,
    ) -> str:
        """
        Get completion status message.

        Args:
            total_frames: Total frames processed
            low_conf_count: Number of low confidence frames
            was_stopped: Whether tracking was stopped early

        Returns:
            Status message string
        """
        if was_stopped:
            return f"Tracking: Stopped ({total_frames} frames processed)"
        elif low_conf_count > 0:
            return f"Tracking: Complete ({total_frames} frames, {low_conf_count} warnings)"
        else:
            return f"Tracking: Complete ({total_frames} frames)"
