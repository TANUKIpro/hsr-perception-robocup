"""
Annotation state management with undo history.

Tracks foreground/background points, current mask, and provides
undo functionality for iterative refinement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AnnotationState:
    """
    Manages annotation state with undo history.

    Attributes:
        foreground_points: List of (x, y) foreground points
        background_points: List of (x, y) background points
        current_mask: Current segmentation mask
        current_iou: IoU score of current mask
        history: Undo history stack
        max_history: Maximum history entries
    """

    foreground_points: List[Tuple[int, int]] = field(default_factory=list)
    background_points: List[Tuple[int, int]] = field(default_factory=list)
    current_mask: Optional[np.ndarray] = None
    current_iou: float = 0.0
    history: List[Dict] = field(default_factory=list)
    max_history: int = 10

    def add_foreground_point(self, x: int, y: int) -> None:
        """Add foreground point and save to history."""
        self._save_to_history()
        self.foreground_points.append((x, y))

    def add_background_point(self, x: int, y: int) -> None:
        """Add background point and save to history."""
        self._save_to_history()
        self.background_points.append((x, y))

    def _save_to_history(self) -> None:
        """Save current state to history."""
        state = {
            "fg": self.foreground_points.copy(),
            "bg": self.background_points.copy(),
            "mask": self.current_mask.copy() if self.current_mask is not None else None,
            "iou": self.current_iou,
        }
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self) -> bool:
        """
        Undo last point addition.

        Returns:
            True if undo was successful, False if no history
        """
        if not self.history:
            return False

        state = self.history.pop()
        self.foreground_points = state["fg"]
        self.background_points = state["bg"]
        self.current_mask = state["mask"]
        self.current_iou = state["iou"]
        return True

    def reset(self) -> None:
        """Reset all points and mask."""
        self.foreground_points.clear()
        self.background_points.clear()
        self.current_mask = None
        self.current_iou = 0.0
        self.history.clear()

    def has_points(self) -> bool:
        """Check if any points exist."""
        return len(self.foreground_points) > 0 or len(self.background_points) > 0

    def get_point_counts(self) -> Tuple[int, int]:
        """Get counts of foreground and background points."""
        return len(self.foreground_points), len(self.background_points)
