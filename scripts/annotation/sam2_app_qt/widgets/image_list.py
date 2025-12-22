"""
Image list widget for file navigation.

Provides a QListWidget-based widget for displaying and selecting images
with annotation status indicators.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor


class ImageListWidget(QListWidget):
    """
    Image list widget with annotation status.

    Signals:
        image_selected: Emitted with index when image is selected
        image_double_clicked: Emitted with index on double-click
    """

    image_selected = pyqtSignal(int)
    image_double_clicked = pyqtSignal(int)

    # Status colors
    COLOR_ANNOTATED = QColor("black")
    COLOR_TRACKED = QColor("green")
    COLOR_LOW_CONF = QColor("orange")
    COLOR_EXCLUDED = QColor("gray")
    COLOR_PENDING = QColor("black")

    def __init__(self, parent: Optional[object] = None):
        """Initialize image list widget."""
        super().__init__(parent)

        self.image_paths: List[Path] = []
        self.annotated_images: Set[Path] = set()
        self.tracking_results: Dict[int, object] = {}
        self.excluded_frames: Set[int] = set()
        self.low_confidence_frames: List[int] = []

        # Setup widget
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def set_images(self, image_paths: List[Path]) -> None:
        """
        Set the list of image paths.

        Args:
            image_paths: List of image file paths
        """
        self.image_paths = image_paths
        self._update_display()

    def set_annotated(self, annotated: Set[Path]) -> None:
        """
        Set the annotated images.

        Args:
            annotated: Set of annotated image paths
        """
        self.annotated_images = annotated
        self._update_display()

    def set_tracking_state(
        self,
        results: Dict[int, object],
        excluded: Set[int],
        low_conf: List[int],
    ) -> None:
        """
        Set tracking state for display.

        Args:
            results: Tracking results dict
            excluded: Set of excluded frame indices
            low_conf: List of low confidence frame indices
        """
        self.tracking_results = results
        self.excluded_frames = excluded
        self.low_confidence_frames = low_conf
        self._update_display()

    def select_index(self, index: int) -> None:
        """
        Select an item by index.

        Args:
            index: Index to select
        """
        if 0 <= index < self.count():
            self.setCurrentRow(index)
            self.scrollToItem(self.item(index))

    def _update_display(self) -> None:
        """Update the list display."""
        self.clear()

        for i, path in enumerate(self.image_paths):
            prefix, color = self._get_status_display(i, path)
            display_name = f"{prefix} {path.name}"

            item = QListWidgetItem(display_name)
            item.setForeground(color)
            self.addItem(item)

    def _get_status_display(
        self, index: int, path: Path
    ) -> tuple[str, QColor]:
        """
        Get status prefix and color for an item.

        Args:
            index: Item index
            path: Image path

        Returns:
            Tuple of (prefix, color)
        """
        if path in self.annotated_images:
            return "[OK]", self.COLOR_ANNOTATED

        if index in self.tracking_results:
            is_excluded = index in self.excluded_frames
            is_low_conf = index in self.low_confidence_frames

            if is_excluded:
                if is_low_conf:
                    return "[X!]", self.COLOR_EXCLUDED
                return "[X]", self.COLOR_EXCLUDED
            else:
                if is_low_conf:
                    return "[T!]", self.COLOR_LOW_CONF
                return "[T]", self.COLOR_TRACKED

        return "[ ]", self.COLOR_PENDING

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        """Handle item click."""
        index = self.row(item)
        self.image_selected.emit(index)

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        """Handle item double-click."""
        index = self.row(item)
        self.image_double_clicked.emit(index)
