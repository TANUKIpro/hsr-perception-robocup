"""
Control panel widget for annotation actions.

Provides buttons and controls for annotation operations.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QFrame,
)
from PyQt6.QtCore import pyqtSignal


class ControlPanel(QWidget):
    """
    Control panel widget with annotation controls.

    Signals:
        reset_clicked: Emitted when Reset button is clicked
        undo_clicked: Emitted when Undo button is clicked
        accept_clicked: Emitted when Accept button is clicked
        skip_clicked: Emitted when Skip button is clicked
        prev_clicked: Emitted when Prev button is clicked
        next_clicked: Emitted when Next button is clicked
        mask_toggle_changed: Emitted when mask visibility is toggled
    """

    reset_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    accept_clicked = pyqtSignal()
    skip_clicked = pyqtSignal()
    prev_clicked = pyqtSignal()
    next_clicked = pyqtSignal()
    mask_toggle_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize control panel."""
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Point info
        info_frame = QFrame()
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(0, 0, 0, 0)

        self.point_label = QLabel("Points: FG=0, BG=0")
        info_layout.addWidget(self.point_label)

        self.iou_label = QLabel("IoU: --")
        self.iou_label.setStyleSheet("margin-left: 15px;")
        info_layout.addWidget(self.iou_label)

        layout.addWidget(info_frame)
        layout.addSpacing(20)

        # Action buttons
        self.reset_btn = QPushButton("Reset (Esc)")
        self.reset_btn.clicked.connect(self.reset_clicked.emit)
        layout.addWidget(self.reset_btn)

        self.undo_btn = QPushButton("Undo (Ctrl+Z)")
        self.undo_btn.clicked.connect(self.undo_clicked.emit)
        layout.addWidget(self.undo_btn)

        self.accept_btn = QPushButton("Accept (Enter)")
        self.accept_btn.clicked.connect(self.accept_clicked.emit)
        layout.addWidget(self.accept_btn)

        self.skip_btn = QPushButton("Skip (S)")
        self.skip_btn.clicked.connect(self.skip_clicked.emit)
        layout.addWidget(self.skip_btn)

        layout.addSpacing(20)

        # Navigation buttons
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(self.prev_clicked.emit)
        layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.next_clicked.emit)
        layout.addWidget(self.next_btn)

        layout.addStretch()

        # Mask toggle
        self.mask_checkbox = QCheckBox("Show Mask")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.toggled.connect(self.mask_toggle_changed.emit)
        layout.addWidget(self.mask_checkbox)

    def update_point_info(self, fg_count: int, bg_count: int) -> None:
        """Update point count display."""
        self.point_label.setText(f"Points: FG={fg_count}, BG={bg_count}")

    def update_iou(self, iou: Optional[float]) -> None:
        """Update IoU display."""
        if iou is not None:
            self.iou_label.setText(f"IoU: {iou:.3f}")
        else:
            self.iou_label.setText("IoU: --")

    def is_mask_visible(self) -> bool:
        """Check if mask visibility is enabled."""
        return self.mask_checkbox.isChecked()

    def set_mask_visible(self, visible: bool) -> None:
        """Set mask visibility."""
        self.mask_checkbox.setChecked(visible)
