"""
Tracking panel widget for video tracking controls.

Provides controls for tracking mode, batch processing, and frame exclusion.
"""

from typing import Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QCheckBox,
    QFrame,
)
from PyQt6.QtCore import pyqtSignal


class TrackingPanel(QWidget):
    """
    Tracking panel widget with tracking controls.

    Signals:
        tracking_toggled: Emitted when tracking mode is toggled
        start_tracking: Emitted when Start Tracking is clicked
        apply_results: Emitted when Apply is clicked
        cancel_tracking: Emitted when Cancel is clicked
        stop_tracking: Emitted when Stop is clicked
        toggle_exclusion: Emitted when Toggle Selected is clicked
        exclude_low_conf: Emitted when Exclude Low-Conf is clicked
        include_all: Emitted when Include All is clicked
        continue_batch: Emitted when Continue Batch is clicked
        stop_at_batch: Emitted when Stop at Batch is clicked
    """

    tracking_toggled = pyqtSignal(bool)
    start_tracking = pyqtSignal()
    apply_results = pyqtSignal()
    cancel_tracking = pyqtSignal()
    stop_tracking = pyqtSignal()
    toggle_exclusion = pyqtSignal()
    exclude_low_conf = pyqtSignal()
    include_all = pyqtSignal()
    continue_batch = pyqtSignal()
    stop_at_batch = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize tracking panel."""
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Main tracking controls group
        group = QGroupBox("Tracking Mode")
        group_layout = QVBoxLayout(group)

        # Top row: main controls
        top_row = QHBoxLayout()

        self.tracking_checkbox = QCheckBox("Enable Tracking Mode")
        self.tracking_checkbox.toggled.connect(self.tracking_toggled.emit)
        top_row.addWidget(self.tracking_checkbox)

        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking.emit)
        self.start_btn.setEnabled(False)
        top_row.addWidget(self.start_btn)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_results.emit)
        self.apply_btn.setEnabled(False)
        top_row.addWidget(self.apply_btn)

        self.cancel_btn = QPushButton("Cancel Tracking")
        self.cancel_btn.clicked.connect(self.cancel_tracking.emit)
        self.cancel_btn.setEnabled(False)
        top_row.addWidget(self.cancel_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_tracking.emit)
        self.stop_btn.setEnabled(False)
        top_row.addWidget(self.stop_btn)

        top_row.addStretch()

        # Status labels
        self.low_conf_label = QLabel("")
        self.low_conf_label.setStyleSheet("color: orange; font-weight: bold;")
        top_row.addWidget(self.low_conf_label)

        self.status_label = QLabel("Tracking: OFF")
        top_row.addWidget(self.status_label)

        group_layout.addLayout(top_row)

        # Exclusion controls frame (initially hidden)
        self.exclusion_frame = QFrame()
        exclusion_layout = QHBoxLayout(self.exclusion_frame)
        exclusion_layout.setContentsMargins(0, 5, 0, 0)

        self.exclusion_stats_label = QLabel("")
        exclusion_layout.addWidget(self.exclusion_stats_label)

        self.toggle_selected_btn = QPushButton("Toggle Selected")
        self.toggle_selected_btn.clicked.connect(self.toggle_exclusion.emit)
        self.toggle_selected_btn.setEnabled(False)
        exclusion_layout.addWidget(self.toggle_selected_btn)

        self.exclude_low_btn = QPushButton("Exclude Low-Conf")
        self.exclude_low_btn.clicked.connect(self.exclude_low_conf.emit)
        self.exclude_low_btn.setEnabled(False)
        exclusion_layout.addWidget(self.exclude_low_btn)

        self.include_all_btn = QPushButton("Include All")
        self.include_all_btn.clicked.connect(self.include_all.emit)
        self.include_all_btn.setEnabled(False)
        exclusion_layout.addWidget(self.include_all_btn)

        exclusion_layout.addStretch()
        self.exclusion_frame.hide()
        group_layout.addWidget(self.exclusion_frame)

        # Batch pause frame (shown between batches)
        self.batch_frame = QFrame()
        batch_layout = QHBoxLayout(self.batch_frame)
        batch_layout.setContentsMargins(0, 5, 0, 0)

        self.batch_info_label = QLabel("")
        self.batch_info_label.setStyleSheet("font-weight: bold;")
        batch_layout.addWidget(self.batch_info_label)

        self.batch_summary_label = QLabel("")
        batch_layout.addWidget(self.batch_summary_label)

        self.continue_btn = QPushButton("Continue to Next Batch")
        self.continue_btn.clicked.connect(self.continue_batch.emit)
        batch_layout.addWidget(self.continue_btn)

        self.stop_at_btn = QPushButton("Stop Here")
        self.stop_at_btn.clicked.connect(self.stop_at_batch.emit)
        batch_layout.addWidget(self.stop_at_btn)

        batch_layout.addStretch()
        self.batch_frame.hide()
        group_layout.addWidget(self.batch_frame)

        main_layout.addWidget(group)

    def set_tracking_enabled(self, enabled: bool) -> None:
        """Set tracking mode checkbox state."""
        self.tracking_checkbox.setChecked(enabled)

    def update_ui_state(
        self,
        is_enabled: bool,
        has_mask: bool,
        has_results: bool,
        is_processing: bool,
        is_paused: bool,
    ) -> None:
        """
        Update UI state based on tracking state.

        Args:
            is_enabled: Whether tracking mode is enabled
            has_mask: Whether a mask is available
            has_results: Whether tracking results exist
            is_processing: Whether tracking is in progress
            is_paused: Whether paused between batches
        """
        # Start tracking
        can_start = is_enabled and has_mask and not has_results and not is_processing
        self.start_btn.setEnabled(can_start)

        # Stop button
        can_stop = is_processing and not is_paused
        self.stop_btn.setEnabled(can_stop)

        # Apply and Cancel
        can_apply = has_results and not is_processing
        self.apply_btn.setEnabled(can_apply)
        self.cancel_btn.setEnabled(can_apply)

        # Exclusion controls
        show_exclusion = (has_results and not is_processing) or is_paused
        if show_exclusion:
            self.exclusion_frame.show()
            self.toggle_selected_btn.setEnabled(True)
            self.exclude_low_btn.setEnabled(True)
            self.include_all_btn.setEnabled(True)
        else:
            self.exclusion_frame.hide()
            self.toggle_selected_btn.setEnabled(False)
            self.exclude_low_btn.setEnabled(False)
            self.include_all_btn.setEnabled(False)

    def update_status(self, text: str) -> None:
        """Update status label."""
        self.status_label.setText(text)

    def update_low_conf_warning(self, count: int) -> None:
        """Update low confidence warning."""
        if count > 0:
            self.low_conf_label.setText(f"Warning: {count} low confidence frames")
        else:
            self.low_conf_label.setText("")

    def update_exclusion_stats(
        self, included: int, excluded: int, low_conf_included: int
    ) -> None:
        """Update exclusion statistics display."""
        total = included + excluded
        text = f"Included: {included}/{total}"
        if excluded > 0:
            text += f" ({excluded} excluded)"
        if low_conf_included > 0:
            text += f" | {low_conf_included} low-conf included"
        self.exclusion_stats_label.setText(text)

    def show_batch_pause(
        self, batch_idx: int, num_batches: int, total_frames: int, low_conf: int
    ) -> None:
        """Show batch pause controls."""
        self.batch_info_label.setText(f"Batch {batch_idx + 1}/{num_batches} completed")
        self.batch_summary_label.setText(
            f"({total_frames} frames, {low_conf} low confidence)"
        )
        self.batch_frame.show()

    def hide_batch_pause(self) -> None:
        """Hide batch pause controls."""
        self.batch_frame.hide()
