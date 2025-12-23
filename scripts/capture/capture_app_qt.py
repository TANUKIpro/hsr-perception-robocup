#!/usr/bin/env python3
"""
ROS2 Image Capture Application (PyQt6 Version)

A PyQt6-based GUI application for capturing images from ROS2 topics.
Features:
- Real-time camera preview with center reticle
- Topic selection from available ROS2 image topics
- Configurable burst capture parameters
- Single image capture
- Integration with ObjectRegistry

Usage:
    python3 capture_app_qt.py

Controls:
    - Select topic from dropdown
    - Set class name and output directory
    - Adjust burst parameters (count, interval)
    - Click "Single Capture" or "Start Burst" to capture
    - Press 'q' or close window to exit
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from object_registry import ObjectRegistry, RegisteredObject
from gui_framework_qt import ROS2App
from gui_framework_qt.components import TopicSelector, StatusBar
from gui_framework_qt.utils import numpy_to_pixmap

# Import drawing utilities from original gui_framework
from gui_framework.utils import draw_reticle, draw_countdown

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw_captures"


class CaptureApp(ROS2App):
    """Main capture application with PyQt6 GUI."""

    def __init__(self) -> None:
        """Initialize the capture application."""
        # Capture state
        self.is_capturing_burst = False
        self.burst_count = 0
        self.burst_target = 0
        self.last_capture_time = 0.0
        self.burst_interval = 0.2

        # Countdown state
        self.countdown_active = False
        self.countdown_remaining = 0
        self.countdown_start_time = 0.0

        # Current output info
        self.current_output_dir: Optional[Path] = None
        self.current_class_name: Optional[str] = None
        self.burst_start_num = 1
        self.pending_burst_count = 0
        self.pending_burst_interval = 0.2

        # Registry state
        self.registry: Optional[ObjectRegistry] = None
        self.registered_objects: list[RegisteredObject] = []

        # Initialize ROS2 app (this calls _build_gui)
        super().__init__(
            title="ROS2 Image Capture",
            node_name="capture_app",
            size=(800, 750),
            min_size=(600, 700),
        )

        # Preview timer
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(33)  # ~30 FPS

    def _build_gui(self) -> None:
        """Build the GUI layout."""
        # === Topic Selection ===
        topic_group = QGroupBox("Topic Selection")
        topic_layout = QVBoxLayout(topic_group)
        self.topic_selector = TopicSelector(
            parent=topic_group,
            get_topics_callback=self.get_image_topics,
            subscribe_callback=self.subscribe_to_topic,
        )
        self.topic_selector.topic_changed.connect(self._on_topic_changed)
        topic_layout.addWidget(self.topic_selector)
        self.main_layout.addWidget(topic_group)

        # === Preview Area ===
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("No image - Select a topic")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1;"
        )
        self.preview_label.setMinimumSize(640, 360)
        preview_layout.addWidget(self.preview_label)
        self.main_layout.addWidget(preview_group, stretch=1)

        # === Capture Settings ===
        settings_group = QGroupBox("Capture Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Class name row
        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class Name:"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(250)
        class_row.addWidget(self.class_combo, stretch=1)

        refresh_registry_btn = QPushButton("Refresh")
        refresh_registry_btn.setMaximumWidth(80)
        refresh_registry_btn.clicked.connect(self._refresh_registry)
        class_row.addWidget(refresh_registry_btn)
        settings_layout.addLayout(class_row)

        # Output directory row
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output Directory:"))
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        output_row.addWidget(self.output_edit, stretch=1)

        browse_btn = QPushButton("Browse")
        browse_btn.setMaximumWidth(80)
        browse_btn.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse_btn)
        settings_layout.addLayout(output_row)

        self.main_layout.addWidget(settings_group)

        # === Burst Parameters ===
        burst_group = QGroupBox("Burst Capture Parameters")
        burst_layout = QHBoxLayout(burst_group)

        # Number of images
        burst_layout.addWidget(QLabel("Images:"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 500)
        self.count_spin.setValue(50)
        self.count_spin.valueChanged.connect(self._update_estimated_time)
        burst_layout.addWidget(self.count_spin)

        burst_layout.addSpacing(20)

        # Interval
        burst_layout.addWidget(QLabel("Interval (s):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.05, 5.0)
        self.interval_spin.setSingleStep(0.05)
        self.interval_spin.setValue(0.2)
        self.interval_spin.valueChanged.connect(self._update_estimated_time)
        burst_layout.addWidget(self.interval_spin)

        burst_layout.addSpacing(20)

        # Estimated time
        burst_layout.addWidget(QLabel("Est. Time:"))
        self.est_time_label = QLabel("10.0s")
        burst_layout.addWidget(self.est_time_label)

        burst_layout.addSpacing(20)

        # Overwrite checkbox
        self.overwrite_check = QCheckBox("Overwrite")
        self.overwrite_check.setChecked(True)
        burst_layout.addWidget(self.overwrite_check)

        burst_layout.addStretch()

        self.main_layout.addWidget(burst_group)

        # === Capture Buttons ===
        button_layout = QHBoxLayout()

        self.single_btn = QPushButton("Single Capture")
        self.single_btn.setMinimumHeight(40)
        self.single_btn.clicked.connect(self._single_capture)
        button_layout.addWidget(self.single_btn)

        button_layout.addSpacing(15)

        self.burst_btn = QPushButton("Start Burst Capture")
        self.burst_btn.setMinimumHeight(40)
        self.burst_btn.clicked.connect(self._toggle_burst_capture)
        button_layout.addWidget(self.burst_btn)

        self.main_layout.addLayout(button_layout)

        # === Status Bar ===
        self.status_bar = StatusBar(self)
        self.main_layout.addWidget(self.status_bar)

        # Initial setup
        self.topic_selector.refresh_topics()
        self._refresh_registry()
        self._update_estimated_time()

    @pyqtSlot(str)
    def _on_topic_changed(self, topic: str) -> None:
        """Handle topic selection change."""
        if topic:
            self.status_bar.set_status(f"Subscribed to {topic}")

    def _refresh_registry(self) -> None:
        """Load objects from registry and populate the class dropdown."""
        try:
            self.registry = ObjectRegistry()
            self.registered_objects = self.registry.get_all_objects()

            self.class_combo.clear()

            if not self.registered_objects:
                self._disable_capture_buttons()
                self.status_bar.set_status(
                    "No objects in registry. Please add objects first."
                )
                self.show_warning(
                    "No Objects",
                    "No objects are registered in the Registry.\n"
                    "Please add objects in the Registry first.",
                )
            else:
                values = [
                    f"{obj.name} ({obj.display_name})"
                    for obj in self.registered_objects
                ]
                self.class_combo.addItems(values)
                self._enable_capture_buttons()
                self.status_bar.set_status(
                    f"Loaded {len(self.registered_objects)} object(s) from registry"
                )
        except Exception as e:
            self.class_combo.clear()
            self._disable_capture_buttons()
            self.status_bar.set_status(f"Failed to load registry: {e}")

    def _disable_capture_buttons(self) -> None:
        """Disable capture buttons when no objects are available."""
        self.single_btn.setEnabled(False)
        self.burst_btn.setEnabled(False)

    def _enable_capture_buttons(self) -> None:
        """Enable capture buttons."""
        self.single_btn.setEnabled(True)
        self.burst_btn.setEnabled(True)

    def _get_selected_class_name(self) -> Optional[str]:
        """Get the selected object name (for filenames)."""
        selected = self.class_combo.currentText()
        if not selected:
            return None
        return selected.split(" (")[0]

    def _browse_output_dir(self) -> None:
        """Open directory browser for output path."""
        current = self.output_edit.text()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            current if Path(current).is_dir() else str(DEFAULT_OUTPUT_DIR),
        )
        if directory:
            self.output_edit.setText(directory)

    def _update_estimated_time(self) -> None:
        """Update the estimated time label."""
        count = self.count_spin.value()
        interval = self.interval_spin.value()
        est_time = count * interval
        self.est_time_label.setText(f"{est_time:.1f}s")

    @pyqtSlot()
    def _update_preview(self) -> None:
        """Update the preview image."""
        frame = self.get_frame()

        if frame is not None:
            # Draw reticle
            display = draw_reticle(frame.copy())

            # Add countdown overlay if counting down
            if self.countdown_active:
                display = draw_countdown(display, self.countdown_remaining)

            # Add capture status overlay if capturing
            elif self.is_capturing_burst:
                display = self._draw_capture_status(display)

            # Resize for display
            display = self._resize_for_display(display)

            # Convert to QPixmap and display
            pixmap = numpy_to_pixmap(display)
            self.preview_label.setPixmap(pixmap)

        # Handle countdown
        if self.countdown_active:
            self._process_countdown()

        # Handle burst capture
        if self.is_capturing_burst:
            self._process_burst_capture()

    def _draw_capture_status(self, frame: np.ndarray) -> np.ndarray:
        """Draw capture status overlay."""
        # Red recording indicator
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

        # Progress text
        text = f"Capturing: {self.burst_count}/{self.burst_target}"
        cv2.putText(
            frame,
            text,
            (50, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        return frame

    def _process_countdown(self) -> None:
        """Process countdown timer."""
        elapsed = time.time() - self.countdown_start_time
        new_remaining = 3 - int(elapsed)

        if new_remaining != self.countdown_remaining:
            self.countdown_remaining = new_remaining
            if self.countdown_remaining > 0:
                self.status_bar.set_status(
                    f"Starting in {self.countdown_remaining}..."
                )

        if elapsed >= 3.0:
            self.countdown_active = False
            self._begin_actual_capture()

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit display area dynamically."""
        label_width = self.preview_label.width()
        label_height = self.preview_label.height()

        if label_width <= 1 or label_height <= 1:
            return frame

        h, w = frame.shape[:2]
        scale = min(label_width / w, label_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != w or new_h != h:
            frame = cv2.resize(
                frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )

        return frame

    def _get_next_file_number(self, output_dir: Path, class_name: str) -> int:
        """Get the next file number for the given class."""
        if self.overwrite_check.isChecked():
            return 1

        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$")
        max_num = 0

        if output_dir.exists():
            for f in output_dir.iterdir():
                match = pattern.match(f.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)

        return max_num + 1

    def _single_capture(self) -> None:
        """Capture a single image."""
        frame = self.get_frame()
        if frame is None:
            self.show_warning("No Image", "No image available to capture")
            return

        class_name = self._get_selected_class_name()
        if not class_name:
            self.show_warning(
                "Invalid Input", "Please select a class from the dropdown"
            )
            return

        output_dir = Path(self.output_edit.text()) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        file_num = self._get_next_file_number(output_dir, class_name)
        filename = f"{class_name}_{file_num}.jpg"
        filepath = output_dir / filename

        cv2.imwrite(str(filepath), frame)
        self.status_bar.set_status(f"Saved: {filename}")

    def _toggle_burst_capture(self) -> None:
        """Start or stop burst capture."""
        if self.is_capturing_burst or self.countdown_active:
            self._stop_burst_capture()
        else:
            self._start_burst_capture()

    def _start_burst_capture(self) -> None:
        """Start burst capture with countdown."""
        if self.get_frame() is None:
            self.show_warning(
                "No Image", "No image available. Check topic connection."
            )
            return

        class_name = self._get_selected_class_name()
        if not class_name:
            self.show_warning(
                "Invalid Input", "Please select a class from the dropdown"
            )
            return

        count = self.count_spin.value()
        interval = self.interval_spin.value()

        output_dir = Path(self.output_edit.text()) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.current_output_dir = output_dir
        self.current_class_name = class_name

        self.pending_burst_count = count
        self.pending_burst_interval = interval

        self.countdown_active = True
        self.countdown_remaining = 3
        self.countdown_start_time = time.time()

        self.burst_btn.setText("Cancel")
        self.single_btn.setEnabled(False)
        self.status_bar.show_progress(0, 100)
        self.status_bar.set_status("Starting in 3...")

    def _begin_actual_capture(self) -> None:
        """Begin the actual burst capture after countdown."""
        self.burst_start_num = self._get_next_file_number(
            self.current_output_dir,
            self.current_class_name,
        )

        self.is_capturing_burst = True
        self.burst_count = 0
        self.burst_target = self.pending_burst_count
        self.burst_interval = self.pending_burst_interval
        self.last_capture_time = 0.0

        self.burst_btn.setText("Stop Capture")
        self.status_bar.set_status(f"Capturing: 0/{self.burst_target}")

    def _process_burst_capture(self) -> None:
        """Process burst capture (called from preview loop)."""
        if not self.is_capturing_burst:
            return

        current_time = time.time()
        if current_time - self.last_capture_time >= self.burst_interval:
            frame = self.get_frame()
            if frame is not None:
                file_num = self.burst_start_num + self.burst_count
                filename = f"{self.current_class_name}_{file_num}.jpg"
                filepath = self.current_output_dir / filename

                cv2.imwrite(str(filepath), frame)
                self.burst_count += 1
                self.last_capture_time = current_time

                progress = (self.burst_count / self.burst_target) * 100
                self.status_bar.update_progress(progress)
                self.status_bar.set_status(
                    f"Capturing: {self.burst_count}/{self.burst_target}"
                )

                if self.burst_count >= self.burst_target:
                    self._stop_burst_capture()
                    self.status_bar.set_status(
                        f"Burst complete: {self.burst_count} images saved"
                    )

    def _stop_burst_capture(self) -> None:
        """Stop burst capture or cancel countdown."""
        was_countdown = self.countdown_active
        was_capturing = self.is_capturing_burst

        self.countdown_active = False
        self.is_capturing_burst = False
        self.burst_btn.setText("Start Burst Capture")
        self.single_btn.setEnabled(True)
        self.status_bar.hide_progress()

        if was_countdown:
            self.status_bar.set_status("Capture cancelled")
        elif was_capturing and self.burst_count < self.burst_target:
            self.status_bar.set_status(
                f"Burst stopped: {self.burst_count}/{self.burst_target} images captured"
            )

    def _on_close(self) -> None:
        """Handle window close."""
        self._preview_timer.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ROS2 Image Capture Application"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Default output directory for captured images",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    window = CaptureApp()

    if args.output_dir:
        window.output_edit.setText(args.output_dir)

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
