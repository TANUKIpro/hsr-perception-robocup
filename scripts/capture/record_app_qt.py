#!/usr/bin/env python3
"""
ROS2 Video Recording Application (PyQt6 Version)

A PyQt6-based GUI application for recording video from ROS2 topics
and extracting frames for training data.

Features:
- Real-time camera preview with center reticle
- Topic selection from available ROS2 image topics
- Video recording with countdown timer
- Automatic frame extraction at uniform intervals
- MP4 video saving

Usage:
    python3 record_app_qt.py

Controls:
    - Select topic from dropdown
    - Set class name and target frame count
    - Click "Start Recording" to begin (3s countdown)
    - Click "Stop Recording" to finish and extract frames
    - Press 'q' or close window to exit
"""

import argparse
import re
import sys
import time
from datetime import datetime
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
    QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from object_registry import ObjectRegistry, RegisteredObject
from gui_framework_qt import ROS2App
from gui_framework_qt.components import TopicSelector, StatusBar
from gui_framework_qt.utils import numpy_to_pixmap

# Import drawing utilities from original gui_framework
from gui_framework.utils import draw_reticle, draw_countdown, draw_recording_indicator

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw_captures"
DEFAULT_VIDEO_DIR = Path(__file__).parent.parent.parent / "data" / "videos"


class RecordApp(ROS2App):
    """Main recording application with PyQt6 GUI."""

    def __init__(self) -> None:
        """Initialize the recording application."""
        # Recording state
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_start_time = 0.0
        self.frame_count = 0
        self.current_video_path: Optional[Path] = None
        self.current_class_name: Optional[str] = None

        # Countdown state
        self.countdown_active = False
        self.countdown_remaining = 0
        self.countdown_start_time = 0.0

        # Extraction state
        self.is_extracting = False

        # Registry state
        self.registry: Optional[ObjectRegistry] = None
        self.registered_objects: list[RegisteredObject] = []

        # Initialize ROS2 app (this calls _build_gui)
        super().__init__(
            title="ROS2 Video Recording",
            node_name="record_app",
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

        # === Recording Settings ===
        settings_group = QGroupBox("Recording Settings")
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

        # Image output directory row
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Image Output:"))
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR))
        output_row.addWidget(self.output_edit, stretch=1)

        browse_btn = QPushButton("Browse")
        browse_btn.setMaximumWidth(80)
        browse_btn.clicked.connect(self._browse_output_dir)
        output_row.addWidget(browse_btn)
        settings_layout.addLayout(output_row)

        # Video output directory row
        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video Output:"))
        self.video_dir_edit = QLineEdit(str(DEFAULT_VIDEO_DIR))
        video_row.addWidget(self.video_dir_edit, stretch=1)

        browse_video_btn = QPushButton("Browse")
        browse_video_btn.setMaximumWidth(80)
        browse_video_btn.clicked.connect(self._browse_video_dir)
        video_row.addWidget(browse_video_btn)
        settings_layout.addLayout(video_row)

        self.main_layout.addWidget(settings_group)

        # === Extraction Parameters ===
        extract_group = QGroupBox("Frame Extraction Parameters")
        extract_layout = QHBoxLayout(extract_group)

        extract_layout.addWidget(QLabel("Target Frames:"))
        self.target_frames_spin = QSpinBox()
        self.target_frames_spin.setRange(10, 500)
        self.target_frames_spin.setValue(50)
        extract_layout.addWidget(self.target_frames_spin)

        extract_layout.addSpacing(20)
        extract_layout.addWidget(
            QLabel("(Frames extracted uniformly from video)")
        )
        extract_layout.addStretch()

        self.main_layout.addWidget(extract_group)

        # === Recording Controls ===
        control_layout = QHBoxLayout()

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setMinimumHeight(40)
        self.record_btn.clicked.connect(self._toggle_recording)
        control_layout.addWidget(self.record_btn)

        control_layout.addSpacing(15)

        self.time_label = QLabel("00:00")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.time_label.setFont(font)
        control_layout.addWidget(self.time_label)

        control_layout.addStretch()

        self.main_layout.addLayout(control_layout)

        # === Status Bar ===
        self.status_bar = StatusBar(self)
        self.main_layout.addWidget(self.status_bar)

        # Initial setup
        self.topic_selector.refresh_topics()
        self._refresh_registry()

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
                self._disable_record_button()
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
                self._enable_record_button()
                self.status_bar.set_status(
                    f"Loaded {len(self.registered_objects)} object(s) from registry"
                )
        except Exception as e:
            self.class_combo.clear()
            self._disable_record_button()
            self.status_bar.set_status(f"Failed to load registry: {e}")

    def _disable_record_button(self) -> None:
        """Disable record button when no objects are available."""
        self.record_btn.setEnabled(False)

    def _enable_record_button(self) -> None:
        """Enable record button."""
        self.record_btn.setEnabled(True)

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
            "Select Image Output Directory",
            current if Path(current).is_dir() else str(DEFAULT_OUTPUT_DIR),
        )
        if directory:
            self.output_edit.setText(directory)

    def _browse_video_dir(self) -> None:
        """Open directory browser for video output path."""
        current = self.video_dir_edit.text()
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Video Output Directory",
            current if Path(current).is_dir() else str(DEFAULT_VIDEO_DIR),
        )
        if directory:
            self.video_dir_edit.setText(directory)

    @pyqtSlot()
    def _update_preview(self) -> None:
        """Update the preview image."""
        frame = self.get_frame()

        if frame is not None:
            # Write frame to video if recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)
                self.frame_count += 1

            # Draw reticle
            display = draw_reticle(frame.copy())

            # Add countdown overlay if counting down
            if self.countdown_active:
                display = draw_countdown(display, self.countdown_remaining)

            # Add recording status overlay if recording
            elif self.is_recording:
                elapsed = time.time() - self.recording_start_time
                display = draw_recording_indicator(
                    display, elapsed, self.frame_count
                )

            # Resize for display
            display = self._resize_for_display(display)

            # Convert to QPixmap and display
            pixmap = numpy_to_pixmap(display)
            self.preview_label.setPixmap(pixmap)

        # Handle countdown
        if self.countdown_active:
            self._process_countdown()

        # Update recording time
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self.time_label.setText(f"{mins:02d}:{secs:02d}")

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
            self._begin_actual_recording()

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

    def _toggle_recording(self) -> None:
        """Start or stop recording."""
        if self.is_recording or self.countdown_active:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        """Start recording with countdown."""
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

        video_dir = Path(self.video_dir_edit.text())
        video_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H-%M")
        video_filename = f"{class_name}_{timestamp}.mp4"
        self.current_video_path = video_dir / video_filename
        self.current_class_name = class_name

        self.countdown_active = True
        self.countdown_remaining = 3
        self.countdown_start_time = time.time()

        self.record_btn.setText("Cancel")
        self.status_bar.show_progress(0, 100)
        self.status_bar.set_status("Starting in 3...")

    def _begin_actual_recording(self) -> None:
        """Begin the actual recording after countdown."""
        frame = self.get_frame()
        if frame is None:
            self.status_bar.set_status("Failed to start recording: no frame")
            self.record_btn.setText("Start Recording")
            return

        h, w = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            str(self.current_video_path),
            fourcc,
            30.0,
            (w, h),
        )

        if not self.video_writer.isOpened():
            self.status_bar.set_status("Failed to create video file")
            self.record_btn.setText("Start Recording")
            return

        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_count = 0

        self.record_btn.setText("Stop Recording")
        self.status_bar.set_status("Recording...")

    def _get_next_file_number(self, output_dir: Path, class_name: str) -> int:
        """Get the next available file number for accumulation."""
        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$")
        max_num = 0

        if output_dir.exists():
            for f in output_dir.iterdir():
                match = pattern.match(f.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)

        return max_num + 1

    def _stop_recording(self) -> None:
        """Stop recording and extract frames."""
        was_countdown = self.countdown_active
        was_recording = self.is_recording

        self.countdown_active = False
        self.is_recording = False
        self.record_btn.setText("Start Recording")
        self.time_label.setText("00:00")

        if was_countdown:
            self.status_bar.set_status("Recording cancelled")
            self.status_bar.hide_progress()
            return

        if was_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

            if self.frame_count > 0:
                self.status_bar.set_status(
                    f"Recording saved: {self.current_video_path.name} "
                    f"({self.frame_count} frames)"
                )
                self._extract_frames()
            else:
                self.status_bar.set_status("No frames recorded")
                if self.current_video_path.exists():
                    self.current_video_path.unlink()

    def _extract_frames(self) -> None:
        """Extract frames from recorded video."""
        if not self.current_video_path or not self.current_video_path.exists():
            return

        self.is_extracting = True
        self.status_bar.set_status("Extracting frames...")
        self.status_bar.show_progress(0, 100)

        target_count = self.target_frames_spin.value()
        class_name = self.current_class_name
        output_dir = Path(self.output_edit.text()) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        start_num = self._get_next_file_number(output_dir, class_name)

        cap = cv2.VideoCapture(str(self.current_video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            self.status_bar.set_status("Error: Video has no frames")
            self.is_extracting = False
            return

        if total_frames <= target_count:
            indices = list(range(total_frames))
        else:
            indices = [
                int(i * total_frames / target_count)
                for i in range(target_count)
            ]

        extracted = 0
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                filename = f"{class_name}_{start_num + i}.jpg"
                cv2.imwrite(str(output_dir / filename), frame)
                extracted += 1

            progress = ((i + 1) / len(indices)) * 100
            self.status_bar.update_progress(progress)
            QApplication.processEvents()

        cap.release()
        self.is_extracting = False

        self.status_bar.set_status(
            f"Extraction complete: {extracted} images saved "
            f"(#{start_num}-{start_num + extracted - 1})"
        )
        self.status_bar.update_progress(100)

    def _on_close(self) -> None:
        """Handle window close."""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()

        self._preview_timer.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ROS2 Video Recording Application"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Default output directory for extracted images",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=str(DEFAULT_VIDEO_DIR),
        help="Default output directory for video files",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    window = RecordApp()

    if args.output_dir:
        window.output_edit.setText(args.output_dir)
    if args.video_dir:
        window.video_dir_edit.setText(args.video_dir)

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
