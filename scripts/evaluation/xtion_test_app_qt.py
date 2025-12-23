#!/usr/bin/env python3
"""
Xtion Live Test Application (PyQt6 Version)

A PyQt6-based GUI application for real-time YOLO inference on ROS2 camera topics.
Features:
- Real-time camera preview with detection overlay
- Topic selection from available ROS2 image topics
- Configurable confidence threshold
- FPS and detection statistics display

Usage:
    python3 xtion_test_app_qt.py --model path/to/model.pt --conf 0.25
"""

import argparse
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
    QSlider,
    QTextEdit,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot

# YOLO import
from ultralytics import YOLO

# Add gui_framework paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_framework_qt import ROS2App
from gui_framework_qt.components import TopicSelector, PreviewPanel
from gui_framework_qt.utils import numpy_to_pixmap


class XtionTestApp(ROS2App):
    """Main Xtion test application with PyQt6 GUI."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
    ) -> None:
        """
        Initialize the Xtion test application.

        Args:
            model_path: Path to YOLO model file
            conf_threshold: Initial confidence threshold
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # YOLO model (loaded later)
        self.model: Optional[YOLO] = None

        # FPS calculation
        self.frame_times: list[float] = []
        self.fps = 0.0

        # Detection results
        self.latest_detections: list[dict] = []

        # Load YOLO model before building GUI
        self._load_model()

        # Initialize ROS2 app (this calls _build_gui)
        super().__init__(
            title="Xtion Live Test - YOLO Inference",
            node_name="xtion_test_app",
            size=(900, 750),
            min_size=(700, 550),
        )

        # Preview timer
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._update_preview)
        self._preview_timer.start(33)  # ~30 FPS

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            if not Path(self.model_path).exists():
                from PyQt6.QtWidgets import QMessageBox

                QMessageBox.critical(
                    None,
                    "Model Error",
                    f"Model not found: {self.model_path}",
                )
                sys.exit(1)

            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox

            QMessageBox.critical(
                None,
                "Model Error",
                f"Failed to load model: {e}",
            )
            sys.exit(1)

    def _build_gui(self) -> None:
        """Build the GUI layout."""
        # === Model Info ===
        model_group = QGroupBox("Model")
        model_layout = QHBoxLayout(model_group)
        model_name = Path(self.model_path).name
        model_layout.addWidget(QLabel(f"Model: {model_name}"))
        model_layout.addStretch()
        self.main_layout.addWidget(model_group)

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
        preview_group = QGroupBox("Camera Preview + Detection")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("No image - Select a topic")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #2c3e50; color: #ecf0f1;"
        )
        self.preview_label.setMinimumSize(640, 360)
        preview_layout.addWidget(self.preview_label)
        self.main_layout.addWidget(preview_group, stretch=1)

        # === Confidence Threshold ===
        conf_group = QGroupBox("Confidence Threshold")
        conf_layout = QHBoxLayout(conf_group)

        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(int(self.conf_threshold * 100))
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_layout.addWidget(self.conf_slider, stretch=1)

        self.conf_label = QLabel(f"{self.conf_threshold:.2f}")
        self.conf_label.setMinimumWidth(40)
        conf_layout.addWidget(self.conf_label)
        self.main_layout.addWidget(conf_group)

        # === Detection Results ===
        detection_group = QGroupBox("Detections")
        detection_layout = QVBoxLayout(detection_group)
        self.detection_text = QTextEdit()
        self.detection_text.setReadOnly(True)
        self.detection_text.setMaximumHeight(100)
        detection_layout.addWidget(self.detection_text)
        self.main_layout.addWidget(detection_group)

        # === Status Bar ===
        status_layout = QHBoxLayout()
        self.fps_label = QLabel("FPS: --")
        status_layout.addWidget(self.fps_label)
        status_layout.addStretch()
        self.status_label = QLabel("Status: Waiting for topic")
        status_layout.addWidget(self.status_label)
        self.main_layout.addLayout(status_layout)

        # Initial topic refresh
        self.topic_selector.refresh_topics()

    @pyqtSlot(str)
    def _on_topic_changed(self, topic: str) -> None:
        """Handle topic selection change."""
        if topic:
            self.status_label.setText(f"Status: Connected to {topic}")

    @pyqtSlot(int)
    def _on_conf_changed(self, value: int) -> None:
        """Handle confidence threshold change."""
        self.conf_threshold = value / 100.0
        self.conf_label.setText(f"{self.conf_threshold:.2f}")

    @pyqtSlot()
    def _update_preview(self) -> None:
        """Update the preview with latest frame and detection results."""
        frame = self.get_frame()

        if frame is None:
            return

        start_time = time.time()

        # Run YOLO inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        result = results[0]

        # Draw detections on frame
        annotated_frame = result.plot()

        # Extract detection info
        self.latest_detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = self.model.names[class_id]
            conf = box.conf.item()
            self.latest_detections.append(
                {"class": class_name, "confidence": conf}
            )

        # Update detection text
        self._update_detection_text()

        # Calculate FPS
        elapsed = time.time() - start_time
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        avg_time = sum(self.frame_times) / len(self.frame_times)
        self.fps = 1.0 / avg_time if avg_time > 0 else 0
        self.fps_label.setText(f"FPS: {self.fps:.1f}")

        # Resize for display
        display_frame = self._resize_for_display(annotated_frame)

        # Convert to QPixmap and display
        pixmap = numpy_to_pixmap(display_frame)
        self.preview_label.setPixmap(pixmap)

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
                frame, (new_w, new_h), interpolation=cv2.INTER_AREA
            )

        return frame

    def _update_detection_text(self) -> None:
        """Update the detection results text."""
        self.detection_text.clear()

        if self.latest_detections:
            lines = []
            for det in self.latest_detections:
                lines.append(f"  - {det['class']}: {det['confidence']:.2f}")
            self.detection_text.setText("\n".join(lines))
        else:
            self.detection_text.setText("  No detections")

    def _on_close(self) -> None:
        """Handle window close event."""
        self._preview_timer.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Xtion Live Test - YOLO Inference"
    )
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)

    window = XtionTestApp(args.model, args.conf)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
