#!/usr/bin/env python3
"""
Xtion Live Test Application

A Tkinter-based GUI application for real-time YOLO inference on ROS2 camera topics.
Features:
- Real-time camera preview with detection overlay
- Topic selection from available ROS2 image topics
- Configurable confidence threshold
- FPS and detection statistics display

Usage:
    python3 xtion_test_app.py --model path/to/model.pt --conf 0.25
"""

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# YOLO import
from ultralytics import YOLO

# Add gui_framework to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gui_framework import ROS2App, AppTheme
from gui_framework.components import TopicSelector, StatusBar


class XtionTestApp(ROS2App):
    """Main Xtion test application with Tkinter GUI."""

    def __init__(
        self,
        root: tk.Tk,
        model_path: str,
        conf_threshold: float = 0.25,
    ) -> None:
        """
        Initialize the Xtion test application.

        Args:
            root: Tkinter root window
            model_path: Path to YOLO model file
            conf_threshold: Initial confidence threshold
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # YOLO model (loaded later)
        self.model: Optional[YOLO] = None

        # Preview state
        self.preview_active = False

        # FPS calculation
        self.frame_times: list[float] = []
        self.fps = 0.0

        # Detection results
        self.latest_detections: list[dict] = []

        # Load YOLO model before building GUI
        self._load_model()

        # Initialize ROS2 app (this calls _build_gui)
        super().__init__(
            root,
            title="Xtion Live Test - YOLO Inference",
            node_name="xtion_test_app",
            geometry="900x750",
            min_size=(700, 550),
        )

        # Start preview update loop
        self._start_preview_loop()

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            if not Path(self.model_path).exists():
                messagebox.showerror(
                    "Model Error", f"Model not found: {self.model_path}"
                )
                sys.exit(1)

            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")

        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            sys.exit(1)

    def _build_gui(self) -> None:
        """Build the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Model Info ===
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="5")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        model_name = Path(self.model_path).name
        ttk.Label(model_frame, text=f"Model: {model_name}").pack(side=tk.LEFT)

        # === Topic Selection (using component) ===
        topic_frame = ttk.LabelFrame(
            main_frame, text="Topic Selection", padding="5"
        )
        topic_frame.pack(fill=tk.X, pady=(0, 10))

        self.topic_selector = TopicSelector(
            topic_frame,
            ros_node=self.ros_node,
            on_change=self._on_topic_changed,
        )
        self.topic_selector.pack(fill=tk.X)

        # === Preview Area ===
        preview_frame = ttk.LabelFrame(
            main_frame, text="Camera Preview + Detection", padding="5"
        )
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.preview_label = ttk.Label(
            preview_frame, text="No image - Select a topic", anchor="center"
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # === Confidence Threshold ===
        conf_frame = ttk.LabelFrame(
            main_frame, text="Confidence Threshold", padding="5"
        )
        conf_frame.pack(fill=tk.X, pady=(0, 10))

        self.conf_var = tk.DoubleVar(value=self.conf_threshold)
        self.conf_scale = ttk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            variable=self.conf_var,
            orient=tk.HORIZONTAL,
            command=self._on_conf_changed,
        )
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.conf_label = ttk.Label(conf_frame, text=f"{self.conf_threshold:.2f}")
        self.conf_label.pack(side=tk.LEFT)

        # === Detection Results ===
        detection_frame = ttk.LabelFrame(
            main_frame, text="Detections", padding="5"
        )
        detection_frame.pack(fill=tk.X, pady=(0, 10))

        self.detection_text = tk.Text(detection_frame, height=5, state=tk.DISABLED)
        self.detection_text.pack(fill=tk.X)

        # === Status Bar ===
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)

        self.fps_label = ttk.Label(status_frame, text="FPS: --")
        self.fps_label.pack(side=tk.LEFT)

        self.status_label = ttk.Label(status_frame, text="Status: Waiting for topic")
        self.status_label.pack(side=tk.RIGHT)

        # Initial topic refresh
        self.topic_selector.refresh_topics()

    def _on_topic_changed(self, topic: str) -> None:
        """Handle topic selection change."""
        if topic:
            self.status_label.config(text=f"Status: Connected to {topic}")
            self.preview_active = True

    def _on_conf_changed(self, value: str) -> None:
        """Handle confidence threshold change."""
        self.conf_threshold = float(value)
        self.conf_label.config(text=f"{self.conf_threshold:.2f}")

    def _start_preview_loop(self) -> None:
        """Start the preview update loop."""
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the preview with latest frame and detection results."""
        if self.preview_active:
            frame = self.get_frame()

            if frame is not None:
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
                self.fps_label.config(text=f"FPS: {self.fps:.1f}")

                # Resize for display
                display_frame = self._resize_for_display(annotated_frame)

                # Convert to PhotoImage
                img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                photo = ImageTk.PhotoImage(img)

                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo  # Keep reference

        # Schedule next update (~30 FPS)
        self.root.after(33, self._update_preview)

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit display area dynamically."""
        # Get preview label size
        label_width = self.preview_label.winfo_width()
        label_height = self.preview_label.winfo_height()

        # Skip if size is too small
        if label_width <= 1 or label_height <= 1:
            return frame

        h, w = frame.shape[:2]
        scale = min(label_width / w, label_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != w or new_h != h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def _update_detection_text(self) -> None:
        """Update the detection results text."""
        self.detection_text.config(state=tk.NORMAL)
        self.detection_text.delete(1.0, tk.END)

        if self.latest_detections:
            for det in self.latest_detections:
                line = f"  - {det['class']}: {det['confidence']:.2f}\n"
                self.detection_text.insert(tk.END, line)
        else:
            self.detection_text.insert(tk.END, "  No detections\n")

        self.detection_text.config(state=tk.DISABLED)

    def _on_close(self) -> None:
        """Handle window close event."""
        self.preview_active = False
        self._shutdown_ros2()
        self.root.destroy()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Xtion Live Test - YOLO Inference")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    root = tk.Tk()

    # Apply theme
    style = ttk.Style()
    AppTheme.apply(style)

    app = XtionTestApp(root, args.model, args.conf)
    app.run()


if __name__ == "__main__":
    main()
