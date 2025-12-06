#!/usr/bin/env python3
"""
ROS2 Image Capture Application

A Tkinter-based GUI application for capturing images from ROS2 topics.
Features:
- Real-time camera preview with center reticle
- Topic selection from available ROS2 image topics
- Configurable burst capture parameters
- Single image capture

Usage:
    python3 capture_app.py

Controls:
    - Select topic from dropdown
    - Set class name and output directory
    - Adjust burst parameters (count, interval)
    - Click "Single Capture" or "Start Burst" to capture
    - Press 'q' or close window to exit
"""

import argparse
import os
import re
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image as RosImage

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw_captures"


def imgmsg_to_cv2(msg: RosImage) -> np.ndarray:
    """Convert sensor_msgs/Image to OpenCV image without cv_bridge."""
    if msg.encoding in ['16UC1']:
        dtype = np.uint16
    elif msg.encoding in ['32FC1']:
        dtype = np.float32
    else:
        dtype = np.uint8

    if msg.encoding in ['rgb8', 'bgr8']:
        channels = 3
    elif msg.encoding in ['rgba8', 'bgra8']:
        channels = 4
    elif msg.encoding in ['mono8', 'mono16', '16UC1', '32FC1']:
        channels = 1
    else:
        channels = 3

    if channels == 1:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    else:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)

    if msg.encoding == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == 'rgba8':
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif msg.encoding == 'bgra8':
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif msg.encoding == 'mono8':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif msg.encoding == 'mono16':
        img = (img / 256).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif msg.encoding in ['16UC1', '32FC1']:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


class ROS2ImageSubscriber(Node):
    """ROS2 node for subscribing to image topics."""

    def __init__(self):
        super().__init__('capture_app')
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.subscription = None
        self.current_topic: Optional[str] = None

    def subscribe_to_topic(self, topic: str):
        """Subscribe to a new image topic."""
        if self.subscription is not None:
            self.destroy_subscription(self.subscription)
            self.subscription = None

        self.current_topic = topic
        self.latest_frame = None

        if topic:
            self.subscription = self.create_subscription(
                RosImage,
                topic,
                self._image_callback,
                10
            )
            self.get_logger().info(f"Subscribed to {topic}")

    def _image_callback(self, msg: RosImage):
        """Handle incoming image message."""
        try:
            frame = imgmsg_to_cv2(msg)
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (thread-safe)."""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def get_image_topics(self) -> List[str]:
        """Get list of available image topics."""
        topics = self.get_topic_names_and_types()
        image_topics = []
        for name, types in topics:
            for t in types:
                if 'sensor_msgs/msg/Image' in t or 'sensor_msgs/Image' in t:
                    image_topics.append(name)
                    break
        return sorted(image_topics)


class CaptureApp:
    """Main capture application with Tkinter GUI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ROS2 Image Capture")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # ROS2 setup
        self.ros_node: Optional[ROS2ImageSubscriber] = None
        self.executor: Optional[SingleThreadedExecutor] = None
        self.ros_thread: Optional[threading.Thread] = None
        self.ros_running = False

        # Capture state
        self.is_capturing_burst = False
        self.burst_count = 0
        self.burst_target = 0
        self.last_capture_time = 0.0

        # Countdown state
        self.countdown_active = False
        self.countdown_remaining = 0
        self.countdown_start_time = 0.0

        # Preview state
        self.preview_active = False
        self.preview_label: Optional[tk.Label] = None

        # Initialize ROS2
        self._init_ros2()

        # Build GUI
        self._build_gui()

        # Start preview update loop
        self._start_preview_loop()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_ros2(self):
        """Initialize ROS2 node and executor."""
        try:
            rclpy.init()
            self.ros_node = ROS2ImageSubscriber()
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.ros_node)

            # Start ROS2 spinning in background thread
            self.ros_running = True
            self.ros_thread = threading.Thread(target=self._ros_spin_thread, daemon=True)
            self.ros_thread.start()

        except Exception as e:
            messagebox.showerror("ROS2 Error", f"Failed to initialize ROS2: {e}")
            sys.exit(1)

    def _ros_spin_thread(self):
        """Background thread for ROS2 spinning."""
        while self.ros_running and rclpy.ok():
            self.executor.spin_once(timeout_sec=0.01)

    def _build_gui(self):
        """Build the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Topic Selection ===
        topic_frame = ttk.LabelFrame(main_frame, text="Topic Selection", padding="5")
        topic_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(topic_frame, text="Image Topic:").pack(side=tk.LEFT)

        self.topic_var = tk.StringVar()
        self.topic_combo = ttk.Combobox(
            topic_frame,
            textvariable=self.topic_var,
            state="readonly",
            width=50
        )
        self.topic_combo.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        self.topic_combo.bind("<<ComboboxSelected>>", self._on_topic_changed)

        refresh_btn = ttk.Button(topic_frame, text="Refresh", command=self._refresh_topics)
        refresh_btn.pack(side=tk.LEFT)

        # === Preview Area ===
        preview_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.preview_label = ttk.Label(preview_frame, text="No image - Select a topic", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # === Capture Settings ===
        settings_frame = ttk.LabelFrame(main_frame, text="Capture Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Class name
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Class Name:", width=15).pack(side=tk.LEFT)
        self.class_var = tk.StringVar(value="object")
        self.class_entry = ttk.Entry(row1, textvariable=self.class_var, width=30)
        self.class_entry.pack(side=tk.LEFT, padx=(5, 0))

        # Output directory
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Output Directory:", width=15).pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.output_entry = ttk.Entry(row2, textvariable=self.output_var, width=40)
        self.output_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        browse_btn = ttk.Button(row2, text="Browse", command=self._browse_output_dir)
        browse_btn.pack(side=tk.LEFT)

        # === Burst Parameters ===
        burst_frame = ttk.LabelFrame(main_frame, text="Burst Capture Parameters", padding="5")
        burst_frame.pack(fill=tk.X, pady=(0, 10))

        params_row = ttk.Frame(burst_frame)
        params_row.pack(fill=tk.X)

        # Number of images
        ttk.Label(params_row, text="Images:").pack(side=tk.LEFT)
        self.count_var = tk.IntVar(value=50)
        self.count_spin = ttk.Spinbox(
            params_row,
            from_=1,
            to=500,
            textvariable=self.count_var,
            width=8
        )
        self.count_spin.pack(side=tk.LEFT, padx=(5, 20))

        # Interval
        ttk.Label(params_row, text="Interval (s):").pack(side=tk.LEFT)
        self.interval_var = tk.DoubleVar(value=0.2)
        self.interval_spin = ttk.Spinbox(
            params_row,
            from_=0.05,
            to=5.0,
            increment=0.05,
            textvariable=self.interval_var,
            width=8
        )
        self.interval_spin.pack(side=tk.LEFT, padx=(5, 20))

        # Estimated time
        ttk.Label(params_row, text="Est. Time:").pack(side=tk.LEFT)
        self.est_time_label = ttk.Label(params_row, text="10.0s")
        self.est_time_label.pack(side=tk.LEFT, padx=(5, 20))

        # Overwrite checkbox
        self.overwrite_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            params_row,
            text="Overwrite",
            variable=self.overwrite_var
        ).pack(side=tk.LEFT)

        # Update estimated time when parameters change
        self.count_var.trace_add("write", self._update_estimated_time)
        self.interval_var.trace_add("write", self._update_estimated_time)

        # === Capture Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        # Custom style for large buttons
        style = ttk.Style()
        style.configure(
            "Large.TButton",
            font=("TkDefaultFont", 12, "bold"),
            padding=(20, 10)
        )

        self.single_btn = ttk.Button(
            button_frame,
            text="Single Capture",
            command=self._single_capture,
            style="Large.TButton"
        )
        self.single_btn.pack(side=tk.LEFT, padx=(0, 15), ipady=5)

        self.burst_btn = ttk.Button(
            button_frame,
            text="Start Burst Capture",
            command=self._toggle_burst_capture,
            style="Large.TButton"
        )
        self.burst_btn.pack(side=tk.LEFT, padx=(0, 10), ipady=5)

        # === Status Bar ===
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            relief="sunken",
            padding=(5, 2)
        )
        self.status_label.pack(fill=tk.X)

        # Progress bar for burst capture
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # Initial topic refresh
        self._refresh_topics()

    def _refresh_topics(self):
        """Refresh the list of available image topics."""
        if self.ros_node:
            topics = self.ros_node.get_image_topics()
            self.topic_combo['values'] = topics

            if topics:
                if self.topic_var.get() not in topics:
                    self.topic_var.set(topics[0])
                    self._on_topic_changed(None)
                self.status_var.set(f"Found {len(topics)} image topic(s)")
            else:
                self.topic_var.set("")
                self.status_var.set("No image topics found")

    def _on_topic_changed(self, event):
        """Handle topic selection change."""
        topic = self.topic_var.get()
        if topic and self.ros_node:
            self.ros_node.subscribe_to_topic(topic)
            self.status_var.set(f"Subscribed to {topic}")

    def _browse_output_dir(self):
        """Open directory browser for output path."""
        current = self.output_var.get()
        directory = filedialog.askdirectory(
            initialdir=current if os.path.isdir(current) else str(DEFAULT_OUTPUT_DIR),
            title="Select Output Directory"
        )
        if directory:
            self.output_var.set(directory)

    def _update_estimated_time(self, *args):
        """Update the estimated time label."""
        try:
            count = self.count_var.get()
            interval = self.interval_var.get()
            est_time = count * interval
            self.est_time_label.config(text=f"{est_time:.1f}s")
        except (tk.TclError, ValueError):
            pass

    def _start_preview_loop(self):
        """Start the preview update loop."""
        self.preview_active = True
        self._update_preview()

    def _update_preview(self):
        """Update the preview image."""
        if not self.preview_active:
            return

        if self.ros_node:
            frame = self.ros_node.get_frame()

            if frame is not None:
                # Draw reticle
                display = self._draw_reticle(frame.copy())

                # Add countdown overlay if counting down
                if self.countdown_active:
                    display = self._draw_countdown(display)

                # Add capture status overlay if capturing
                elif self.is_capturing_burst:
                    display = self._draw_capture_status(display)

                # Resize for display
                display = self._resize_for_display(display, max_width=760, max_height=400)

                # Convert to PhotoImage
                image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                photo = ImageTk.PhotoImage(image)

                self.preview_label.config(image=photo, text="")
                self.preview_label.image = photo  # Keep reference

        # Handle countdown
        if self.countdown_active:
            self._process_countdown()

        # Handle burst capture
        if self.is_capturing_burst:
            self._process_burst_capture()

        # Schedule next update
        self.root.after(33, self._update_preview)  # ~30 FPS

    def _draw_reticle(self, frame: np.ndarray) -> np.ndarray:
        """Draw center crosshair reticle on frame."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        size = min(w, h) // 20
        color = (0, 255, 0)  # Green
        thickness = 1

        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)

        return frame

    def _draw_capture_status(self, frame: np.ndarray) -> np.ndarray:
        """Draw capture status overlay."""
        h, w = frame.shape[:2]

        # Red recording indicator
        cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

        # Progress text
        text = f"Capturing: {self.burst_count}/{self.burst_target}"
        cv2.putText(
            frame, text, (50, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        return frame

    def _draw_countdown(self, frame: np.ndarray) -> np.ndarray:
        """Draw countdown overlay on frame."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        # Large countdown number in center
        text = str(self.countdown_remaining)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 8.0
        thickness = 15

        # Get text size for centering
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = cx - text_w // 2
        text_y = cy + text_h // 2

        # Draw text with outline
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 5)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 255, 255), thickness)

        # "Get Ready!" text
        ready_text = "Get Ready!"
        ready_scale = 1.5
        (ready_w, ready_h), _ = cv2.getTextSize(ready_text, font, ready_scale, 3)
        ready_x = cx - ready_w // 2
        ready_y = cy - text_h // 2 - 40

        cv2.putText(frame, ready_text, (ready_x, ready_y), font, ready_scale, (0, 0, 0), 5)
        cv2.putText(frame, ready_text, (ready_x, ready_y), font, ready_scale, (255, 255, 255), 3)

        return frame

    def _process_countdown(self):
        """Process countdown timer."""
        elapsed = time.time() - self.countdown_start_time
        new_remaining = 3 - int(elapsed)

        if new_remaining != self.countdown_remaining:
            self.countdown_remaining = new_remaining
            if self.countdown_remaining > 0:
                self.status_var.set(f"Starting in {self.countdown_remaining}...")

        if elapsed >= 3.0:
            self.countdown_active = False
            self._begin_actual_capture()

    def _resize_for_display(
        self,
        frame: np.ndarray,
        max_width: int,
        max_height: int
    ) -> np.ndarray:
        """Resize frame to fit display area."""
        h, w = frame.shape[:2]

        scale_w = max_width / w
        scale_h = max_height / h
        scale = min(scale_w, scale_h, 1.0)

        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        return frame

    def _get_next_file_number(self, output_dir: Path, class_name: str) -> int:
        """
        Get the next file number for the given class.

        If overwrite is enabled, returns 1 (start fresh).
        If overwrite is disabled, finds the highest existing number and returns next.
        """
        if self.overwrite_var.get():
            # Overwrite mode: start from 1
            return 1

        # Find existing files and get max number
        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$")
        max_num = 0

        if output_dir.exists():
            for f in output_dir.iterdir():
                match = pattern.match(f.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)

        return max_num + 1

    def _single_capture(self):
        """Capture a single image."""
        if not self.ros_node:
            return

        frame = self.ros_node.get_frame()
        if frame is None:
            messagebox.showwarning("No Image", "No image available to capture")
            return

        # Get parameters
        class_name = self.class_var.get().strip()
        if not class_name:
            messagebox.showwarning("Invalid Input", "Please enter a class name")
            return

        output_dir = Path(self.output_var.get()) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with number
        file_num = self._get_next_file_number(output_dir, class_name)
        filename = f"{class_name}_{file_num}.jpg"
        filepath = output_dir / filename

        # Save image
        cv2.imwrite(str(filepath), frame)
        self.status_var.set(f"Saved: {filename}")

    def _toggle_burst_capture(self):
        """Start or stop burst capture."""
        if self.is_capturing_burst or self.countdown_active:
            self._stop_burst_capture()
        else:
            self._start_burst_capture()

    def _start_burst_capture(self):
        """Start burst capture with countdown."""
        if not self.ros_node or self.ros_node.get_frame() is None:
            messagebox.showwarning("No Image", "No image available. Check topic connection.")
            return

        # Get parameters
        class_name = self.class_var.get().strip()
        if not class_name:
            messagebox.showwarning("Invalid Input", "Please enter a class name")
            return

        try:
            count = self.count_var.get()
            interval = self.interval_var.get()
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid Input", "Invalid count or interval value")
            return

        # Prepare output directory
        output_dir = Path(self.output_var.get()) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.current_output_dir = output_dir
        self.current_class_name = class_name

        # Store parameters for after countdown
        self.pending_burst_count = count
        self.pending_burst_interval = interval

        # Start countdown
        self.countdown_active = True
        self.countdown_remaining = 3
        self.countdown_start_time = time.time()

        # Update UI
        self.burst_btn.config(text="Cancel")
        self.single_btn.config(state="disabled")
        self.progress_var.set(0)
        self.status_var.set("Starting in 3...")

    def _begin_actual_capture(self):
        """Begin the actual burst capture after countdown."""
        # Get starting file number
        self.burst_start_num = self._get_next_file_number(
            self.current_output_dir,
            self.current_class_name
        )

        # Start capture
        self.is_capturing_burst = True
        self.burst_count = 0
        self.burst_target = self.pending_burst_count
        self.burst_interval = self.pending_burst_interval
        self.last_capture_time = 0.0

        # Update UI
        self.burst_btn.config(text="Stop Capture")
        self.status_var.set(f"Capturing: 0/{self.burst_target}")

    def _process_burst_capture(self):
        """Process burst capture (called from preview loop)."""
        if not self.is_capturing_burst:
            return

        current_time = time.time()
        if current_time - self.last_capture_time >= self.burst_interval:
            frame = self.ros_node.get_frame()
            if frame is not None:
                # Generate filename with sequential number
                file_num = self.burst_start_num + self.burst_count
                filename = f"{self.current_class_name}_{file_num}.jpg"
                filepath = self.current_output_dir / filename

                # Save image
                cv2.imwrite(str(filepath), frame)
                self.burst_count += 1
                self.last_capture_time = current_time

                # Update progress
                progress = (self.burst_count / self.burst_target) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Capturing: {self.burst_count}/{self.burst_target}")

                # Check if done
                if self.burst_count >= self.burst_target:
                    self._stop_burst_capture()
                    self.status_var.set(
                        f"Burst complete: {self.burst_count} images saved to {self.current_output_dir}"
                    )

    def _stop_burst_capture(self):
        """Stop burst capture or cancel countdown."""
        was_countdown = self.countdown_active
        was_capturing = self.is_capturing_burst

        self.countdown_active = False
        self.is_capturing_burst = False
        self.burst_btn.config(text="Start Burst Capture")
        self.single_btn.config(state="normal")

        if was_countdown:
            self.status_var.set("Capture cancelled")
        elif was_capturing and self.burst_count < self.burst_target:
            self.status_var.set(f"Burst stopped: {self.burst_count}/{self.burst_target} images captured")

    def _on_close(self):
        """Handle window close."""
        self.preview_active = False
        self.ros_running = False

        if self.ros_node:
            self.ros_node.destroy_node()

        if self.executor:
            self.executor.shutdown()

        try:
            rclpy.shutdown()
        except Exception:
            pass

        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="ROS2 Image Capture Application")
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Default output directory for captured images"
    )
    args = parser.parse_args()

    # Create main window
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    available_themes = style.theme_names()
    if 'clam' in available_themes:
        style.theme_use('clam')

    # Create app
    app = CaptureApp(root)

    # Set initial output directory from args
    if args.output_dir:
        app.output_var.set(args.output_dir)

    # Run main loop
    root.mainloop()


if __name__ == '__main__':
    main()
