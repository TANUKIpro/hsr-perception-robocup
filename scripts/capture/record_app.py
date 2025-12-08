#!/usr/bin/env python3
"""
ROS2 Video Recording Application

A Tkinter-based GUI application for recording video from ROS2 topics
and extracting frames for training data.

Features:
- Real-time camera preview with center reticle
- Topic selection from available ROS2 image topics
- Video recording with countdown timer
- Automatic frame extraction at uniform intervals
- MP4 video saving

Usage:
    python3 record_app.py

Controls:
    - Select topic from dropdown
    - Set class name and target frame count
    - Click "Start Recording" to begin (3s countdown)
    - Click "Stop Recording" to finish and extract frames
    - Press 'q' or close window to exit
"""

import argparse
import os
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

# Add app directory to path for importing ObjectRegistry
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))
from object_registry import ObjectRegistry, RegisteredObject

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw_captures"
DEFAULT_VIDEO_DIR = Path(__file__).parent.parent.parent / "data" / "videos"


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
        super().__init__('record_app')
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


class RecordApp:
    """Main recording application with Tkinter GUI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ROS2 Video Recording")
        self.root.geometry("800x750")
        self.root.resizable(True, True)

        # ROS2 setup
        self.ros_node: Optional[ROS2ImageSubscriber] = None
        self.executor: Optional[SingleThreadedExecutor] = None
        self.ros_thread: Optional[threading.Thread] = None
        self.ros_running = False

        # Recording state
        self.is_recording = False
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.recording_start_time = 0.0
        self.frame_count = 0
        self.current_video_path: Optional[Path] = None

        # Countdown state
        self.countdown_active = False
        self.countdown_remaining = 0
        self.countdown_start_time = 0.0

        # Extraction state
        self.is_extracting = False

        # Preview state
        self.preview_active = False
        self.preview_label: Optional[tk.Label] = None

        # Registry state
        self.registry: Optional[ObjectRegistry] = None
        self.registered_objects: List[RegisteredObject] = []

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

        # === Recording Settings ===
        settings_frame = ttk.LabelFrame(main_frame, text="Recording Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))

        # Class name (dropdown from registry)
        row1 = ttk.Frame(settings_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="Class Name:", width=15).pack(side=tk.LEFT)
        self.class_var = tk.StringVar()
        self.class_combo = ttk.Combobox(
            row1,
            textvariable=self.class_var,
            state="readonly",
            width=35
        )
        self.class_combo.pack(side=tk.LEFT, padx=(5, 5))

        # Refresh button for registry
        refresh_registry_btn = ttk.Button(
            row1,
            text="↻",
            width=3,
            command=self._refresh_registry
        )
        refresh_registry_btn.pack(side=tk.LEFT)

        # Output directory (for images)
        row2 = ttk.Frame(settings_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="Image Output:", width=15).pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.output_entry = ttk.Entry(row2, textvariable=self.output_var, width=40)
        self.output_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        browse_btn = ttk.Button(row2, text="Browse", command=self._browse_output_dir)
        browse_btn.pack(side=tk.LEFT)

        # Video directory
        row3 = ttk.Frame(settings_frame)
        row3.pack(fill=tk.X, pady=2)
        ttk.Label(row3, text="Video Output:", width=15).pack(side=tk.LEFT)
        self.video_dir_var = tk.StringVar(value=str(DEFAULT_VIDEO_DIR))
        self.video_dir_entry = ttk.Entry(row3, textvariable=self.video_dir_var, width=40)
        self.video_dir_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        browse_video_btn = ttk.Button(row3, text="Browse", command=self._browse_video_dir)
        browse_video_btn.pack(side=tk.LEFT)

        # === Extraction Parameters ===
        extract_frame = ttk.LabelFrame(main_frame, text="Frame Extraction Parameters", padding="5")
        extract_frame.pack(fill=tk.X, pady=(0, 10))

        params_row = ttk.Frame(extract_frame)
        params_row.pack(fill=tk.X)

        # Target frame count
        ttk.Label(params_row, text="Target Frames:").pack(side=tk.LEFT)
        self.target_frames_var = tk.IntVar(value=50)
        self.target_frames_spin = ttk.Spinbox(
            params_row,
            from_=10,
            to=500,
            textvariable=self.target_frames_var,
            width=8
        )
        self.target_frames_spin.pack(side=tk.LEFT, padx=(5, 20))

        # Info label
        ttk.Label(params_row, text="(Frames extracted uniformly from video)").pack(side=tk.LEFT)

        # === Recording Controls ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        # Custom style for large buttons
        style = ttk.Style()
        style.configure(
            "Large.TButton",
            font=("TkDefaultFont", 12, "bold"),
            padding=(20, 10)
        )
        style.configure(
            "Recording.TButton",
            font=("TkDefaultFont", 12, "bold"),
            padding=(20, 10)
        )

        self.record_btn = ttk.Button(
            button_frame,
            text="Start Recording",
            command=self._toggle_recording,
            style="Large.TButton"
        )
        self.record_btn.pack(side=tk.LEFT, padx=(0, 15), ipady=5)

        # Recording time display
        self.time_var = tk.StringVar(value="00:00")
        self.time_label = ttk.Label(
            button_frame,
            textvariable=self.time_var,
            font=("TkDefaultFont", 16, "bold")
        )
        self.time_label.pack(side=tk.LEFT, padx=(0, 15))

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

        # Progress bar for extraction
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        # Initial topic refresh
        self._refresh_topics()

        # Initial registry load
        self._refresh_registry()

    def _refresh_registry(self):
        """Load objects from registry and populate the class dropdown."""
        try:
            self.registry = ObjectRegistry()
            self.registered_objects = self.registry.get_all_objects()

            if not self.registered_objects:
                # No objects registered
                self.class_combo['values'] = []
                self.class_combo.set("")
                self._disable_record_button()
                self.status_var.set("No objects in registry. Please add objects first.")
                messagebox.showwarning(
                    "No Objects",
                    "No objects are registered in the Registry.\n"
                    "Please add objects in the Registry first."
                )
            else:
                # Populate dropdown with "name (display_name)" format
                values = [f"{obj.name} ({obj.display_name})" for obj in self.registered_objects]
                self.class_combo['values'] = values
                if not self.class_var.get() or self.class_var.get() not in values:
                    self.class_combo.set(values[0])
                self._enable_record_button()
                self.status_var.set(f"Loaded {len(self.registered_objects)} object(s) from registry")
        except Exception as e:
            self.class_combo['values'] = []
            self.class_combo.set("")
            self._disable_record_button()
            self.status_var.set(f"Failed to load registry: {e}")

    def _disable_record_button(self):
        """Disable record button when no objects are available."""
        self.record_btn.config(state="disabled")

    def _enable_record_button(self):
        """Enable record button."""
        self.record_btn.config(state="normal")

    def _get_selected_class_name(self) -> Optional[str]:
        """Get the selected object name (for filenames)."""
        selected = self.class_var.get()
        if not selected:
            return None
        # Extract name from "name (display_name)" format
        return selected.split(" (")[0]

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
            title="Select Image Output Directory"
        )
        if directory:
            self.output_var.set(directory)

    def _browse_video_dir(self):
        """Open directory browser for video output path."""
        current = self.video_dir_var.get()
        directory = filedialog.askdirectory(
            initialdir=current if os.path.isdir(current) else str(DEFAULT_VIDEO_DIR),
            title="Select Video Output Directory"
        )
        if directory:
            self.video_dir_var.set(directory)

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
                # Write frame to video if recording
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(frame)
                    self.frame_count += 1

                # Draw reticle
                display = self._draw_reticle(frame.copy())

                # Add countdown overlay if counting down
                if self.countdown_active:
                    display = self._draw_countdown(display)

                # Add recording status overlay if recording
                elif self.is_recording:
                    display = self._draw_recording_status(display)

                # Resize for display
                display = self._resize_for_display(display)

                # Convert to PhotoImage
                image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                photo = ImageTk.PhotoImage(image)

                self.preview_label.config(image=photo, text="")
                self.preview_label.image = photo  # Keep reference

        # Handle countdown
        if self.countdown_active:
            self._process_countdown()

        # Update recording time
        if self.is_recording:
            elapsed = time.time() - self.recording_start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            self.time_var.set(f"{mins:02d}:{secs:02d}")

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

    def _draw_recording_status(self, frame: np.ndarray) -> np.ndarray:
        """Draw recording status overlay."""
        h, w = frame.shape[:2]

        # Red recording indicator (blinking)
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)

        # Recording text
        elapsed = time.time() - self.recording_start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        text = f"REC {mins:02d}:{secs:02d} | {self.frame_count} frames"
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
            self._begin_actual_recording()

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit display area dynamically."""
        # Get preview area size directly (same as SAM2 annotation app)
        preview_width = self.preview_label.winfo_width()
        preview_height = self.preview_label.winfo_height()

        # Early return if size is too small
        if preview_width <= 1 or preview_height <= 1:
            return frame

        h, w = frame.shape[:2]
        scale_w = preview_width / w
        scale_h = preview_height / h
        scale = min(scale_w, scale_h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != w or new_h != h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return frame

    def _toggle_recording(self):
        """Start or stop recording."""
        if self.is_recording or self.countdown_active:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Start recording with countdown."""
        if not self.ros_node or self.ros_node.get_frame() is None:
            messagebox.showwarning("No Image", "No image available. Check topic connection.")
            return

        # Get parameters
        class_name = self._get_selected_class_name()
        if not class_name:
            messagebox.showwarning("Invalid Input", "Please select a class from the dropdown")
            return

        # Prepare video output directory
        video_dir = Path(self.video_dir_var.get())
        video_dir.mkdir(parents=True, exist_ok=True)

        # Generate video filename: classname_yyyymmdd-hh-mm.mp4
        timestamp = datetime.now().strftime("%Y%m%d-%H-%M")
        video_filename = f"{class_name}_{timestamp}.mp4"
        self.current_video_path = video_dir / video_filename
        self.current_class_name = class_name

        # Start countdown
        self.countdown_active = True
        self.countdown_remaining = 3
        self.countdown_start_time = time.time()

        # Update UI
        self.record_btn.config(text="Cancel")
        self.progress_var.set(0)
        self.status_var.set("Starting in 3...")

    def _begin_actual_recording(self):
        """Begin the actual recording after countdown."""
        # Get frame dimensions
        frame = self.ros_node.get_frame()
        if frame is None:
            self.status_var.set("Failed to start recording: no frame")
            self.record_btn.config(text="Start Recording")
            return

        h, w = frame.shape[:2]

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(self.current_video_path),
            fourcc,
            30.0,  # fps
            (w, h)
        )

        if not self.video_writer.isOpened():
            self.status_var.set("Failed to create video file")
            self.record_btn.config(text="Start Recording")
            return

        # Start recording
        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_count = 0

        # Update UI
        self.record_btn.config(text="Stop Recording")
        self.status_var.set("Recording...")

    def _get_next_file_number(self, output_dir: Path, class_name: str) -> int:
        """Get the next available file number for accumulation."""
        import re
        pattern = re.compile(rf"^{re.escape(class_name)}_(\d+)\.jpg$")
        max_num = 0

        if output_dir.exists():
            for f in output_dir.iterdir():
                match = pattern.match(f.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)

        return max_num + 1

    def _stop_recording(self):
        """Stop recording and extract frames."""
        was_countdown = self.countdown_active
        was_recording = self.is_recording

        self.countdown_active = False
        self.is_recording = False
        self.record_btn.config(text="Start Recording")
        self.time_var.set("00:00")

        if was_countdown:
            self.status_var.set("Recording cancelled")
            return

        if was_recording and self.video_writer is not None:
            # Close video writer
            self.video_writer.release()
            self.video_writer = None

            if self.frame_count > 0:
                self.status_var.set(f"Recording saved: {self.current_video_path.name} ({self.frame_count} frames)")

                # Extract frames
                self._extract_frames()
            else:
                self.status_var.set("No frames recorded")
                # Delete empty video file
                if self.current_video_path.exists():
                    self.current_video_path.unlink()

    def _extract_frames(self):
        """Extract frames from recorded video."""
        if not self.current_video_path or not self.current_video_path.exists():
            return

        self.is_extracting = True
        self.status_var.set("Extracting frames...")
        self.progress_var.set(0)

        # Get parameters
        target_count = self.target_frames_var.get()
        class_name = self.current_class_name
        output_dir = Path(self.output_var.get()) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get starting file number for accumulation
        start_num = self._get_next_file_number(output_dir, class_name)

        # Open video
        cap = cv2.VideoCapture(str(self.current_video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            self.status_var.set("Error: Video has no frames")
            self.is_extracting = False
            return

        # Calculate frame indices (uniform distribution)
        if total_frames <= target_count:
            # If video has fewer frames than target, use all frames
            indices = list(range(total_frames))
        else:
            # Uniform sampling
            indices = [int(i * total_frames / target_count) for i in range(target_count)]

        # Extract frames (accumulate from start_num)
        extracted = 0
        for i, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                filename = f"{class_name}_{start_num + i}.jpg"
                cv2.imwrite(str(output_dir / filename), frame)
                extracted += 1

            # Update progress
            progress = ((i + 1) / len(indices)) * 100
            self.progress_var.set(progress)
            self.root.update_idletasks()

        cap.release()
        self.is_extracting = False

        self.status_var.set(
            f"Extraction complete: {extracted} images saved to {output_dir} (#{start_num}-{start_num + extracted - 1})"
        )
        self.progress_var.set(100)

    def _on_close(self):
        """Handle window close."""
        # Stop recording if active
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()

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
    parser = argparse.ArgumentParser(description="ROS2 Video Recording Application")
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Default output directory for extracted images"
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default=str(DEFAULT_VIDEO_DIR),
        help="Default output directory for video files"
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
    app = RecordApp(root)

    # Set initial directories from args
    if args.output_dir:
        app.output_var.set(args.output_dir)
    if args.video_dir:
        app.video_dir_var.set(args.video_dir)

    # Run main loop
    root.mainloop()


if __name__ == '__main__':
    main()
