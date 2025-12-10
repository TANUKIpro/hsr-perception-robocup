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
import re
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "app"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from object_registry import ObjectRegistry, RegisteredObject
from gui_framework import ROS2App, AppTheme
from gui_framework.components import TopicSelector, StatusBar
from gui_framework.utils import draw_reticle, draw_countdown, draw_recording_indicator

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw_captures"
DEFAULT_VIDEO_DIR = Path(__file__).parent.parent.parent / "data" / "videos"


class RecordApp(ROS2App):
    """Main recording application with Tkinter GUI."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the recording application."""
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

        # Registry state
        self.registry: Optional[ObjectRegistry] = None
        self.registered_objects: list[RegisteredObject] = []

        # Initialize ROS2 app (this calls _build_gui)
        super().__init__(
            root,
            title="ROS2 Video Recording",
            node_name="record_app",
            geometry="800x750",
            min_size=(600, 700),
        )

        # Start preview update loop
        self._start_preview_loop()

    def _build_gui(self) -> None:
        """Build the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Topic Selection (using component) ===
        topic_frame = ttk.LabelFrame(main_frame, text="Topic Selection", padding="5")
        topic_frame.pack(fill=tk.X, pady=(0, 10))

        self.topic_selector = TopicSelector(
            topic_frame,
            ros_node=self.ros_node,
            on_change=self._on_topic_changed,
        )
        self.topic_selector.pack(fill=tk.X)

        # === Preview Area ===
        preview_frame = ttk.LabelFrame(main_frame, text="Camera Preview", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.preview_label = ttk.Label(
            preview_frame, text="No image - Select a topic", anchor="center"
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # === Recording Settings ===
        settings_frame = ttk.LabelFrame(
            main_frame, text="Recording Settings", padding="5"
        )
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
            width=35,
        )
        self.class_combo.pack(side=tk.LEFT, padx=(5, 5))

        refresh_registry_btn = ttk.Button(
            row1,
            text="\u21bb",
            width=3,
            command=self._refresh_registry,
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
        self.video_dir_entry = ttk.Entry(
            row3, textvariable=self.video_dir_var, width=40
        )
        self.video_dir_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        browse_video_btn = ttk.Button(
            row3, text="Browse", command=self._browse_video_dir
        )
        browse_video_btn.pack(side=tk.LEFT)

        # === Extraction Parameters ===
        extract_frame = ttk.LabelFrame(
            main_frame, text="Frame Extraction Parameters", padding="5"
        )
        extract_frame.pack(fill=tk.X, pady=(0, 10))

        params_row = ttk.Frame(extract_frame)
        params_row.pack(fill=tk.X)

        ttk.Label(params_row, text="Target Frames:").pack(side=tk.LEFT)
        self.target_frames_var = tk.IntVar(value=50)
        self.target_frames_spin = ttk.Spinbox(
            params_row,
            from_=10,
            to=500,
            textvariable=self.target_frames_var,
            width=8,
        )
        self.target_frames_spin.pack(side=tk.LEFT, padx=(5, 20))

        ttk.Label(params_row, text="(Frames extracted uniformly from video)").pack(
            side=tk.LEFT
        )

        # === Recording Controls ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))

        self.record_btn = ttk.Button(
            button_frame,
            text="Start Recording",
            command=self._toggle_recording,
            style="Large.TButton",
        )
        self.record_btn.pack(side=tk.LEFT, padx=(0, 15), ipady=5)

        self.time_var = tk.StringVar(value="00:00")
        self.time_label = ttk.Label(
            button_frame,
            textvariable=self.time_var,
            font=("TkDefaultFont", 16, "bold"),
        )
        self.time_label.pack(side=tk.LEFT, padx=(0, 15))

        # === Status Bar ===
        self.status_bar = StatusBar(main_frame)
        self.status_bar.pack(fill=tk.X)

        # Initial topic refresh
        self.topic_selector.refresh_topics()

        # Initial registry load
        self._refresh_registry()

    def _on_topic_changed(self, topic: str) -> None:
        """Handle topic selection change."""
        if topic:
            self.status_bar.set_status(f"Subscribed to {topic}")

    def _refresh_registry(self) -> None:
        """Load objects from registry and populate the class dropdown."""
        try:
            self.registry = ObjectRegistry()
            self.registered_objects = self.registry.get_all_objects()

            if not self.registered_objects:
                self.class_combo["values"] = []
                self.class_combo.set("")
                self._disable_record_button()
                self.status_bar.set_status(
                    "No objects in registry. Please add objects first."
                )
                messagebox.showwarning(
                    "No Objects",
                    "No objects are registered in the Registry.\n"
                    "Please add objects in the Registry first.",
                )
            else:
                values = [
                    f"{obj.name} ({obj.display_name})"
                    for obj in self.registered_objects
                ]
                self.class_combo["values"] = values
                if not self.class_var.get() or self.class_var.get() not in values:
                    self.class_combo.set(values[0])
                self._enable_record_button()
                self.status_bar.set_status(
                    f"Loaded {len(self.registered_objects)} object(s) from registry"
                )
        except Exception as e:
            self.class_combo["values"] = []
            self.class_combo.set("")
            self._disable_record_button()
            self.status_bar.set_status(f"Failed to load registry: {e}")

    def _disable_record_button(self) -> None:
        """Disable record button when no objects are available."""
        self.record_btn.config(state="disabled")

    def _enable_record_button(self) -> None:
        """Enable record button."""
        self.record_btn.config(state="normal")

    def _get_selected_class_name(self) -> Optional[str]:
        """Get the selected object name (for filenames)."""
        selected = self.class_var.get()
        if not selected:
            return None
        return selected.split(" (")[0]

    def _browse_output_dir(self) -> None:
        """Open directory browser for output path."""
        current = self.output_var.get()
        directory = filedialog.askdirectory(
            initialdir=current if os.path.isdir(current) else str(DEFAULT_OUTPUT_DIR),
            title="Select Image Output Directory",
        )
        if directory:
            self.output_var.set(directory)

    def _browse_video_dir(self) -> None:
        """Open directory browser for video output path."""
        current = self.video_dir_var.get()
        directory = filedialog.askdirectory(
            initialdir=current if os.path.isdir(current) else str(DEFAULT_VIDEO_DIR),
            title="Select Video Output Directory",
        )
        if directory:
            self.video_dir_var.set(directory)

    def _start_preview_loop(self) -> None:
        """Start the preview update loop."""
        self.preview_active = True
        self._update_preview()

    def _update_preview(self) -> None:
        """Update the preview image."""
        if not self.preview_active:
            return

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

            # Convert to PhotoImage
            image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image)

            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo

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
        self.root.after(33, self._update_preview)

    def _process_countdown(self) -> None:
        """Process countdown timer."""
        elapsed = time.time() - self.countdown_start_time
        new_remaining = 3 - int(elapsed)

        if new_remaining != self.countdown_remaining:
            self.countdown_remaining = new_remaining
            if self.countdown_remaining > 0:
                self.status_bar.set_status(f"Starting in {self.countdown_remaining}...")

        if elapsed >= 3.0:
            self.countdown_active = False
            self._begin_actual_recording()

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit display area dynamically."""
        preview_width = self.preview_label.winfo_width()
        preview_height = self.preview_label.winfo_height()

        if preview_width <= 1 or preview_height <= 1:
            return frame

        h, w = frame.shape[:2]
        scale = min(preview_width / w, preview_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != w or new_h != h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

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
            messagebox.showwarning(
                "No Image", "No image available. Check topic connection."
            )
            return

        class_name = self._get_selected_class_name()
        if not class_name:
            messagebox.showwarning(
                "Invalid Input", "Please select a class from the dropdown"
            )
            return

        video_dir = Path(self.video_dir_var.get())
        video_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H-%M")
        video_filename = f"{class_name}_{timestamp}.mp4"
        self.current_video_path = video_dir / video_filename
        self.current_class_name = class_name

        self.countdown_active = True
        self.countdown_remaining = 3
        self.countdown_start_time = time.time()

        self.record_btn.config(text="Cancel")
        self.status_bar.show_progress(0, 100)
        self.status_bar.set_status("Starting in 3...")

    def _begin_actual_recording(self) -> None:
        """Begin the actual recording after countdown."""
        frame = self.get_frame()
        if frame is None:
            self.status_bar.set_status("Failed to start recording: no frame")
            self.record_btn.config(text="Start Recording")
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
            self.record_btn.config(text="Start Recording")
            return

        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_count = 0

        self.record_btn.config(text="Stop Recording")
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
        self.record_btn.config(text="Start Recording")
        self.time_var.set("00:00")

        if was_countdown:
            self.status_bar.set_status("Recording cancelled")
            self.status_bar.hide_progress()
            return

        if was_recording and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

            if self.frame_count > 0:
                self.status_bar.set_status(
                    f"Recording saved: {self.current_video_path.name} ({self.frame_count} frames)"
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

        target_count = self.target_frames_var.get()
        class_name = self.current_class_name
        output_dir = Path(self.output_var.get()) / class_name
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
            indices = [int(i * total_frames / target_count) for i in range(target_count)]

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
            self.root.update_idletasks()

        cap.release()
        self.is_extracting = False

        self.status_bar.set_status(
            f"Extraction complete: {extracted} images saved (#{start_num}-{start_num + extracted - 1})"
        )
        self.status_bar.update_progress(100)

    def _on_close(self) -> None:
        """Handle window close."""
        if self.is_recording and self.video_writer is not None:
            self.video_writer.release()

        self.preview_active = False
        self._shutdown_ros2()
        self.root.destroy()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ROS2 Video Recording Application")
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

    root = tk.Tk()

    # Apply theme
    style = ttk.Style()
    AppTheme.apply(style)

    app = RecordApp(root)

    if args.output_dir:
        app.output_var.set(args.output_dir)
    if args.video_dir:
        app.video_dir_var.set(args.video_dir)

    app.run()


if __name__ == "__main__":
    main()
