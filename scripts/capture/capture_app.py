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
import time
import tkinter as tk
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
from gui_framework.utils import draw_reticle, draw_countdown

# Default paths
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "raw_captures"


class CaptureApp(ROS2App):
    """Main capture application with Tkinter GUI."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the capture application."""
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

        # Registry state
        self.registry: Optional[ObjectRegistry] = None
        self.registered_objects: list[RegisteredObject] = []

        # Initialize ROS2 app (this calls _build_gui)
        super().__init__(
            root,
            title="ROS2 Image Capture",
            node_name="capture_app",
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

        self.preview_label = ttk.Label(
            preview_frame, text="No image - Select a topic", anchor="center"
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # === Capture Settings ===
        settings_frame = ttk.LabelFrame(main_frame, text="Capture Settings", padding="5")

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

        # Refresh button for registry
        refresh_registry_btn = ttk.Button(
            row1,
            text="\u21bb",
            width=3,
            command=self._refresh_registry,
        )
        refresh_registry_btn.pack(side=tk.LEFT)

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
        burst_frame = ttk.LabelFrame(
            main_frame, text="Burst Capture Parameters", padding="5"
        )

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
            width=8,
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
            width=8,
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
            variable=self.overwrite_var,
        ).pack(side=tk.LEFT)

        # Update estimated time when parameters change
        self.count_var.trace_add("write", self._update_estimated_time)
        self.interval_var.trace_add("write", self._update_estimated_time)

        # === Capture Buttons ===
        button_frame = ttk.Frame(main_frame)

        self.single_btn = ttk.Button(
            button_frame,
            text="Single Capture",
            command=self._single_capture,
            style="Large.TButton",
        )
        self.single_btn.pack(side=tk.LEFT, padx=(0, 15), ipady=5)

        self.burst_btn = ttk.Button(
            button_frame,
            text="Start Burst Capture",
            command=self._toggle_burst_capture,
            style="Large.TButton",
        )
        self.burst_btn.pack(side=tk.LEFT, padx=(0, 10), ipady=5)

        # === Status Bar ===
        self.status_bar = StatusBar(main_frame)

        # === Pack frames in correct order ===
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
        burst_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
        settings_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

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
                self._disable_capture_buttons()
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
                self._enable_capture_buttons()
                self.status_bar.set_status(
                    f"Loaded {len(self.registered_objects)} object(s) from registry"
                )
        except Exception as e:
            self.class_combo["values"] = []
            self.class_combo.set("")
            self._disable_capture_buttons()
            self.status_bar.set_status(f"Failed to load registry: {e}")

    def _disable_capture_buttons(self) -> None:
        """Disable capture buttons when no objects are available."""
        self.single_btn.config(state="disabled")
        self.burst_btn.config(state="disabled")

    def _enable_capture_buttons(self) -> None:
        """Enable capture buttons."""
        self.single_btn.config(state="normal")
        self.burst_btn.config(state="normal")

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
            title="Select Output Directory",
        )
        if directory:
            self.output_var.set(directory)

    def _update_estimated_time(self, *args) -> None:
        """Update the estimated time label."""
        try:
            count = self.count_var.get()
            interval = self.interval_var.get()
            est_time = count * interval
            self.est_time_label.config(text=f"{est_time:.1f}s")
        except (tk.TclError, ValueError):
            pass

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

            # Convert to PhotoImage
            image = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            photo = ImageTk.PhotoImage(image)

            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo

        # Handle countdown
        if self.countdown_active:
            self._process_countdown()

        # Handle burst capture
        if self.is_capturing_burst:
            self._process_burst_capture()

        # Schedule next update
        self.root.after(33, self._update_preview)

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
                self.status_bar.set_status(f"Starting in {self.countdown_remaining}...")

        if elapsed >= 3.0:
            self.countdown_active = False
            self._begin_actual_capture()

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

    def _get_next_file_number(self, output_dir: Path, class_name: str) -> int:
        """Get the next file number for the given class."""
        if self.overwrite_var.get():
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
            messagebox.showwarning("No Image", "No image available to capture")
            return

        class_name = self._get_selected_class_name()
        if not class_name:
            messagebox.showwarning(
                "Invalid Input", "Please select a class from the dropdown"
            )
            return

        output_dir = Path(self.output_var.get()) / class_name
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

        try:
            count = self.count_var.get()
            interval = self.interval_var.get()
        except (tk.TclError, ValueError):
            messagebox.showwarning("Invalid Input", "Invalid count or interval value")
            return

        output_dir = Path(self.output_var.get()) / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self.current_output_dir = output_dir
        self.current_class_name = class_name

        self.pending_burst_count = count
        self.pending_burst_interval = interval

        self.countdown_active = True
        self.countdown_remaining = 3
        self.countdown_start_time = time.time()

        self.burst_btn.config(text="Cancel")
        self.single_btn.config(state="disabled")
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

        self.burst_btn.config(text="Stop Capture")
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
        self.burst_btn.config(text="Start Burst Capture")
        self.single_btn.config(state="normal")
        self.status_bar.hide_progress()

        if was_countdown:
            self.status_bar.set_status("Capture cancelled")
        elif was_capturing and self.burst_count < self.burst_target:
            self.status_bar.set_status(
                f"Burst stopped: {self.burst_count}/{self.burst_target} images captured"
            )

    def _on_close(self) -> None:
        """Handle window close."""
        self.preview_active = False
        self._shutdown_ros2()
        self.root.destroy()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ROS2 Image Capture Application")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Default output directory for captured images",
    )
    args = parser.parse_args()

    root = tk.Tk()

    # Apply theme
    style = ttk.Style()
    AppTheme.apply(style)

    app = CaptureApp(root)

    if args.output_dir:
        app.output_var.set(args.output_dir)

    app.run()


if __name__ == "__main__":
    main()
