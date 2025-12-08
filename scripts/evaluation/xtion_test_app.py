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
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image as RosImage

# YOLO import
from ultralytics import YOLO


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
        super().__init__('xtion_test_app')
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


class XtionTestApp:
    """Main Xtion test application with Tkinter GUI."""

    def __init__(self, root: tk.Tk, model_path: str, conf_threshold: float = 0.25):
        self.root = root
        self.root.title("Xtion Live Test - YOLO Inference")
        self.root.geometry("900x750")
        self.root.resizable(True, True)

        self.model_path = model_path
        self.conf_threshold = conf_threshold

        # ROS2 setup
        self.ros_node: Optional[ROS2ImageSubscriber] = None
        self.executor: Optional[SingleThreadedExecutor] = None
        self.ros_thread: Optional[threading.Thread] = None
        self.ros_running = False

        # YOLO model
        self.model: Optional[YOLO] = None

        # Preview state
        self.preview_active = False
        self.preview_label: Optional[tk.Label] = None

        # FPS calculation
        self.frame_times: List[float] = []
        self.fps = 0.0

        # Detection results
        self.latest_detections: List[dict] = []

        # Initialize ROS2
        self._init_ros2()

        # Load YOLO model
        self._load_model()

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

    def _load_model(self):
        """Load YOLO model."""
        try:
            if not Path(self.model_path).exists():
                messagebox.showerror("Model Error", f"Model not found: {self.model_path}")
                sys.exit(1)

            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")

        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            sys.exit(1)

    def _build_gui(self):
        """Build the GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # === Model Info ===
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding="5")
        model_frame.pack(fill=tk.X, pady=(0, 10))

        model_name = Path(self.model_path).name
        ttk.Label(model_frame, text=f"Model: {model_name}").pack(side=tk.LEFT)

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
        preview_frame = ttk.LabelFrame(main_frame, text="Camera Preview + Detection", padding="5")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.preview_label = ttk.Label(preview_frame, text="No image - Select a topic", anchor="center")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # === Confidence Threshold ===
        conf_frame = ttk.LabelFrame(main_frame, text="Confidence Threshold", padding="5")
        conf_frame.pack(fill=tk.X, pady=(0, 10))

        self.conf_var = tk.DoubleVar(value=self.conf_threshold)
        self.conf_scale = ttk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            variable=self.conf_var,
            orient=tk.HORIZONTAL,
            command=self._on_conf_changed
        )
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        self.conf_label = ttk.Label(conf_frame, text=f"{self.conf_threshold:.2f}")
        self.conf_label.pack(side=tk.LEFT)

        # === Detection Results ===
        detection_frame = ttk.LabelFrame(main_frame, text="Detections", padding="5")
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

        # Initial topic refresh (must be after status_label is created)
        self._refresh_topics()

    def _refresh_topics(self):
        """Refresh the list of available topics."""
        topics = self.ros_node.get_image_topics()

        # Add common Xtion/RealSense topics if not in list
        common_topics = [
            "/camera/color/image_raw",
            "/camera/rgb/image_raw",
            "/hsrb/head_rgbd_sensor/rgb/image_rect_color",
        ]

        all_topics = list(set(topics + [t for t in common_topics if t not in topics]))
        all_topics.sort()

        self.topic_combo['values'] = all_topics

        if all_topics and not self.topic_var.get():
            # Auto-select first available topic
            for topic in all_topics:
                if topic in topics:  # Prioritize actually available topics
                    self.topic_var.set(topic)
                    self._on_topic_changed(None)
                    break

    def _on_topic_changed(self, event):
        """Handle topic selection change."""
        topic = self.topic_var.get()
        if topic:
            self.ros_node.subscribe_to_topic(topic)
            self.status_label.config(text=f"Status: Connected to {topic}")
            self.preview_active = True

    def _on_conf_changed(self, value):
        """Handle confidence threshold change."""
        self.conf_threshold = float(value)
        self.conf_label.config(text=f"{self.conf_threshold:.2f}")

    def _start_preview_loop(self):
        """Start the preview update loop."""
        self._update_preview()

    def _update_preview(self):
        """Update the preview with latest frame and detection results."""
        if self.preview_active:
            frame = self.ros_node.get_frame()

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
                    self.latest_detections.append({
                        'class': class_name,
                        'confidence': conf
                    })

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

                self.preview_label.configure(image=photo)
                self.preview_label.image = photo

        # Schedule next update (~30 FPS)
        self.root.after(33, self._update_preview)

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to fit display area dynamically."""
        # Get window size
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        # Reserved space for other UI components (padding included)
        # model_frame + topic_frame + confidence + results + margins
        reserved_height = 300
        reserved_width = 40

        # Calculate available size
        max_width = max(window_width - reserved_width, 400)
        max_height = max(window_height - reserved_height, 200)

        h, w = frame.shape[:2]
        scale = min(max_width / w, max_height / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        if new_w != w or new_h != h:
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return frame

    def _update_detection_text(self):
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

    def _on_close(self):
        """Handle window close event."""
        self.preview_active = False
        self.ros_running = False

        if self.ros_thread:
            self.ros_thread.join(timeout=1.0)

        if self.ros_node:
            self.ros_node.destroy_node()

        try:
            rclpy.shutdown()
        except Exception:
            pass

        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Xtion Live Test - YOLO Inference")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    root = tk.Tk()
    app = XtionTestApp(root, args.model, args.conf)
    root.mainloop()


if __name__ == "__main__":
    main()
