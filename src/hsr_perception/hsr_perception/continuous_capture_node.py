#!/usr/bin/env python3
"""
Continuous Capture Node

ROS2 node for continuous image capture during competition day data collection.
Supports single capture and burst capture modes with class selection.
"""

import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger

# Import custom service types (generated from .srv files)
from hsr_perception.srv import GetStatus, SetClass, StartBurst


class ContinuousCaptureNode(Node):
    """
    ROS2 node for continuous image capture.

    Subscribes to HSR camera topics and provides services for:
    - Setting current capture class
    - Single frame capture
    - Burst capture mode
    - Status retrieval
    """

    def __init__(self):
        super().__init__("continuous_capture_node")

        # Declare parameters
        self.declare_parameter("class_config_path", "config/object_classes.json")
        self.declare_parameter("output_base_dir", "datasets/raw_captures")
        self.declare_parameter(
            "image_topic", "/hsrb/head_rgbd_sensor/rgb/image_rect_color"
        )
        self.declare_parameter("burst_interval", 0.2)
        self.declare_parameter("burst_default_count", 50)
        self.declare_parameter("image_format", "jpg")
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("preview_enabled", True)

        # Get parameters
        self.class_config_path = (
            self.get_parameter("class_config_path").get_parameter_value().string_value
        )
        self.output_base_dir = (
            self.get_parameter("output_base_dir").get_parameter_value().string_value
        )
        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.burst_interval = (
            self.get_parameter("burst_interval").get_parameter_value().double_value
        )
        self.burst_default_count = (
            self.get_parameter("burst_default_count").get_parameter_value().integer_value
        )
        self.image_format = (
            self.get_parameter("image_format").get_parameter_value().string_value
        )
        self.jpeg_quality = (
            self.get_parameter("jpeg_quality").get_parameter_value().integer_value
        )
        self.preview_enabled = (
            self.get_parameter("preview_enabled").get_parameter_value().bool_value
        )

        # Initialize state
        self.cv_bridge = CvBridge()
        self.latest_image: Optional[np.ndarray] = None
        self.image_lock = Lock()

        self.current_class_id: int = -1
        self.current_class_name: str = ""
        self.is_capturing: bool = False
        self.burst_count: int = 0
        self.burst_target: int = 0
        self.burst_timer = None

        # Load class configuration
        self.class_config: Dict = {}
        self.class_counts: Dict[str, int] = {}
        self._load_class_config()

        # Create output directory
        Path(self.output_base_dir).mkdir(parents=True, exist_ok=True)

        # QoS for image subscription (best effort for camera streams)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_callback, image_qos
        )

        # Create publishers
        self.status_pub = self.create_publisher(String, "~/status", 10)
        self.count_pub = self.create_publisher(Int32, "~/count", 10)
        if self.preview_enabled:
            self.preview_pub = self.create_publisher(Image, "~/preview", 1)

        # Create services
        self.set_class_srv = self.create_service(
            SetClass, "~/set_class", self._handle_set_class
        )
        self.start_burst_srv = self.create_service(
            StartBurst, "~/start_burst", self._handle_start_burst
        )
        self.stop_burst_srv = self.create_service(
            Trigger, "~/stop_burst", self._handle_stop_burst
        )
        self.single_capture_srv = self.create_service(
            Trigger, "~/single_capture", self._handle_single_capture
        )
        self.get_status_srv = self.create_service(
            GetStatus, "~/get_status", self._handle_get_status
        )
        self.reload_config_srv = self.create_service(
            Trigger, "~/reload_config", self._handle_reload_config
        )

        self.get_logger().info(
            f"ContinuousCaptureNode initialized. Subscribing to: {self.image_topic}"
        )
        self.get_logger().info(f"Output directory: {self.output_base_dir}")
        self.get_logger().info(f"Loaded {len(self.class_config.get('objects', []))} classes")

    def _load_class_config(self) -> None:
        """Load class configuration from JSON file."""
        try:
            config_path = Path(self.class_config_path)
            if not config_path.is_absolute():
                # Try relative to workspace
                workspace_path = Path(__file__).parent.parent.parent.parent.parent
                config_path = workspace_path / self.class_config_path

            with open(config_path, "r") as f:
                self.class_config = json.load(f)

            # Initialize counts
            self._update_class_counts()

            self.get_logger().info(f"Loaded config from: {config_path}")

        except Exception as e:
            self.get_logger().error(f"Failed to load class config: {e}")
            self.class_config = {"objects": [], "categories": [], "settings": {}}

    def _update_class_counts(self) -> None:
        """Update class counts from filesystem."""
        self.class_counts = {}
        output_path = Path(self.output_base_dir)

        for obj in self.class_config.get("objects", []):
            class_name = obj["class_name"]
            class_dir = output_path / class_name

            if class_dir.exists():
                count = len(
                    [
                        f
                        for f in class_dir.iterdir()
                        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                    ]
                )
            else:
                count = 0

            self.class_counts[class_name] = count

    def _image_callback(self, msg: Image) -> None:
        """Handle incoming camera images."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.image_lock:
                self.latest_image = cv_image.copy()

            # Publish preview if enabled
            if self.preview_enabled and self.current_class_name:
                preview = self._create_preview(cv_image)
                preview_msg = self.cv_bridge.cv2_to_imgmsg(preview, "bgr8")
                self.preview_pub.publish(preview_msg)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def _create_preview(self, image: np.ndarray) -> np.ndarray:
        """Create preview image with overlay information."""
        preview = image.copy()
        h, w = preview.shape[:2]

        # Add overlay text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 255, 0) if self.is_capturing else (255, 255, 255)

        # Class name
        cv2.putText(
            preview,
            f"Class: {self.current_class_name}",
            (10, 30),
            font,
            font_scale,
            color,
            thickness,
        )

        # Count
        current_count = self.class_counts.get(self.current_class_name, 0)
        target = self._get_target_samples(self.current_class_name)
        cv2.putText(
            preview,
            f"Count: {current_count}/{target}",
            (10, 60),
            font,
            font_scale,
            color,
            thickness,
        )

        # Capture status
        if self.is_capturing:
            status = f"CAPTURING ({self.burst_count}/{self.burst_target})"
            cv2.putText(preview, status, (10, 90), font, font_scale, (0, 0, 255), thickness)

        return preview

    def _get_target_samples(self, class_name: str) -> int:
        """Get target sample count for a class."""
        for obj in self.class_config.get("objects", []):
            if obj["class_name"] == class_name:
                return obj.get(
                    "target_samples",
                    self.class_config.get("settings", {}).get("default_target_samples", 100),
                )
        return 100

    def _get_class_info(self, class_id: int) -> Optional[Dict]:
        """Get class info by ID."""
        for obj in self.class_config.get("objects", []):
            if obj["class_id"] == class_id:
                return obj
        return None

    def _save_image(self, image: np.ndarray, class_name: str) -> str:
        """Save image to disk."""
        class_dir = Path(self.output_base_dir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{class_name}_{timestamp}.{self.image_format}"
        filepath = class_dir / filename

        # Save image
        if self.image_format.lower() in ["jpg", "jpeg"]:
            cv2.imwrite(
                str(filepath),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
        else:
            cv2.imwrite(str(filepath), image)

        # Update count
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        return str(filepath)

    def _handle_set_class(
        self, request: SetClass.Request, response: SetClass.Response
    ) -> SetClass.Response:
        """Handle set_class service request."""
        class_info = self._get_class_info(request.class_id)

        if class_info is None:
            response.success = False
            response.message = f"Class ID {request.class_id} not found"
            response.class_name = ""
            response.current_count = 0
            response.target_count = 0
            return response

        self.current_class_id = request.class_id
        self.current_class_name = class_info["class_name"]

        response.success = True
        response.class_name = self.current_class_name
        response.current_count = self.class_counts.get(self.current_class_name, 0)
        response.target_count = self._get_target_samples(self.current_class_name)
        response.message = f"Set class to: {self.current_class_name}"

        self.get_logger().info(f"Class set to: {self.current_class_name} (ID: {request.class_id})")

        # Publish status
        self.status_pub.publish(String(data=f"Class: {self.current_class_name}"))

        return response

    def _handle_start_burst(
        self, request: StartBurst.Request, response: StartBurst.Response
    ) -> StartBurst.Response:
        """Handle start_burst service request."""
        # Set class if specified
        if request.class_id >= 0:
            class_info = self._get_class_info(request.class_id)
            if class_info is None:
                response.success = False
                response.message = f"Class ID {request.class_id} not found"
                return response
            self.current_class_id = request.class_id
            self.current_class_name = class_info["class_name"]

        if not self.current_class_name:
            response.success = False
            response.message = "No class selected"
            return response

        if self.is_capturing:
            response.success = False
            response.message = "Already capturing"
            return response

        # Set burst parameters
        self.burst_target = request.num_images if request.num_images > 0 else self.burst_default_count
        interval = request.interval_seconds if request.interval_seconds > 0 else self.burst_interval
        self.burst_count = 0
        self.is_capturing = True

        # Create burst timer
        self.burst_timer = self.create_timer(interval, self._burst_capture_callback)

        response.success = True
        response.message = f"Started burst capture: {self.burst_target} images at {1/interval:.1f} FPS"
        response.class_id = self.current_class_id
        response.class_name = self.current_class_name

        self.get_logger().info(response.message)

        return response

    def _burst_capture_callback(self) -> None:
        """Timer callback for burst capture."""
        if not self.is_capturing:
            return

        with self.image_lock:
            if self.latest_image is None:
                self.get_logger().warn("No image available for capture")
                return
            image = self.latest_image.copy()

        # Save image
        filepath = self._save_image(image, self.current_class_name)
        self.burst_count += 1

        # Publish count
        self.count_pub.publish(Int32(data=self.class_counts[self.current_class_name]))

        self.get_logger().debug(f"Captured: {filepath} ({self.burst_count}/{self.burst_target})")

        # Check if done
        if self.burst_target > 0 and self.burst_count >= self.burst_target:
            self._stop_burst()
            self.get_logger().info(
                f"Burst complete: {self.burst_count} images captured for {self.current_class_name}"
            )

    def _stop_burst(self) -> None:
        """Stop burst capture."""
        self.is_capturing = False
        if self.burst_timer:
            self.burst_timer.cancel()
            self.burst_timer = None

    def _handle_stop_burst(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle stop_burst service request."""
        if not self.is_capturing:
            response.success = False
            response.message = "Not currently capturing"
            return response

        captured = self.burst_count
        self._stop_burst()

        response.success = True
        response.message = f"Stopped burst capture. {captured} images captured."

        self.get_logger().info(response.message)

        return response

    def _handle_single_capture(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle single_capture service request."""
        if not self.current_class_name:
            response.success = False
            response.message = "No class selected"
            return response

        with self.image_lock:
            if self.latest_image is None:
                response.success = False
                response.message = "No image available"
                return response
            image = self.latest_image.copy()

        filepath = self._save_image(image, self.current_class_name)

        response.success = True
        response.message = f"Captured: {filepath}"

        # Publish count
        self.count_pub.publish(Int32(data=self.class_counts[self.current_class_name]))

        return response

    def _handle_get_status(
        self, request: GetStatus.Request, response: GetStatus.Response
    ) -> GetStatus.Response:
        """Handle get_status service request."""
        response.current_class = self.current_class_name
        response.current_class_id = self.current_class_id
        response.current_count = self.class_counts.get(self.current_class_name, 0)
        response.target_count = self._get_target_samples(self.current_class_name)
        response.is_capturing = self.is_capturing
        response.output_directory = self.output_base_dir

        # All classes
        response.all_class_names = []
        response.all_class_counts = []
        response.all_target_counts = []

        for obj in self.class_config.get("objects", []):
            name = obj["class_name"]
            response.all_class_names.append(name)
            response.all_class_counts.append(self.class_counts.get(name, 0))
            response.all_target_counts.append(self._get_target_samples(name))

        return response

    def _handle_reload_config(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Handle reload_config service request."""
        try:
            self._load_class_config()
            response.success = True
            response.message = f"Reloaded config. {len(self.class_config.get('objects', []))} classes."
        except Exception as e:
            response.success = False
            response.message = f"Failed to reload config: {e}"

        return response


def main(args=None):
    """Entry point for continuous capture node."""
    rclpy.init(args=args)
    node = ContinuousCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
