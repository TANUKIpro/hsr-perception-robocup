"""
ROS2 Image Utilities.

Provides image message conversion and a thread-safe ROS2 image subscriber.
"""

import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage


def imgmsg_to_cv2(msg: RosImage) -> np.ndarray:
    """
    Convert sensor_msgs/Image to OpenCV image without cv_bridge.

    Args:
        msg: ROS2 Image message

    Returns:
        BGR OpenCV image as numpy array
    """
    # Determine dtype based on encoding
    if msg.encoding in ["16UC1"]:
        dtype = np.uint16
    elif msg.encoding in ["32FC1"]:
        dtype = np.float32
    else:
        dtype = np.uint8

    # Determine number of channels
    if msg.encoding in ["rgb8", "bgr8"]:
        channels = 3
    elif msg.encoding in ["rgba8", "bgra8"]:
        channels = 4
    elif msg.encoding in ["mono8", "mono16", "16UC1", "32FC1"]:
        channels = 1
    else:
        channels = 3

    # Reshape data to image
    if channels == 1:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    else:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(
            msg.height, msg.width, channels
        )

    # Convert to BGR
    if msg.encoding == "rgb8":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif msg.encoding == "rgba8":
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif msg.encoding == "bgra8":
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif msg.encoding == "mono8":
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif msg.encoding == "mono16":
        img = (img / 256).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif msg.encoding in ["16UC1", "32FC1"]:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


class ROS2ImageSubscriber(Node):
    """
    ROS2 node for subscribing to image topics.

    Thread-safe implementation with frame locking.

    Example:
        node = ROS2ImageSubscriber("my_app")
        node.subscribe_to_topic("/camera/image_raw")
        frame = node.get_frame()
    """

    def __init__(self, node_name: str = "gui_app") -> None:
        """
        Initialize the ROS2 image subscriber.

        Args:
            node_name: Name for the ROS2 node
        """
        super().__init__(node_name)
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.subscription = None
        self.current_topic: Optional[str] = None

    def subscribe_to_topic(self, topic: str) -> None:
        """
        Subscribe to a new image topic.

        Args:
            topic: ROS2 image topic name
        """
        # Unsubscribe from previous topic
        if self.subscription is not None:
            self.destroy_subscription(self.subscription)
            self.subscription = None

        self.current_topic = topic
        self.latest_frame = None

        if topic:
            self.subscription = self.create_subscription(
                RosImage, topic, self._image_callback, 10
            )
            self.get_logger().info(f"Subscribed to {topic}")

    def _image_callback(self, msg: RosImage) -> None:
        """Handle incoming image message."""
        try:
            frame = imgmsg_to_cv2(msg)
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame (thread-safe).

        Returns:
            Copy of the latest frame, or None if no frame received
        """
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def get_image_topics(self) -> list[str]:
        """
        Get list of available image topics.

        Returns:
            Sorted list of image topic names
        """
        topics = self.get_topic_names_and_types()
        image_topics = []
        for name, types in topics:
            for t in types:
                if "sensor_msgs/msg/Image" in t or "sensor_msgs/Image" in t:
                    image_topics.append(name)
                    break
        return sorted(image_topics)
