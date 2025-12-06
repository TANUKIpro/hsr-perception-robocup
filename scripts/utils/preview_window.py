#!/usr/bin/env python3
"""
Real-time camera preview window with center reticle.

A lightweight OpenCV-based preview window for ROS2 image topics.
Displays a center crosshair reticle to help align objects.

Usage:
    python3 preview_window.py --topic /camera/rgb/image_raw

Controls:
    ESC or Q: Close window
"""

import argparse
import sys

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
    """
    Convert sensor_msgs/Image to OpenCV image without cv_bridge.

    This avoids NumPy version compatibility issues with cv_bridge.

    Args:
        msg: ROS2 Image message

    Returns:
        OpenCV BGR image (numpy array)
    """
    # Determine dtype based on encoding
    if msg.encoding in ['16UC1']:
        dtype = np.uint16
    elif msg.encoding in ['32FC1']:
        dtype = np.float32
    else:
        dtype = np.uint8

    # Determine number of channels
    if msg.encoding in ['rgb8', 'bgr8']:
        channels = 3
    elif msg.encoding in ['rgba8', 'bgra8']:
        channels = 4
    elif msg.encoding in ['mono8', 'mono16', '16UC1', '32FC1']:
        channels = 1
    else:
        # Default to 3 channels
        channels = 3

    # Convert to numpy array
    if channels == 1:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
    else:
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)

    # Convert to BGR for OpenCV display
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
        # Depth image - normalize for visualization
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # bgr8 needs no conversion

    return img


class PreviewWindow(Node):
    """ROS2 node for displaying camera preview with reticle."""

    def __init__(self, topic: str):
        super().__init__('preview_window')
        self.window_name = f"Preview: {topic}"
        self.latest_frame = None

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 640, 480)

        # Subscribe to image topic
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.callback,
            10
        )
        self.get_logger().info(f"Subscribing to {topic}")
        self.get_logger().info("Press ESC or Q to close")

    def callback(self, msg: Image):
        """Handle incoming image message."""
        try:
            self.latest_frame = imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def draw_reticle(self, frame):
        """Draw center crosshair reticle on frame."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Size proportional to image
        size = min(w, h) // 20
        color = (0, 255, 0)  # Green
        thickness = 1

        # Horizontal line
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, thickness)
        # Vertical line
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, thickness)

        return frame

    def spin_once_with_display(self) -> bool:
        """
        Process one ROS2 callback and update display.

        Returns:
            True to continue, False to exit
        """
        rclpy.spin_once(self, timeout_sec=0.03)

        if self.latest_frame is not None:
            # Draw reticle on a copy
            display = self.draw_reticle(self.latest_frame.copy())
            cv2.imshow(self.window_name, display)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF

        # ESC (27) or 'q' to quit
        if key in [27, ord('q'), ord('Q')]:
            return False

        # Check if window was closed
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Real-time camera preview with center reticle"
    )
    parser.add_argument(
        '--topic',
        required=True,
        help="Image topic to subscribe"
    )
    args = parser.parse_args()

    rclpy.init()
    node = PreviewWindow(args.topic)

    try:
        while rclpy.ok():
            if not node.spin_once_with_display():
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == '__main__':
    sys.exit(main())
