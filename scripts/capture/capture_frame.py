#!/usr/bin/env python3
"""
Capture a single frame from ROS2 image topic.

Usage:
    python3 capture_frame.py --topic /camera/rgb/image_raw --output /tmp/preview.jpg
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class FrameCapture(Node):
    """Node to capture a single frame from an image topic."""

    def __init__(self, topic: str, output_path: str):
        super().__init__('frame_capture')
        self.bridge = CvBridge()
        self.output_path = output_path
        self.received = False

        self.subscription = self.create_subscription(
            Image,
            topic,
            self.callback,
            1
        )
        self.get_logger().info(f"Subscribing to {topic}")

    def callback(self, msg: Image):
        """Handle incoming image message."""
        if not self.received:
            try:
                # Try different encodings
                if msg.encoding in ['rgb8', 'bgr8', 'mono8']:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                elif msg.encoding in ['16UC1', '32FC1']:
                    # Depth image - normalize for visualization
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                    cv_image = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX)
                    cv_image = cv_image.astype('uint8')
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                else:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

                cv2.imwrite(self.output_path, cv_image)
                self.received = True
                self.get_logger().info(f"Saved frame to {self.output_path}")

            except Exception as e:
                self.get_logger().error(f"Failed to convert image: {e}")


def main():
    parser = argparse.ArgumentParser(description="Capture a single frame from ROS2 image topic")
    parser.add_argument('--topic', required=True, help="Image topic to subscribe")
    parser.add_argument('--output', default='/tmp/preview_frame.jpg', help="Output file path")
    parser.add_argument('--timeout', type=float, default=2.0, help="Timeout in seconds")
    args = parser.parse_args()

    rclpy.init()
    node = FrameCapture(args.topic, args.output)

    # Spin until image received or timeout
    start = node.get_clock().now()
    while rclpy.ok() and not node.received:
        rclpy.spin_once(node, timeout_sec=0.1)
        elapsed = (node.get_clock().now() - start).nanoseconds / 1e9
        if elapsed > args.timeout:
            node.get_logger().warn(f"Timeout after {args.timeout}s - no image received")
            break

    success = node.received
    node.destroy_node()
    rclpy.shutdown()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
