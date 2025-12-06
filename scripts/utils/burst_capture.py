#!/usr/bin/env python3
"""
Burst capture from ROS2 image topic.

Captures multiple images from a specified topic at regular intervals.

Usage:
    python3 burst_capture.py --topic /camera/rgb/image_raw --output-dir ./captures --class-name cup --count 50 --interval 0.2
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def imgmsg_to_cv2(msg: Image) -> np.ndarray:
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


class BurstCapture(Node):
    """ROS2 node for burst capturing images from a topic."""

    def __init__(self, topic: str, output_dir: Path, class_name: str, count: int, interval: float):
        super().__init__('burst_capture')
        self.output_dir = output_dir
        self.class_name = class_name
        self.target_count = count
        self.interval = interval
        self.captured_count = 0
        self.latest_frame = None
        self.last_capture_time = 0.0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Subscribe to image topic
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.callback,
            10
        )
        self.get_logger().info(f"Subscribing to {topic}")
        self.get_logger().info(f"Will capture {count} images at {interval}s intervals")
        self.get_logger().info(f"Saving to {output_dir}")

    def callback(self, msg: Image):
        """Handle incoming image message."""
        try:
            self.latest_frame = imgmsg_to_cv2(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")

    def capture_if_ready(self) -> bool:
        """
        Capture an image if interval has passed.

        Returns:
            True if all images captured, False otherwise
        """
        if self.latest_frame is None:
            return False

        current_time = time.time()
        if current_time - self.last_capture_time >= self.interval:
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.class_name}_{timestamp}.jpg"
            filepath = self.output_dir / filename

            # Save image
            cv2.imwrite(str(filepath), self.latest_frame)
            self.captured_count += 1
            self.last_capture_time = current_time

            self.get_logger().info(f"Captured {self.captured_count}/{self.target_count}: {filename}")

            if self.captured_count >= self.target_count:
                return True

        return False

    def is_done(self) -> bool:
        return self.captured_count >= self.target_count


def main():
    parser = argparse.ArgumentParser(description="Burst capture from ROS2 image topic")
    parser.add_argument('--topic', required=True, help="Image topic to subscribe")
    parser.add_argument('--output-dir', required=True, help="Output directory for captured images")
    parser.add_argument('--class-name', required=True, help="Class name for filename prefix")
    parser.add_argument('--count', type=int, default=50, help="Number of images to capture")
    parser.add_argument('--interval', type=float, default=0.2, help="Interval between captures in seconds")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    rclpy.init()
    node = BurstCapture(
        topic=args.topic,
        output_dir=output_dir,
        class_name=args.class_name,
        count=args.count,
        interval=args.interval
    )

    try:
        while rclpy.ok() and not node.is_done():
            rclpy.spin_once(node, timeout_sec=0.05)
            node.capture_if_ready()
    except KeyboardInterrupt:
        node.get_logger().info("Capture interrupted by user")
    finally:
        node.get_logger().info(f"Captured {node.captured_count} images total")
        node.destroy_node()
        rclpy.shutdown()

    return 0 if node.captured_count > 0 else 1


if __name__ == '__main__':
    sys.exit(main())
