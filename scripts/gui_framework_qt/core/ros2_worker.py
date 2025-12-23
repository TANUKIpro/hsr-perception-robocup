"""
ROS2 Worker Thread for PyQt6.

Provides a QThread-based worker for spinning ROS2 nodes
in the background, with thread-safe frame access.
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker


class ROS2Worker(QThread):
    """
    Worker thread for ROS2 node spinning.

    Runs the ROS2 executor in a background thread and emits
    signals when new frames are received.

    Signals:
        frame_received: Emitted when a new frame is available
        error_occurred: Emitted when an error occurs
        topics_updated: Emitted when available topics change

    Example:
        worker = ROS2Worker("my_node")
        worker.frame_received.connect(self._on_frame)
        worker.start()
    """

    # Signal emitted when a new frame is received
    frame_received = pyqtSignal(np.ndarray)

    # Signal emitted when an error occurs
    error_occurred = pyqtSignal(str)

    # Signal emitted when topic list is updated
    topics_updated = pyqtSignal(list)

    def __init__(self, node_name: str = "gui_app") -> None:
        """
        Initialize the ROS2 worker.

        Args:
            node_name: Name for the ROS2 node
        """
        super().__init__()
        self.node_name = node_name
        self.ros_node: Optional["ROS2ImageSubscriber"] = None  # noqa: F821
        self.executor: Optional["SingleThreadedExecutor"] = None  # noqa: F821
        self._running = False
        self._mutex = QMutex()
        self._latest_frame: Optional[np.ndarray] = None

    def run(self) -> None:
        """Thread main function - initialize ROS2 and spin."""
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor

            # Import here to avoid circular imports
            import sys
            from pathlib import Path

            # Add gui_framework path if needed
            gui_framework_path = Path(__file__).parent.parent.parent
            if str(gui_framework_path) not in sys.path:
                sys.path.insert(0, str(gui_framework_path))

            from gui_framework.utils.ros2_image import ROS2ImageSubscriber

            # Initialize ROS2
            rclpy.init()
            self.ros_node = ROS2ImageSubscriber(self.node_name)
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.ros_node)

            self._running = True

            # Spin loop
            while self._running:
                try:
                    self.executor.spin_once(timeout_sec=0.05)

                    # Get frame and emit signal if available
                    frame = self.ros_node.get_frame()
                    if frame is not None:
                        with QMutexLocker(self._mutex):
                            self._latest_frame = frame.copy()
                        self.frame_received.emit(frame)

                except Exception:
                    if self._running:
                        continue
                    break

        except Exception as e:
            self.error_occurred.emit(f"ROS2 initialization failed: {e}")
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Stop the worker thread."""
        self._running = False
        self.quit()
        self.wait(2000)  # Wait up to 2 seconds

    def _cleanup(self) -> None:
        """Cleanup ROS2 resources."""
        try:
            import rclpy

            if self.executor is not None:
                try:
                    self.executor.shutdown()
                except Exception:
                    pass
                self.executor = None

            if self.ros_node is not None:
                try:
                    self.ros_node.destroy_node()
                except Exception:
                    pass
                self.ros_node = None

            try:
                rclpy.shutdown()
            except Exception:
                pass
        except ImportError:
            pass

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame (thread-safe).

        Returns:
            Latest frame as numpy array, or None if not available
        """
        with QMutexLocker(self._mutex):
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        return None

    def get_image_topics(self) -> list[str]:
        """
        Get list of available image topics.

        Returns:
            List of image topic names
        """
        with QMutexLocker(self._mutex):
            if self.ros_node is not None:
                try:
                    return self.ros_node.get_image_topics()
                except Exception:
                    pass
        return []

    def subscribe_to_topic(self, topic: str) -> None:
        """
        Subscribe to an image topic.

        Args:
            topic: ROS2 image topic name
        """
        with QMutexLocker(self._mutex):
            if self.ros_node is not None:
                try:
                    self.ros_node.subscribe_to_topic(topic)
                except Exception as e:
                    self.error_occurred.emit(f"Failed to subscribe: {e}")
