"""
ROS2 Application Base Class for PyQt6.

Provides a foundation for PyQt6 applications that integrate with ROS2,
using QThread-based background spinning for the ROS2 executor.
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import pyqtSlot

from gui_framework_qt.core.base_app import BaseApp
from gui_framework_qt.core.ros2_worker import ROS2Worker


class ROS2App(BaseApp):
    """
    Base class for ROS2-integrated PyQt6 applications.

    Extends BaseApp with ROS2 initialization, spinning, and cleanup.
    Uses QThread for background ROS2 node management.

    Subclasses must implement:
        - _build_gui(): Build the GUI layout
        - _on_close(): Handle window close (automatically calls ROS2 shutdown)

    Example:
        class MyROS2App(ROS2App):
            def _build_gui(self):
                # Build your GUI here
                pass

            def _on_close(self):
                # Additional cleanup if needed (ROS2 shutdown is automatic)
                pass

        if __name__ == "__main__":
            app = QApplication(sys.argv)
            window = MyROS2App(title="My ROS2 App", node_name="my_node")
            window.show()
            sys.exit(app.exec())
    """

    def __init__(
        self,
        title: str,
        node_name: str = "gui_app",
        size: tuple[int, int] = (800, 750),
        min_size: Optional[tuple[int, int]] = (600, 550),
        resizable: bool = True,
    ) -> None:
        """
        Initialize the ROS2 application.

        Args:
            title: Window title
            node_name: ROS2 node name
            size: Window size as (width, height)
            min_size: Minimum window size as (width, height)
            resizable: Whether the window is resizable
        """
        self.node_name = node_name
        self.ros2_worker: Optional[ROS2Worker] = None
        self._latest_frame: Optional[np.ndarray] = None

        # Initialize base app (this calls _build_gui)
        super().__init__(title, size, min_size, resizable)

        # Start ROS2 worker
        self._start_ros2_worker()

    def _start_ros2_worker(self) -> None:
        """Start the ROS2 worker thread."""
        self.ros2_worker = ROS2Worker(self.node_name)
        self.ros2_worker.frame_received.connect(self._on_frame_received)
        self.ros2_worker.error_occurred.connect(self._on_ros2_error)
        self.ros2_worker.start()

    def _shutdown_ros2(self) -> None:
        """Shutdown the ROS2 worker thread."""
        if self.ros2_worker is not None:
            self.ros2_worker.stop()
            self.ros2_worker = None

    @pyqtSlot(np.ndarray)
    def _on_frame_received(self, frame: np.ndarray) -> None:
        """
        Handle frame received from ROS2.

        Override this method to process frames as they arrive.

        Args:
            frame: Received frame as numpy array (BGR format)
        """
        self._latest_frame = frame

    @pyqtSlot(str)
    def _on_ros2_error(self, error: str) -> None:
        """
        Handle ROS2 error.

        Args:
            error: Error message
        """
        self.show_error("ROS2 Error", error)

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the latest frame from the ROS2 subscriber.

        Returns:
            Latest frame as numpy array, or None if not available
        """
        if self.ros2_worker is not None:
            return self.ros2_worker.get_frame()
        return self._latest_frame

    def get_image_topics(self) -> list[str]:
        """
        Get list of available image topics.

        Returns:
            List of image topic names
        """
        if self.ros2_worker is not None:
            return self.ros2_worker.get_image_topics()
        return []

    def subscribe_to_topic(self, topic: str) -> None:
        """
        Subscribe to an image topic.

        Args:
            topic: ROS2 image topic name
        """
        if self.ros2_worker is not None:
            self.ros2_worker.subscribe_to_topic(topic)

    def closeEvent(self, event) -> None:
        """Handle window close event with ROS2 cleanup."""
        self._shutdown_ros2()
        super().closeEvent(event)
