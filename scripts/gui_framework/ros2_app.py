"""
ROS2 Application Base Class.

Provides a foundation for Tkinter applications that interact with ROS2.
"""

import threading
import tkinter as tk
from abc import abstractmethod
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor

from gui_framework.base_app import BaseApp
from gui_framework.utils.ros2_image import ROS2ImageSubscriber


class ROS2App(BaseApp):
    """
    Base class for ROS2-integrated Tkinter applications.

    Extends BaseApp with ROS2 initialization, spinning, and cleanup.

    Subclasses must implement:
        - _build_gui(): Build the GUI layout
        - _on_close(): Handle window close (must call _shutdown_ros2())

    Example:
        class MyROS2App(ROS2App):
            def _build_gui(self):
                # Build your GUI here
                pass

            def _on_close(self):
                self._shutdown_ros2()
                self.root.destroy()

        if __name__ == "__main__":
            root = tk.Tk()
            app = MyROS2App(root, "My ROS2 App", "my_node")
            app.run()
    """

    def __init__(
        self,
        root: tk.Tk,
        title: str,
        node_name: str = "gui_app",
        geometry: str = "800x750",
        min_size: tuple[int, int] | None = (600, 550),
        resizable: tuple[bool, bool] = (True, True),
    ) -> None:
        """
        Initialize the ROS2 application.

        Args:
            root: Tkinter root window
            title: Window title
            node_name: ROS2 node name
            geometry: Window geometry string
            min_size: Minimum window size as (width, height)
            resizable: Tuple of (width_resizable, height_resizable)
        """
        # Initialize ROS2 first
        self.ros_node: Optional[ROS2ImageSubscriber] = None
        self.executor: Optional[SingleThreadedExecutor] = None
        self._ros_running = False
        self._ros_thread: Optional[threading.Thread] = None

        self._init_ros2(node_name)

        # Initialize base app (this calls _build_gui)
        super().__init__(root, title, geometry, min_size, resizable)

    def _init_ros2(self, node_name: str) -> None:
        """
        Initialize ROS2 node and executor.

        Args:
            node_name: Name for the ROS2 node
        """
        try:
            rclpy.init()
            self.ros_node = ROS2ImageSubscriber(node_name)
            self.executor = SingleThreadedExecutor()
            self.executor.add_node(self.ros_node)

            # Start spinning in background thread
            self._ros_running = True
            self._ros_thread = threading.Thread(
                target=self._ros_spin_thread, daemon=True
            )
            self._ros_thread.start()
        except Exception as e:
            print(f"Failed to initialize ROS2: {e}")
            self.ros_node = None
            self.executor = None

    def _ros_spin_thread(self) -> None:
        """Background thread for ROS2 spinning."""
        while self._ros_running and self.executor is not None:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception:
                break

    def _shutdown_ros2(self) -> None:
        """Shutdown ROS2 node and executor."""
        self._ros_running = False

        if self._ros_thread is not None:
            self._ros_thread.join(timeout=1.0)
            self._ros_thread = None

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

    def get_frame(self) -> Optional["numpy.ndarray"]:  # noqa: F821
        """
        Get the latest frame from the ROS2 subscriber.

        Returns:
            Latest frame as numpy array, or None if not available
        """
        if self.ros_node is not None:
            return self.ros_node.get_frame()
        return None

    def get_image_topics(self) -> list[str]:
        """
        Get list of available image topics.

        Returns:
            List of image topic names
        """
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
        if self.ros_node is not None:
            self.ros_node.subscribe_to_topic(topic)

    @abstractmethod
    def _build_gui(self) -> None:
        """Build the GUI layout."""
        pass

    @abstractmethod
    def _on_close(self) -> None:
        """
        Handle window close event.

        Must call _shutdown_ros2() before destroying the window.
        """
        pass
