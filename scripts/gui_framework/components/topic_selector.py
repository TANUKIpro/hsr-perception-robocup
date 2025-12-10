"""
Topic Selector Component.

Provides a ROS2 image topic selection widget.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional

from gui_framework.utils.ros2_image import ROS2ImageSubscriber


# Common ROS2 image topics
COMMON_TOPICS = [
    "/camera/color/image_raw",
    "/camera/image_raw",
    "/image_raw",
    "/xtion/rgb/image_raw",
    "/xtion/depth/image_raw",
    "/head_camera/rgb/image_raw",
    "/hand_camera/image_raw",
]


class TopicSelector(ttk.Frame):
    """
    Topic selection widget for ROS2 image topics.

    Provides a combobox for topic selection and a refresh button.

    Example:
        selector = TopicSelector(
            parent,
            ros_node=my_ros_node,
            on_change=on_topic_changed
        )
        selector.pack(fill=tk.X)
    """

    def __init__(
        self,
        parent: tk.Widget,
        ros_node: Optional[ROS2ImageSubscriber] = None,
        on_change: Optional[Callable[[str], None]] = None,
        label_text: str = "Image Topic:",
        width: int = 50,
        **kwargs,
    ) -> None:
        """
        Initialize the topic selector.

        Args:
            parent: Parent widget
            ros_node: ROS2 image subscriber node
            on_change: Callback when topic selection changes
            label_text: Label text for the selector
            width: Width of the combobox
            **kwargs: Additional arguments passed to ttk.Frame
        """
        super().__init__(parent, **kwargs)

        self._ros_node = ros_node
        self._on_change = on_change

        # Label
        ttk.Label(self, text=label_text).pack(side=tk.LEFT)

        # Topic combobox
        self.topic_var = tk.StringVar()
        self.topic_combo = ttk.Combobox(
            self,
            textvariable=self.topic_var,
            state="readonly",
            width=width,
        )
        self.topic_combo.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        self.topic_combo.bind("<<ComboboxSelected>>", self._on_selection_changed)

        # Refresh button
        self.refresh_btn = ttk.Button(
            self,
            text="Refresh",
            command=self.refresh_topics,
        )
        self.refresh_btn.pack(side=tk.LEFT)

    def set_ros_node(self, ros_node: ROS2ImageSubscriber) -> None:
        """
        Set the ROS2 node for topic discovery.

        Args:
            ros_node: ROS2 image subscriber node
        """
        self._ros_node = ros_node

    def refresh_topics(self) -> None:
        """Refresh the list of available topics."""
        topics = []

        if self._ros_node is not None:
            try:
                topics = self._ros_node.get_image_topics()
            except Exception:
                pass

        # Add common topics that might not be published yet
        for topic in COMMON_TOPICS:
            if topic not in topics:
                topics.append(topic)

        topics.sort()
        self.topic_combo["values"] = topics

        # Keep current selection if still valid
        current = self.topic_var.get()
        if current and current in topics:
            self.topic_var.set(current)
        elif topics:
            # Select first available topic
            self.topic_var.set(topics[0])
            self._on_selection_changed(None)

    def get_selected_topic(self) -> str:
        """
        Get the currently selected topic.

        Returns:
            Selected topic name
        """
        return self.topic_var.get()

    def set_selected_topic(self, topic: str) -> None:
        """
        Set the selected topic.

        Args:
            topic: Topic name to select
        """
        self.topic_var.set(topic)
        self._on_selection_changed(None)

    def _on_selection_changed(self, event) -> None:
        """Handle topic selection change."""
        topic = self.topic_var.get()

        # Subscribe to the topic
        if self._ros_node is not None and topic:
            self._ros_node.subscribe_to_topic(topic)

        # Call user callback
        if self._on_change is not None:
            self._on_change(topic)

    def set_on_change(self, callback: Callable[[str], None]) -> None:
        """
        Set the change callback.

        Args:
            callback: Function to call when selection changes
        """
        self._on_change = callback
