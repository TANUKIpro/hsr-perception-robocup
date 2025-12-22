"""
Topic selector component for PyQt6 GUI framework.

Provides a ROS2 image topic selection widget with refresh capability.
"""

from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QPushButton,
)
from PyQt6.QtCore import pyqtSignal


class TopicSelector(QWidget):
    """
    ROS2 image topic selector widget.

    Features:
    - Dropdown for topic selection
    - Refresh button to update available topics
    - Signal emission on topic change

    Signals:
        topic_changed: Emitted when a new topic is selected

    Example:
        selector = TopicSelector(
            parent,
            get_topics_callback=ros_node.get_image_topics,
            subscribe_callback=ros_node.subscribe_to_topic,
        )
        selector.topic_changed.connect(self._on_topic_changed)
    """

    # Signal emitted when topic selection changes
    topic_changed = pyqtSignal(str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        label_text: str = "Image Topic:",
        get_topics_callback: Optional[Callable[[], list[str]]] = None,
        subscribe_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Initialize the topic selector.

        Args:
            parent: Parent widget
            label_text: Label text for the selector
            get_topics_callback: Callback to get available topics
            subscribe_callback: Callback to subscribe to a topic
        """
        super().__init__(parent)

        self._get_topics = get_topics_callback
        self._subscribe = subscribe_callback

        self._build_ui(label_text)

    def _build_ui(self, label_text: str) -> None:
        """Build the UI layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Label
        label = QLabel(label_text)
        layout.addWidget(label)

        # Combobox
        self.topic_combo = QComboBox()
        self.topic_combo.setMinimumWidth(300)
        self.topic_combo.currentTextChanged.connect(self._on_selection_changed)
        layout.addWidget(self.topic_combo, stretch=1)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setMaximumWidth(80)
        self.refresh_btn.clicked.connect(self.refresh_topics)
        layout.addWidget(self.refresh_btn)

    def refresh_topics(self) -> None:
        """Refresh the list of available topics."""
        if self._get_topics is None:
            return

        topics = self._get_topics()
        current = self.topic_combo.currentText()

        # Block signals during update
        self.topic_combo.blockSignals(True)
        self.topic_combo.clear()
        self.topic_combo.addItems(sorted(topics))

        # Restore selection if possible
        if current in topics:
            self.topic_combo.setCurrentText(current)
        elif topics:
            self.topic_combo.setCurrentIndex(0)

        self.topic_combo.blockSignals(False)

        # Trigger subscription for current selection
        if self.topic_combo.currentText():
            self._on_selection_changed(self.topic_combo.currentText())

    def _on_selection_changed(self, topic: str) -> None:
        """Handle topic selection change."""
        if self._subscribe and topic:
            self._subscribe(topic)
        self.topic_changed.emit(topic)

    def get_selected_topic(self) -> str:
        """
        Get the currently selected topic.

        Returns:
            Currently selected topic name
        """
        return self.topic_combo.currentText()

    def set_callbacks(
        self,
        get_topics: Callable[[], list[str]],
        subscribe: Callable[[str], None],
    ) -> None:
        """
        Set the callback functions.

        Args:
            get_topics: Callback to get available topics
            subscribe: Callback to subscribe to a topic
        """
        self._get_topics = get_topics
        self._subscribe = subscribe

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the selector.

        Args:
            enabled: True to enable, False to disable
        """
        self.topic_combo.setEnabled(enabled)
        self.refresh_btn.setEnabled(enabled)
