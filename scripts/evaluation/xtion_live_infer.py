#!/usr/bin/env python3
"""Xtion live YOLO inference viewer.

Subscribes to a ROS2 sensor_msgs/Image topic (typically the Xtion RGB
stream), runs a fine-tuned YOLOv8 model on each frame, and renders the
annotated frame in a PyQt6 window with topic selection, confidence
slider, FPS counter, and per-detection list.

Usage:
    python3 scripts/evaluation/xtion_live_infer.py \
        --model models/finetuned/<run>/weights/best.pt --conf 0.25

Prereqs (host or container):
    - ROS2 Humble (rclpy, sensor_msgs)
    - PyQt6
    - ultralytics, opencv-python, numpy

See CLAUDE.md -> "Xtion live inference" for the Docker workflow.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional


def _die(msg: str, code: int = 1) -> None:
    print(f"[xtion_live_infer] {msg}", file=sys.stderr)
    sys.exit(code)


try:
    import cv2
    import numpy as np
except ImportError as e:  # pragma: no cover
    _die(f"missing cv2/numpy: {e}")

try:
    from PyQt6.QtCore import Qt, QThread, QMutex, QMutexLocker, QTimer, pyqtSignal, pyqtSlot
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSlider,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )
except ImportError as e:  # pragma: no cover
    _die(
        "PyQt6 is not installed. Install it via `pip install PyQt6` "
        f"or run inside the hsr-perception container. ({e})"
    )

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
except ImportError as e:  # pragma: no cover
    _die(
        "rclpy / sensor_msgs is not available. Source a ROS2 Humble "
        "environment (`source /opt/ros/humble/setup.bash`) or run "
        f"inside the hsr-perception container. ({e})"
    )

try:
    from ultralytics import YOLO
except ImportError as e:  # pragma: no cover
    _die(f"ultralytics is required: {e}")


# ---------------------------------------------------------------------------
# ROS2 image helpers
# ---------------------------------------------------------------------------


def _imgmsg_to_bgr(msg: RosImage) -> np.ndarray:
    """Convert sensor_msgs/Image to a BGR numpy array without cv_bridge."""
    enc = msg.encoding

    if enc in ("16UC1",):
        dtype, channels = np.uint16, 1
    elif enc in ("32FC1",):
        dtype, channels = np.float32, 1
    elif enc in ("mono8",):
        dtype, channels = np.uint8, 1
    elif enc in ("mono16",):
        dtype, channels = np.uint16, 1
    elif enc in ("rgba8", "bgra8"):
        dtype, channels = np.uint8, 4
    else:  # rgb8 / bgr8 / unknown -> assume 3-channel uint8
        dtype, channels = np.uint8, 3

    buf = np.frombuffer(msg.data, dtype=dtype)
    if channels == 1:
        img = buf.reshape(msg.height, msg.width)
    else:
        img = buf.reshape(msg.height, msg.width, channels)

    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc == "mono8":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc == "mono16":
        img8 = (img / 256).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    if enc in ("16UC1", "32FC1"):
        norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return img  # bgr8 or unknown 3ch


class _ImageSubscriberNode(Node):
    """Thread-safe ROS2 node that tracks the latest image on a single topic."""

    def __init__(self, node_name: str) -> None:
        super().__init__(node_name)
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._sub = None
        self._topic: Optional[str] = None

    def subscribe_to(self, topic: str) -> None:
        if self._sub is not None:
            self.destroy_subscription(self._sub)
            self._sub = None
        self._topic = topic or None
        with self._lock:
            self._frame = None
        if topic:
            self._sub = self.create_subscription(RosImage, topic, self._on_image, 10)
            self.get_logger().info(f"subscribed to {topic}")

    def _on_image(self, msg: RosImage) -> None:
        try:
            bgr = _imgmsg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(f"image decode failed: {e}")
            return
        with self._lock:
            self._frame = bgr

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def list_image_topics(self) -> list[str]:
        out: list[str] = []
        for name, types in self.get_topic_names_and_types():
            if not any("sensor_msgs/msg/Image" in t or "sensor_msgs/Image" in t for t in types):
                continue
            if self.get_publishers_info_by_topic(name):
                out.append(name)
        return sorted(out)


# ---------------------------------------------------------------------------
# Qt worker for the ROS2 executor
# ---------------------------------------------------------------------------


class _ROS2Worker(QThread):
    error = pyqtSignal(str)
    ready = pyqtSignal()

    def __init__(self, node_name: str) -> None:
        super().__init__()
        self._node_name = node_name
        self._running = False
        self._mutex = QMutex()
        self.node: Optional[_ImageSubscriberNode] = None
        self._executor: Optional[SingleThreadedExecutor] = None

    def run(self) -> None:
        try:
            if not rclpy.ok():
                rclpy.init()
            with QMutexLocker(self._mutex):
                self.node = _ImageSubscriberNode(self._node_name)
                self._executor = SingleThreadedExecutor()
                self._executor.add_node(self.node)
            self._running = True
            self.ready.emit()
            while self._running:
                try:
                    self._executor.spin_once(timeout_sec=0.05)
                except Exception:
                    if not self._running:
                        break
        except Exception as e:
            self.error.emit(f"ROS2 init failed: {e}")
        finally:
            self._cleanup()

    def stop(self) -> None:
        self._running = False
        self.quit()
        self.wait(2000)

    def _cleanup(self) -> None:
        with QMutexLocker(self._mutex):
            if self._executor is not None:
                try:
                    self._executor.shutdown()
                except Exception:
                    pass
                self._executor = None
            if self.node is not None:
                try:
                    self.node.destroy_node()
                except Exception:
                    pass
                self.node = None
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class XtionLiveWindow(QMainWindow):
    def __init__(self, model_path: str, conf: float) -> None:
        super().__init__()
        self.setWindowTitle("Xtion Live - YOLO Inference")
        self.resize(900, 780)

        if not Path(model_path).exists():
            QMessageBox.critical(None, "Model", f"Model not found: {model_path}")
            sys.exit(1)
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            QMessageBox.critical(None, "Model", f"Failed to load model: {e}")
            sys.exit(1)
        print(f"[xtion_live_infer] model loaded: {model_path}")

        self.model_path = model_path
        self.conf = float(conf)
        self._frame_times: list[float] = []

        self._build_ui()

        self.worker = _ROS2Worker("xtion_live_infer")
        self.worker.error.connect(self._on_worker_error)
        self.worker.ready.connect(self._on_worker_ready)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)  # ~30 FPS

    # -- UI -----------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Model info
        mbox = QGroupBox("Model")
        ml = QHBoxLayout(mbox)
        ml.addWidget(QLabel(f"Model: {Path(self.model_path).name}"))
        ml.addStretch()
        root.addWidget(mbox)

        # Topic
        tbox = QGroupBox("Topic")
        tl = QHBoxLayout(tbox)
        self.topic_cb = QComboBox()
        self.topic_cb.setEditable(True)
        self.topic_cb.setMinimumWidth(420)
        self.topic_cb.activated.connect(self._on_topic_activated)
        tl.addWidget(self.topic_cb, stretch=1)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._refresh_topics)
        tl.addWidget(self.refresh_btn)
        self.subscribe_btn = QPushButton("Subscribe")
        self.subscribe_btn.clicked.connect(self._subscribe_current)
        tl.addWidget(self.subscribe_btn)
        root.addWidget(tbox)

        # Preview
        pbox = QGroupBox("Preview")
        pl = QVBoxLayout(pbox)
        self.preview = QLabel("Waiting for frames...")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background:#2c3e50; color:#ecf0f1;")
        self.preview.setMinimumSize(640, 360)
        pl.addWidget(self.preview)
        root.addWidget(pbox, stretch=1)

        # Confidence
        cbox = QGroupBox("Confidence")
        cl = QHBoxLayout(cbox)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setValue(int(self.conf * 100))
        self.slider.valueChanged.connect(self._on_conf_changed)
        cl.addWidget(self.slider, stretch=1)
        self.conf_lbl = QLabel(f"{self.conf:.2f}")
        self.conf_lbl.setMinimumWidth(40)
        cl.addWidget(self.conf_lbl)
        root.addWidget(cbox)

        # Detections
        dbox = QGroupBox("Detections")
        dl = QVBoxLayout(dbox)
        self.det_text = QTextEdit()
        self.det_text.setReadOnly(True)
        self.det_text.setMaximumHeight(120)
        dl.addWidget(self.det_text)
        root.addWidget(dbox)

        # Status
        sbar = QHBoxLayout()
        self.fps_lbl = QLabel("FPS: --")
        sbar.addWidget(self.fps_lbl)
        sbar.addStretch()
        self.status_lbl = QLabel("Status: starting ROS2...")
        sbar.addWidget(self.status_lbl)
        root.addLayout(sbar)

    # -- Slots --------------------------------------------------------------

    @pyqtSlot()
    def _on_worker_ready(self) -> None:
        self.status_lbl.setText("Status: ROS2 ready")
        self._refresh_topics()

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str) -> None:
        self.status_lbl.setText(f"Status: {msg}")
        QMessageBox.critical(self, "ROS2", msg)

    @pyqtSlot(int)
    def _on_conf_changed(self, v: int) -> None:
        self.conf = v / 100.0
        self.conf_lbl.setText(f"{self.conf:.2f}")

    @pyqtSlot()
    def _refresh_topics(self) -> None:
        node = self.worker.node
        if node is None:
            return
        topics = node.list_image_topics()
        current = self.topic_cb.currentText()
        self.topic_cb.blockSignals(True)
        self.topic_cb.clear()
        self.topic_cb.addItems(topics)
        if current:
            idx = self.topic_cb.findText(current)
            if idx >= 0:
                self.topic_cb.setCurrentIndex(idx)
            else:
                self.topic_cb.setEditText(current)
        self.topic_cb.blockSignals(False)
        self.status_lbl.setText(f"Status: {len(topics)} image topic(s) found")

    @pyqtSlot(int)
    def _on_topic_activated(self, _idx: int) -> None:
        self._subscribe_current()

    @pyqtSlot()
    def _subscribe_current(self) -> None:
        node = self.worker.node
        if node is None:
            return
        topic = self.topic_cb.currentText().strip()
        if not topic:
            return
        node.subscribe_to(topic)
        self.status_lbl.setText(f"Status: subscribed {topic}")

    # -- Preview loop -------------------------------------------------------

    @pyqtSlot()
    def _tick(self) -> None:
        node = self.worker.node
        if node is None:
            return
        frame = node.get_frame()
        if frame is None:
            return

        t0 = time.time()
        try:
            results = self.model(frame, conf=self.conf, verbose=False)
        except Exception as e:
            self.status_lbl.setText(f"Status: inference error: {e}")
            return
        result = results[0]
        annotated = result.plot()

        dets: list[str] = []
        for box in result.boxes:
            cid = int(box.cls.item())
            name = self.model.names.get(cid, str(cid)) if isinstance(self.model.names, dict) else self.model.names[cid]
            dets.append(f"  - {name}: {box.conf.item():.2f}")
        self.det_text.setText("\n".join(dets) if dets else "  (no detections)")

        dt = time.time() - t0
        self._frame_times.append(dt)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self.fps_lbl.setText(f"FPS: {1.0 / avg:.1f}" if avg > 0 else "FPS: --")

        pixmap = self._to_pixmap(annotated)
        self.preview.setPixmap(pixmap)

    def _to_pixmap(self, bgr: np.ndarray) -> QPixmap:
        lw, lh = self.preview.width(), self.preview.height()
        if lw > 1 and lh > 1:
            h, w = bgr.shape[:2]
            scale = min(lw / w, lh / h)
            nw, nh = int(w * scale), int(h * scale)
            if nw != w or nh != h:
                bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return QPixmap.fromImage(qimg)

    # -- Teardown -----------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        self.timer.stop()
        self.worker.stop()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Xtion live YOLO inference viewer")
    parser.add_argument("--model", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Initial confidence threshold")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    win = XtionLiveWindow(args.model, args.conf)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
