"""
SAM2 Annotation Application - Main Window.

PyQt6-based main window for interactive SAM2 annotation.
"""

import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QGroupBox,
    QProgressBar,
    QLabel,
    QMessageBox,
    QApplication,
)
from PyQt6.QtCore import Qt, pyqtSlot, QTimer
from PyQt6.QtGui import QShortcut, QKeySequence

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from annotation_utils import (
    bbox_to_yolo,
    mask_to_bbox,
    write_yolo_label,
    batch_save_yolo_labels,
    read_yolo_label,
    yolo_to_bbox,
)
from video_tracking_predictor import (
    VideoTrackingPredictor,
    TrackingResult,
    BatchInfo,
)

from sam2_app_qt.state import AnnotationState, TrackingState
from sam2_app_qt.predictors import SAM2InteractivePredictor
from sam2_app_qt.widgets import (
    ImageCanvas,
    ImageListWidget,
    ControlPanel,
    TrackingPanel,
)
from sam2_app_qt.workers import ModelLoaderWorker, TrackingWorker, SaveWorker
from sam2_app_qt.utils import draw_mask_overlay, draw_points


class SAM2AnnotationWindow(QMainWindow):
    """
    Main window for SAM2 interactive annotation.

    Provides GUI for point-based segmentation with real-time preview,
    undo/reset functionality, and batch image processing.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        class_id: int = 0,
        model_path: str = "sam2_b.pt",
        device: str = "cuda",
    ):
        """
        Initialize the main window.

        Args:
            input_dir: Directory containing images
            output_dir: Directory for output labels
            class_id: YOLO class ID
            model_path: Path to SAM2 model
            device: Device to run model on
        """
        super().__init__()

        # Configuration
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.class_id = class_id
        self.model_path = model_path
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.predictor: Optional[SAM2InteractivePredictor] = None
        self.state = AnnotationState()
        self.tracking_state = TrackingState()
        self.video_predictor: Optional[VideoTrackingPredictor] = None

        # Image state
        self.current_image: Optional[np.ndarray] = None
        self.current_image_path: Optional[Path] = None
        self.image_list: List[Path] = []
        self.current_index: int = 0
        self.points_frame_index: Optional[int] = None
        self.annotated_images: Set[Path] = set()
        self.tracking_frame_map: Dict[int, Path] = {}

        # Workers
        self._model_loader: Optional[ModelLoaderWorker] = None
        self._tracking_worker: Optional[TrackingWorker] = None
        self._save_worker: Optional[SaveWorker] = None

        # Setup UI
        self.setWindowTitle("SAM2 Interactive Annotator")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        self._build_ui()
        self._setup_shortcuts()
        self._load_image_list()
        self._load_model_async()

    def _build_ui(self) -> None:
        """Build the main UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Image list
        left_panel = QGroupBox("Images")
        left_layout = QVBoxLayout(left_panel)
        self.image_list_widget = ImageListWidget()
        self.image_list_widget.image_selected.connect(self._on_image_selected)
        self.image_list_widget.image_double_clicked.connect(
            self._on_image_double_clicked
        )
        left_layout.addWidget(self.image_list_widget)
        left_panel.setMinimumWidth(200)
        left_panel.setMaximumWidth(300)
        splitter.addWidget(left_panel)

        # Center panel: Canvas
        center_panel = QGroupBox("Annotation Canvas")
        center_layout = QVBoxLayout(center_panel)
        self.canvas = ImageCanvas()
        self.canvas.left_clicked.connect(self._on_left_click)
        self.canvas.right_clicked.connect(self._on_right_click)
        self.canvas.resized.connect(self._update_display)
        center_layout.addWidget(self.canvas)
        splitter.addWidget(center_panel)

        splitter.setSizes([200, 800])
        main_layout.addWidget(splitter, stretch=1)

        # Control panel
        self.control_panel = ControlPanel()
        self.control_panel.reset_clicked.connect(self._on_reset)
        self.control_panel.undo_clicked.connect(self._on_undo)
        self.control_panel.accept_clicked.connect(self._on_accept)
        self.control_panel.skip_clicked.connect(self._on_skip)
        self.control_panel.prev_clicked.connect(self._on_prev_image)
        self.control_panel.next_clicked.connect(self._on_next_image)
        self.control_panel.mask_toggle_changed.connect(self._update_display)
        main_layout.addWidget(self.control_panel)

        # Tracking panel
        self.tracking_panel = TrackingPanel()
        self.tracking_panel.tracking_toggled.connect(self._on_toggle_tracking)
        self.tracking_panel.start_tracking.connect(self._on_start_tracking)
        self.tracking_panel.apply_results.connect(self._on_apply_results)
        self.tracking_panel.cancel_tracking.connect(self._on_cancel_tracking)
        self.tracking_panel.stop_tracking.connect(self._on_stop_tracking)
        self.tracking_panel.toggle_exclusion.connect(self._on_toggle_exclusion)
        self.tracking_panel.exclude_low_conf.connect(self._on_exclude_low_conf)
        self.tracking_panel.include_all.connect(self._on_include_all)
        self.tracking_panel.continue_batch.connect(self._on_continue_batch)
        self.tracking_panel.stop_at_batch.connect(self._on_stop_at_batch)
        main_layout.addWidget(self.tracking_panel)

        # Status bar
        status_frame = QWidget()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(5, 2, 5, 2)

        self.status_label = QLabel("Loading...")
        status_layout.addWidget(self.status_label, stretch=1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.hide()
        status_layout.addWidget(self.progress_bar)

        main_layout.addWidget(status_frame)

    def _setup_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        shortcuts = [
            ("Ctrl+Z", self._on_undo),
            ("Escape", self._on_reset),
            ("Return", self._on_accept),
            ("Right", self._on_next_image),
            ("Left", self._on_prev_image),
            ("Down", self._on_next_image),
            ("Up", self._on_prev_image),
            ("N", self._on_next_image),
            ("P", self._on_prev_image),
            ("S", self._on_skip),
            ("Space", self._toggle_mask_overlay),
            ("M", self._toggle_mask_overlay),
            ("Q", self.close),
        ]

        for key, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)

    def _load_model_async(self) -> None:
        """Load SAM2 model in background thread."""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        self._model_loader = ModelLoaderWorker(
            model_path=self.model_path,
            device=self.device,
            parent=self,
        )
        self._model_loader.progress.connect(self._on_model_progress)
        self._model_loader.finished.connect(self._on_model_loaded)
        self._model_loader.error.connect(self._on_model_error)
        self._model_loader.start()

    @pyqtSlot(str)
    def _on_model_progress(self, msg: str) -> None:
        """Handle model loading progress."""
        self.status_label.setText(msg)

    @pyqtSlot(object)
    def _on_model_loaded(self, predictor: SAM2InteractivePredictor) -> None:
        """Handle model loaded."""
        self.predictor = predictor
        self.progress_bar.hide()
        self.status_label.setText("Model loaded. Select an image to start.")

        if self.image_list:
            self._load_current_image()

    @pyqtSlot(str)
    def _on_model_error(self, error: str) -> None:
        """Handle model loading error."""
        self.progress_bar.hide()
        QMessageBox.critical(self, "Model Error", f"Failed to load model:\n{error}")

    def _load_image_list(self) -> None:
        """Load list of images from input directory."""
        def natural_sort_key(path: Path):
            return [
                int(c) if c.isdigit() else c.lower()
                for c in re.split(r"(\d+)", path.name)
            ]

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        self.image_list = sorted(
            [f for f in self.input_dir.iterdir() if f.suffix.lower() in extensions],
            key=natural_sort_key,
        )

        # Check for existing annotations
        labels_dir = self.output_dir / "labels"
        for img_path in self.image_list:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                label_path = self.output_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.annotated_images.add(img_path)

        # Update list widget
        self.image_list_widget.set_images(self.image_list)
        self.image_list_widget.set_annotated(self.annotated_images)

        if not self.image_list:
            QMessageBox.warning(
                self, "No Images", f"No images found in {self.input_dir}"
            )

    @pyqtSlot(int)
    def _on_image_selected(self, index: int) -> None:
        """Handle image selection from list."""
        self.current_index = index
        self._load_current_image()

    @pyqtSlot(int)
    def _on_image_double_clicked(self, index: int) -> None:
        """Handle image double-click (toggle exclusion in tracking mode)."""
        if not self.tracking_state.is_tracking_initialized:
            return

        if index not in self.tracking_state.tracking_results:
            return

        self.tracking_state.toggle_frame_exclusion(index)
        self._update_image_list()
        self._update_display()

    def _load_current_image(self) -> None:
        """Load the current image for annotation."""
        if not self.image_list or self.predictor is None:
            return

        self.current_image_path = self.image_list[self.current_index]

        # Load image
        image = cv2.imread(str(self.current_image_path))
        if image is None:
            QMessageBox.critical(
                self, "Error", f"Failed to load: {self.current_image_path}"
            )
            return

        # Convert to RGB
        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.canvas.set_image(self.current_image)

        # Reset annotation state
        self.state.reset()
        self.points_frame_index = None
        self.predictor.reset_mask_state()

        # Set image in predictor
        self.status_label.setText("Computing image embedding...")
        QApplication.processEvents()

        try:
            self.predictor.set_image(self.current_image)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process image: {e}")
            return

        # Update display
        self._update_display()
        self._update_status()
        self._update_image_list()

    @pyqtSlot(int, int)
    def _on_left_click(self, x: int, y: int) -> None:
        """Handle left click: add foreground point."""
        if self.points_frame_index is None:
            self.points_frame_index = self.current_index

        self.state.add_foreground_point(x, y)
        self._run_segmentation()

    @pyqtSlot(int, int)
    def _on_right_click(self, x: int, y: int) -> None:
        """Handle right click: add background point."""
        if self.points_frame_index is None:
            self.points_frame_index = self.current_index

        self.state.add_background_point(x, y)
        self._run_segmentation()

    def _run_segmentation(self) -> None:
        """Run segmentation with current points."""
        if not self.state.has_points() or self.predictor is None:
            return

        try:
            use_prev = self.state.current_mask is not None
            mask, iou, _ = self.predictor.predict(
                self.state.foreground_points,
                self.state.background_points,
                use_previous_mask=use_prev,
            )

            self.state.current_mask = mask
            self.state.current_iou = iou

            self._update_display()
            self._update_point_info()
            self._update_tracking_ui()

        except Exception as e:
            self.status_label.setText(f"Segmentation failed: {e}")

    @pyqtSlot()
    def _update_display(self) -> None:
        """Update canvas display."""
        if self.current_image is None:
            return

        display = self.current_image.copy()
        img_h, img_w = display.shape[:2]

        # Determine mask to display
        display_mask = None
        mask_color = (0, 255, 0)

        if self.control_panel.is_mask_visible():
            if (
                self.tracking_state.is_tracking_initialized
                and self.current_index in self.tracking_state.tracking_results
            ):
                result = self.tracking_state.tracking_results[self.current_index]
                display_mask = result.mask
                if result.is_low_confidence:
                    mask_color = (255, 200, 0)
            elif self.state.current_mask is not None:
                display_mask = self.state.current_mask

        # Draw mask overlay
        if display_mask is not None:
            display = draw_mask_overlay(display, display_mask, mask_color)

        # Draw points on current frame
        if self.points_frame_index is None or self.current_index == self.points_frame_index:
            display = draw_points(
                display,
                self.state.foreground_points,
                self.state.background_points,
            )

        # Draw saved annotation bbox (orange)
        if (
            self.state.current_mask is None
            and self.current_image_path in self.annotated_images
        ):
            label_path = self.output_dir / "labels" / f"{self.current_image_path.stem}.txt"
            if not label_path.exists():
                label_path = self.output_dir / f"{self.current_image_path.stem}.txt"
            if label_path.exists():
                labels = read_yolo_label(str(label_path))
                for class_id, x_c, y_c, w, h in labels:
                    x1, y1, x2, y2 = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)

        self.canvas.update_display(display)

    def _update_point_info(self) -> None:
        """Update point count display."""
        fg, bg = self.state.get_point_counts()
        self.control_panel.update_point_info(fg, bg)

        if self.state.current_mask is not None:
            self.control_panel.update_iou(self.state.current_iou)
        else:
            self.control_panel.update_iou(None)

    def _update_status(self) -> None:
        """Update status bar."""
        annotated = len(self.annotated_images)
        total = len(self.image_list)
        current = self.current_index + 1 if self.image_list else 0

        status = f"Image {current}/{total} | Annotated: {annotated}/{total}"
        if self.current_image_path:
            status += f" | Current: {self.current_image_path.name}"
        status += f" | Class ID: {self.class_id}"

        self.status_label.setText(status)

    def _update_image_list(self) -> None:
        """Update image list display."""
        self.image_list_widget.set_annotated(self.annotated_images)
        self.image_list_widget.set_tracking_state(
            self.tracking_state.tracking_results,
            self.tracking_state.excluded_frames,
            self.tracking_state.low_confidence_frames,
        )
        self.image_list_widget.select_index(self.current_index)

    def _update_tracking_ui(self) -> None:
        """Update tracking panel UI state."""
        self.tracking_panel.update_ui_state(
            is_enabled=self.tracking_state.is_tracking_mode,
            has_mask=self.state.current_mask is not None,
            has_results=self.tracking_state.is_tracking_initialized,
            is_processing=self.tracking_state.is_processing,
            is_paused=self.tracking_state.is_paused_between_batches,
        )

        if self.tracking_state.is_tracking_initialized:
            low_conf = len(self.tracking_state.low_confidence_frames)
            self.tracking_panel.update_low_conf_warning(low_conf)

            included, excluded, low_incl = self.tracking_state.get_exclusion_stats()
            self.tracking_panel.update_exclusion_stats(included, excluded, low_incl)

    @pyqtSlot()
    def _on_undo(self) -> None:
        """Handle undo action."""
        if self.state.undo():
            if not self.state.has_points():
                self.predictor.reset_mask_state()
            self._update_display()
            self._update_point_info()
            self.status_label.setText("Undo: last point removed")
        else:
            self.status_label.setText("Nothing to undo")

    @pyqtSlot()
    def _on_reset(self) -> None:
        """Handle reset action."""
        self.state.reset()
        self.points_frame_index = None
        if self.predictor:
            self.predictor.reset_mask_state()
        self._update_display()
        self._update_point_info()
        self.status_label.setText("Points cleared. Click to start over.")

    @pyqtSlot()
    def _on_accept(self) -> None:
        """Handle accept and save action."""
        if self.state.current_mask is None:
            QMessageBox.warning(self, "No Mask", "No mask to save. Add points first.")
            return

        bbox = mask_to_bbox(self.state.current_mask, use_contour=True)
        if bbox is None:
            QMessageBox.warning(self, "Invalid Mask", "Could not extract bounding box.")
            return

        x_min, y_min, x_max, y_max = bbox
        img_h, img_w = self.current_image.shape[:2]

        # Convert to YOLO format
        yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

        # Save label file
        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{self.current_image_path.stem}.txt"
        write_yolo_label(str(label_path), self.class_id, yolo_bbox)

        # Save mask file
        masks_dir = self.output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_path = masks_dir / f"{self.current_image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), self.state.current_mask.astype(np.uint8) * 255)

        # Copy original image
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        dst_image_path = images_dir / self.current_image_path.name
        if not dst_image_path.exists():
            shutil.copy2(self.current_image_path, dst_image_path)

        # Mark as annotated
        self.annotated_images.add(self.current_image_path)
        self.status_label.setText(f"Saved: {label_path.name} + mask + image")

        self._on_next_image()

    @pyqtSlot()
    def _on_skip(self) -> None:
        """Handle skip action."""
        self._on_next_image()

    @pyqtSlot()
    def _on_next_image(self) -> None:
        """Navigate to next image."""
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self._load_current_image()

    @pyqtSlot()
    def _on_prev_image(self) -> None:
        """Navigate to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()

    @pyqtSlot()
    def _toggle_mask_overlay(self) -> None:
        """Toggle mask overlay visibility."""
        visible = self.control_panel.is_mask_visible()
        self.control_panel.set_mask_visible(not visible)

    # === Tracking Methods ===

    @pyqtSlot(bool)
    def _on_toggle_tracking(self, enabled: bool) -> None:
        """Handle tracking mode toggle."""
        if enabled:
            self.tracking_state.enable_tracking()
            self.tracking_panel.update_status("Tracking: Ready (add points)")
        else:
            self._on_cancel_tracking()
        self._update_tracking_ui()

    @pyqtSlot()
    def _on_start_tracking(self) -> None:
        """Start tracking from current mask."""
        if not self.tracking_state.is_tracking_mode:
            return

        if self.state.current_mask is None:
            QMessageBox.warning(
                self, "No Mask",
                "Add points to create a mask first, then start tracking."
            )
            return

        self.tracking_panel.update_status("Tracking: Initializing...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.tracking_state.set_processing(True)
        self.tracking_state.clear_stop_request()
        self._update_tracking_ui()

        # Initialize video predictor if needed
        if self.video_predictor is None:
            self.video_predictor = VideoTrackingPredictor(
                model_path=self.model_path,
                device=self.device,
            )
            self.video_predictor.load_model(
                progress_callback=lambda msg: self.tracking_panel.update_status(f"Tracking: {msg}")
            )

        # Start tracking worker
        self._tracking_worker = TrackingWorker(
            video_predictor=self.video_predictor,
            image_list=self.image_list,
            start_frame=self.current_index,
            foreground_points=self.state.foreground_points,
            background_points=self.state.background_points,
            obj_id=self.tracking_state.current_obj_id,
            parent=self,
        )
        self._tracking_worker.progress.connect(self._on_tracking_progress)
        self._tracking_worker.finished.connect(self._on_tracking_complete)
        self._tracking_worker.error.connect(self._on_tracking_error)
        self._tracking_worker.status.connect(
            lambda msg: self.tracking_panel.update_status(f"Tracking: {msg}")
        )
        self._tracking_worker.start()

    @pyqtSlot(int, int, int, object)
    def _on_tracking_progress(
        self, current: int, total: int, frame_idx: int, mask: np.ndarray
    ) -> None:
        """Handle tracking progress update."""
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.tracking_panel.update_status(f"Tracking: {current}/{total} frames")

        # Update display to show current tracking result
        if frame_idx < len(self.image_list):
            self.current_index = frame_idx
            self.current_image_path = self.image_list[frame_idx]

            img = cv2.imread(str(self.current_image_path))
            if img is not None:
                self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.canvas.set_image(self.current_image)

                temp_result = TrackingResult.from_mask(
                    mask,
                    reference_area=self.video_predictor.reference_mask_area,
                )
                self.tracking_state.tracking_results[frame_idx] = temp_result
                self.tracking_state.is_tracking_initialized = True
                self.tracking_frame_map[frame_idx] = self.current_image_path

                self._update_display()
                self._update_image_list()

    @pyqtSlot(dict)
    def _on_tracking_complete(self, results: Dict[int, TrackingResult]) -> None:
        """Handle tracking completion."""
        self.tracking_state.set_processing(False)
        self.tracking_state.set_tracking_results(results)
        self.progress_bar.hide()

        low_conf = len(self.tracking_state.low_confidence_frames)
        total = len(results)

        status = f"Tracking: Complete ({total} frames"
        if low_conf > 0:
            status += f", {low_conf} warnings)"
        else:
            status += ")"

        self.tracking_panel.update_status(status)
        self._update_tracking_ui()
        self._update_image_list()
        self._update_display()

        self.status_label.setText("Tracking complete. Review and click 'Apply' to save.")

    @pyqtSlot(str)
    def _on_tracking_error(self, error: str) -> None:
        """Handle tracking error."""
        self.progress_bar.hide()
        self.tracking_state.set_processing(False)
        self.tracking_panel.update_status("Tracking: Failed")
        self._update_tracking_ui()

        QMessageBox.critical(self, "Tracking Error", f"Tracking failed:\n{error}")

    @pyqtSlot()
    def _on_apply_results(self) -> None:
        """Apply tracking results and save annotations."""
        if not self.tracking_state.is_tracking_initialized:
            return

        included, excluded, low_incl = self.tracking_state.get_exclusion_stats()

        msg = f"Save annotations for {included} frames?"
        if excluded > 0:
            msg += f"\n\n({excluded} frames excluded)"
        if low_incl > 0:
            msg += f"\nWarning: {low_incl} low confidence frames included"

        result = QMessageBox.question(
            self, "Confirm Save", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if result != QMessageBox.StandardButton.Yes:
            return

        self.status_label.setText("Saving annotations...")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        # Filter excluded frames
        included_results = {
            idx: res for idx, res in self.tracking_state.tracking_results.items()
            if not self.tracking_state.is_frame_excluded(idx)
        }
        included_map = {
            idx: path for idx, path in self.tracking_frame_map.items()
            if not self.tracking_state.is_frame_excluded(idx)
        }

        self._save_worker = SaveWorker(
            tracking_results=included_results,
            frame_map=included_map,
            output_dir=str(self.output_dir),
            class_id=self.class_id,
            copy_images=True,
            parent=self,
        )
        self._save_worker.finished.connect(
            lambda r: self._on_save_complete(r, excluded)
        )
        self._save_worker.error.connect(self._on_save_error)
        self._save_worker.start()

    def _on_save_complete(self, result, excluded_count: int) -> None:
        """Handle save completion."""
        self.progress_bar.hide()

        for idx in self.tracking_state.tracking_results.keys():
            if idx in self.tracking_frame_map:
                if not self.tracking_state.is_frame_excluded(idx):
                    self.annotated_images.add(self.tracking_frame_map[idx])

        self._update_image_list()
        self._on_cancel_tracking()

        summary = f"Saved {result.successful}/{result.total_images} annotations"
        if excluded_count > 0:
            summary += f" ({excluded_count} excluded)"

        QMessageBox.information(self, "Save Complete", result.summary())
        self.status_label.setText(summary)

    @pyqtSlot(str)
    def _on_save_error(self, error: str) -> None:
        """Handle save error."""
        self.progress_bar.hide()
        QMessageBox.critical(self, "Save Error", f"Save failed:\n{error}")

    @pyqtSlot()
    def _on_cancel_tracking(self) -> None:
        """Cancel tracking and reset state."""
        self.tracking_state.disable_tracking()
        self.tracking_panel.set_tracking_enabled(False)
        self.tracking_panel.update_status("Tracking: OFF")

        self.state.reset()
        if self.predictor:
            self.predictor.reset_mask_state()

        if self.video_predictor:
            self.video_predictor.reset()

        self.tracking_frame_map.clear()
        self._update_tracking_ui()
        self._update_image_list()
        self._update_display()

    @pyqtSlot()
    def _on_stop_tracking(self) -> None:
        """Stop tracking in progress."""
        if self._tracking_worker:
            self._tracking_worker.request_stop()

    @pyqtSlot()
    def _on_toggle_exclusion(self) -> None:
        """Toggle exclusion of selected frame."""
        if not self.tracking_state.is_tracking_initialized:
            return

        if self.current_index not in self.tracking_state.tracking_results:
            self.status_label.setText("Selected frame has no tracking results")
            return

        is_excluded = self.tracking_state.toggle_frame_exclusion(self.current_index)
        self._update_image_list()
        self._update_display()
        self._update_tracking_ui()

        status = "excluded from" if is_excluded else "included in"
        self.status_label.setText(f"Frame {self.current_index} {status} training")

    @pyqtSlot()
    def _on_exclude_low_conf(self) -> None:
        """Exclude all low confidence frames."""
        count = self.tracking_state.exclude_all_low_confidence()
        self._update_image_list()
        self._update_display()
        self._update_tracking_ui()

        if count > 0:
            self.status_label.setText(f"Excluded {count} low confidence frames")
        else:
            self.status_label.setText("No additional frames to exclude")

    @pyqtSlot()
    def _on_include_all(self) -> None:
        """Include all frames."""
        count = self.tracking_state.include_all()
        self._update_image_list()
        self._update_display()
        self._update_tracking_ui()

        if count > 0:
            self.status_label.setText(f"Included {count} previously excluded frames")
        else:
            self.status_label.setText("All frames already included")

    @pyqtSlot()
    def _on_continue_batch(self) -> None:
        """Continue to next batch."""
        self.tracking_panel.hide_batch_pause()
        self.tracking_state.resume_from_pause()
        self._update_tracking_ui()

    @pyqtSlot()
    def _on_stop_at_batch(self) -> None:
        """Stop at current batch."""
        self.tracking_panel.hide_batch_pause()
        self.tracking_state.resume_from_pause()
        self.tracking_state.request_stop()

    def closeEvent(self, event) -> None:
        """Handle window close."""
        if self.video_predictor:
            self.video_predictor.reset()
        event.accept()
