#!/usr/bin/env python3
"""
SAM2 Interactive Annotation Tool

A Tkinter-based GUI application for interactive object annotation using SAM2.
Features:
- Point-click based segmentation (foreground/background points)
- Real-time mask preview with overlay
- Undo/Reset functionality for refinement
- Batch image navigation
- YOLO format output

Usage:
    python sam2_interactive_app.py --input-dir images/ --output-dir labels/ --class-id 0

Controls:
    - Left click: Add foreground point (include in mask)
    - Right click: Add background point (exclude from mask)
    - Ctrl+Z: Undo last point
    - Escape: Reset all points
    - Enter: Accept and save annotation
    - Arrow keys: Navigate between images
"""

import argparse
import shutil
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
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
    BatchTrackingProgress,
)
from gui_framework import AppTheme


# =============================================================================
# SAM2 Interactive Predictor
# =============================================================================


class SAM2InteractivePredictor:
    """
    Wrapper for SAM2 Image Predictor with point-based segmentation.

    Provides interactive segmentation using foreground/background points.
    Supports iterative refinement using previous mask as input.
    """

    def __init__(self, model_path: str = "sam2_b.pt", device: str = "cuda"):
        """
        Initialize SAM2 predictor.

        Args:
            model_path: Path to SAM2 model checkpoint
            device: Device to run model on ("cuda" or "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.predictor = None
        self.model = None
        self.current_image: Optional[np.ndarray] = None
        self.low_res_mask: Optional[np.ndarray] = None

    def load_model(self, progress_callback: Optional[callable] = None) -> None:
        """
        Load SAM2 model.

        Args:
            progress_callback: Optional callback for progress updates
        """
        if self.predictor is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Install with:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        if progress_callback:
            progress_callback("Loading SAM2 model...")

        # Determine model config based on model path
        model_path_lower = self.model_path.lower()
        if "sam2.1" in model_path_lower or "sam2_1" in model_path_lower:
            if "base" in model_path_lower or "_b" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "large" in model_path_lower or "_l" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            elif "small" in model_path_lower or "_s" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
            elif "tiny" in model_path_lower or "_t" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        else:
            if "sam2_b" in model_path_lower or "base" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            elif "sam2_l" in model_path_lower or "large" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            elif "sam2_t" in model_path_lower or "tiny" in model_path_lower:
                model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            else:
                model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

        self.model = build_sam2(model_cfg, self.model_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

        if progress_callback:
            progress_callback("SAM2 model loaded successfully")

    def set_image(self, image_rgb: np.ndarray) -> None:
        """
        Set image for segmentation.

        Args:
            image_rgb: RGB image array (H, W, 3)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self.predictor.set_image(image_rgb)
        self.current_image = image_rgb
        self.low_res_mask = None  # Reset for new image

    def predict(
        self,
        foreground_points: List[Tuple[int, int]],
        background_points: List[Tuple[int, int]],
        use_previous_mask: bool = False,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Predict mask from points.

        Args:
            foreground_points: List of (x, y) foreground points
            background_points: List of (x, y) background points
            use_previous_mask: Whether to use previous mask for refinement

        Returns:
            Tuple of (best_mask, iou_score, low_res_mask)
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if not foreground_points and not background_points:
            raise ValueError("At least one point is required")

        # Prepare points and labels
        all_points = list(foreground_points) + list(background_points)
        all_labels = [1] * len(foreground_points) + [0] * len(background_points)

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)

        # Use previous mask for refinement if available
        mask_input = self.low_res_mask if use_previous_mask else None

        masks, iou_predictions, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            multimask_output=True,
        )

        # Select best mask (highest IoU)
        best_idx = int(np.argmax(iou_predictions))
        # Convert to boolean for use as NumPy index (SAM2 returns float mask)
        best_mask = masks[best_idx].astype(bool)
        best_iou = float(iou_predictions[best_idx])

        # Store low_res_mask for future refinement
        self.low_res_mask = low_res_masks[best_idx : best_idx + 1]

        return best_mask, best_iou, self.low_res_mask

    def reset_mask_state(self) -> None:
        """Reset the stored low_res_mask for new annotation."""
        self.low_res_mask = None


# =============================================================================
# Annotation State
# =============================================================================


@dataclass
class AnnotationState:
    """
    Manages annotation state with undo history.

    Tracks foreground/background points, current mask, and provides
    undo functionality for iterative refinement.
    """

    foreground_points: List[Tuple[int, int]] = field(default_factory=list)
    background_points: List[Tuple[int, int]] = field(default_factory=list)
    current_mask: Optional[np.ndarray] = None
    current_iou: float = 0.0
    history: List[Dict] = field(default_factory=list)
    max_history: int = 10

    def add_foreground_point(self, x: int, y: int) -> None:
        """Add foreground point and save to history."""
        self._save_to_history()
        self.foreground_points.append((x, y))

    def add_background_point(self, x: int, y: int) -> None:
        """Add background point and save to history."""
        self._save_to_history()
        self.background_points.append((x, y))

    def _save_to_history(self) -> None:
        """Save current state to history."""
        state = {
            "fg": self.foreground_points.copy(),
            "bg": self.background_points.copy(),
            "mask": self.current_mask.copy() if self.current_mask is not None else None,
            "iou": self.current_iou,
        }
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    def undo(self) -> bool:
        """
        Undo last point addition.

        Returns:
            True if undo was successful, False if no history
        """
        if not self.history:
            return False

        state = self.history.pop()
        self.foreground_points = state["fg"]
        self.background_points = state["bg"]
        self.current_mask = state["mask"]
        self.current_iou = state["iou"]
        return True

    def reset(self) -> None:
        """Reset all points and mask."""
        self.foreground_points.clear()
        self.background_points.clear()
        self.current_mask = None
        self.current_iou = 0.0
        self.history.clear()

    def has_points(self) -> bool:
        """Check if any points exist."""
        return len(self.foreground_points) > 0 or len(self.background_points) > 0

    def get_point_counts(self) -> Tuple[int, int]:
        """Get counts of foreground and background points."""
        return len(self.foreground_points), len(self.background_points)


# =============================================================================
# Tracking State
# =============================================================================


@dataclass
class TrackingState:
    """
    Manages video tracking state for sequential annotation.

    Tracks tracking mode status, results, low confidence frames,
    and excluded frames for training data filtering.
    """

    is_tracking_mode: bool = False
    is_tracking_initialized: bool = False
    tracking_results: Dict[int, TrackingResult] = field(default_factory=dict)
    low_confidence_frames: List[int] = field(default_factory=list)
    confirmed_frames: Set[int] = field(default_factory=set)
    excluded_frames: Set[int] = field(default_factory=set)
    current_obj_id: int = 1

    # Stop/pause control fields
    is_processing: bool = False
    stop_requested: bool = False
    is_paused_between_batches: bool = False
    current_batch_index: int = 0
    total_batches: int = 0

    def enable_tracking(self) -> None:
        """Enable tracking mode."""
        self.is_tracking_mode = True

    def disable_tracking(self) -> None:
        """Disable tracking mode and clear results."""
        self.is_tracking_mode = False
        self.is_tracking_initialized = False
        self.tracking_results.clear()
        self.low_confidence_frames.clear()
        self.confirmed_frames.clear()
        self.excluded_frames.clear()

    def set_tracking_results(
        self, results: Dict[int, TrackingResult]
    ) -> None:
        """Set tracking results and identify low confidence frames."""
        self.tracking_results = results
        self.is_tracking_initialized = True
        self.low_confidence_frames = [
            idx for idx, result in results.items()
            if result.is_low_confidence
        ]
        # Clear excluded frames when new results are set
        self.excluded_frames.clear()

    def get_frame_status(self, frame_idx: int) -> str:
        """
        Get status indicator for a frame.

        Returns:
            Status string: 'confirmed', 'excluded', 'tracked', 'low_confidence', or 'pending'
        """
        if frame_idx in self.confirmed_frames:
            return "confirmed"
        if frame_idx in self.excluded_frames:
            return "excluded"
        if frame_idx in self.low_confidence_frames:
            return "low_confidence"
        if frame_idx in self.tracking_results:
            return "tracked"
        return "pending"

    def toggle_frame_exclusion(self, frame_idx: int) -> bool:
        """
        Toggle exclusion state of a frame.

        Args:
            frame_idx: Frame index to toggle

        Returns:
            True if frame is now excluded, False if now included
        """
        if frame_idx in self.excluded_frames:
            self.excluded_frames.discard(frame_idx)
            return False
        else:
            self.excluded_frames.add(frame_idx)
            return True

    def is_frame_excluded(self, frame_idx: int) -> bool:
        """Check if frame is excluded."""
        return frame_idx in self.excluded_frames

    def exclude_all_low_confidence(self) -> int:
        """
        Exclude all low confidence frames.

        Returns:
            Number of frames excluded
        """
        count = 0
        for frame_idx in self.low_confidence_frames:
            if frame_idx not in self.excluded_frames:
                self.excluded_frames.add(frame_idx)
                count += 1
        return count

    def include_all(self) -> int:
        """
        Include all frames (clear exclusions).

        Returns:
            Number of frames that were excluded
        """
        count = len(self.excluded_frames)
        self.excluded_frames.clear()
        return count

    def get_included_frames(self) -> List[int]:
        """Get list of frame indices that are not excluded."""
        return [
            idx for idx in self.tracking_results.keys()
            if idx not in self.excluded_frames
        ]

    def get_exclusion_stats(self) -> Tuple[int, int, int]:
        """
        Get exclusion statistics.

        Returns:
            Tuple of (included_count, excluded_count, low_confidence_included_count)
        """
        total = len(self.tracking_results)
        excluded = len(self.excluded_frames)
        included = total - excluded

        # Count low confidence frames that are still included
        low_conf_included = sum(
            1 for idx in self.low_confidence_frames
            if idx not in self.excluded_frames
        )

        return (included, excluded, low_conf_included)

    # Stop/pause control methods
    def request_stop(self) -> None:
        """Request processing to stop."""
        self.stop_requested = True

    def clear_stop_request(self) -> None:
        """Clear stop request flag."""
        self.stop_requested = False

    def is_stop_requested(self) -> bool:
        """Check if stop has been requested."""
        return self.stop_requested

    def set_processing(self, processing: bool) -> None:
        """Set processing state."""
        self.is_processing = processing
        if not processing:
            self.stop_requested = False
            self.is_paused_between_batches = False

    def pause_for_batch_review(self, batch_idx: int, total_batches: int) -> None:
        """Set state for batch review pause."""
        self.is_paused_between_batches = True
        self.current_batch_index = batch_idx
        self.total_batches = total_batches

    def resume_from_pause(self) -> None:
        """Resume from batch review pause."""
        self.is_paused_between_batches = False


# =============================================================================
# Main Application
# =============================================================================


class SAM2AnnotationApp:
    """
    Main Tkinter application for SAM2 interactive annotation.

    Provides GUI for point-based segmentation with real-time preview,
    undo/reset functionality, and batch image processing.
    """

    def __init__(
        self,
        root: tk.Tk,
        input_dir: str,
        output_dir: str,
        class_id: int = 0,
        model_path: str = "sam2_b.pt",
        device: str = "cuda",
    ):
        """
        Initialize application.

        Args:
            root: Tkinter root window
            input_dir: Directory containing images
            output_dir: Directory for output labels
            class_id: YOLO class ID
            model_path: Path to SAM2 model
            device: Device to run model on
        """
        self.root = root
        self.root.title("SAM2 Interactive Annotator")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

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
        self.current_image: Optional[np.ndarray] = None
        self.current_image_path: Optional[Path] = None
        self.image_list: List[Path] = []
        self.current_index: int = 0
        self.points_frame_index: Optional[int] = None  # Frame where points were added
        self.annotated_images: set = set()

        # Tracking mode state
        self.tracking_state = TrackingState()
        self.video_predictor: Optional[VideoTrackingPredictor] = None
        self.tracking_frame_map: Dict[int, Path] = {}
        self._batch_continue_event = threading.Event()

        # Display state
        self.display_image: Optional[np.ndarray] = None
        self.scale_factor: float = 1.0
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.show_mask_overlay: bool = True

        # Build GUI
        self._build_gui()

        # Load images
        self._load_image_list()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Bind keyboard shortcuts
        self._bind_shortcuts()

        # Load model in background
        self._load_model_async()

    def _build_gui(self):
        """Build the GUI layout."""
        # Configure grid
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(1, weight=1)  # Main content row

        # === Left Panel: Image List ===
        left_frame = ttk.Frame(self.root, width=200)
        left_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        left_frame.grid_propagate(False)

        ttk.Label(left_frame, text="Images", font=("TkDefaultFont", 10, "bold")).pack(
            pady=(0, 5)
        )

        # Image listbox with scrollbar
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.image_listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            font=("TkDefaultFont", 9),
        )
        self.image_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.image_listbox.yview)

        self.image_listbox.bind("<<ListboxSelect>>", self._on_image_select)
        self.image_listbox.bind("<Double-Button-1>", self._on_listbox_double_click)

        # === Main Panel: Canvas ===
        main_frame = ttk.Frame(self.root)
        main_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, bg="gray20", cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # === Control Panel ===
        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Point info
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(side=tk.LEFT, padx=10)

        self.point_label = ttk.Label(info_frame, text="Points: FG=0, BG=0")
        self.point_label.pack(side=tk.LEFT, padx=(0, 15))

        self.iou_label = ttk.Label(info_frame, text="IoU: --")
        self.iou_label.pack(side=tk.LEFT)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=20)

        ttk.Button(button_frame, text="Reset (Esc)", command=self._on_reset).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(button_frame, text="Undo (Ctrl+Z)", command=self._on_undo).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            button_frame, text="Accept & Save (Enter)", command=self._on_accept
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Skip (S)", command=self._on_skip).pack(
            side=tk.LEFT, padx=2
        )

        # Navigation buttons
        nav_frame = ttk.Frame(control_frame)
        nav_frame.pack(side=tk.LEFT, padx=20)

        ttk.Button(nav_frame, text="< Prev", command=self._on_prev_image).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(nav_frame, text="Next >", command=self._on_next_image).pack(
            side=tk.LEFT, padx=2
        )

        # Toggle mask overlay
        self.show_mask_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            control_frame,
            text="Show Mask",
            variable=self.show_mask_var,
            command=self._update_display,
        ).pack(side=tk.RIGHT, padx=10)

        # === Tracking Mode Controls ===
        tracking_frame = ttk.LabelFrame(self.root, text="Tracking Mode", padding=5)
        tracking_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 5))

        # Top row: main tracking controls
        tracking_row1 = ttk.Frame(tracking_frame)
        tracking_row1.pack(fill=tk.X, pady=(0, 3))

        # Tracking mode toggle
        self.tracking_mode_var = tk.BooleanVar(value=False)
        self.tracking_toggle = ttk.Checkbutton(
            tracking_row1,
            text="Enable Tracking Mode",
            variable=self.tracking_mode_var,
            command=self._on_toggle_tracking_mode,
        )
        self.tracking_toggle.pack(side=tk.LEFT, padx=5)

        # Start tracking button
        self.start_tracking_btn = ttk.Button(
            tracking_row1,
            text="Start Tracking",
            command=self._on_start_tracking,
            state=tk.DISABLED,
        )
        self.start_tracking_btn.pack(side=tk.LEFT, padx=5)

        # Apply all button
        self.apply_all_btn = ttk.Button(
            tracking_row1,
            text="Apply",
            command=self._on_apply_tracking_results,
            state=tk.DISABLED,
        )
        self.apply_all_btn.pack(side=tk.LEFT, padx=5)

        # Cancel tracking button
        self.cancel_tracking_btn = ttk.Button(
            tracking_row1,
            text="Cancel Tracking",
            command=self._on_cancel_tracking,
            state=tk.DISABLED,
        )
        self.cancel_tracking_btn.pack(side=tk.LEFT, padx=5)

        # Stop tracking button (visible only during processing)
        self.stop_tracking_btn = ttk.Button(
            tracking_row1,
            text="Stop",
            command=self._on_stop_tracking,
            state=tk.DISABLED,
        )
        self.stop_tracking_btn.pack(side=tk.LEFT, padx=5)

        # Tracking status label
        self.tracking_status_var = tk.StringVar(value="Tracking: OFF")
        ttk.Label(
            tracking_row1,
            textvariable=self.tracking_status_var,
            font=("TkDefaultFont", 9),
        ).pack(side=tk.RIGHT, padx=10)

        # Low confidence warning
        self.low_conf_label = ttk.Label(
            tracking_row1,
            text="",
            foreground="orange",
            font=("TkDefaultFont", 9, "bold"),
        )
        self.low_conf_label.pack(side=tk.RIGHT, padx=5)

        # Bottom row: exclusion controls (initially hidden)
        self.exclusion_frame = ttk.Frame(tracking_frame)
        # Will be shown after tracking completes

        # Exclusion stats label
        self.exclusion_stats_var = tk.StringVar(value="")
        self.exclusion_stats_label = ttk.Label(
            self.exclusion_frame,
            textvariable=self.exclusion_stats_var,
            font=("TkDefaultFont", 9),
        )
        self.exclusion_stats_label.pack(side=tk.LEFT, padx=5)

        # Toggle Selected button
        self.toggle_selected_btn = ttk.Button(
            self.exclusion_frame,
            text="Toggle Selected",
            command=self._on_toggle_frame_exclusion,
            state=tk.DISABLED,
        )
        self.toggle_selected_btn.pack(side=tk.LEFT, padx=5)

        # Exclude All Low-Confidence button
        self.exclude_low_conf_btn = ttk.Button(
            self.exclusion_frame,
            text="Exclude All Low-Confidence",
            command=self._on_exclude_all_low_confidence,
            state=tk.DISABLED,
        )
        self.exclude_low_conf_btn.pack(side=tk.LEFT, padx=5)

        # Include All button
        self.include_all_btn = ttk.Button(
            self.exclusion_frame,
            text="Include All",
            command=self._on_include_all,
            state=tk.DISABLED,
        )
        self.include_all_btn.pack(side=tk.LEFT, padx=5)

        # Batch pause frame (shown between batches)
        self.batch_pause_frame = ttk.Frame(tracking_frame)
        # Will be shown/hidden dynamically during batch processing

        # Batch info label
        self.batch_info_var = tk.StringVar(value="")
        ttk.Label(
            self.batch_pause_frame,
            textvariable=self.batch_info_var,
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side=tk.LEFT, padx=5)

        # Batch summary label
        self.batch_summary_var = tk.StringVar(value="")
        ttk.Label(
            self.batch_pause_frame,
            textvariable=self.batch_summary_var,
            font=("TkDefaultFont", 9),
        ).pack(side=tk.LEFT, padx=5)

        # Continue to Next Batch button
        self.continue_batch_btn = ttk.Button(
            self.batch_pause_frame,
            text="Continue to Next Batch",
            command=self._on_continue_batch,
        )
        self.continue_batch_btn.pack(side=tk.LEFT, padx=10)

        # Stop Here button
        self.stop_at_batch_btn = ttk.Button(
            self.batch_pause_frame,
            text="Stop Here",
            command=self._on_stop_at_batch,
        )
        self.stop_at_batch_btn.pack(side=tk.LEFT, padx=5)

        # === Status Bar ===
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=2)

        self.status_var = tk.StringVar(value="Loading...")
        self.status_label = ttk.Label(
            status_frame, textvariable=self.status_var, relief="sunken", padding=(5, 2)
        )
        self.status_label.pack(fill=tk.X)

        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100
        )

    def _bind_shortcuts(self):
        """Bind keyboard shortcuts."""
        self.root.bind("<Control-z>", lambda e: self._on_undo())
        self.root.bind("<Escape>", lambda e: self._on_reset())
        self.root.bind("<Return>", lambda e: self._on_accept())
        self.root.bind("<Right>", lambda e: self._on_next_image())
        self.root.bind("<Left>", lambda e: self._on_prev_image())
        self.root.bind("<Down>", lambda e: self._on_next_image())
        self.root.bind("<Up>", lambda e: self._on_prev_image())
        self.root.bind("<n>", lambda e: self._on_next_image())
        self.root.bind("<p>", lambda e: self._on_prev_image())
        self.root.bind("<s>", lambda e: self._on_skip())
        self.root.bind("<space>", lambda e: self._toggle_mask_overlay())
        self.root.bind("<m>", lambda e: self._toggle_mask_overlay())

    def _load_model_async(self):
        """Load SAM2 model in background thread."""
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        def load_task():
            try:
                self.predictor = SAM2InteractivePredictor(
                    model_path=self.model_path, device=self.device
                )
                self.predictor.load_model(
                    progress_callback=lambda msg: self.root.after(
                        0, lambda: self.status_var.set(msg)
                    )
                )
                self.root.after(0, self._on_model_loaded)
            except Exception as e:
                self.root.after(
                    0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}")
                )

        thread = threading.Thread(target=load_task, daemon=True)
        thread.start()

    def _on_model_loaded(self):
        """Called when model is loaded."""
        self.progress_bar.pack_forget()
        self.status_var.set("Model loaded. Select an image to start annotating.")

        # Load first image
        if self.image_list:
            self._load_current_image()

    def _load_image_list(self):
        """Load list of images from input directory."""
        import re

        def natural_sort_key(path):
            """Sort key for natural/alphanumeric ordering."""
            return [
                int(c) if c.isdigit() else c.lower()
                for c in re.split(r"(\d+)", path.name)
            ]

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        self.image_list = sorted(
            [
                f
                for f in self.input_dir.iterdir()
                if f.suffix.lower() in extensions
            ],
            key=natural_sort_key,
        )

        # Check for existing annotations (check both output_dir and output_dir/labels)
        labels_dir = self.output_dir / "labels"
        for img_path in self.image_list:
            # Check in labels subdirectory first (batch save location)
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                # Fallback to output_dir directly (legacy location)
                label_path = self.output_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.annotated_images.add(img_path)

        # Update listbox
        self._update_image_listbox()

        if not self.image_list:
            messagebox.showwarning(
                "No Images", f"No images found in {self.input_dir}"
            )

    def _on_image_select(self, event):
        """Handle image selection from listbox."""
        selection = self.image_listbox.curselection()
        if selection:
            self.current_index = selection[0]
            self._load_current_image()

    def _load_current_image(self):
        """Load the current image for annotation."""
        if not self.image_list or self.predictor is None:
            return

        self.current_image_path = self.image_list[self.current_index]

        # Load image
        image = cv2.imread(str(self.current_image_path))
        if image is None:
            messagebox.showerror("Error", f"Failed to load: {self.current_image_path}")
            return

        # Convert to RGB
        self.current_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reset annotation state
        self.state.reset()
        self.points_frame_index = None
        self.predictor.reset_mask_state()

        # Set image in predictor
        self.status_var.set("Computing image embedding...")
        self.root.update()

        try:
            self.predictor.set_image(self.current_image)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
            return

        # Update display
        self._update_display()
        self._update_status()
        self._update_image_listbox()

    def _on_canvas_resize(self, event):
        """Handle canvas resize."""
        self._update_display()

    def _update_display(self):
        """Update canvas display with current image and annotations."""
        if self.current_image is None:
            return

        # Get canvas size
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        # Create display image
        display = self.current_image.copy()
        img_h, img_w = display.shape[:2]

        # Determine which mask to display
        display_mask = None
        mask_color = [0, 255, 0]  # Default: green

        if self.show_mask_var.get():
            # Prioritize tracking results when tracking is active
            if (
                self.tracking_state.is_tracking_initialized
                and self.current_index in self.tracking_state.tracking_results
            ):
                # Use tracking result mask
                result = self.tracking_state.tracking_results[self.current_index]
                display_mask = result.mask
                # Use yellow for low confidence
                if result.is_low_confidence:
                    mask_color = [255, 200, 0]
            elif self.state.current_mask is not None:
                # Use current annotation mask (for manual annotation mode)
                display_mask = self.state.current_mask

        # Draw mask overlay
        if display_mask is not None:
            mask_overlay = np.zeros_like(display)
            # Ensure mask is boolean type for NumPy indexing
            mask_bool = display_mask.astype(bool)
            mask_overlay[mask_bool] = mask_color
            display = cv2.addWeighted(display, 1.0, mask_overlay, 0.4, 0)

            # Draw bounding box
            bbox = mask_to_bbox(display_mask, use_contour=True)
            if bbox:
                x1, y1, x2, y2 = bbox
                box_color = tuple(mask_color)
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

        # Draw points only on the frame where they were added
        if self.points_frame_index is None or self.current_index == self.points_frame_index:
            # Draw foreground points (green with plus)
            for x, y in self.state.foreground_points:
                cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
                cv2.circle(display, (x, y), 5, (255, 255, 255), 2)
                cv2.line(display, (x - 3, y), (x + 3, y), (255, 255, 255), 2)
                cv2.line(display, (x, y - 3), (x, y + 3), (255, 255, 255), 2)

            # Draw background points (red with minus)
            for x, y in self.state.background_points:
                cv2.circle(display, (x, y), 5, (255, 0, 0), -1)
                cv2.circle(display, (x, y), 5, (255, 255, 255), 2)
                cv2.line(display, (x - 3, y), (x + 3, y), (255, 255, 255), 2)

        # Draw saved annotation bounding box (orange) if no active annotation
        if (
            self.state.current_mask is None
            and self.current_image_path is not None
            and self.current_image_path in self.annotated_images
        ):
            # Check in labels subdirectory first (batch save location)
            label_path = self.output_dir / "labels" / f"{self.current_image_path.stem}.txt"
            if not label_path.exists():
                # Fallback to output_dir directly (legacy location)
                label_path = self.output_dir / f"{self.current_image_path.stem}.txt"
            if label_path.exists():
                labels = read_yolo_label(str(label_path))
                for class_id, x_c, y_c, w, h in labels:
                    x1, y1, x2, y2 = yolo_to_bbox(x_c, y_c, w, h, img_w, img_h)
                    # Orange color for saved annotations
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 165, 0), 2)

        # Calculate scale to fit canvas (allow scaling up and down)
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale_factor = min(scale_w, scale_h)

        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)

        # Resize for display
        display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Calculate offset for centering
        self.offset_x = (canvas_w - new_w) // 2
        self.offset_y = (canvas_h - new_h) // 2

        # Convert to PhotoImage
        image = Image.fromarray(display)
        photo = ImageTk.PhotoImage(image)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            self.offset_x, self.offset_y, anchor=tk.NW, image=photo
        )
        self.canvas.image = photo  # Keep reference

    def _canvas_to_image_coords(
        self, canvas_x: int, canvas_y: int
    ) -> Optional[Tuple[int, int]]:
        """Convert canvas coordinates to image coordinates."""
        if self.current_image is None:
            return None

        img_h, img_w = self.current_image.shape[:2]

        # Remove offset and scale
        img_x = int((canvas_x - self.offset_x) / self.scale_factor)
        img_y = int((canvas_y - self.offset_y) / self.scale_factor)

        # Check bounds
        if 0 <= img_x < img_w and 0 <= img_y < img_h:
            return (img_x, img_y)

        return None

    def _on_left_click(self, event):
        """Handle left click: add foreground point."""
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return

        # Record frame where points were first added
        if self.points_frame_index is None:
            self.points_frame_index = self.current_index

        self.state.add_foreground_point(*coords)
        self._run_segmentation()

    def _on_right_click(self, event):
        """Handle right click: add background point."""
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return

        # Record frame where points were first added
        if self.points_frame_index is None:
            self.points_frame_index = self.current_index

        self.state.add_background_point(*coords)
        self._run_segmentation()

    def _run_segmentation(self):
        """Run segmentation with current points."""
        if not self.state.has_points():
            return

        if self.predictor is None:
            return

        try:
            # Use previous mask for refinement if available
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
            self._update_tracking_ui_state()

        except Exception as e:
            self.status_var.set(f"Segmentation failed: {e}")

    def _update_point_info(self):
        """Update point count and IoU display."""
        fg, bg = self.state.get_point_counts()
        self.point_label.config(text=f"Points: FG={fg}, BG={bg}")

        if self.state.current_mask is not None:
            self.iou_label.config(text=f"IoU: {self.state.current_iou:.3f}")
        else:
            self.iou_label.config(text="IoU: --")

    def _update_status(self):
        """Update status bar."""
        annotated = len(self.annotated_images)
        total = len(self.image_list)
        current = self.current_index + 1 if self.image_list else 0

        status = f"Image {current}/{total} | Annotated: {annotated}/{total}"
        if self.current_image_path:
            status += f" | Current: {self.current_image_path.name}"
        status += f" | Class ID: {self.class_id}"

        self.status_var.set(status)

    def _on_undo(self):
        """Handle undo action."""
        if self.state.undo():
            # Also reset predictor mask state if no points left
            if not self.state.has_points():
                self.predictor.reset_mask_state()

            self._update_display()
            self._update_point_info()
            self.status_var.set("Undo: last point removed")
        else:
            self.status_var.set("Nothing to undo")

    def _on_reset(self):
        """Handle reset action."""
        self.state.reset()
        self.points_frame_index = None
        if self.predictor:
            self.predictor.reset_mask_state()
        self._update_display()
        self._update_point_info()
        self.status_var.set("Points cleared. Click to start over.")

    def _on_accept(self):
        """Handle accept and save action."""
        if self.state.current_mask is None:
            messagebox.showwarning("No Mask", "No mask to save. Add points first.")
            return

        # Get bounding box from mask
        bbox = mask_to_bbox(self.state.current_mask, use_contour=True)
        if bbox is None:
            messagebox.showwarning("Invalid Mask", "Could not extract bounding box.")
            return

        x_min, y_min, x_max, y_max = bbox
        img_h, img_w = self.current_image.shape[:2]

        # Convert to YOLO format
        yolo_bbox = bbox_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

        # Save label file to labels subdirectory (consistent with batch_save_yolo_labels)
        labels_dir = self.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{self.current_image_path.stem}.txt"
        write_yolo_label(str(label_path), self.class_id, yolo_bbox)

        # Save mask file to masks subdirectory
        masks_dir = self.output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_path = masks_dir / f"{self.current_image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), self.state.current_mask.astype(np.uint8) * 255)

        # Copy original image to images subdirectory
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        dst_image_path = images_dir / self.current_image_path.name
        if not dst_image_path.exists():
            shutil.copy2(self.current_image_path, dst_image_path)

        # Mark as annotated
        self.annotated_images.add(self.current_image_path)

        self.status_var.set(f"Saved: {label_path.name} + mask + image")

        # Move to next image
        self._on_next_image()

    def _on_skip(self):
        """Handle skip action."""
        self._on_next_image()

    def _on_next_image(self):
        """Navigate to next image."""
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self._load_current_image()

    def _on_prev_image(self):
        """Navigate to previous image."""
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()

    def _toggle_mask_overlay(self):
        """Toggle mask overlay visibility."""
        self.show_mask_var.set(not self.show_mask_var.get())
        self._update_display()

    # =========================================================================
    # Tracking Mode Methods
    # =========================================================================

    def _on_toggle_tracking_mode(self):
        """Handle tracking mode toggle."""
        if self.tracking_mode_var.get():
            self.tracking_state.enable_tracking()
            self._update_tracking_ui_state()
            self.tracking_status_var.set("Tracking: Ready (add points to first frame)")
        else:
            self._on_cancel_tracking()

    def _update_tracking_ui_state(self):
        """Update tracking UI based on current state."""
        is_enabled = self.tracking_state.is_tracking_mode
        has_mask = self.state.current_mask is not None
        has_results = self.tracking_state.is_tracking_initialized
        is_processing = self.tracking_state.is_processing
        is_paused = self.tracking_state.is_paused_between_batches

        # Start tracking: enabled when tracking mode is on, have mask, no results, not processing
        if is_enabled and has_mask and not has_results and not is_processing:
            self.start_tracking_btn.config(state=tk.NORMAL)
        else:
            self.start_tracking_btn.config(state=tk.DISABLED)

        # Stop button: enabled only during processing (not paused)
        if is_processing and not is_paused:
            self.stop_tracking_btn.config(state=tk.NORMAL)
        else:
            self.stop_tracking_btn.config(state=tk.DISABLED)

        # Apply all and Cancel: enabled when we have results and not processing
        if has_results and not is_processing:
            self.apply_all_btn.config(state=tk.NORMAL)
            self.cancel_tracking_btn.config(state=tk.NORMAL)
        else:
            self.apply_all_btn.config(state=tk.DISABLED)
            self.cancel_tracking_btn.config(state=tk.DISABLED)

        # Update low confidence warning
        if has_results:
            low_conf_count = len(self.tracking_state.low_confidence_frames)
            if low_conf_count > 0:
                self.low_conf_label.config(
                    text=f"Warning: {low_conf_count} low confidence frames"
                )
            else:
                self.low_conf_label.config(text="")
        else:
            self.low_conf_label.config(text="")

        # Show/hide exclusion controls based on tracking results and not processing
        if has_results and not is_processing:
            self.exclusion_frame.pack(fill=tk.X, pady=(3, 0))
            self.toggle_selected_btn.config(state=tk.NORMAL)
            self.exclude_low_conf_btn.config(state=tk.NORMAL)
            self.include_all_btn.config(state=tk.NORMAL)
        elif is_paused:
            # During batch pause, show exclusion controls for review
            self.exclusion_frame.pack(fill=tk.X, pady=(3, 0))
            self.toggle_selected_btn.config(state=tk.NORMAL)
            self.exclude_low_conf_btn.config(state=tk.NORMAL)
            self.include_all_btn.config(state=tk.NORMAL)
        else:
            self.exclusion_frame.pack_forget()
            self.toggle_selected_btn.config(state=tk.DISABLED)
            self.exclude_low_conf_btn.config(state=tk.DISABLED)
            self.include_all_btn.config(state=tk.DISABLED)

    def _on_start_tracking(self):
        """Start tracking from current mask."""
        if not self.tracking_state.is_tracking_mode:
            return

        if self.state.current_mask is None:
            messagebox.showwarning(
                "No Mask",
                "Add points to create a mask first, then start tracking."
            )
            return

        # Disable UI during tracking and set processing state
        self.start_tracking_btn.config(state=tk.DISABLED)
        self.tracking_status_var.set("Tracking: Initializing...")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.progress_var.set(0)

        # Set processing state
        self.tracking_state.set_processing(True)
        self.tracking_state.clear_stop_request()
        self._update_tracking_ui_state()

        def tracking_task():
            try:
                # Initialize video predictor if not already done
                if self.video_predictor is None:
                    self.video_predictor = VideoTrackingPredictor(
                        model_path=self.model_path,
                        device=self.device,
                    )
                    self.root.after(0, lambda: self.tracking_status_var.set(
                        "Tracking: Loading video model..."
                    ))
                    self.video_predictor.load_model(
                        progress_callback=lambda msg: self.root.after(
                            0, lambda: self.status_var.set(msg)
                        )
                    )

                # Check VRAM usage
                self.root.after(0, lambda: self.tracking_status_var.set(
                    "Tracking: Checking VRAM..."
                ))

                # Calculate frame range: from current frame to end
                start_frame = self.current_index
                end_frame = len(self.image_list) - 1
                num_frames = end_frame - start_frame + 1

                # Get first image to estimate size
                if self.current_image is not None:
                    img_size = self.current_image.shape[:2]

                    vram_estimate = self.video_predictor.estimate_vram_usage(
                        num_frames, img_size
                    )

                    if vram_estimate.needs_split:
                        # Show warning dialog on main thread
                        self.root.after(0, lambda: self._show_vram_warning_dialog(
                            vram_estimate, num_frames, start_frame
                        ))
                        return

                # Initialize sequence from current frame to end
                self.root.after(0, lambda: self.tracking_status_var.set(
                    f"Tracking: Preparing frames {start_frame}-{end_frame}..."
                ))

                # Create BatchInfo for partial sequence
                batch_info = BatchInfo(
                    batch_index=0,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    frame_count=num_frames,
                    is_first_batch=True,
                    is_last_batch=True,
                )

                num_frames_in_batch, global_frame_map = self.video_predictor.init_batch_sequence(
                    image_paths=self.image_list,
                    batch_info=batch_info,
                    progress_callback=lambda msg: self.root.after(
                        0, lambda: self.status_var.set(msg)
                    )
                )

                # Store frame map (global indices)
                self.tracking_frame_map.update(global_frame_map)

                # Set initial prompt on first frame of the batch (local index 0)
                self.root.after(0, lambda: self.tracking_status_var.set(
                    "Tracking: Setting initial prompt..."
                ))
                self.video_predictor.set_initial_prompt(
                    frame_idx=0,  # Local index within batch
                    obj_id=self.tracking_state.current_obj_id,
                    foreground_points=self.state.foreground_points,
                    background_points=self.state.background_points,
                )

                # Propagate tracking
                self.root.after(0, lambda: self.tracking_status_var.set(
                    "Tracking: Propagating..."
                ))

                def progress_callback(current, total, local_frame_idx, mask):
                    # Convert local index to global index
                    global_frame_idx = start_frame + local_frame_idx

                    def update_ui():
                        # Update progress
                        progress = (current / total) * 100
                        self.progress_var.set(progress)
                        self.tracking_status_var.set(
                            f"Tracking: {current}/{total} frames"
                        )

                        # Switch to current frame and display tracking result
                        if global_frame_idx < len(self.image_list):
                            self.current_index = global_frame_idx
                            self.current_image_path = self.image_list[global_frame_idx]

                            # Load image for display
                            img = cv2.imread(str(self.current_image_path))
                            if img is not None:
                                self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                                # Store tracking result for display
                                temp_result = TrackingResult.from_mask(
                                    mask, reference_area=self.video_predictor.reference_mask_area
                                )
                                self.tracking_state.tracking_results[global_frame_idx] = temp_result
                                self.tracking_state.is_tracking_initialized = True

                                # Update display and listbox
                                self._update_display()
                                self._update_image_listbox()

                                # Update listbox selection
                                self.image_listbox.selection_clear(0, tk.END)
                                self.image_listbox.selection_set(global_frame_idx)
                                self.image_listbox.see(global_frame_idx)

                    self.root.after(0, update_ui)

                # Define stop check callback
                def stop_check():
                    return self.tracking_state.is_stop_requested()

                local_results = self.video_predictor.propagate_tracking(
                    progress_callback=progress_callback,
                    stop_check=stop_check,
                )

                # Convert local results to global frame indices
                global_results = {
                    start_frame + local_idx: result
                    for local_idx, result in local_results.items()
                }

                # Check if stopped
                if self.tracking_state.is_stop_requested():
                    self.root.after(0, lambda r=global_results: self._on_tracking_stopped(r))
                    return

                # Update state on main thread
                self.root.after(0, lambda: self._on_tracking_complete(global_results))

            except Exception as e:
                self.root.after(0, lambda: self._on_tracking_error(str(e)))
            finally:
                self.root.after(0, lambda: self.tracking_state.set_processing(False))

        thread = threading.Thread(target=tracking_task, daemon=True)
        thread.start()

    def _on_tracking_complete(self, results: Dict[int, TrackingResult]):
        """Called when tracking is complete."""
        self.tracking_state.set_processing(False)
        self.tracking_state.set_tracking_results(results)
        self.progress_bar.pack_forget()
        self.progress_var.set(0)

        low_conf_count = len(self.tracking_state.low_confidence_frames)
        total_frames = len(results)

        if low_conf_count > 0:
            self.tracking_status_var.set(
                f"Tracking: Complete ({total_frames} frames, {low_conf_count} warnings)"
            )
        else:
            self.tracking_status_var.set(
                f"Tracking: Complete ({total_frames} frames)"
            )

        self._update_tracking_ui_state()
        self._update_image_listbox()
        self._update_display()

        self.status_var.set(
            f"Tracking complete. Review results and click 'Apply' to save."
        )

    def _on_tracking_error(self, error_msg: str):
        """Called when tracking fails."""
        self.progress_bar.pack_forget()
        self.progress_var.set(0)
        self.tracking_status_var.set("Tracking: Failed")
        self._update_tracking_ui_state()

        messagebox.showerror("Tracking Error", f"Tracking failed:\n{error_msg}")

    def _show_vram_warning_dialog(self, vram_estimate, num_frames: int, start_frame: int = 0):
        """Show VRAM warning dialog for batch splitting."""
        end_frame = start_frame + num_frames - 1
        message = (
            f"VRAM Capacity Warning\n\n"
            f"Target: frames {start_frame}-{end_frame} ({num_frames} frames)\n"
            f"Estimated VRAM usage: {vram_estimate.estimated_usage_gb:.1f}GB\n"
            f"Available VRAM: {vram_estimate.available_gb:.1f}GB\n\n"
            f"Processing with batch splitting:\n"
        )

        batch_size = vram_estimate.recommended_batch_size
        for i in range(vram_estimate.num_batches):
            batch_start = start_frame + i * batch_size
            batch_end = min(start_frame + (i + 1) * batch_size - 1, end_frame)
            count = batch_end - batch_start + 1
            message += f"- Batch {i + 1}: frames {batch_start}-{batch_end} ({count} images)\n"

        result = messagebox.askyesno(
            "VRAM Capacity Warning",
            message + "\nDo you want to continue?",
            icon="warning"
        )

        if result:
            # Start batch processing
            self._run_batch_tracking(vram_estimate, num_frames, start_frame)
        else:
            self.tracking_status_var.set("Tracking: Cancelled")
            self.progress_bar.pack_forget()
            self._update_tracking_ui_state()

    def _run_batch_tracking(self, vram_estimate, num_frames: int, global_start_frame: int = 0):
        """
        Run tracking in batches to handle large sequences with limited VRAM.

        Args:
            vram_estimate: VRAMEstimate with batch configuration
            num_frames: Total number of frames to process
            global_start_frame: Global frame index to start tracking from
        """
        self.tracking_status_var.set("Tracking: Starting batch processing...")
        self.progress_var.set(0)

        # Set processing state and clear stop request
        self.tracking_state.set_processing(True)
        self.tracking_state.clear_stop_request()
        self._batch_continue_event.clear()
        self._update_tracking_ui_state()

        def batch_tracking_task():
            try:
                batch_size = vram_estimate.recommended_batch_size
                num_batches = vram_estimate.num_batches
                all_results: Dict[int, TrackingResult] = {}
                current_mask = None
                low_conf_batches = []

                for batch_idx in range(num_batches):
                    # Check for stop request at start of each batch
                    if self.tracking_state.is_stop_requested():
                        self.root.after(0, lambda r=dict(all_results): self._on_tracking_stopped(r))
                        return

                    # Calculate batch frame range (global indices)
                    batch_start = global_start_frame + batch_idx * batch_size
                    batch_end = min(global_start_frame + (batch_idx + 1) * batch_size - 1,
                                   global_start_frame + num_frames - 1)
                    frame_count = batch_end - batch_start + 1

                    batch_info = BatchInfo(
                        batch_index=batch_idx,
                        start_frame=batch_start,
                        end_frame=batch_end,
                        frame_count=frame_count,
                        is_first_batch=(batch_idx == 0),
                        is_last_batch=(batch_idx == num_batches - 1),
                    )

                    # Update UI for batch start
                    self.root.after(0, lambda b=batch_idx, n=num_batches: (
                        self.tracking_status_var.set(
                            f"Tracking: Batch {b + 1}/{n} - Initializing..."
                        )
                    ))

                    # Initialize batch sequence
                    self.video_predictor.init_batch_sequence(
                        image_paths=self.image_list,
                        batch_info=batch_info,
                        progress_callback=lambda msg: self.root.after(
                            0, lambda m=msg: self.status_var.set(m)
                        ),
                    )

                    # Set initial prompt or mask
                    if batch_info.is_first_batch:
                        # First batch: use user's points (at local index 0)
                        self.root.after(0, lambda: self.tracking_status_var.set(
                            f"Tracking: Batch 1/{num_batches} - Setting initial prompt..."
                        ))

                        self.video_predictor.set_initial_prompt(
                            frame_idx=0,  # Local index within batch
                            obj_id=self.tracking_state.current_obj_id,
                            foreground_points=self.state.foreground_points,
                            background_points=self.state.background_points,
                        )
                    else:
                        # Subsequent batches: use last mask from previous batch
                        self.root.after(0, lambda b=batch_idx, n=num_batches: (
                            self.tracking_status_var.set(
                                f"Tracking: Batch {b + 1}/{n} - Setting initial mask..."
                            )
                        ))
                        if current_mask is not None:
                            self.video_predictor.set_initial_mask(
                                frame_idx=0,
                                obj_id=self.tracking_state.current_obj_id,
                                mask=current_mask,
                            )
                        else:
                            raise RuntimeError("No mask from previous batch")

                    # Propagate tracking within batch
                    frames_processed_before = len(all_results)

                    def batch_progress_callback(current, total, local_frame_idx, mask):
                        # Convert local to global frame index
                        global_frame_idx = batch_start + local_frame_idx
                        overall_processed = frames_processed_before + current

                        def update_ui():
                            # Update progress bar
                            overall_pct = (overall_processed / num_frames) * 100
                            self.progress_var.set(overall_pct)

                            # Update status
                            self.tracking_status_var.set(
                                f"Tracking: Batch {batch_idx + 1}/{num_batches} "
                                f"({current}/{total} frames)"
                            )

                            # Display current frame
                            if global_frame_idx < len(self.image_list):
                                self.current_index = global_frame_idx
                                self.current_image_path = self.image_list[global_frame_idx]

                                img = cv2.imread(str(self.current_image_path))
                                if img is not None:
                                    self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                                    temp_result = TrackingResult.from_mask(
                                        mask,
                                        reference_area=self.video_predictor.reference_mask_area,
                                    )
                                    self.tracking_state.tracking_results[global_frame_idx] = temp_result
                                    self.tracking_state.is_tracking_initialized = True

                                    self._update_display()
                                    self._update_image_listbox()

                                    self.image_listbox.selection_clear(0, tk.END)
                                    self.image_listbox.selection_set(global_frame_idx)
                                    self.image_listbox.see(global_frame_idx)

                        self.root.after(0, update_ui)

                    # Define stop check callback
                    def stop_check():
                        return self.tracking_state.is_stop_requested()

                    batch_results = self.video_predictor.propagate_tracking(
                        progress_callback=batch_progress_callback,
                        stop_check=stop_check,
                    )

                    # Convert local results to global frame indices
                    for local_idx, result in batch_results.items():
                        global_idx = batch_start + local_idx
                        all_results[global_idx] = result
                        self.tracking_frame_map[global_idx] = self.image_list[global_idx]

                    # Check if stopped during this batch
                    if self.tracking_state.is_stop_requested():
                        self.root.after(0, lambda r=dict(all_results): self._on_tracking_stopped(r))
                        return

                    # Check for low confidence in this batch
                    batch_low_conf = [
                        batch_start + local_idx
                        for local_idx, result in batch_results.items()
                        if result.is_low_confidence
                    ]
                    if batch_low_conf:
                        low_conf_batches.append((batch_idx + 1, len(batch_low_conf)))

                    # Save last mask for next batch
                    if batch_results:
                        last_local_idx = max(batch_results.keys())
                        current_mask = batch_results[last_local_idx].mask

                    # Clear VRAM and pause for user review (except for last batch)
                    if not batch_info.is_last_batch:
                        self.root.after(0, lambda b=batch_idx, n=num_batches: (
                            self.status_var.set(
                                f"Batch {b + 1}/{n} complete. Clearing VRAM..."
                            )
                        ))
                        self.video_predictor.clear_vram()

                        # Show batch pause dialog for user review
                        batch_results_copy = dict(batch_results)
                        self.root.after(0, lambda b=batch_idx, n=num_batches, r=batch_results_copy: (
                            self._show_batch_pause_dialog(b, n, r)
                        ))

                        # Wait for user to continue or stop
                        self._batch_continue_event.clear()
                        self._batch_continue_event.wait()

                        # Check if user chose to stop
                        if self.tracking_state.is_stop_requested():
                            self.root.after(0, lambda r=dict(all_results): self._on_tracking_stopped(r))
                            return

                # Show warning if any batches had low confidence frames
                if low_conf_batches:
                    warning_msg = "Low confidence frames detected:\n"
                    for batch_num, count in low_conf_batches:
                        warning_msg += f"- Batch {batch_num}: {count} frames\n"
                    self.root.after(0, lambda: self.status_var.set(warning_msg.strip()))

                # Complete - update state on main thread
                self.root.after(0, lambda: self._on_tracking_complete(all_results))

            except Exception as e:
                self.root.after(0, lambda: self._on_tracking_error(str(e)))
            finally:
                self.root.after(0, lambda: self.tracking_state.set_processing(False))

        thread = threading.Thread(target=batch_tracking_task, daemon=True)
        thread.start()

    def _on_apply_tracking_results(self):
        """Apply tracking results and save all annotations (excluding excluded frames)."""
        if not self.tracking_state.is_tracking_initialized:
            return

        # Get exclusion stats
        included_count, excluded_count, low_conf_included = self.tracking_state.get_exclusion_stats()

        # Filter out excluded frames
        included_results = {
            idx: result
            for idx, result in self.tracking_state.tracking_results.items()
            if not self.tracking_state.is_frame_excluded(idx)
        }

        included_frame_map = {
            idx: path
            for idx, path in self.tracking_frame_map.items()
            if not self.tracking_state.is_frame_excluded(idx)
        }

        # Build confirmation message
        confirm_msg = f"Save annotations for {included_count} frames?"
        if excluded_count > 0:
            confirm_msg += f"\n\n({excluded_count} frames excluded)"
        if low_conf_included > 0:
            confirm_msg += f"\nWarning: {low_conf_included} low confidence frames included"

        result = messagebox.askyesno(
            "Confirm Save",
            confirm_msg,
            icon="question"
        )
        if not result:
            return

        # Save all results (excluding excluded frames)
        self.status_var.set("Saving annotations...")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        def save_task():
            try:
                result = batch_save_yolo_labels(
                    tracking_results=included_results,
                    frame_map=included_frame_map,
                    output_dir=str(self.output_dir),
                    class_id=self.class_id,
                    copy_images=True,  # Copy images for Copy-Paste augmentation
                )

                self.root.after(0, lambda: self._on_save_complete(result, excluded_count))

            except Exception as e:
                self.root.after(0, lambda: self._on_save_error(str(e)))

        thread = threading.Thread(target=save_task, daemon=True)
        thread.start()

    def _on_save_complete(self, result, excluded_count: int = 0):
        """Called when save is complete."""
        self.progress_bar.pack_forget()

        # Update annotated images set (only for included frames)
        for frame_idx in self.tracking_state.tracking_results.keys():
            if frame_idx in self.tracking_frame_map:
                # Only add to annotated if not excluded
                if not self.tracking_state.is_frame_excluded(frame_idx):
                    self.annotated_images.add(self.tracking_frame_map[frame_idx])

        self._update_image_listbox()
        self._on_cancel_tracking()

        # Build summary message
        summary_msg = f"Save Complete\n\n{result.summary()}"
        if excluded_count > 0:
            summary_msg += f"\n\n{excluded_count} frames were excluded and not saved."

        messagebox.showinfo("Save Complete", summary_msg)

        status_msg = f"Saved {result.successful}/{result.total_images} annotations"
        if excluded_count > 0:
            status_msg += f" ({excluded_count} excluded)"
        self.status_var.set(status_msg)

    def _on_save_error(self, error_msg: str):
        """Called when save fails."""
        self.progress_bar.pack_forget()
        messagebox.showerror("Save Error", f"Save failed:\n{error_msg}")

    # =========================================================================
    # Frame Exclusion Methods
    # =========================================================================

    def _on_toggle_frame_exclusion(self):
        """Toggle exclusion state of the currently selected frame."""
        if not self.tracking_state.is_tracking_initialized:
            return

        selection = self.image_listbox.curselection()
        if not selection:
            self.status_var.set("Select a frame to toggle exclusion")
            return

        frame_idx = selection[0]

        # Check if this frame has tracking results
        if frame_idx not in self.tracking_state.tracking_results:
            self.status_var.set("Selected frame has no tracking results")
            return

        is_now_excluded = self.tracking_state.toggle_frame_exclusion(frame_idx)

        # Update display
        self._update_image_listbox()
        self._update_display()

        if is_now_excluded:
            self.status_var.set(f"Frame {frame_idx} excluded from training")
        else:
            self.status_var.set(f"Frame {frame_idx} included in training")

    def _on_exclude_all_low_confidence(self):
        """Exclude all low confidence frames."""
        if not self.tracking_state.is_tracking_initialized:
            return

        count = self.tracking_state.exclude_all_low_confidence()

        # Update display
        self._update_image_listbox()
        self._update_display()

        if count > 0:
            self.status_var.set(f"Excluded {count} low confidence frames")
        else:
            self.status_var.set("No additional frames to exclude")

    def _on_include_all(self):
        """Include all frames (clear exclusions)."""
        if not self.tracking_state.is_tracking_initialized:
            return

        count = self.tracking_state.include_all()

        # Update display
        self._update_image_listbox()
        self._update_display()

        if count > 0:
            self.status_var.set(f"Included {count} previously excluded frames")
        else:
            self.status_var.set("All frames already included")

    def _update_exclusion_stats(self):
        """Update the exclusion statistics label."""
        if not self.tracking_state.is_tracking_initialized:
            self.exclusion_stats_var.set("")
            return

        included, excluded, low_conf_included = self.tracking_state.get_exclusion_stats()
        total = included + excluded

        if excluded > 0:
            stats_text = f"Included: {included}/{total} ({excluded} excluded)"
        else:
            stats_text = f"Included: {included}/{total}"

        if low_conf_included > 0:
            stats_text += f" | {low_conf_included} low-conf included"

        self.exclusion_stats_var.set(stats_text)

    def _on_listbox_double_click(self, event):
        """Handle double-click on listbox to toggle frame exclusion."""
        if not self.tracking_state.is_tracking_initialized:
            return

        # Get the item that was double-clicked
        index = self.image_listbox.nearest(event.y)
        if index < 0:
            return

        # Check if this frame has tracking results
        if index not in self.tracking_state.tracking_results:
            return

        # Toggle exclusion
        is_now_excluded = self.tracking_state.toggle_frame_exclusion(index)

        # Update display
        self._update_image_listbox()
        self._update_display()

        if is_now_excluded:
            self.status_var.set(f"Frame {index} excluded from training")
        else:
            self.status_var.set(f"Frame {index} included in training")

    def _on_cancel_tracking(self):
        """Cancel tracking and reset state."""
        self.tracking_state.disable_tracking()
        self.tracking_mode_var.set(False)
        self.tracking_status_var.set("Tracking: OFF")

        # Reset annotation state (clear current_mask so saved rectangles can be shown)
        self.state.reset()
        if self.predictor:
            self.predictor.reset_mask_state()

        if self.video_predictor:
            self.video_predictor.reset()

        self.tracking_frame_map.clear()
        self._update_tracking_ui_state()
        self._update_image_listbox()
        self._update_display()

    def _on_stop_tracking(self):
        """Handle stop button click during tracking."""
        if not self.tracking_state.is_processing:
            return

        # Pause processing BEFORE showing dialog
        self.tracking_state.request_stop()
        self.status_var.set("Paused - waiting for confirmation...")

        result = messagebox.askyesno(
            "Stop Tracking",
            "Processing is paused.\n"
            "Results obtained so far will be preserved.\n\n"
            "Do you want to stop?",
            icon="warning"
        )

        if result:
            # User confirmed stop
            self.status_var.set("Stopping...")
            self.stop_tracking_btn.config(state=tk.DISABLED)
        else:
            # User cancelled - resume processing
            self.tracking_state.clear_stop_request()
            self.status_var.set("Resumed tracking...")

    def _show_batch_pause_dialog(
        self, batch_idx: int, num_batches: int, batch_results: Dict
    ):
        """Show batch pause dialog for user review and decision."""
        self.tracking_state.pause_for_batch_review(batch_idx, num_batches)

        # Update batch info
        self.batch_info_var.set(f"Batch {batch_idx + 1}/{num_batches} completed")

        # Calculate summary
        total_frames = len(batch_results)
        low_conf_count = sum(1 for r in batch_results.values() if r.is_low_confidence)
        self.batch_summary_var.set(
            f"({total_frames} frames, {low_conf_count} low confidence)"
        )

        # Show batch pause frame
        self.batch_pause_frame.pack(fill=tk.X, pady=(3, 0))

        # Update UI state
        self._update_tracking_ui_state()

        # Update status
        self.tracking_status_var.set(
            f"Batch {batch_idx + 1}/{num_batches} complete - Review and continue"
        )
        self.status_var.set(
            f"Processed {total_frames} frames. "
            f"Low confidence: {low_conf_count}. "
            f"Review results and click Continue or Stop."
        )

    def _hide_batch_pause_frame(self):
        """Hide the batch pause control frame."""
        self.batch_pause_frame.pack_forget()

    def _on_continue_batch(self):
        """Continue to the next batch after pause."""
        self._hide_batch_pause_frame()
        self.tracking_state.resume_from_pause()
        self.tracking_state.set_processing(True)
        self._update_tracking_ui_state()

        # Signal worker thread to continue
        self._batch_continue_event.set()

    def _on_stop_at_batch(self):
        """Stop processing at the current batch."""
        self._hide_batch_pause_frame()
        self.tracking_state.resume_from_pause()
        self.tracking_state.request_stop()

        # Signal worker thread to exit
        self._batch_continue_event.set()

    def _on_tracking_stopped(self, results: Dict):
        """Called when tracking is stopped by user."""
        self.tracking_state.set_tracking_results(results)
        self.tracking_state.set_processing(False)
        self.progress_bar.pack_forget()
        self.progress_var.set(0)

        total_frames = len(results)
        low_conf_count = len(self.tracking_state.low_confidence_frames)

        self.tracking_status_var.set(
            f"Tracking: Stopped ({total_frames} frames processed)"
        )

        self._update_tracking_ui_state()
        self._update_image_listbox()
        self._update_display()

        # Show exclusion frame for review
        if total_frames > 0:
            self.exclusion_frame.pack(fill=tk.X, pady=(3, 0))

        messagebox.showinfo(
            "Tracking Stopped",
            f"Tracking stopped by user.\n\n"
            f"Frames processed: {total_frames}\n"
            f"Low confidence: {low_conf_count}\n\n"
            f"You can review and apply the results obtained so far."
        )

        self.status_var.set(
            f"Tracking stopped. {total_frames} frames processed. "
            f"Click 'Apply' to save results."
        )

    def _update_image_listbox(self):
        """Update image list display with tracking status and exclusion checkboxes."""
        self.image_listbox.delete(0, tk.END)

        for i, path in enumerate(self.image_list):
            # Determine status prefix and color
            if path in self.annotated_images:
                prefix = "[✓]"
                color = "black"
            elif i in self.tracking_state.tracking_results:
                # Only show tracking status for frames that have actual results
                is_excluded = self.tracking_state.is_frame_excluded(i)
                is_low_conf = i in self.tracking_state.low_confidence_frames

                if is_excluded:
                    # Excluded frames: unchecked box
                    if is_low_conf:
                        prefix = "[☐!]"  # Excluded low confidence
                    else:
                        prefix = "[☐]"   # Excluded normal
                    color = "gray"
                else:
                    # Included frames: checked box
                    if is_low_conf:
                        prefix = "[☑!]"  # Included low confidence
                        color = "orange"
                    else:
                        prefix = "[☑]"   # Included normal
                        color = "green"
            else:
                prefix = "[ ]"
                color = "black"

            display_name = f"{prefix} {path.name}"
            self.image_listbox.insert(tk.END, display_name)
            self.image_listbox.itemconfig(i, foreground=color)

        # Select current image
        if self.image_list:
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)

        # Update exclusion stats
        self._update_exclusion_stats()

    def _on_close(self):
        """Handle window close."""
        if self.video_predictor:
            self.video_predictor.reset()
        self.root.destroy()


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SAM2 Interactive Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python sam2_interactive_app.py \\
      --input-dir datasets/raw_captures/object \\
      --output-dir datasets/annotated/object \\
      --class-id 0

  # With custom model
  python sam2_interactive_app.py \\
      --input-dir images/ \\
      --output-dir labels/ \\
      --model models/sam2_b.pt \\
      --device cuda

Controls:
  Left click:  Add foreground point (include in mask)
  Right click: Add background point (exclude from mask)
  Ctrl+Z:      Undo last point
  Escape:      Reset all points
  Enter:       Accept and save annotation
  Arrow keys:  Navigate between images
  Space:       Toggle mask overlay
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing images to annotate",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory for output YOLO label files",
    )
    parser.add_argument(
        "--class-id",
        "-c",
        type=int,
        default=0,
        help="YOLO class ID (default: 0)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="sam2_b.pt",
        help="Path to SAM2 model checkpoint (default: sam2_b.pt)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run model on (default: cuda)",
    )

    args = parser.parse_args()

    # Check input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Check model file
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent.parent
        # Handle both "sam2.1_xxx.pt" and "models/sam2.1_xxx.pt" formats
        model_name = Path(args.model).name
        model_path = project_root / "models" / model_name
        if not model_path.exists():
            print(f"Error: Model file not found: {args.model}")
            print(f"Tried: {model_path}")
            sys.exit(1)

    # Create main window
    root = tk.Tk()

    # Apply theme
    style = ttk.Style()
    AppTheme.apply(style)

    # Create app
    app = SAM2AnnotationApp(
        root=root,
        input_dir=str(input_dir),
        output_dir=args.output_dir,
        class_id=args.class_id,
        model_path=str(model_path),
        device=args.device,
    )

    # Run main loop
    root.mainloop()


if __name__ == "__main__":
    main()
