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
from annotation_utils import (
    bbox_to_yolo,
    write_yolo_label,
    batch_save_yolo_labels,
    read_yolo_label,
    yolo_to_bbox,
)
from video_tracking_predictor import VideoTrackingPredictor, TrackingResult


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

    Tracks tracking mode status, results, and low confidence frames.
    """

    is_tracking_mode: bool = False
    is_tracking_initialized: bool = False
    tracking_results: Dict[int, TrackingResult] = field(default_factory=dict)
    low_confidence_frames: List[int] = field(default_factory=list)
    confirmed_frames: Set[int] = field(default_factory=set)
    current_obj_id: int = 1

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

    def get_frame_status(self, frame_idx: int) -> str:
        """
        Get status indicator for a frame.

        Returns:
            Status string: 'confirmed', 'tracked', 'low_confidence', or 'pending'
        """
        if frame_idx in self.confirmed_frames:
            return "confirmed"
        if frame_idx in self.low_confidence_frames:
            return "low_confidence"
        if frame_idx in self.tracking_results:
            return "tracked"
        return "pending"


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
        self.annotated_images: set = set()

        # Tracking mode state
        self.tracking_state = TrackingState()
        self.video_predictor: Optional[VideoTrackingPredictor] = None
        self.tracking_frame_map: Dict[int, Path] = {}

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

        # Tracking mode toggle
        self.tracking_mode_var = tk.BooleanVar(value=False)
        self.tracking_toggle = ttk.Checkbutton(
            tracking_frame,
            text="Enable Tracking Mode",
            variable=self.tracking_mode_var,
            command=self._on_toggle_tracking_mode,
        )
        self.tracking_toggle.pack(side=tk.LEFT, padx=5)

        # Start tracking button
        self.start_tracking_btn = ttk.Button(
            tracking_frame,
            text="Start Tracking",
            command=self._on_start_tracking,
            state=tk.DISABLED,
        )
        self.start_tracking_btn.pack(side=tk.LEFT, padx=5)

        # Apply all button
        self.apply_all_btn = ttk.Button(
            tracking_frame,
            text="Apply All",
            command=self._on_apply_tracking_results,
            state=tk.DISABLED,
        )
        self.apply_all_btn.pack(side=tk.LEFT, padx=5)

        # Cancel tracking button
        self.cancel_tracking_btn = ttk.Button(
            tracking_frame,
            text="Cancel Tracking",
            command=self._on_cancel_tracking,
            state=tk.DISABLED,
        )
        self.cancel_tracking_btn.pack(side=tk.LEFT, padx=5)

        # Tracking status label
        self.tracking_status_var = tk.StringVar(value="Tracking: OFF")
        ttk.Label(
            tracking_frame,
            textvariable=self.tracking_status_var,
            font=("TkDefaultFont", 9),
        ).pack(side=tk.RIGHT, padx=10)

        # Low confidence warning
        self.low_conf_label = ttk.Label(
            tracking_frame,
            text="",
            foreground="orange",
            font=("TkDefaultFont", 9, "bold"),
        )
        self.low_conf_label.pack(side=tk.RIGHT, padx=5)

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
            bbox = self._mask_to_bbox(display_mask)
            if bbox:
                x1, y1, x2, y2 = bbox
                box_color = tuple(mask_color)
                cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

        # Draw foreground points (green with plus)
        for x, y in self.state.foreground_points:
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
            cv2.line(display, (x - 5, y), (x + 5, y), (255, 255, 255), 2)
            cv2.line(display, (x, y - 5), (x, y + 5), (255, 255, 255), 2)

        # Draw background points (red with minus)
        for x, y in self.state.background_points:
            cv2.circle(display, (x, y), 8, (255, 0, 0), -1)
            cv2.circle(display, (x, y), 8, (255, 255, 255), 2)
            cv2.line(display, (x - 5, y), (x + 5, y), (255, 255, 255), 2)

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

        # Calculate scale to fit canvas
        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        self.scale_factor = min(scale_w, scale_h, 1.0)

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

        self.state.add_foreground_point(*coords)
        self._run_segmentation()

    def _on_right_click(self, event):
        """Handle right click: add background point."""
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords is None:
            return

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

    def _mask_to_bbox(
        self, mask: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """Convert mask to bounding box."""
        # Find contours
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            # Fallback: use mask bounds
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                return None

            y_indices = np.where(rows)[0]
            x_indices = np.where(cols)[0]
            return (
                int(x_indices[0]),
                int(y_indices[0]),
                int(x_indices[-1]),
                int(y_indices[-1]),
            )

        # Get bounding rect of largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        return (x, y, x + w, y + h)

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
        bbox = self._mask_to_bbox(self.state.current_mask)
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

        # Mark as annotated
        self.annotated_images.add(self.current_image_path)

        self.status_var.set(f"Saved: {label_path.name}")

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

        # Start tracking: enabled when tracking mode is on and we have a mask
        if is_enabled and has_mask and not has_results:
            self.start_tracking_btn.config(state=tk.NORMAL)
        else:
            self.start_tracking_btn.config(state=tk.DISABLED)

        # Apply all: enabled when we have tracking results
        if has_results:
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

        # Disable UI during tracking
        self.start_tracking_btn.config(state=tk.DISABLED)
        self.tracking_status_var.set("Tracking: Initializing...")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.progress_var.set(0)

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

                # Get first image to estimate size
                if self.current_image is not None:
                    img_size = self.current_image.shape[:2]
                    num_frames = len(self.image_list)

                    vram_estimate = self.video_predictor.estimate_vram_usage(
                        num_frames, img_size
                    )

                    if vram_estimate.needs_split:
                        # Show warning dialog on main thread
                        self.root.after(0, lambda: self._show_vram_warning_dialog(
                            vram_estimate, num_frames
                        ))
                        return

                # Initialize sequence
                self.root.after(0, lambda: self.tracking_status_var.set(
                    "Tracking: Preparing frames..."
                ))
                num_frames, _ = self.video_predictor.init_sequence(
                    str(self.input_dir),
                    progress_callback=lambda msg: self.root.after(
                        0, lambda: self.status_var.set(msg)
                    )
                )

                # Store frame map
                self.tracking_frame_map = self.video_predictor.frame_map.copy()

                # Set initial prompt on current frame
                self.root.after(0, lambda: self.tracking_status_var.set(
                    "Tracking: Setting initial prompt..."
                ))
                self.video_predictor.set_initial_prompt(
                    frame_idx=self.current_index,
                    obj_id=self.tracking_state.current_obj_id,
                    foreground_points=self.state.foreground_points,
                    background_points=self.state.background_points,
                )

                # Propagate tracking
                self.root.after(0, lambda: self.tracking_status_var.set(
                    "Tracking: Propagating..."
                ))

                def progress_callback(current, total, frame_idx, mask):
                    def update_ui():
                        # Update progress
                        progress = (current / total) * 100
                        self.progress_var.set(progress)
                        self.tracking_status_var.set(
                            f"Tracking: {current}/{total} frames"
                        )

                        # Switch to current frame and display tracking result
                        if frame_idx < len(self.image_list):
                            self.current_index = frame_idx
                            self.current_image_path = self.image_list[frame_idx]

                            # Load image for display
                            img = cv2.imread(str(self.current_image_path))
                            if img is not None:
                                self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                                # Store tracking result for display
                                temp_result = TrackingResult.from_mask(
                                    mask, reference_area=self.video_predictor.reference_mask_area
                                )
                                self.tracking_state.tracking_results[frame_idx] = temp_result
                                self.tracking_state.is_tracking_initialized = True

                                # Update display and listbox
                                self._update_display()
                                self._update_image_listbox()

                                # Update listbox selection
                                self.image_listbox.selection_clear(0, tk.END)
                                self.image_listbox.selection_set(frame_idx)
                                self.image_listbox.see(frame_idx)

                    self.root.after(0, update_ui)

                results = self.video_predictor.propagate_tracking(
                    progress_callback=progress_callback
                )

                # Update state on main thread
                self.root.after(0, lambda: self._on_tracking_complete(results))

            except Exception as e:
                self.root.after(0, lambda: self._on_tracking_error(str(e)))

        thread = threading.Thread(target=tracking_task, daemon=True)
        thread.start()

    def _on_tracking_complete(self, results: Dict[int, TrackingResult]):
        """Called when tracking is complete."""
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
            f"Tracking complete. Review results and click 'Apply All' to save."
        )

    def _on_tracking_error(self, error_msg: str):
        """Called when tracking fails."""
        self.progress_bar.pack_forget()
        self.progress_var.set(0)
        self.tracking_status_var.set("Tracking: Failed")
        self._update_tracking_ui_state()

        messagebox.showerror("Tracking Error", f"Tracking failed:\n{error_msg}")

    def _show_vram_warning_dialog(self, vram_estimate, num_frames: int):
        """Show VRAM warning dialog for batch splitting."""
        message = (
            f"VRAM容量警告\n\n"
            f"処理対象: {num_frames}フレーム\n"
            f"推定VRAM使用量: {vram_estimate.estimated_usage_gb:.1f}GB\n"
            f"利用可能VRAM: {vram_estimate.available_gb:.1f}GB\n\n"
            f"バッチ分割で処理します:\n"
        )

        batch_size = vram_estimate.recommended_batch_size
        for i in range(vram_estimate.num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size - 1, num_frames - 1)
            count = end - start + 1
            message += f"- バッチ{i + 1}: フレーム {start}-{end} ({count}枚)\n"

        result = messagebox.askyesno(
            "VRAM容量警告",
            message + "\n続行しますか？",
            icon="warning"
        )

        if result:
            # TODO: Implement batch processing
            self.status_var.set("Batch processing not yet implemented")
            self.tracking_status_var.set("Tracking: Cancelled (batch needed)")
        else:
            self.tracking_status_var.set("Tracking: Cancelled")

        self.progress_bar.pack_forget()
        self._update_tracking_ui_state()

    def _on_apply_tracking_results(self):
        """Apply tracking results and save all annotations."""
        if not self.tracking_state.is_tracking_initialized:
            return

        # Confirm with user
        total = len(self.tracking_state.tracking_results)
        low_conf = len(self.tracking_state.low_confidence_frames)

        if low_conf > 0:
            result = messagebox.askyesno(
                "Confirm Save",
                f"{total}フレームのアノテーションを保存しますか？\n\n"
                f"注意: {low_conf}フレームは低信頼度です。",
                icon="question"
            )
            if not result:
                return

        # Save all results
        self.status_var.set("Saving annotations...")
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))

        def save_task():
            try:
                result = batch_save_yolo_labels(
                    tracking_results=self.tracking_state.tracking_results,
                    frame_map=self.tracking_frame_map,
                    output_dir=str(self.output_dir),
                    class_id=self.class_id,
                    copy_images=False,  # Don't copy images, just save labels
                )

                self.root.after(0, lambda: self._on_save_complete(result))

            except Exception as e:
                self.root.after(0, lambda: self._on_save_error(str(e)))

        thread = threading.Thread(target=save_task, daemon=True)
        thread.start()

    def _on_save_complete(self, result):
        """Called when save is complete."""
        self.progress_bar.pack_forget()

        # Update annotated images set
        for frame_idx in self.tracking_state.tracking_results.keys():
            if frame_idx in self.tracking_frame_map:
                self.annotated_images.add(self.tracking_frame_map[frame_idx])

        self._update_image_listbox()
        self._on_cancel_tracking()

        messagebox.showinfo(
            "Save Complete",
            f"保存完了\n\n{result.summary()}"
        )

        self.status_var.set(
            f"Saved {result.successful}/{result.total_images} annotations"
        )

    def _on_save_error(self, error_msg: str):
        """Called when save fails."""
        self.progress_bar.pack_forget()
        messagebox.showerror("Save Error", f"Save failed:\n{error_msg}")

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

    def _update_image_listbox(self):
        """Update image list display with tracking status."""
        self.image_listbox.delete(0, tk.END)

        for i, path in enumerate(self.image_list):
            # Determine status prefix
            if path in self.annotated_images:
                prefix = "[✓]"
            elif self.tracking_state.is_tracking_initialized:
                status = self.tracking_state.get_frame_status(i)
                if status == "low_confidence":
                    prefix = "[!]"
                elif status == "tracked":
                    prefix = "[T]"
                else:
                    prefix = "[ ]"
            else:
                prefix = "[ ]"

            display_name = f"{prefix} {path.name}"
            self.image_listbox.insert(tk.END, display_name)

            # Color code for tracking status
            if self.tracking_state.is_tracking_initialized:
                status = self.tracking_state.get_frame_status(i)
                if status == "low_confidence":
                    self.image_listbox.itemconfig(i, foreground="orange")
                elif status == "tracked":
                    self.image_listbox.itemconfig(i, foreground="green")

        # Select current image
        if self.image_list:
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(self.current_index)
            self.image_listbox.see(self.current_index)

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
        model_path = project_root / "models" / args.model
        if not model_path.exists():
            print(f"Error: Model file not found: {args.model}")
            print(f"Tried: {model_path}")
            sys.exit(1)

    # Create main window
    root = tk.Tk()

    # Set theme
    style = ttk.Style()
    available_themes = style.theme_names()
    if "clam" in available_themes:
        style.theme_use("clam")

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
