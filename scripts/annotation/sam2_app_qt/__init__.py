"""
SAM2 Interactive Annotation Application (PyQt6 Version).

A PyQt6-based GUI application for interactive object annotation using SAM2.
Features:
- Point-click based segmentation (foreground/background points)
- Real-time mask preview with overlay
- Undo/Reset functionality for refinement
- Video tracking mode for batch annotation
- Batch image navigation
- YOLO format output

Usage:
    python -m sam2_app_qt --input-dir images/ --output-dir labels/ --class-id 0

Controls:
    - Left click: Add foreground point (include in mask)
    - Right click: Add background point (exclude from mask)
    - Ctrl+Z: Undo last point
    - Escape: Reset all points
    - Enter: Accept and save annotation
    - Arrow keys: Navigate between images
"""

from sam2_app_qt.main_window import SAM2AnnotationWindow

__all__ = ["SAM2AnnotationWindow"]
__version__ = "2.0.0"
