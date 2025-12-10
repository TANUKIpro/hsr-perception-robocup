"""
HSR Perception - Annotation Module

Auto-annotation tools for competition day workflow.
"""

from .annotation_utils import (
    bbox_to_yolo,
    yolo_to_bbox,
    write_yolo_label,
    read_yolo_label,
    validate_yolo_annotation,
    create_dataset_yaml,
    split_dataset,
    AnnotationResult,
    mask_to_bbox,
)

from .base_annotator import BaseAnnotator

from .visualization import (
    AnnotationColors,
    AnnotationStyle,
    draw_bbox_on_image,
    draw_mask_overlay,
    visualize_annotation_result,
)

__all__ = [
    # Core utilities
    "bbox_to_yolo",
    "yolo_to_bbox",
    "write_yolo_label",
    "read_yolo_label",
    "validate_yolo_annotation",
    "create_dataset_yaml",
    "split_dataset",
    "AnnotationResult",
    "mask_to_bbox",
    # Base class
    "BaseAnnotator",
    # Visualization
    "AnnotationColors",
    "AnnotationStyle",
    "draw_bbox_on_image",
    "draw_mask_overlay",
    "visualize_annotation_result",
]
