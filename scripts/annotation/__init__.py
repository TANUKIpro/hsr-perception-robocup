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
)

__all__ = [
    "bbox_to_yolo",
    "yolo_to_bbox",
    "write_yolo_label",
    "read_yolo_label",
    "validate_yolo_annotation",
    "create_dataset_yaml",
    "split_dataset",
    "AnnotationResult",
]
