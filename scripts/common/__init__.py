"""
HSR Perception - Common Utilities Module

Shared utilities for annotation, training, and evaluation.
Centralizes commonly used constants, image processing, and configuration.
"""

from .constants import (
    # Image extensions
    IMAGE_EXTENSIONS,
    IMAGE_EXTENSIONS_ALL,
    # Competition defaults
    DEFAULT_TARGET_SAMPLES,
    DEFAULT_BURST_INTERVAL,
    MIN_SAMPLES_FOR_TRAINING,
    DEFAULT_TRAIN_RATIO,
    # Model/inference defaults
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_IOU_THRESHOLD,
    TARGET_MAP50,
    TARGET_INFERENCE_MS,
    # Annotation defaults
    DEFAULT_BBOX_MARGIN_RATIO,
    DEFAULT_MIN_CONTOUR_AREA,
    DEFAULT_MAX_CONTOUR_AREA_RATIO,
    # Visualization defaults
    DEFAULT_BBOX_COLOR,
    DEFAULT_MASK_COLOR,
    DEFAULT_MASK_ALPHA,
    DEFAULT_BBOX_THICKNESS,
    DEFAULT_FONT_SCALE,
    DEFAULT_FONT_THICKNESS,
    # SAM2 defaults
    SAM2_DEFAULT_POINTS_PER_SIDE,
    SAM2_DEFAULT_PRED_IOU_THRESH,
    SAM2_DEFAULT_STABILITY_SCORE_THRESH,
    SAM2_DEFAULT_MIN_MASK_REGION_AREA,
    # Training defaults
    DEFAULT_YOLO_MODEL,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PATIENCE,
)

from .device_utils import (
    check_cuda_available,
    get_default_device,
    log_gpu_status,
    get_gpu_info,
    get_optimal_batch_size,
)

from .image_utils import (
    mask_to_bbox,
    find_object_bbox,
    draw_bbox,
    draw_mask_overlay,
    draw_detections,
    list_image_files,
    load_image,
    save_image,
)

from .config_utils import (
    # Annotator configs
    AnnotatorConfig,
    BackgroundSubtractionConfig,
    SAM2Config,
    get_sam2_model_config,
    # Training config
    TrainingConfig,
    # Evaluation config
    EvaluationConfig,
    # Class config loading
    load_class_config,
    get_class_names,
    get_class_id_map,
)

from .validation import (
    ErrorSeverity,
    PipelineError,
    ValidationResult,
    validate_dataset_yaml,
    validate_yolo_annotation,
    validate_model_path,
)

from .model_utils import (
    resolve_model_path,
    get_pretrained_model_path,
    get_finetuned_model_path,
    is_model_cached,
)

__all__ = [
    # Constants
    "IMAGE_EXTENSIONS",
    "IMAGE_EXTENSIONS_ALL",
    "DEFAULT_TARGET_SAMPLES",
    "DEFAULT_BURST_INTERVAL",
    "MIN_SAMPLES_FOR_TRAINING",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "TARGET_MAP50",
    "TARGET_INFERENCE_MS",
    "DEFAULT_BBOX_MARGIN_RATIO",
    "DEFAULT_MIN_CONTOUR_AREA",
    "DEFAULT_MAX_CONTOUR_AREA_RATIO",
    "DEFAULT_BBOX_COLOR",
    "DEFAULT_MASK_COLOR",
    "DEFAULT_MASK_ALPHA",
    "DEFAULT_BBOX_THICKNESS",
    "DEFAULT_FONT_SCALE",
    "DEFAULT_FONT_THICKNESS",
    "SAM2_DEFAULT_POINTS_PER_SIDE",
    "SAM2_DEFAULT_PRED_IOU_THRESH",
    "SAM2_DEFAULT_STABILITY_SCORE_THRESH",
    "SAM2_DEFAULT_MIN_MASK_REGION_AREA",
    "DEFAULT_YOLO_MODEL",
    "DEFAULT_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_PATIENCE",
    # Device utilities
    "check_cuda_available",
    "get_default_device",
    "log_gpu_status",
    "get_gpu_info",
    "get_optimal_batch_size",
    # Image utilities
    "mask_to_bbox",
    "find_object_bbox",
    "draw_bbox",
    "draw_mask_overlay",
    "draw_detections",
    "list_image_files",
    "load_image",
    "save_image",
    # Config utilities
    "AnnotatorConfig",
    "BackgroundSubtractionConfig",
    "SAM2Config",
    "get_sam2_model_config",
    "TrainingConfig",
    "EvaluationConfig",
    "load_class_config",
    "get_class_names",
    "get_class_id_map",
    # Validation
    "ErrorSeverity",
    "PipelineError",
    "ValidationResult",
    "validate_dataset_yaml",
    "validate_yolo_annotation",
    "validate_model_path",
    # Model utilities
    "resolve_model_path",
    "get_pretrained_model_path",
    "get_finetuned_model_path",
    "is_model_cached",
]
