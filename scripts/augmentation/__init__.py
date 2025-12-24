"""
Copy-Paste Augmentation Module

Provides functionality for Copy-Paste data augmentation:
- ObjectExtractor: Extract objects from images using SAM2 masks
- CopyPasteAugmentor: Paste objects onto background images with blending
- ParallelSyntheticGenerator: Parallel synthetic image generation
"""

from .object_extractor import ObjectExtractor, ExtractedObject, ObjectReference
from .copy_paste_augmentor import CopyPasteAugmentor, CopyPasteConfig
from .parallel_generator import (
    ParallelSyntheticGenerator,
    GenerationTask,
    GenerationResult,
)

__all__ = [
    "ObjectExtractor",
    "ExtractedObject",
    "ObjectReference",
    "CopyPasteAugmentor",
    "CopyPasteConfig",
    "ParallelSyntheticGenerator",
    "GenerationTask",
    "GenerationResult",
]
