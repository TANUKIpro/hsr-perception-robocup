"""
Copy-Paste Augmentation Module

Provides functionality for Copy-Paste data augmentation:
- ObjectExtractor: Extract objects from images using SAM2 masks
- CopyPasteAugmentor: Paste objects onto background images with blending
"""

from .object_extractor import ObjectExtractor, ExtractedObject
from .copy_paste_augmentor import CopyPasteAugmentor, CopyPasteConfig

__all__ = [
    "ObjectExtractor",
    "ExtractedObject",
    "CopyPasteAugmentor",
    "CopyPasteConfig",
]
