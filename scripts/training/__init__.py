"""
HSR Perception - Training Module

Competition day fine-tuning scripts for YOLOv8 models.
"""

from .quick_finetune import CompetitionTrainer, COMPETITION_CONFIG

__all__ = [
    "CompetitionTrainer",
    "COMPETITION_CONFIG",
]
