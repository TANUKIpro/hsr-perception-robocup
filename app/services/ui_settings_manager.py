"""
UI Settings Manager

Manages UI settings persistence for each profile.
Settings are stored in app_data/ui_settings.json within each profile directory.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING
import json
import streamlit as st

if TYPE_CHECKING:
    from .path_coordinator import PathCoordinator


# =============================================================================
# Data Classes for Settings
# =============================================================================

@dataclass
class TrainingAdvancedParams:
    """Advanced training parameters matching PRESETS in training_advanced_params.py."""
    # Data Augmentation - HSV
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    # Data Augmentation - Geometric
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    # Data Augmentation - Flip & Advanced
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.7
    mixup: float = 0.1
    # Optimizer - Learning Rate
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.001
    # Optimizer - LLRD
    llrd_enabled: bool = False
    llrd_decay_rate: float = 0.9
    # Optimizer - SWA
    swa_enabled: bool = True
    swa_start_epoch: int = 10
    swa_lr: float = 0.0005
    # Overfitting Prevention
    label_smoothing: float = 0.05
    cos_lr: bool = True
    multi_scale: bool = False
    # Performance
    workers: int = 8
    cache: bool = True
    amp: bool = True
    imgsz: int = 640
    patience: int = 10
    close_mosaic: int = 20
    freeze: int = 10
    warmup_epochs: int = 3
    # Checkpoint
    save: bool = True
    save_period: int = 5
    exist_ok: bool = True


@dataclass
class SyntheticParams:
    """Synthetic image generation parameters."""
    enabled: bool = True
    ratio: float = 2.0
    max_objects: int = 3
    min_objects: int = 1
    scale_min: float = 0.5
    scale_max: float = 1.5
    rotation_min: int = -15
    rotation_max: int = 15
    enable_horizontal_flip: bool = True
    enable_vertical_flip: bool = False
    enable_white_balance: bool = True
    white_balance_strength: float = 0.7
    edge_blur_sigma: float = 2.0
    allow_overlap: bool = False
    overlap_iou_threshold: float = 0.1
    output_quality: int = 95


@dataclass
class DatasetPreparationParams:
    """Dataset preparation parameters."""
    validation_ratio: float = 0.2
    group_continuous_frames: bool = True
    group_interval_sec: float = 2.0


@dataclass
class EvaluationParams:
    """Evaluation page parameters."""
    visual_conf: float = 0.25
    video_conf: float = 0.25
    robustness_conf: float = 0.25
    xtion_conf: float = 0.25


@dataclass
class UISettings:
    """Root class containing all UI settings."""
    version: str = "1.0.0"
    updated_at: str = ""
    current_preset: str = "Competition"
    training_params: TrainingAdvancedParams = field(default_factory=TrainingAdvancedParams)
    training_synthetic: SyntheticParams = field(default_factory=SyntheticParams)
    annotation_synthetic: SyntheticParams = field(default_factory=SyntheticParams)
    dataset_preparation: DatasetPreparationParams = field(default_factory=DatasetPreparationParams)
    evaluation: EvaluationParams = field(default_factory=EvaluationParams)


# =============================================================================
# UI Settings Manager
# =============================================================================

class UISettingsManager:
    """
    Manages UI settings persistence for each profile.

    Settings are stored in app_data/ui_settings.json within each profile directory.
    This ensures settings are preserved across:
    - Page reloads
    - Application restarts
    - Profile exports/imports
    """

    SETTINGS_FILE = "ui_settings.json"

    def __init__(self, path_coordinator: "PathCoordinator"):
        """
        Initialize UISettingsManager.

        Args:
            path_coordinator: PathCoordinator instance for resolving paths
        """
        self.path_coordinator = path_coordinator

    def _get_settings_path(self) -> Path:
        """Get the path to the settings file for the current profile."""
        app_data_dir = self.path_coordinator.get_path("app_data_dir")
        return app_data_dir / self.SETTINGS_FILE

    def load(self) -> UISettings:
        """
        Load settings from file.

        Returns:
            UISettings instance (defaults if file doesn't exist)
        """
        settings_path = self._get_settings_path()

        if not settings_path.exists():
            return UISettings()

        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return self._dict_to_settings(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Return defaults if file is corrupted
            return UISettings()

    def save(self, settings: UISettings) -> None:
        """
        Save settings to file.

        Args:
            settings: UISettings instance to save
        """
        settings.updated_at = datetime.now().isoformat()
        settings_path = self._get_settings_path()

        # Ensure directory exists
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        data = self._settings_to_dict(settings)

        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_to_session_state(self) -> None:
        """
        Load settings from file and apply to Streamlit session state.

        This populates all UI widget values from the saved settings.
        """
        settings = self.load()

        # Current preset
        st.session_state.current_preset = settings.current_preset

        # Training advanced params (adv_* prefix)
        params = settings.training_params
        for key, value in asdict(params).items():
            st.session_state[f"adv_{key}"] = value

        # Training synthetic params (synthetic_* prefix)
        synth = settings.training_synthetic
        st.session_state.enable_dynamic_synthetic = synth.enabled
        st.session_state.synthetic_ratio = synth.ratio
        st.session_state.synthetic_max_objects = synth.max_objects
        st.session_state.synthetic_min_objects = synth.min_objects
        st.session_state.synthetic_scale_min = synth.scale_min
        st.session_state.synthetic_scale_max = synth.scale_max
        st.session_state.synthetic_rotation_min = synth.rotation_min
        st.session_state.synthetic_rotation_max = synth.rotation_max
        st.session_state.synthetic_enable_hflip = synth.enable_horizontal_flip
        st.session_state.synthetic_enable_vflip = synth.enable_vertical_flip
        st.session_state.synthetic_enable_wb = synth.enable_white_balance
        st.session_state.synthetic_wb_strength = synth.white_balance_strength
        st.session_state.synthetic_edge_blur = synth.edge_blur_sigma

        # Annotation synthetic params (synth_* prefix)
        ann_synth = settings.annotation_synthetic
        st.session_state.synth_ratio = ann_synth.ratio
        st.session_state.synth_max_obj = ann_synth.max_objects
        st.session_state.synth_min_obj = ann_synth.min_objects
        st.session_state.synth_scale_min = ann_synth.scale_min
        st.session_state.synth_scale_max = ann_synth.scale_max
        st.session_state.synth_rot_min = ann_synth.rotation_min
        st.session_state.synth_rot_max = ann_synth.rotation_max
        st.session_state.synth_hflip = ann_synth.enable_horizontal_flip
        st.session_state.synth_vflip = ann_synth.enable_vertical_flip
        st.session_state.synth_wb = ann_synth.enable_white_balance
        st.session_state.synth_wb_strength = ann_synth.white_balance_strength
        st.session_state.synth_edge_blur = ann_synth.edge_blur_sigma
        st.session_state.synth_overlap = ann_synth.allow_overlap
        st.session_state.synth_overlap_iou = ann_synth.overlap_iou_threshold
        st.session_state.synth_jpeg_quality = ann_synth.output_quality

        # Dataset preparation params
        ds_prep = settings.dataset_preparation
        st.session_state.val_ratio = ds_prep.validation_ratio
        st.session_state.group_continuous = ds_prep.group_continuous_frames
        st.session_state.group_interval = ds_prep.group_interval_sec

        # Evaluation params
        ev = settings.evaluation
        st.session_state.visual_conf = ev.visual_conf
        st.session_state.video_conf = ev.video_conf
        st.session_state.robustness_conf = ev.robustness_conf
        st.session_state.xtion_conf_threshold = ev.xtion_conf

        # Mark as loaded
        st.session_state._ui_settings_loaded = True

    def save_from_session_state(self) -> None:
        """
        Collect settings from Streamlit session state and save to file.

        This gathers all UI widget values and persists them.
        """
        settings = UISettings()

        # Current preset
        settings.current_preset = st.session_state.get("current_preset", "Competition")

        # Training advanced params
        params = settings.training_params
        for key in asdict(params).keys():
            session_key = f"adv_{key}"
            if session_key in st.session_state:
                setattr(params, key, st.session_state[session_key])

        # Training synthetic params
        synth = settings.training_synthetic
        synth.enabled = st.session_state.get("enable_dynamic_synthetic", True)
        synth.ratio = st.session_state.get("synthetic_ratio", 2.0)
        synth.max_objects = st.session_state.get("synthetic_max_objects", 3)
        synth.min_objects = st.session_state.get("synthetic_min_objects", 1)
        synth.scale_min = st.session_state.get("synthetic_scale_min", 0.5)
        synth.scale_max = st.session_state.get("synthetic_scale_max", 1.5)
        synth.rotation_min = st.session_state.get("synthetic_rotation_min", -15)
        synth.rotation_max = st.session_state.get("synthetic_rotation_max", 15)
        synth.enable_horizontal_flip = st.session_state.get("synthetic_enable_hflip", True)
        synth.enable_vertical_flip = st.session_state.get("synthetic_enable_vflip", False)
        synth.enable_white_balance = st.session_state.get("synthetic_enable_wb", True)
        synth.white_balance_strength = st.session_state.get("synthetic_wb_strength", 0.7)
        synth.edge_blur_sigma = st.session_state.get("synthetic_edge_blur", 2.0)

        # Annotation synthetic params
        ann_synth = settings.annotation_synthetic
        ann_synth.ratio = st.session_state.get("synth_ratio", 2.0)
        ann_synth.max_objects = st.session_state.get("synth_max_obj", 3)
        ann_synth.min_objects = st.session_state.get("synth_min_obj", 1)
        ann_synth.scale_min = st.session_state.get("synth_scale_min", 0.5)
        ann_synth.scale_max = st.session_state.get("synth_scale_max", 1.5)
        ann_synth.rotation_min = st.session_state.get("synth_rot_min", -15)
        ann_synth.rotation_max = st.session_state.get("synth_rot_max", 15)
        ann_synth.enable_horizontal_flip = st.session_state.get("synth_hflip", True)
        ann_synth.enable_vertical_flip = st.session_state.get("synth_vflip", False)
        ann_synth.enable_white_balance = st.session_state.get("synth_wb", True)
        ann_synth.white_balance_strength = st.session_state.get("synth_wb_strength", 0.7)
        ann_synth.edge_blur_sigma = st.session_state.get("synth_edge_blur", 2.0)
        ann_synth.allow_overlap = st.session_state.get("synth_overlap", False)
        ann_synth.overlap_iou_threshold = st.session_state.get("synth_overlap_iou", 0.1)
        ann_synth.output_quality = st.session_state.get("synth_jpeg_quality", 95)

        # Dataset preparation params
        ds_prep = settings.dataset_preparation
        ds_prep.validation_ratio = st.session_state.get("val_ratio", 0.2)
        ds_prep.group_continuous_frames = st.session_state.get("group_continuous", True)
        ds_prep.group_interval_sec = st.session_state.get("group_interval", 2.0)

        # Evaluation params
        ev = settings.evaluation
        ev.visual_conf = st.session_state.get("visual_conf", 0.25)
        ev.video_conf = st.session_state.get("video_conf", 0.25)
        ev.robustness_conf = st.session_state.get("robustness_conf", 0.25)
        ev.xtion_conf = st.session_state.get("xtion_conf_threshold", 0.25)

        self.save(settings)

    def _settings_to_dict(self, settings: UISettings) -> dict:
        """Convert UISettings to a dictionary for JSON serialization."""
        return {
            "version": settings.version,
            "updated_at": settings.updated_at,
            "current_preset": settings.current_preset,
            "training": {
                "advanced_params": asdict(settings.training_params),
                "synthetic": asdict(settings.training_synthetic),
            },
            "annotation": {
                "synthetic_generation": asdict(settings.annotation_synthetic),
                "dataset_preparation": asdict(settings.dataset_preparation),
            },
            "evaluation": asdict(settings.evaluation),
        }

    def _dict_to_settings(self, data: dict) -> UISettings:
        """Convert a dictionary to UISettings, handling missing fields gracefully."""
        settings = UISettings()
        settings.version = data.get("version", "1.0.0")
        settings.updated_at = data.get("updated_at", "")
        settings.current_preset = data.get("current_preset", "Competition")

        # Training params
        if "training" in data:
            training = data["training"]
            if "advanced_params" in training:
                for key, value in training["advanced_params"].items():
                    if hasattr(settings.training_params, key):
                        setattr(settings.training_params, key, value)
            if "synthetic" in training:
                for key, value in training["synthetic"].items():
                    if hasattr(settings.training_synthetic, key):
                        setattr(settings.training_synthetic, key, value)

        # Annotation params
        if "annotation" in data:
            annotation = data["annotation"]
            if "synthetic_generation" in annotation:
                for key, value in annotation["synthetic_generation"].items():
                    if hasattr(settings.annotation_synthetic, key):
                        setattr(settings.annotation_synthetic, key, value)
            if "dataset_preparation" in annotation:
                for key, value in annotation["dataset_preparation"].items():
                    if hasattr(settings.dataset_preparation, key):
                        setattr(settings.dataset_preparation, key, value)

        # Evaluation params
        if "evaluation" in data:
            for key, value in data["evaluation"].items():
                if hasattr(settings.evaluation, key):
                    setattr(settings.evaluation, key, value)

        return settings
