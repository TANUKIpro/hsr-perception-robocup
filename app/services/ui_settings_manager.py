"""UI settings persistence.

Stores a small JSON under `app_data/ui_settings.json` so advanced-training
sliders and evaluation confidence thresholds survive reloads. Registry /
profile / synthetic persistence was dropped with the rest of the
collection pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from .path_coordinator import PathCoordinator


@dataclass
class TrainingAdvancedParams:
    # Data augmentation — HSV
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    # Data augmentation — geometric
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    # Data augmentation — flip & advanced
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 0.7
    mixup: float = 0.1
    # Optimizer — learning rate
    optimizer: str = "AdamW"
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.001
    # LLRD
    llrd_enabled: bool = False
    llrd_decay_rate: float = 0.9
    # SWA
    swa_enabled: bool = True
    swa_start_epoch: int = 10
    swa_lr: float = 0.0005
    # Overfitting prevention
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
class EvaluationParams:
    visual_conf: float = 0.25


@dataclass
class UISettings:
    version: str = "2.0.0"
    updated_at: str = ""
    current_preset: str = "Competition"
    training_params: TrainingAdvancedParams = field(default_factory=TrainingAdvancedParams)
    evaluation: EvaluationParams = field(default_factory=EvaluationParams)


class UISettingsManager:
    """Persists UI widget state between Streamlit runs."""

    SETTINGS_FILE = "ui_settings.json"

    def __init__(self, path_coordinator: "PathCoordinator") -> None:
        self.path_coordinator = path_coordinator

    def _settings_path(self) -> Path:
        app_data_dir = self.path_coordinator.project_root / "app_data"
        app_data_dir.mkdir(parents=True, exist_ok=True)
        return app_data_dir / self.SETTINGS_FILE

    def load(self) -> UISettings:
        path = self._settings_path()
        if not path.exists():
            return UISettings()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return UISettings()
        return self._from_dict(data)

    def save(self, settings: UISettings) -> None:
        settings.updated_at = datetime.now().isoformat()
        self._settings_path().write_text(
            json.dumps(self._to_dict(settings), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def load_to_session_state(self) -> None:
        settings = self.load()
        st.session_state.current_preset = settings.current_preset
        for key, value in asdict(settings.training_params).items():
            st.session_state[f"adv_{key}"] = value
        st.session_state.visual_conf = settings.evaluation.visual_conf
        st.session_state._ui_settings_loaded = True

    def save_from_session_state(self) -> None:
        settings = UISettings()
        settings.current_preset = st.session_state.get("current_preset", "Competition")
        for key in asdict(settings.training_params):
            session_key = f"adv_{key}"
            if session_key in st.session_state:
                setattr(settings.training_params, key, st.session_state[session_key])
        settings.evaluation.visual_conf = st.session_state.get("visual_conf", 0.25)
        self.save(settings)

    @staticmethod
    def _to_dict(settings: UISettings) -> dict:
        return {
            "version": settings.version,
            "updated_at": settings.updated_at,
            "current_preset": settings.current_preset,
            "training": {"advanced_params": asdict(settings.training_params)},
            "evaluation": asdict(settings.evaluation),
        }

    @staticmethod
    def _from_dict(data: dict) -> UISettings:
        settings = UISettings()
        settings.version = data.get("version", settings.version)
        settings.updated_at = data.get("updated_at", "")
        settings.current_preset = data.get("current_preset", "Competition")
        training = data.get("training", {})
        for key, value in training.get("advanced_params", {}).items():
            if hasattr(settings.training_params, key):
                setattr(settings.training_params, key, value)
        for key, value in data.get("evaluation", {}).items():
            if hasattr(settings.evaluation, key):
                setattr(settings.evaluation, key, value)
        return settings
