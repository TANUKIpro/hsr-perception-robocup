"""Path resolution helpers for the train + evaluate pipeline.

The repository is now a consumer of datasets produced by `pybullet_hsr`.
All paths here are project-root relative or absolute; there is no
per-profile isolation (profiles were removed along with the collection UI).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PYBULLET_HSR_ROOT = Path(
    os.environ.get("PYBULLET_HSR_ROOT", "/home/roboworks/repos/pybullet_hsr")
)
PYBULLET_HSR_ANNOTATION_ROOT = PYBULLET_HSR_ROOT / "annotation_data"

DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
FINETUNED_DIR = MODELS_DIR / "finetuned"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
TASKS_DIR = PROJECT_ROOT / "app_data" / "tasks"

# Wire the data scripts dir onto sys.path so we can import `manifest` from
# both CLI (relative import works) and Streamlit (no package context).
_SCRIPTS_DATA_DIR = PROJECT_ROOT / "scripts" / "data"
if str(_SCRIPTS_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DATA_DIR))


@st.cache_data(ttl=30, show_spinner=False)
def _cached_get_trained_models(finetuned_dir: str) -> List[Dict[str, Optional[str]]]:
    finetuned_path = Path(finetuned_dir)
    if not finetuned_path.exists():
        return []
    models: List[Dict[str, Optional[str]]] = []
    for model_dir in sorted(finetuned_path.iterdir(), reverse=True):
        if not model_dir.is_dir():
            continue
        best_pt = model_dir / "weights" / "best.pt"
        last_pt = model_dir / "weights" / "last.pt"
        if best_pt.exists() or last_pt.exists():
            models.append({
                "name": model_dir.name,
                "best_path": str(best_pt) if best_pt.exists() else None,
                "last_path": str(last_pt) if last_pt.exists() else None,
                "created": datetime.fromtimestamp(model_dir.stat().st_ctime).isoformat(),
            })
    return models


@st.cache_data(ttl=30, show_spinner=False)
def _cached_get_datasets(datasets_dir: str) -> List[Dict[str, str]]:
    path = Path(datasets_dir)
    if not path.exists():
        return []
    out: List[Dict[str, str]] = []
    for d in sorted(path.iterdir(), reverse=True):
        yaml_path = d / "data.yaml"
        if d.is_dir() and yaml_path.exists():
            out.append({
                "name": d.name,
                "path": str(d),
                "data_yaml": str(yaml_path),
                "created": datetime.fromtimestamp(d.stat().st_ctime).isoformat(),
            })
    return out


def _read_prepare_meta(dataset_dir: Path) -> Optional[Dict[str, Any]]:
    import json as _json
    path = dataset_dir / ".prepare_meta.json"
    if not path.is_file():
        return None
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass
class PathConfig:
    datasets_dir: Path = DATASETS_DIR
    models_dir: Path = MODELS_DIR
    finetuned_dir: Path = FINETUNED_DIR
    pretrained_dir: Path = PRETRAINED_DIR
    tasks_dir: Path = TASKS_DIR


class PathCoordinator:
    """Small façade around the new path layout."""

    def __init__(self, project_root: Optional[Union[str, Path]] = None) -> None:
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT
        self.config = PathConfig()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        for d in (
            self.config.datasets_dir,
            self.config.models_dir,
            self.config.finetuned_dir,
            self.config.pretrained_dir,
            self.config.tasks_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> Path:
        if not hasattr(self.config, key):
            raise KeyError(f"Unknown path key: {key}")
        return getattr(self.config, key)

    def resolve_path(self, path: Union[str, Path]) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.project_root / p

    def get_datasets(self) -> List[Dict[str, str]]:
        return _cached_get_datasets(str(self.config.datasets_dir))

    def get_trained_models(self) -> List[Dict[str, Optional[str]]]:
        return _cached_get_trained_models(str(self.config.finetuned_dir))

    def get_available_pybullet_hsr_dumps(self) -> List[Dict[str, Any]]:
        """Scan `${PYBULLET_HSR_ROOT}/annotation_data/` for dumps with a valid manifest.json.

        Returns records in the form `{"path": Path, "manifest": dict}`,
        sorted by `manifest.created_at` descending.
        """
        from manifest import discover_dumps  # lazy — avoid startup cost
        return discover_dumps(PYBULLET_HSR_ANNOTATION_ROOT)

    def get_dataset_meta(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Return the `.prepare_meta.json` sidecar for a prepared dataset, or None."""
        return _read_prepare_meta(self.config.datasets_dir / dataset_name)

    def get_latest_sync_state(self) -> Dict[str, Any]:
        """Compare the newest pybullet_hsr dump against the matching local dataset.

        Returns a dict with keys:
          - `state`: one of "no_dumps", "no_local", "stale", "up_to_date"
          - `dump`: the newest dump record ({"path", "manifest"}) or None
          - `local_meta`: the matching dataset's sidecar dict, or None
          - `local_dataset_name`: the dataset name we expect under `datasets/`
          - `reason`: short human-readable staleness reason (when stale)

        `stale` is reported when the dump's manifest created_at, path, or
        num_images differs from what the local sidecar recorded — that's
        enough to tell the user "there's a newer dump you haven't synced."
        """
        dumps = self.get_available_pybullet_hsr_dumps()
        if not dumps:
            return {"state": "no_dumps", "dump": None, "local_meta": None,
                    "local_dataset_name": None, "reason": ""}

        latest = dumps[0]
        manifest = latest["manifest"]
        dump_path = latest["path"]
        dataset_name = manifest.get("dataset_name") or dump_path.name
        meta = self.get_dataset_meta(dataset_name)

        base = {
            "dump": latest,
            "local_meta": meta,
            "local_dataset_name": dataset_name,
        }
        if not meta:
            return {**base, "state": "no_local",
                    "reason": f"datasets/{dataset_name} not prepared yet"}

        if meta.get("source_dump") != str(dump_path.resolve()):
            return {**base, "state": "stale",
                    "reason": f"local sidecar points at {meta.get('source_dump')!r}"}
        if meta.get("manifest_created_at") != manifest.get("created_at", ""):
            return {**base, "state": "stale",
                    "reason": f"dump created_at has advanced to {manifest.get('created_at')}"}
        stats = manifest.get("stats", {}) or {}
        if stats.get("num_images") and stats["num_images"] != meta.get("num_images_in_dump"):
            return {**base, "state": "stale",
                    "reason": "dump num_images changed since last prepare"}

        return {**base, "state": "up_to_date", "reason": ""}

    def get_pretrained_models(self) -> List[str]:
        models: List[str] = []
        if self.config.pretrained_dir.exists():
            for model_file in self.config.pretrained_dir.glob("*.pt"):
                models.append(str(model_file))
        for name in (
            "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
            "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        ):
            if name not in [Path(m).name for m in models]:
                models.append(name)
        return models

    def get_training_paths(self, dataset_yaml: Union[str, Path]) -> Dict[str, str]:
        yaml_path = self.resolve_path(dataset_yaml)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")
        return {
            "dataset_yaml": str(yaml_path),
            "output_dir": str(self.config.finetuned_dir),
        }

    def get_path_summary(self) -> Dict[str, str]:
        return {
            "datasets": str(self.config.datasets_dir),
            "models": str(self.config.models_dir),
            "finetuned": str(self.config.finetuned_dir),
            "pretrained": str(self.config.pretrained_dir),
            "pybullet_hsr_root": str(PYBULLET_HSR_ROOT),
            "pybullet_hsr_annotation_root": str(PYBULLET_HSR_ANNOTATION_ROOT),
        }
