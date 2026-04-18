"""Unit tests for scripts/data/manifest.py (reader + discover)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DATA = Path(__file__).resolve().parents[3] / "scripts" / "data"
if str(_SCRIPTS_DATA) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DATA))

from manifest import (  # noqa: E402
    LABEL_FORMAT_NAMES,
    LABEL_FORMAT_NUMERIC,
    ManifestSchemaError,
    discover_dumps,
    load_manifest,
    name_to_id,
    ordered_names,
)


def _write_manifest(dump_dir: Path, overrides: dict | None = None) -> dict:
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "fake",
        "created_at": "2026-04-19T00:00:00Z",
        "generator": {"repo": "test", "commit": "", "command": "t", "config": ""},
        "paths": {
            "images_subdir": "images/train_images",
            "labels_subdir": "labels/train_images",
        },
        "label_format": LABEL_FORMAT_NUMERIC,
        "image_extension": "png",
        "image_size": [640, 480],
        "classes": [
            {"id": 0, "name": "a"},
            {"id": 1, "name": "b"},
            {"id": 2, "name": "c"},
        ],
        "stats": {
            "num_images": 3,
            "num_labels": 3,
            "num_boxes": 4,
            "per_class_box_count": {"a": 2, "b": 1, "c": 1},
        },
    }
    if overrides:
        manifest.update(overrides)
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_load_manifest_valid(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    m = load_manifest(tmp_path)
    assert m["dataset_name"] == "fake"
    assert m["label_format"] == LABEL_FORMAT_NUMERIC
    assert name_to_id(m) == {"a": 0, "b": 1, "c": 2}
    assert ordered_names(m) == ["a", "b", "c"]


def test_load_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(ManifestSchemaError, match="not found"):
        load_manifest(tmp_path)


def test_load_manifest_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text("{not json")
    with pytest.raises(ManifestSchemaError, match="invalid JSON"):
        load_manifest(tmp_path)


def test_load_manifest_unsupported_version(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"schema_version": "2.0"})
    with pytest.raises(ManifestSchemaError, match="schema_version"):
        load_manifest(tmp_path)


def test_load_manifest_missing_required_key(tmp_path: Path) -> None:
    m = _write_manifest(tmp_path)
    del m["classes"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ManifestSchemaError, match="missing required keys"):
        load_manifest(tmp_path)


def test_load_manifest_bad_label_format(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"label_format": "coco"})
    with pytest.raises(ManifestSchemaError, match="unknown label_format"):
        load_manifest(tmp_path)


def test_ordered_names_requires_contiguous_ids(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {
        "classes": [{"id": 0, "name": "a"}, {"id": 2, "name": "c"}],
    })
    m = load_manifest(tmp_path)
    with pytest.raises(ManifestSchemaError, match="contiguous"):
        ordered_names(m)


def test_discover_dumps_sorts_by_created_at_desc(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "dump_old", {"created_at": "2026-01-01T00:00:00Z"})
    _write_manifest(tmp_path / "dump_new", {"created_at": "2026-04-01T00:00:00Z"})
    # A directory without a manifest must be skipped.
    (tmp_path / "empty").mkdir()
    results = discover_dumps(tmp_path)
    assert [r["path"].name for r in results] == ["dump_new", "dump_old"]


def test_discover_dumps_skips_invalid(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "good")
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "manifest.json").write_text("{broken")
    results = discover_dumps(tmp_path)
    assert [r["path"].name for r in results] == ["good"]


def test_discover_dumps_returns_empty_for_missing_root(tmp_path: Path) -> None:
    assert discover_dumps(tmp_path / "nope") == []


def test_yolo_names_format_roundtrip(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"label_format": LABEL_FORMAT_NAMES})
    m = load_manifest(tmp_path)
    assert m["label_format"] == LABEL_FORMAT_NAMES
    assert name_to_id(m)["b"] == 1
