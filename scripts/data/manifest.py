"""Reader + discovery for pybullet_hsr dataset `manifest.json` (schema v1.0).

We treat the manifest as the single source of truth for a dump's layout,
label format, and class list. The schema lives in the pybullet_hsr repo
(`src/blenderproc_hsr/manifest.py`); we intentionally don't import it
here to keep the two repositories loosely coupled — only the stable
parts of v1.0 are consumed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_SCHEMA_VERSIONS = {"1.0"}

MANIFEST_FILENAME = "manifest.json"

LABEL_FORMAT_NUMERIC = "yolo_numeric"
LABEL_FORMAT_NAMES = "yolo_names"

_REQUIRED_TOP_LEVEL = (
    "schema_version",
    "dataset_name",
    "created_at",
    "paths",
    "label_format",
    "image_extension",
    "classes",
    "stats",
)
_REQUIRED_PATHS = ("images_subdir", "labels_subdir")


class ManifestSchemaError(RuntimeError):
    """Raised when a manifest is missing, malformed, or on an unsupported schema version."""


def load_manifest(dump_dir: Path) -> dict[str, Any]:
    """Read `<dump_dir>/manifest.json` and minimally validate it.

    Raises `ManifestSchemaError` on any of: missing file, invalid JSON,
    unsupported `schema_version`, or missing required top-level keys.
    """
    path = dump_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise ManifestSchemaError(
            f"{path} not found. Run pybullet_hsr/scripts/write_manifest.py "
            f"--dump-dir {dump_dir} to generate it."
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestSchemaError(f"{path}: invalid JSON — {exc}") from exc

    if not isinstance(manifest, dict):
        raise ManifestSchemaError(f"{path}: top-level value must be a JSON object")

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in manifest]
    if missing:
        raise ManifestSchemaError(f"{path}: missing required keys {missing}")

    version = manifest["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestSchemaError(
            f"{path}: schema_version {version!r} not supported "
            f"(this repo reads {sorted(SUPPORTED_SCHEMA_VERSIONS)})"
        )

    paths = manifest["paths"]
    if not isinstance(paths, dict):
        raise ManifestSchemaError(f"{path}: 'paths' must be an object")
    missing_paths = [k for k in _REQUIRED_PATHS if k not in paths]
    if missing_paths:
        raise ManifestSchemaError(f"{path}: paths is missing {missing_paths}")

    lfmt = manifest["label_format"]
    if lfmt not in (LABEL_FORMAT_NUMERIC, LABEL_FORMAT_NAMES):
        raise ManifestSchemaError(f"{path}: unknown label_format {lfmt!r}")

    classes = manifest["classes"]
    if not isinstance(classes, list) or not classes:
        raise ManifestSchemaError(f"{path}: 'classes' must be a non-empty list")
    for i, cls in enumerate(classes):
        if not isinstance(cls, dict) or "id" not in cls or "name" not in cls:
            raise ManifestSchemaError(f"{path}: classes[{i}] missing id or name")

    return manifest


def name_to_id(manifest: dict[str, Any]) -> dict[str, int]:
    """Return {class_name: class_id}."""
    return {c["name"]: int(c["id"]) for c in manifest["classes"]}


def ordered_names(manifest: dict[str, Any]) -> list[str]:
    """Return class names ordered by id (contiguous 0..N-1)."""
    pairs = sorted((int(c["id"]), c["name"]) for c in manifest["classes"])
    ids = [i for i, _ in pairs]
    if ids != list(range(len(ids))):
        raise ManifestSchemaError(
            f"manifest classes ids are not contiguous 0..N-1: {ids}"
        )
    return [name for _, name in pairs]


def discover_dumps(annotation_root: Path) -> list[dict[str, Any]]:
    """Scan `annotation_root` for sub-directories with a valid manifest.json.

    Returns a list of `{path, manifest}` records, sorted by manifest
    `created_at` descending. Invalid / absent manifests are skipped
    silently — callers that want to surface problems should look at
    `list(annotation_root.iterdir())` separately.
    """
    if not annotation_root.is_dir():
        return []
    results: list[dict[str, Any]] = []
    for entry in annotation_root.iterdir():
        if not entry.is_dir():
            continue
        try:
            manifest = load_manifest(entry)
        except ManifestSchemaError:
            continue
        results.append({"path": entry, "manifest": manifest})
    results.sort(key=lambda r: r["manifest"].get("created_at", ""), reverse=True)
    return results
