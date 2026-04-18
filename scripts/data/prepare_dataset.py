#!/usr/bin/env python3
"""Convert a pybullet_hsr dataset dump into a YOLO train/val split + data.yaml.

Driven by the dump's `manifest.json` (schema v1.0) — the manifest carries
the images/labels layout, label format, and class list, so no external
`--classes-yaml` is needed. Pass `--latest` with an `annotation_data/`
root to auto-pick the newest manifest-bearing dump.

On a successful prepare, a sidecar `<dest>/.prepare_meta.json` is written
recording which dump + settings produced the split. Subsequent runs with
the same source are no-ops unless `--force` is passed, which makes it
safe to run this script on every training attempt (pipeline-style).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

try:
    from .manifest import (
        LABEL_FORMAT_NAMES,
        LABEL_FORMAT_NUMERIC,
        ManifestSchemaError,
        discover_dumps,
        load_manifest,
        name_to_id,
        ordered_names,
    )
except ImportError:
    # Allow direct invocation (`python scripts/data/prepare_dataset.py ...`).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from manifest import (  # type: ignore[no-redef]
        LABEL_FORMAT_NAMES,
        LABEL_FORMAT_NUMERIC,
        ManifestSchemaError,
        discover_dumps,
        load_manifest,
        name_to_id,
        ordered_names,
    )


PREPARE_META_FILENAME = ".prepare_meta.json"
DEFAULT_VAL_RATIO = 0.1
DEFAULT_SEED = 42


def place_file(src: Path, dst: Path, symlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def translate_label(src: Path, dst: Path, mapping: dict[str, int]) -> None:
    """Rewrite a label file, mapping each first-token class name to its id."""
    out_lines: list[str] = []
    for lineno, raw in enumerate(src.read_text().splitlines(), start=1):
        parts = raw.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise SystemExit(f"{src}:{lineno}: expected 5 tokens, got {len(parts)}")
        name = parts[0]
        if name not in mapping:
            raise SystemExit(f"{src}:{lineno}: class {name!r} not in manifest")
        out_lines.append(f"{mapping[name]} {' '.join(parts[1:])}")
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


def resolve_source(source: Path, latest: bool) -> Path:
    """Return the actual dump directory: source itself, or its newest child if --latest."""
    if not latest:
        return source
    candidates = discover_dumps(source)
    if not candidates:
        raise SystemExit(
            f"no manifest-bearing dumps under {source}. "
            f"Run pybullet_hsr/scripts/write_manifest.py on an existing dump first."
        )
    return candidates[0]["path"]


def _build_meta(
    *,
    dump_dir: Path,
    manifest: dict[str, Any],
    dest: Path,
    val_ratio: float,
    seed: int,
    symlink: bool,
    counts: dict[str, int],
) -> dict[str, Any]:
    stats = manifest.get("stats", {}) or {}
    return {
        "source_dump": str(dump_dir.resolve()),
        "dataset_name": manifest.get("dataset_name"),
        "manifest_created_at": manifest.get("created_at", ""),
        "label_format": manifest.get("label_format"),
        "num_images_in_dump": stats.get("num_images"),
        "val_ratio": val_ratio,
        "seed": seed,
        "symlink": symlink,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "counts": counts,
        "schema_version": 1,
    }


def read_meta(dest: Path) -> dict[str, Any] | None:
    path = dest / PREPARE_META_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_fresh(
    dest: Path,
    *,
    dump_dir: Path,
    manifest: dict[str, Any],
    val_ratio: float,
    seed: int,
    symlink: bool,
) -> bool:
    """Return True if `<dest>` was prepared from this dump with matching settings.

    A missing or unreadable sidecar counts as stale. Val ratio / seed /
    symlink mismatches also count as stale because the split and on-disk
    layout would differ.
    """
    meta = read_meta(dest)
    if not meta:
        return False
    if meta.get("source_dump") != str(dump_dir.resolve()):
        return False
    if meta.get("manifest_created_at") != manifest.get("created_at", ""):
        return False
    if float(meta.get("val_ratio", -1)) != float(val_ratio):
        return False
    if int(meta.get("seed", -1)) != int(seed):
        return False
    if bool(meta.get("symlink", None)) != bool(symlink):
        return False
    # Also require the dataset yaml to still exist — if someone rm'd the
    # dest, the sidecar alone shouldn't trick us into skipping.
    return (dest / "data.yaml").is_file()


def ensure_dataset(
    *,
    source: Path,
    dest: Path | None = None,
    latest: bool = False,
    val_ratio: float = DEFAULT_VAL_RATIO,
    seed: int = DEFAULT_SEED,
    symlink: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Prepare the YOLO split if needed and return metadata about what happened.

    Returns a dict with keys:
      - action: "prepared" | "up-to-date"
      - dest: Path
      - data_yaml: Path
      - dump_dir: Path
      - dataset_name: str
      - counts: {"train": int, "val": int, "skipped_no_label": int}
      - meta: the sidecar dict
    """
    dump_dir = resolve_source(source, latest)
    manifest = load_manifest(dump_dir)

    dataset_name = manifest["dataset_name"]
    label_format = manifest["label_format"]
    images_subdir = manifest["paths"]["images_subdir"]
    labels_subdir = manifest["paths"]["labels_subdir"]
    image_extension = manifest["image_extension"]
    names = ordered_names(manifest)
    mapping = name_to_id(manifest) if label_format == LABEL_FORMAT_NAMES else {}

    dest = dest or Path(__file__).resolve().parents[2] / "datasets" / dataset_name

    if not force and is_fresh(
        dest,
        dump_dir=dump_dir,
        manifest=manifest,
        val_ratio=val_ratio,
        seed=seed,
        symlink=symlink,
    ):
        meta = read_meta(dest) or {}
        return {
            "action": "up-to-date",
            "dest": dest,
            "data_yaml": dest / "data.yaml",
            "dump_dir": dump_dir,
            "dataset_name": dataset_name,
            "counts": meta.get("counts", {}),
            "meta": meta,
        }

    src_images = dump_dir / images_subdir
    src_labels = dump_dir / labels_subdir
    images = sorted(src_images.glob(f"*.{image_extension}"))
    if not images:
        raise SystemExit(f"no .{image_extension} files under {src_images}")

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    val_set = set(shuffled[:n_val])

    dst_dirs = {
        ("images", "train"): dest / "images" / "train",
        ("images", "val"): dest / "images" / "val",
        ("labels", "train"): dest / "labels" / "train",
        ("labels", "val"): dest / "labels" / "val",
    }
    for d in dst_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0, "skipped_no_label": 0}
    for img in images:
        label = src_labels / f"{img.stem}.txt"
        if not label.exists():
            counts["skipped_no_label"] += 1
            continue
        split = "val" if img in val_set else "train"
        place_file(img, dst_dirs[("images", split)] / img.name, symlink)
        if label_format == LABEL_FORMAT_NUMERIC:
            place_file(label, dst_dirs[("labels", split)] / label.name, symlink)
        else:
            translate_label(label, dst_dirs[("labels", split)] / label.name, mapping)
        counts[split] += 1

    data_yaml = {
        "path": str(dest.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(names)},
    }
    (dest / "data.yaml").write_text(
        yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True)
    )

    meta = _build_meta(
        dump_dir=dump_dir,
        manifest=manifest,
        dest=dest,
        val_ratio=val_ratio,
        seed=seed,
        symlink=symlink,
        counts=counts,
    )
    (dest / PREPARE_META_FILENAME).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {
        "action": "prepared",
        "dest": dest,
        "data_yaml": dest / "data.yaml",
        "dump_dir": dump_dir,
        "dataset_name": dataset_name,
        "counts": counts,
        "meta": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source", type=Path, required=True,
        help="pybullet_hsr dump root (contains manifest.json), "
             "or an annotation_data/ directory when --latest is used",
    )
    ap.add_argument(
        "--dest", type=Path, default=None,
        help="destination root for the YOLO dataset "
             "(defaults to datasets/<dataset_name> relative to repo root)",
    )
    ap.add_argument(
        "--latest", action="store_true",
        help="treat --source as an annotation_data/ root and pick the newest manifest-bearing dump",
    )
    ap.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--symlink", action="store_true",
        help="symlink images/labels instead of copying (images always "
             "symlinkable; labels only when label_format is yolo_numeric)",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="re-prepare even if the existing dataset is up-to-date with the source dump",
    )
    args = ap.parse_args()

    try:
        result = ensure_dataset(
            source=args.source,
            dest=args.dest,
            latest=args.latest,
            val_ratio=args.val_ratio,
            seed=args.seed,
            symlink=args.symlink,
            force=args.force,
        )
    except ManifestSchemaError as exc:
        raise SystemExit(str(exc))

    dump_dir = result["dump_dir"]
    dest = result["dest"]
    action = result["action"]
    counts = result.get("counts", {})

    if action == "up-to-date":
        print(
            f"source={dump_dir} dataset={result['dataset_name']} "
            f"status=up-to-date (use --force to rebuild)"
        )
        print(f"data.yaml={dest / 'data.yaml'}")
        return

    print(
        f"source={dump_dir} dataset={result['dataset_name']} status=prepared"
    )
    print(
        f"train={counts.get('train', 0)} val={counts.get('val', 0)} "
        f"skipped_no_label={counts.get('skipped_no_label', 0)}"
    )
    print(f"wrote {dest / 'data.yaml'}")


if __name__ == "__main__":
    main()
