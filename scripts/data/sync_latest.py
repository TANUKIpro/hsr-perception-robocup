#!/usr/bin/env python3
"""Prepare the newest pybullet_hsr dump under an annotation_data/ root.

Thin wrapper around `prepare_dataset.ensure_dataset(..., latest=True)`.
Safe to run on every training attempt: if the existing local dataset is
already in sync with the newest dump, this is a no-op (prints status
and exits 0). Pass `--force` to rebuild anyway.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import ensure_dataset  # noqa: E402
from manifest import ManifestSchemaError  # noqa: E402


def _default_annotation_root() -> Path:
    # Mirrors PathCoordinator / CLAUDE.md defaults so the UI and the CLI agree.
    env = os.environ.get("PYBULLET_HSR_ROOT")
    if env:
        return Path(env) / "annotation_data"
    for candidate in (
        Path("/pybullet_hsr/annotation_data"),  # in-container default
        Path("/home/roboworks/repos/pybullet_hsr/annotation_data"),  # host default
    ):
        if candidate.is_dir():
            return candidate
    return Path("/pybullet_hsr/annotation_data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--annotation-root", type=Path, default=None,
        help="pybullet_hsr annotation_data/ directory "
             "(defaults to $PYBULLET_HSR_ROOT/annotation_data)",
    )
    ap.add_argument("--dest", type=Path, default=None,
                    help="destination root (defaults to datasets/<dataset_name>)")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-symlink", action="store_true",
                    help="copy files instead of symlinking (default: symlink)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the existing dataset is already in sync")
    args = ap.parse_args()

    annotation_root = args.annotation_root or _default_annotation_root()
    if not annotation_root.is_dir():
        raise SystemExit(
            f"annotation root {annotation_root} not found. "
            f"Set PYBULLET_HSR_ROOT or pass --annotation-root."
        )

    try:
        result = ensure_dataset(
            source=annotation_root,
            dest=args.dest,
            latest=True,
            val_ratio=args.val_ratio,
            seed=args.seed,
            symlink=not args.no_symlink,
            force=args.force,
        )
    except ManifestSchemaError as exc:
        raise SystemExit(str(exc))

    dump = result["dump_dir"]
    dest = result["dest"]
    action = result["action"]
    counts = result.get("counts", {})

    if action == "up-to-date":
        print(
            f"[sync_latest] dump={dump.name} dataset={result['dataset_name']} "
            f"status=up-to-date (use --force to rebuild)"
        )
    else:
        print(
            f"[sync_latest] dump={dump.name} dataset={result['dataset_name']} "
            f"status=prepared train={counts.get('train', 0)} val={counts.get('val', 0)}"
        )
    print(f"[sync_latest] data.yaml={dest / 'data.yaml'}")


if __name__ == "__main__":
    main()
