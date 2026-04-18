"""End-to-end tests for scripts/data/prepare_dataset.py as a subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _PROJECT_ROOT / "scripts" / "data" / "prepare_dataset.py"


def _make_png(path: Path, color: tuple[int, int, int]) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), color).save(path)


def _seed_pair(dump: Path, stem: str, color: tuple[int, int, int], label_text: str) -> None:
    _make_png(dump / "images" / "train_images" / f"{stem}.png", color)
    label_path = dump / "labels" / "train_images" / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label_text)


def _write_manifest(dump_dir: Path, classes: list[tuple[int, str]], label_format: str) -> None:
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "t",
        "created_at": "2026-04-19T00:00:00Z",
        "generator": {"repo": "test", "commit": "", "command": "t", "config": ""},
        "paths": {
            "images_subdir": "images/train_images",
            "labels_subdir": "labels/train_images",
        },
        "label_format": label_format,
        "image_extension": "png",
        "image_size": [640, 480],
        "classes": [{"id": cid, "name": n} for cid, n in classes],
        "stats": {"num_images": 0, "num_labels": 0, "num_boxes": 0, "per_class_box_count": {}},
    }
    dump_dir.mkdir(parents=True, exist_ok=True)
    (dump_dir / "manifest.json").write_text(json.dumps(manifest))


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
    )


@pytest.fixture
def numeric_dump(tmp_path: Path) -> Path:
    dump = tmp_path / "dump"
    _write_manifest(dump, [(0, "a"), (1, "b"), (2, "c")], label_format="yolo_numeric")
    for i in range(1, 6):
        _seed_pair(dump, str(i), (i * 30, 50, 80), f"{i % 3} 0.5 0.5 0.1 0.1\n")
    return dump


def test_numeric_dump_splits_and_writes_data_yaml(numeric_dump: Path, tmp_path: Path) -> None:
    dest = tmp_path / "out"
    result = _run([
        "--source", str(numeric_dump),
        "--dest", str(dest),
        "--symlink",
        "--val-ratio", "0.2",
    ])
    assert result.returncode == 0, result.stderr
    data_yaml = yaml.safe_load((dest / "data.yaml").read_text())
    assert data_yaml["names"] == {0: "a", 1: "b", 2: "c"}
    train_images = list((dest / "images" / "train").iterdir())
    val_images = list((dest / "images" / "val").iterdir())
    assert len(train_images) + len(val_images) == 5


def test_yolo_names_labels_translated_to_numeric(tmp_path: Path) -> None:
    dump = tmp_path / "dump"
    _write_manifest(dump, [(0, "cat"), (1, "dog")], label_format="yolo_names")
    _seed_pair(dump, "1", (10, 20, 30), "cat 0.5 0.5 0.1 0.1\n")
    _seed_pair(dump, "2", (40, 50, 60), "dog 0.3 0.3 0.2 0.2\n")

    dest = tmp_path / "out"
    result = _run(["--source", str(dump), "--dest", str(dest), "--val-ratio", "0.5"])
    assert result.returncode == 0, result.stderr

    all_labels = list((dest / "labels" / "train").glob("*.txt")) + \
                 list((dest / "labels" / "val").glob("*.txt"))
    seen_first_tokens = {p.read_text().split()[0] for p in all_labels}
    assert seen_first_tokens == {"0", "1"}


def test_missing_manifest_errors_with_guidance(tmp_path: Path) -> None:
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    result = _run(["--source", str(empty), "--dest", str(tmp_path / "out")])
    assert result.returncode != 0
    assert "write_manifest.py" in (result.stdout + result.stderr)


def test_latest_picks_newest_manifest(tmp_path: Path) -> None:
    root = tmp_path / "annotation_data"
    root.mkdir()
    old = root / "dump_old"
    new = root / "dump_new"
    _write_manifest(old, [(0, "a")], label_format="yolo_numeric")
    _write_manifest(new, [(0, "a")], label_format="yolo_numeric")
    # Override created_at so `new` wins regardless of filesystem order.
    new_manifest = json.loads((new / "manifest.json").read_text())
    new_manifest["created_at"] = "2026-12-31T00:00:00Z"
    (new / "manifest.json").write_text(json.dumps(new_manifest))
    old_manifest = json.loads((old / "manifest.json").read_text())
    old_manifest["created_at"] = "2026-01-01T00:00:00Z"
    (old / "manifest.json").write_text(json.dumps(old_manifest))
    # Seed both dumps with a single image/label so prepare doesn't error.
    for dump in (old, new):
        _seed_pair(dump, "1", (1, 1, 1), "0 0.5 0.5 0.1 0.1\n")

    dest = tmp_path / "out"
    result = _run([
        "--source", str(root),
        "--latest",
        "--dest", str(dest),
        "--symlink",
    ])
    assert result.returncode == 0, result.stderr
    assert "dump_new" in result.stdout or "dump_new" in result.stderr


def test_latest_errors_when_no_manifest_found(tmp_path: Path) -> None:
    root = tmp_path / "annotation_data"
    (root / "dump").mkdir(parents=True)
    result = _run([
        "--source", str(root),
        "--latest",
        "--dest", str(tmp_path / "out"),
    ])
    assert result.returncode != 0
    assert "no manifest-bearing dumps" in (result.stdout + result.stderr).lower()
