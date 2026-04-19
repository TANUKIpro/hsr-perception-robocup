# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
Although the text is written in English, please respond to any questions or clarifications from users in Japanese.

## Project Overview

HSR (Human Support Robot) perception pipeline for RoboCup@Home competitions. This branch is a
**training + evaluation consumer** of datasets produced by the separate `pybullet_hsr` repository
(BlenderProc-based synthetic data generation). Data collection, auto-annotation, and the ROS2
capture node used to live here but have been removed — they are no longer in scope.

**Environment**: Ubuntu 22.04 / Python 3.10 / CUDA 12.1 (Docker)

**Companion repo**: `/home/roboworks/repos/pybullet_hsr`
- Dataset dumps live under `pybullet_hsr/annotation_data/<name>_<YYYYMMDD_HHMM>/`
- Each dump carries a **`manifest.json` (schema v1.0)** declaring paths, label format, class list, and
  stats. This repo reads the manifest as the single source of truth; `configs/datasets/*.yaml` is
  not consumed directly anymore.
- If a dump was generated before the manifest was added, run pybullet_hsr's
  `scripts/write_manifest.py --dump-dir <dir> --classes-yaml configs/datasets/<name>.yaml`
  once to emit the manifest retroactively.

## Pipeline (3 steps)

```
prepare-dataset → train → evaluate
  (scripts/data/  (scripts/training/  (scripts/evaluation/
   prepare_dataset.py)  quick_finetune.py)   evaluate_model.py)
```

1. **prepare-dataset** — Read `<dump>/manifest.json`, split images/labels into train/val, write
   `data.yaml`. Labels in `yolo_numeric` dumps are copied/symlinked as-is; `yolo_names` dumps
   are translated to numeric ids using the manifest's class list.
2. **train** — YOLOv8 fine-tuning with GPU auto-scaling, OOM recovery, TensorBoard, optional LLRD/SWA.
3. **evaluate** — mAP@50 / mAP@50-95 / inference-time check against the competition targets
   (mAP ≥ 85%, inference ≤ 100ms).

## Build & Run

```bash
# Local (requires Python 3.10 venv with torch + ultralytics):
pip install -r requirements.txt

# --- Fast path: sync the newest dump + train in one go -----------------
# `sync_latest.py` is a no-op when the local dataset is already in sync
# with the newest manifest-bearing dump, so it's safe to run on every
# training attempt. Use --force to rebuild unconditionally.
python scripts/data/sync_latest.py                             # prepare only
./start.sh sync                                                # same, in Docker
./start.sh train-latest -- --fast --epochs 1                   # sync + train, Docker
docker compose run --rm app train-latest --fast --epochs 1     # same, explicit form

# --- Manual step-by-step (equivalent) ----------------------------------

# 1. Prepare dataset. --latest picks the newest manifest-bearing dump under
#    $PYBULLET_HSR_ROOT/annotation_data/ automatically. If a prior prepare
#    for the same dump + settings exists, this is a no-op (pass --force to
#    rebuild).
python scripts/data/prepare_dataset.py \
    --source /home/roboworks/repos/pybullet_hsr/annotation_data --latest --symlink

# or point at a specific dump:
python scripts/data/prepare_dataset.py \
    --source /home/roboworks/repos/pybullet_hsr/annotation_data/<name>_<ts> --symlink

# 2. Train (dataset defaults to datasets/<dataset_name>/data.yaml).
python scripts/training/quick_finetune.py --dataset datasets/<dataset_name>/data.yaml --fast --epochs 1

# 3. Evaluate.
python scripts/evaluation/evaluate_model.py \
    --model models/finetuned/<run>/weights/best.pt \
    --dataset datasets/<dataset_name>/data.yaml

# Streamlit UI (auto-discovers dumps via manifest + exposes a one-click
# "Sync latest dump" / "Prepare & Train on latest dump" button on
# Dashboard and Training):
./run_app.sh

# Docker (recommended — matches CI):
./start.sh                      # build + run Streamlit at http://localhost:8501
./start.sh --tensorboard        # also bring TensorBoard up on 6006
docker compose run --rm app prepare-dataset --source /pybullet_hsr/annotation_data --latest --symlink
docker compose run --rm app train --dataset datasets/<dataset_name>/data.yaml --fast
```

### Dataset freshness tracking

`prepare_dataset.py` writes a `.prepare_meta.json` sidecar into the
destination (`datasets/<name>/.prepare_meta.json`) recording the source
dump path, manifest `created_at`, val ratio, seed, and symlink mode.
Subsequent runs with the same source detect this and exit early — that's
what makes `sync` / `train-latest` idempotent. Pass `--force` to rebuild
unconditionally (e.g. after tweaking val ratio or seed).

The Streamlit Dashboard compares the sidecar against the newest dump and
shows ✅ in sync / ⚠️ stale / ⚠️ not prepared yet. The Training page
exposes the same status at the top with a "Prepare & Train on latest
dump" primary button.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/data/manifest.py` | Manifest reader + dump discovery (schema v1.0) |
| `scripts/data/prepare_dataset.py` | Manifest-driven YOLO dataset preparer + `.prepare_meta.json` sidecar |
| `scripts/data/sync_latest.py` | Thin wrapper that preps the newest dump (no-op if fresh) |
| `scripts/training/quick_finetune.py` | YOLOv8 fine-tuning entrypoint |
| `scripts/evaluation/evaluate_model.py` | mAP / inference-time evaluator |
| `scripts/evaluation/xtion_live_infer.py` | PyQt6 live viewer: YOLO over ROS2 image topic (Xtion) |
| `app/main.py` + `app/pages/` | Streamlit UI (Dashboard, Training, Evaluation, Settings) |
| `docker/Dockerfile`, `docker-compose.yml` | Container build + runtime wiring (ROS2 + CUDA + YOLO in one image) |
| `docker/99-xtion.rules` | Xtion udev rules (installed on host for USB access) |
| `config/fastdds_profile.xml` | FastDDS profile (SHM disabled) for ROS2 discovery stability |

## Xtion live inference

Once a model is trained, `scripts/evaluation/xtion_live_infer.py` subscribes to
a ROS2 `sensor_msgs/Image` topic (typically from an Xtion PRO LIVE), runs the
`.pt` weights on each frame, and renders the annotated stream in a PyQt6
window with a topic selector, confidence slider, FPS counter, and per-detection
list.

The base `hsr-perception:latest` image now bundles ROS2 Humble + OpenNI2 +
PyQt6 directly — there is no separate overlay image.

```bash
# One-time host setup
sudo cp docker/99-xtion.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG video $USER    # logout/login required

# PC + Xtion (default) — single command starts the publisher AND viewer
./start.sh xtion-live -- \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25

# HSR / external publisher — skip the in-container openni2_camera
./start.sh xtion-live -- --no-camera \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25
```

`start.sh` runs `xhost +local:docker` before launching, and compose gives
the container `network_mode: host`, USB passthrough (`/dev/bus/usb`), and
X11 forwarding so rclpy can discover publishers on the host / same DDS
domain.

**Host path** — if ROS2 Humble + PyQt6 are already installed locally:

```bash
source /opt/ros/humble/setup.bash
python scripts/evaluation/xtion_live_infer.py \
    --model models/finetuned/<run>/weights/best.pt --conf 0.25
```

## Environment variables

- `PYBULLET_HSR_ROOT` — path to the pybullet_hsr clone. Default: `/home/roboworks/repos/pybullet_hsr`
  (inside Docker: `/pybullet_hsr`, mounted read-only).

## Manifest schema (v1.0 contract)

Consumed fields (read via `scripts/data/manifest.py`):

- `schema_version`: must be `"1.0"` (reader fails fast on mismatch).
- `dataset_name`: used as the default output directory name.
- `paths.images_subdir` / `paths.labels_subdir`: where to find the image and label files inside the dump.
- `label_format`: `"yolo_numeric"` (labels already carry class_ids) or `"yolo_names"` (labels carry
  class names, reader translates to numeric ids using `classes[]`).
- `image_extension`: e.g., `"png"`.
- `classes[]`: ordered `{id, name}` list, must be contiguous from 0.
- `stats`: displayed in the Dashboard / Training picker; not load-bearing.

If the schema bumps to v2.0 upstream, update `SUPPORTED_SCHEMA_VERSIONS` in
`scripts/data/manifest.py` and adjust consumers accordingly.

## Tech Stack

- YOLOv8 (Ultralytics ≥ 8.3.0) with LLRD + SWA + OOM recovery hooks
- PyTorch + CUDA 12.1 (Docker image)
- Streamlit UI
- TensorBoard for live monitoring

## Branch Strategy

- `main` — Stable
- `develop` — Development
- `feature/*` — Feature development (this branch lives here)
- `competition/*` — Competition-specific adjustments

Note: This branch (`feature/pybullet-hsr-dataset-integration`) has diverged significantly from `main`;
the pre-existing collection / annotation / ROS2 subsystems were removed and are intentionally not
merged back. Operate via branch switching, not merges.

## Commit Message Convention

```
feat: New feature
fix: Bug fix
docs: Documentation
refactor: Refactoring
test: Test additions/modifications
```
