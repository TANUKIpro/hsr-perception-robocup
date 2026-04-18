#!/bin/bash
# HSR Perception Pipeline - Entrypoint Script
# Training + evaluation container. No ROS2, no capture, no SAM2.

set -e

export PYTHONPATH=/workspace:/workspace/scripts:${PYTHONPATH}
export PYBULLET_HSR_ROOT=${PYBULLET_HSR_ROOT:-/pybullet_hsr}

check_gpu() {
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        echo "  [OK] GPU available"
    else
        echo "  [WARN] GPU not available - training will be slow on CPU"
    fi
}

sync_pretrained_models() {
    local cache_dir="/opt/model-cache"
    local target_dir="/workspace/models/pretrained"
    [ -d "$cache_dir" ] || return 0
    mkdir -p "$target_dir"
    for model in "$cache_dir"/*.pt; do
        [ -f "$model" ] || continue
        local name
        name=$(basename "$model")
        if [ ! -f "$target_dir/$name" ]; then
            echo "  Copying cached model: $name"
            cp "$model" "$target_dir/$name"
        fi
    done
}

print_info() {
    echo "=============================================="
    echo "HSR Perception - Training + Evaluation"
    echo "=============================================="
    echo "PYBULLET_HSR_ROOT: ${PYBULLET_HSR_ROOT}"
    echo "Python: $(python3 --version)"
    check_gpu
    echo "=============================================="
}

sync_pretrained_models

case "${1:-streamlit}" in
    streamlit)
        print_info
        echo "Starting Streamlit UI at http://localhost:8501 ..."
        cd /workspace
        exec streamlit run app/main.py \
            --server.headless true \
            --server.port 8501 \
            --server.address 0.0.0.0
        ;;

    prepare-dataset)
        shift
        print_info
        exec python3 /workspace/scripts/data/prepare_dataset.py "$@"
        ;;

    sync)
        shift
        print_info
        exec python3 /workspace/scripts/data/sync_latest.py "$@"
        ;;

    train)
        shift
        print_info
        exec python3 -m scripts.training.quick_finetune "$@"
        ;;

    train-latest)
        shift
        print_info
        echo "[train-latest] syncing newest dump ..."
        python3 /workspace/scripts/data/sync_latest.py \
            --annotation-root "${PYBULLET_HSR_ROOT}/annotation_data"
        data_yaml=$(python3 - "$PYBULLET_HSR_ROOT" <<'PY'
import json, os, sys
from pathlib import Path
sys.path.insert(0, "/workspace/scripts/data")
from manifest import discover_dumps  # type: ignore
root = Path(sys.argv[1]) / "annotation_data"
dumps = discover_dumps(root)
if not dumps:
    sys.exit("no manifest-bearing dumps found")
name = dumps[0]["manifest"].get("dataset_name") or dumps[0]["path"].name
print(f"/workspace/datasets/{name}/data.yaml")
PY
)
        echo "[train-latest] training with ${data_yaml}"
        exec python3 -m scripts.training.quick_finetune --dataset "${data_yaml}" "$@"
        ;;

    evaluate)
        shift
        print_info
        exec python3 /workspace/scripts/evaluation/evaluate_model.py "$@"
        ;;

    verify)
        shift
        print_info
        exec python3 /workspace/scripts/evaluation/visual_verification.py "$@"
        ;;

    test)
        shift
        print_info
        cd /workspace
        exec python3 -m pytest "$@"
        ;;

    tensorboard)
        shift
        print_info
        LOGDIR="${1:-/workspace/runs}"
        echo "Starting TensorBoard at http://localhost:6006 ..."
        exec tensorboard --logdir="${LOGDIR}" --bind_all --port=6006
        ;;

    bash|sh)
        print_info
        exec /bin/bash
        ;;

    python|python3)
        shift
        exec python3 "$@"
        ;;

    help|--help|-h)
        cat <<EOF
HSR Perception Pipeline - Docker Entrypoint

Usage: docker run hsr-perception [COMMAND] [ARGS]

Commands:
  streamlit         Start Streamlit UI (default)
  prepare-dataset   Turn a manifest-bearing pybullet_hsr dump into a YOLO dataset.
                    Reads <dump>/manifest.json (schema v1.0) for paths + classes.
                    Pass --latest with an annotation_data/ root to auto-pick the
                    newest dump. No --classes-yaml flag — the manifest carries it.
  sync              Prepare the newest manifest-bearing dump under
                    \$PYBULLET_HSR_ROOT/annotation_data/. No-op if the local
                    dataset is already in sync (add --force to rebuild).
  train             Run YOLOv8 training
  train-latest      Sync the newest dump, then run training against it. Extra
                    args (e.g. --fast --epochs 1) are forwarded to quick_finetune.
  evaluate          Run model evaluation
  verify            Run visual verification
  test              Run pytest tests
  tensorboard       Start TensorBoard server
  bash              Start interactive shell
  python [cmd]      Run Python script
  help              Show this help message

Examples:
  docker compose up
  docker compose run --rm app sync
  docker compose run --rm app train-latest --fast --epochs 1
  docker compose run --rm app prepare-dataset \\
      --source /pybullet_hsr/annotation_data --latest --symlink
  docker compose run --rm app train --dataset datasets/<dataset_name>/data.yaml --fast --epochs 1
EOF
        ;;

    *)
        exec "$@"
        ;;
esac
