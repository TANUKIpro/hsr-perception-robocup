#!/bin/bash
# HSR Perception Pipeline - Entrypoint Script
# Training + evaluation + ROS2-based Xtion live inference.

set -e

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
if [ -f /opt/ros/humble/setup.bash ]; then
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
fi

export PYTHONPATH=/workspace:/workspace/scripts:${PYTHONPATH}
export PYBULLET_HSR_ROOT=${PYBULLET_HSR_ROOT:-/pybullet_hsr}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-/workspace/config/fastdds_profile.xml}

check_gpu() {
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        echo "  [OK] GPU available"
    else
        echo "  [WARN] GPU not available - training will be slow on CPU"
    fi
}

check_xtion() {
    echo "Checking for Xtion camera..."
    if lsusb 2>/dev/null | grep -qi "1d27"; then
        echo "  [OK] Xtion camera detected (vendor 1d27)"
    else
        echo "  [WARN] Xtion camera not detected on USB"
    fi
}

check_display() {
    echo "Checking X11 display..."
    if [ -z "${DISPLAY:-}" ]; then
        echo "  [WARN] DISPLAY not set - GUI apps will not work"
    else
        echo "  [OK] DISPLAY=${DISPLAY}"
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
    echo "ROS_DISTRO: ${ROS_DISTRO:-<unset>} (ROS_DOMAIN_ID=${ROS_DOMAIN_ID})"
    echo "FastDDS profile: ${FASTRTPS_DEFAULT_PROFILES_FILE}"
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

    xtion-live)
        shift
        print_info

        # Parse our own flag (--no-camera) out of the forwarded args.
        SKIP_CAMERA=false
        FILTERED_ARGS=()
        for arg in "$@"; do
            case "$arg" in
                --no-camera) SKIP_CAMERA=true ;;
                *)           FILTERED_ARGS+=("$arg") ;;
            esac
        done

        CAMERA_PID=""
        cleanup_camera() {
            if [ -n "$CAMERA_PID" ] && kill -0 "$CAMERA_PID" 2>/dev/null; then
                echo "[xtion-live] Stopping openni2_camera (pid $CAMERA_PID)..."
                kill -INT "$CAMERA_PID" 2>/dev/null || true
                for _ in 1 2 3 4 5; do
                    kill -0 "$CAMERA_PID" 2>/dev/null || break
                    sleep 1
                done
                if kill -0 "$CAMERA_PID" 2>/dev/null; then
                    kill -TERM "$CAMERA_PID" 2>/dev/null || true
                fi
                # Mop up any surviving children of ros2 launch.
                pkill -TERM -f 'openni2_camera|component_container|static_transform_publisher' 2>/dev/null || true
            fi
        }
        trap cleanup_camera EXIT INT TERM

        if [ "$SKIP_CAMERA" = false ]; then
            check_xtion || true
            echo "[xtion-live] Starting openni2_camera publisher in background..."
            : > /tmp/openni2.log
            ros2 launch openni2_camera camera_only.launch.py \
                >/tmp/openni2.log 2>&1 &
            CAMERA_PID=$!

            echo "[xtion-live] Waiting (up to 20s) for publisher to advertise..."
            ready=false
            for _ in $(seq 1 40); do
                if grep -q "Loaded node '/camera/driver'" /tmp/openni2.log 2>/dev/null; then
                    ready=true
                    break
                fi
                if ! kill -0 "$CAMERA_PID" 2>/dev/null; then
                    echo "[xtion-live] openni2_camera exited early; see /tmp/openni2.log:"
                    tail -40 /tmp/openni2.log
                    exit 1
                fi
                sleep 0.5
            done
            if [ "$ready" = true ]; then
                echo "[xtion-live] Publisher ready; /camera/rgb/image_raw should be live."
            else
                echo "[xtion-live] WARN: publisher did not report ready in 20s; continuing anyway."
            fi
        else
            echo "[xtion-live] --no-camera: skipping openni2 publisher (HSR / external publisher)."
        fi

        check_display || true
        echo "[xtion-live] Launching PyQt6 inference viewer..."
        # Don't exec — we want the trap to fire for cleanup on exit.
        python3 /workspace/scripts/evaluation/xtion_live_infer.py "${FILTERED_ARGS[@]}"
        ;;

    ros2-camera)
        shift
        print_info
        check_xtion || true
        echo "Starting ROS2 OpenNI2 camera node..."
        exec ros2 launch openni2_camera camera.launch.py "$@"
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

    ros2)
        shift
        exec ros2 "$@"
        ;;

    bash|sh)
        shell="$1"; shift
        if [ $# -gt 0 ]; then
            # e.g. `docker compose run --rm app bash -c 'script'`
            exec "/bin/$shell" "$@"
        fi
        print_info
        exec "/bin/$shell"
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
                    Pass --latest with an annotation_data/ root to auto-pick the
                    newest dump.
  sync              Prepare the newest manifest-bearing dump under
                    \$PYBULLET_HSR_ROOT/annotation_data/. No-op if fresh.
  train             Run YOLOv8 training
  train-latest      Sync the newest dump, then run training against it.
  evaluate          Run model evaluation
  verify            Run visual verification
  xtion-live        Live YOLO inference viewer over a ROS2 image topic.
                    Auto-starts openni2_camera publisher in the same
                    container. Pass --no-camera to skip (HSR / external).
  ros2-camera       Launch openni2_camera standalone (foreground).
  ros2 [args]       ros2 CLI passthrough
  test              Run pytest tests
  tensorboard       Start TensorBoard server
  bash              Start interactive shell
  python [cmd]      Run Python script
  help              Show this help message

Examples:
  docker compose up
  docker compose run --rm app sync
  docker compose run --rm app train-latest --fast --epochs 1
  docker compose run --rm app xtion-live --model models/finetuned/<run>/weights/best.pt
EOF
        ;;

    *)
        exec "$@"
        ;;
esac
