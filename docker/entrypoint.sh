#!/bin/bash
# =============================================================================
# HSR Perception Pipeline - Entrypoint Script
# Supports multiple run modes for different use cases
# =============================================================================

set -e

# =============================================================================
# Environment Setup
# =============================================================================

# Source ROS2 environment
echo "Sourcing ROS2 Humble environment..."
source /opt/ros/humble/setup.bash

# Source workspace overlay if built
if [ -f /workspace/install/setup.bash ]; then
    echo "Sourcing workspace overlay..."
    source /workspace/install/setup.bash
fi

# Set FastDDS profile for SHM conflict avoidance
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-/workspace/config/fastdds_profile.xml}
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# Add scripts to PYTHONPATH
export PYTHONPATH=/workspace:/workspace/scripts:${PYTHONPATH}

# =============================================================================
# Helper Functions
# =============================================================================

# Check if Xtion camera is connected
check_xtion() {
    echo "Checking for Xtion camera..."
    if lsusb 2>/dev/null | grep -q "1d27"; then
        echo "  [OK] Xtion camera detected"
        return 0
    else
        echo "  [WARN] Xtion camera not detected"
        return 1
    fi
}

# Check if GPU is available
check_gpu() {
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
        echo "  [OK] GPU available"
    else
        echo "  [WARN] GPU not available - running in CPU mode"
    fi
}

# Check if X11 display is available
check_display() {
    echo "Checking X11 display..."
    if [ -z "$DISPLAY" ]; then
        echo "  [WARN] DISPLAY not set - GUI apps will not work"
        return 1
    fi
    if ! xdpyinfo &>/dev/null 2>&1; then
        echo "  [WARN] Cannot connect to X server - run 'xhost +local:docker' on host"
        return 1
    fi
    echo "  [OK] X11 display available (${DISPLAY})"
    return 0
}

# Print environment info
print_info() {
    echo "=============================================="
    echo "HSR Perception Pipeline"
    echo "=============================================="
    echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID}"
    echo "FastDDS Profile: ${FASTRTPS_DEFAULT_PROFILES_FILE}"
    echo "Python: $(python3 --version)"
    check_gpu
    check_display || true
    echo "=============================================="
}

# =============================================================================
# Command Handlers
# =============================================================================

case "${1:-streamlit}" in
    # -------------------------------------------------------------------------
    # Streamlit Web UI (Default)
    # -------------------------------------------------------------------------
    streamlit)
        print_info
        echo ""
        echo "Starting HSR Object Manager (Streamlit)..."
        echo "Access the UI at: http://localhost:8501"
        echo ""
        cd /workspace
        exec streamlit run app/main.py \
            --server.headless true \
            --server.port 8501 \
            --server.address 0.0.0.0
        ;;

    # -------------------------------------------------------------------------
    # ROS2 OpenNI2 Camera Node (Xtion)
    # -------------------------------------------------------------------------
    ros2-camera)
        print_info
        check_xtion
        echo ""
        echo "Starting ROS2 OpenNI2 Camera Node..."
        exec ros2 launch openni2_camera camera_only.launch.py
        ;;

    # -------------------------------------------------------------------------
    # ROS2 HSR Capture Node
    # -------------------------------------------------------------------------
    ros2-capture)
        print_info
        echo ""
        echo "Starting ROS2 HSR Capture Node..."
        exec ros2 launch hsr_perception capture.launch.py
        ;;

    # -------------------------------------------------------------------------
    # Training Mode
    # -------------------------------------------------------------------------
    train)
        shift
        print_info
        echo ""
        echo "Starting YOLOv8 Training..."
        exec python3 /workspace/scripts/training/quick_finetune.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # Annotation Mode
    # -------------------------------------------------------------------------
    annotate)
        shift
        print_info
        echo ""
        echo "Starting Auto-Annotation..."
        exec python3 /workspace/scripts/annotation/auto_annotate.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # Evaluation Mode
    # -------------------------------------------------------------------------
    evaluate)
        shift
        print_info
        echo ""
        echo "Starting Model Evaluation..."
        exec python3 /workspace/scripts/evaluation/evaluate_model.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # Visual Verification Mode
    # -------------------------------------------------------------------------
    verify)
        shift
        print_info
        echo ""
        echo "Starting Visual Verification..."
        exec python3 /workspace/scripts/evaluation/visual_verification.py "$@"
        ;;

    # -------------------------------------------------------------------------
    # TensorBoard
    # -------------------------------------------------------------------------
    tensorboard)
        shift
        print_info
        LOGDIR="${1:-/workspace/runs}"
        echo ""
        echo "Starting TensorBoard..."
        echo "Log directory: ${LOGDIR}"
        echo "Access at: http://localhost:6006"
        exec tensorboard --logdir="${LOGDIR}" --bind_all --port=6006
        ;;

    # -------------------------------------------------------------------------
    # Interactive Shell
    # -------------------------------------------------------------------------
    bash|sh)
        print_info
        echo ""
        echo "Starting interactive shell..."
        exec /bin/bash
        ;;

    # -------------------------------------------------------------------------
    # ROS2 Command Pass-through
    # -------------------------------------------------------------------------
    ros2)
        shift
        exec ros2 "$@"
        ;;

    # -------------------------------------------------------------------------
    # Python Script Pass-through
    # -------------------------------------------------------------------------
    python|python3)
        shift
        exec python3 "$@"
        ;;

    # -------------------------------------------------------------------------
    # Help
    # -------------------------------------------------------------------------
    help|--help|-h)
        echo "HSR Perception Pipeline - Docker Entrypoint"
        echo ""
        echo "Usage: docker run hsr-perception [COMMAND] [ARGS]"
        echo ""
        echo "Commands:"
        echo "  streamlit     Start Streamlit Web UI (default)"
        echo "  ros2-camera   Start ROS2 OpenNI2 camera node"
        echo "  ros2-capture  Start ROS2 HSR capture node"
        echo "  train         Run YOLOv8 training"
        echo "  annotate      Run auto-annotation"
        echo "  evaluate      Run model evaluation"
        echo "  verify        Run visual verification"
        echo "  tensorboard   Start TensorBoard server"
        echo "  bash          Start interactive shell"
        echo "  ros2 [cmd]    Run ROS2 command"
        echo "  python [cmd]  Run Python script"
        echo "  help          Show this help message"
        echo ""
        echo "Examples:"
        echo "  docker compose up                    # Start Streamlit UI"
        echo "  docker compose run hsr-perception train --dataset /workspace/datasets/data.yaml"
        echo "  docker compose exec hsr-perception bash"
        ;;

    # -------------------------------------------------------------------------
    # Custom Command
    # -------------------------------------------------------------------------
    *)
        exec "$@"
        ;;
esac
