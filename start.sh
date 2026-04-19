#!/bin/bash
# HSR Perception Docker Startup Script
#
# Usage:
#   ./start.sh                    # Build (if needed) and run Streamlit
#   ./start.sh --build            # Force rebuild
#   ./start.sh --tensorboard      # Also start TensorBoard
#   ./start.sh -d                 # Detach (background)
#   ./start.sh sync               # Prepare the newest pybullet_hsr dump (no-op if fresh)
#   ./start.sh train-latest -- --fast --epochs 1
#                                 # Sync newest dump + start training in one go
#   ./start.sh xtion-live -- --model <path>
#                                 # Live YOLO inference over a ROS2 image topic

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="hsr-perception:latest"

FLAG_BUILD=false
FLAG_TENSORBOARD=false
FLAG_DETACH=false
SUBCOMMAND=""
SUBCOMMAND_ARGS=()

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

show_help() {
    cat <<EOF
HSR Perception Docker Startup Script

Usage:
  ./start.sh [options]                  # Launch the Streamlit UI
  ./start.sh sync [-- args...]          # Prepare the newest pybullet_hsr dump
  ./start.sh train-latest [-- args...]  # Sync newest dump + run training
  ./start.sh xtion-live [-- args...]    # Live YOLO inference over ROS2 image topic
  ./start.sh ros2-camera                # Launch openni2_camera (Xtion publisher)

Options:
  --build         Force rebuild of the Docker image
  --tensorboard   Also start the TensorBoard service (port 6006)
  --detach, -d    Run in the background
  --help, -h      Show this help message

Subcommands (pass extra args after --; they are forwarded to the container):
  sync            docker compose run --rm app sync [args...]
                  No-op if the local dataset is already in sync with the
                  newest manifest-bearing dump; pass --force to rebuild.
  train-latest    docker compose run --rm app train-latest [args...]
                  e.g. ./start.sh train-latest -- --fast --epochs 1
  xtion-live      docker compose run --rm app xtion-live [args...]
                  e.g. ./start.sh xtion-live -- \\
                        --model models/finetuned/<run>/weights/best.pt
                  Needs a ROS2 publisher (e.g. openni2_camera) on the host
                  or inside a sibling container, plus X11 forwarding.
  ros2-camera     docker compose run --rm app ros2-camera
                  Launches openni2_camera inside the container for the
                  connected Xtion PRO LIVE.

The script mounts PYBULLET_HSR_ROOT (default: /home/roboworks/repos/pybullet_hsr)
into the container read-only at /pybullet_hsr so the app can read the source
dataset and class mapping YAML.
EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)        FLAG_BUILD=true; shift ;;
            --tensorboard)  FLAG_TENSORBOARD=true; shift ;;
            --detach|-d)    FLAG_DETACH=true; shift ;;
            --help|-h)      show_help; exit 0 ;;
            sync|train-latest|xtion-live|ros2-camera)
                SUBCOMMAND="$1"; shift
                if [[ "${1:-}" == "--" ]]; then shift; fi
                SUBCOMMAND_ARGS=("$@")
                return
                ;;
            *)              warn "Unknown option: $1"; shift ;;
        esac
    done
}

check_dependencies() {
    info "Checking dependencies..."
    command -v docker &>/dev/null || error "Docker is not installed."
    success "Docker: installed"
    docker compose version &>/dev/null || error "Docker Compose is not available."
    success "Docker Compose: available"
    if docker info 2>/dev/null | grep -q "nvidia"; then
        success "NVIDIA Container Toolkit: available"
    else
        warn "NVIDIA Container Toolkit not detected. GPU features may be limited."
    fi
}

build_image() {
    local need_build=false
    if [ "$FLAG_BUILD" = true ]; then
        info "Force rebuild specified"; need_build=true
    elif ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        info "Docker image not found"; need_build=true
    fi
    if [ "$need_build" = true ]; then
        info "Building Docker image (ROS2 + CUDA + YOLO; this may take 5-15 min)..."
        cd "$PROJECT_ROOT" && docker compose build
        success "Docker image build completed"
    else
        success "Docker image: already built"
    fi
}

start_services() {
    local profiles=""
    local detach_flag=""
    [ "$FLAG_TENSORBOARD" = true ] && profiles="$profiles --profile tensorboard"
    [ "$FLAG_DETACH" = true ] && detach_flag="-d"

    echo
    echo "========================================"
    echo "  Starting services"
    echo "========================================"
    echo "  Web UI:      http://localhost:8501"
    [ "$FLAG_TENSORBOARD" = true ] && echo "  TensorBoard: http://localhost:6006"
    echo

    cd "$PROJECT_ROOT"
    docker compose $profiles up $detach_flag
}

cleanup() {
    echo; info "Shutting down..."
    cd "$PROJECT_ROOT" && docker compose down
    info "Shutdown complete"
    exit 0
}

trap cleanup SIGINT SIGTERM

allow_x11() {
    if command -v xhost &>/dev/null; then
        xhost +local:docker >/dev/null 2>&1 || true
    else
        warn "xhost not found - the GUI may fail to open. Install x11-xserver-utils on the host."
    fi
}

run_subcommand() {
    info "Running '${SUBCOMMAND}' inside the container..."
    cd "$PROJECT_ROOT"
    if [ "$SUBCOMMAND" = "xtion-live" ] || [ "$SUBCOMMAND" = "ros2-camera" ]; then
        allow_x11
    fi
    exec docker compose run --rm app "$SUBCOMMAND" "${SUBCOMMAND_ARGS[@]}"
}

main() {
    parse_args "$@"
    check_dependencies
    build_image
    if [ -n "$SUBCOMMAND" ]; then
        run_subcommand
    fi
    start_services
}

main "$@"
