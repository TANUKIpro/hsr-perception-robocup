#!/bin/bash
# =============================================================================
# HSR Perception Docker Startup Script
# =============================================================================
#
# First-time startup:
#   - Automatically builds Docker image
#   - Automatically installs Xtion udev rules (sudo required)
#
# Usage:
#   ./start.sh              # Normal startup
#   ./start.sh --build      # Force rebuild
#   ./start.sh --tensorboard # Start with TensorBoard
#   ./start.sh -d           # Start in background
#
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="hsr-perception:latest"
XTION_VENDOR_ID="1d27"  # PrimeSense/ASUS Xtion
UDEV_RULES_FILE="/etc/udev/rules.d/99-xtion.rules"

# Option flags
FLAG_BUILD=false
FLAG_TENSORBOARD=false
FLAG_DETACH=false
FLAG_HELP=false

# -----------------------------------------------------------------------------
# Colored message functions
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# -----------------------------------------------------------------------------
# Show help
# -----------------------------------------------------------------------------
show_help() {
    cat << EOF
========================================
  HSR Perception Docker Startup Script
========================================

Usage: ./start.sh [options]

Options:
  --build         Force rebuild Docker image
  --tensorboard   Also start TensorBoard (port 6006)
  --detach, -d    Start in background
  --help, -h      Show this help

Examples:
  ./start.sh                      # Normal startup
  ./start.sh --build              # Rebuild image and start
  ./start.sh --tensorboard        # Start with TensorBoard
  ./start.sh -d --tensorboard     # Start in background with TensorBoard

How to stop:
  Foreground mode: Ctrl+C
  Background mode: docker compose down

EOF
}

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --build)
                FLAG_BUILD=true
                shift
                ;;
            --tensorboard)
                FLAG_TENSORBOARD=true
                shift
                ;;
            --detach|-d)
                FLAG_DETACH=true
                shift
                ;;
            --help|-h)
                FLAG_HELP=true
                shift
                ;;
            *)
                warn "Unknown option: $1"
                shift
                ;;
        esac
    done
}

# -----------------------------------------------------------------------------
# Check dependencies
# -----------------------------------------------------------------------------
check_dependencies() {
    info "Checking dependencies..."

    # Docker
    if ! command -v docker &>/dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    success "Docker: installed"

    # Docker Compose
    if ! docker compose version &>/dev/null; then
        error "Docker Compose is not available. Please install Docker Desktop or docker-compose-plugin."
    fi
    success "Docker Compose: available"

    # NVIDIA Container Toolkit (warning only)
    if docker info 2>/dev/null | grep -q "nvidia"; then
        success "NVIDIA Container Toolkit: available"
    else
        warn "NVIDIA Container Toolkit not detected. GPU features may be limited."
    fi
}

# -----------------------------------------------------------------------------
# Setup udev rules (first-time only)
# -----------------------------------------------------------------------------
setup_xtion_udev() {
    if [ -f "$UDEV_RULES_FILE" ]; then
        return 0
    fi

    local source_rules="$PROJECT_ROOT/docker/99-xtion.rules"
    if [ ! -f "$source_rules" ]; then
        warn "udev rules file not found: $source_rules"
        return 0
    fi

    echo ""
    info "First-time setup: Installing Xtion udev rules"
    echo "   (sudo password required)"
    echo ""

    if sudo cp "$source_rules" "$UDEV_RULES_FILE" && \
       sudo udevadm control --reload-rules && \
       sudo udevadm trigger; then
        success "udev rules installed"

        # Check if user is in video group
        if ! groups | grep -q video; then
            info "Attempting to add user to video group..."
            if sudo usermod -aG video "$USER"; then
                warn "Added to video group. Please log out and log back in for changes to take effect."
            fi
        fi
    else
        warn "Failed to install udev rules. Please install manually:"
        echo "   sudo cp $source_rules $UDEV_RULES_FILE"
        echo "   sudo udevadm control --reload-rules"
        echo "   sudo udevadm trigger"
    fi
}

# -----------------------------------------------------------------------------
# Build Docker image
# -----------------------------------------------------------------------------
build_image() {
    local need_build=false

    if [ "$FLAG_BUILD" = true ]; then
        info "Force rebuild specified"
        need_build=true
    elif ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        info "Docker image not found"
        need_build=true
    fi

    if [ "$need_build" = true ]; then
        echo ""
        info "Building Docker image (this may take 10-20 minutes)..."
        echo ""

        cd "$PROJECT_ROOT"
        if docker compose build; then
            success "Docker image build completed"
        else
            error "Docker image build failed"
        fi
    else
        success "Docker image: already built"
    fi
}

# -----------------------------------------------------------------------------
# X11 setup
# -----------------------------------------------------------------------------
setup_x11() {
    info "Setting up X11 access..."

    # Check if xhost is available
    if command -v xhost &>/dev/null; then
        if xhost +local:docker &>/dev/null; then
            success "X11: Docker access granted"
        else
            warn "Failed to set X11 access (possibly headless environment)"
        fi
    else
        warn "xhost not found. GUI features may be limited."
    fi
}

# -----------------------------------------------------------------------------
# Check Xtion connection
# -----------------------------------------------------------------------------
check_xtion() {
    if lsusb 2>/dev/null | grep -q "$XTION_VENDOR_ID"; then
        return 0  # Connected
    else
        return 1  # Not connected
    fi
}

# -----------------------------------------------------------------------------
# Start services
# -----------------------------------------------------------------------------
start_services() {
    local compose_args=""
    local profiles=""
    local detach_flag=""

    # Profile settings
    if [ "$FLAG_TENSORBOARD" = true ]; then
        profiles="$profiles --profile training"
    fi

    # Check Xtion connection
    local xtion_connected=false
    if check_xtion; then
        xtion_connected=true
        success "Xtion camera: detected"
    else
        echo ""
        warn "Xtion camera not detected"
        echo ""
        echo "   To use camera features:"
        echo "   1. Connect Xtion camera to this machine via USB"
        echo "   2. Run the following in another terminal:"
        echo ""
        echo "      cd $PROJECT_ROOT"
        echo "      docker compose run --rm hsr-perception ros2-camera"
        echo ""
    fi

    # Detach flag
    if [ "$FLAG_DETACH" = true ]; then
        detach_flag="-d"
    fi

    # Startup message
    echo ""
    echo "========================================"
    echo "  Starting services"
    echo "========================================"
    echo ""
    echo "  Web app:      http://localhost:8501"
    if [ "$FLAG_TENSORBOARD" = true ]; then
        echo "  TensorBoard:  http://localhost:6006"
    fi
    if [ "$xtion_connected" = true ]; then
        echo "  ROS2 camera:  starting"
    fi
    echo ""

    if [ "$FLAG_DETACH" != true ]; then
        echo "  Press Ctrl+C to stop"
        echo ""
    fi

    # Start Docker Compose
    cd "$PROJECT_ROOT"

    # Start camera in background if Xtion is connected
    if [ "$xtion_connected" = true ]; then
        info "Starting ROS2 camera node in background..."
        docker compose run -d --rm hsr-perception ros2-camera
    fi

    # Start web app
    docker compose $profiles up $detach_flag
}

# -----------------------------------------------------------------------------
# Cleanup on exit
# -----------------------------------------------------------------------------
cleanup() {
    echo ""
    info "Shutting down..."

    # Stop all containers
    cd "$PROJECT_ROOT"
    docker compose down

    # Revoke X11 access
    if command -v xhost &>/dev/null; then
        xhost -local:docker &>/dev/null || true
    fi

    info "Shutdown complete"
    exit 0
}

# Handle Ctrl+C
trap cleanup SIGINT SIGTERM

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
main() {
    parse_args "$@"

    # Show help
    if [ "$FLAG_HELP" = true ]; then
        show_help
        exit 0
    fi

    # Display banner
    echo ""
    echo "========================================"
    echo "  HSR Perception Docker Startup Script"
    echo "========================================"
    echo ""

    # Change to project directory
    cd "$PROJECT_ROOT"

    # Run setup steps
    check_dependencies
    echo ""
    setup_xtion_udev
    echo ""
    build_image
    echo ""
    setup_x11
    echo ""
    start_services
}

main "$@"
