#!/bin/bash
# Launch HSR Object Manager

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configure FastDDS to use UDP only (disable Shared Memory)
# This fixes race conditions when Streamlit starts before ROS2 camera nodes
export FASTRTPS_DEFAULT_PROFILES_FILE="$SCRIPT_DIR/config/fastdds_profile.xml"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing..."
    pip install streamlit Pillow
fi

# Run the app
echo "Starting HSR Object Manager..."
echo "FastDDS Profile: $FASTRTPS_DEFAULT_PROFILES_FILE"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "Open http://localhost:8501 in your browser"
echo ""

cd "$SCRIPT_DIR"
streamlit run app/main.py --server.headless true
