#!/bin/bash
# Launch the HSR Perception Streamlit UI locally.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYBULLET_HSR_ROOT="${PYBULLET_HSR_ROOT:-/home/roboworks/repos/pybullet_hsr}"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/scripts:$PYTHONPATH"

if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Install with: pip install -r requirements.txt"
    exit 1
fi

echo "PYBULLET_HSR_ROOT=$PYBULLET_HSR_ROOT"
echo "Starting HSR Perception — http://localhost:8501"
cd "$SCRIPT_DIR"
exec streamlit run app/main.py --server.headless true
