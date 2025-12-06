#!/bin/bash
# Launch HSR Object Manager

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit not found. Installing..."
    pip install streamlit Pillow
fi

# Run the app
echo "Starting HSR Object Manager..."
echo "Open http://localhost:8501 in your browser"
echo ""

cd "$SCRIPT_DIR"
streamlit run app/main.py --server.headless true
