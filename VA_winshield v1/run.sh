#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=========================================================="
echo "   WII-AI · Windshield YOLO Extractor & Dataset Studio"
echo "=========================================================="

# Check for local venv, Video Analytics v4 venv, or python3
PYTHON_BIN=""
if [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN=".venv/bin/python3"
elif [ -f "../Video Analytics v4/.venv/bin/python3" ]; then
    PYTHON_BIN="../Video Analytics v4/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_BIN="python3"
else
    echo "Python 3 environment not found. Please set up a virtual environment."
    exit 1
fi

PORT="${PORT:-5056}"
echo "Using Python: $PYTHON_BIN"
echo "Starting web server on http://localhost:${PORT}"
echo "Open your browser to http://localhost:${PORT}"
echo "Press Ctrl+C to stop."

exec "$PYTHON_BIN" app.py
