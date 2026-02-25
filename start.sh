#!/bin/bash
# Start AI Bot - Full website + all bots locally (Linux/Mac)
# Usage: ./start.sh

set -e
cd "$(dirname "$0")"

echo "=== AI Bot - Local Mode ==="
echo ""

# Create venv if missing
if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Installing dependencies (first run: may take 5-10 minutes)..."
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/pip install -r requirements.txt
fi

# Ensure database exists with correct schema
if [ ! -f "botdb.sqlite3" ]; then
    echo "Initializing database..."
    .venv/bin/python init_db.py
fi

echo ""
echo "Starting at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop"
echo ""

exec .venv/bin/uvicorn one_server:app --reload --port 8000 --host 0.0.0.0
