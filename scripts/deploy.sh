#!/bin/bash
# Atomic deploy: git pull, pip, preflight, restart. Never requires reboot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Deploy (atomic) ==="
echo "Project: $PROJECT_ROOT"

# 1. Git pull (if deployed from git)
if [ -d ".git" ]; then
  echo "git pull..."
  git pull || true
fi

# 2. Ensure venv
if [ ! -d "venv" ]; then
  echo "Creating venv..."
  python3 -m venv venv
fi

# 3. Activate and upgrade pip
source venv/bin/activate
pip install --upgrade pip

# 4. Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# 5. Preflight
echo "Running preflight..."
python scripts/preflight.py

# 6. Restart bot-worker
echo "Restarting bot-worker..."
sudo systemctl restart bot-worker

# 7. Status
echo ""
echo "=== Service status ==="
sudo systemctl --no-pager -l status bot-worker || true

echo ""
echo "=== Last 200 log lines ==="
sudo journalctl -u bot-worker -n 200 --no-pager || true

echo ""
echo "=== Ports 8000 / 9001 ==="
sudo ss -ltnp | grep -E ":9001|:8000" || true

echo ""
echo "=== Memory & disk ==="
free -m
df -h

echo ""
echo "=== Deploy complete ==="
