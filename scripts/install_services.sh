#!/bin/bash
# Install and enable systemd services: tradingserver (UI) + bot-worker.
# Run from project root or set DEPLOY_PATH.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_PATH="${DEPLOY_PATH:-/home/ubuntu/local_3comas_clone_v2}"

echo "=== Install Services (DEPLOY_PATH=$DEPLOY_PATH) ==="
cd "$PROJECT_ROOT"
chmod +x scripts/keys_check.sh scripts/health_watchdog.sh 2>/dev/null || true

# Substitute path in service files
for name in tradingserver; do
  SRC="$PROJECT_ROOT/${name}.service"
  [ -f "$SRC" ] || SRC="$PROJECT_ROOT/ops/${name}.service"
  if [ -f "$SRC" ]; then
    echo "Installing $name.service..."
    sed "s|/home/ubuntu/local_3comas_clone_v2|$DEPLOY_PATH|g" "$SRC" | sudo tee "/etc/systemd/system/${name}.service" >/dev/null
  fi
done

for name in bot-worker bot-health; do
  SRC="$PROJECT_ROOT/ops/${name}.service"
  if [ -f "$SRC" ]; then
    echo "Installing $name.service..."
    sed "s|/home/ubuntu/local_3comas_clone_v2|$DEPLOY_PATH|g" "$SRC" | sudo tee "/etc/systemd/system/${name}.service" >/dev/null
  fi
done

[ -f "$PROJECT_ROOT/ops/bot-health.timer" ] && sudo cp "$PROJECT_ROOT/ops/bot-health.timer" /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable tradingserver
sudo systemctl enable bot-worker 2>/dev/null || true
sudo systemctl enable bot-health.timer 2>/dev/null || true
sudo systemctl start bot-health.timer 2>/dev/null || true

sudo systemctl restart tradingserver
sudo systemctl restart bot-worker 2>/dev/null || true

echo ""
echo "=== Service status ==="
sudo systemctl --no-pager -l status tradingserver | head -15
echo ""
sudo systemctl --no-pager -l status bot-worker 2>/dev/null | head -15 || true
echo ""
echo "=== Last 200 lines tradingserver ==="
sudo journalctl -u tradingserver -n 200 --no-pager
echo ""
echo "=== Last 50 lines bot-worker ==="
sudo journalctl -u bot-worker -n 50 --no-pager 2>/dev/null || true
