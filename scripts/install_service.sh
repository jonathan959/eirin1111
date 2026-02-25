#!/bin/bash
# Install bot-worker and bot-health (service + timer).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_PATH="${DEPLOY_PATH:-$PROJECT_ROOT}"

echo "Deploy path: $DEPLOY_PATH"

# Ensure health_watchdog.sh is executable
chmod +x "$PROJECT_ROOT/scripts/health_watchdog.sh" 2>/dev/null || true

# Substitute deploy path in service files
for name in bot-worker bot-health; do
  SRC="$PROJECT_ROOT/ops/${name}.service"
  if [ -f "$SRC" ]; then
    echo "Installing $name.service..."
    sed "s|/home/ubuntu/local_3comas_clone_v2|$DEPLOY_PATH|g" "$SRC" | sudo tee "/etc/systemd/system/${name}.service" >/dev/null
  fi
done

# Timer (no path substitution needed)
if [ -f "$PROJECT_ROOT/ops/bot-health.timer" ]; then
  echo "Installing bot-health.timer..."
  sudo cp "$PROJECT_ROOT/ops/bot-health.timer" /etc/systemd/system/
fi

sudo systemctl daemon-reload
sudo systemctl enable bot-worker
sudo systemctl enable bot-health.timer
sudo systemctl start bot-health.timer 2>/dev/null || true
sudo systemctl restart bot-worker

echo ""
echo "=== bot-worker status ==="
sudo systemctl --no-pager -l status bot-worker
echo ""
echo "=== bot-health.timer ==="
systemctl list-timers | grep bot-health || true
