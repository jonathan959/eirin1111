#!/bin/bash
# Auto-recovery watchdog: if health fails, restart bot-worker and log resources.

HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:9001/health}"
SERVICE="${SERVICE:-bot-worker}"

if curl -fsS -m 10 "$HEALTH_URL" >/dev/null 2>&1; then
  exit 0
fi

echo "[$(date -Iseconds)] health_watchdog: $HEALTH_URL FAILED - restarting $SERVICE"
echo "=== free -m ==="
free -m
echo "=== df -h ==="
df -h
echo "=== restarting $SERVICE ==="
sudo systemctl restart "$SERVICE"
exit 0
