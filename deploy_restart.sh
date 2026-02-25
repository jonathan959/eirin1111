#!/bin/bash
# Deploy restart: force-clear port 8000, reset systemd, restart service. No reboot needed.
# Run this when deploy leaves server unresponsive.

set -e
REMOTE_DIR="${1:-/home/ubuntu/local_3comas_clone_v2}"
cd "$REMOTE_DIR"

echo "=== Deploy Restart (no reboot) ==="

# 1. Reset systemd failed state (so restart can proceed even after StartLimitBurst)
sudo systemctl reset-failed ai-bot 2>/dev/null || true
sudo systemctl reset-failed tradingserver 2>/dev/null || true

# 2. Stop services gracefully
echo "Stopping ai-bot / tradingserver..."
sudo systemctl stop ai-bot 2>/dev/null || true
sudo systemctl stop tradingserver 2>/dev/null || true
sleep 5

# 3. Force-kill anything on port 8000 (orphaned uvicorn, stuck processes)
if command -v fuser &>/dev/null; then
  echo "Force-clearing port 8000..."
  sudo fuser -k 8000/tcp 2>/dev/null || true
  sleep 2
elif command -v lsof &>/dev/null; then
  for pid in $(sudo lsof -t -i:8000 2>/dev/null); do
    echo "Killing process on 8000: $pid"
    sudo kill -9 "$pid" 2>/dev/null || true
  done
  sleep 2
fi

# 4. Ensure port is free
for i in $(seq 1 10); do
  if ! (ss -tlnp 2>/dev/null | grep -q ":8000 ") && ! (netstat -tlnp 2>/dev/null | grep -q ":8000 "); then
    echo "Port 8000 is free"
    break
  fi
  sleep 2
done

# 5. Start ai-bot (or tradingserver if you prefer that as default)
echo "Starting ai-bot..."
sudo systemctl start ai-bot || true

# 6. Wait for health (up to 90s). If not up after 20s, try tradingserver as fallback.
echo "Waiting for app..."
for i in $(seq 1 45); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || curl -sf -m 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "App ready after ${i}*2s"
    exit 0
  fi
  if [ "$i" -eq 10 ]; then
    echo "App not up after 20s; starting tradingserver as fallback (one_server:app)."
    sudo systemctl stop ai-bot 2>/dev/null || true
    sleep 2
    sudo systemctl start tradingserver 2>/dev/null || true
  fi
  sleep 2
done

echo "WARNING: App may still be starting. Check: sudo journalctl -u ai-bot -n 30"
exit 0


