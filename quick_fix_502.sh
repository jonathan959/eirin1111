#!/bin/bash
# Quick 502/504 fix: force-clear port, restart service, no reboot needed.

echo "=== Quick 502/504 Fix (no reboot) ==="
# Fix temp dirs (often broken after reboot - causes pip/uvicorn to fail)
FIX_TEMP="$(dirname "$0")/fix_temp_dirs.sh"
if [ -x "$FIX_TEMP" ]; then
  sudo bash "$FIX_TEMP" 2>/dev/null || true
else
  sudo chmod 1777 /tmp 2>/dev/null || true
  mkdir -p /home/ubuntu/local_3comas_clone_v2/tmp 2>/dev/null; chmod 700 /home/ubuntu/local_3comas_clone_v2/tmp 2>/dev/null || true
fi
# Ensure ai-bot binds correctly (127.0.0.1 behind nginx is correct)
sudo systemctl daemon-reload 2>/dev/null || true
sudo systemctl reset-failed ai-bot 2>/dev/null || true
sudo systemctl reset-failed tradingserver 2>/dev/null || true

sudo systemctl stop ai-bot 2>/dev/null || true
sudo systemctl stop tradingserver 2>/dev/null || true
sleep 5

# Force-kill anything on port 8000 (prevents "Address already in use" after deploy)
if command -v fuser &>/dev/null; then
  sudo fuser -k 8000/tcp 2>/dev/null || true
elif command -v lsof &>/dev/null; then
  for pid in $(sudo lsof -t -i:8000 2>/dev/null); do sudo kill -9 $pid 2>/dev/null; done
fi
sleep 2

echo "Starting ai-bot..."
sudo systemctl start ai-bot

echo "Waiting for app (up to 60s)..."
for i in $(seq 1 30); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1 || curl -sf -m 5 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    echo "App ready after ${i}*2s"
    break
  fi
  sleep 2
done

echo "Reloading nginx..."
sudo systemctl reload nginx 2>/dev/null || true

echo ""
echo "=== Health Check ==="
curl -sf -m 10 http://127.0.0.1:8000/health 2>/dev/null | python3 -m json.tool 2>/dev/null || curl -sf -m 10 http://127.0.0.1:8000/api/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Health check failed - check: sudo journalctl -u ai-bot -n 50"
echo ""
echo "=== Done ==="
