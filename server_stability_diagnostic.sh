#!/bin/bash
# server_stability_diagnostic.sh - Run on the EC2 instance to diagnose stability issues
# Usage: ssh ubuntu@3.148.6.246 'bash -s' < server_stability_diagnostic.sh
# Or: scp to server and run: bash server_stability_diagnostic.sh

set -e
echo "=== Server Stability Diagnostic ($(date)) ==="
echo ""

echo "[1] Memory usage"
free -h
echo ""

echo "[2] Process memory (tradingserver / uvicorn)"
ps aux | grep -E "uvicorn|python.*one_server" | grep -v grep || true
echo ""

echo "[3] OOM killer history (if any)"
dmesg 2>/dev/null | grep -i "out of memory\|oom" | tail -20 || echo "Cannot read dmesg (need sudo)"
echo ""

echo "[4] Tradingserver service status"
sudo systemctl status tradingserver --no-pager 2>/dev/null || true
echo ""

echo "[5] Recent restarts (systemd)"
journalctl -u tradingserver --since "24 hours ago" --no-pager 2>/dev/null | grep -E "Starting|Stopping|Failed|status=0/0" | tail -30
echo ""

echo "[6] Recent errors in tradingserver logs"
journalctl -u tradingserver -n 100 --no-pager 2>/dev/null | grep -iE "error|exception|traceback|killed|signal" | tail -30
echo ""

echo "[7] Open file descriptors (tradingserver PID)"
PID=$(pgrep -f "uvicorn one_server" | head -1)
if [ -n "$PID" ]; then
  echo "PID=$PID"
  ls -la /proc/$PID/fd 2>/dev/null | wc -l || echo "Cannot read fd count"
else
  echo "Process not found"
fi
echo ""

echo "[8] Thread count (tradingserver)"
if [ -n "$PID" ]; then
  cat /proc/$PID/status 2>/dev/null | grep -E "Threads|VmRSS|VmSize" || true
fi
echo ""

echo "=== Diagnostic complete ==="
