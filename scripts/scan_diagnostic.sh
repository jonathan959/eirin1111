#!/bin/bash
# Market Screener diagnostic - run on EC2 via: ssh ... 'bash -s' < scan_diagnostic.sh
# Captures: RAM, CPU, threads, logs, process state

echo "========== STEP 1: SYSTEM STATE =========="
echo ""

# Find the app process (ai-bot, tradingserver, or uvicorn)
APP_PID=$(pgrep -f "one_server|uvicorn.*8000" | head -1)
if [ -z "$APP_PID" ]; then
  APP_PID=$(pgrep -f "worker_api" | head -1)
fi
if [ -z "$APP_PID" ]; then
  echo "WARN: Could not find app process by name. Listing python processes:"
  ps aux | grep -E python | grep -v grep
  echo ""
  APP_PID=$(ps aux | grep -E 'uvicorn|one_server' | grep -v grep | awk '{print $2}' | head -1)
fi

if [ -n "$APP_PID" ]; then
  echo "=== App PID: $APP_PID ==="
  echo ""
  echo "--- RAM usage (MB) ---"
  ps -o pid,rss,vsz,pcpu,etime,comm -p $APP_PID 2>/dev/null || true
  RSS_KB=$(ps -o rss= -p $APP_PID 2>/dev/null | tr -d ' ')
  if [ -n "$RSS_KB" ]; then
    RSS_MB=$((RSS_KB / 1024))
    echo "RSS (MB): $RSS_MB"
  fi
  echo ""
  echo "--- Thread count ---"
  ps -T -p $APP_PID 2>/dev/null | wc -l || echo "N/A"
  echo ""
  echo "--- Process uptime ---"
  ps -o etime= -p $APP_PID 2>/dev/null || true
  echo ""
else
  echo "ERROR: No app process found"
fi

echo ""
echo "--- CPU usage (top 5 processes) ---"
ps aux --sort=-%cpu | head -6
echo ""

echo "--- Active threads (all python) ---"
ps -eLf 2>/dev/null | grep -E "python|uvicorn" | grep -v grep | wc -l || echo "N/A"
echo ""

echo "========== STEP 2: SCHEDULER / LOGS =========="
echo ""
echo "--- Last 200 lines of journal (ai-bot or tradingserver) ---"
for unit in ai-bot tradingserver one_server; do
  if systemctl is-active --quiet $unit 2>/dev/null; then
    echo "=== Unit: $unit ==="
    journalctl -u $unit -n 200 --no-pager 2>/dev/null | tail -200
    break
  fi
done
if ! systemctl is-active --quiet ai-bot 2>/dev/null && ! systemctl is-active --quiet tradingserver 2>/dev/null; then
  echo "Trying journalctl for common names..."
  journalctl -u ai-bot -n 100 --no-pager 2>/dev/null || journalctl -u tradingserver -n 100 --no-pager 2>/dev/null || true
fi
echo ""

echo "========== STEP 3: DATA / DB =========="
echo ""
echo "--- DB file size ---"
ls -la /home/ubuntu/local_3comas_clone_v2/botdb.sqlite3 2>/dev/null || ls -la ~/local_3comas_clone_v2/botdb.sqlite3 2>/dev/null || echo "DB not found in default paths"
echo ""
echo "--- Recommendations table row count (if sqlite3 available) ---"
sqlite3 /home/ubuntu/local_3comas_clone_v2/botdb.sqlite3 "SELECT COUNT(*) FROM recommendations_snapshots;" 2>/dev/null || echo "sqlite3 or DB unavailable"
echo ""
