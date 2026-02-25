#!/bin/bash
# Health watchdog: restart tradingserver if /health fails (fixes intermittent 502)
# Run via cron: */2 * * * * /home/ubuntu/local_3comas_clone_v2/health_watchdog.sh
# Rate limit: at most 1 restart per 5 min

DIR="${1:-/home/ubuntu/local_3comas_clone_v2}"
LOCK="${DIR}/.watchdog_restart.lock"
cd "$DIR" 2>/dev/null || exit 0

if curl -sf -m 8 http://127.0.0.1:8000/health >/dev/null 2>&1 || \
   curl -sf -m 8 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
  rm -f "$LOCK" 2>/dev/null || true
  exit 0
fi

now=$(date +%s)
if [ -f "$LOCK" ]; then
  last=$(cat "$LOCK" 2>/dev/null || echo 0)
  [ $((now - last)) -lt 300 ] && exit 0
fi
echo "$now" > "$LOCK"

logger -t ai-bot-watchdog "Health failed, restarting ai-bot"
sudo systemctl restart ai-bot 2>/dev/null || true
sudo systemctl restart tradingserver 2>/dev/null || true
sudo systemctl restart bot-worker 2>/dev/null || true
