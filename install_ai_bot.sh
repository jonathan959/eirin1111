#!/bin/bash
# One-shot installer: ai-bot service + nginx. Run on EC2 after deploy.
# Usage: cd /home/ubuntu/local_3comas_clone_v2 && bash install_ai_bot.sh

set -e
DIR="/home/ubuntu/local_3comas_clone_v2"
cd "$DIR"

echo "=== Installing AI Bot service + nginx ==="

# 1. Stop old service to free port 8000
sudo systemctl stop tradingserver 2>/dev/null || true
sudo systemctl stop ai-bot 2>/dev/null || true
sleep 2
if command -v fuser &>/dev/null; then
  sudo fuser -k 8000/tcp 2>/dev/null || true
  sleep 2
fi

# 2. Ensure venv exists
if [ ! -f "$DIR/venv/bin/python" ]; then
  echo "Creating venv..."
  python3 -m venv "$DIR/venv"
  "$DIR/venv/bin/pip" install -q -r requirements.txt
fi

# 3. Install systemd service
sudo cp "$DIR/ai-bot.service" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-bot
sudo systemctl start ai-bot

# 4. Wait for app
echo "Waiting for app (up to 60s)..."
for i in $(seq 1 30); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "App ready."
    break
  fi
  sleep 2
done

# 5. Install nginx config
if [ -f "$DIR/nginx-ai-bot.conf" ]; then
  sudo cp "$DIR/nginx-ai-bot.conf" /etc/nginx/sites-available/ai-bot
  sudo ln -sf /etc/nginx/sites-available/ai-bot /etc/nginx/sites-enabled/ai-bot
  sudo rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
  sudo rm -f /etc/nginx/sites-enabled/tradingserver 2>/dev/null || true
  sudo nginx -t && sudo systemctl reload nginx
fi

# 6. UFW (if enabled)
sudo ufw allow 80/tcp 2>/dev/null || true
sudo ufw allow 22/tcp 2>/dev/null || true
sudo ufw reload 2>/dev/null || true

echo ""
echo "=== Status ==="
sudo systemctl status ai-bot --no-pager | head -12
echo ""
curl -sS http://127.0.0.1:8000/health | python3 -m json.tool 2>/dev/null || echo "Health check failed"
echo ""
echo "=== Done. Test: curl http://127.0.0.1/health ==="
