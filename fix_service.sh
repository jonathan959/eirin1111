#!/bin/bash
# fix_service.sh - Emergency 502 recovery: ensure 0.0.0.0, restart tradingserver, wait, check health.

set -e
SERVICE_NAME="tradingserver"
SERVICE_DIR="/home/ubuntu/local_3comas_clone_v2"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "=== Trading Server 502 Recovery ==="

if [ ! -f "$SERVICE_FILE" ]; then
    echo "Creating service file..."
    sudo cp "${SERVICE_DIR}/tradingserver.service" "$SERVICE_FILE"
    sudo systemctl daemon-reload
fi

# Ensure app binds 0.0.0.0 (not 127.0.0.1) so Nginx can proxy
sudo sed -i 's/--host 127.0.0.1/--host 0.0.0.0/g' "$SERVICE_FILE" 2>/dev/null || true
sudo sed -i 's/--host localhost/--host 0.0.0.0/g' "$SERVICE_FILE" 2>/dev/null || true
sudo systemctl daemon-reload 2>/dev/null || true

echo "Stopping and starting tradingserver..."
sudo systemctl stop "$SERVICE_NAME" 2>/dev/null || true
sleep 2
sudo systemctl start "$SERVICE_NAME"

echo "Waiting 25s for startup..."
sleep 25

echo ""
echo "=== Service Status ==="
sudo systemctl status "$SERVICE_NAME" --no-pager -l | head -30

echo ""
echo "=== Health Check ==="
if curl -sf -m 10 http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "OK - Health responding"
    curl -s http://127.0.0.1:8000/health | python3 -m json.tool 2>/dev/null || curl -s http://127.0.0.1:8000/health
else
    echo "FAIL - Health not responding. Last 80 log lines:"
    sudo journalctl -u "$SERVICE_NAME" -n 80 --no-pager || true
fi

sudo systemctl enable "$SERVICE_NAME" 2>/dev/null || true
echo ""
echo "=== Done ==="
