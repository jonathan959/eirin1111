#!/bin/bash
# deploy.sh - Deploy trading bot to ubuntu@3.148.6.246
# Follow DEPLOYMENT_RULES.md - SSH-only, no Docker

set -e
HOST="ubuntu@3.148.6.246"
REMOTE_PATH="/home/ubuntu/bot"
SERVICE="tradingserver"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Trading Bot Deploy ==="
echo "Host: $HOST"
echo "Path: $REMOTE_PATH"
echo ""

# 1. Test SSH connection first
echo "[1/6] Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" "echo OK"; then
    echo "ERROR: SSH connection failed. Run SSH_DIAGNOSTIC.sh for troubleshooting."
    exit 1
fi
echo "SSH OK"
echo ""

# 2. Backup all files with timestamp
echo "[2/6] Backing up current app..."
ssh "$HOST" "cd $REMOTE_PATH && mkdir -p backups && cp -a . backups/pre_${TIMESTAMP} 2>/dev/null || true"
echo "Backup: backups/pre_${TIMESTAMP}"
echo ""

# 3. Upload core Python files
echo "[3/6] Uploading app files..."
scp -q app.py worker_api.py bot_manager.py db.py strategies.py intelligence_layer.py constants.py "$HOST:$REMOTE_PATH/"
echo "Uploaded: app.py, worker_api.py, bot_manager.py, db.py, strategies.py, intelligence_layer.py, constants.py"
echo ""

# 4. Restart service
echo "[4/6] Restarting $SERVICE..."
ssh "$HOST" "sudo systemctl restart $SERVICE"
sleep 3
echo ""

# 5. Check service status
echo "[5/6] Checking service status..."
if ssh "$HOST" "sudo systemctl is-active --quiet $SERVICE"; then
    echo "SUCCESS: $SERVICE is running"
else
    echo "WARNING: $SERVICE may not be running. Checking status..."
    ssh "$HOST" "sudo systemctl status $SERVICE --no-pager" || true
    exit 1
fi
echo ""

# 6. Final message
echo "[6/6] Deployment complete."
echo "Verify: ssh $HOST 'tail -20 /home/ubuntu/botdata/logs/worker.log'"
echo ""
