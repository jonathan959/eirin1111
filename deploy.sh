#!/bin/bash
# deploy.sh - Deploy trading bot to ubuntu@3.148.6.246
# Uploads ALL necessary files (Python, templates, static, config)

set -e
HOST="ubuntu@3.148.6.246"
REMOTE_PATH="/home/ubuntu/local_3comas_clone_v2"
SERVICE="tradingserver"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Trading Bot Deploy ==="
echo "Host: $HOST"
echo "Path: $REMOTE_PATH"
echo ""

# 1. Test SSH connection
echo "[1/6] Testing SSH connection..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$HOST" "echo OK" 2>/dev/null; then
    echo "ERROR: SSH connection failed."
    exit 1
fi
echo "SSH OK"
echo ""

# 2. Backup
echo "[2/6] Backing up current app..."
ssh "$HOST" "cd $REMOTE_PATH && mkdir -p backups && tar czf backups/pre_${TIMESTAMP}.tar.gz --exclude='backups' --exclude='venv' --exclude='.venv' --exclude='__pycache__' --exclude='botdb.sqlite3' *.py templates/ static/ .env 2>/dev/null || true"
echo "Backup: backups/pre_${TIMESTAMP}.tar.gz"
echo ""

# 3. Upload ALL Python files
echo "[3/6] Uploading Python files..."
scp -q \
    one_server.py \
    one_server_v2.py \
    worker_api.py \
    bot_manager.py \
    db.py \
    executor.py \
    intelligence_layer.py \
    strategies.py \
    alpaca_client.py \
    unified_alpaca_client.py \
    alpaca_adapter.py \
    alpaca_rate_limiter.py \
    kraken_client.py \
    symbol_classifier.py \
    market_data.py \
    autopilot.py \
    explore_v2.py \
    risk_circuit_breaker.py \
    circuit_breaker.py \
    risk_engine.py \
    portfolio_risk_manager.py \
    env_utils.py \
    constants.py \
    init_db.py \
    discord_notifications.py \
    notification_manager.py \
    health_monitor.py \
    data_cache.py \
    websocket_manager.py \
    ml_predictor.py \
    ml_ensemble.py \
    capital_allocator.py \
    kelly_criterion.py \
    "$HOST:$REMOTE_PATH/" 2>/dev/null

# Upload remaining Python files (optional modules - skip missing)
for f in \
    phase1_intelligence.py phase2_data_fetcher.py sentiment_analyzer.py \
    correlation_trading.py pairs_trading.py zscore_trading.py \
    momentum_ranking.py volatility_forecaster.py order_flow_analyzer.py \
    high_frequency.py advanced_strategies.py portfolio_correlation.py \
    enhanced_rate_limiter.py backtest.py trade_reasoning.py \
    adaptive_scorer.py adaptive_parameters.py; do
    [ -f "$f" ] && scp -q "$f" "$HOST:$REMOTE_PATH/" 2>/dev/null || true
done

echo "Uploaded Python files"

# 4. Upload templates and static
echo "[4/6] Uploading templates and static..."
ssh "$HOST" "mkdir -p $REMOTE_PATH/templates $REMOTE_PATH/static $REMOTE_PATH/tests $REMOTE_PATH/scripts"
scp -q templates/*.html "$HOST:$REMOTE_PATH/templates/" 2>/dev/null || true
scp -rq static/ "$HOST:$REMOTE_PATH/static/" 2>/dev/null || true

# Upload config files (not .env - keep server's .env)
scp -q requirements.txt start.sh tradingserver.service AGENTS.md "$HOST:$REMOTE_PATH/" 2>/dev/null || true

echo "Uploaded templates, static, and config"
echo ""

# 5. Run DB migration and restart
echo "[5/6] Running DB migration and restarting..."
ssh "$HOST" "cd $REMOTE_PATH && venv/bin/python init_db.py 2>/dev/null || true"
ssh "$HOST" "sudo systemctl restart $SERVICE"
sleep 5

if ssh "$HOST" "sudo systemctl is-active --quiet $SERVICE"; then
    echo "SUCCESS: $SERVICE is running"
else
    echo "WARNING: $SERVICE may not be running"
    ssh "$HOST" "sudo journalctl -u $SERVICE --no-pager -n 20" 2>/dev/null || true
    exit 1
fi
echo ""

# 6. Health check
echo "[6/6] Health check..."
sleep 3
ssh "$HOST" "curl -s http://localhost:8000/api/health | python3 -m json.tool 2>/dev/null | head -10" || echo "Health check skipped"
echo ""
echo "=== Deployment complete ==="
echo "Dashboard: http://3.148.6.246:8000/"
