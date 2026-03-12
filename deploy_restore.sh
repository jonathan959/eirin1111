#!/bin/bash
# Restore from .previous (rollback). Run from deploy.ps1.
set -e
cd /home/ubuntu/local_3comas_clone_v2

FILES="one_server.py worker_api.py worker.py bot_manager.py executor.py strategies.py symbol_classifier.py alpaca_adapter.py alpaca_client.py price_predictor.py strategy_optimizer.py portfolio_optimizer.py anomaly_detector.py db.py kraken_client.py intelligence_layer.py phase1_intelligence.py portfolio_correlation.py env_utils.py multi_timeframe.py sentiment_analyzer.py order_book_analyzer.py ml_predictor.py kelly_criterion.py app.py requirements.txt"
SCRIPTS="validate_before_restart.sh fix_service.sh quick_fix_502.sh deploy_restart.sh health_watchdog.sh check_nginx.sh setup_nginx.sh fix_port80.sh"

for f in $FILES $SCRIPTS; do
  [ -f .previous/"$f" ] && cp -p .previous/"$f" . 2>/dev/null || true
done
[ -d .previous/templates ] && rm -rf templates && cp -rp .previous/templates .
[ -d .previous/static ] && rm -rf static && cp -rp .previous/static .
[ -f .previous/tradingserver.service ] && cp -p .previous/tradingserver.service .
# Restore DB from backup if present (recovers bots/data after bad deploy)
if [ -f .previous/botdb.sqlite3 ]; then
  echo "Restoring database from .previous/botdb.sqlite3 (recover bots)..."
  cp -p .previous/botdb.sqlite3 . 2>/dev/null || true
  for w in .previous/botdb.sqlite3-wal .previous/botdb.sqlite3-shm; do
    [ -f "$w" ] && cp -p "$w" . 2>/dev/null || true
  done
fi

sudo cp tradingserver.service /etc/systemd/system/
sudo systemctl daemon-reload
[ -x deploy_restart.sh ] && bash deploy_restart.sh || sudo systemctl restart tradingserver
echo "RESTORED"
