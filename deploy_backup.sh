#!/bin/bash
# Lightweight backup of deployed files only. Run from deploy.ps1 with: timeout 25 bash deploy_backup.sh
set -e
cd /home/ubuntu/local_3comas_clone_v2
rm -rf .previous 2>/dev/null || true
mkdir -p .previous

FILES="one_server.py worker_api.py worker.py bot_manager.py executor.py strategies.py symbol_classifier.py alpaca_adapter.py alpaca_client.py price_predictor.py strategy_optimizer.py portfolio_optimizer.py anomaly_detector.py db.py kraken_client.py intelligence_layer.py phase1_intelligence.py portfolio_correlation.py env_utils.py multi_timeframe.py sentiment_analyzer.py order_book_analyzer.py ml_predictor.py kelly_criterion.py app.py requirements.txt"
SCRIPTS="validate_before_restart.sh fix_service.sh quick_fix_502.sh deploy_restart.sh health_watchdog.sh check_nginx.sh setup_nginx.sh fix_port80.sh"

for f in $FILES $SCRIPTS; do
  [ -f "$f" ] && cp -p "$f" .previous/ 2>/dev/null || true
done
[ -d templates ] && cp -rp templates .previous/ 2>/dev/null || true
[ -d static ] && cp -rp static .previous/ 2>/dev/null || true
[ -f tradingserver.service ] && cp -p tradingserver.service .previous/ 2>/dev/null || true

echo "BACKUP_DONE"
