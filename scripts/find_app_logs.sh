#!/bin/bash
# Run on the server to find what is serving the app and where its logs are.
# Usage: bash find_app_logs.sh   (or run via: ssh ... 'bash -s' < find_app_logs.sh)

echo "=== 1. What is listening on port 8000 and 80? ==="
sudo ss -tlnp 2>/dev/null | grep -E ':8000|:80 ' || true
sudo lsof -i :8000 -i :80 2>/dev/null | head -20 || true

echo ""
echo "=== 2. Python/uvicorn processes ==="
ps aux | grep -E 'python|uvicorn' | grep -v grep || true

echo ""
echo "=== 3. Systemd units that might be the app ==="
systemctl list-units --all --no-pager 2>/dev/null | grep -iE 'trading|bot|uvicorn|ai-bot|one_server' || true
ls -la /etc/systemd/system/*.service 2>/dev/null | grep -iE 'trading|bot|ai|uvicorn' || true

echo ""
echo "=== 4. Look for log files in project and home ==="
for dir in ~/local_3comas_clone_v2 ~/local_3comas_clone_v2_staging ~; do
  for f in deploy.log nohup.out deploy_aws.log .deploy.log; do
    [ -f "$dir/$f" ] && echo "FOUND: $dir/$f" && tail -5 "$dir/$f" && echo "---"
  done
done

echo ""
echo "=== 5. If a unit like 'tradingserver' or 'ai-bot' exists, run: ==="
echo "   sudo journalctl -u tradingserver -n 100 --no-pager"
echo "   sudo journalctl -u ai-bot -n 100 --no-pager"
echo ""
echo "=== 6. All systemd service names (to find the right one) ==="
systemctl list-unit-files --type=service --no-pager 2>/dev/null | grep -E 'trading|bot|ai|uvicorn|gunicorn' || echo "(none found)"
