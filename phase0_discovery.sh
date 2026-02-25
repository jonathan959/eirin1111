#!/bin/bash
# Phase 0 Discovery - run on the EC2 server to diagnose deployment state.
# Usage: bash phase0_discovery.sh

echo "=== Phase 0: Discovery ==="
echo ""

echo "1) Key files / entrypoint"
echo "---"
ls -la /home/ubuntu/local_3comas_clone_v2/one_server_v2.py 2>/dev/null || echo "one_server_v2.py NOT FOUND"
ls -la /home/ubuntu/local_3comas_clone_v2/one_server.py 2>/dev/null || echo "one_server.py NOT FOUND"
ls -la /home/ubuntu/local_3comas_clone_v2/worker_api.py 2>/dev/null || echo "worker_api.py NOT FOUND"
echo ""

echo "2) Python / venv"
echo "---"
ls -la /home/ubuntu/local_3comas_clone_v2/venv/bin/python 2>/dev/null || echo "venv NOT FOUND"
[ -f /home/ubuntu/local_3comas_clone_v2/requirements.txt ] && head -10 /home/ubuntu/local_3comas_clone_v2/requirements.txt
echo ""

echo "3) Listening ports"
echo "---"
sudo ss -lntp 2>/dev/null | head -30
echo "---"
sudo lsof -i -P -n 2>/dev/null | head -80
echo ""

echo "4) OS / release"
echo "---"
lsb_release -a 2>/dev/null || cat /etc/os-release 2>/dev/null | head -10
echo ""

echo "5) Firewall (UFW)"
echo "---"
sudo ufw status verbose 2>/dev/null || echo "ufw not available"
echo ""

echo "6) Nginx status + config"
echo "---"
sudo systemctl status nginx --no-pager 2>/dev/null | head -15
ls -la /etc/nginx/sites-enabled /etc/nginx/sites-available 2>/dev/null
echo ""

echo "7) Service logs (ai-bot, tradingserver, nginx)"
echo "---"
sudo journalctl -u ai-bot --no-pager -n 80 2>/dev/null || echo "ai-bot: no logs"
echo "---"
sudo journalctl -u tradingserver --no-pager -n 80 2>/dev/null || echo "tradingserver: no logs"
echo "---"
sudo journalctl -u nginx --no-pager -n 50 2>/dev/null || echo "nginx: no logs"
echo ""
echo "=== Discovery complete ==="
