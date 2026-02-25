#!/bin/bash
# Setup nginx reverse proxy for AI bot (3.148.6.246)

set -e

REMOTE_DIR="${1:-/home/ubuntu/local_3comas_clone_v2}"
cd "$REMOTE_DIR" 2>/dev/null || true
SERVICE_PORT=8000
NGINX_CONFIG="/etc/nginx/sites-available/ai-bot"
PUBLIC_IP=$(curl -s --max-time 5 http://checkip.amazonaws.com 2>/dev/null || echo "3.148.6.246")

echo "=== Setting up Nginx Reverse Proxy (ai-bot) ==="
echo "Server IP: $PUBLIC_IP"
echo "Service Port: $SERVICE_PORT"
echo ""

# Install nginx if not installed
if ! command -v nginx &> /dev/null; then
    echo "Installing nginx..."
    sudo apt-get update
    sudo apt-get install -y nginx
fi

# Use repo nginx-ai-bot.conf if present, else create inline
if [ -f "$REMOTE_DIR/nginx-ai-bot.conf" ]; then
    echo "Using nginx-ai-bot.conf from repo..."
    sudo cp "$REMOTE_DIR/nginx-ai-bot.conf" "$NGINX_CONFIG"
else
    echo "Creating nginx configuration..."
    sudo tee "$NGINX_CONFIG" > /dev/null <<EOF
upstream ai_bot_backend {
    server 127.0.0.1:$SERVICE_PORT;
    keepalive 32;
}

server {
    listen 80;
    listen [::]:80;
    server_name $PUBLIC_IP _;

    proxy_connect_timeout 10s;
    proxy_send_timeout 300s;
    proxy_read_timeout 3600s;
    client_max_body_size 10M;

    proxy_next_upstream error timeout http_502 http_503 http_504;
    proxy_next_upstream_tries 2;
    proxy_next_upstream_timeout 30s;

    location / {
        proxy_pass http://ai_bot_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_read_timeout 3600;
    }

    location /health {
        proxy_pass http://ai_bot_backend/health;
        proxy_connect_timeout 5s;
        proxy_read_timeout 10s;
        access_log off;
    }
}
EOF
fi

# Enable ai-bot site, remove old/default
sudo ln -sf "$NGINX_CONFIG" /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo rm -f /etc/nginx/sites-enabled/tradingserver 2>/dev/null || true

# Test nginx configuration
echo ""
echo "Testing nginx configuration..."
if sudo nginx -t; then
    echo "✓ Nginx configuration is valid"
    
    echo "Reloading nginx..."
    sudo systemctl reload nginx
    
    echo ""
    echo "=== Nginx Status ==="
    sudo systemctl status nginx --no-pager -l | head -10
    
    echo ""
    echo "=== Testing Endpoints ==="
    echo "Testing http://$PUBLIC_IP/health..."
    curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://$PUBLIC_IP/health || echo "Failed to connect"
    
    echo ""
    echo "=== Health Watchdog (502 auto-recovery) ==="
    WATCHDOG_SCRIPT="${REMOTE_DIR}/health_watchdog.sh"
    if [ -f "$WATCHDOG_SCRIPT" ]; then
      chmod +x "$WATCHDOG_SCRIPT" 2>/dev/null || true
      CRON_ENTRY="*/2 * * * * $WATCHDOG_SCRIPT"
      if ! crontab -l 2>/dev/null | grep -q health_watchdog; then
        (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab - 2>/dev/null && echo "Watchdog cron added (every 2 min)" || echo "Watchdog: add manually: $CRON_ENTRY"
      else
        echo "Watchdog cron already present"
      fi
    else
      echo "Watchdog script not found at $WATCHDOG_SCRIPT"
    fi

    echo ""
    echo "=== Setup Complete ==="
    echo "Site accessible at: http://$PUBLIC_IP"
else
    echo "✗ Nginx configuration has errors."
    exit 1
fi
