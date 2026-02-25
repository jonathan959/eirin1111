#!/bin/bash
# Configure nginx reverse proxy: :80 -> UI (8000), timeouts, headers.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVICE_PORT="${SERVICE_PORT:-8000}"
NGINX_CONFIG="/etc/nginx/sites-available/tradingserver"
PUBLIC_IP=$(curl -s --max-time 5 http://checkip.amazonaws.com 2>/dev/null || echo "3.148.6.246")

echo "=== Install Nginx (port $SERVICE_PORT) ==="
sudo apt-get update -qq
sudo apt-get install -y nginx 2>/dev/null || true

sudo tee "$NGINX_CONFIG" >/dev/null <<EOF
upstream tradingserver_backend {
    server 127.0.0.1:$SERVICE_PORT;
    keepalive 8;
}

server {
    listen 80;
    listen [::]:80;
    server_name $PUBLIC_IP _;

    proxy_connect_timeout 10s;
    proxy_send_timeout 120s;
    proxy_read_timeout 120s;
    client_max_body_size 10M;

    proxy_next_upstream error timeout http_502 http_503 http_504;
    proxy_next_upstream_tries 2;
    proxy_next_upstream_timeout 30s;

    location / {
        proxy_pass http://tradingserver_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_buffering off;
    }

    location /health {
        proxy_pass http://tradingserver_backend/health;
        proxy_connect_timeout 5s;
        proxy_read_timeout 10s;
        access_log off;
    }

    location /api/ {
        proxy_pass http://tradingserver_backend/api/;
        proxy_connect_timeout 15s;
        proxy_read_timeout 90s;
        proxy_send_timeout 90s;
    }
}
EOF

sudo ln -sf "$NGINX_CONFIG" /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
sudo nginx -t && sudo systemctl reload nginx
echo "Nginx configured. Site: http://$PUBLIC_IP"
