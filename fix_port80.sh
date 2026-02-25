#!/bin/bash
# Fix port 80 conflict and start nginx. Avoid killing nginx (reload instead).

echo "=== Fixing Port 80 Conflict ==="

PORT80_PROCESS=$(sudo lsof -i :80 -t 2>/dev/null | head -1)
if [ -n "$PORT80_PROCESS" ]; then
    PROCESS_NAME=$(ps -p $PORT80_PROCESS -o comm= 2>/dev/null || echo "unknown")
    echo "Process $PORT80_PROCESS ($PROCESS_NAME) using port 80"

    if [ "$PROCESS_NAME" = "nginx" ]; then
        echo "Nginx already on 80. Reloading (no kill)..."
        sudo systemctl reload nginx 2>/dev/null || sudo systemctl restart nginx
        echo "Nginx reloaded."
        exit 0
    fi

    if [ "$PROCESS_NAME" = "apache2" ] || [ "$PROCESS_NAME" = "httpd" ]; then
        echo "Stopping Apache..."
        sudo systemctl stop apache2 2>/dev/null || sudo systemctl stop httpd 2>/dev/null || true
        sudo systemctl disable apache2 2>/dev/null || sudo systemctl disable httpd 2>/dev/null || true
    fi

    if sudo lsof -i :80 -t &>/dev/null; then
        echo "Killing non-nginx process on port 80..."
        sudo kill -9 $(sudo lsof -i :80 -t) 2>/dev/null || true
        sleep 2
    fi
fi

if sudo lsof -i :80 -t &>/dev/null; then
    echo "Warning: Port 80 still in use"
    sudo lsof -i :80
else
    echo "Port 80 is now free"
fi

echo ""
echo "Starting nginx..."
sudo systemctl start nginx

# Check nginx status
echo ""
echo "=== Nginx Status ==="
sudo systemctl status nginx --no-pager -l | head -15

# Test the website
echo ""
echo "=== Testing Website ==="
sleep 2
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/ --max-time 5 2>/dev/null || echo "000")
echo "Local test: HTTP $HTTP_CODE"

HTTP_CODE_IP=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1/ --max-time 5 2>/dev/null || echo "000")
echo "Local nginx test: HTTP $HTTP_CODE_IP"

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE_IP" = "200" ]; then
    echo ""
    echo "=== SUCCESS: Website is accessible ==="
else
    echo ""
    echo "=== Checking for issues ==="
    
    # Check if uvicorn is running
    if pgrep -f "uvicorn" > /dev/null; then
        echo "✓ Uvicorn is running"
    else
        echo "✗ Uvicorn is NOT running - restarting service..."
        sudo systemctl restart tradingserver
        sleep 3
    fi
    
    # Check nginx config
    sudo nginx -t
    
    # Show nginx error log
    echo ""
    echo "Recent nginx errors:"
    sudo tail -5 /var/log/nginx/error.log 2>/dev/null || echo "No error log"
fi

echo ""
echo "=== Fix Complete ==="
