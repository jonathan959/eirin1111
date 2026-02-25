#!/bin/bash
# Check and fix nginx configuration

echo "=== Checking Nginx Configuration ==="

# Check if nginx is installed
if ! command -v nginx &> /dev/null; then
    echo "Nginx is not installed. Installing..."
    sudo apt-get update
    sudo apt-get install -y nginx
fi

# Check nginx status
echo ""
echo "=== Nginx Status ==="
sudo systemctl status nginx --no-pager -l | head -15 || true

# Check if service is running on port 8000
echo ""
echo "=== Checking if service is listening on port 8000 ==="
sudo netstat -tlnp | grep :8000 || ss -tlnp | grep :8000 || echo "Port 8000 not found"

# Check nginx config
echo ""
echo "=== Checking Nginx Configuration ==="
if [ -f /etc/nginx/sites-available/tradingserver ]; then
    echo "Found tradingserver config:"
    cat /etc/nginx/sites-available/tradingserver
elif [ -f /etc/nginx/sites-available/default ]; then
    echo "Found default config:"
    cat /etc/nginx/sites-available/default | head -50
else
    echo "No nginx config found for tradingserver"
fi

echo ""
echo "=== Testing Service Directly ==="
curl -s http://localhost:8000/health | python3 -m json.tool || echo "Service not responding on localhost:8000"

echo ""
echo "=== Check Complete ==="
