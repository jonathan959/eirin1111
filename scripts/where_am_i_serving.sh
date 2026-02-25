#!/bin/bash
# Port/proxy consistency check: what is listening, and do health endpoints respond?
set -euo pipefail

echo "=== Processes on 8000 and 9001 ==="
sudo ss -ltnp 2>/dev/null | grep -E ":8000|:9001" || echo "(none)"
echo ""
sudo lsof -i :8000 -i :9001 2>/dev/null || true

echo ""
echo "=== Health check :8000 ==="
curl -fsS -m 5 "http://127.0.0.1:8000/health" 2>/dev/null | head -c 200 || echo "FAILED or no /health"
echo ""
curl -fsS -m 5 "http://127.0.0.1:8000/api/health" 2>/dev/null | head -c 200 || echo "FAILED or no /api/health"
echo ""

echo ""
echo "=== Health check :9001 ==="
curl -fsS -m 5 "http://127.0.0.1:9001/health" 2>/dev/null | head -c 200 || echo "FAILED or no /health"
echo ""
curl -fsS -m 5 "http://127.0.0.1:9001/api/health" 2>/dev/null | head -c 200 || echo "FAILED or no /api/health"
echo ""
