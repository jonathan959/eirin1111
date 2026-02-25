#!/usr/bin/env bash
# Local verification (WSL / dev machine). Run from project root.
# Usage: bash scripts/verify_local.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "=== Local verification ($(pwd)) ==="
echo ""
echo "=== Project ==="
[ -d venv ] && echo "venv: OK" || echo "venv: MISSING"
[ -f .env ] && echo ".env: OK" || echo ".env: MISSING"
[ -f requirements.txt ] && echo "requirements.txt: OK" || echo "requirements.txt: MISSING"
echo ""
echo "=== Ports (local dev) ==="
(ss -lntp 2>/dev/null || netstat -tlnp 2>/dev/null) | grep -E ':8000|:9001' || echo "(nothing on 8000/9001 - start with: uvicorn one_server:app --port 8000)"
echo ""
echo "=== Health (if server running on 8000) ==="
curl -sS -o /dev/null -w "http://127.0.0.1:8000/health: %{http_code}\n" -m 3 http://127.0.0.1:8000/health 2>/dev/null || echo "http://127.0.0.1:8000/health: not reachable"
echo ""
echo "=== Done ==="
