#!/bin/bash
# Smoke test: curl /health and /api/bots/summary. Exit non-zero on failure.
# Usage: ./smoke_test.sh [base_url]
#   base_url defaults to http://127.0.0.1:8000

BASE="${1:-http://127.0.0.1:8000}"
FAIL=0

echo "=== Smoke test: $BASE ==="

if ! curl -sf -m 8 "$BASE/health" >/dev/null 2>&1; then
  echo "FAIL: $BASE/health not reachable"
  FAIL=1
else
  echo "OK: /health"
fi

if ! curl -sf -m 8 "$BASE/api/bots/summary" >/dev/null 2>&1; then
  echo "FAIL: $BASE/api/bots/summary not reachable"
  FAIL=1
else
  echo "OK: /api/bots/summary"
fi

if [ $FAIL -eq 1 ]; then
  exit 1
fi
echo "=== All checks passed ==="
exit 0
