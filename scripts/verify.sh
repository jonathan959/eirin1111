#!/bin/bash
# Verify preflight and health endpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source venv/bin/activate
python scripts/preflight.py

# worker.py defaults to port 9001
PORT="${WORKER_PORT:-9001}"
curl -fsS "http://127.0.0.1:${PORT}/health"
curl -fsS "http://127.0.0.1:${PORT}/api/health"

echo "Verify OK"
exit 0
