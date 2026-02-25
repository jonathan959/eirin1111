#!/bin/bash
# Validate app imports (and optionally /health) before we restart the live service.
# Lightweight: import-only to avoid OOM on smaller instances. Exit 0 = OK to restart.

set -e
cd /home/ubuntu/local_3comas_clone_v2
export PYTHONUNBUFFERED=1
# Fallback TMPDIR - /tmp often full/unwritable after EC2 reboot
export TMPDIR="/home/ubuntu/local_3comas_clone_v2/tmp"
mkdir -p "$TMPDIR" 2>/dev/null; chmod 700 "$TMPDIR" 2>/dev/null
# Minimal import path for validation (avoids OOM/heavy deps on small instances)
export DISABLE_PHASE2=1
export DISABLE_PHASE3=1
export ENABLE_PHASE4_ULTIMATE=0

echo "=== Validate before restart (import-only) ==="

source venv/bin/activate
# Skip pip - temp dir often broken/full after EC2 reboot; deps should already be installed
# pip install -q -r requirements.txt 2>/dev/null || true

if python -c "import sys; sys.path.insert(0, '.'); from one_server_v2 import app; assert app is not None; print('Import OK')" 2>&1; then
  echo "=== Validation passed ==="
  exit 0
fi

echo "Validation failed: import or startup error."
exit 1
