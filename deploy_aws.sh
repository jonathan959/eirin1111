#!/bin/bash
# Deploy on AWS (Linux) - bind 0.0.0.0 so app is reachable at http://3.151.143.63:8000
# Run on EC2: ./deploy_aws.sh   or   bash deploy_aws.sh

cd "$(dirname "$0")"
export DEPLOY_AWS=1
export PORT="${PORT:-8000}"

echo "=== Deploy one_server_v2 (AWS) ==="
echo "URL: http://3.151.143.63:${PORT}  (or your instance public IP)"
echo ""

python -m uvicorn one_server_v2:app --reload --port "$PORT" --host 0.0.0.0
