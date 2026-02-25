#!/bin/bash
# Print key status (SET/NOT SET) - never print secrets. Used by ExecStartPre.
set -e
cd "${1:-/home/ubuntu/local_3comas_clone_v2}"
[ -f .env ] && set -a && source .env && set +a
echo "KEYS: KRAKEN_KEY=${KRAKEN_API_KEY:+SET} KRAKEN_SECRET=${KRAKEN_API_SECRET:+SET}"
echo "KEYS: ALPACA_PAPER_KEY=${ALPACA_API_KEY_PAPER:+SET} ALPACA_PAPER_SECRET=${ALPACA_API_SECRET_PAPER:+SET}"
echo "KEYS: ALPACA_LIVE_KEY=${ALPACA_API_KEY_LIVE:+SET} ALPACA_LIVE_SECRET=${ALPACA_API_SECRET_LIVE:+SET}"
