#!/bin/bash
# Run inside the same environment as the service (venv + .env).
# Use to compare paper vs live on server for TOYO 1h bars.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source venv/bin/activate
set -a
source .env
set +a

echo "=== Key status (SET=present, blank=missing) ==="
echo "ALPACA_API_KEY_PAPER=${ALPACA_API_KEY_PAPER:+SET}"
echo "ALPACA_API_SECRET_PAPER=${ALPACA_API_SECRET_PAPER:+SET}"
echo "ALPACA_API_KEY_LIVE=${ALPACA_API_KEY_LIVE:+SET}"
echo "ALPACA_API_SECRET_LIVE=${ALPACA_API_SECRET_LIVE:+SET}"
echo "ALPACA_DATA_FEED=${ALPACA_DATA_FEED:-(empty)}"
echo ""

python3 -c "
import os
print('Keys from Python:', 'PAPER=' + ('SET' if os.getenv('ALPACA_API_KEY_PAPER') else 'blank'), 'LIVE=' + ('SET' if os.getenv('ALPACA_API_KEY_LIVE') else 'blank'))

from alpaca_client import AlpacaClient

# Paper
try:
    cp = AlpacaClient(mode='paper')
    bars_p = cp.get_ohlcv('TOYO', '1h', 50)
    print('PAPER get_ohlcv(TOYO, 1h, 50):', len(bars_p), 'bars')
except Exception as e:
    print('PAPER get_ohlcv(TOYO) error:', e)

# Live
try:
    cl = AlpacaClient(mode='live')
    bars_l = cl.get_ohlcv('TOYO', '1h', 50)
    print('LIVE get_ohlcv(TOYO, 1h, 50):', len(bars_l), 'bars')
except Exception as e:
    print('LIVE get_ohlcv(TOYO) error:', e)

print('Done.')
"
