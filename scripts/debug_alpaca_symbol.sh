#!/bin/bash
# Quick debug: test Alpaca symbol and feed. Run from project root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "ALPACA_DATA_FEED=${ALPACA_DATA_FEED:-}(empty=default)"
source venv/bin/activate 2>/dev/null || true

python -c "
import os
print('ALPACA_DATA_FEED=', repr(os.getenv('ALPACA_DATA_FEED', '')))
try:
    from alpaca_client import AlpacaClient
    c = AlpacaClient(mode='paper')
    print('get_asset(TOYO):', c.get_asset('TOYO'))
except Exception as e:
    print('get_asset(TOYO) error:', e)
try:
    from alpaca_client import AlpacaClient
    c = AlpacaClient(mode='paper')
    n = len(c.get_ohlcv('TOYO', '1h', 50))
    print('get_ohlcv(TOYO, 1h, 50):', n, 'candles')
except Exception as e:
    print('get_ohlcv(TOYO) error:', e)
try:
    from alpaca_client import AlpacaClient
    c = AlpacaClient(mode='paper')
    n = len(c.get_ohlcv('TM', '1h', 50))
    print('get_ohlcv(TM, 1h, 50):', n, 'candles (known ticker)')
except Exception as e:
    print('get_ohlcv(TM) error:', e)
"
