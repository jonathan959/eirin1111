#!/bin/bash
# Fetches candles for a symbol across 1h, 4h, 1d. Prints counts, staleness, provider, errors.
# Usage: ./scripts/data_probe.sh [SYMBOL]
# Default: AAPL (or pass TOYO, TM, etc.)
set -e

SYMBOL="${1:-AAPL}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source venv/bin/activate 2>/dev/null || true
[ -f .env ] && set -a && source .env && set +a

echo "=== Data probe: $SYMBOL ==="
echo "Timeframes: 1h, 4h, 1d"
echo ""

python3 -c "
import os
import time
from market_data import MarketDataRouter, get_last_data_error
from alpaca_client import AlpacaClient

sym = '$SYMBOL'
router = None
try:
    client = AlpacaClient(mode='paper')
    router = MarketDataRouter(kraken_client=None, alpaca_paper=client, alpaca_live=None)
except Exception as e:
    print('Router init error:', e)
    exit(1)

now = int(time.time())
for tf in ['1h', '4h', '1d']:
    try:
        candles = router.get_candles(sym, tf, 100, market_type='stocks') if router else []
        cnt = len(candles) if candles else 0
        last_ts = candles[-1][0] // 1000 if candles and len(candles[-1]) > 0 else 0
        age_min = (now - last_ts) / 60 if last_ts else -1
        stale = 'STALE' if age_min > 120 else 'OK'
        print(f'{tf}: {cnt} bars, last_candle_age_min={age_min:.0f}, {stale}')
    except Exception as e:
        print(f'{tf}: ERROR -', e)
err = get_last_data_error(sym)
if err:
    print('Data error:', err)
print('Done.')
"
