import sys
import os
import json
import time

# Add path
sys.path.append(os.getcwd())

from intelligence_layer import IntelligenceLayer, IntelligenceContext
# Mock or Minimal imports from worker_api if needed, but we can verify DB/Intelligence directly if decoupled
# For now, let's just inspect the IntelligenceLayer output locally if possible,
# or simulate what worker_api recommendations scan does.

print("=== START SMOKE TEST ===")

try:
    il = IntelligenceLayer()
    print("Intelligence Layer initialized.")
except Exception as e:
    print(f"FATAL: Intelligence Layer init failed: {e}")
    sys.exit(1)

# Inspect what symbols it thinks it should scan
# This might be in worker_api's globals or passed in.
# Let's check RECO_SYMBOLS env var or default from worker_api if we can import it
try:
    from worker_api import RECO_SYMBOLS
    print(f"RECO_SYMBOLS count: {len(RECO_SYMBOLS)}")
    print(f"First 10 symbols: {RECO_SYMBOLS[:10]}")
    
    # Check for Stocks
    stocks = [s for s in RECO_SYMBOLS if not s.endswith("/USD") and not s.endswith("USDT")]
    print(f"Potential Stocks found in list: {len(stocks)}")
    if len(stocks) > 0:
        print(f"Sample stocks: {stocks[:5]}")
except ImportError:
    print("Could not import RECO_SYMBOLS from worker_api. Checking env...")
    env_syms = os.getenv("RECO_SYMBOLS", "")
    if env_syms:
        print(f"RECO_SYMBOLS from Env: {len(env_syms.split(','))}")
    else:
        print("RECO_SYMBOLS not found.")

# Check IntelligenceLayer capabilities
print(f"Has Analyzer: {'analyzer' in dir(il)}")
# print(f"Registered Models: {il.models.keys() if hasattr(il, 'models') else 'Unknown'}")

# Try to gen a recommendation for a dummy asset
ctx = IntelligenceContext(
    symbol="BTC/USD",
    last_price=50000.0,
    candles_1h=[],
    candles_4h=[],
    candles_1d=[],
    candles_1w=[],
    btc_context={"risk_off": False},
    now_ts=int(time.time()),
    last_price_ts=int(time.time()),
    dry_run=True
)

res = il.generate_recommendation(ctx, "short")
print("Dummy Scan Result:")
print(json.dumps(res, indent=2))

print("=== END SMOKE TEST ===")
