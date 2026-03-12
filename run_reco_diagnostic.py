#!/usr/bin/env python3
"""
Run recommendation scan diagnostic - shows exactly where crypto/stock scan succeeds or fails.
Run: python run_reco_diagnostic.py
"""
import os
import sys
import logging

# Force logging to stdout with RECO_DEBUG visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
    force=True,
)

# Load env before imports
from env_utils import load_env
load_env()

# Now import worker_api - this will trigger module init
print("\n=== RECO DIAGNOSTIC: Importing worker_api ===")
import worker_api

print("\n=== Checking trading clients ===")
kr = worker_api._kraken_ready()
alp = bool(worker_api.alpaca_live or worker_api.alpaca_paper)
print(f"Kraken ready: {kr}")
print(f"Alpaca ready: {alp}")
if not kr and not alp:
    print("WARNING: No trading clients. Crypto requires Kraken; stocks require Alpaca.")
    print("Crypto will show 0 symbols. Stocks will show 0 if Alpaca not configured.")

print("\n=== Calling _reco_symbols ===")
symbols = worker_api._reco_symbols(quote="USD")
crypto = [s for s in symbols if "/" in s]
stocks = [s for s in symbols if "/" not in s and len(s) < 6]
print(f"Total symbols: {len(symbols)}")
print(f"Crypto: {len(crypto)} (e.g. {crypto[:5] if crypto else 'NONE'})")
print(f"Stocks: {len(stocks)} (e.g. {stocks[:15] if stocks else 'NONE'})")

print("\n=== After momentum filter ===")
symbols2 = worker_api._apply_momentum_filter_to_universe(symbols)
print(f"After stock momentum filter: {len(symbols2)} symbols")

symbols3 = worker_api._apply_crypto_momentum_filter(symbols2)
print(f"After crypto momentum filter: {len(symbols3)} symbols")

print("\n=== Running ONE scan iteration (short horizon) ===")
worker_api._scan_recommendations("short")

print("\n=== DB recommendation counts ===")
from db import count_recommendations_by_horizon
counts = count_recommendations_by_horizon()
print(f"short={counts.get('short',0)} medium={counts.get('medium',0)} long={counts.get('long',0)}")

print("\n=== Sample crypto recommendations ===")
from db import list_recommendations
import json
all_rows = list_recommendations(horizon="short", limit=200)
def _mt(r):
    m = r.get("metrics")
    if isinstance(m, str):
        try: return json.loads(m).get("market_type")
        except Exception: return None
    return (m or {}).get("market_type")
rows = [r for r in all_rows if _mt(r) == "crypto"]
for r in (rows or [])[:5]:
    print(f"  {r.get('symbol')} score={r.get('score')}")

print("\n=== Sample stock recommendations ===")
rows2 = [r for r in all_rows if _mt(r) == "stocks"]
for r in (rows2 or [])[:10]:
    print(f"  {r.get('symbol')} score={r.get('score')}")

print("\n=== DIAGNOSTIC COMPLETE ===")
