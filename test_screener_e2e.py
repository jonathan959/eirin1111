#!/usr/bin/env python3
"""
End-to-end test for market screener: DB counts, API response, whitelist, horizons.

Run: python test_screener_e2e.py

Expects server on http://127.0.0.1:8000 or set BASE_URL env.
"""
import os
import sys

# Load env before imports
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().replace('"', "").replace("'", ""))

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")


def main():
    print("=" * 60)
    print("Market Screener End-to-End Test")
    print("=" * 60)

    # 1. Direct DB query
    print("\n1. Database record counts")
    try:
        from db import count_recommendations_by_horizon
        counts = count_recommendations_by_horizon()
        for h in ("short", "medium", "long"):
            n = counts.get(h, 0)
            status = "OK" if n > 0 else "EMPTY"
            print(f"   {h}: {n} records [{status}]")
    except Exception as e:
        print(f"   DB ERROR: {e}")
        return 1

    # 2. API calls per horizon
    print("\n2. API response per horizon (min_score=0 = Any)")
    api_ok = False
    try:
        import urllib.request
        import json

        for horizon in ("short", "medium", "long"):
            url = f"{BASE_URL}/api/recommendations?horizon={horizon}&min_score=0&signal=buy&limit=50&market_type=all"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode())
            items = data.get("items") or []
            ok = data.get("ok", False)
            status = data.get("status", "?")
            top30 = data.get("top30_whitelist_active", False)
            print(f"   {horizon}: ok={ok} status={status} items={len(items)} top30_active={top30}")
            if items:
                symbols = [it.get("symbol", "?") for it in items[:5]]
                print(f"      Sample: {symbols}")
            api_ok = True
    except Exception as e:
        print(f"   API ERROR: {e}")
        if "Connection refused" in str(e) or "timed out" in str(e).lower():
            print("   (Server not running on", BASE_URL + " - start with: uvicorn one_server:app --port 8000)")

    # 3. Whitelist verification (inline to avoid worker_api startup)
    print("\n3. Whitelist check (crypto symbols must be in top 30)")
    TOP_30 = frozenset({"XBT","ETH","SOL","XRP","ADA","AVAX","DOGE","DOT","LINK","MATIC","UNI","LTC","ATOM","BCH","ALGO","XLM","ICP","FIL","VET","SAND","MANA","AXS","THETA","EOS","AAVE","MKR","SNX","COMP","YFI","SUSHI","BTC"})
    def base_from_sym(s):
        s = (s or "").strip().upper()
        if "/" in s: return (s.split("/")[0] or "").strip()
        if "-" in s: return (s.split("-")[0] or "").strip()
        for suf in ("USD","USDT","USDC"):
            if s.endswith(suf) and len(s)>len(suf): return s[:-len(suf)].strip()
        return s
    for sym in ["BTC/USD", "XBT/USD", "ETH/USD", "NIGHT/USD", "SKY/USD", "BTCUSD"]:
        base = base_from_sym(sym)
        passed = base in TOP_30
        print(f"   {sym} -> base={base} -> {'PASS' if passed else 'BLOCKED'}")

    # 4. Summary
    print("\n4. Summary")
    all_ok = all(counts.get(h, 0) > 0 for h in ("short", "medium", "long"))
    if all_ok:
        print("   All horizons have data. OK.")
    else:
        empty = [h for h in ("short", "medium", "long") if counts.get(h, 0) == 0]
        print(f"   EMPTY: {empty}. Start server - bootstrap will scan medium. Or: POST /api/recommendations/scan?horizon=medium")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
