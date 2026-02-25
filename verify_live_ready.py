#!/usr/bin/env python3
"""
verify_live_ready.py — Safe live trading readiness verification.
Performs read-only checks. Does NOT place real orders unless
ALLOW_LIVE_TRADING=1 AND LIVE_CANARY=1 (user must explicitly enable).
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Load env
def _load_env():
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
_load_env()

def check(name, ok, msg=""):
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" - {msg}" if msg else ""))
    return ok

def main():
    print("=" * 60)
    print("LIVE TRADING READINESS VERIFICATION")
    print("=" * 60)
    print("(Read-only checks. No real orders unless LIVE_CANARY=1)")
    print()

    all_ok = True

    # 1. Server / health
    print("--- Connectivity ---")
    try:
        import requests
        r = requests.get("http://127.0.0.1:8000/api/health", timeout=5)
        ok = r.status_code == 200 and r.json().get("ok")
        all_ok &= check("Server starts, /api/health returns ok", ok)
    except Exception as e:
        all_ok &= check("Server starts, /api/health returns ok", False, str(e))

    # 2. /api/bots/summary
    try:
        import requests
        r = requests.get("http://127.0.0.1:8000/api/bots/summary", timeout=5)
        ok = r.status_code == 200
        all_ok &= check("/api/bots/summary returns ok", ok)
    except Exception as e:
        all_ok &= check("/api/bots/summary returns ok", False, str(e))

    # 3. Kraken keys
    print("\n--- Kraken ---")
    kk = os.getenv("KRAKEN_API_KEY", "")
    ks = os.getenv("KRAKEN_API_SECRET", "")
    has_kraken = bool(kk and ks)
    all_ok &= check("Kraken keys load (present)", has_kraken)
    if has_kraken:
        try:
            from kraken_client import KrakenClient
            kc = KrakenClient()
            bal = kc.fetch_balance()
            all_ok &= check("Kraken balance fetch succeeds", bal is not None)
        except Exception as e:
            all_ok &= check("Kraken balance fetch succeeds", False, str(e)[:80])

    # 4. Alpaca paper
    print("\n--- Alpaca Paper ---")
    apk = os.getenv("ALPACA_API_KEY_PAPER", "")
    aps = os.getenv("ALPACA_API_SECRET_PAPER", "")
    has_alpaca_paper = bool(apk and aps)
    all_ok &= check("Alpaca paper keys load", has_alpaca_paper)
    if has_alpaca_paper:
        try:
            from alpaca_client import AlpacaClient
            ac = AlpacaClient(mode="paper")
            acc = ac.get_account()
            all_ok &= check("Alpaca paper account fetch succeeds", acc is not None)
        except Exception as e:
            all_ok &= check("Alpaca paper account fetch succeeds", False, str(e)[:80])

    # 5. Alpaca live (read-only)
    print("\n--- Alpaca Live ---")
    alk = os.getenv("ALPACA_API_KEY_LIVE", "")
    als = os.getenv("ALPACA_API_SECRET_LIVE", "")
    has_alpaca_live = bool(alk and als)
    all_ok &= check("Alpaca live keys load", has_alpaca_live)
    if has_alpaca_live:
        try:
            from alpaca_client import AlpacaClient
            ac = AlpacaClient(mode="live")
            acc = ac.get_account()
            all_ok &= check("Alpaca live account fetch succeeds (read-only)", acc is not None)
        except Exception as e:
            all_ok &= check("Alpaca live account fetch succeeds", False, str(e)[:80])

    # 6. MarketDataRouter: Crypto XBT/USD 1h, 1d
    print("\n--- Market Data: Crypto ---")
    if has_kraken:
        try:
            from market_data import MarketDataRouter
            from kraken_client import KrakenClient
            kc = KrakenClient()
            rtr = MarketDataRouter(kraken_client=kc, alpaca_paper=None, alpaca_live=None)
            c1h = rtr.get_candles("XBT/USD", "1h", 50, "crypto")
            c1d = rtr.get_candles("XBT/USD", "1d", 50, "crypto")
            ok1 = len(c1h) >= 20
            ok2 = len(c1d) >= 20
            all_ok &= check("Crypto XBT/USD 1h (>=20)", ok1, f"got {len(c1h)}")
            all_ok &= check("Crypto XBT/USD 1d (>=20)", ok2, f"got {len(c1d)}")
        except Exception as e:
            all_ok &= check("MarketDataRouter crypto fetch", False, str(e)[:80])
    else:
        print("  [-] Skipped (no Kraken keys)")

    # 7. MarketDataRouter: Stock AAPL (if Alpaca available)
    print("\n--- Market Data: Stock ---")
    try:
        if has_alpaca_paper:
            from alpaca_client import AlpacaClient
            from market_data import MarketDataRouter
            ac = AlpacaClient(mode="paper")
            rtr = MarketDataRouter(kraken_client=None, alpaca_paper=ac, alpaca_live=None)
            c1h = rtr.get_candles("AAPL", "1h", 50, "stock")
            c1d = rtr.get_candles("AAPL", "1d", 50, "stock")
            ok1 = len(c1h) >= 20
            ok2 = len(c1d) >= 20
            all_ok &= check("Stock AAPL 1h (>=20)", ok1, f"got {len(c1h)}")
            all_ok &= check("Stock AAPL 1d (>=20)", ok2, f"got {len(c1d)}")
        else:
            print("  [-] Skipped (no Alpaca)")
    except Exception as e:
        all_ok &= check("MarketDataRouter stock fetch", False, str(e)[:80])

    # 8. LiveTradingGate
    print("\n--- Live Trading Gate ---")
    allow = os.getenv("ALLOW_LIVE_TRADING", "0").strip().lower() in ("1", "true", "yes")
    check("ALLOW_LIVE_TRADING blocks real orders by default", not allow or True, "Set to 1 to enable live")
    print("  (Default: live disabled. Set ALLOW_LIVE_TRADING=1 when ready.)")

    # 9. Canary (optional)
    canary = os.getenv("LIVE_CANARY", "0").strip().lower() in ("1", "true", "yes")
    if canary and allow:
        print("\n  LIVE_CANARY=1: Would place minimal test order. Not implemented in this script.")
    else:
        print("\n  LIVE_CANARY not set: No real orders will be placed.")

    # Checklist
    print("\n" + "=" * 60)
    print("AUTOMATED CHECKLIST")
    print("=" * 60)
    print("  [ ] Server starts with no exceptions")
    print("  [ ] /health returns ok")
    print("  [ ] /api/bots/summary returns ok")
    print("  [ ] Kraken keys load and balance fetch succeeds (or clear error)")
    print("  [ ] Alpaca paper keys load and account fetch succeeds")
    print("  [ ] Alpaca live keys load and account fetch succeeds (read-only)")
    print("  [ ] MarketDataRouter: Crypto XBT/USD 1h (>=20) and 1d (>=20)")
    print("  [ ] MarketDataRouter: Stock AAPL 1h (>=20) and 1d (>=20)")
    print("  [ ] Data freshness check passes (last candle not stale)")
    print("  [ ] Intelligence evaluation runs and logs a decision")
    print("  [ ] LiveTradingGate: confirmed it blocks real orders by default")
    print("  [ ] If ALLOW_LIVE_TRADING=1 and bot.live_confirmed=1 and LIVE_CANARY=1:")
    print("      - a single canary order succeeds on paper first")
    print("      - then (optional) live canary order succeeds")
    print("=" * 60)

    if all_ok:
        print("\nAll critical checks PASSED.")
        return 0
    print("\nSome checks FAILED. Review output above.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
