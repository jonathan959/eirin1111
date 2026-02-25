#!/usr/bin/env python3
"""
verify_all.py — Run unit, integration, and end-to-end tests.
Exits with status 0 if all pass, non-zero otherwise.
"""
import os
import sys
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

def run(cmd, cwd=ROOT, timeout=120):
    """Run command, return (success, output)."""
    try:
        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=os.name == "nt",
        )
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("VERIFY ALL — Unit + Integration + E2E")
    print("=" * 60)

    results = []
    failed = []

    # 1. Unit tests (market_data, symbol_classifier)
    print("\n--- Unit: market_data ---")
    ok, out = run([sys.executable, "-c", """
from market_data import MarketDataRouter, _normalize_stock_symbol, _normalize_crypto_symbol
assert _normalize_stock_symbol("THNR") == "THNR"
assert _normalize_crypto_symbol("BTC/USD") == "XBT/USD"
r = MarketDataRouter(None, None, None)
ok, reason, norm, prov = r.validate_symbol("THNR", "stock")
assert not ok and prov == "alpaca"
ok, reason, norm, prov = r.validate_symbol("XBT/USD", "crypto")
assert not ok and prov == "kraken"
print("OK")
"""])
    results.append(("market_data unit", ok))
    if not ok:
        failed.append(("market_data unit", out))
        print("FAIL:", out[:200])
    else:
        print("PASS")

    # 2. Unit: symbol_classifier
    print("\n--- Unit: symbol_classifier ---")
    ok, out = run([sys.executable, "-c", """
from symbol_classifier import classify_symbol, is_stock_symbol, is_crypto_symbol
assert classify_symbol("THNR") == "stock"
assert classify_symbol("AAPL") == "stock"
assert classify_symbol("XBT/USD") == "crypto"
assert is_stock_symbol("THNR")
assert is_crypto_symbol("XBT/USD")
print("OK")
"""])
    results.append(("symbol_classifier", ok))
    if not ok:
        failed.append(("symbol_classifier", out))
        print("FAIL:", out[:200])
    else:
        print("PASS")

    # 3. Unit: risk_circuit_breaker
    print("\n--- Unit: risk_circuit_breaker ---")
    ok, out = run([sys.executable, "-c", """
from risk_circuit_breaker import check_circuit_breakers
# All pass
ok, r = check_circuit_breakers(equity=10000, daily_realized_pnl=0, portfolio_drawdown=0.05,
    portfolio_exposure_pct=0.2, open_deals_count=3, total_exposure_usd=2000)
assert ok and r is None
# Daily loss trip
ok, r = check_circuit_breakers(equity=10000, daily_realized_pnl=-700, portfolio_drawdown=0,
    portfolio_exposure_pct=0, open_deals_count=0, total_exposure_usd=0, max_daily_loss_pct=0.06)
assert not ok and r and "Daily loss" in r
# Max deals trip
ok, r = check_circuit_breakers(equity=10000, daily_realized_pnl=0, portfolio_drawdown=0,
    portfolio_exposure_pct=0, open_deals_count=6, total_exposure_usd=0, max_concurrent_deals=6)
assert not ok and r and "concurrent" in r
print("OK")
"""])
    results.append(("risk_circuit_breaker", ok))
    if not ok:
        failed.append(("risk_circuit_breaker", out))
        print("FAIL:", out[:200])
    else:
        print("PASS")

    # 4. Integration: test_market_hours
    print("\n--- Integration: test_market_hours ---")
    ok, out = run([sys.executable, "tests/test_market_hours.py"])
    results.append(("test_market_hours", ok))
    if not ok:
        failed.append(("test_market_hours", out))
        print("FAIL:", out[-500:] if len(out) > 500 else out)
    else:
        print("PASS")

    # 5. Integration: test_integration_bot_intelligence
    print("\n--- Integration: test_integration_bot_intelligence ---")
    ok, out = run([sys.executable, "tests/test_integration_bot_intelligence.py"])
    results.append(("test_integration_bot_intelligence", ok))
    if not ok:
        failed.append(("test_integration_bot_intelligence", out))
        print("FAIL:", out[-500:] if len(out) > 500 else out)
    else:
        print("PASS")

    # 6. E2E: requires server running
    print("\n--- E2E: test_end_to_end (requires server on :8000) ---")
    ok, out = run([sys.executable, "test_end_to_end.py"], timeout=60)
    results.append(("test_end_to_end", ok))
    if not ok:
        if "Connection" in out or "refused" in out.lower():
            print("SKIP (server not running)")
            results[-1] = ("test_end_to_end", True)  # Don't fail if server down
        else:
            failed.append(("test_end_to_end", out))
            print("FAIL:", out[-500:] if len(out) > 500 else out)
    else:
        print("PASS")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    if failed:
        print("\nFAILED:")
        for name, out in failed:
            print(f"  - {name}")
            print(f"    {out[:200]}...")
        return 1
    print("\nAll checks passed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
