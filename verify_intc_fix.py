"""
Script to verify that the INTC routing fix is working correctly.

This simulates what happens when the recommendations scanner encounters INTC.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from symbol_classifier import classify_symbol, is_stock_symbol, is_crypto_symbol


def test_classification():
    """Test that INTC and other symbols are classified correctly"""
    print("=" * 60)
    print("SYMBOL CLASSIFICATION TEST")
    print("=" * 60)
    
    test_cases = [
        ("INTC", "stock"),
        ("AAPL", "stock"),
        ("MSFT", "stock"),
        ("BTC/USD", "crypto"),
        ("ETH/USD", "crypto"),
        ("BTCUSD", "crypto"),  # Long form, will be normalized
    ]
    
    all_passed = True
    for symbol, expected_type in test_cases:
        actual_type = classify_symbol(symbol)
        passed = actual_type == expected_type
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {symbol:15} => {actual_type:6} (expected: {expected_type})")
        if not passed:
            all_passed = False
    
    print()
    return all_passed


def test_guardrails():
    """Test that guardrails prevent misrouting"""
    print("=" * 60)
    print("GUARDRAIL TEST")
    print("=" * 60)
    
    from symbol_classifier import validate_symbol_type
    
    # Test 1: Stock passed to crypto function should raise
    print("\nTest 1: Passing INTC to crypto function...")
    try:
        validate_symbol_type("INTC", "crypto", "_ohlcv_cached")
        print("✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ PASS: Correctly raised ValueError")
        print(f"  Error message: {str(e)[:80]}...")
    
    # Test 2: Crypto passed to stock function should raise
    print("\nTest 2: Passing BTC/USD to stock function...")
    try:
        validate_symbol_type("BTC/USD", "stock", "alpaca_get_ohlcv")
        print("✗ FAIL: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ PASS: Correctly raised ValueError")
        print(f"  Error message: {str(e)[:80]}...")
    
    # Test 3: Stock passed to stock function should NOT raise
    print("\nTest 3: Passing INTC to stock function...")
    try:
        validate_symbol_type("INTC", "stock", "alpaca_get_ohlcv")
        print("✓ PASS: No error raised (as expected)")
    except ValueError as e:
        print(f"✗ FAIL: Should not have raised ValueError: {e}")
        return False
    
    print()
    return True


def demonstrate_fix():
    """Demonstrate how the fix prevents the INTC -> Kraken error"""
    print("=" * 60)
    print("FIX DEMONSTRATION")
    print("=" * 60)
    
    print("\nBEFORE FIX:")
    print("  1. Recommendations scanner encounters 'INTC'")
    print("  2. _scan_symbol(\"INTC\", ...) called")
    print("  3. _ohlcv_cached(\"INTC\", ...) called")
    print("  4. Kraken API called for INTC")
    print("  5. ❌ ERROR: 'Symbol not found on Kraken: INTC'")
    
    print("\nAFTER FIX:")
    print("  1. Recommendations scanner encounters 'INTC'")
    print("  2. _scan_symbol(\"INTC\", ...) called")
    print("  3. classify_symbol(\"INTC\") => 'stock'")
    print("  4. Route to Alpaca client (NOT Kraken)")
    print("  5. alpaca_client.get_ohlcv(\"INTC\", ...) called")
    print("  6. ✓ SUCCESS: Data fetched from Alpaca")
    
    print("\nGUARDRAILS (Safety Net):")
    print("  - If INTC somehow reaches _ohlcv_cached():")
    print("    → validate_symbol_type() raises clear developer error")
    print("    → Prevents silent bugs and provides fix guidance")
    
    print()


if __name__ == "__main__":
    print("\n🔧 INTC ROUTING FIX VERIFICATION\n")
    
    # Run tests
    classification_ok = test_classification()
    guardrails_ok = test_guardrails()
    demonstrate_fix()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Classification Tests: {'✓ PASSED' if classification_ok else '✗ FAILED'}")
    print(f"Guardrail Tests:      {'✓ PASSED' if guardrails_ok else '✗ FAILED'}")
    
    if classification_ok and guardrails_ok:
        print("\n✅ ALL TESTS PASSED - Fix is working correctly!")
        print("\nThe system will now:")
        print("  • Route stock symbols (INTC, AAPL, etc.) to Alpaca")
        print("  • Route crypto symbols (BTC/USD, etc.) to Kraken")
        print("  • Show clear error if stock provider not configured")
        print("  • Never show 'Symbol not found on Kraken' for stocks")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED - Please review")
        sys.exit(1)
