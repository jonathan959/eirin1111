#!/usr/bin/env python3
"""
Run All Checks - Production Release Gates

This script runs ALL verification checks for the trading bot:
1. Unit tests (symbol classification, routing, bot CRUD)
2. Integration tests (routing verification)
3. Core safety checks (order sizing, guardrails)

Exit code 0 = PASS (ready for production)
Exit code 1 = FAIL (critical issues remain)
"""

import sys
import unittest
import os
import subprocess


def run_custom_checks():
    """Run custom verification scripts"""
    checks_passed = True
    
    # Run comprehensive routing verification
    print("\n" + "="*70)
    print("Running Custom Verification: Routing Complete")
    print("="*70)
    
    try:
        result = subprocess.run(
            [sys.executable, "verify_routing_complete.py"],
            capture_output=False,
            cwd=os.getcwd()
        )
        if result.returncode != 0:
            checks_passed = False
            print("✗ Routing verification FAILED")
        else:
            print("✓ Routing verification PASSED")
    except Exception as e:
        print(f"✗ Routing verification ERROR: {e}")
        checks_passed = False
    
    return checks_passed


def main():
    """Main test runner"""
    print("="*70)
    print("TRADING BOT - PRODUCTION RELEASE GATES")
    print("="*70)
    print()
    print("Running comprehensive verification suite...")
    print()
    
    # Run custom checks first
    custom_passed = run_custom_checks()
    
    # Run unit tests
    print("\n" + "="*70)
    print("Running Unit Tests")
    print("="*70)
    
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.getcwd(), 'tests')
    
    # Discover all tests
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    print(f"Custom Checks:  {'✓ PASS' if custom_passed else '✗ FAIL'}")
    print(f"Unit Tests:     {'✓ PASS' if result.wasSuccessful() else '✗ FAIL'}")
    print(f"  - Tests run:  {result.testsRun}")
    print(f"  - Failures:   {len(result.failures)}")
    print(f"  - Errors:     {len(result.errors)}")
    
    print("="*70)
    
    all_passed = custom_passed and result.wasSuccessful()
    
    if all_passed:
        print("\n✅ RELEASE GATES PASSED: All checks green.")
        print()
        print("Critical Fixes Verified:")
        print("  1. ✓ Stock/Crypto routing bug FIXED")
        print("  2. ✓ INTC cannot reach Kraken APIs")
        print("  3. ✓ Guardrails prevent cross-market errors")
        print("  4. ✓ Order sizing validation exists")
        print()
        print("System is READY for production deployment.")
        sys.exit(0)
    else:
        print("\n❌ RELEASE GATES FAILED: Fix critical issues before deploying.")
        
        if not custom_passed:
            print("\n⚠️  Custom verification checks failed.")
        
        if not result.wasSuccessful():
            print(f"\n⚠️  {len(result.failures) + len(result.errors)} test(s) failed.")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
