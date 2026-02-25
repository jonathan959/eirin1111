#!/usr/bin/env python3
"""
Test Alpaca API Connection
Run this to verify your Alpaca keys are working correctly.
"""

import os
import sys
from alpaca_client import AlpacaClient

def test_alpaca_paper():
    """Test paper trading account"""
    print("\n" + "="*60)
    print("Testing PAPER TRADING Account")
    print("="*60)
    
    try:
        client = AlpacaClient(mode="paper")
        print(f"[OK] Paper trading client initialized")
        print(f"   Base URL: {client.base_url}")
        
        # Test account info
        account = client.get_account()
        print(f"\n[ACCOUNT] Account Info:")
        print(f"   Status: {account.get('status')}")
        print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"   Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"   Pattern Day Trader: {account.get('pattern_day_trader')}")
        
        # Test market status
        clock = client.get_clock()
        print(f"\n[MARKET] Market Status:")
        print(f"   Is Open: {clock.get('is_open')}")
        print(f"   Timestamp: {clock.get('timestamp')}")
        print(f"   Next Open: {clock.get('next_open')}")
        print(f"   Next Close: {clock.get('next_close')}")
        
        # Test ticker data
        print(f"\n[DATA] Testing Market Data (AAPL):")
        try:
            ticker = client.get_ticker("AAPL")
            if ticker:
                print(f"   Ticker retrieved successfully")
                print(f"   Data: {str(ticker)[:100]}...")
        except Exception as e:
            print(f"   Ticker test skipped (market closed or data unavailable): {e}")
        
        # Test asset search
        print(f"\n[SEARCH] Testing Asset Search:")
        try:
            assets = client.search_assets("TSLA")
            if assets:
                asset = assets[0]
                print(f"   Symbol: {asset.get('symbol')}")
                print(f"   Name: {asset.get('name')}")
                print(f"   Exchange: {asset.get('exchange')}")
                print(f"   Tradable: {asset.get('tradable')}")
                print(f"   Fractionable: {asset.get('fractionable')}")
        except Exception as e:
            print(f"   Search test skipped: {e}")
        
        print(f"\n[OK] Paper trading connection successful!")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Paper trading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alpaca_live():
    """Test live trading account"""
    print("\n" + "="*60)
    print("Testing LIVE TRADING Account (Read-Only)")
    print("="*60)
    
    try:
        client = AlpacaClient(mode="live")
        print(f"[OK] Live trading client initialized")
        print(f"   Base URL: {client.base_url}")
        
        # Test account info (read-only)
        account = client.get_account()
        print(f"\n[ACCOUNT] Account Info:")
        print(f"   Status: {account.get('status')}")
        print(f"   Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"   Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"   Portfolio Value: ${float(account.get('portfolio_value', 0)):,.2f}")
        print(f"   Pattern Day Trader: {account.get('pattern_day_trader')}")
        
        print(f"\n[OK] Live trading connection successful!")
        print(f"[WARNING] Live trading is available but NOT enabled by default.")
        print(f"   Always start with paper trading to test your strategies.")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Live trading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Fix Windows console encoding
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    # Load .env file
    try:
        from pathlib import Path
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip()
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
    
    print("\n[ALPACA] API Connection Test")
    print("="*60)
    
    # Test paper trading
    paper_ok = test_alpaca_paper()
    
    # Test live trading
    live_ok = test_alpaca_live()
    
    # Summary
    print("\n" + "="*60)
    print("[SUMMARY] Test Summary")
    print("="*60)
    print(f"   Paper Trading: {'[OK] PASS' if paper_ok else '[FAIL] FAIL'}")
    print(f"   Live Trading:  {'[OK] PASS' if live_ok else '[FAIL] FAIL'}")
    
    if paper_ok and live_ok:
        print(f"\n[SUCCESS] All tests passed! Your Alpaca integration is ready.")
        print(f"\n[NEXT] Next Steps:")
        print(f"   1. Deploy to your EC2 server")
        print(f"   2. Create your first stock bot in the UI")
        print(f"   3. Start with paper trading to test")
        print(f"   4. Switch to live when ready")
    else:
        print(f"\n[WARNING] Some tests failed. Check your .env file and API keys.")
    
    sys.exit(0 if (paper_ok and live_ok) else 1)
