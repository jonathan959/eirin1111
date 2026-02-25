#!/usr/bin/env python3
"""
Comprehensive Feature Test Suite
Tests all major features of the trading bot system
"""

import sys
import os
import time
import json
from typing import Dict, Any, List, Optional

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Load environment
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
_load_env()

from db import (
    init_db,
    create_bot,
    get_bot,
    list_bots,
    update_bot,
    delete_bot,
    add_log,
    list_logs,
)
from kraken_client import KrakenClient
from alpaca_client import AlpacaClient
from bot_manager import BotManager
from strategies import (
    detect_regime,
    select_strategy,
    smart_decide,
    get_strategy,
    DcaConfig,
    SmartDcaConfig,
)
from intelligence_layer import IntelligenceLayer, IntelligenceContext
from executor import OrderExecutor

# Test results
TEST_RESULTS = {
    "passed": [],
    "failed": [],
    "warnings": [],
}

def test_result(name: str, passed: bool, message: str = "", warning: bool = False):
    """Record test result"""
    if warning:
        TEST_RESULTS["warnings"].append(f"⚠️ {name}: {message}")
        print(f"⚠️ WARNING: {name} - {message}")
    elif passed:
        TEST_RESULTS["passed"].append(name)
        print(f"✅ PASS: {name}")
    else:
        TEST_RESULTS["failed"].append(f"{name}: {message}")
        print(f"❌ FAIL: {name} - {message}")

def test_database():
    """Test database operations"""
    print("\n=== Testing Database ===")
    try:
        init_db()
        test_result("Database initialization", True)
    except Exception as e:
        test_result("Database initialization", False, str(e))
        return False
    
    # Test bot creation
    try:
        bot_data = {
            "name": "Test Bot",
            "symbol": "BTC/USD",
            "strategy_mode": "classic_dca",
            "base_quote": 25.0,
            "safety_quote": 25.0,
            "max_safety": 3,
            "first_dev": 0.015,
            "step_mult": 1.2,
            "tp": 0.012,
            "dry_run": 1,
            "enabled": 0,
            "market_type": "crypto",
        }
        bot_id = create_bot(**bot_data)
        test_result("Bot creation (crypto)", True, f"Created bot ID: {bot_id}")
        
        # Test bot retrieval
        bot = get_bot(bot_id)
        if bot and bot.get("name") == "Test Bot":
            test_result("Bot retrieval", True)
        else:
            test_result("Bot retrieval", False, "Bot data mismatch")
        
        # Test bot update
        update_bot(bot_id, name="Updated Test Bot")
        bot = get_bot(bot_id)
        if bot and bot.get("name") == "Updated Test Bot":
            test_result("Bot update", True)
        else:
            test_result("Bot update", False, "Update failed")
        
        # Test bot deletion
        delete_bot(bot_id)
        bot = get_bot(bot_id)
        if not bot:
            test_result("Bot deletion", True)
        else:
            test_result("Bot deletion", False, "Bot still exists")
            
    except Exception as e:
        test_result("Bot CRUD operations", False, str(e))
    
    # Test stock bot creation
    try:
        stock_bot_data = {
            "name": "Test Stock Bot",
            "symbol": "AAPL",
            "strategy_mode": "smart_dca",
            "base_quote": 50.0,
            "safety_quote": 50.0,
            "max_safety": 3,
            "first_dev": 0.015,
            "step_mult": 1.2,
            "tp": 0.012,
            "dry_run": 1,
            "enabled": 0,
            "market_type": "stocks",
            "alpaca_mode": "paper",
        }
        stock_bot_id = create_bot(**stock_bot_data)
        test_result("Bot creation (stocks)", True, f"Created stock bot ID: {stock_bot_id}")
        delete_bot(stock_bot_id)
    except Exception as e:
        test_result("Bot creation (stocks)", False, str(e))
    
    return True

def test_kraken_client():
    """Test Kraken client"""
    print("\n=== Testing Kraken Client ===")
    try:
        kc = KrakenClient()
        if not kc.client:
            test_result("Kraken client initialization", False, "Client not initialized")
            return False
        test_result("Kraken client initialization", True)
        
        # Test market fetch
        try:
            markets = kc.fetch_markets()
            if markets and len(markets) > 0:
                test_result("Kraken fetch markets", True, f"Found {len(markets)} markets")
            else:
                test_result("Kraken fetch markets", False, "No markets returned")
        except Exception as e:
            test_result("Kraken fetch markets", False, str(e))
        
        # Test ticker fetch
        try:
            ticker = kc.fetch_ticker("BTC/USD")
            if ticker and "last" in ticker:
                test_result("Kraken fetch ticker", True, f"Price: {ticker.get('last')}")
            else:
                test_result("Kraken fetch ticker", False, "Invalid ticker data")
        except Exception as e:
            test_result("Kraken fetch ticker", False, str(e))
            
    except Exception as e:
        test_result("Kraken client", False, str(e))
        return False
    
    return True

def test_alpaca_client():
    """Test Alpaca client"""
    print("\n=== Testing Alpaca Client ===")
    try:
        # Test paper client
        alpaca_paper = AlpacaClient(paper=True)
        if not alpaca_paper.client:
            test_result("Alpaca paper client initialization", False, "Client not initialized")
            return False
        test_result("Alpaca paper client initialization", True)
        
        # Test live client
        try:
            alpaca_live = AlpacaClient(paper=False)
            if alpaca_live.client:
                test_result("Alpaca live client initialization", True)
            else:
                test_result("Alpaca live client initialization", False, "Client not initialized")
        except Exception as e:
            test_result("Alpaca live client initialization", False, str(e), warning=True)
        
        # Test symbol search
        try:
            symbols = alpaca_paper.search_assets(status="active", asset_class="us_equity")
            if symbols and len(symbols) > 0:
                test_result("Alpaca symbol search", True, f"Found {len(symbols)} symbols")
            else:
                test_result("Alpaca symbol search", False, "No symbols returned")
        except Exception as e:
            test_result("Alpaca symbol search", False, str(e))
            
    except Exception as e:
        test_result("Alpaca client", False, str(e))
        return False
    
    return True

def test_strategies():
    """Test strategy implementations"""
    print("\n=== Testing Strategies ===")
    
    # Test regime detection
    try:
        # Mock candles
        candles = [[time.time() - 3600 * i, 100, 102, 99, 101, 1000] for i in range(100, 0, -1)]
        regime = detect_regime(candles)
        if regime:
            test_result("Regime detection", True, f"Detected: {regime}")
        else:
            test_result("Regime detection", False, "No regime detected")
    except Exception as e:
        test_result("Regime detection", False, str(e))
    
    # Test strategy selection
    try:
        candles_1h = [[time.time() - 3600 * i, 100, 102, 99, 101, 1000] for i in range(50, 0, -1)]
        candles_4h = [[time.time() - 14400 * i, 100, 103, 98, 102, 4000] for i in range(50, 0, -1)]
        candles_1d = [[time.time() - 86400 * i, 100, 105, 97, 104, 10000] for i in range(50, 0, -1)]
        
        strategy = select_strategy(candles_1h, candles_4h, candles_1d)
        if strategy:
            test_result("Strategy selection", True, f"Selected: {strategy}")
        else:
            test_result("Strategy selection", False, "No strategy selected")
    except Exception as e:
        test_result("Strategy selection", False, str(e))
    
    # Test DCA config
    try:
        config = DcaConfig(
            base_quote=25.0,
            safety_quote=25.0,
            max_safety=3,
            first_dev=0.015,
            step_mult=1.2,
            tp=0.012,
        )
        if config.base_quote == 25.0:
            test_result("DCA config creation", True)
        else:
            test_result("DCA config creation", False, "Config mismatch")
    except Exception as e:
        test_result("DCA config creation", False, str(e))
    
    # Test Smart DCA
    try:
        smart_config = SmartDcaConfig(
            base_quote=25.0,
            safety_quote=25.0,
            max_safety=3,
            first_dev=0.015,
            step_mult=1.2,
            tp=0.012,
            vol_gap_mult=1.0,
            tp_vol_mult=1.0,
        )
        if smart_config.base_quote == 25.0:
            test_result("Smart DCA config creation", True)
        else:
            test_result("Smart DCA config creation", False, "Config mismatch")
    except Exception as e:
        test_result("Smart DCA config creation", False, str(e))
    
    return True

def test_intelligence_layer():
    """Test intelligence layer"""
    print("\n=== Testing Intelligence Layer ===")
    try:
        layer = IntelligenceLayer()
        test_result("Intelligence layer initialization", True)
        
        # Test context building (would need real data)
        # This is a basic test - full testing would require market data
        test_result("Intelligence layer", True, "Basic initialization passed")
        
    except Exception as e:
        test_result("Intelligence layer", False, str(e))
        return False
    
    return True

def test_bot_manager():
    """Test BotManager"""
    print("\n=== Testing Bot Manager ===")
    try:
        kc = KrakenClient()
        if not kc.client:
            test_result("BotManager initialization (needs Kraken)", False, "Kraken client not available")
            return False
        
        bm = BotManager(kc)
        test_result("BotManager initialization", True)
        
        # Test bot listing
        bots = bm.list_running()
        test_result("BotManager list running", True, f"Found {len(bots)} running bots")
        
    except Exception as e:
        test_result("BotManager", False, str(e))
        return False
    
    return True

def test_order_executor():
    """Test OrderExecutor"""
    print("\n=== Testing Order Executor ===")
    try:
        kc = KrakenClient()
        if not kc.client:
            test_result("OrderExecutor initialization (needs Kraken)", False, "Kraken client not available")
            return False
        
        executor = OrderExecutor(kc)
        test_result("OrderExecutor initialization", True)
        
        # Note: Actual order execution would require live trading or dry-run mode
        test_result("OrderExecutor", True, "Initialization passed (order execution requires dry-run/live mode)")
        
    except Exception as e:
        test_result("OrderExecutor", False, str(e))
        return False
    
    return True

def test_phase1_intelligence():
    """Test Phase 1 intelligence features"""
    print("\n=== Testing Phase 1 Intelligence ===")
    try:
        from phase1_intelligence import Phase1Intelligence
        
        phase1 = Phase1Intelligence()
        test_result("Phase 1 Intelligence initialization", True)
        
        # Test trailing stop
        from phase1_intelligence import TrailingStopLoss
        trailing = TrailingStopLoss()
        test_result("Trailing stop loss", True)
        
        # Test cooldown manager
        from phase1_intelligence import CooldownManager
        cooldown = CooldownManager()
        test_result("Cooldown manager", True)
        
    except ImportError as e:
        test_result("Phase 1 Intelligence", False, f"Import error: {e}")
    except Exception as e:
        test_result("Phase 1 Intelligence", False, str(e))
    
    return True

def test_phase2_intelligence():
    """Test Phase 2 intelligence features"""
    print("\n=== Testing Phase 2 Intelligence ===")
    try:
        from multi_timeframe import MultiTimeframeAnalyzer
        analyzer = MultiTimeframeAnalyzer()
        test_result("Multi-timeframe analyzer", True)
        
        from kelly_criterion import KellyPositionSizer
        kelly = KellyPositionSizer()
        test_result("Kelly position sizer", True)
        
        from sentiment_analyzer import SentimentAnalyzer
        sentiment = SentimentAnalyzer()
        test_result("Sentiment analyzer", True)
        
        from portfolio_correlation import PortfolioCorrelationAnalyzer
        correlation = PortfolioCorrelationAnalyzer()
        test_result("Portfolio correlation analyzer", True)
        
        from order_book_analyzer import OrderBookAnalyzer
        orderbook = OrderBookAnalyzer()
        test_result("Order book analyzer", True)
        
    except ImportError as e:
        test_result("Phase 2 Intelligence", False, f"Import error: {e}")
    except Exception as e:
        test_result("Phase 2 Intelligence", False, str(e))
    
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("COMPREHENSIVE FEATURE TEST SUITE")
    print("=" * 60)
    
    # Run all tests
    test_database()
    test_kraken_client()
    test_alpaca_client()
    test_strategies()
    test_intelligence_layer()
    test_bot_manager()
    test_order_executor()
    test_phase1_intelligence()
    test_phase2_intelligence()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {len(TEST_RESULTS['passed'])}")
    print(f"❌ Failed: {len(TEST_RESULTS['failed'])}")
    print(f"⚠️  Warnings: {len(TEST_RESULTS['warnings'])}")
    
    if TEST_RESULTS['failed']:
        print("\n❌ FAILED TESTS:")
        for fail in TEST_RESULTS['failed']:
            print(f"  - {fail}")
    
    if TEST_RESULTS['warnings']:
        print("\n⚠️  WARNINGS:")
        for warn in TEST_RESULTS['warnings']:
            print(f"  - {warn}")
    
    print("\n" + "=" * 60)
    
    # Return exit code
    if TEST_RESULTS['failed']:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
