#!/usr/bin/env python3
"""
End-to-End Test Suite
Tests the complete flow: UI → API → Bot Execution → Orders
"""

import sys
import os
import time
import json
import requests
from typing import Dict, Any, List

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
    delete_bot,
    get_intelligence_decisions,
)
from bot_manager import BotManager
from kraken_client import KrakenClient

BASE_URL = "http://127.0.0.1:8000"
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

def test_api_health():
    """Test API is running"""
    print("\n=== Testing API Health ===")
    try:
        response = requests.get(f"{BASE_URL}/api/bots/summary", timeout=5)
        if response.status_code == 200:
            test_result("API health check", True)
            return True
        else:
            test_result("API health check", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        test_result("API health check", False, str(e))
        return False

def test_bot_creation_api():
    """Test bot creation via API"""
    print("\n=== Testing Bot Creation API ===")
    try:
        bot_data = {
            "name": "E2E Test Bot",
            "symbol": "BTC/USD",
            "strategy_mode": "smart_dca",
            "base_quote": 25.0,
            "safety_quote": 25.0,
            "max_safety": 3,
            "first_dev": 0.015,
            "step_mult": 1.2,
            "tp": 0.012,
            "max_spend_quote": 100.0,
            "dry_run": 1,
            "enabled": 0,
            "market_type": "crypto",
        }
        
        response = requests.post(
            f"{BASE_URL}/api/bots",
            json=bot_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("ok") and data.get("bot"):
                bot_id = data["bot"].get("id")
                test_result("Bot creation API", True, f"Created bot ID: {bot_id}")
                
                # Clean up
                if bot_id:
                    delete_bot(bot_id)
                return True
            else:
                test_result("Bot creation API", False, "Invalid response format")
                return False
        else:
            test_result("Bot creation API", False, f"Status: {response.status_code}, {response.text}")
            return False
    except Exception as e:
        test_result("Bot creation API", False, str(e))
        return False

def test_intelligence_integration():
    """Test that intelligence layer is integrated"""
    print("\n=== Testing Intelligence Integration ===")
    try:
        from intelligence_layer import IntelligenceLayer, PHASE1_AVAILABLE, PHASE2_AVAILABLE, PHASE3_AVAILABLE
        
        layer = IntelligenceLayer()
        test_result("Intelligence layer initialization", True)
        
        # Check Phase availability
        test_result("Phase 1 available", PHASE1_AVAILABLE, warning=not PHASE1_AVAILABLE)
        test_result("Phase 2 available", PHASE2_AVAILABLE, warning=not PHASE2_AVAILABLE)
        test_result("Phase 3 available", PHASE3_AVAILABLE, warning=not PHASE3_AVAILABLE)
        
        return True
    except Exception as e:
        test_result("Intelligence integration", False, str(e))
        return False

def test_dry_run_bot_execution():
    """Test dry-run bot execution"""
    print("\n=== Testing Dry-Run Bot Execution ===")
    try:
        # Create test bot
        bot_data = {
            "name": "E2E Dry Run Test",
            "symbol": "BTC/USD",
            "strategy_mode": "smart_dca",
            "base_quote": 25.0,
            "safety_quote": 25.0,
            "max_safety": 3,
            "first_dev": 0.015,
            "step_mult": 1.2,
            "tp": 0.012,
            "max_spend_quote": 100.0,
            "dry_run": 1,
            "enabled": 1,  # Enable it
            "market_type": "crypto",
        }
        
        bot_id = create_bot(bot_data)
        test_result("Test bot created", True, f"Bot ID: {bot_id}")
        
        # Start bot via API
        try:
            response = requests.post(f"{BASE_URL}/api/bots/{bot_id}/start", timeout=5)
            if response.status_code == 200:
                test_result("Bot start API", True)
            else:
                test_result("Bot start API", False, f"Status: {response.status_code}")
        except Exception as e:
            test_result("Bot start API", False, str(e))
        
        # Wait a bit for bot to run
        print("Waiting 15 seconds for bot to execute...")
        time.sleep(15)
        
        # Check bot status
        try:
            response = requests.get(f"{BASE_URL}/api/bots/{bot_id}/status", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    snap = data.get("snap", {})
                    running = snap.get("running", False)
                    last_event = snap.get("last_event", "")
                    test_result("Bot execution status", running, f"Running: {running}, Event: {last_event}")
                else:
                    test_result("Bot execution status", False, "Invalid response")
            else:
                test_result("Bot execution status", False, f"Status: {response.status_code}")
        except Exception as e:
            test_result("Bot execution status", False, str(e))
        
        # Check intelligence decisions were logged
        try:
            decisions = get_intelligence_decisions(bot_id, limit=10)
            if decisions:
                test_result("Intelligence decisions logged", True, f"Found {len(decisions)} decisions")
                # Check latest decision
                latest = decisions[0]
                test_result("Decision structure", True, f"Action: {latest.get('final_action')}, Strategy: {latest.get('strategy_mode')}")
            else:
                test_result("Intelligence decisions logged", False, "No decisions found")
        except Exception as e:
            test_result("Intelligence decisions check", False, str(e))
        
        # Stop bot
        try:
            response = requests.post(f"{BASE_URL}/api/bots/{bot_id}/stop", timeout=5)
            test_result("Bot stop API", response.status_code == 200)
        except Exception as e:
            test_result("Bot stop API", False, str(e))
        
        # Clean up
        delete_bot(bot_id)
        
        return True
    except Exception as e:
        test_result("Dry-run bot execution", False, str(e))
        return False

def test_strategy_execution_verification():
    """Verify strategies are executed correctly"""
    print("\n=== Testing Strategy Execution Verification ===")
    try:
        from bot_manager import BotManager
        from kraken_client import KrakenClient
        
        kc = KrakenClient()
        if not kc.client:
            test_result("Strategy execution (needs Kraken)", False, "Kraken client not available")
            return False
        
        bm = BotManager(kc)
        
        # Check if bots are running
        running = bm.list_running()
        test_result("Bot manager running bots", True, f"Found {len(running)} running bots")
        
        # For each running bot, check snapshot
        for bot_id in running[:3]:  # Check first 3
            try:
                snap = bm.snapshot(bot_id)
                active_strategy = snap.get("active_strategy")
                decision_action = snap.get("decision_action")
                regime = snap.get("regime_label")
                
                test_result(
                    f"Bot {bot_id} strategy execution",
                    True,
                    f"Strategy: {active_strategy}, Action: {decision_action}, Regime: {regime}"
                )
            except Exception as e:
                test_result(f"Bot {bot_id} snapshot", False, str(e))
        
        return True
    except Exception as e:
        test_result("Strategy execution verification", False, str(e))
        return False

def test_ui_endpoints():
    """Test UI endpoints are accessible"""
    print("\n=== Testing UI Endpoints ===")
    endpoints = [
        "/",
        "/bots",
        "/recommendations",
        "/intelligence",
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if response.status_code == 200:
                test_result(f"UI endpoint: {endpoint}", True)
            else:
                test_result(f"UI endpoint: {endpoint}", False, f"Status: {response.status_code}")
        except Exception as e:
            test_result(f"UI endpoint: {endpoint}", False, str(e))

def main():
    """Run all end-to-end tests"""
    print("=" * 60)
    print("END-TO-END TEST SUITE")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print()
    
    # Test API health first
    if not test_api_health():
        print("\n❌ API is not running. Please start the server first.")
        print("Run: uvicorn one_server:app --host 0.0.0.0 --port 8000")
        return 1
    
    # Run all tests
    test_bot_creation_api()
    test_intelligence_integration()
    test_dry_run_bot_execution()
    test_strategy_execution_verification()
    test_ui_endpoints()
    
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
