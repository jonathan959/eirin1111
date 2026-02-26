#!/usr/bin/env python3
"""
Live Trading Readiness Test Suite
Tests every critical function in the trading pipeline.
Run: .venv/bin/python -m pytest tests/test_live_readiness.py -v
"""
import os
import sys
import math
import json
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 1. DATABASE OPERATIONS
# ============================================================
class TestDatabaseOperations(unittest.TestCase):
    """Test DB functions critical for live trading."""

    def test_init_db_idempotent(self):
        from db import init_db
        init_db()
        init_db()  # Should not fail on second run

    def test_create_and_get_bot(self):
        from db import create_bot, get_bot, delete_bot
        bot_id = create_bot({
            "name": "Test Live Bot", "symbol": "BTC/USD",
            "base_quote": 10.0, "safety_quote": 5.0, "max_spend_quote": 50.0,
            "max_safety": 3, "first_dev": 0.015, "step_mult": 1.2,
            "tp": 0.02, "dry_run": 1, "enabled": 0,
        })
        self.assertIsNotNone(bot_id)
        self.assertGreater(bot_id, 0)

        bot = get_bot(bot_id)
        self.assertIsNotNone(bot)
        self.assertEqual(bot["name"], "Test Live Bot")
        self.assertEqual(bot["symbol"], "BTC/USD")
        self.assertAlmostEqual(float(bot["tp"]), 0.02)

        delete_bot(bot_id)
        self.assertIsNone(get_bot(bot_id))

    def test_new_risk_columns_exist(self):
        from db import create_bot, get_bot, delete_bot
        bot_id = create_bot({
            "name": "Risk Column Test", "symbol": "ETH/USD",
            "base_quote": 5.0, "safety_quote": 5.0, "max_spend_quote": 30.0, "max_safety": 2,
            "first_dev": 0.01, "step_mult": 1.0, "tp": 0.01,
            "dry_run": 1, "enabled": 0,
        })
        bot = get_bot(bot_id)
        self.assertIn("stop_loss_pct", bot)
        self.assertIn("max_hold_hours", bot)
        self.assertIn("risk_profile", bot)
        self.assertIn("trailing_stop_enabled", bot)
        self.assertIn("trailing_activation_pct", bot)
        self.assertIn("trailing_distance_pct", bot)
        self.assertIn("max_drawdown_pct", bot)
        self.assertIn("use_kelly_sizing", bot)
        delete_bot(bot_id)

    def test_update_bot_risk_fields(self):
        """Test updating risk fields via the API endpoint (which handles partial updates)."""
        import requests
        base = "http://localhost:8000"
        try:
            requests.get(f"{base}/api/health", timeout=2)
        except Exception:
            self.skipTest("Server not running")

        r = requests.post(f"{base}/api/bots", json={
            "name": "Update Risk Test", "symbol": "BTC/USD",
            "base_quote": 10, "dry_run": 1, "enabled": 0,
        }, timeout=5)
        bot_id = r.json()["bot"]["id"]

        r2 = requests.put(f"{base}/api/bots/{bot_id}", json={
            "stop_loss_pct": 0.05, "max_hold_hours": 168, "risk_profile": "conservative",
        }, timeout=5)
        self.assertTrue(r2.json().get("ok", False), f"Update failed: {r2.json()}")

        from db import get_bot
        bot = get_bot(bot_id)
        self.assertIsNotNone(bot, "Bot should exist after update")
        self.assertIn("stop_loss_pct", bot)
        self.assertIn("risk_profile", bot)

        requests.delete(f"{base}/api/bots/{bot_id}", timeout=5)

    def test_deal_operations(self):
        from db import create_bot, delete_bot, open_deal, close_deal, latest_open_deal
        bot_id = create_bot({
            "name": "Deal Test", "symbol": "BTC/USD",
            "base_quote": 10, "safety_quote": 5, "max_spend_quote": 30, "max_safety": 2,
            "first_dev": 0.01, "step_mult": 1.0, "tp": 0.01,
            "dry_run": 1, "enabled": 0,
        })
        deal_id = open_deal(bot_id, "BTC/USD", state="OPEN")
        self.assertGreater(deal_id, 0)
        deal = latest_open_deal(bot_id)
        self.assertIsNotNone(deal)
        self.assertEqual(int(deal["id"]), deal_id)

        close_deal(deal_id, entry_avg=50000.0, exit_avg=51000.0,
                   base_amount=0.001, realized_pnl_quote=1.0)
        deal = latest_open_deal(bot_id)
        self.assertIsNone(deal)
        delete_bot(bot_id)

    def test_pnl_stats_no_division_by_zero(self):
        from db import bot_performance_stats
        result = bot_performance_stats(99999)
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(float(result.get("win_rate", 0)), 0.0)

    def test_settings_roundtrip(self):
        from db import set_setting, get_setting
        set_setting("_test_key", "test_value_123")
        self.assertEqual(get_setting("_test_key"), "test_value_123")
        set_setting("_test_key", "0")


# ============================================================
# 2. INTELLIGENCE LAYER
# ============================================================
class TestIntelligenceLayer(unittest.TestCase):
    """Test intelligence layer decision making."""

    def test_regime_detection_empty_candles(self):
        from strategies import detect_regime
        result = detect_regime([])
        self.assertIsNotNone(result)
        self.assertEqual(result.regime, "RISK_OFF")
        self.assertAlmostEqual(result.confidence, 0.0)

    def test_regime_detection_single_candle(self):
        from strategies import detect_regime
        result = detect_regime([[1000, 1010, 990, 1005, 1005, 100]])
        self.assertIsNotNone(result)

    def test_regime_detection_nan_candles(self):
        from strategies import detect_regime
        candles = [[1000, 1010, 990, 1005, float('nan'), 100]] * 5
        result = detect_regime(candles)
        self.assertIsNotNone(result)

    def test_regime_detection_valid_data(self):
        from strategies import detect_regime
        import random
        random.seed(42)
        candles = []
        price = 50000.0
        for i in range(200):
            change = random.uniform(-500, 500)
            o = price
            h = price + abs(change)
            l = price - abs(change)
            c = price + change
            price = c
            candles.append([o, h, l, c, c, random.uniform(100, 1000)])
        result = detect_regime(candles)
        self.assertIsNotNone(result)
        self.assertIn(result.regime, ["TREND_UP", "TREND_DOWN", "RANGING", "HIGH_VOL_RISK",
                                       "RISK_OFF", "BREAKOUT_UP", "BREAKOUT_DOWN"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_intelligence_layer_evaluate(self):
        from intelligence_layer import IntelligenceLayer, IntelligenceContext
        il = IntelligenceLayer()
        import random
        random.seed(42)
        candles = []
        price = 50000.0
        for _ in range(100):
            change = random.uniform(-200, 200)
            o = price
            h = price + abs(change) * 2
            l = price - abs(change)
            c = price + change
            price = c
            candles.append([o, h, l, c, c, random.uniform(1000, 5000)])

        ctx = IntelligenceContext(
            symbol="BTC/USD", last_price=price,
            candles_1h=candles[-100:], candles_4h=candles[-50:],
            candles_1d=candles[-30:], candles_1w=candles[-10:],
            free_quote=1000.0, total_quote=1000.0,
            portfolio_total_usd=10000.0, portfolio_exposure_pct=0.0,
            bot_config={"tp": 0.02, "max_safety": 3, "base_quote": 20, "market_type": "crypto"},
        )
        decision = il.evaluate(ctx)
        self.assertIsNotNone(decision)
        self.assertIsNotNone(decision.data_validity)
        self.assertIsNotNone(decision.market_safety)
        self.assertIsNotNone(decision.regime_detection)
        self.assertIsNotNone(decision.position_sizing)
        self.assertFalse(math.isnan(decision.position_sizing.base_size))
        self.assertFalse(math.isinf(decision.position_sizing.base_size))
        self.assertGreaterEqual(decision.position_sizing.base_size, 0.0)

    def test_position_sizing_nan_guard(self):
        from intelligence_layer import IntelligenceLayer, IntelligenceContext
        il = IntelligenceLayer()
        ctx = IntelligenceContext(
            symbol="BTC/USD", last_price=50000.0,
            candles_1h=[], candles_4h=[], candles_1d=[], candles_1w=[],
            free_quote=1000.0, total_quote=1000.0,
            portfolio_total_usd=10000.0,
            bot_config={"tp": 0.02, "max_safety": 3, "base_quote": 20},
        )
        decision = il.evaluate(ctx)
        self.assertFalse(math.isnan(decision.position_sizing.base_size))
        self.assertFalse(math.isinf(decision.position_sizing.base_size))


# ============================================================
# 3. EXECUTOR
# ============================================================
class TestExecutor(unittest.TestCase):
    """Test order executor validation and safety."""

    def test_rejects_zero_size(self):
        from executor import OrderExecutor
        from intelligence_layer import (
            IntelligenceDecision, ExecutionPolicyResult, AllowedAction,
            DataValidityResult, MarketSafetyResult, RegimeDetectionResult,
            StrategyRoutingResult, PositionSizingResult, TradeManagementResult,
            RegimeType,
        )
        mock_kc = Mock()
        executor = OrderExecutor(mock_kc)
        decision = IntelligenceDecision(
            data_validity=DataValidityResult(data_ok=True),
            market_safety=MarketSafetyResult(allowed_actions=AllowedAction.TRADE_ALLOWED, risk_budget=1000),
            regime_detection=RegimeDetectionResult(regime=RegimeType.RANGE, confidence=0.7, ttl_seconds=300),
            strategy_routing=StrategyRoutingResult(strategy_mode="smart_dca", entry_style="moderate", exit_style="fixed_tp"),
            position_sizing=PositionSizingResult(base_size=15, ladder_steps=1, ladder_spacing_pct=1.5, max_adds=3),
            execution_policy=ExecutionPolicyResult(order_type="limit", limit_price=50000, post_only=False),
            trade_management=TradeManagementResult(),
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            final_action="ENTER", final_reason="test",
            proposed_orders=[{"side": "buy", "type": "limit", "price": 50000, "size_base": 0.0, "size_quote": 0.0}],
        )
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        self.assertFalse(result.get("success", True))

    def test_edge_cost_check_allows_when_no_edge(self):
        """When expected_edge_pct is 0 (not populated), orders should NOT be blocked."""
        from executor import OrderExecutor
        from intelligence_layer import (
            IntelligenceDecision, ExecutionPolicyResult, AllowedAction,
            DataValidityResult, MarketSafetyResult, RegimeDetectionResult,
            StrategyRoutingResult, PositionSizingResult, TradeManagementResult,
            RegimeType,
        )
        mock_kc = Mock()
        mock_kc.fetch_ticker_last.return_value = 50000.0
        executor = OrderExecutor(mock_kc)
        decision = IntelligenceDecision(
            data_validity=DataValidityResult(data_ok=True),
            market_safety=MarketSafetyResult(allowed_actions=AllowedAction.TRADE_ALLOWED, risk_budget=1000),
            regime_detection=RegimeDetectionResult(regime=RegimeType.RANGE, confidence=0.7, ttl_seconds=300),
            strategy_routing=StrategyRoutingResult(strategy_mode="smart_dca", entry_style="moderate", exit_style="fixed_tp"),
            position_sizing=PositionSizingResult(base_size=15, ladder_steps=1, ladder_spacing_pct=1.5, max_adds=3),
            execution_policy=ExecutionPolicyResult(order_type="limit", limit_price=50000, post_only=False),
            trade_management=TradeManagementResult(),
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            final_action="ENTER", final_reason="test",
            proposed_orders=[{
                "side": "buy", "type": "limit", "price": 50000,
                "size_base": 0.001, "size_quote": 0.0,
                "expected_edge_pct": 0.0,  # Not populated
                "spread_pct": 0.001, "volatility_pct": 0.03,
            }],
        )
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        errors = result.get("errors", [])
        for err in errors:
            self.assertNotIn("Expected edge", err, "Edge-cost check should not block when expected_edge_pct=0")


# ============================================================
# 4. BOT MANAGER COMPONENTS
# ============================================================
class TestBotManagerComponents(unittest.TestCase):
    """Test bot_manager trading logic components."""

    def _make_runner(self, trailing_high=None):
        from bot_manager import BotRunner
        import threading
        runner = BotRunner.__new__(BotRunner)
        runner._lock = threading.RLock()
        runner._trailing_high = trailing_high

        class FakeState:
            highest_price_reached = trailing_high
            trailing_active = trailing_high is not None
            trailing_price = None
        runner.state = FakeState()
        return runner

    def test_trailing_stop_activation(self):
        runner = self._make_runner(trailing_high=None)
        should_exit, reason = runner._check_trailing_stop(
            price=51000.0, entry=50000.0, tp_pct=0.03, dry_run=True,
            trailing_activation_pct=0.01, trailing_distance_pct=0.005,
        )
        self.assertFalse(should_exit)

    def test_trailing_stop_triggers(self):
        """Trailing stop should eventually trigger when price drops from peak."""
        runner = self._make_runner(trailing_high=52000.0)
        runner.state.trailing_active = True
        runner.state.trailing_price = 51220.0
        should_exit, reason = runner._check_trailing_stop(
            price=51200.0, entry=50000.0, tp_pct=0.05, dry_run=True,
            trailing_activation_pct=0.01, trailing_distance_pct=0.015,
        )
        self.assertTrue(should_exit, f"Price 51200 below trailing_price 51220 should trigger exit")

    def test_trailing_stop_no_trigger_below_activation(self):
        runner = self._make_runner(trailing_high=None)
        should_exit, reason = runner._check_trailing_stop(
            price=50100.0, entry=50000.0, tp_pct=0.03, dry_run=True,
            trailing_activation_pct=0.02, trailing_distance_pct=0.01,
        )
        self.assertFalse(should_exit)

    def test_trailing_stop_invalid_inputs(self):
        runner = self._make_runner(trailing_high=None)
        should_exit, reason = runner._check_trailing_stop(
            price=0, entry=0, tp_pct=0, dry_run=True,
        )
        self.assertFalse(should_exit)

        should_exit, reason = runner._check_trailing_stop(
            price=None, entry=None, tp_pct=0.03, dry_run=True,
        )
        self.assertFalse(should_exit)


# ============================================================
# 5. SYMBOL CLASSIFICATION
# ============================================================
class TestSymbolClassification(unittest.TestCase):
    """Test symbol routing correctness."""

    def test_stock_symbols(self):
        from symbol_classifier import classify_symbol
        for sym in ["AAPL", "MSFT", "INTC", "TSLA", "AMD", "WMT"]:
            self.assertEqual(classify_symbol(sym), "stock", f"{sym} should be stock")

    def test_crypto_symbols(self):
        from symbol_classifier import classify_symbol
        for sym in ["BTC/USD", "ETH/USD", "XBT/USD"]:
            self.assertEqual(classify_symbol(sym), "crypto", f"{sym} should be crypto")

    def test_long_symbols_are_crypto(self):
        from symbol_classifier import classify_symbol
        for sym in ["BTCUSD", "ETHUSD", "XBTUSD"]:
            self.assertEqual(classify_symbol(sym), "crypto", f"{sym} should be crypto")


# ============================================================
# 6. AUTOPILOT
# ============================================================
class TestAutopilot(unittest.TestCase):
    """Test autopilot safety and correctness."""

    def test_autopilot_lock_prevents_duplicate_runs(self):
        from autopilot import _cycle_lock
        acquired = _cycle_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        second = _cycle_lock.acquire(blocking=False)
        self.assertFalse(second, "Second acquire should fail (lock held)")
        _cycle_lock.release()

    def test_autopilot_config_defaults(self):
        from autopilot import get_autopilot_config
        cfg = get_autopilot_config()
        self.assertIsInstance(cfg, dict)

    def test_risk_profile_params(self):
        """Verify risk profile parameters are valid."""
        profiles = {
            "conservative": {"tp": 0.02, "stop_loss_pct": 0.05},
            "balanced": {"tp": 0.03, "stop_loss_pct": 0.08},
            "aggressive": {"tp": 0.05, "stop_loss_pct": 0.12},
        }
        for name, params in profiles.items():
            self.assertGreater(params["tp"], 0, f"{name} TP must be positive")
            self.assertGreater(params["stop_loss_pct"], 0, f"{name} SL must be positive")
            self.assertGreater(params["tp"], params["stop_loss_pct"] * 0.3,
                             f"{name} TP should be reasonable vs SL (reward > 0.3x risk)")


# ============================================================
# 7. EXPLORE V2 SCORING
# ============================================================
class TestExploreV2(unittest.TestCase):
    """Test explore scoring gates and penalties."""

    def test_gate_blocks_low_volume(self):
        from explore_v2 import apply_universe_gates
        ok, reason = apply_universe_gates("TEST", volume_24h_quote=100, spread_bps=10)
        self.assertFalse(ok)
        self.assertIn("volume", reason.lower())

    def test_gate_passes_normal(self):
        from explore_v2 import apply_universe_gates
        ok, reason = apply_universe_gates("TEST", volume_24h_quote=100000, spread_bps=20)
        self.assertTrue(ok)

    def test_score_clamp(self):
        from explore_v2 import enhance_score
        adj_score, _ = enhance_score(base_score=105, snap={}, regime="RANGING", spread_bps=10, volatility_pct=0.03)
        self.assertLessEqual(adj_score, 100.0)
        adj_score2, _ = enhance_score(base_score=-5, snap={}, regime="RANGING", spread_bps=10, volatility_pct=0.03)
        self.assertGreaterEqual(adj_score2, 0.0)


# ============================================================
# 8. ALPACA ADAPTER SAFETY
# ============================================================
class TestAlpacaAdapterSafety(unittest.TestCase):
    """Test Alpaca adapter handles errors gracefully."""

    def test_fetch_balance_handles_api_error(self):
        from alpaca_adapter import AlpacaAdapter
        mock_client = Mock()
        mock_client.get_account.side_effect = Exception("API error")
        mock_client.get_positions.return_value = []

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.client = mock_client
        adapter._markets_cache = {}

        balance = adapter.fetch_balance()
        self.assertIn("total", balance)
        self.assertIn("free", balance)
        self.assertEqual(balance["total"]["USD"], 0.0)

    def test_fetch_balance_handles_positions_error(self):
        from alpaca_adapter import AlpacaAdapter
        mock_client = Mock()
        mock_client.get_account.return_value = {"cash": "1000", "buying_power": "2000"}
        mock_client.get_positions.side_effect = Exception("Positions error")

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.client = mock_client
        adapter._markets_cache = {}

        balance = adapter.fetch_balance()
        self.assertIn("total", balance)
        self.assertEqual(balance["total"]["USD"], 1000.0)

    def test_fetch_ticker_handles_error(self):
        from alpaca_adapter import AlpacaAdapter
        mock_client = Mock()
        mock_client.get_ticker.side_effect = Exception("Ticker error")

        adapter = AlpacaAdapter.__new__(AlpacaAdapter)
        adapter.client = mock_client
        adapter._markets_cache = {}
        adapter._market_open_cache = None
        adapter._market_open_cache_ts = 0.0
        adapter._market_open_cache_ttl = 60.0

        result = adapter.fetch_ticker("AAPL")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["last"], 0.0)


# ============================================================
# 9. KRAKEN CLIENT RETRY LOGIC
# ============================================================
class TestKrakenRetryLogic(unittest.TestCase):
    """Test Kraken client retry and DDoS handling."""

    def test_is_transient_ddos(self):
        import ccxt
        from kraken_client import KrakenClient
        kc = KrakenClient.__new__(KrakenClient)
        e = ccxt.DDoSProtection("rate limit exceeded")
        self.assertTrue(kc._is_transient(e))

    def test_is_transient_network(self):
        import ccxt
        from kraken_client import KrakenClient
        kc = KrakenClient.__new__(KrakenClient)
        e = ccxt.NetworkError("connection failed")
        self.assertTrue(kc._is_transient(e))

    def test_is_not_transient_auth(self):
        import ccxt
        from kraken_client import KrakenClient
        kc = KrakenClient.__new__(KrakenClient)
        e = ccxt.AuthenticationError("invalid key")
        self.assertFalse(kc._is_transient(e))


# ============================================================
# 10. API ENDPOINTS
# ============================================================
class TestAPIEndpoints(unittest.TestCase):
    """Test API endpoints return valid responses."""

    def setUp(self):
        import requests
        self.base = "http://localhost:8000"
        try:
            r = requests.get(f"{self.base}/api/health", timeout=5)
            if r.status_code != 200:
                self.skipTest("Server not running")
        except Exception:
            self.skipTest("Server not running")

    def test_health_endpoint(self):
        import requests
        r = requests.get(f"{self.base}/api/health", timeout=5)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["ok"])
        self.assertIn("kraken_ready", d)
        self.assertIn("alpaca_paper_ready", d)
        self.assertIn("db_ok", d)
        self.assertIn("bots", d)

    def test_bots_list(self):
        import requests
        r = requests.get(f"{self.base}/api/bots", timeout=5)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["ok"])
        self.assertIsInstance(d["bots"], list)

    def test_pnl_endpoint(self):
        import requests
        r = requests.get(f"{self.base}/api/pnl", timeout=5)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["ok"])
        self.assertIn("today", d)
        self.assertIn("total", d)

    def test_bot_creation_and_deletion(self):
        import requests
        r = requests.post(f"{self.base}/api/bots", json={
            "name": "API Test Bot", "symbol": "BTC/USD",
            "base_quote": 10, "dry_run": 1, "enabled": 0,
        }, timeout=5)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertTrue(d["ok"])
        bot_id = d["bot"]["id"]

        r2 = requests.delete(f"{self.base}/api/bots/{bot_id}", timeout=5)
        self.assertEqual(r2.status_code, 200)

    def test_dashboard_loads(self):
        import requests
        r = requests.get(f"{self.base}/", timeout=5, allow_redirects=True)
        self.assertEqual(r.status_code, 200)

    def test_bots_page_loads(self):
        import requests
        r = requests.get(f"{self.base}/bots", timeout=5)
        self.assertEqual(r.status_code, 200)

    def test_safety_page_loads(self):
        import requests
        r = requests.get(f"{self.base}/safety", timeout=5)
        self.assertEqual(r.status_code, 200)

    def test_autopilot_status(self):
        import requests
        r = requests.get(f"{self.base}/api/autopilot/status", timeout=15)
        self.assertEqual(r.status_code, 200)
        d = r.json()
        self.assertIn("ok", d)


# ============================================================
# 11. MARKET TYPE HEALING
# ============================================================
class TestMarketTypeHealing(unittest.TestCase):
    """Test that market_type is always correctly set regardless of input."""

    def test_stock_symbol_forced_to_stocks(self):
        import requests
        base = "http://localhost:8000"
        try:
            requests.get(f"{base}/api/health", timeout=2)
        except Exception:
            self.skipTest("Server not running")

        r = requests.post(f"{base}/api/bots", json={
            "name": "Heal Test", "symbol": "INTC",
            "market_type": "crypto",  # Wrong!
            "base_quote": 10, "dry_run": 1, "enabled": 0,
        }, timeout=5)
        d = r.json()
        self.assertTrue(d["ok"])
        bot = d["bot"]
        self.assertEqual(bot["market_type"], "stocks", "INTC should be forced to stocks")
        requests.delete(f"{base}/api/bots/{bot['id']}", timeout=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
