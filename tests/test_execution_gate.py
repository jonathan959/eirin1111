"""Tests for the unified Execution Gate and bot config validator."""

import time
import pytest


class TestExecutionGate:
    """Unit tests for execution_gate.check_execution_gate()."""

    def test_gate_blocks_no_price_data(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate("ETH/USD", bid=None, ask=None, last_price=None)
        assert not r.allowed
        assert "No price data" in r.reason

    def test_gate_blocks_invalid_bid(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate("ETH/USD", bid=-1.0, ask=100.0)
        assert not r.allowed
        assert "Invalid bid" in r.reason

    def test_gate_blocks_wide_spread(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate(
            "ETH/USD", bid=100.0, ask=102.0,
            bot_spread_guard_pct=0.005,
        )
        assert not r.allowed
        assert "Spread too wide" in r.reason
        assert r.spread_pct is not None
        assert r.spread_pct > 0.005

    def test_gate_allows_tight_spread(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate(
            "ETH/USD", bid=100.0, ask=100.10,
            bot_spread_guard_pct=0.005,
        )
        assert r.allowed
        assert "spread" in r.checks_passed

    def test_gate_blocks_stale_ticker(self):
        from execution_gate import check_execution_gate
        old_ts = time.time() - 300
        r = check_execution_gate(
            "ETH/USD", bid=100.0, ask=100.10,
            ticker_ts=old_ts,
        )
        assert not r.allowed
        assert "Stale ticker" in r.reason

    def test_gate_allows_fresh_ticker(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate(
            "ETH/USD", bid=100.0, ask=100.10,
            ticker_ts=time.time() - 5,
        )
        assert r.allowed

    def test_gate_blocks_stale_candle(self):
        from execution_gate import check_execution_gate
        old_ts = time.time() - 50000
        r = check_execution_gate(
            "ETH/USD", bid=100.0, ask=100.10,
            last_candle_ts=old_ts,
        )
        assert not r.allowed
        assert "Stale candle" in r.reason

    def test_gate_blocks_low_volume(self):
        import os
        os.environ["EXECUTION_MIN_VOLUME_24H"] = "100000"
        import importlib
        import execution_gate
        importlib.reload(execution_gate)
        r = execution_gate.check_execution_gate(
            "ETH/USD", bid=100.0, ask=100.10,
            volume_24h=5000,
        )
        assert not r.allowed
        assert "Low 24h volume" in r.reason
        os.environ["EXECUTION_MIN_VOLUME_24H"] = "0"
        importlib.reload(execution_gate)

    def test_gate_blocks_market_order_slippage(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate(
            "MICRO/USD", bid=1.0, ask=1.20,
            order_type="market", dry_run=False,
            bot_spread_guard_pct=0.25,
        )
        assert not r.allowed
        assert ("slippage" in r.reason.lower() or "spread" in r.reason.lower())

    def test_gate_result_to_dict(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate("ETH/USD", bid=100.0, ask=100.10)
        d = r.to_dict()
        assert "allowed" in d
        assert "spread_pct" in d
        assert "checks_passed" in d

    def test_gate_adaptive_spread_high_volatility(self):
        from execution_gate import check_execution_gate
        r = check_execution_gate(
            "ETH/USD", bid=100.0, ask=100.80,
            bot_spread_guard_pct=0.003,
            volatility_pct=0.12,
        )
        assert r.allowed, f"Should allow wider spread in high vol: {r.reason}"

    def test_gate_disabled(self):
        import os
        os.environ["EXECUTION_GATE_ENABLED"] = "0"
        import importlib
        import execution_gate
        importlib.reload(execution_gate)
        r = execution_gate.check_execution_gate("ETH/USD", bid=None, ask=None)
        assert r.allowed
        os.environ["EXECUTION_GATE_ENABLED"] = "1"
        importlib.reload(execution_gate)


class TestBotConfigValidator:
    """Unit tests for bot_config_validator."""

    def test_valid_config_passes(self):
        from bot_config_validator import validate_bot_config
        cfg = {
            "symbol": "ETH/USD",
            "strategy_mode": "smart_dca",
            "market_type": "crypto",
            "tp": 0.03,
            "first_dev": 0.015,
            "step_mult": 1.2,
            "base_quote": 20.0,
            "safety_quote": 10.0,
            "max_safety": 3,
            "poll_seconds": 10,
            "spread_guard_pct": 0.003,
            "stop_loss_pct": 0.08,
        }
        cleaned, issues = validate_bot_config(cfg)
        fatal = [i for i in issues if i.startswith("ERROR:")]
        assert len(fatal) == 0
        assert cleaned["strategy_mode"] == "smart_dca"

    def test_invalid_strategy_defaults(self):
        from bot_config_validator import validate_bot_config
        cfg = {"symbol": "ETH/USD", "strategy_mode": "unknown_garbage"}
        cleaned, issues = validate_bot_config(cfg)
        assert cleaned["strategy_mode"] == "smart_dca"
        assert any("not recognized" in i for i in issues)

    def test_empty_symbol_is_fatal(self):
        from bot_config_validator import validate_bot_config
        cfg = {"symbol": ""}
        _, issues = validate_bot_config(cfg)
        fatal = [i for i in issues if i.startswith("ERROR:")]
        assert len(fatal) > 0

    def test_values_clamped_to_range(self):
        from bot_config_validator import validate_bot_config
        cfg = {
            "symbol": "ETH/USD",
            "tp": 99.0,
            "first_dev": -1.0,
            "step_mult": 100.0,
            "stop_loss_pct": 0.0001,
        }
        cleaned, issues = validate_bot_config(cfg)
        assert 0.001 <= cleaned["tp"] <= 0.50
        assert 0.001 <= cleaned["first_dev"] <= 0.20
        assert 1.0 <= cleaned["step_mult"] <= 5.0
        assert len(issues) > 0

    def test_market_type_stock_normalized(self):
        from bot_config_validator import validate_bot_config
        cfg = {"symbol": "AAPL", "market_type": "stock"}
        cleaned, _ = validate_bot_config(cfg)
        assert cleaned["market_type"] == "stocks"


class TestExploreV2Scoring:
    """Tests for explore_v2 scoring improvements."""

    def test_crash_penalty_applied(self):
        from explore_v2 import enhance_score
        base = 70.0
        snap = {"return_30d": -0.25}
        score, reasons = enhance_score(base, snap, "BULL")
        assert score < base
        assert any("30d crash" in r.lower() or "30d" in r.lower() for r in reasons)

    def test_multi_tf_alignment_bonus(self):
        from explore_v2 import enhance_score
        base = 50.0
        snap = {"regime_1d": "BULL", "regime_4h": "BREAKOUT"}
        score, reasons = enhance_score(base, snap, "BULL")
        assert score > base
        assert any("alignment" in r.lower() or "multi" in r.lower() for r in reasons)

    def test_tf_divergence_penalty(self):
        from explore_v2 import enhance_score
        base = 50.0
        snap = {"regime_1d": "BULL", "regime_4h": "BEAR"}
        score, reasons = enhance_score(base, snap, "BULL")
        assert score < base

    def test_diversify_caps_low_liquidity(self):
        from explore_v2 import diversify_picks
        items = []
        for i in range(20):
            items.append({"symbol": f"MICRO{i}/USD", "score": 80 - i, "volume": 1000})
        result = diversify_picks(items, top_k=10)
        assert len(result) <= 10

    def test_score_breakdown_returns_list(self):
        from explore_v2 import compute_score_breakdown
        bd = compute_score_breakdown(75.0, {"weekly_trend": "Up", "atr_pct": 0.03}, "BULL")
        assert isinstance(bd, list)
        assert len(bd) >= 1


class TestGateResult:
    """Test that GateResult serializes properly for the API."""

    def test_to_dict_complete(self):
        from execution_gate import GateResult
        r = GateResult(
            allowed=False,
            reason="test",
            bid=100.0,
            ask=101.0,
            spread_pct=0.01,
            spread_threshold=0.005,
        )
        d = r.to_dict()
        assert d["allowed"] is False
        assert d["bid"] == 100.0
        assert d["spread_pct"] == 0.01
