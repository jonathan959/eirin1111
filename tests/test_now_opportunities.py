#!/usr/bin/env python3
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_reco(symbol: str, score: float, horizon: str = "short", market_type: str = "crypto") -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "score": score,
        "created_ts": 1700000000,
        "metrics_json": '{"market_type": "%s", "recommended_strategy": "smart_dca"}' % market_type,
    }


def test_now_opportunities_basic(monkeypatch):
    import autopilot

    recos_by_horizon = {
        "short": [
            _make_reco("XBT/USD", 90, "short", "crypto"),
            _make_reco("ETH/USD", 85, "short", "crypto"),
        ],
        "long": [
            _make_reco("AAPL", 88, "long", "stock"),
        ],
    }

    def fake_get_autopilot_config() -> Dict[str, Any]:
        return {"min_score": 80, "asset_types": "both", "sectors_avoid": []}

    def fake_list_bots() -> List[Dict[str, Any]]:
        return [{"symbol": "ETH/USD", "enabled": 1}]  # already has a bot, should be excluded

    def fake_list_recommendations(horizon: str, limit: int = 200, exclude_bases=None):
        return recos_by_horizon.get(horizon, [])

    monkeypatch.setattr(autopilot, "get_autopilot_config", fake_get_autopilot_config)
    monkeypatch.setattr("db.list_bots", fake_list_bots)
    monkeypatch.setattr("db.list_recommendations", fake_list_recommendations)

    items = autopilot.get_now_opportunities(asset_filter="both", max_count=3)
    # ETH should be excluded (already has a bot); XBT and AAPL are candidates
    symbols = {i["symbol"] for i in items}
    assert "ETH/USD" not in symbols
    assert "XBT/USD" in symbols
    assert "AAPL" in symbols


def test_now_opportunities_min_score(monkeypatch):
    import autopilot

    recos_by_horizon = {
        "short": [
            _make_reco("LOW/USD", 60, "short", "crypto"),
        ],
        "long": [],
    }

    def fake_get_autopilot_config() -> Dict[str, Any]:
        # Very high min_score so LOW/USD is filtered out
        return {"min_score": 90, "asset_types": "both", "sectors_avoid": []}

    def fake_list_bots() -> List[Dict[str, Any]]:
        return []

    def fake_list_recommendations(horizon: str, limit: int = 200, exclude_bases=None):
        return recos_by_horizon.get(horizon, [])

    monkeypatch.setattr(autopilot, "get_autopilot_config", fake_get_autopilot_config)
    monkeypatch.setattr("db.list_bots", fake_list_bots)
    monkeypatch.setattr("db.list_recommendations", fake_list_recommendations)

    items = autopilot.get_now_opportunities(asset_filter="both", max_count=3)
    # No items should pass the min_score filter
    assert items == []

