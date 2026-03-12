"""
Tests for autopilot purposed-entry integration:
- Candidates not passing preflight are skipped
- WATCH candidates go to watchlist, not bots
- READY candidates create bots with rationale
"""

import json
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)


@pytest.fixture
def mock_db():
    """Mock DB functions for autopilot testing."""
    with patch.dict(os.environ, {
        "AUTOPILOT_MODE": "STRICT_READY",
        "WATCHLIST_ENABLED": "1",
        "MIN_ENTRY_CONFIDENCE": "0.65",
    }):
        import importlib
        import autopilot
        importlib.reload(autopilot)
        yield autopilot


class TestAutopilotReadyOnly:
    def test_not_ready_candidate_skipped_strict_mode(self, mock_db):
        """In STRICT_READY mode, candidates that aren't ready should be skipped."""
        scanner_setup_not_ready = {
            "ready_now": False,
            "ready_reason": "confidence 0.45 < 0.65",
            "edge_score": 0.45,
            "confidence": 0.45,
            "entry_type": "NO_TRADE",
        }

        with patch.object(mock_db, '_evaluate_candidate_readiness',
                          return_value=scanner_setup_not_ready):
            with patch.object(mock_db, '_add_to_watchlist') as mock_watchlist:
                with patch('db.list_bots', return_value=[]):
                    with patch('db.get_setting', return_value="{}"):
                        with patch('db.set_setting'):
                            with patch('db.list_recommendations', return_value=[
                                {"symbol": "BTC/USD", "score": 85, "metrics_json": '{"market_type":"crypto"}'},
                            ]):
                                with patch('db.get_recommendation', return_value=None):
                                    with patch('db.add_autopilot_audit_log'):
                                        result = mock_db.run_autopilot_cycle(
                                            create_bot_fn=MagicMock(),
                                            delete_bot_fn=MagicMock(),
                                            start_bot_fn=MagicMock(),
                                            stop_bot_fn=MagicMock(),
                                            get_portfolio_total_fn=lambda: 10000,
                                            notify_fn=MagicMock(),
                                            force_run=True,
                                        )
                                        assert result["created"] == 0
                                        assert any("not_ready" in s.get("reason", "")
                                                   for s in result.get("skipped", []))

    def test_ready_candidate_creates_bot(self, mock_db):
        """Candidates marked READY should create bots."""
        scanner_setup_ready = {
            "ready_now": True,
            "ready_reason": "all_criteria_met",
            "edge_score": 0.75,
            "confidence": 0.75,
            "entry_type": "BREAKOUT",
            "evidence": ["Regime: TREND_UP", "Volume breakout"],
            "target_levels": {"tp1": 105, "tp2": 110},
            "invalidation_level": 95,
        }

        created_bots = []

        def fake_create(config):
            created_bots.append(config)
            return len(created_bots)

        with patch.object(mock_db, '_evaluate_candidate_readiness',
                          return_value=scanner_setup_ready):
            with patch('db.list_bots', return_value=[]):
                with patch('db.get_setting', return_value="{}"):
                    with patch('db.set_setting'):
                        with patch('db.list_recommendations', return_value=[
                            {"symbol": "BTC/USD", "score": 85, "metrics_json": '{"market_type":"crypto"}'},
                        ]):
                            with patch('db.get_recommendation', return_value=None):
                                with patch('db.add_autopilot_audit_log'):
                                    result = mock_db.run_autopilot_cycle(
                                        create_bot_fn=fake_create,
                                        delete_bot_fn=MagicMock(),
                                        start_bot_fn=MagicMock(),
                                        stop_bot_fn=MagicMock(),
                                        get_portfolio_total_fn=lambda: 10000,
                                        notify_fn=MagicMock(),
                                        force_run=True,
                                    )
                                    assert result["created"] == 1
                                    assert len(created_bots) == 1
                                    assert created_bots[0]["symbol"] == "BTC/USD"

    def test_scanner_none_falls_through(self, mock_db):
        """When scanner is unavailable (returns None), bot creation proceeds as before."""
        created_bots = []

        def fake_create(config):
            created_bots.append(config)
            return len(created_bots)

        with patch.object(mock_db, '_evaluate_candidate_readiness', return_value=None):
            with patch('db.list_bots', return_value=[]):
                with patch('db.get_setting', return_value="{}"):
                    with patch('db.set_setting'):
                        with patch('db.list_recommendations', return_value=[
                            {"symbol": "BTC/USD", "score": 85, "metrics_json": '{"market_type":"crypto"}'},
                        ]):
                            with patch('db.get_recommendation', return_value=None):
                                with patch('db.add_autopilot_audit_log'):
                                    result = mock_db.run_autopilot_cycle(
                                        create_bot_fn=fake_create,
                                        delete_bot_fn=MagicMock(),
                                        start_bot_fn=MagicMock(),
                                        stop_bot_fn=MagicMock(),
                                        get_portfolio_total_fn=lambda: 10000,
                                        notify_fn=MagicMock(),
                                        force_run=True,
                                    )
                                    assert result["created"] == 1

    def test_watch_candidate_goes_to_watchlist(self, mock_db):
        """WATCH candidates (not ready) should be added to the watchlist, not create bots."""
        scanner_setup_watch = {
            "ready_now": False,
            "ready_reason": "no_entry_type",
            "edge_score": 0.55,
            "confidence": 0.55,
            "entry_type": "NO_TRADE",
            "trigger_conditions": "Wait for ADX > 20",
            "regime": "RANGE",
        }

        with patch.object(mock_db, '_evaluate_candidate_readiness',
                          return_value=scanner_setup_watch):
            with patch.object(mock_db, '_add_to_watchlist') as mock_add_wl:
                with patch('db.list_bots', return_value=[]):
                    with patch('db.get_setting', return_value="{}"):
                        with patch('db.set_setting'):
                            with patch('db.list_recommendations', return_value=[
                                {"symbol": "SOL/USD", "score": 80, "metrics_json": '{"market_type":"crypto"}'},
                            ]):
                                with patch('db.get_recommendation', return_value=None):
                                    with patch('db.add_autopilot_audit_log'):
                                        result = mock_db.run_autopilot_cycle(
                                            create_bot_fn=MagicMock(),
                                            delete_bot_fn=MagicMock(),
                                            start_bot_fn=MagicMock(),
                                            stop_bot_fn=MagicMock(),
                                            get_portfolio_total_fn=lambda: 10000,
                                            notify_fn=MagicMock(),
                                            force_run=True,
                                        )
                                        assert result["created"] == 0
                                        assert mock_add_wl.called
                                        call_args = mock_add_wl.call_args
                                        assert call_args[0][0] == "SOL/USD"


class TestAllowWatchDryRun:
    def test_dry_run_bot_for_watch_candidate(self):
        """In ALLOW_WATCH_DRYRUN mode, WATCH candidates get dry-run bots."""
        with patch.dict(os.environ, {
            "AUTOPILOT_MODE": "ALLOW_WATCH_DRYRUN",
            "WATCHLIST_ENABLED": "1",
        }):
            import importlib
            import autopilot
            importlib.reload(autopilot)

            scanner_setup_watch = {
                "ready_now": False,
                "ready_reason": "confidence too low",
                "edge_score": 0.50,
                "confidence": 0.50,
            }

            created_bots = []

            def fake_create(config):
                created_bots.append(config)
                return len(created_bots)

            with patch.object(autopilot, '_evaluate_candidate_readiness',
                              return_value=scanner_setup_watch):
                with patch.object(autopilot, '_add_to_watchlist'):
                    with patch('db.list_bots', return_value=[]):
                        with patch('db.get_setting', return_value="{}"):
                            with patch('db.set_setting'):
                                with patch('db.list_recommendations', return_value=[
                                    {"symbol": "BTC/USD", "score": 85, "metrics_json": '{"market_type":"crypto"}'},
                                ]):
                                    with patch('db.get_recommendation', return_value=None):
                                        with patch('db.add_autopilot_audit_log'):
                                            result = autopilot.run_autopilot_cycle(
                                                create_bot_fn=fake_create,
                                                delete_bot_fn=MagicMock(),
                                                start_bot_fn=MagicMock(),
                                                stop_bot_fn=MagicMock(),
                                                get_portfolio_total_fn=lambda: 10000,
                                                notify_fn=MagicMock(),
                                                force_run=True,
                                            )
                                            assert result["created"] == 1
                                            assert created_bots[0]["dry_run"] == 1


class TestNotificationPayload:
    def test_notify_includes_evidence(self, mock_db):
        """Notification for created bot should include evidence and entry details."""
        scanner_setup = {
            "ready_now": True,
            "ready_reason": "all_criteria_met",
            "edge_score": 0.75,
            "confidence": 0.75,
            "entry_type": "BREAKOUT",
            "evidence": ["Regime: TREND_UP", "Volume breakout", "EMA bullish"],
            "target_levels": {"tp1": 105, "tp2": 110},
            "invalidation_level": 95,
        }

        notify_calls = []

        def fake_notify(event, payload):
            notify_calls.append((event, payload))

        with patch.object(mock_db, '_evaluate_candidate_readiness',
                          return_value=scanner_setup):
            with patch('db.list_bots', return_value=[]):
                with patch('db.get_setting', return_value="{}"):
                    with patch('db.set_setting'):
                        with patch('db.list_recommendations', return_value=[
                            {"symbol": "BTC/USD", "score": 85, "metrics_json": '{"market_type":"crypto"}'},
                        ]):
                            with patch('db.get_recommendation', return_value=None):
                                with patch('db.add_autopilot_audit_log'):
                                    mock_db.run_autopilot_cycle(
                                        create_bot_fn=lambda c: 1,
                                        delete_bot_fn=MagicMock(),
                                        start_bot_fn=MagicMock(),
                                        stop_bot_fn=MagicMock(),
                                        get_portfolio_total_fn=lambda: 10000,
                                        notify_fn=fake_notify,
                                        force_run=True,
                                    )
                                    assert len(notify_calls) == 1
                                    event, payload = notify_calls[0]
                                    assert event == "autopilot_bot_created"
                                    assert "evidence" in payload
                                    assert len(payload["evidence"]) <= 3
                                    assert payload["entry_type"] == "BREAKOUT"
                                    assert payload["confidence"] == 0.75
