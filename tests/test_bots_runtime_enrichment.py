"""Runtime enrichment for bot list APIs (includes bot_status on SSE payload)."""
import json
import unittest
from unittest.mock import MagicMock, patch


class TestBotsRuntimeEnrichment(unittest.TestCase):
    def test_enrich_sets_bot_status_signal_and_action(self):
        import worker_api as wa

        snap = {
            "running": True,
            "base_pos": 0.0,
            "risk_level": "OK",
            "risk_reason": "",
            "risk_state": None,
            "gate_details": {"allowed": True},
            "decision_action": "ENTER",
            "decision_reason": "",
            "intelligence_allowed": "TRADE_ALLOWED",
            "last_event": None,
            "last_tick_ts": None,
            "unrealized_pnl_pct": None,
        }
        mock_bm = MagicMock()
        mock_bm.snapshot.return_value = snap
        bots = [{"id": 7, "dry_run": 0, "symbol": "BTC/USD", "enabled": 1}]
        with patch.object(wa, "bm", mock_bm), patch.object(
            wa, "load_latest_signal_for_bot", return_value={"score": 88.0, "conviction_grade": "A"}
        ), patch.object(wa, "_pause_state", return_value=False), patch.object(
            wa, "_kill_switch_state", return_value=False
        ), patch.object(wa, "latest_open_deal", return_value=None):
            wa._enrich_bots_runtime_fields(bots)

        bs = bots[0].get("bot_status") or {}
        self.assertEqual(bs.get("signal"), "Strong Buy")
        self.assertEqual(bs.get("action_state"), "Trade")
        self.assertIn("reason", bs)

    def test_sse_json_contains_bot_status_fields(self):
        """Mirrors /api/bots/stream payload shape after enrichment."""
        import worker_api as wa

        snap = {
            "running": True,
            "base_pos": 0.0,
            "risk_level": "OK",
            "risk_reason": "",
            "risk_state": None,
            "gate_details": {"allowed": True},
            "decision_action": "ENTER",
            "decision_reason": "",
            "intelligence_allowed": "TRADE_ALLOWED",
            "last_event": None,
            "last_tick_ts": None,
            "unrealized_pnl_pct": None,
        }
        mock_bm = MagicMock()
        mock_bm.snapshot.return_value = snap

        def _fake_list():
            return [{"id": 9, "dry_run": 0, "symbol": "ETH/USD", "enabled": 1}]

        bots = _fake_list()
        with patch.object(wa, "bm", mock_bm), patch.object(
            wa, "load_latest_signal_for_bot", return_value={"score": 60.0, "conviction_grade": "B"}
        ), patch.object(wa, "_pause_state", return_value=False), patch.object(
            wa, "_kill_switch_state", return_value=False
        ), patch.object(wa, "latest_open_deal", return_value=None):
            wa._enrich_bots_runtime_fields(bots)

        line = json.dumps(bots)
        decoded = json.loads(line)
        self.assertEqual(len(decoded), 1)
        bs = decoded[0]["bot_status"]
        self.assertEqual(bs["signal"], "Buy")
        self.assertEqual(bs["action_state"], "Trade")


if __name__ == "__main__":
    unittest.main()
