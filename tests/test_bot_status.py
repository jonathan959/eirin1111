"""Tests for services/bot_status canonical signal + action_state."""
import unittest

from services import bot_status as bs


class TestBotStatus(unittest.TestCase):
    def test_strong_buy_wait_confidence(self):
        b = {"dry_run": 0, "symbol": "BTC/USD"}
        sig = {"score": 92.0, "conviction_grade": "A"}
        intel = {"decision_action": "PAUSE", "decision_reason": "Entry confidence 0.50 < threshold 0.65"}
        out = bs.compute_bot_status(b, sig, intel, {"level": "OK"}, {}, allow_live_trading=True)
        self.assertEqual(out.signal, bs.SIGNAL_STRONG_BUY)
        self.assertEqual(out.action_state, bs.ACTION_WAIT)

    def test_strong_buy_trade(self):
        b = {"dry_run": 0, "symbol": "ETH/USD"}
        sig = {"score": 90.0, "conviction_grade": "A"}
        intel = {"decision_action": "ENTER", "allowed_actions": "TRADE_ALLOWED"}
        out = bs.compute_bot_status(b, sig, intel, {"level": "OK"}, {"allowed": True}, allow_live_trading=True)
        self.assertEqual(out.signal, bs.SIGNAL_STRONG_BUY)
        self.assertEqual(out.action_state, bs.ACTION_TRADE)

    def test_buy_watch_gate(self):
        b = {"dry_run": 0}
        sig = {"score": 60.0, "conviction_grade": "B"}
        gate = {"allowed": False, "reason": "Spread too wide", "spread_pct": 0.015}
        out = bs.compute_bot_status(b, sig, {}, {"level": "OK"}, gate, allow_live_trading=True)
        self.assertEqual(out.signal, bs.SIGNAL_BUY)
        self.assertEqual(out.action_state, bs.ACTION_WATCH)
        self.assertIn("Spread", out.reason)

    def test_neutral_blocked_risk(self):
        b = {"dry_run": 0}
        sig = {"score": 45.0, "conviction_grade": "C"}
        out = bs.compute_bot_status(
            b,
            sig,
            {},
            {"level": "CRITICAL", "reason": "Risk engine: daily loss limit exceeded."},
            {"allowed": True},
            allow_live_trading=True,
        )
        self.assertEqual(out.signal, bs.SIGNAL_NEUTRAL)
        self.assertEqual(out.action_state, bs.ACTION_BLOCKED)

    def test_dry_run_always_watch(self):
        b = {"dry_run": 1}
        sig = {"score": 99.0, "conviction_grade": "A"}
        out = bs.compute_bot_status(b, sig, {}, {"level": "OK"}, {}, allow_live_trading=True)
        self.assertEqual(out.signal, bs.SIGNAL_STRONG_BUY)
        self.assertEqual(out.action_state, bs.ACTION_WATCH)
        self.assertIn("Dry run", out.reason)


if __name__ == "__main__":
    unittest.main()
