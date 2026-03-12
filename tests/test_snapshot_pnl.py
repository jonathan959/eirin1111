"""Tests for unrealized PnL fields in BotRunner.snapshot()."""
import os
import sys
import threading

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from bot_manager import RuntimeState


def _make_snapshot(last_price, avg_entry, base_pos):
    """Helper: build a RuntimeState, set fields, return snapshot dict."""
    state = RuntimeState()
    state.last_price = last_price
    state.avg_entry = avg_entry
    state.base_pos = base_pos
    state.running = True

    class FakeRunner:
        def __init__(self, st):
            self.state = st
            self._lock = threading.RLock()

        def snapshot(self):
            with self._lock:
                lp = self.state.last_price
                ae = self.state.avg_entry
                bp = self.state.base_pos
                upnl_q = None
                upnl_p = None
                pnl_st = "FLAT"
                if ae and ae > 0 and bp and bp > 0 and lp is not None:
                    upnl_q = (lp - ae) * bp
                    upnl_p = ((lp / ae) - 1.0) * 100.0
                    pnl_st = "UP" if upnl_q > 0 else ("DOWN" if upnl_q < 0 else "FLAT")
                return {
                    "last_price": lp,
                    "avg_entry": ae,
                    "base_pos": bp,
                    "unrealized_pnl_quote": upnl_q,
                    "unrealized_pnl_pct": upnl_p,
                    "pnl_status": pnl_st,
                }

    return FakeRunner(state).snapshot()


class TestSnapshotPnl:
    def test_profit_position(self):
        snap = _make_snapshot(last_price=102, avg_entry=100, base_pos=0.5)
        assert snap["unrealized_pnl_quote"] == 1.0
        assert abs(snap["unrealized_pnl_pct"] - 2.0) < 1e-10
        assert snap["pnl_status"] == "UP"

    def test_loss_position(self):
        snap = _make_snapshot(last_price=98, avg_entry=100, base_pos=0.5)
        assert snap["unrealized_pnl_quote"] == -1.0
        assert abs(snap["unrealized_pnl_pct"] - (-2.0)) < 1e-10
        assert snap["pnl_status"] == "DOWN"

    def test_flat_position(self):
        snap = _make_snapshot(last_price=100, avg_entry=100, base_pos=0.5)
        assert snap["unrealized_pnl_quote"] == 0.0
        assert snap["unrealized_pnl_pct"] == 0.0
        assert snap["pnl_status"] == "FLAT"

    def test_no_position(self):
        snap = _make_snapshot(last_price=100, avg_entry=0, base_pos=0)
        assert snap["unrealized_pnl_quote"] is None
        assert snap["unrealized_pnl_pct"] is None
        assert snap["pnl_status"] == "FLAT"

    def test_no_avg_entry(self):
        snap = _make_snapshot(last_price=100, avg_entry=None, base_pos=1.0)
        assert snap["unrealized_pnl_quote"] is None
        assert snap["unrealized_pnl_pct"] is None
        assert snap["pnl_status"] == "FLAT"

    def test_no_price(self):
        snap = _make_snapshot(last_price=None, avg_entry=100, base_pos=1.0)
        assert snap["unrealized_pnl_quote"] is None
        assert snap["unrealized_pnl_pct"] is None
        assert snap["pnl_status"] == "FLAT"

    def test_large_crypto_position(self):
        snap = _make_snapshot(last_price=65432.10, avg_entry=64000.00, base_pos=0.015)
        expected_q = (65432.10 - 64000.00) * 0.015
        expected_p = ((65432.10 / 64000.00) - 1.0) * 100.0
        assert abs(snap["unrealized_pnl_quote"] - expected_q) < 1e-6
        assert abs(snap["unrealized_pnl_pct"] - expected_p) < 1e-6
        assert snap["pnl_status"] == "UP"

    def test_small_penny_stock(self):
        snap = _make_snapshot(last_price=0.45, avg_entry=0.50, base_pos=1000)
        expected_q = (0.45 - 0.50) * 1000
        expected_p = ((0.45 / 0.50) - 1.0) * 100.0
        assert abs(snap["unrealized_pnl_quote"] - expected_q) < 1e-6
        assert abs(snap["unrealized_pnl_pct"] - expected_p) < 1e-6
        assert snap["pnl_status"] == "DOWN"
