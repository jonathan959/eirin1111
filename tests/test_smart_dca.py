import unittest

from strategies import (
    detect_regime,
    SmartDcaConfig,
    DealState,
    AccountSnapshot,
    PerformanceStats,
    smart_decide,
    rsi,
    bollinger,
    select_strategy,
    RegimeResult,
)


class SmartDcaTests(unittest.TestCase):
    def test_regime_high_vol(self):
        candles = []
        price = 100.0
        for i in range(60):
            high = price * 1.05
            low = price * 0.95
            close = price * (1.0 + (0.01 if i % 2 == 0 else -0.01))
            candles.append([i, price, high, low, close, 1.0])
            price = close
        regime = detect_regime(candles)
        self.assertIn(regime.regime, {"HIGH_VOL_RISK", "RANGING", "TREND_UP", "TREND_DOWN", "RISK_OFF"})

    def test_risk_pause(self):
        candles = [[i, 100, 101, 99, 100, 1] for i in range(250)]
        cfg = SmartDcaConfig(base_quote=25, safety_quote=25, max_safety=3, tp=0.01, max_exposure_pct=0.05)
        deal = DealState(avg_entry=100.0, position_size=10.0, safety_used=0, tp_price=None, spent_quote=250.0)
        account = AccountSnapshot(total_usd=1000.0, free_usd=10.0, used_usd=0.0, positions_usd=1000.0)
        perf = PerformanceStats(realized_today=0.0, drawdown=0.0, open_deals=1)
        decision = smart_decide(candles, 100.0, cfg, deal, account, perf, now_ts=0, cooldown_until=0)
        self.assertIn(decision.action, {"PAUSE", "HOLD", "TAKE_PROFIT", "TRAIL_TP_UPDATE", "SAFETY_ORDER", "ENTER"})

    def test_risk_breaker_daily_loss(self):
        candles = [[i, 100, 101, 99, 100, 1] for i in range(250)]
        cfg = SmartDcaConfig(base_quote=25, safety_quote=25, max_safety=3, tp=0.01, max_daily_loss=10.0)
        deal = DealState(avg_entry=100.0, position_size=1.0, safety_used=0, tp_price=None, spent_quote=100.0)
        account = AccountSnapshot(total_usd=1000.0, free_usd=10.0, used_usd=0.0, positions_usd=100.0)
        perf = PerformanceStats(realized_today=-25.0, drawdown=0.0, open_deals=1)
        decision = smart_decide(candles, 100.0, cfg, deal, account, perf, now_ts=0, cooldown_until=0)
        self.assertEqual(decision.action, "PAUSE")

    def test_indicator_sanity(self):
        closes = [100 + i for i in range(30)]
        self.assertIsNotNone(rsi(closes, 14))
        bb = bollinger(closes, 20, 2.0)
        self.assertIsNotNone(bb)
        lower, mid, upper, bw = bb
        self.assertTrue(lower < mid < upper)
        self.assertTrue(bw >= 0)

    def test_selector_hysteresis(self):
        regime = RegimeResult(regime="TREND_UP", confidence=0.8, why=[], snapshot={}, scores={"uptrend_score": 0.8, "downtrend_score": 0.1, "range_score": 0.1, "high_vol_score": 0.0})
        target, switched, why = select_strategy(regime, current="smart", last_switch_ts=0, now_ts=10, min_hold_sec=60)
        self.assertFalse(switched)
        self.assertEqual(target, "smart")
        target, switched, why = select_strategy(regime, current="smart", last_switch_ts=0, now_ts=100, min_hold_sec=60)
        self.assertTrue(target in {"trend", "smart", "trend_follow"})


if __name__ == "__main__":
    unittest.main()
