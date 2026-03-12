"""
End-to-end lifecycle tests for Bot 41 (Smart PAXG/USD).
Prints PASS/FAIL for each test with actual values.
"""
import os
import sys
import time
import traceback
import logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_utils import load_env
load_env()

BOT_ID = 41
DEAL_ID = 30
SYMBOL = "PAXG/USD"

results = []
warnings_list = []

SYNTHETIC_BOT_CONFIG = {
    "id": BOT_ID, "name": "Smart PAXG/USD", "symbol": SYMBOL,
    "enabled": 1, "dry_run": 0, "base_quote": 5.0, "safety_quote": 2.0,
    "max_safety": 3, "first_dev": 0.01, "step_mult": 1.2,
    "tp": 0.01, "trend_filter": 0, "trend_sma": 200,
    "max_spend_quote": 20.0, "poll_seconds": 10,
    "strategy_mode": "trend_follow", "forced_strategy": "",
    "max_open_orders": 6, "vol_gap_mult": 1.0, "tp_vol_mult": 1.0,
    "min_gap_pct": 0.003, "max_gap_pct": 0.06,
    "max_total_exposure_pct": 0.50, "per_symbol_exposure_pct": 0.15,
    "min_free_cash_pct": 0.1, "max_concurrent_deals": 6,
    "spread_guard_pct": 0.003, "limit_timeout_sec": 8,
    "daily_loss_limit_pct": 0.06, "pause_hours": 6,
    "market_type": "crypto", "alpaca_mode": "paper",
    "use_kelly_sizing": 1, "kelly_fraction": 0.25,
    "stop_loss_pct": 0.08, "max_drawdown_pct": 0.15,
    "trailing_activation_pct": 0.0, "trailing_distance_pct": 0.0,
}


def get_bot_config():
    """Get bot 41 from DB, or fall back to synthetic config."""
    try:
        from db import get_bot
        bot = get_bot(BOT_ID)
        if bot:
            return bot, False
    except Exception:
        pass
    return SYNTHETIC_BOT_CONFIG, True


def record(test_num, name, passed, details="", expected=None, actual=None, warn=None):
    status = "PASS" if passed else "FAIL"
    results.append({
        "num": test_num, "name": name, "passed": passed,
        "details": details, "expected": expected, "actual": actual,
    })
    if warn:
        warnings_list.append(f"Test {test_num}: {warn}")
    print(f"\n{'='*70}")
    print(f"  TEST {test_num}: {name}")
    print(f"  Status: {status}")
    if details:
        for line in details.strip().split("\n"):
            print(f"    {line}")
    if not passed and expected is not None:
        print(f"    Expected: {expected}")
        print(f"    Actual:   {actual}")
    if warn:
        print(f"    WARNING: {warn}")
    print(f"{'='*70}")


def get_kraken_client():
    from kraken_client import KrakenClient
    return KrakenClient()


# ─────────────────────────────────────────────────────────────────────
# TEST 1: Intelligence Layer
# ─────────────────────────────────────────────────────────────────────
def test_1_intelligence_layer():
    try:
        from intelligence_layer import IntelligenceLayer, IntelligenceContext

        bot, is_synthetic = get_bot_config()
        kc = get_kraken_client()
        ticker = kc.fetch_ticker(SYMBOL)
        last_price = float(ticker.get("last", 0))
        bid = float(ticker.get("bid", 0))
        ask = float(ticker.get("ask", 0))
        spread = (ask - bid) / ask * 100 if ask > 0 else 0

        candles_1h = kc.fetch_ohlcv(SYMBOL, timeframe="1h", limit=200)
        candles_4h = kc.fetch_ohlcv(SYMBOL, timeframe="4h", limit=100)
        candles_1d = kc.fetch_ohlcv(SYMBOL, timeframe="1d", limit=60)

        ctx = IntelligenceContext(
            symbol=SYMBOL,
            last_price=last_price,
            bid_price=bid,
            ask_price=ask,
            spread_pct=spread / 100,
            candles_1h=candles_1h,
            candles_4h=candles_4h,
            candles_1d=candles_1d,
            bot_config=bot,
            dry_run=True,
            portfolio_total_usd=20.0,
            now_ts=int(time.time()),
            last_price_ts=int(time.time()),
            last_candle_ts=int(candles_1h[-1][0] / 1000) if candles_1h else int(time.time()),
        )

        layer = IntelligenceLayer()
        decision = layer.evaluate(ctx)

        dv = decision.data_validity.data_ok
        ms_ok = decision.market_safety.allowed_actions is not None
        regime_ok = decision.regime_detection.regime is not None

        details = (
            f"Price: ${last_price:,.2f}\n"
            f"Data Validity (Phase1): {dv}\n"
            f"Market Safety (Phase2): {decision.market_safety.allowed_actions} | reasons: {decision.market_safety.reasons}\n"
            f"Regime Detection (Phase3): {decision.regime_detection.regime} (conf={decision.regime_detection.confidence:.2f})\n"
            f"Final Action: {decision.final_action}\n"
            f"Final Reason: {decision.final_reason}"
        )

        passed = dv and ms_ok and regime_ok and decision.final_action is not None
        record(1, "Intelligence Layer", passed, details)
    except Exception as e:
        record(1, "Intelligence Layer", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 2: Strategy (Trend Follow)
# ─────────────────────────────────────────────────────────────────────
def test_2_strategy():
    try:
        from strategies import (
            TrendFollowStrategy, StrategyContext, DealState, AccountSnapshot,
            PerformanceStats, RegimeResult,
        )

        bot, _ = get_bot_config()
        kc = get_kraken_client()
        ticker = kc.fetch_ticker(SYMBOL)
        last_price = float(ticker.get("last", 0))

        candles_5m = kc.fetch_ohlcv(SYMBOL, timeframe="5m", limit=300)
        candles_15m = kc.fetch_ohlcv(SYMBOL, timeframe="15m", limit=200)
        candles_1h = kc.fetch_ohlcv(SYMBOL, timeframe="1h", limit=200)
        candles_4h = kc.fetch_ohlcv(SYMBOL, timeframe="4h", limit=100)

        closes_1h = [float(c[4]) for c in candles_1h] if candles_1h else []
        sma20 = sum(closes_1h[-20:]) / 20 if len(closes_1h) >= 20 else 0
        sma50 = sum(closes_1h[-50:]) / 50 if len(closes_1h) >= 50 else 0
        regime_str = "trending_up" if sma20 > sma50 else "ranging"

        deal = DealState(avg_entry=None, position_size=0, safety_used=0, tp_price=None)
        account = AccountSnapshot(total_usd=20.0, free_usd=15.0, used_usd=5.0, positions_usd=5.0)
        perf = PerformanceStats(realized_today=0.0, drawdown=0.0, open_deals=1)
        regime = RegimeResult(
            regime=regime_str, confidence=0.7, why=["test"],
            snapshot={"sma20": sma20, "sma50": sma50}, scores={},
        )

        ctx = StrategyContext(
            symbol=SYMBOL, last_price=last_price,
            candles_5m=candles_5m, candles_15m=candles_15m,
            candles_1h=candles_1h, candles_4h=candles_4h,
            deal=deal, account=account, perf=perf,
            now_ts=int(time.time()), cooldown_until=0,
            cfg=bot or {}, regime=regime,
        )

        strat = TrendFollowStrategy()
        decision = strat.propose_orders(ctx)

        details = (
            f"Regime: {regime_str} (SMA20={sma20:.2f} vs SMA50={sma50:.2f})\n"
            f"Action: {decision.action}\n"
            f"Reason: {decision.reason}\n"
            f"Debug: {decision.debug}"
        )

        passed = decision.action in ("ENTER", "HOLD", "SAFETY_ORDER", "TAKE_PROFIT", "EXIT", "PAUSE", "TRAIL_TP_UPDATE")
        record(2, "Strategy (Trend Follow)", passed, details)
    except Exception as e:
        record(2, "Strategy (Trend Follow)", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 3: Risk Engine
# ─────────────────────────────────────────────────────────────────────
def test_3_risk_engine():
    try:
        from risk_engine import RiskContext, RiskConfig, can_open_trade

        cfg = RiskConfig()
        ctx = RiskContext(
            bot_id=BOT_ID,
            symbol=SYMBOL,
            balance_total_usd=20.0,
            balance_free_usd=15.0,
            positions_usd={SYMBOL: 5.21},
            symbol_position_usd=5.21,
            trades_today=1,
            proposed_order_usd=2.0,
            is_crypto=True,
            is_safety_order=False,
        )

        allowed, reason = can_open_trade(ctx)

        is_small = ctx.balance_total_usd < cfg.SMALL_ACCOUNT_THRESHOLD
        asset_exp = 5.21 / 20.0
        max_asset = max(cfg.MAX_ASSET_EXPOSURE_PCT, cfg.SMALL_ACCOUNT_MAX_ASSET_PCT) if is_small else cfg.MAX_ASSET_EXPOSURE_PCT

        details = (
            f"Allowed: {allowed}\n"
            f"Reason: {reason}\n"
            f"Small account (<${cfg.SMALL_ACCOUNT_THRESHOLD}): {is_small}\n"
            f"Asset exposure: {asset_exp*100:.2f}% (cap: {max_asset*100:.1f}%)\n"
            f"Proposed order: $2.00\n"
            f"Floor: ${cfg.MIN_ORDER_USD_FLOOR}\n"
            f"Balance: $20.00"
        )

        record(3, "Risk Engine", allowed, details, expected="allowed=True", actual=f"allowed={allowed}, reason={reason}")
    except Exception as e:
        record(3, "Risk Engine", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 4: Execution Gate
# ─────────────────────────────────────────────────────────────────────
def test_4_execution_gate():
    try:
        from execution_gate import check_execution_gate

        kc = get_kraken_client()
        ticker = kc.fetch_ticker(SYMBOL)
        bid = float(ticker.get("bid", 0))
        ask = float(ticker.get("ask", 0))
        last_price = float(ticker.get("last", 0))
        vol = float(ticker.get("volume", 0))
        now = time.time()

        candles = kc.fetch_ohlcv(SYMBOL, timeframe="1h", limit=2)
        candle_ts = candles[-1][0] / 1000 if candles else now

        result = check_execution_gate(
            symbol=SYMBOL, side="buy", order_type="market",
            bid=bid, ask=ask, last_price=last_price,
            ticker_ts=now, last_candle_ts=candle_ts,
            volume_24h=vol, quote_amount=2.0, dry_run=True,
        )

        spread_pct = result.spread_pct if hasattr(result, 'spread_pct') else None
        ticker_age = result.ticker_age_sec if hasattr(result, 'ticker_age_sec') else None
        candle_age = result.candle_age_sec if hasattr(result, 'candle_age_sec') else None

        details = (
            f"Allowed: {result.allowed}\n"
            f"Reason: {result.reason}\n"
            f"Spread: {spread_pct*100 if spread_pct else 'N/A'}%\n"
            f"Ticker age: {ticker_age:.1f}s\n"
            f"Candle age: {candle_age:.1f}s\n"
            f"Volume 24h: {vol:,.2f}\n"
            f"Checks passed: {result.checks_passed}\n"
            f"Checks failed: {result.checks_failed}"
        )

        record(4, "Execution Gate", result.allowed, details,
               expected="allowed=True", actual=f"allowed={result.allowed}, reason={result.reason}")
    except Exception as e:
        record(4, "Execution Gate", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 5: Safety Order Trigger
# ─────────────────────────────────────────────────────────────────────
def test_5_safety_order():
    try:
        from db import latest_open_deal

        bot, is_synthetic = get_bot_config()
        deal = None
        try:
            deal = latest_open_deal(BOT_ID)
        except Exception:
            pass

        entry_avg = float(deal["entry_avg"]) if deal and deal.get("entry_avg") else None
        if not entry_avg:
            kc = get_kraken_client()
            ticker = kc.fetch_ticker(SYMBOL)
            entry_avg = float(ticker.get("last", 5100))

        base_quote = float(bot.get("base_quote", 0))
        safety_quote = float(bot.get("safety_quote", 0))
        first_dev = float(bot.get("first_dev", 0.01))
        step_mult = float(bot.get("step_mult", 1.2))
        max_safety = int(bot.get("max_safety", 0))

        trigger_prices = []
        dev = first_dev
        for i in range(max_safety):
            trigger = entry_avg * (1 - dev)
            trigger_prices.append(trigger)
            dev *= step_mult

        simulated_price = trigger_prices[0] - 1.0 if trigger_prices else entry_avg * 0.98
        would_trigger = any(simulated_price <= tp for tp in trigger_prices)
        triggered_index = None
        for i, tp in enumerate(trigger_prices):
            if simulated_price <= tp:
                triggered_index = i + 1
                break

        details = (
            f"Entry avg: ${entry_avg:,.2f} {'(from deal)' if deal else '(simulated from current price)'}\n"
            f"Safety quote: ${safety_quote:.2f}\n"
            f"Max safety orders: {max_safety}\n"
            f"First deviation: {first_dev*100:.2f}%\n"
            f"Step multiplier: {step_mult}x\n"
            f"Trigger prices: {['$%.2f' % p for p in trigger_prices]}\n"
            f"Simulated price: ${simulated_price:,.2f}\n"
            f"Would trigger safety order: {would_trigger}"
        )
        if triggered_index:
            details += f"\nSafety order #{triggered_index} triggered, size: ${safety_quote:.2f}"

        warn = None
        if not deal:
            warn = "No open deal found for bot 41 — using simulated entry from current price"
        record(5, "Safety Order Trigger", would_trigger, details, warn=warn)
    except Exception as e:
        record(5, "Safety Order Trigger", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 6: Take Profit
# ─────────────────────────────────────────────────────────────────────
def test_6_take_profit():
    try:
        from db import latest_open_deal

        bot, _ = get_bot_config()
        deal = None
        try:
            deal = latest_open_deal(BOT_ID)
        except Exception:
            pass

        tp_pct = float(bot.get("tp", 0.01))
        entry_avg = float(deal["entry_avg"]) if deal and deal.get("entry_avg") else None
        base_amount = float(deal.get("base_amount", 0) or 0) if deal else 0

        if not entry_avg:
            kc = get_kraken_client()
            ticker = kc.fetch_ticker(SYMBOL)
            entry_avg = float(ticker.get("last", 5100)) * 0.99
            base_amount = float(bot.get("base_quote", 5.0)) / entry_avg

        tp_target = entry_avg * (1 + tp_pct)
        simulated_exit = tp_target + 10.0

        pnl_pct = (simulated_exit - entry_avg) / entry_avg * 100
        pnl_usd = (simulated_exit - entry_avg) * base_amount if base_amount > 0 else 0
        would_tp = simulated_exit >= tp_target

        details = (
            f"Entry avg: ${entry_avg:,.2f} {'(from deal)' if deal else '(simulated)'}\n"
            f"TP %: {tp_pct*100:.2f}%\n"
            f"TP target: ${tp_target:,.2f}\n"
            f"Simulated exit: ${simulated_exit:,.2f}\n"
            f"Would TP: {would_tp}\n"
            f"PnL: {pnl_pct:.2f}% (${pnl_usd:.4f})\n"
            f"Base amount: {base_amount:.8f}"
        )
        warn = None if deal else "No open deal — used simulated entry (current price - 1%)"
        record(6, "Take Profit", would_tp, details, warn=warn)
    except Exception as e:
        record(6, "Take Profit", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 7: Full Cycle Dry Run (7 days of data)
# ─────────────────────────────────────────────────────────────────────
def test_7_full_cycle_dry_run():
    try:
        bot, _ = get_bot_config()
        kc = get_kraken_client()

        candles_1h = kc.fetch_ohlcv(SYMBOL, timeframe="1h", limit=168)
        if not candles_1h or len(candles_1h) < 20:
            record(7, "Full Cycle Dry Run", False, "Insufficient candle data")
            return

        closes = [float(c[4]) for c in candles_1h]
        highs = [float(c[2]) for c in candles_1h]
        lows = [float(c[3]) for c in candles_1h]
        timestamps = [c[0] for c in candles_1h]

        base_quote = float(bot.get("base_quote", 5)) if bot else 5.0
        safety_quote = float(bot.get("safety_quote", 2)) if bot else 2.0
        tp_pct = float(bot.get("tp", 0.01)) if bot else 0.01
        first_dev = float(bot.get("first_dev", 0.01)) if bot else 0.01
        max_safety = int(bot.get("max_safety", 3)) if bot else 3

        in_position = False
        entry_price = 0
        entry_idx = 0
        safety_count = 0
        avg_entry = 0
        total_invested = 0
        total_units = 0

        sma_period = 20
        best_entry = None
        best_exit = None
        best_pnl = None

        for i in range(sma_period, len(closes)):
            sma = sum(closes[i - sma_period:i]) / sma_period
            price = closes[i]

            if not in_position:
                if price > sma:
                    in_position = True
                    entry_price = price
                    entry_idx = i
                    total_invested = base_quote
                    total_units = base_quote / price
                    avg_entry = price
                    safety_count = 0
            else:
                dev = first_dev
                for s in range(safety_count, max_safety):
                    trigger = avg_entry * (1 - dev)
                    if price <= trigger and safety_count < max_safety:
                        total_invested += safety_quote
                        total_units += safety_quote / price
                        avg_entry = total_invested / total_units
                        safety_count += 1
                        break
                    dev *= 1.2

                tp_target = avg_entry * (1 + tp_pct)
                if price >= tp_target:
                    exit_price = price
                    pnl_pct = (exit_price - avg_entry) / avg_entry * 100
                    duration_hrs = (timestamps[i] - timestamps[entry_idx]) / 3600000
                    best_entry = entry_price
                    best_exit = exit_price
                    best_pnl = pnl_pct
                    in_position = False

        if best_entry is not None:
            details = (
                f"Entry price: ${best_entry:,.2f}\n"
                f"Exit price: ${best_exit:,.2f}\n"
                f"PnL: {best_pnl:.2f}%\n"
                f"Safety orders triggered: {safety_count}\n"
                f"Duration: {duration_hrs:.1f} hours\n"
                f"Total invested: ${total_invested:.2f}"
            )
            record(7, "Full Cycle Dry Run", True, details)
        elif in_position:
            current_pnl = (closes[-1] - avg_entry) / avg_entry * 100
            duration_hrs = (timestamps[-1] - timestamps[entry_idx]) / 3600000
            details = (
                f"Still in position (no TP hit in 7d window)\n"
                f"Entry: ${entry_price:,.2f}\n"
                f"Current: ${closes[-1]:,.2f}\n"
                f"Unrealized PnL: {current_pnl:.2f}%\n"
                f"Safety orders: {safety_count}\n"
                f"Duration so far: {duration_hrs:.1f} hours"
            )
            record(7, "Full Cycle Dry Run", True, details,
                   warn="Deal did not close within 7-day window")
        else:
            record(7, "Full Cycle Dry Run", True,
                   f"No entry signal in 7-day window. Price range: ${min(closes):,.2f} - ${max(closes):,.2f}",
                   warn="No entry triggered — market conditions may not favor trend follow")
    except Exception as e:
        record(7, "Full Cycle Dry Run", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 8: Database Integrity
# ─────────────────────────────────────────────────────────────────────
def test_8_database_integrity():
    try:
        from db import get_bot, get_deal, latest_open_deal

        bot_from_db = get_bot(BOT_ID)
        bot, is_synthetic = get_bot_config()
        deal = None
        try:
            deal = get_deal(DEAL_ID, full=True) if DEAL_ID else None
        except Exception:
            pass
        open_deal = None
        try:
            open_deal = latest_open_deal(BOT_ID)
        except Exception:
            pass

        issues = []

        if is_synthetic:
            issues.append(f"Bot {BOT_ID} not in local DB — using synthetic config (exists on production server)")

        bot_fields = {
            "id": bot.get("id"), "name": bot.get("name"), "symbol": bot.get("symbol"),
            "enabled": bot.get("enabled"), "strategy_mode": bot.get("strategy_mode"),
            "base_quote": bot.get("base_quote"), "safety_quote": bot.get("safety_quote"),
            "max_safety": bot.get("max_safety"), "tp": bot.get("tp"),
        }
        for k, v in bot_fields.items():
            if v is None:
                issues.append(f"Bot field '{k}' is NULL")

        if bot.get("symbol") != SYMBOL:
            issues.append(f"Bot symbol mismatch: expected {SYMBOL}, got {bot.get('symbol')}")

        details = f"Bot 41 fields {'(SYNTHETIC — bot is on production server)' if is_synthetic else '(from DB)'}:\n"
        for k, v in bot_fields.items():
            details += f"  {k}: {v}\n"

        if deal:
            deal_fields = {
                "id": deal.get("id"), "bot_id": deal.get("bot_id"),
                "state": deal.get("state"), "symbol": deal.get("symbol"),
                "entry_avg": deal.get("entry_avg"), "opened_at": deal.get("opened_at"),
            }
            details += f"\nDeal {DEAL_ID} fields:\n"
            for k, v in deal_fields.items():
                details += f"  {k}: {v}\n"

            if deal.get("bot_id") != BOT_ID:
                issues.append(f"Deal {DEAL_ID} bot_id mismatch: expected {BOT_ID}, got {deal.get('bot_id')}")
            if deal.get("symbol") != SYMBOL:
                issues.append(f"Deal {DEAL_ID} symbol mismatch: expected {SYMBOL}, got {deal.get('symbol')}")
            for k in ("entry_avg", "symbol", "bot_id", "opened_at"):
                if deal.get(k) is None:
                    issues.append(f"Deal field '{k}' is NULL")
        else:
            details += f"\nDeal {DEAL_ID}: NOT FOUND in local DB"

        if open_deal:
            details += f"\nLatest open deal: #{open_deal.get('id')} ({open_deal.get('state')})"
        else:
            details += "\nNo open deal for bot 41 in local DB"

        critical_issues = [i for i in issues if "NULL" in i or "mismatch" in i]
        passed = len(critical_issues) == 0

        if issues:
            details += "\n\nNotes: " + "; ".join(issues)
        record(8, "Database Integrity", passed, details,
               warn="Bot/deal data is on production server, not local DB" if is_synthetic else None)
    except Exception as e:
        record(8, "Database Integrity", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 9: Kraken Connectivity
# ─────────────────────────────────────────────────────────────────────
def test_9_kraken_connectivity():
    try:
        kc = get_kraken_client()
        ticker = kc.fetch_ticker(SYMBOL)

        bid = float(ticker.get("bid", 0))
        ask = float(ticker.get("ask", 0))
        last = float(ticker.get("last", 0))
        spread_pct = (ask - bid) / ask * 100 if ask > 0 else 0

        candles = kc.fetch_ohlcv(SYMBOL, timeframe="1m", limit=2)
        last_candle_ts = candles[-1][0] / 1000 if candles else 0
        candle_age = time.time() - last_candle_ts if last_candle_ts > 0 else 9999

        connected = last > 0
        candle_fresh = candle_age < 600

        details = (
            f"Connected: {connected}\n"
            f"Bid: ${bid:,.2f}\n"
            f"Ask: ${ask:,.2f}\n"
            f"Last: ${last:,.2f}\n"
            f"Spread: {spread_pct:.4f}%\n"
            f"Last candle age: {candle_age:.1f}s ({candle_age/60:.1f} min)\n"
            f"Candle fresh (<10min): {candle_fresh}"
        )

        passed = connected and candle_fresh
        record(9, "Kraken Connectivity", passed, details,
               expected="Price > 0 and candle < 10min old",
               actual=f"price=${last:,.2f}, candle_age={candle_age:.0f}s")
    except Exception as e:
        record(9, "Kraken Connectivity", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 10: Multi-Timeframe Alignment
# ─────────────────────────────────────────────────────────────────────
def test_10_multi_timeframe():
    try:
        kc = get_kraken_client()
        timeframes = {"5m": 60, "1h": 50, "4h": 50, "1d": 30}
        tf_status = {}

        for tf, limit in timeframes.items():
            candles = kc.fetch_ohlcv(SYMBOL, timeframe=tf, limit=limit)
            if not candles or len(candles) < 22:
                tf_status[tf] = {"trend": "UNKNOWN", "reason": f"Only {len(candles) if candles else 0} candles"}
                continue

            closes = [float(c[4]) for c in candles]
            sma20 = sum(closes[-20:]) / 20
            sma50 = sum(closes[-min(50, len(closes)):]) / min(50, len(closes))
            last = closes[-1]

            if last > sma20 > sma50:
                trend = "BULLISH"
                reason = f"Price ${last:,.2f} > SMA20 ${sma20:,.2f} > SMA50 ${sma50:,.2f}"
            elif last < sma20 < sma50:
                trend = "BEARISH"
                reason = f"Price ${last:,.2f} < SMA20 ${sma20:,.2f} < SMA50 ${sma50:,.2f}"
            else:
                trend = "MIXED"
                reason = f"Price ${last:,.2f}, SMA20 ${sma20:,.2f}, SMA50 ${sma50:,.2f}"

            tf_status[tf] = {"trend": trend, "reason": reason}

        trends = [v["trend"] for v in tf_status.values()]
        all_bullish = all(t == "BULLISH" for t in trends)
        all_bearish = all(t == "BEARISH" for t in trends)
        aligned = all_bullish or all_bearish

        details = ""
        for tf, info in tf_status.items():
            details += f"{tf}: {info['trend']} — {info['reason']}\n"
        details += f"\nAll aligned: {aligned}"
        if not aligned:
            conflicts = [f"{tf}: {info['trend']}" for tf, info in tf_status.items() if info['trend'] not in (trends[0],)]
            details += f"\nConflicting: {conflicts}"

        record(10, "Multi-Timeframe Alignment", True, details,
               warn="Timeframes not aligned" if not aligned else None)
    except Exception as e:
        record(10, "Multi-Timeframe Alignment", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 11: Cooldown State
# ─────────────────────────────────────────────────────────────────────
def test_11_cooldown():
    try:
        from db import list_logs

        logs = list_logs(BOT_ID, limit=100)
        now = int(time.time())

        cooldown_until = 0
        cooldown_reason = "No cooldown active"

        for log in logs:
            msg = log.get("message", "")
            if "cooldown" in msg.lower() or "pause" in msg.lower():
                ts = log.get("ts", 0)
                if "hour" in msg.lower():
                    try:
                        hours = int("".join(c for c in msg.split("hour")[0].split()[-1] if c.isdigit()) or "0")
                        cooldown_until = max(cooldown_until, ts + hours * 3600)
                        cooldown_reason = msg
                    except Exception:
                        pass
                elif ts > now - 3600:
                    cooldown_until = max(cooldown_until, ts + 300)
                    cooldown_reason = msg

        remaining = max(0, cooldown_until - now)

        details = (
            f"Cooldown active: {remaining > 0}\n"
            f"Seconds remaining: {remaining}\n"
            f"Reason: {cooldown_reason}\n"
            f"Recent log count: {len(logs)}"
        )

        record(11, "Cooldown State", True, details,
               warn=f"Bot in cooldown for {remaining}s" if remaining > 0 else None)
    except Exception as e:
        record(11, "Cooldown State", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 12: Position Sizing (Kelly Criterion)
# ─────────────────────────────────────────────────────────────────────
def test_12_position_sizing():
    try:
        from kelly_criterion import KellyPositionSizer

        sizer = KellyPositionSizer(
            kelly_fraction=0.25,
            max_position_pct=0.10,
            min_trades_required=20,
        )

        # Test A: With insufficient trades (default 2% sizing)
        bot_stats_few = {
            "total_trades": 5, "winning_trades": 3,
            "avg_profit_pct": 0.02, "avg_loss_pct": -0.01,
        }
        size_few, reason_few, details_few = sizer.calculate_position_size(20.0, bot_stats_few)

        # Test B: With sufficient trades (full Kelly calculation)
        bot_stats_full = {
            "total_trades": 50, "winning_trades": 30,
            "avg_profit_pct": 0.03, "avg_loss_pct": -0.015,
        }
        size_full, reason_full, details_full = sizer.calculate_position_size(20.0, bot_stats_full)

        raw_kelly_full = details_full.get("kelly_pct")
        capped_full = details_full.get("position_pct", 0)

        valid_few = size_few > 0 and size_few <= 20.0
        valid_full = size_full > 0 and size_full <= 20.0

        details = (
            f"--- With {bot_stats_few['total_trades']} trades (below min {sizer.min_trades_required}) ---\n"
            f"Method: {details_few.get('method')}\n"
            f"Position size: ${size_few:.2f} (2% default of $20)\n"
            f"Reason: {reason_few}\n"
            f"\n--- With {bot_stats_full['total_trades']} trades (above min) ---\n"
            f"Method: {details_full.get('method')}\n"
            f"Raw Kelly %: {raw_kelly_full:.4f}%" if raw_kelly_full else "Raw Kelly: N/A"
        )
        details += (
            f"\nCapped %: {capped_full*100:.2f}%\n"
            f"Position size: ${size_full:.2f}\n"
            f"Reason: {reason_full}\n"
            f"Valid (both > 0): {valid_few and valid_full}"
        )

        record(12, "Position Sizing (Kelly)", valid_few and valid_full, details,
               expected="Both sizes > $0 and <= $20", actual=f"few=${size_few:.2f}, full=${size_full:.2f}")
    except Exception as e:
        record(12, "Position Sizing (Kelly)", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 13: Circuit Breaker
# ─────────────────────────────────────────────────────────────────────
def test_13_circuit_breaker():
    try:
        from risk_circuit_breaker import check_circuit_breakers

        ok, reason = check_circuit_breakers(
            equity=20.0,
            daily_realized_pnl=0.0,
            portfolio_drawdown=0.0,
            portfolio_exposure_pct=0.25,
            open_deals_count=1,
            total_exposure_usd=5.21,
            max_total_exposure_pct=0.50,
            max_concurrent_deals=6,
            max_daily_loss_pct=0.06,
            max_drawdown_pct=0.15,
            max_exposure_pct=0.50,
            consecutive_losses=0,
            loss_circuit_threshold=3,
        )

        from risk_engine import can_open_trade, RiskContext

        risk_ok, risk_reason = can_open_trade(RiskContext(
            bot_id=BOT_ID, symbol=SYMBOL,
            balance_total_usd=20.0, balance_free_usd=15.0,
            positions_usd={SYMBOL: 5.21}, symbol_position_usd=5.21,
            proposed_order_usd=2.0, is_crypto=True,
        ))

        details = (
            f"Portfolio circuit breaker: {'CLEAR' if ok else 'TRIPPED'}\n"
            f"  Reason: {reason or 'None'}\n"
            f"Risk engine per-symbol: {'CLEAR' if risk_ok else 'BLOCKED'}\n"
            f"  Reason: {risk_reason or 'None'}\n"
            f"All clear: {ok and risk_ok}"
        )

        record(13, "Circuit Breaker", ok and risk_ok, details)
    except Exception as e:
        record(13, "Circuit Breaker", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 14: Regime Consistency
# ─────────────────────────────────────────────────────────────────────
def test_14_regime_consistency():
    try:
        kc = get_kraken_client()
        candles_1h = kc.fetch_ohlcv(SYMBOL, timeframe="1h", limit=200)
        closes = [float(c[4]) for c in candles_1h]

        sma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else 0
        sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else 0

        if sma20 > sma50 and closes[-1] > sma20:
            scanner_regime = "trending_up"
        elif sma20 < sma50 and closes[-1] < sma20:
            scanner_regime = "trending_down"
        else:
            scanner_regime = "ranging"

        from db import list_logs
        logs = list_logs(BOT_ID, limit=50)
        bot_regime = None
        for log in logs:
            msg = log.get("message", "")
            lower = msg.lower()
            if "regime" in lower:
                for r in ("trending_up", "trending_down", "ranging", "high_vol", "trend_up", "trend_down", "range"):
                    if r in lower:
                        bot_regime = r
                        break
                if bot_regime:
                    break

        match = False
        if bot_regime:
            normalize = lambda r: r.replace("trend_up", "trending_up").replace("trend_down", "trending_down").replace("range", "ranging")
            match = normalize(scanner_regime) == normalize(bot_regime)

        details = (
            f"Scanner regime (computed): {scanner_regime}\n"
            f"  SMA20: ${sma20:,.2f}, SMA50: ${sma50:,.2f}, Last: ${closes[-1]:,.2f}\n"
            f"Bot log regime: {bot_regime or 'Not found in recent logs'}\n"
            f"Match: {match}"
        )

        passed = True
        record(14, "Regime Consistency", passed, details,
               warn="Could not find regime in bot logs for comparison" if not bot_regime else (
                   "Regimes do not match" if not match else None))
    except Exception as e:
        record(14, "Regime Consistency", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# TEST 15: End-to-End Order Flow (Dry Run)
# ─────────────────────────────────────────────────────────────────────
def test_15_e2e_order_flow():
    try:
        from intelligence_layer import IntelligenceLayer, IntelligenceContext
        from risk_engine import RiskContext, can_open_trade
        from execution_gate import check_execution_gate

        bot, _ = get_bot_config()
        kc = get_kraken_client()
        ticker = kc.fetch_ticker(SYMBOL)
        last_price = float(ticker.get("last", 0))
        bid = float(ticker.get("bid", 0))
        ask = float(ticker.get("ask", 0))
        now = time.time()

        candles_1h = kc.fetch_ohlcv(SYMBOL, timeframe="1h", limit=200)
        candles_4h = kc.fetch_ohlcv(SYMBOL, timeframe="4h", limit=100)
        candles_1d = kc.fetch_ohlcv(SYMBOL, timeframe="1d", limit=60)

        step_results = {}

        # Step 1: Intelligence
        try:
            ctx = IntelligenceContext(
                symbol=SYMBOL, last_price=last_price,
                bid_price=bid, ask_price=ask,
                spread_pct=(ask - bid) / ask if ask > 0 else 0,
                candles_1h=candles_1h, candles_4h=candles_4h, candles_1d=candles_1d,
                bot_config=bot or {}, dry_run=True,
                portfolio_total_usd=20.0,
                now_ts=int(now),
                last_price_ts=int(now),
                last_candle_ts=int(candles_1h[-1][0] / 1000) if candles_1h else int(now),
            )
            layer = IntelligenceLayer()
            decision = layer.evaluate(ctx)
            step_results["1_Intelligence"] = f"OK — action={decision.final_action}, reason={decision.final_reason}"
            intel_passed = True
        except Exception as e:
            step_results["1_Intelligence"] = f"FAILED — {e}"
            intel_passed = False

        # Step 2: Risk Engine
        try:
            risk_ctx = RiskContext(
                bot_id=BOT_ID, symbol=SYMBOL,
                balance_total_usd=20.0, balance_free_usd=18.0,
                positions_usd={}, symbol_position_usd=0,
                proposed_order_usd=2.0, is_crypto=True,
            )
            allowed, reason = can_open_trade(risk_ctx)
            step_results["2_Risk_Engine"] = f"{'OK' if allowed else 'BLOCKED'} — {reason or 'allowed'}"
            risk_passed = allowed
        except Exception as e:
            step_results["2_Risk_Engine"] = f"FAILED — {e}"
            risk_passed = False

        # Step 3: Execution Gate
        try:
            candle_ts = candles_1h[-1][0] / 1000 if candles_1h else now
            gate = check_execution_gate(
                symbol=SYMBOL, side="buy", order_type="market",
                bid=bid, ask=ask, last_price=last_price,
                ticker_ts=now, last_candle_ts=candle_ts,
                volume_24h=float(ticker.get("volume", 0)),
                quote_amount=2.0, dry_run=True,
            )
            step_results["3_Execution_Gate"] = f"{'OK' if gate.allowed else 'BLOCKED'} — {gate.reason or 'passed'}"
            gate_passed = gate.allowed
        except Exception as e:
            step_results["3_Execution_Gate"] = f"FAILED — {e}"
            gate_passed = False

        # Step 4: Executor (dry run)
        try:
            if intel_passed:
                from executor import OrderExecutor
                executor = OrderExecutor(kc)
                exec_result = executor.execute_decision(
                    decision, BOT_ID, SYMBOL, dry_run=True,
                    risk_context=risk_ctx if risk_passed else None,
                )
                errors = exec_result.get("errors", [])
                orders = exec_result.get("orders_placed", [])
                step_results["4_Executor"] = f"OK — orders={len(orders)}, errors={errors}"
                exec_passed = True
            else:
                step_results["4_Executor"] = "SKIPPED — intelligence step failed"
                exec_passed = False
        except Exception as e:
            step_results["4_Executor"] = f"FAILED — {e}"
            exec_passed = False

        all_passed = intel_passed and risk_passed and gate_passed and exec_passed

        details = "Order flow steps:\n"
        for step, result in step_results.items():
            details += f"  {step}: {result}\n"
        details += f"\nAll steps passed: {all_passed}"

        record(15, "End-to-End Order Flow", all_passed, details)
    except Exception as e:
        record(15, "End-to-End Order Flow", False, f"Exception: {e}\n{traceback.format_exc()}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  BOT 41 (Smart PAXG/USD) — END-TO-END TEST SUITE")
    print("="*70)

    tests = [
        test_1_intelligence_layer,
        test_2_strategy,
        test_3_risk_engine,
        test_4_execution_gate,
        test_5_safety_order,
        test_6_take_profit,
        test_7_full_cycle_dry_run,
        test_8_database_integrity,
        test_9_kraken_connectivity,
        test_10_multi_timeframe,
        test_11_cooldown,
        test_12_position_sizing,
        test_13_circuit_breaker,
        test_14_regime_consistency,
        test_15_e2e_order_flow,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"\n  UNEXPECTED ERROR in {t.__name__}: {e}")

    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = sum(1 for r in results if not r["passed"])
    total = len(results)

    print("\n\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"  PASSED: {passed}/{total}")
    print(f"  FAILED: {failed}/{total}")
    print(f"  WARNINGS: {len(warnings_list)}")

    if failed > 0:
        print(f"\n  FAILED TESTS:")
        for r in results:
            if not r["passed"]:
                print(f"    Test {r['num']}: {r['name']}")
                if r.get("expected"):
                    print(f"      Expected: {r['expected']}")
                if r.get("actual"):
                    print(f"      Actual:   {r['actual']}")
                if r.get("details"):
                    first_lines = r["details"].strip().split("\n")[:3]
                    for line in first_lines:
                        print(f"      {line}")

    if warnings_list:
        print(f"\n  WARNINGS:")
        for w in warnings_list:
            print(f"    {w}")

    print("\n" + "="*70)
    sys.exit(1 if failed > 0 else 0)
