# bot_manager.py  (REPLACE ENTIRE FILE)
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

from db import (
    add_log,
    latest_open_deal,
    open_deal,
    close_deal,
    get_bot,
    pnl_summary,
    set_bot_running,
    get_setting,
    set_setting,
    add_regime_snapshot,
    add_strategy_decision,
    get_expected_edge,
    add_strategy_trade,
    add_order_event,
    add_intelligence_decision,
    update_bot,
    list_deals,
    list_strategy_decisions,
    get_intelligence_decisions,
    get_symbols_with_open_deals_excluding,
    count_orders_today,
    bot_performance_stats,
)
from alpaca_adapter import AlpacaAdapter
from kraken_client import KrakenClient
from alpaca_client import AlpacaClient
from strategies import (
    DcaConfig,
    safety_levels,
    trend_ok,
    SmartDcaConfig,
    build_smart_dca_config,
    DealState,
    AccountSnapshot,
    PerformanceStats,
    smart_decide,
    detect_regime,
    get_strategy,
    select_strategy,
    dominant_regime,
    StrategyContext,
    _atr,
)
from intelligence_layer import (
    IntelligenceLayer,
    IntelligenceContext,
    IntelligenceDecision,
    AllowedAction,
)

# Check Phase availability
try:
    from intelligence_layer import PHASE1_AVAILABLE, PHASE2_AVAILABLE, PHASE3_AVAILABLE
except ImportError:
    PHASE1_AVAILABLE = False
    PHASE2_AVAILABLE = False
    PHASE3_AVAILABLE = False
from executor import OrderExecutor
import json
from symbol_classifier import classify_symbol, is_stock_symbol

try:
    from portfolio_correlation import PortfolioCorrelationAnalyzer
    CORRELATION_AVAILABLE = True
except ImportError:
    CORRELATION_AVAILABLE = False
    PortfolioCorrelationAnalyzer = None

try:
    from risk_circuit_breaker import check_circuit_breakers, trip_and_alert
except ImportError:
    check_circuit_breakers = None
    trip_and_alert = None

try:
    from order_book_analyzer import OrderBookAnalyzer
    ORDER_BOOK_AVAILABLE = True
except ImportError:
    ORDER_BOOK_AVAILABLE = False
    OrderBookAnalyzer = None


# =========================
# Risk / Stability Controls
# =========================
MAX_ERRORS_BEFORE_HALT = int(os.getenv("MAX_ERRORS_BEFORE_HALT", "5"))

# Global default daily loss kill-switch (quote units, usually USD)
MAX_DAILY_LOSS_QUOTE = float(os.getenv("MAX_DAILY_LOSS_QUOTE", "25.0"))

# Order book slippage gating (per bot or env)
MAX_SLIPPAGE_PCT_MAJORS = float(os.getenv("MAX_SLIPPAGE_PCT_MAJORS", "0.40")) / 100.0
MAX_SLIPPAGE_PCT_ALTS = float(os.getenv("MAX_SLIPPAGE_PCT_ALTS", "0.80")) / 100.0
MIN_LIQUIDITY_MULT = float(os.getenv("MIN_LIQUIDITY_MULT", "3.0"))

# Stop join timeout
STOP_JOIN_TIMEOUT_SEC = int(os.getenv("STOP_JOIN_TIMEOUT_SEC", "10"))

# Backoff on errors (seconds)
BACKOFF_MIN_SEC = float(os.getenv("BACKOFF_MIN_SEC", "2"))
BACKOFF_MAX_SEC = float(os.getenv("BACKOFF_MAX_SEC", "30"))

# Live trading guardrail:
# - If a bot is set to dry_run=0 (LIVE), this env must be TRUE to allow real orders.
# - This lets you configure bots as LIVE in UI but still block execution until you’re confident.
ALLOW_LIVE_TRADING = os.getenv("ALLOW_LIVE_TRADING", "0").strip().lower() in ("1", "true", "yes", "y", "on")

# ATR-based position sizing: scale order size down when volatility is high
ATR_POSITION_SIZING_ENABLED = os.getenv("ATR_POSITION_SIZING_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")
# Correlation check: block new entries when portfolio is over-exposed to correlated assets
CORRELATION_CHECK_ENABLED = os.getenv("CORRELATION_CHECK_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")
# Adaptive take-profit: scale TP by volatility (higher vol = larger TP target)
ADAPTIVE_TP_ENABLED = os.getenv("ADAPTIVE_TP_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")
ADAPTIVE_TP_MIN = float(os.getenv("ADAPTIVE_TP_MIN_PCT", "1.0")) / 100.0
ADAPTIVE_TP_MAX = float(os.getenv("ADAPTIVE_TP_MAX_PCT", "10.0")) / 100.0
ADAPTIVE_TP_ATR_MULT = float(os.getenv("ADAPTIVE_TP_ATR_MULT", "1.5"))


def _today_start_ts_local() -> int:
    lt = time.localtime()
    return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, lt.tm_wday, lt.tm_yday, lt.tm_isdst)))


def _extract_realized_pnl(pnl_obj: Any) -> float:
    """
    db.pnl_summary() returns dict with 'realized' in your db.py.
    Keep defensive compatibility with other keys.
    """
    if not isinstance(pnl_obj, dict):
        return 0.0
    for k in ("realized", "realized_pnl", "pnl", "net", "total"):
        if k in pnl_obj and pnl_obj[k] is not None:
            try:
                return float(pnl_obj[k])
            except Exception:
                pass
    return 0.0


def _check_order_book_market(
    kc: Any,
    symbol: str,
    side: str,
    size_quote: float,
    is_major: bool = False,
) -> Tuple[bool, float, Optional[str]]:
    """
    Check order book before market order. Returns (allowed, effective_size_quote, reason).
    If slippage/liquidity bad: returns (False, size_quote, reason) - caller should block or reduce.
    """
    if not ORDER_BOOK_AVAILABLE or OrderBookAnalyzer is None:
        return True, size_quote, None
    if not hasattr(kc, "fetch_order_book"):
        return True, size_quote, None
    try:
        ob = kc.fetch_order_book(symbol, 50)
        if not ob or (not ob.get("bids") and not ob.get("asks")):
            return True, size_quote, None
        max_slip = MAX_SLIPPAGE_PCT_MAJORS if is_major else MAX_SLIPPAGE_PCT_ALTS
        analyzer = OrderBookAnalyzer(max_slippage_pct=max_slip, min_liquidity_mult=MIN_LIQUIDITY_MULT)
        ok, slippage_pct, reason, details = analyzer.analyze_liquidity(ob, side, size_quote)
        if ok:
            return True, size_quote, f"slippage {slippage_pct*100:.2f}%"
        # Reduce size if insufficient liquidity; block if slippage too high
        if "slippage" in reason.lower():
            return False, size_quote, reason
        avail = details.get("cumulative_volume_quote", 0.0) or 0.0
        if avail > 0 and avail < size_quote:
            reduced = max(0, avail / max(MIN_LIQUIDITY_MULT, 1.5))
            return True, reduced, f"reduced to ${reduced:.2f} (liquidity)"
        return False, size_quote, reason
    except Exception as e:
        return True, size_quote, None  # On error, allow (don't block)


def _atr_scaled_quote(quote_amt: float, symbol: str, kc: Any, min_mult: float = 0.5) -> Tuple[float, Optional[str]]:
    """
    Scale order size by volatility (ATR). Higher vol -> smaller size.
    Returns (scaled_amt, reason_or_None).
    """
    if not ATR_POSITION_SIZING_ENABLED or quote_amt <= 0:
        return quote_amt, None
    try:
        ohlcv = kc.fetch_ohlcv(symbol, timeframe="1d", limit=20)
        if not ohlcv or len(ohlcv) < 15:
            return quote_amt, None
        atr_val = _atr(ohlcv, 14)
        last_price = float(ohlcv[-1][4]) if ohlcv else 0.0
        if not atr_val or not last_price or last_price <= 0:
            return quote_amt, None
        atr_pct = atr_val / last_price
        # mult = 1 / (1 + k*atr_pct): atr 2% -> 0.94, 5% -> 0.83, 10% -> 0.71
        mult = 1.0 / (1.0 + 3.0 * atr_pct)
        mult = max(min_mult, min(1.0, mult))
        scaled = round(quote_amt * mult, 2)
        if scaled < 1.0:
            return quote_amt, None  # Don't scale below $1
        reason = f"ATR {atr_pct*100:.2f}% -> size x{mult:.2f}" if mult < 0.98 else None
        return scaled, reason
    except Exception:
        return quote_amt, None


def _adaptive_tp(base_tp: float, symbol: str, kc: Any) -> Tuple[float, Optional[str]]:
    """
    Scale take-profit by volatility. Higher vol -> larger TP target.
    Returns (effective_tp_fraction, reason_or_None).
    """
    if not ADAPTIVE_TP_ENABLED or base_tp <= 0:
        return base_tp, None
    try:
        ohlcv = kc.fetch_ohlcv(symbol, timeframe="1d", limit=20)
        if not ohlcv or len(ohlcv) < 15:
            return base_tp, None
        atr_val = _atr(ohlcv, 14)
        last_price = float(ohlcv[-1][4]) if ohlcv else 0.0
        if not atr_val or not last_price or last_price <= 0:
            return base_tp, None
        atr_pct = atr_val / last_price
        # TP = max(user_base, min(max_cap, atr_mult * atr_pct))
        vol_based = ADAPTIVE_TP_ATR_MULT * atr_pct
        effective = max(base_tp, min(ADAPTIVE_TP_MAX, max(ADAPTIVE_TP_MIN, vol_based)))
        if abs(effective - base_tp) < 0.001:
            return base_tp, None
        reason = f"ATR {atr_pct*100:.2f}% -> TP {effective*100:.2f}% (base {base_tp*100:.2f}%)"
        return effective, reason
    except Exception:
        return base_tp, None


def _check_correlation_allowed(symbol: str, bot_id: int, kc: Any) -> Tuple[bool, str]:
    """Check if adding this symbol would over-concentrate in correlated assets."""
    if not CORRELATION_CHECK_ENABLED or not CORRELATION_AVAILABLE:
        return True, ""
    existing = get_symbols_with_open_deals_excluding(bot_id)
    if not existing:
        return True, ""
    try:
        analyzer = PortfolioCorrelationAnalyzer(high_correlation_threshold=0.75)
        symbols_to_fetch = list(set(existing + [symbol]))
        price_history: Dict[str, List[float]] = {}
        for sym in symbols_to_fetch:
            try:
                ohlcv = kc.fetch_ohlcv(sym, timeframe="1d", limit=31)
                if ohlcv and len(ohlcv) >= 14:
                    price_history[sym] = [float(c[4]) for c in ohlcv]
            except Exception:
                pass
        allowed, reason, _ = analyzer.should_add_symbol(symbol, existing, price_history)
        if not allowed:
            return False, reason
        return True, ""
    except Exception:
        return True, ""  # On error, allow (fail open)


def _try_resolve_symbol(markets: Dict[str, Any], symbol: str) -> Optional[str]:
    """
    Resolve symbol to an existing ccxt market key.
    Tries BTC<->XBT swap (Kraken commonly uses XBT).
    """
    if not markets or not symbol:
        return None
    if symbol in markets:
        return symbol
    if symbol.startswith("BTC/"):
        alt = symbol.replace("BTC/", "XBT/", 1)
        if alt in markets:
            return alt
    if symbol.startswith("XBT/"):
        alt = symbol.replace("XBT/", "BTC/", 1)
        if alt in markets:
            return alt
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return int(default)


def _safe_enum_val(x: Any) -> str:
    """Use for regime/allowed_actions etc. Handles both Enum and str (e.g. strategies.RegimeResult.regime)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return x.value if hasattr(x, "value") else str(x)


@dataclass
class RuntimeState:
    running: bool = False
    last_price: Optional[float] = None

    avg_entry: Optional[float] = None
    base_pos: float = 0.0

    safety_used: int = 0
    spent_quote: float = 0.0

    tp_price: Optional[float] = None
    tp_order_id: Optional[str] = None

    deal_id: Optional[int] = None
    deal_opened_at: Optional[int] = None  # seconds since epoch
    entry_regime: Optional[str] = None
    entry_strategy: Optional[str] = None
    mae: Optional[float] = None
    mfe: Optional[float] = None

    last_event: str = "Idle"
    errors: int = 0
    decision_action: Optional[str] = None
    decision_reason: Optional[str] = None
    cooldown_until: int = 0
    last_tick_ts: int = 0
    active_strategy: Optional[str] = None
    regime_label: Optional[str] = None
    regime_confidence: Optional[float] = None
    regime_scores: Optional[Dict[str, float]] = None
    risk_state: Optional[str] = None
    forced_strategy: Optional[str] = None
    consecutive_losses: int = 0
    # Trailing stop (Part 1 profit optimization)
    trailing_active: bool = False
    trailing_price: Optional[float] = None
    highest_price_reached: Optional[float] = None
    # Partial profit taking (Part 2)
    partial_initial_position: Optional[float] = None
    partial_levels_hit: List[float] = field(default_factory=list)
    # Swing: multi-day scale-in
    scale_in_tranche_index: int = 0
    scale_in_last_add_ts: Optional[int] = None


class BotRunner:
    def __init__(self, bot_id: int, kc: Any, manager: Optional["BotManager"] = None):
        """
        Initialize BotRunner with a trading client.
        kc can be KrakenClient (crypto) or AlpacaAdapter (stocks).
        """
        self.bot_id = int(bot_id)
        self.kc = kc  # Can be KrakenClient or AlpacaAdapter
        self.manager = manager
        self.intelligence_layer = IntelligenceLayer()
        self.executor = OrderExecutor(kc)

        self.state = RuntimeState()
        self._lock = threading.RLock()  # RLock for thread-safe state transitions (running/enabled)

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._last_reason_key: Optional[str] = None
        self._last_reason_msg: Optional[str] = None
        self._cooldown_until: int = 0
        self._last_decision_action: Optional[str] = None
        self._last_decision_reason: Optional[str] = None
        self._last_tp_update_ts: float = 0.0
        self._equity_peak: float = 0.0
        self._last_discord_ts: float = 0.0
        self._last_discord_msg: Optional[str] = None
        self._last_insufficient_notify_ts: float = 0.0
        self._last_log_msg: Optional[str] = None
        self._last_log_level: Optional[str] = None
        self._last_log_category: Optional[str] = None
        self._repeat_count: int = 0
        self._last_log_ts: float = 0.0
        self._last_wait_reason: Optional[str] = None
        self._active_strategy: str = "smart_dca"
        self._strategy_last_switch_ts: int = 0
        self._grid_orders: Dict[str, Any] = {}
        self._last_regime: Optional[Dict[str, Any]] = None
        self._last_order_sig: Optional[str] = None
        self._pending_partial: Optional[Tuple[float, float, str]] = None
        self._dry_run_safety_used: int = 0
        self._last_safety_buy_ts: float = 0.0
        self._last_order_ts: int = 0
        self._regime_selected: Optional[str] = None
        self._regime_candidate: Optional[str] = None
        self._regime_candidate_ticks: int = 0
        self._regime_history: List[str] = []
        
        # Initialize Intelligence Layer and Executor
        self.intelligence_layer = IntelligenceLayer()
        self.executor = OrderExecutor(kc)

    # -----------------
    # State + logging
    # -----------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "running": self.state.running,
                "last_price": self.state.last_price,
                "avg_entry": self.state.avg_entry,
                "base_pos": self.state.base_pos,
                "safety_used": self.state.safety_used,
                "spent_quote": self.state.spent_quote,
                "tp_price": self.state.tp_price,
                "tp_order_id": self.state.tp_order_id,
                "deal_id": self.state.deal_id,
                "last_event": self.state.last_event,
                "errors": self.state.errors,
                "decision_action": self.state.decision_action,
                "decision_reason": self.state.decision_reason,
                "cooldown_until": self.state.cooldown_until,
                "last_tick_ts": self.state.last_tick_ts,
                "active_strategy": self.state.active_strategy,
                "regime_label": self.state.regime_label,
                "regime_confidence": self.state.regime_confidence,
                "regime_scores": self.state.regime_scores,
                "risk_state": self.state.risk_state,
                "forced_strategy": self.state.forced_strategy,
            }

    def _log(self, msg: str, level: str = "INFO", category: str = "SYSTEM") -> None:
        now = time.time()
        if msg == self._last_log_msg and level == self._last_log_level and category == self._last_log_category:
            self._repeat_count += 1
            if now - self._last_log_ts < 30.0:
                return
            msg = f"{msg} (x{self._repeat_count})"
        else:
            self._repeat_count = 0
        self._last_log_msg = msg
        self._last_log_level = level
        self._last_log_category = category
        self._last_log_ts = now
        add_log(self.bot_id, level, msg, category)

    def _set(self, msg: str, level: str = "INFO", category: str = "SYSTEM") -> None:
        with self._lock:
            self.state.last_event = msg
        self._log(msg, level, category)
        if str(level).upper() == "ERROR" or str(category).upper() == "RISK":
            # Throttle "insufficient funds" type messages to once per hour to avoid spam
            msg_lower = (msg or "").lower()
            is_insufficient = "insufficient" in msg_lower or "no funds" in msg_lower or "available" in msg_lower and "waiting" not in msg_lower
            if is_insufficient:
                now = time.time()
                if now - self._last_insufficient_notify_ts < 3600.0:  # 1 hour
                    return
                self._last_insufficient_notify_ts = now
            self._notify_discord(f"⚠️ {self._bot_label()}: {msg}")

    def _heartbeat(self) -> None:
        with self._lock:
            self.state.last_tick_ts = int(time.time())

    def _global_pause_on(self) -> bool:
        env = os.getenv("PAUSE_ALL_BOTS", "").strip().lower()
        if env in ("1", "true", "yes", "y", "on"):
            return True
        try:
            v = get_setting("global_pause", "0")
            if str(v).strip().lower() in ("1", "true", "yes", "y", "on"):
                until = get_setting("global_pause_until", "0")
                try:
                    until_ts = int(until or 0)
                except Exception:
                    until_ts = 0
                if until_ts and until_ts <= int(time.time()):
                    set_setting("global_pause", "0")
                    set_setting("global_pause_until", "0")
                    return False
                return True
        except Exception:
            pass
        return False

    def _notify_discord(self, message: str, trade_event: bool = False, force: bool = False) -> None:
        if os.getenv("DISCORD_OFF", "0").strip().lower() in ("1", "true", "yes"):
            return
        webhook = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
        if not webhook:
            return
        # DISCORD_TRADES_ONLY=1: only send trades + lifecycle (start/stop/blocked). force=True bypasses.
        trades_only = os.getenv("DISCORD_TRADES_ONLY", "1").strip().lower() in ("1", "true", "yes")
        if trades_only and not trade_event and not force:
            return
        now = time.time()
        if self._last_discord_msg == message and (now - self._last_discord_ts) < 60.0:
            return
        self._last_discord_msg = message
        self._last_discord_ts = now
        try:
            import requests
            requests.post(webhook, json={"content": message}, timeout=3)
        except Exception:
            pass

    def _bot_label(self) -> str:
        try:
            b = get_bot(self.bot_id)
            if b:
                name = b.get("name") or self.bot_id
                symbol = b.get("symbol") or ""
                return f"{name} {symbol}".strip()
        except Exception:
            pass
        return f"Bot {self.bot_id}"

    def _set_once(self, key: str, msg: str, level: str = "INFO", category: str = "STRATEGY") -> None:
        if self._last_reason_key == key and self._last_reason_msg == msg:
            with self._lock:
                self.state.last_event = msg
            return
        self._last_reason_key = key
        self._last_reason_msg = msg
        self._set(msg, level, category)

    def _status(self, msg: str) -> None:
        with self._lock:
            self.state.last_event = msg
    
    def _build_intelligence_context(
        self,
        symbol: str,
        price: float,
        candles_1h: List[List[float]],
        candles_4h: List[List[float]],
        candles_1d: List[List[float]],
        candles_1w: List[List[float]],
        bot: Dict[str, Any],
        account: AccountSnapshot,
        perf: PerformanceStats,
        deal_state: DealState,
        now_ts: int,
    ) -> IntelligenceContext:
        """Build IntelligenceContext from bot state."""
        # Get BTC context (regime + correlation)
        btc_context = {"risk_off": False, "labels": {}, "scores": {}, "corr": 0.0, "regime": None}
        try:
            btc_sym = "XBT/USD"
            btc_1h = self.manager.ohlcv_cached(btc_sym, "1h", limit=220, ttl_sec=120) if self.manager else []
            btc_4h = self.manager.ohlcv_cached(btc_sym, "4h", limit=220, ttl_sec=300) if self.manager else []
            btc_1d = self.manager.ohlcv_cached(btc_sym, "1d", limit=500, ttl_sec=900) if self.manager else []
            btc_1w = self.manager.ohlcv_cached(btc_sym, "1w", limit=300, ttl_sec=1800) if self.manager else []
            btc_price = float(btc_1h[-1][4]) if btc_1h and len(btc_1h[-1]) >= 5 else 0.0
            if btc_1h and len(btc_1h) >= 2 and len(btc_1h[-2]) >= 5:
                prev_close = float(btc_1h[-2][4])
                if prev_close > 0:
                    btc_context["1h_change_pct"] = (btc_price - prev_close) / prev_close
            if btc_price > 0:
                btc_ctx = IntelligenceContext(
                    symbol=btc_sym,
                    last_price=btc_price,
                    candles_1h=btc_1h,
                    candles_4h=btc_4h,
                    candles_1d=btc_1d,
                    candles_1w=btc_1w,
                    now_ts=now_ts,
                )
                btc_reg = self.intelligence_layer._detect_regime(btc_ctx)
                btc_context["regime"] = btc_reg.regime.value if hasattr(btc_reg.regime, "value") else str(btc_reg.regime)
                btc_context["labels"] = {"regime": btc_context["regime"]}
                btc_context["scores"] = btc_reg.scores or {}
                if btc_context["regime"] in ("STRONG_BEAR", "WEAK_BEAR", "BEAR", "HIGH_VOL_DEFENSIVE", "RISK_OFF"):
                    btc_context["risk_off"] = True

            # Correlation to BTC using 1h returns
            if btc_1h and candles_1h:
                try:
                    def _rets(vals):
                        return [(vals[i] / vals[i-1] - 1.0) for i in range(1, len(vals)) if vals[i-1] > 0]
                    btc_close = [float(c[4]) for c in btc_1h if len(c) >= 5][-120:]
                    sym_close = [float(c[4]) for c in candles_1h if len(c) >= 5][-120:]
                    n = min(len(btc_close), len(sym_close))
                    if n >= 30:
                        r1 = _rets(btc_close[-n:])
                        r2 = _rets(sym_close[-n:])
                        m1 = sum(r1) / len(r1)
                        m2 = sum(r2) / len(r2)
                        cov = sum((a - m1) * (b - m2) for a, b in zip(r1, r2)) / max(1, len(r1) - 1)
                        v1 = sum((a - m1) ** 2 for a in r1) / max(1, len(r1) - 1)
                        v2 = sum((b - m2) ** 2 for b in r2) / max(1, len(r2) - 1)
                        if v1 > 0 and v2 > 0:
                            btc_context["corr"] = max(-1.0, min(1.0, cov / ((v1 ** 0.5) * (v2 ** 0.5))))
                except Exception:
                    pass
        except Exception:
            pass
        
        # Get spread
        spread_pct = None
        try:
            ticker = self.kc.fetch_ticker(symbol)
            bid = float(ticker.get("bid", 0))
            ask = float(ticker.get("ask", 0))
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / bid
        except Exception:
            pass
        
        # Get balances
        balances = {}
        free_quote = account.free_usd
        total_quote = account.total_usd
        
        # Get global pause/kill switch
        global_pause = self._global_pause_on()
        kill_switch = bool(get_setting("kill_switch", "0") == "1")
        
        # Build bot config with safety settings
        bot_config = dict(bot)
        bot_config["global_pause"] = global_pause
        bot_config["kill_switch"] = kill_switch
        bot_config["consecutive_losses"] = int(self.state.consecutive_losses)
        # Kelly sizing: add performance stats for position sizing
        try:
            perf_stats = bot_performance_stats(self.bot_id)
            bot_config["win_rate"] = perf_stats.get("win_rate", 0.5)
            bot_config["avg_profit_pct"] = perf_stats.get("avg_profit_pct", 0.02)
            bot_config["avg_loss_pct"] = perf_stats.get("avg_loss_pct", 0.01)
            bot_config["total_trades"] = perf_stats.get("total_trades", 0)
            bot_config["winning_trades"] = perf_stats.get("winning_trades", 0)
        except Exception:
            pass

        # Data health for instrumentation (no API keys)
        market_type_val = bot.get("market_type", "crypto")
        is_stock = market_type_val in ("stock", "stocks") or (len(symbol) < 6 and "/" not in symbol)
        provider = "alpaca" if is_stock else "kraken"
        if is_stock:
            try:
                from earnings_calendar import days_until_earnings
                ed = days_until_earnings(symbol)
                if ed is not None:
                    bot_config["earnings_days"] = ed
            except Exception:
                pass
            try:
                from stock_metadata import get_sector, get_liquidity_tier
                vol_24h = 0.0
                if candles_1d and len(candles_1d[-1]) >= 6 and price:
                    vol_24h = float(candles_1d[-1][5]) * price
                market_breadth_sector = get_sector(symbol)
                market_breadth_liquidity = get_liquidity_tier(price, vol_24h) if price and vol_24h else "unknown"
            except Exception:
                market_breadth_sector = None
                market_breadth_liquidity = "unknown"
        else:
            market_breadth_sector = None
            market_breadth_liquidity = None
        market_breadth = {
            "is_stock": is_stock,
            "market_type": market_type_val,
            "provider": provider,
            "symbol_normalized": symbol.upper().split("/")[0] if "/" in symbol else symbol.upper(),
            "candle_counts": {
                "1h": len(candles_1h or []),
                "4h": len(candles_4h or []),
                "1d": len(candles_1d or []),
            },
            "limits_requested": {"1h": 220, "4h": 220, "1d": 500},
        }
        if is_stock and market_breadth_sector is not None:
            market_breadth["sector"] = market_breadth_sector
        if is_stock and market_breadth_liquidity:
            market_breadth["liquidity_tier"] = market_breadth_liquidity
        # Propagate data fetch error (e.g. Alpaca asset invalid) for intelligence layer
        try:
            from market_data import get_last_data_error
            err = get_last_data_error(symbol)
            if err:
                market_breadth["data_error"] = err
        except Exception:
            pass

        # Fallback: derive price from candles when live ticker returns 0 (market closed, API glitch)
        if (price is None or price <= 0) and (candles_1d or candles_4h or candles_1h):
            if candles_1d and len(candles_1d[-1]) >= 5:
                price = float(candles_1d[-1][4])
            elif candles_4h and len(candles_4h[-1]) >= 5:
                price = float(candles_4h[-1][4])
            elif candles_1h and len(candles_1h[-1]) >= 5:
                price = float(candles_1h[-1][4])
            if price and price > 0:
                logger.info("Using candle-derived price %.2f for %s (ticker returned 0)", price, symbol)

        return IntelligenceContext(
            symbol=symbol,
            last_price=float(price or 0),
            bid_price=None,  # Would need to fetch
            ask_price=None,  # Would need to fetch
            spread_pct=spread_pct,
            candles_1h=candles_1h,
            candles_4h=candles_4h,
            candles_1d=candles_1d,
            candles_1w=candles_1w,
            balances=balances,
            free_quote=free_quote,
            total_quote=total_quote,
            open_positions=[],  # Would need to fetch open positions
            current_position_size=deal_state.position_size,
            avg_entry_price=deal_state.avg_entry,
            unrealized_pnl=0.0,  # Would calculate from price vs entry
            bot_config=bot_config,
            dry_run=bool(bot.get("dry_run", 1)),
            portfolio_total_usd=total_quote,
            portfolio_exposure_pct=(account.positions_usd / total_quote) if total_quote > 0 else 0.0,
            daily_realized_pnl=perf.realized_today,
            portfolio_drawdown=perf.drawdown,
            btc_context=btc_context,
            market_breadth=market_breadth,
            now_ts=now_ts,
            last_price_ts=now_ts,
            last_candle_ts=now_ts,
            exchange_errors=self.state.errors,
            rate_limit_storm=False,  # Would need to track
        )

    # -----------------
    # Lifecycle
    # -----------------
    def start(self) -> str:
        # Prevent duplicate threads
        if self._thread and self._thread.is_alive():
            return "Already running."

        # If a stop is in progress, wait briefly
        if self._stopping:
            t0 = time.time()
            while self._stopping and (time.time() - t0) < 2.0:
                time.sleep(0.05)

        self._stop.clear()
        with self._lock:
            self.state.running = True
            self.state.last_event = "Starting..."
            self.state.errors = 0

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._log("Start command received.", "INFO", "SYSTEM")
        self._log("Bot is turned on.", "INFO", "SYSTEM")
        if os.getenv("DISCORD_NOTIFY_STARTED", "1").strip().lower() in ("1", "true", "yes"):
            self._notify_discord(f"✅ {self._bot_label()} started.", force=True)
        try:
            # Force enable on start to prevent loop lockout
            from db import set_bot_enabled
            set_bot_enabled(self.bot_id, True) 
            set_bot_running(self.bot_id, True)
        except Exception:
            pass
        return "Started."

    def stop(self) -> str:
        self._stopping = True
        self._stop.set()

        with self._lock:
            self.state.last_event = "Stopping..."

        th = self._thread
        if th and th.is_alive():
            th.join(timeout=STOP_JOIN_TIMEOUT_SEC)

        still_alive = bool(th and th.is_alive())
        with self._lock:
            self.state.running = False
            self.state.last_event = "Stopped." if not still_alive else "Stop requested; thread still shutting down..."

        self._log("Bot stopped.", "INFO", "SYSTEM")
        self._notify_discord(f"🛑 {self._bot_label()} stopped.", force=True)
        try:
            set_bot_running(self.bot_id, False)
        except Exception:
            pass
        self._stopping = False

        if not still_alive:
            self._thread = None

        return "Stopped." if not still_alive else "Stop requested; still shutting down."

    # -----------------
    # Exchange helpers
    # -----------------
    def _balance_free_total(self, asset: str) -> Tuple[float, float]:
        bal = self.kc.fetch_balance()
        free = _safe_float((bal.get("free", {}) or {}).get(asset, 0.0), 0.0)
        total = _safe_float((bal.get("total", {}) or {}).get(asset, 0.0), 0.0)
        return free, total

    def _check_trailing_stop(
        self,
        price: float,
        entry: float,
        tp_pct: float,
        dry_run: bool,
        trailing_activation_pct: Optional[float] = None,
        trailing_distance_pct: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check trailing stop logic. Returns (should_exit, reason).
        Activates when profit reaches activation_pct of TP target; trails distance_pct below highest.
        Uses bot-level trailing_activation_pct/trailing_distance_pct when provided; else env or defaults.
        """
        if not entry or entry <= 0 or not price or price <= 0 or tp_pct <= 0:
            return False, None
        if trailing_activation_pct is not None and trailing_activation_pct > 0:
            activation_threshold = float(trailing_activation_pct)  # bot: fraction of profit to activate (e.g. 0.02 = 2%)
        else:
            activation_factor = float(os.getenv("TRAILING_STOP_ACTIVATION_PCT", "0.5"))
            activation_threshold = tp_pct * activation_factor  # env: fraction of TP target (e.g. 0.5 = 50% of TP)
        if trailing_distance_pct is not None and trailing_distance_pct > 0:
            distance_pct = float(trailing_distance_pct)
        else:
            distance_pct = float(os.getenv("TRAILING_STOP_DISTANCE_PCT", "0.02"))
        profit_pct = (price - entry) / entry

        with self._lock:
            highest = self.state.highest_price_reached
            if highest is None or price > highest:
                highest = price
                self.state.highest_price_reached = highest
            if profit_pct >= activation_threshold:
                self.state.trailing_active = True
            if not self.state.trailing_active:
                return False, None
            trail_price = highest * (1.0 - distance_pct)
            # Break-even stop: once profit reaches ~0.5× TP, never let stop go below entry
            if profit_pct >= activation_threshold and entry > 0:
                trail_price = max(trail_price, entry)
            self.state.trailing_price = trail_price
            if price <= trail_price:
                reason = (
                    f"Trailing stop hit at ${price:.2f} (entry ${entry:.2f}, high ${highest:.2f}, "
                    f"profit +{profit_pct*100:.2f}%)"
                )
                return True, reason
        return False, None

    def _check_partial_exit(
        self,
        price: float,
        entry: float,
        tp_pct: float,
        pos_total: float,
    ) -> Optional[Tuple[float, float, str]]:
        """
        Check partial profit-taking milestones. Returns (level, sell_pct, reason) if a
        partial should execute, else None.
        Default: sell 50% at first TP (100% of target), trail the rest.
        """
        if not os.getenv("PARTIAL_EXIT_ENABLED", "true").lower() in ("true", "1", "yes"):
            return None
        if not entry or entry <= 0 or not price or price <= 0 or tp_pct <= 0 or pos_total <= 0:
            return None
        try:
            levels_str = os.getenv("PARTIAL_EXIT_LEVELS", "1.0")
            amounts_str = os.getenv("PARTIAL_EXIT_AMOUNTS", "0.5")
            levels = [float(x.strip()) for x in levels_str.split(",") if x.strip()]
            amounts = [float(x.strip()) for x in amounts_str.split(",") if x.strip()]
        except (ValueError, TypeError):
            levels = [0.5, 0.75, 1.0]
            amounts = [0.25, 0.25, 0.5]
        if len(amounts) < len(levels):
            amounts.extend([0.0] * (len(levels) - len(amounts)))
        profit_pct = (price - entry) / entry
        progress = profit_pct / tp_pct if tp_pct else 0.0
        with self._lock:
            hit = self.state.partial_levels_hit or []
            initial = self.state.partial_initial_position
        for i, level in enumerate(levels):
            if level in hit:
                continue
            if progress < level:
                continue
            sell_pct = amounts[i] if i < len(amounts) else (1.0 / len(levels))
            if initial is None:
                initial = pos_total
            reason = (
                f"Partial exit {level*100:.0f}% TP: selling {sell_pct*100:.0f}% "
                f"(profit +{profit_pct*100:.2f}%, progress {progress*100:.0f}% of TP)"
            )
            return (level, sell_pct, reason)
        return None

    def _cancel_order_safe(self, symbol: str, order_id: str) -> None:
        if not order_id:
            return
        try:
            # Preferred: KrakenClient wrapper (locked)
            self.kc.cancel_order(order_id, symbol)
            return
        except Exception:
            pass
        # Fallback: cancel all open orders (safe but broader)
        try:
            self.kc.cancel_all_open_orders(symbol)
        except Exception:
            pass

    def _fetch_trades_since(self, symbol: str, since_ms: int, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Best-effort: use ccxt since= if available; otherwise fetch and filter locally.
        """
        trades: List[Dict[str, Any]] = []
        try:
            # Use the locked wrapper when possible
            trades = self.kc.fetch_my_trades(symbol, limit=limit)
        except Exception:
            trades = []

        out: List[Dict[str, Any]] = []
        for t in trades or []:
            ts = t.get("timestamp")
            if ts is None:
                out.append(t)
                continue
            try:
                if int(ts) >= int(since_ms):
                    out.append(t)
            except Exception:
                out.append(t)
        return out

    def _deal_metrics_from_trades(
        self, symbol: str, deal_opened_at_sec: int
    ) -> Tuple[Optional[float], float, float, float, float]:
        """
        Returns:
          avg_entry, buy_amount_base, buy_cost_quote, sell_amount_base, sell_proceeds_quote
        Only considers trades since deal opened.
        """
        since_ms = int(deal_opened_at_sec) * 1000
        trades = self._fetch_trades_since(symbol, since_ms, limit=500)

        buy_amt = 0.0
        buy_cost = 0.0
        sell_amt = 0.0
        sell_proceeds = 0.0

        for t in trades:
            side = (t.get("side") or "").lower()
            amt = _safe_float(t.get("amount"), 0.0)
            price = _safe_float(t.get("price"), 0.0)
            if amt <= 0 or price <= 0:
                continue
            if side == "buy":
                buy_amt += amt
                buy_cost += amt * price
            elif side == "sell":
                sell_amt += amt
                sell_proceeds += amt * price

        avg_entry = (buy_cost / buy_amt) if buy_amt > 0 else None
        return avg_entry, buy_amt, buy_cost, sell_amt, sell_proceeds

    def _account_snapshot_simple(self, quote: str, last_price: float, position_size: float) -> AccountSnapshot:
        free_quote, total_quote = self._balance_free_total(quote)
        total_usd = float(total_quote)
        free_usd = float(free_quote)
        used_usd = max(0.0, float(total_quote) - float(free_quote))
        positions_usd = float(position_size) * float(last_price or 0.0)
        return AccountSnapshot(
            total_usd=total_usd,
            free_usd=free_usd,
            used_usd=used_usd,
            positions_usd=positions_usd,
        )

    def _perf_stats(self, quote: str) -> PerformanceStats:
        try:
            ps = pnl_summary(_today_start_ts_local())
            realized_today = _extract_realized_pnl(ps)
        except Exception:
            realized_today = 0.0
        drawdown = 0.0
        if self._equity_peak > 0:
            drawdown = max(0.0, (self._equity_peak - self._equity_now) / self._equity_peak)
        return PerformanceStats(realized_today=realized_today, drawdown=drawdown, open_deals=1)

    @property
    def _equity_now(self) -> float:
        try:
            return float(self.state.spent_quote or 0.0)
        except Exception:
            return 0.0

    def _smart_config(self, bot: Dict[str, Any]) -> SmartDcaConfig:
        return build_smart_dca_config(bot)

    def _ensure_trading_allowed(self) -> None:
        """Check kill-switch/pause immediately before any order. Raises if blocked."""
        if self._global_pause_on():
            self._set("Order blocked by global pause.", "WARN", "RISK")
            raise RuntimeError("blocked by pause")
        if str(get_setting("kill_switch", "0")).strip().lower() in ("1", "true", "yes", "y", "on"):
            self._set("Order blocked by kill-switch.", "WARN", "RISK")
            raise RuntimeError("blocked by kill-switch")

    def _log_decision(self, action: str, reason: str) -> None:
        if action == self._last_decision_action and reason == self._last_decision_reason:
            self._status(reason)
            return
        self._last_decision_action = action
        self._last_decision_reason = reason
        with self._lock:
            self.state.decision_action = action
            self.state.decision_reason = reason
        category = "STRATEGY"
        if action in ("ENTER", "SAFETY_ORDER", "TAKE_PROFIT", "TRAIL_TP_UPDATE"):
            category = "ORDER"
        if action == "PAUSE":
            category = "RISK"
        if action == "STOP_LOSS":
            category = "RISK"
        self._set(reason, "INFO", category)

    def _build_risk_context(
        self,
        symbol: str,
        account: AccountSnapshot,
        price: float,
        bot: Dict[str, Any],
        spread_bps: Optional[float] = None,
        volume_24h: Optional[float] = None,
        volatility_pct: Optional[float] = None,
        volatility_avg_pct: Optional[float] = None,
    ) -> Optional[Any]:
        """Build RiskContext for risk engine when RISK_ENGINE_ENABLED=1."""
        try:
            from risk_engine import RiskContext, is_enabled
            if not is_enabled():
                return None
        except ImportError:
            return None
        total = max(0.0, float(account.total_usd or 0.0))
        free = max(0.0, float(account.free_usd or 0.0))
        pos_usd = float(account.positions_usd or 0.0)
        positions = {symbol: pos_usd} if pos_usd > 0 else {}
        daily_loss_pct = None
        try:
            ps = pnl_summary(_today_start_ts_local())
            realized = float(ps.get("realized", 0.0) or 0.0)
            if total > 0:
                daily_loss_pct = realized / total
        except Exception:
            pass
        return RiskContext(
            bot_id=self.bot_id,
            symbol=symbol,
            balance_total_usd=total,
            balance_free_usd=free,
            positions_usd=positions,
            symbol_position_usd=pos_usd,
            spread_bps=spread_bps,
            volume_24h_quote=volume_24h,
            volatility_pct=volatility_pct,
            volatility_avg_pct=volatility_avg_pct,
            daily_loss_pct=daily_loss_pct,
            trades_today=count_orders_today(self.bot_id),
            last_error_count=int(self.state.errors or 0),
            max_total_exposure_pct=float(bot.get("max_total_exposure_pct") or os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50")),
            per_symbol_exposure_pct=float(bot.get("per_symbol_exposure_pct") or 0.15),
            daily_loss_limit_pct=float(bot.get("daily_loss_limit_pct") or 0.06),
        )

    def _grid_maintain(
        self,
        symbol: str,
        grid_low: float,
        grid_high: float,
        levels: int,
        step: float,
        base: str,
        quote: str,
        last_price: float,
        dry_run: bool,
        max_open_orders: int,
        quote_per_order: float,
        pos_base: float,
    ) -> None:
        if levels <= 0 or step <= 0:
            return
        try:
            open_orders = self.kc.fetch_open_orders(symbol) if not dry_run else []
        except Exception:
            open_orders = []
        existing = {}
        for o in open_orders or []:
            side = (o.get("side") or "").lower()
            price = _safe_float(o.get("price"), 0.0)
            if price > 0:
                existing[(side, round(price, 2))] = True

        targets = []
        for i in range(1, levels):
            px = grid_low + (step * i)
            if px <= 0:
                continue
            side = "buy" if px < last_price else "sell"
            targets.append((side, px))

        if len(open_orders) >= max_open_orders:
            return

        for side, px in targets:
            if len(open_orders) >= max_open_orders:
                break
            key = (side, round(px, 2))
            if key in existing:
                continue
            if side == "buy":
                size_base = quote_per_order / px if px > 0 else 0.0
                if size_base <= 0:
                    continue
                if dry_run:
                    self._set(f"[DRY RUN] Grid buy {size_base:.6f} {base} @ {px:.2f} {quote}.", "INFO", "ORDER")
                else:
                    self._ensure_trading_allowed()
                    o = self.kc.create_limit_buy_base(symbol, size_base, px)
                    add_order_event(self.bot_id, symbol, "buy", "limit", px, size_base, o.get("id"), "grid", "placed", "grid_order", is_live=1)
            else:
                if pos_base <= 0:
                    continue
                size_base = min(pos_base / max(1, levels), quote_per_order / px if px > 0 else 0.0)
                if size_base <= 0:
                    continue
                if dry_run:
                    self._set(f"[DRY RUN] Grid sell {size_base:.6f} {base} @ {px:.2f} {quote}.", "INFO", "ORDER")
                else:
                    self._ensure_trading_allowed()
                    o = self.kc.create_limit_sell_base(symbol, size_base, px)
                    add_order_event(self.bot_id, symbol, "sell", "limit", px, size_base, o.get("id"), "grid", "placed", "grid_order", is_live=1)
            open_orders.append({"side": side, "price": px})

    def _run_loop_smart(self) -> None:
        backoff = float(BACKOFF_MIN_SEC)
        tp_order_id: Optional[str] = None
        try:
            bot = get_bot(self.bot_id)
            if not bot:
                self._set("Bot config not found.", "ERROR", "DATA")
                with self._lock:
                    self.state.running = False
                return

            raw_symbol = str(bot["symbol"])
            dry_run = bool(bot.get("dry_run", 1))

            if not dry_run and not ALLOW_LIVE_TRADING:
                self._set(
                    "LIVE trading blocked by server guardrail. Set ALLOW_LIVE_TRADING=1 in .env to allow real orders.",
                    "ERROR",
                    "RISK",
                )
                with self._lock:
                    self.state.running = False
                return

            cfg = self._smart_config(bot)
            poll = max(1, _safe_int(bot.get("poll_seconds", 10), 10))

            # -----------------------------------------------------
            # SYMBOL CLASSIFICATION & ROUTING
            # -----------------------------------------------------
            # Use robust classifier instead of heuristics
            market_type = classify_symbol(raw_symbol)
            is_stock = (market_type == "stock")
            
            # If bot explicitly overrides, respect it (rare case)
            if bot.get("market_type") == "stocks":
                is_stock = True
            
            if is_stock:
                # Stock bot - use Alpaca
                if not self.manager:
                    self._set("BotManager not available for stock trading", "ERROR", "SYSTEM")
                    with self._lock:
                        self.state.running = False
                    return
                
                try:
                    # Pass detected market_type to ensure correct client is used
                    alpaca_client = self.manager.get_client_for_bot(bot, override_market_type="stock")
                    self.kc = AlpacaAdapter(alpaca_client)
                    self.executor = OrderExecutor(self.kc)
                    
                    # For stocks, symbol is used as-is, no Kraken resolution needed
                    symbol = raw_symbol
                    # Create a simple market entry for compatibility
                    base = raw_symbol  # e.g., "INTC"
                    quote = "USD"
                    
                    self._set(f"Stock bot initialized for {symbol} (Alpaca)", "INFO", "SYSTEM")
                except Exception as e:
                    self._set(f"Failed to initialize Alpaca for {raw_symbol}: {e}", "ERROR", "SYSTEM")
                    with self._lock:
                        self.state.running = False
                    return
            else:
                # Crypto bot - use Kraken
                markets = self.kc.load_markets()
                symbol = _try_resolve_symbol(markets, raw_symbol)
                if not symbol:
                    self._set(f"Symbol not found on Kraken: {raw_symbol}", "ERROR", "DATA")
                    with self._lock:
                        self.state.running = False
                    return

                mk = markets[symbol]
                base = str(mk.get("base"))
                quote = str(mk.get("quote"))

            deal = latest_open_deal(self.bot_id)
            if deal:
                deal_id = int(deal["id"])
                deal_opened_at = int(deal.get("opened_at") or int(time.time()))
                self._set(f"Resuming existing open deal (deal_id={deal_id}).", "INFO", "STRATEGY")
            else:
                deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                deal_opened_at = int(time.time())
                self._set(f"Opened new deal (deal_id={deal_id}).", "INFO", "STRATEGY")

            with self._lock:
                self.state.deal_id = deal_id
                self.state.deal_opened_at = deal_opened_at
                self.state.partial_initial_position = None
                self.state.partial_levels_hit = []

            # Initialize cooldown from persisted last_stop_ts (survives restarts)
            last_stop_ts = int(get_setting(f"bot:{self.bot_id}:last_stop_ts", "0") or 0)
            if last_stop_ts > 0:
                cooldown_sec = int(cfg.cooldown_sec) * 2  # Same as STOP_LOSS path
                self._cooldown_until = last_stop_ts + cooldown_sec
                with self._lock:
                    self.state.cooldown_until = int(self._cooldown_until)

            while not self._stop.is_set():
                bot = get_bot(self.bot_id)
                if not bot:
                    self._set("Bot removed. Stopping.", "ERROR", "DATA")
                    break

                dry_run = bool(bot.get("dry_run", 1))
                if not dry_run and not ALLOW_LIVE_TRADING:
                    self._set(
                        "LIVE trading blocked by server guardrail. Set ALLOW_LIVE_TRADING=1 in .env to allow real orders.",
                        "ERROR",
                        "RISK",
                    )
                    break

                poll = max(1, _safe_int(bot.get("poll_seconds", 10), 10))

                if self._global_pause_on():
                    self._set("Global pause active. Waiting…", "INFO", "RISK")
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    continue

                try:
                    price = float(self.kc.fetch_ticker_last(symbol))
                except Exception as e:
                    self._set(f"Price fetch failed: {type(e).__name__}: {e}", "ERROR", "DATA")
                    with self._lock:
                        self.state.errors += 1
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(backoff)
                    backoff = min(float(BACKOFF_MAX_SEC), max(float(BACKOFF_MIN_SEC), backoff * 2))
                    continue

                backoff = float(BACKOFF_MIN_SEC)
                with self._lock:
                    self.state.last_price = price
                self._heartbeat()

                # Legacy 15m fetch preserved for compatibility logic if needed, 
                # but we prefer MTF for the brain.
                candles_15m = self.kc.fetch_ohlcv(symbol, timeframe="15m", limit=200)
                candles_1h = self.kc.fetch_ohlcv(symbol, timeframe="1h", limit=100)
                candles_4h = self.kc.fetch_ohlcv(symbol, timeframe="4h", limit=100)
                candles_1d = self.kc.fetch_ohlcv(symbol, timeframe="1d", limit=100)
                candles_1w = self.kc.fetch_ohlcv(symbol, timeframe="1w", limit=50)

                deal_opened_at = int(self.state.deal_opened_at or int(time.time()))
                avg_entry, buy_amt, buy_cost, sell_amt, sell_proceeds = self._deal_metrics_from_trades(symbol, deal_opened_at)
                
                # Recalculate safety used
                safety_used_est = 0
                try:
                    trades = self._fetch_trades_since(symbol, int(deal_opened_at) * 1000, limit=500)
                    buy_count = sum(1 for t in trades if (t.get("side") or "").lower() == "buy")
                    safety_used_est = max(0, int(buy_count) - 1)
                except Exception:
                    safety_used_est = 0

                pos_free = 0.0
                pos_total = 0.0
                if not dry_run:
                    try:
                        pos_free, pos_total = self._balance_free_total(base)
                    except Exception:
                        pos_free, pos_total = 0.0, 0.0

                if avg_entry and price > 0 and float(self.state.base_pos or 0.0) > 0:
                    pnl_pct = (price - avg_entry) / avg_entry
                    with self._lock:
                        if self.state.mfe is None or pnl_pct > float(self.state.mfe):
                            self.state.mfe = float(pnl_pct)
                        if self.state.mae is None or pnl_pct < float(self.state.mae):
                            self.state.mae = float(pnl_pct)

                with self._lock:
                    self.state.avg_entry = float(avg_entry) if (avg_entry is not None) else None
                    self.state.base_pos = float(pos_total) if not dry_run else float(max(buy_amt - sell_amt, 0.0))
                    if buy_cost > 0:
                        self.state.spent_quote = float(buy_cost)
                    self.state.safety_used = int(safety_used_est)

                deal_state = DealState(
                    avg_entry=self.state.avg_entry,
                    position_size=float(self.state.base_pos),
                    safety_used=int(self.state.safety_used),
                    tp_price=self.state.tp_price,
                    spent_quote=float(self.state.spent_quote),
                    trailing_active=False,
                    trailing_price=None,
                )
                account = self._account_snapshot_simple(quote, price, float(self.state.base_pos))
                perf = self._perf_stats(quote)

                # Drawdown guard
                try:
                    equity_now = account.total_usd + account.positions_usd
                    if equity_now > self._equity_peak:
                        self._equity_peak = equity_now
                    max_dd = float(os.getenv("MAX_DRAWDOWN_PCT", "0.20"))
                    if self._equity_peak > 0 and (self._equity_peak - equity_now) / self._equity_peak >= max_dd:
                        self._set("Bot halted: drawdown limit reached.", "ERROR", "RISK")
                        break
                except Exception:
                    pass

                # --- Intelligence Layer Decision ---
                ctx = self._build_intelligence_context(
                    symbol=symbol,
                    price=price,
                    candles_1h=candles_1h,
                    candles_4h=candles_4h,
                    candles_1d=candles_1d,
                    candles_1w=candles_1w,
                    bot=bot,
                    account=account,
                    perf=perf,
                    deal_state=deal_state,
                    now_ts=int(time.time())
                )
                
                intel_decision = self.intelligence_layer.evaluate(ctx)
                
                # Persist full trace
                try:
                    import json
                    add_intelligence_decision(
                        bot_id=int(self.bot_id),
                        symbol=str(symbol),
                        allowed_actions=str(intel_decision.allowed_actions.value if hasattr(intel_decision.allowed_actions, "value") else intel_decision.allowed_actions),
                        final_action=str(intel_decision.final_action),
                        final_reason=str(intel_decision.final_reason),
                        data_ok=bool(intel_decision.data_validity.data_ok),
                        data_reasons=json.dumps(intel_decision.data_validity.reasons),
                        safety_allowed=str(intel_decision.market_safety.allowed_actions.value if hasattr(intel_decision.market_safety.allowed_actions, "value") else intel_decision.market_safety.allowed_actions),
                        safety_reasons=json.dumps(intel_decision.market_safety.reasons),
                        regime=str(intel_decision.regime_detection.regime.value if hasattr(intel_decision.regime_detection.regime, "value") else intel_decision.regime_detection.regime),
                        regime_confidence=float(intel_decision.regime_detection.confidence),
                        strategy_mode=str(intel_decision.strategy_routing.strategy_mode),
                        entry_style=str(intel_decision.strategy_routing.entry_style),
                        exit_style=str(intel_decision.strategy_routing.exit_style),
                        base_size=float(intel_decision.position_sizing.base_size),
                        order_type=str(intel_decision.execution_policy.order_type),
                        manage_actions=json.dumps(intel_decision.trade_management.manage_actions, default=str),
                        proposed_orders=json.dumps(intel_decision.proposed_orders, default=str),
                        debug_json=json.dumps(intel_decision.debug, default=str)
                    )
                except Exception as e:
                    self._log(f"Failed to log intelligence decision: {e}", "ERROR", "SYSTEM")
                # D3: Structured decision log for traceability
                try:
                    conf = float(intel_decision.regime_detection.confidence or 0)
                    dh = (intel_decision.debug or {}).get("data_health", "?")
                    add_log(
                        int(self.bot_id), "INFO",
                        f"Decision: {intel_decision.final_action} | {(intel_decision.final_reason or '')[:80]} | conf={conf:.2f} data={dh}",
                        "INTELLIGENCE",
                    )
                except Exception:
                    pass
                
                # Update State with Brain info
                with self._lock:
                    self.state.regime_label = _safe_enum_val(intel_decision.regime_detection.regime)
                    self.state.regime_confidence = float(intel_decision.regime_detection.confidence)
                    self.state.active_strategy = intel_decision.strategy_routing.strategy_mode
                    self.state.risk_state = "RISK_OFF" if _safe_enum_val(intel_decision.market_safety.allowed_actions) == "NO_TRADE" else "RISK_ON"

                # Adapt to Legacy Decision Struct for Loop Compatibility
                @dataclass
                class LegacyDecision:
                    action: str
                    reason: str

                mapped_action = "HOLD"
                reason = intel_decision.final_reason
                
                # Map Actions
                if intel_decision.final_action == "ENTER":
                    mapped_action = "ENTER"
                elif intel_decision.final_action == "ADD":
                    mapped_action = "SAFETY_ORDER"
                elif intel_decision.final_action == "NO_TRADE":
                    # If specific risk flags, map to PAUSE
                    if intel_decision.market_safety.global_pause: mapped_action = "PAUSE"
                    elif intel_decision.market_safety.kill_switch: mapped_action = "PAUSE"
                    else: mapped_action = "HOLD"
                elif intel_decision.final_action == "MANAGE_ONLY":
                    mapped_action = "HOLD"

                # Map Exits (including trailing stop from intelligence)
                if intel_decision.trade_management.manage_actions:
                    for action in intel_decision.trade_management.manage_actions:
                        atype = action.get("action")
                        if atype == "legacy_stop_loss":
                            mapped_action = "STOP_LOSS"
                            break
                        if atype == "exit_all":
                            mapped_action = "TRAILING_EXIT"
                            reason = action.get("reason", "Trailing stop hit")
                            break

                # Part 2: Partial profit taking (before trailing)
                tp_pct_val = float(bot.get("tp", 0.02))
                partial_result = None
                if avg_entry and price and float(self.state.base_pos or 0) > 0:
                    partial_result = self._check_partial_exit(
                        price=price, entry=avg_entry, tp_pct=tp_pct_val, pos_total=float(pos_total)
                    )
                if partial_result:
                    self._pending_partial = partial_result  # (level, sell_pct, reason)
                    mapped_action = "PARTIAL_EXIT"
                    reason = partial_result[2]
                elif avg_entry and price and float(self.state.base_pos or 0) > 0:
                    self._pending_partial = None
                    should_exit_ts, ts_reason = self._check_trailing_stop(
                        price=price, entry=avg_entry, tp_pct=tp_pct_val, dry_run=dry_run,
                        trailing_activation_pct=float(bot.get("trailing_activation_pct") or 0) or None,
                        trailing_distance_pct=float(bot.get("trailing_distance_pct") or 0) or None,
                    )
                    if should_exit_ts and ts_reason:
                        mapped_action = "TRAILING_EXIT"
                        reason = ts_reason
                else:
                    self._pending_partial = None
                
                # Fallback to smart_decide logic for TP if Intelligence didn't explicitly trigger an exit?
                # Actually, legacy smart_decide handled TP internally via 'tp_price'.
                # We need to make sure handle TP logic is preserved.
                # Since IntelligenceLayer.evaluate() calls _manage_trades(), it should handle it.
                # However, our loop below checks `self.state.tp_price`.
                
                decision = LegacyDecision(mapped_action, reason)


                if decision.action == "PAUSE":
                    self._cooldown_until = int(time.time()) + int(cfg.cooldown_sec)
                    with self._lock:
                        self.state.cooldown_until = int(self._cooldown_until)
                    self._log_decision(decision.action, decision.reason)
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    continue

                if decision.action == "HOLD":
                    self._log_decision(decision.action, decision.reason)
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    continue

                if decision.action == "ENTER":
                    self._log_decision(decision.action, decision.reason)
                    if dry_run:
                        self._notify_discord(f"[DRY RUN] {self._bot_label()} Smart DCA entry buy.")
                        time.sleep(poll)
                        continue
                    # Prevent double-buy: skip if open buy order exists (wait for fill)
                    if hasattr(self.kc, "fetch_open_orders"):
                        try:
                            opens = self.kc.fetch_open_orders(symbol) or []
                            if any((o.get("side") or "").lower() == "buy" for o in opens):
                                self._set("Open buy order exists; waiting for fill.", "INFO", "ORDER")
                                time.sleep(poll)
                                continue
                        except Exception:
                            pass
                    eff_base, _ = _atr_scaled_quote(float(cfg.base_quote), symbol, self.kc)
                    free_quote, _ = self._balance_free_total(quote)
                    if free_quote < eff_base:
                        self._set("Insufficient quote balance for entry.", "ERROR", "RISK")
                        time.sleep(poll)
                        continue
                    self._notify_discord(f"🟢 {self._bot_label()} Smart DCA entry buy {eff_base:.2f} {quote}", trade_event=True)
                    self._ensure_trading_allowed()
                    is_major = symbol in ("XBT/USD", "BTC/USD", "ETH/USD")
                    allowed, eff_size, ob_reason = _check_order_book_market(self.kc, symbol, "buy", float(eff_base), is_major)
                    if not allowed:
                        self._set(f"Order book block: {ob_reason}", "WARN", "RISK")
                        time.sleep(poll)
                        continue
                    if eff_size < eff_base and ob_reason:
                        self._set(f"Order book: {ob_reason}", "INFO", "ORDER")
                    self.kc.create_market_buy_quote(symbol, float(eff_size))

                if decision.action == "SAFETY_ORDER":
                    self._log_decision(decision.action, decision.reason)
                    if dry_run:
                        self._notify_discord(f"[DRY RUN] {self._bot_label()} Smart DCA safety buy.")
                        time.sleep(poll)
                        continue
                    eff_safety, _ = _atr_scaled_quote(float(cfg.safety_quote), symbol, self.kc)
                    free_quote, _ = self._balance_free_total(quote)
                    if free_quote < eff_safety:
                        self._set("Safety buy blocked: insufficient quote balance.", "ERROR", "RISK")
                        time.sleep(poll)
                        continue
                    self._notify_discord(f"🟢 {self._bot_label()} Smart DCA safety buy {eff_safety:.2f} {quote}", trade_event=True)
                    self._ensure_trading_allowed()
                    is_major = symbol in ("XBT/USD", "BTC/USD", "ETH/USD")
                    allowed, eff_size, ob_reason = _check_order_book_market(self.kc, symbol, "buy", float(eff_safety), is_major)
                    if not allowed:
                        self._set(f"Order book block: {ob_reason}", "WARN", "RISK")
                        time.sleep(poll)
                        continue
                    self.kc.create_market_buy_quote(symbol, float(eff_size))
                    if tp_order_id:
                        self._cancel_order_safe(symbol, tp_order_id)
                        tp_order_id = None

                if decision.action == "PARTIAL_EXIT":
                    pr = getattr(self, "_pending_partial", None)
                    self._pending_partial = None
                    self._log_decision("PARTIAL_EXIT", decision.reason)
                    self._set(decision.reason, "INFO", "ORDER")
                    add_log(self.bot_id, decision.reason, "ORDER")
                    self._notify_discord(
                        f"📈 {self._bot_label()} {decision.reason}",
                        trade_event=True,
                    )
                    if dry_run or not pos_free or not pr:
                        time.sleep(poll)
                        continue
                    if tp_order_id:
                        self._cancel_order_safe(symbol, tp_order_id)
                        tp_order_id = None
                        with self._lock:
                            self.state.tp_order_id = None
                    level, sell_pct, _ = pr
                    with self._lock:
                        initial = self.state.partial_initial_position
                        if initial is None:
                            initial = float(pos_total)
                            self.state.partial_initial_position = initial
                        hit = list(self.state.partial_levels_hit or [])
                        hit.append(level)
                        self.state.partial_levels_hit = hit
                    sell_amt = initial * sell_pct
                    if sell_amt <= 0 or sell_amt > pos_total:
                        sell_amt = min(initial * sell_pct, pos_total)
                    if sell_amt <= 0:
                        time.sleep(poll)
                        continue
                    self._ensure_trading_allowed()
                    try:
                        self.kc.create_market_sell_base(symbol, float(sell_amt), f"partial_{self.bot_id}")
                        is_final = level >= 0.99  # 100% TP level
                        if is_final:
                            deal_id = self.state.deal_id
                            if deal_id:
                                close_deal(
                                    deal_id,
                                    entry_avg=float(avg_entry or 0),
                                    exit_avg=float(price),
                                    base_amount=float(sell_amt),
                                    realized_pnl_quote=float(price - (avg_entry or 0)) * float(sell_amt),
                                    exit_strategy="partial_exit",
                                )
                            with self._lock:
                                self.state.trailing_active = False
                                self.state.trailing_price = None
                                self.state.highest_price_reached = None
                                self.state.partial_initial_position = None
                                self.state.partial_levels_hit = []
                                self.state.deal_id = None
                                self.state.base_pos = 0.0
                                self.state.spent_quote = 0.0
                                self.state.tp_price = None
                                self.state.tp_order_id = None
                        else:
                            with self._lock:
                                self.state.trailing_active = True
                    except Exception as e:
                        self._set(f"Partial exit sell failed: {e}", "ERROR", "ORDER")
                        with self._lock:
                            hit = list(self.state.partial_levels_hit or [])
                            if level in hit:
                                hit.remove(level)
                            self.state.partial_levels_hit = hit
                            if self.state.partial_initial_position == float(pos_total):
                                self.state.partial_initial_position = None
                    time.sleep(poll)
                    continue

                if decision.action == "TRAILING_EXIT":
                    self._log_decision("TRAILING_EXIT", decision.reason)
                    self._set(decision.reason, "INFO", "ORDER")
                    add_log(self.bot_id, decision.reason, "ORDER")
                    self._notify_discord(
                        f"📈 {self._bot_label()} {decision.reason}",
                        trade_event=True,
                    )
                    if dry_run or not pos_free:
                        with self._lock:
                            self.state.trailing_active = False
                            self.state.trailing_price = None
                            self.state.highest_price_reached = None
                            self.state.partial_initial_position = None
                            self.state.partial_levels_hit = []
                        time.sleep(poll)
                        continue
                    self._ensure_trading_allowed()
                    try:
                        self.kc.create_market_sell_base(symbol, float(pos_total), f"trail_{self.bot_id}")
                        deal_id = self.state.deal_id
                        if deal_id:
                            close_deal(
                                deal_id,
                                entry_avg=float(avg_entry or 0),
                                exit_avg=float(price),
                                base_amount=float(pos_total),
                                realized_pnl_quote=float(price - (avg_entry or 0)) * float(pos_total),
                                exit_strategy="trailing_exit",
                            )
                        with self._lock:
                            self.state.trailing_active = False
                            self.state.trailing_price = None
                            self.state.highest_price_reached = None
                            self.state.partial_initial_position = None
                            self.state.partial_levels_hit = []
                            self.state.deal_id = None
                            self.state.base_pos = 0.0
                            self.state.spent_quote = 0.0
                            self.state.tp_price = None
                            self.state.tp_order_id = None
                    except Exception as e:
                        self._set(f"Trailing exit sell failed: {e}", "ERROR", "ORDER")
                    time.sleep(poll)
                    continue

                if decision.action == "STOP_LOSS":
                    self._log_decision(decision.action, decision.reason)
                    now_ts = int(time.time())
                    self._cooldown_until = now_ts + int(cfg.cooldown_sec) * 2
                    with self._lock:
                        self.state.cooldown_until = int(self._cooldown_until)
                    set_setting(f"bot:{self.bot_id}:last_stop_ts", str(now_ts))
                    add_log(self.bot_id, "STOP_LOSS executed", "RISK")
                    if dry_run or not pos_free:
                        self._notify_discord(f"[DRY RUN] {self._bot_label()} stop-loss triggered.")
                        time.sleep(poll)
                        continue
                    self._notify_discord(f"🛑 {self._bot_label()} stop-loss market sell.", trade_event=True)
                    try:
                        self._ensure_trading_allowed()
                        size_quote_sell = float(pos_free) * float(self.kc.fetch_ticker_last(symbol) or 0)
                        is_major = symbol in ("XBT/USD", "BTC/USD", "ETH/USD")
                        allowed, _, ob_reason = _check_order_book_market(self.kc, symbol, "sell", size_quote_sell, is_major)
                        if not allowed:
                            self._set(f"Order book block: {ob_reason}", "WARN", "RISK")
                            time.sleep(poll)
                            continue
                        self.kc.create_market_sell_base(symbol, float(pos_free))
                    except Exception as e:
                        self._set(f"Stop-loss sell failed: {type(e).__name__}: {e}", "ERROR", "ORDER")
                        time.sleep(poll)
                        continue
                    try:
                        realized = float(sell_proceeds - buy_cost)
                        avg_exit = (sell_proceeds / sell_amt) if sell_amt > 0 else None
                        od = latest_open_deal(self.bot_id)
                        if od:
                            hold_sec = int(time.time()) - int(self.state.deal_opened_at or int(time.time()))
                            close_deal(
                                int(od["id"]),
                                float(avg_entry) if avg_entry is not None else None,
                                float(avg_exit) if avg_exit is not None else None,
                                float(buy_amt),
                                float(realized),
                                entry_regime=self.state.entry_regime,
                                exit_regime=self.state.regime_label,
                                entry_strategy=self.state.entry_strategy,
                                exit_strategy=self.state.active_strategy,
                                mae=self.state.mae,
                                mfe=self.state.mfe,
                                hold_sec=hold_sec,
                                safety_count=self.state.safety_used,
                            )
                            try:
                                strat = self.state.entry_strategy or self.state.active_strategy or "unknown"
                                add_strategy_trade(self.bot_id, strat, float(realized))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    if int(bot.get("auto_restart", 0)) == 1 and not self._stop.is_set():
                        new_deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                        with self._lock:
                            self.state.deal_id = new_deal_id
                            self.state.deal_opened_at = int(time.time())
                            self.state.safety_used = 0
                            self.state.spent_quote = 0.0
                            self.state.tp_order_id = None
                            self.state.entry_regime = None
                            self.state.entry_strategy = None
                            self.state.mae = None
                            self.state.mfe = None
                            self.state.partial_initial_position = None
                            self.state.partial_levels_hit = []
                        self._set("Auto-restart after stop-loss: opened new deal.", "INFO", "SYSTEM")
                        time.sleep(poll)
                        continue
                    with self._lock:
                        self.state.running = False
                        self.state.last_event = "Stop-loss executed. Bot stopped."
                    return

                if decision.action == "TAKE_PROFIT":
                    self._log_decision(decision.action, decision.reason)
                    if dry_run or not pos_free:
                        self._notify_discord(f"[DRY RUN] {self._bot_label()} Smart DCA take profit.")
                        time.sleep(poll)
                        continue
                    price_tp = float(decision.order.get("price") or 0.0) if decision.order else 0.0
                    if price_tp > 0:
                        self._notify_discord(f"🟣 {self._bot_label()} Smart DCA TP sell @ {price_tp:.2f} {quote}", trade_event=True)
                        self._ensure_trading_allowed()
                        o = self.kc.create_limit_sell_base(symbol, pos_free, price_tp)
                        tp_order_id = o.get("id")
                        with self._lock:
                            self.state.tp_order_id = tp_order_id

                if decision.action == "TRAIL_TP_UPDATE":
                    self._log_decision(decision.action, decision.reason)
                    if dry_run or not pos_free:
                        time.sleep(poll)
                        continue
                    trail_price = float(decision.order.get("trail_price") or 0.0) if decision.order else 0.0
                    if trail_price > 0 and (time.time() - self._last_tp_update_ts) > 30:
                        if tp_order_id:
                            self._cancel_order_safe(symbol, tp_order_id)
                            tp_order_id = None
                        self._ensure_trading_allowed()
                        o = self.kc.create_limit_sell_base(symbol, pos_free, trail_price)
                        tp_order_id = o.get("id")
                        self._last_tp_update_ts = time.time()
                        with self._lock:
                            self.state.tp_order_id = tp_order_id

                # Deal closure detection (live only)
                if not dry_run:
                    try:
                        if buy_amt > 0 and pos_total <= 0.0:
                            realized = float(sell_proceeds - buy_cost)
                            avg_exit = (sell_proceeds / sell_amt) if sell_amt > 0 else None
                            self._set(f"Deal closed. Realized PnL (est): {realized:.2f} {quote}.", "INFO", "ORDER")
                            self._notify_discord(
                                f"✅ {self._bot_label()} deal closed. Realized: {realized:.2f} {quote}",
                                trade_event=True,
                            )
                            od = latest_open_deal(self.bot_id)
                            if od:
                                hold_sec = int(time.time()) - int(self.state.deal_opened_at or int(time.time()))
                                close_deal(
                                    int(od["id"]),
                                    float(avg_entry) if avg_entry is not None else None,
                                    float(avg_exit) if avg_exit is not None else None,
                                    float(buy_amt),
                                    float(realized),
                                    entry_regime=self.state.entry_regime,
                                    exit_regime=self.state.regime_label,
                                    entry_strategy=self.state.entry_strategy,
                                    exit_strategy=self.state.active_strategy,
                                    mae=self.state.mae,
                                    mfe=self.state.mfe,
                                    hold_sec=hold_sec,
                                    safety_count=self.state.safety_used,
                                )
                                try:
                                    strat = self.state.entry_strategy or self.state.active_strategy or "unknown"
                                    notional = float(avg_entry or 0.0) * float(buy_amt or 0.0)
                                    pnl_pct = (float(realized) / notional) if notional > 0 else None
                                    add_strategy_trade(
                                        self.bot_id,
                                        strat,
                                        float(realized),
                                        symbol=symbol,
                                        regime=_safe_enum_val(self.state.entry_regime or self.state.regime_label),
                                        pnl_pct=pnl_pct,
                                    )
                                except Exception:
                                    pass
                                with self._lock:
                                    if float(realized) < 0:
                                        self.state.consecutive_losses += 1
                                    else:
                                        self.state.consecutive_losses = 0
                            if int(bot.get("auto_restart", 0)) == 1 and not self._stop.is_set():
                                new_deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                                with self._lock:
                                    self.state.deal_id = new_deal_id
                                    self.state.deal_opened_at = int(time.time())
                                    self.state.safety_used = 0
                                    self.state.spent_quote = 0.0
                                    self.state.tp_order_id = None
                                    self.state.entry_regime = None
                                    self.state.entry_strategy = None
                                    self.state.mae = None
                                    self.state.mfe = None
                                self._set("Auto-restart: opened new deal.", "INFO", "SYSTEM")
                                time.sleep(poll)
                                continue
                            with self._lock:
                                self.state.running = False
                                self.state.last_event = "Deal closed. Bot stopped."
                            return
                    except Exception:
                        pass

                        time.sleep(poll)
                        continue
                    with self._lock:
                        self.state.running = False
                        self.state.last_event = "Stop-loss executed. Bot stopped."
                    return

                if decision.action == "TAKE_PROFIT":
                    self._log_decision(decision.action, decision.reason)
                    if dry_run or not pos_free:
                        self._notify_discord(f"[DRY RUN] {self._bot_label()} Smart DCA take profit.")
                        time.sleep(poll)
                        continue
                    price_tp = float(decision.order.get("price") or 0.0) if decision.order else 0.0
                    if price_tp > 0:
                        self._notify_discord(f"🟣 {self._bot_label()} Smart DCA TP sell @ {price_tp:.2f} {quote}", trade_event=True)
                        self._ensure_trading_allowed()
                        o = self.kc.create_limit_sell_base(symbol, pos_free, price_tp)
                        tp_order_id = o.get("id")
                        with self._lock:
                            self.state.tp_order_id = tp_order_id

                if decision.action == "TRAIL_TP_UPDATE":
                    self._log_decision(decision.action, decision.reason)
                    if dry_run or not pos_free:
                        time.sleep(poll)
                        continue
                    trail_price = float(decision.order.get("trail_price") or 0.0) if decision.order else 0.0
                    if trail_price > 0 and (time.time() - self._last_tp_update_ts) > 30:
                        if tp_order_id:
                            self._cancel_order_safe(symbol, tp_order_id)
                            tp_order_id = None
                        self._ensure_trading_allowed()
                        o = self.kc.create_limit_sell_base(symbol, pos_free, trail_price)
                        tp_order_id = o.get("id")
                        self._last_tp_update_ts = time.time()
                        with self._lock:
                            self.state.tp_order_id = tp_order_id

                # Deal closure detection (live only)
                if not dry_run:
                    try:
                        if buy_amt > 0 and pos_total <= 0.0:
                            realized = float(sell_proceeds - buy_cost)
                            avg_exit = (sell_proceeds / sell_amt) if sell_amt > 0 else None
                            self._set(f"Deal closed. Realized PnL (est): {realized:.2f} {quote}.", "INFO", "ORDER")
                            self._notify_discord(
                                f"✅ {self._bot_label()} deal closed. Realized: {realized:.2f} {quote}",
                                trade_event=True,
                            )
                            od = latest_open_deal(self.bot_id)
                            if od:
                                hold_sec = int(time.time()) - int(self.state.deal_opened_at or int(time.time()))
                                close_deal(
                                    int(od["id"]),
                                    float(avg_entry) if avg_entry is not None else None,
                                    float(avg_exit) if avg_exit is not None else None,
                                    float(buy_amt),
                                    float(realized),
                                    entry_regime=self.state.entry_regime,
                                    exit_regime=self.state.regime_label,
                                    entry_strategy=self.state.entry_strategy,
                                    exit_strategy=self.state.active_strategy,
                                    mae=self.state.mae,
                                    mfe=self.state.mfe,
                                    hold_sec=hold_sec,
                                    safety_count=self.state.safety_used,
                                )
                                try:
                                    strat = self.state.entry_strategy or self.state.active_strategy or "unknown"
                                    notional = float(avg_entry or 0.0) * float(buy_amt or 0.0)
                                    pnl_pct = (float(realized) / notional) if notional > 0 else None
                                    add_strategy_trade(
                                        self.bot_id,
                                        strat,
                                        float(realized),
                                        symbol=symbol,
                                        regime=_safe_enum_val(self.state.entry_regime or self.state.regime_label),
                                        pnl_pct=pnl_pct,
                                    )
                                except Exception:
                                    pass
                                with self._lock:
                                    if float(realized) < 0:
                                        self.state.consecutive_losses += 1
                                    else:
                                        self.state.consecutive_losses = 0
                            if int(bot.get("auto_restart", 0)) == 1 and not self._stop.is_set():
                                new_deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                                with self._lock:
                                    self.state.deal_id = new_deal_id
                                    self.state.deal_opened_at = int(time.time())
                                    self.state.safety_used = 0
                                    self.state.spent_quote = 0.0
                                    self.state.tp_order_id = None
                                    self.state.entry_regime = None
                                    self.state.entry_strategy = None
                                    self.state.mae = None
                                    self.state.mfe = None
                                self._set("Auto-restart: opened new deal.", "INFO", "SYSTEM")
                                time.sleep(poll)
                                continue
                            with self._lock:
                                self.state.running = False
                                self.state.last_event = "Deal closed. Bot stopped."
                            return
                    except Exception:
                        pass

                time.sleep(poll)

        except Exception as e:
            with self._lock:
                self.state.errors += 1
                self.state.running = False
                self.state.last_event = f"Fatal error: {type(e).__name__}: {e}"
            add_log(self.bot_id, "ERROR", f"Fatal error: {type(e).__name__}: {e}", "DATA")

        finally:
            with self._lock:
                self.state.running = False
            self._stopping = False

    def _run_loop_multi(self) -> None:
        backoff = float(BACKOFF_MIN_SEC)
        tp_order_id: Optional[str] = None
        last_regime_log_ts = 0
        last_decision_log_ts = 0
        try:
            bot = get_bot(self.bot_id)
            if not bot:
                self._set("Bot config not found.", "ERROR", "DATA")
                with self._lock:
                    self.state.running = False
                return

            raw_symbol = str(bot["symbol"])
            dry_run = bool(bot.get("dry_run", 1))

            if not dry_run and not ALLOW_LIVE_TRADING:
                self._set(
                    "LIVE trading blocked by server guardrail. Set ALLOW_LIVE_TRADING=1 in .env to allow real orders.",
                    "ERROR",
                    "RISK",
                )
                with self._lock:
                    self.state.running = False
                return

            if int(bot.get("enabled", 0)) == 0:
                with self._lock:
                    self.state.running = False
                self._set("Bot disabled via config.", "INFO", "SYSTEM")
                return


            # Inject Alpaca Adapter if needed
            # ---------------------------------------------------
            # Auto-detect market type if not set or invalid
            # ---------------------------------------------------
            raw_symbol = str(bot.get("symbol") or "")
            normalized = raw_symbol
            
            # Initial client setup
            if self.manager:
                # First pass: try to respect config OR classify based on symbol
                stored_market_type = bot.get("market_type", "crypto")
                detected_market_type = classify_symbol(raw_symbol)
                # Use either stored "stocks" or detected "stock"
                is_stock = stored_market_type in ("stock", "stocks") or detected_market_type == "stock"
                if is_stock:
                    try:
                        # Pass override to ensure correct client is used
                        real_client = self.manager.get_client_for_bot(bot, override_market_type="stock")
                        self.kc = AlpacaAdapter(real_client)
                        self.executor = OrderExecutor(self.kc) # Update executor
                    except Exception as e:
                        self._log(f"Failed to load Alpaca adapter: {e}", "ERROR")
                
                # Validation & Auto-Correction
                try:
                    # Check if symbol works with current client
                    mk = self.kc.load_markets()
                    resolved = _try_resolve_symbol(mk, raw_symbol)
                    
                    if not resolved:
                        # Symbol not found in current client. Try switching?
                        if market_type == "crypto": 
                            # Failed on Kraken, check Alpaca
                            self._log(f"Symbol {raw_symbol} not found on Kraken. Checking Alpaca...", "WARN")
                            alpaca_client = getattr(self.manager, "alpaca_live", None) or getattr(self.manager, "alpaca_paper", None)
                            if alpaca_client:
                                adapter = AlpacaAdapter(alpaca_client)
                                # Check if valid on Alpaca
                                if adapter.is_tradeable(raw_symbol):
                                    self._log(f"Found {raw_symbol} on Alpaca. Switching to STOCKS mode.", "INFO")
                                    self.kc = adapter
                                    self.executor = OrderExecutor(self.kc)
                                    adapter.ensure_market(raw_symbol) # enhance cache
                                    resolved = raw_symbol
                                    # Persist change so we don't check every time
                                try:
                                    bot["market_type"] = "stocks"
                                    # Update DB so worker_api sees it too
                                    update_bot(self.bot_id, {"market_type": "stocks"})
                                except Exception as e: 
                                    self._log(f"Failed to persist market_type: {e}", "WARN")
                                    pass
                                else:
                                    self._log(f"Symbol {raw_symbol} not found on Alpaca either.", "ERROR")
                except Exception as e:
                    self._log(f"Market auto-detection error: {e}", "ERROR")
                    
            # Final resolution check
            markets = self.kc.load_markets()
            symbol = _try_resolve_symbol(markets, raw_symbol)
            if not symbol and isinstance(self.kc, AlpacaAdapter):
                 # For Alpaca, we might need to trust the symbol if it was validated above
                 symbol = raw_symbol

            if not symbol:
                 self._set(f"Symbol not found/active on {type(self.kc).__name__}: {raw_symbol}", "ERROR", "DATA")
                 with self._lock:
                     self.state.running = False
                 return
            
            # Crash protection: ensure symbol is in markets
            if symbol not in markets:
                if isinstance(self.kc, AlpacaAdapter):
                    # Should have been added by ensure_market above, but force it just in case
                    self.kc.ensure_market(symbol)
                    markets = self.kc.load_markets()
            
            mk = markets.get(symbol)
            if not mk:
                 self._set(f"Critical: Symbol '{symbol}' validated but missing from markets map.", "ERROR", "SYSTEM")
                 with self._lock:
                     self.state.running = False
                 return

            base = str(mk.get("base"))
            quote = str(mk.get("quote"))

            deal = latest_open_deal(self.bot_id)
            if deal:
                deal_id = int(deal["id"])
                deal_opened_at = int(deal.get("opened_at") or int(time.time()))
                self._set(f"Resuming existing open deal (deal_id={deal_id}).", "INFO", "STRATEGY")
            else:
                deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                deal_opened_at = int(time.time())
                self._set(f"Opened new deal (deal_id={deal_id}).", "INFO", "STRATEGY")

            with self._lock:
                self.state.deal_id = deal_id
                self.state.deal_opened_at = deal_opened_at
                self.state.tp_order_id = None
                self.state.trailing_active = False
                self.state.trailing_price = None
                self.state.highest_price_reached = None
                self.state.partial_initial_position = None
                self.state.partial_levels_hit = []

            while not self._stop.is_set():
                bot = get_bot(self.bot_id)
                if not bot:
                    self._set("Bot removed. Stopping.", "ERROR", "DATA")
                    break

                dry_run = bool(bot.get("dry_run", 1))
                if not dry_run and not ALLOW_LIVE_TRADING:
                    self._set(
                        "LIVE trading blocked by server guardrail. Set ALLOW_LIVE_TRADING=1 in .env to allow real orders.",
                        "ERROR",
                        "RISK",
                    )
                    break

                poll = max(1, _safe_int(bot.get("poll_seconds", 10), 10))
                max_spend = _safe_float(bot.get("max_spend_quote", 0.0), 0.0)
                max_open_orders = int(bot.get("max_open_orders", 6))

                if self._global_pause_on():
                    self._set("Global pause active. Waiting…", "INFO", "RISK")
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    continue

                # Kill switch
                try:
                    if str(get_setting("kill_switch", "0")).strip().lower() in ("1", "true", "yes", "y", "on"):
                        self._set("Global kill switch enabled. Pausing entries.", "ERROR", "RISK")
                        time.sleep(poll)
                        continue
                except Exception:
                    pass

                # Price
                try:
                    price = float(self.kc.fetch_ticker_last(symbol))
                except Exception as e:
                    self._set(f"Price fetch failed: {type(e).__name__}: {e}", "ERROR", "DATA")
                    with self._lock:
                        self.state.errors += 1
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(backoff)
                    backoff = min(float(BACKOFF_MAX_SEC), max(float(BACKOFF_MIN_SEC), backoff * 2))
                    continue

                backoff = float(BACKOFF_MIN_SEC)
                with self._lock:
                    self.state.last_price = price
                self._heartbeat()

                # candles + regime
                try:
                    mt = bot.get("market_type", "crypto")
                    candles_5m = self.manager.ohlcv_cached(symbol, "5m", limit=220, ttl_sec=15, market_type=mt) if self.manager else []
                    candles_15m = self.manager.ohlcv_cached(symbol, "15m", limit=220, ttl_sec=30, market_type=mt) if self.manager else []
                    candles_1h = self.manager.ohlcv_cached(symbol, "1h", limit=220, ttl_sec=60, market_type=mt) if self.manager else []
                    candles_4h = self.manager.ohlcv_cached(symbol, "4h", limit=220, ttl_sec=180, market_type=mt) if self.manager else []
                    candles_1d = self.manager.ohlcv_cached(symbol, "1d", limit=500, ttl_sec=900, market_type=mt) if self.manager else []
                    candles_1w = self.manager.ohlcv_cached(symbol, "1w", limit=300, ttl_sec=1800, market_type=mt) if self.manager else []
                    day_trading = int(bot.get("day_trading_mode", 0) or 0)
                    strat_mode = str(bot.get("strategy_mode") or bot.get("forced_strategy") or "").lower()
                    scalping = strat_mode == "scalping"
                    candles_1m = []
                    if day_trading or scalping:
                        candles_1m = self.manager.ohlcv_cached(symbol, "1m", limit=120, ttl_sec=10, market_type=mt) if self.manager else []
                except Exception:
                    candles_5m, candles_15m, candles_1h, candles_4h, candles_1d, candles_1w, candles_1m = [], [], [], [], [], [], []

                regime_short = detect_regime(candles_15m or candles_5m)
                regime_mid = detect_regime(candles_1h or candles_15m)
                regime_long = detect_regime(candles_4h or candles_1h)
                regime = regime_short
                now_ts = int(time.time())
                # Hysteresis / anti-flip
                switch_threshold = float(bot.get("regime_switch_threshold", 0.6))
                switch_ticks = int(bot.get("regime_switch_ticks", 2))
                hold_ticks = int(bot.get("regime_hold_candles", 2))
                dominant_label = regime_short.regime
                dominant_score = 0.0
                try:
                    scores = regime_short.scores or {}
                    dominant_score = max(scores.values()) if scores else 0.0
                except Exception:
                    dominant_score = 0.0

                if self._regime_selected is None:
                    self._regime_selected = dominant_label
                if dominant_label != self._regime_selected:
                    if dominant_label == self._regime_candidate:
                        self._regime_candidate_ticks += 1
                    else:
                        self._regime_candidate = dominant_label
                        self._regime_candidate_ticks = 1
                    if dominant_score >= switch_threshold and self._regime_candidate_ticks >= max(switch_ticks, hold_ticks):
                        self._regime_selected = dominant_label
                        self._regime_candidate = None
                        self._regime_candidate_ticks = 0
                else:
                    self._regime_candidate = None
                    self._regime_candidate_ticks = 0

                if self._regime_selected:
                    regime.regime = self._regime_selected

                try:
                    add_regime_snapshot(
                        self.bot_id,
                        symbol,
                        regime.regime,
                        float(regime.confidence),
                        json.dumps(regime.why or []),
                        json.dumps(
                            {
                                **(regime.snapshot or {}),
                                "scores": regime.scores or {},
                                "tf_short": regime_short.regime,
                                "tf_mid": regime_mid.regime,
                                "tf_long": regime_long.regime,
                            }
                        ),
                    )
                except Exception:
                    pass
                last_regime_log_ts = now_ts

                with self._lock:
                    self.state.regime_label = regime.regime
                    self.state.regime_confidence = float(regime.confidence)
                    self.state.regime_scores = regime.scores or {}
                    if self.state.entry_regime is None and float(self.state.base_pos or 0.0) > 0:
                        self.state.entry_regime = regime.regime

                # positions
                deal_opened_at = int(self.state.deal_opened_at or int(time.time()))
                avg_entry, buy_amt, buy_cost, sell_amt, sell_proceeds = self._deal_metrics_from_trades(symbol, deal_opened_at)
                safety_used_est = 0
                try:
                    trades = self._fetch_trades_since(symbol, int(deal_opened_at) * 1000, limit=500)
                    buy_count = sum(1 for t in trades if (t.get("side") or "").lower() == "buy")
                    safety_used_est = max(0, int(buy_count) - 1)
                except Exception:
                    safety_used_est = 0
                pos_free = 0.0
                pos_total = 0.0
                if not dry_run:
                    try:
                        pos_free, pos_total = self._balance_free_total(base)
                    except Exception:
                        pos_free, pos_total = 0.0, 0.0

                with self._lock:
                    self.state.avg_entry = float(avg_entry) if avg_entry is not None else None
                    self.state.base_pos = float(pos_total) if not dry_run else float(max(buy_amt - sell_amt, 0.0))
                    if buy_cost > 0:
                        self.state.spent_quote = float(buy_cost)
                    self.state.safety_used = int(safety_used_est)

                account = self._account_snapshot_simple(quote, price, float(self.state.base_pos))
                perf = self._perf_stats(quote)

                # Portfolio-level risk checks
                risk_reason = None
                equity = float(account.total_usd) + float(account.positions_usd)
                position_value = float(self.state.base_pos or 0.0) * float(price or 0.0)

                # B3: Centralized circuit breakers (daily loss, drawdown, exposure, max deals)
                if check_circuit_breakers and trip_and_alert and self.manager and equity > 0:
                    try:
                        ps = pnl_summary(_today_start_ts_local())
                        daily_realized = _extract_realized_pnl(ps)
                        total_exposure = float(self.manager.total_exposure_usd())
                        open_deals = int(self.manager.open_positions_count())
                        exp_pct = total_exposure / equity if equity > 0 else 0.0
                        _max_exp = float(bot.get("max_total_exposure_pct") or os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50"))
                        _floor = float(os.getenv("FLOOR_MAX_EXPOSURE_PCT", "0.50"))
                        _max_exp = max(_max_exp, _floor)
                        logger.debug("Exposure check: %s = %.2f%% (limit: %.2f%%)", self._bot_label(), exp_pct * 100, _max_exp * 100)
                        ok, cb_reason = check_circuit_breakers(
                            equity=equity,
                            daily_realized_pnl=daily_realized,
                            portfolio_drawdown=float(perf.drawdown or 0.0),
                            portfolio_exposure_pct=exp_pct,
                            open_deals_count=open_deals,
                            total_exposure_usd=total_exposure,
                            max_total_exposure_pct=_max_exp,
                            max_exposure_pct=_max_exp,
                            max_concurrent_deals=int(bot.get("max_concurrent_deals", 6)),
                            max_daily_loss_pct=float(bot.get("daily_loss_limit_pct", 0.06)),
                            max_drawdown_pct=float(bot.get("max_drawdown_pct", 0) or 0) or 1.0,  # 0 = disabled (use 1.0 so check never trips)
                        )
                        if not ok and cb_reason:
                            risk_reason = cb_reason
                            trip_and_alert(cb_reason, pause_hours=int(bot.get("pause_hours", 6)), bot_label=bot.get("label", str(bot_id)))
                    except Exception as e:
                        import logging
                        logging.getLogger(__name__).debug("Circuit breaker check failed: %s", e)

                if not risk_reason:
                    try:
                        max_total_pct = float(bot.get("max_total_exposure_pct") or os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50"))
                        max_total_pct = max(max_total_pct, float(os.getenv("FLOOR_MAX_EXPOSURE_PCT", "0.50")))
                        override_pct = os.getenv("OVERRIDE_MAX_EXPOSURE_PCT", "").strip()
                        if override_pct:
                            try:
                                max_total_pct = max(max_total_pct, float(override_pct))
                            except ValueError:
                                pass
                        if max_total_pct > 0 and self.manager and equity > 0:
                            total_exposure = float(self.manager.total_exposure_usd())
                            exp_ratio = total_exposure / equity
                            if exp_ratio >= max_total_pct:
                                risk_reason = "Global exposure cap reached."
                            logger.debug("Exposure check: %s = %.2f%% (limit: %.2f%%)", self._bot_label(), exp_ratio * 100, max_total_pct * 100)
                    except Exception:
                        pass

                if not risk_reason:
                    try:
                        max_deals = int(bot.get("max_concurrent_deals", 6))
                        if max_deals > 0 and self.manager:
                            if self.manager.open_positions_count() >= max_deals:
                                risk_reason = "Max concurrent deals reached."
                    except Exception:
                        pass

                try:
                    per_symbol_pct = float(bot.get("per_symbol_exposure_pct", 0.15))
                    if per_symbol_pct > 0 and equity > 0:
                        if (position_value / equity) >= per_symbol_pct:
                            risk_reason = "Per-symbol exposure cap reached."
                except Exception:
                    pass

                try:
                    # Paper/dry_run: skip or relax reserve check (no real money at risk)
                    is_paper = dry_run or (
                        hasattr(self.kc, "client") and hasattr(self.kc.client, "mode")
                        and getattr(self.kc.client, "mode", "") == "paper"
                    )
                    min_free_pct = float(bot.get("min_free_cash_pct", 0.1))
                    try:
                        from portfolio_manager import MIN_CASH_RESERVE_PCT
                        min_free_pct = max(min_free_pct, MIN_CASH_RESERVE_PCT)
                    except ImportError:
                        pass
                    if min_free_pct > 0 and equity > 0 and not is_paper:
                        free_ratio = float(account.free_usd) / equity
                        if free_ratio <= min_free_pct:
                            risk_reason = "Minimum free cash reserve reached."
                            self._log(
                                f"Reserve check: account=live balance={equity:.2f} free={account.free_usd:.2f} "
                                f"reserve_req={min_free_pct*100:.1f}% ratio={free_ratio*100:.1f}%",
                                "DEBUG", "RISK"
                            )
                    elif is_paper and equity > 0 and equity < 5000:
                        # Log only when paper balance is low (helps debug virtual funds)
                        free_ratio = float(account.free_usd) / equity if equity > 0 else 0
                        import logging
                        logging.getLogger(__name__).debug(
                            "Reserve check (paper): balance=%.2f free=%.2f ratio=%.1f%% - skipped",
                            equity, account.free_usd, free_ratio * 100
                        )
                except Exception:
                    pass

                # Daily loss limit (realized + unrealized for this bot) — circuit breaker handles realized; this catches unrealized
                if not risk_reason:
                    try:
                        ps = pnl_summary(_today_start_ts_local())
                        realized_today = _extract_realized_pnl(ps)
                        unrealized = float(position_value - float(self.state.spent_quote or 0.0))
                        loss_total = float(realized_today + unrealized)
                        loss_limit_pct = float(bot.get("daily_loss_limit_pct", 0.06))
                        if equity > 0 and loss_limit_pct > 0 and loss_total <= -(equity * loss_limit_pct):
                            risk_reason = "Daily loss circuit breaker (realized + unrealized)."
                            try:
                                pause_hours = int(bot.get("pause_hours", 6))
                                if trip_and_alert:
                                    trip_and_alert(risk_reason, pause_hours=pause_hours, bot_label=bot.get("label", str(bot_id)))
                                else:
                                    set_setting("global_pause", "1")
                                    set_setting("global_pause_until", str(int(time.time()) + (pause_hours * 3600)))
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Spread guard
                try:
                    max_spread = float(bot.get("spread_guard_pct", 0.003))
                    t = self.kc.fetch_ticker(symbol)
                    bid = float(t.get("bid") or 0.0)
                    ask = float(t.get("ask") or 0.0)
                    mid = (bid + ask) / 2 if bid and ask else 0.0
                    if mid > 0 and ((ask - bid) / mid) >= max_spread:
                        risk_reason = "Spread too wide."
                except Exception:
                    pass

                # Open orders guard
                try:
                    opens = self.kc.fetch_open_orders(symbol) if not dry_run else []
                    if len(opens) > max_open_orders:
                        risk_reason = "Too many open orders."
                except Exception:
                    pass

                if risk_reason:
                    with self._lock:
                        self.state.risk_state = risk_reason
                    self._log_decision("PAUSE", risk_reason)
                    # D3: Discord alert for risk pause (rate limited)
                    if os.getenv("DISCORD_NOTIFY_RISK", "1").strip().lower() in ("1", "true", "yes", "y", "on"):
                        key = f"risk_pause:{risk_reason[:60]}"
                        now = time.time()
                        if getattr(self, "_last_risk_pause_notify", None) != key or (now - getattr(self, "_last_risk_pause_notify_ts", 0)) > 3600:
                            self._last_risk_pause_notify = key
                            self._last_risk_pause_notify_ts = now
                            self._notify_discord(f"⚠️ {self._bot_label()}: PAUSE — {risk_reason}", force=True)
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    continue
                else:
                    with self._lock:
                        self.state.risk_state = None

                # Risk scaling (regime + multi-timeframe + BTC correlation)
                risk_mult = 1.0
                scores = regime.scores or {}
                if scores.get("high_vol_score", 0.0) >= 0.6:
                    risk_mult *= 0.6
                if scores.get("downtrend_score", 0.0) >= 0.6:
                    risk_mult *= 0.7
                if float(perf.drawdown or 0.0) >= 0.1:
                    risk_mult *= 0.7
                try:
                    max_total_pct = float(bot.get("max_total_exposure_pct") or os.getenv("MAX_TOTAL_EXPOSURE_PCT", "0.50"))
                    if max_total_pct > 0 and self.manager and equity > 0:
                        total_exposure = float(self.manager.total_exposure_usd())
                        if (total_exposure / equity) >= (max_total_pct * 0.9):
                            risk_mult *= 0.8
                except Exception:
                    pass

                tf_disagree = False
                try:
                    short_dom = dominant_regime(regime_short.scores)
                    mid_dom = dominant_regime(regime_mid.scores)
                    if short_dom != mid_dom and float(regime_mid.confidence) >= 0.55:
                        tf_disagree = True
                        risk_mult *= 0.8
                except Exception:
                    tf_disagree = False
                if tf_disagree:
                    with self._lock:
                        self.state.risk_state = "TF disagreement: size reduced"

                # BTC correlation rule (alts reduce size on BTC risk-off)
                try:
                    if self.manager and symbol not in ("XBT/USD", "BTC/USD"):
                        btc_candles = self.manager.ohlcv_cached("XBT/USD", "1h", limit=120, ttl_sec=120)
                        btc_regime = detect_regime(btc_candles)
                        btc_scores = btc_regime.scores or {}
                        if btc_scores.get("downtrend_score", 0.0) >= 0.6 or btc_scores.get("high_vol_score", 0.0) >= 0.6:
                            risk_mult *= 0.75
                except Exception:
                    pass

                # Strategy selection handled by Intelligence Layer (below)
                active = self._active_strategy
                forced = str(bot.get("forced_strategy") or "").strip().lower() or None

                # Strategy decision
                bot_cfg = dict(bot)
                bot_cfg["base_quote_mult"] = risk_mult
                bot_cfg["safety_quote_mult"] = risk_mult
                bot_cfg["vol_gap_mult"] = float(bot.get("vol_gap_mult", 1.0))
                bot_cfg["tp_vol_mult"] = float(bot.get("tp_vol_mult", 1.0))
                bot_cfg["min_gap_pct"] = float(bot.get("min_gap_pct", 0.003))
                bot_cfg["max_gap_pct"] = float(bot.get("max_gap_pct", 0.06))
                bot_cfg["safety_cooldown_sec"] = int(bot.get("safety_cooldown_sec", 120))
                if scores.get("high_vol_score", 0.0) >= 0.6:
                    bot_cfg["max_safety"] = max(0, int(bot.get("max_safety", 0)) - 1)

                deal_state = DealState(
                    avg_entry=avg_entry,
                    position_size=float(self.state.base_pos),
                    safety_used=int(self.state.safety_used),
                    tp_price=self.state.tp_price,
                    spent_quote=float(self.state.spent_quote),
                )

                # Part 2: Partial profit taking (before trailing)
                tp_pct_val = float(bot.get("tp", 0.02))
                partial_result = None
                if avg_entry and price and float(self.state.base_pos or 0) > 0:
                    partial_result = self._check_partial_exit(
                        price=price, entry=avg_entry, tp_pct=tp_pct_val, pos_total=float(pos_total)
                    )
                if partial_result:
                    level, sell_pct, partial_reason = partial_result
                    self._set(partial_reason, "INFO", "ORDER")
                    add_log(self.bot_id, partial_reason, "ORDER")
                    self._notify_discord(f"📈 {self._bot_label()} {partial_reason}", trade_event=True)
                    if dry_run or not pos_free:
                        self._heartbeat()
                        time.sleep(poll)
                        continue
                    if tp_order_id:
                        self._cancel_order_safe(symbol, tp_order_id)
                        tp_order_id = None
                        with self._lock:
                            self.state.tp_order_id = None
                    with self._lock:
                        initial = self.state.partial_initial_position
                        if initial is None:
                            initial = float(pos_total)
                            self.state.partial_initial_position = initial
                        hit = list(self.state.partial_levels_hit or [])
                        hit.append(level)
                        self.state.partial_levels_hit = hit
                    sell_amt = min(initial * sell_pct, pos_total)
                    if sell_amt > 0:
                        try:
                            self._ensure_trading_allowed()
                            self.kc.create_market_sell_base(symbol, float(sell_amt), f"partial_{self.bot_id}")
                            is_final = level >= 0.99
                            if is_final:
                                deal_id = self.state.deal_id
                                if deal_id:
                                    close_deal(
                                        deal_id,
                                        entry_avg=float(avg_entry or 0),
                                        exit_avg=float(price),
                                        base_amount=float(sell_amt),
                                        realized_pnl_quote=float(price - (avg_entry or 0)) * float(sell_amt),
                                        exit_strategy="partial_exit",
                                    )
                                with self._lock:
                                    self.state.trailing_active = False
                                    self.state.trailing_price = None
                                    self.state.highest_price_reached = None
                                    self.state.partial_initial_position = None
                                    self.state.partial_levels_hit = []
                                    self.state.deal_id = None
                                    self.state.base_pos = 0.0
                                    self.state.spent_quote = 0.0
                                    self.state.tp_price = None
                                    self.state.tp_order_id = None
                                tp_order_id = None
                            else:
                                with self._lock:
                                    self.state.trailing_active = True
                        except Exception as e:
                            self._set(f"Partial exit sell failed: {e}", "ERROR", "ORDER")
                            with self._lock:
                                hit = list(self.state.partial_levels_hit or [])
                                if level in hit:
                                    hit.remove(level)
                                self.state.partial_levels_hit = hit
                                if self.state.partial_initial_position == float(pos_total):
                                    self.state.partial_initial_position = None
                    self._heartbeat()
                    time.sleep(poll)
                    continue

                # Part 1: Trailing stop check (direct in bot_manager)
                if avg_entry and price and float(self.state.base_pos or 0) > 0:
                    should_exit_ts, ts_reason = self._check_trailing_stop(
                        price=price, entry=avg_entry, tp_pct=tp_pct_val, dry_run=dry_run,
                        trailing_activation_pct=float(bot.get("trailing_activation_pct") or 0) or None,
                        trailing_distance_pct=float(bot.get("trailing_distance_pct") or 0) or None,
                    )
                    if should_exit_ts and ts_reason:
                        self._set(ts_reason, "INFO", "ORDER")
                        add_log(self.bot_id, ts_reason, "ORDER")
                        self._notify_discord(f"📈 {self._bot_label()} {ts_reason}", trade_event=True)
                        if dry_run or not pos_free:
                            with self._lock:
                                self.state.trailing_active = False
                                self.state.trailing_price = None
                                self.state.highest_price_reached = None
                                self.state.partial_initial_position = None
                                self.state.partial_levels_hit = []
                            self._heartbeat()
                            time.sleep(poll)
                            continue
                        self._ensure_trading_allowed()
                        try:
                            if tp_order_id:
                                self._cancel_order_safe(symbol, tp_order_id)
                                tp_order_id = None
                            self.kc.create_market_sell_base(symbol, float(pos_total), f"trail_{self.bot_id}")
                            deal_id = self.state.deal_id
                            if deal_id:
                                close_deal(
                                    deal_id,
                                    entry_avg=float(avg_entry or 0),
                                    exit_avg=float(price),
                                    base_amount=float(pos_total),
                                    realized_pnl_quote=float(price - (avg_entry or 0)) * float(pos_total),
                                    exit_strategy="trailing_exit",
                                )
                            with self._lock:
                                self.state.trailing_active = False
                                self.state.trailing_price = None
                                self.state.highest_price_reached = None
                                self.state.partial_initial_position = None
                                self.state.partial_levels_hit = []
                                self.state.deal_id = None
                                self.state.base_pos = 0.0
                                self.state.spent_quote = 0.0
                                self.state.tp_price = None
                                self.state.tp_order_id = None
                        except Exception as e:
                            self._set(f"Trailing exit sell failed: {e}", "ERROR", "ORDER")
                        self._heartbeat()
                        time.sleep(poll)
                        continue

                # ============================================================
                # INTELLIGENCE LAYER EVALUATION (NEW)
                # ============================================================
                # Build IntelligenceContext

                intel_context = self._build_intelligence_context(
                    symbol=symbol,
                    price=price,
                    candles_1h=candles_1h,
                    candles_4h=candles_4h,
                    candles_1d=candles_1d,
                    candles_1w=candles_1w,
                    bot=bot,
                    account=account,
                    perf=perf,
                    deal_state=deal_state,
                    now_ts=now_ts,
                )
                
                # Evaluate with Intelligence Layer
                intel_decision = self.intelligence_layer.evaluate(intel_context)

                # Scope guard: stocks are analysis-only unless Alpaca integration is active
                if is_stock_symbol(symbol) and not isinstance(self.kc, AlpacaAdapter):
                    intel_decision.allowed_actions = AllowedAction.NO_TRADE
                    intel_decision.final_action = "NO_TRADE"
                    intel_decision.final_reason = "Stocks are analysis-only without Alpaca"
                
                # Log intelligence decision
                try:
                    import json
                    add_intelligence_decision(
                        self.bot_id,
                        symbol,
                        _safe_enum_val(intel_decision.allowed_actions),
                        intel_decision.final_action,
                        intel_decision.final_reason,
                        intel_decision.data_validity.data_ok,
                        json.dumps(intel_decision.data_validity.reasons),
                        _safe_enum_val(intel_decision.market_safety.allowed_actions),
                        json.dumps(intel_decision.market_safety.reasons),
                        _safe_enum_val(intel_decision.regime_detection.regime),
                        float(intel_decision.regime_detection.confidence),
                        intel_decision.strategy_routing.strategy_mode,
                        intel_decision.strategy_routing.entry_style,
                        intel_decision.strategy_routing.exit_style,
                        float(intel_decision.position_sizing.base_size),
                        intel_decision.execution_policy.order_type,
                        json.dumps(intel_decision.trade_management.manage_actions),
                        json.dumps(intel_decision.proposed_orders),
                        json.dumps(intel_decision.debug),
                    )
                except Exception as e:
                    self._set(f"Failed to log intelligence decision: {e}", "ERROR", "SYSTEM")
                # D3: Structured decision log for traceability
                try:
                    conf = float(intel_decision.regime_detection.confidence or 0)
                    dh = (intel_decision.debug or {}).get("data_health", "?")
                    add_log(
                        self.bot_id, "INFO",
                        f"Decision: {intel_decision.final_action} | {(intel_decision.final_reason or '')[:80]} | conf={conf:.2f} data={dh}",
                        "INTELLIGENCE",
                    )
                except Exception:
                    pass
                
                # Check if trading is allowed
                if _safe_enum_val(intel_decision.allowed_actions) == "NO_TRADE":
                    block_msg = f"Trading blocked by Intelligence Layer: {intel_decision.final_reason}"
                    self._set(block_msg, "INFO", "INTELLIGENCE")
                    if os.getenv("DISCORD_NOTIFY_BLOCKED", "1").strip().lower() in ("1", "true", "yes", "y", "on"):
                        key = f"blocked:{(intel_decision.final_reason or '')[:80]}"
                        now = time.time()
                        if getattr(self, "_last_blocked_notify", None) != key or (now - getattr(self, "_last_blocked_notify_ts", 0)) > 3600:
                            self._last_blocked_notify = key
                            self._last_blocked_notify_ts = now
                            self._notify_discord(f"⚠️ {self._bot_label()}: {block_msg}", force=True)
                    # Still execute trade management actions
                    if intel_decision.trade_management.manage_actions:
                        exec_result = self.executor.execute_decision(intel_decision, self.bot_id, symbol, dry_run)
                        if exec_result.get("errors"):
                            for err in exec_result["errors"]:
                                # Don't log market closed as ERROR
                                if "Market is closed" in str(err):
                                    self._set(f"Market closed: {err}", "INFO", "SYSTEM")
                                else:
                                    self._set(f"Trade management errors: {err}", "ERROR", "INTELLIGENCE")
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    continue
                
                # Update strategy mode based on Intelligence Layer routing
                if intel_decision.strategy_routing.strategy_mode != "none":
                    active = intel_decision.strategy_routing.strategy_mode
                    self._active_strategy = active
                    with self._lock:
                        self.state.active_strategy = active
                
                # AUTO_CLOSE_EOD: Close stock positions before market close (prevents overnight risk)
                auto_close_eod = int(bot.get("auto_close_eod", 0) or 0)
                eod_close_triggered = False
                if auto_close_eod and float(pos_total or 0) > 0 and is_stock_symbol(symbol):
                    try:
                        from phase1_intelligence import should_auto_close_eod
                        if should_auto_close_eod():
                            eod_close_triggered = True
                            decision = Decision(
                                "EXIT",
                                "AUTO_CLOSE_EOD: Closing before market close.",
                                {"side": "sell", "type": "market", "size_base": float(pos_total)},
                                {"regime": regime},
                                "auto_close_eod",
                            )
                            decision.strategy = "auto_close_eod"
                            intel_decision.proposed_orders.append(decision.order)
                            self._set("AUTO_CLOSE_EOD: Closing position before market close.", "INFO", "SYSTEM")
                    except Exception:
                        pass

                # ============================================================
                # STRATEGY DECISION (existing flow, but gated by Intelligence)
                # ============================================================
                if not eod_close_triggered:
                    bot_cfg_with_deal = dict(bot_cfg)
                    bot_cfg_with_deal["deal_opened_at"] = deal_opened_at
                    bot_cfg_with_deal["scale_in_tranche_index"] = getattr(self.state, "scale_in_tranche_index", 0)
                    bot_cfg_with_deal["scale_in_last_add_ts"] = getattr(self.state, "scale_in_last_add_ts", None)
                    try:
                        bot_cfg_with_deal["spread_pct"] = float(getattr(intel_context, "spread_pct", None) or 0.0)
                    except Exception:
                        bot_cfg_with_deal["spread_pct"] = 0.0
                    if candles_1d:
                        bot_cfg_with_deal["candles_1d"] = candles_1d
                    ctx = StrategyContext(
                        symbol=symbol,
                        last_price=price,
                        candles_5m=candles_5m,
                        candles_15m=candles_15m,
                        candles_1h=candles_1h,
                        candles_4h=candles_4h,
                        deal=deal_state,
                        account=account,
                        perf=perf,
                        now_ts=now_ts,
                        cooldown_until=self.state.cooldown_until,
                        cfg=bot_cfg_with_deal,
                        regime=regime,
                        candles_1m=candles_1m if candles_1m else None,
                        deal_opened_at=deal_opened_at,
                    )
                    strategy = get_strategy(active)
                    decision = strategy.decide(ctx)
                
                # ============================================================
                # COMPREHENSIVE LOGGING: Strategy Execution Verification
                # ============================================================
                self._log(
                    f"STRATEGY: {active} → {decision.action} | "
                    f"Regime: {_safe_enum_val(regime.regime)} ({regime.confidence:.0%}) | "
                    f"Intelligence: {_safe_enum_val(intel_decision.allowed_actions)} | "
                    f"Reason: {decision.reason}",
                    "INFO",
                    "STRATEGY"
                )
                
                # Log intelligence decision details
                self._log(
                    f"INTELLIGENCE: Phase1={PHASE1_AVAILABLE}, Phase2={PHASE2_AVAILABLE}, Phase3={PHASE3_AVAILABLE} | "
                    f"Final: {intel_decision.final_action} | "
                    f"Strategy routed: {intel_decision.strategy_routing.strategy_mode} | "
                    f"Position size: ${intel_decision.position_sizing.base_size:.2f}",
                    "INFO",
                    "INTELLIGENCE"
                )
                
                # Add strategy's proposed order to Intelligence decision
                if decision.order:
                    # Attach cost-model context for executor
                    try:
                        atr_val = _atr(candles_1d or candles_4h or [], 14)
                        vol_pct = (atr_val / price) if atr_val and price > 0 else 0.0
                    except Exception:
                        vol_pct = 0.0
                    if "expected_edge_pct" not in decision.order:
                        try:
                            reg_label = _safe_enum_val(intel_decision.regime_detection.regime)
                            edge = get_expected_edge(symbol, reg_label, active, window=100, prior_weight=50)
                            expected_edge = float(edge.get("expected_edge") or 0.0)
                            decision.order["expected_edge_pct"] = expected_edge
                            decision.order["edge_trades"] = int(edge.get("trades") or 0)
                        except Exception:
                            decision.order["expected_edge_pct"] = float(bot_cfg.get("tp") or 0.015)
                    decision.order["spread_pct"] = float(getattr(intel_context, "spread_pct", None) or 0.0)
                    decision.order["volatility_pct"] = float(vol_pct or 0.0)
                    decision.order["quote_ts"] = float(now_ts)
                    if not eod_close_triggered:
                        intel_decision.proposed_orders.append(decision.order)
                    self._log(
                        f"ORDER PROPOSED: {decision.order.get('side', 'unknown')} "
                        f"{decision.order.get('type', 'unknown')} @ "
                        f"${decision.order.get('price', 0):.6f} "
                        f"({decision.order.get('size_quote', 0):.2f} {quote})",
                        "INFO",
                        "ORDER"
                    )

                try:
                    add_strategy_decision(
                        self.bot_id,
                        decision.strategy or active,
                        decision.action,
                        decision.reason,
                        regime.regime,
                        float(regime.confidence),
                        json.dumps(decision.debug or {}),
                    )
                except Exception:
                    pass

                self._log_decision(decision.action, decision.reason)

                # ============================================================
                # EXECUTE THROUGH CENTRALIZED EXECUTOR (NEW)
                # ============================================================
                # Execute Intelligence decision through executor
                if _safe_enum_val(intel_decision.allowed_actions) in ("TRADE_ALLOWED", "MANAGE_ONLY"):
                    vol_pct = None
                    if decision.order:
                        vol_pct = decision.order.get("volatility_pct")
                        vol_pct = float(vol_pct) if vol_pct is not None else None
                    risk_ctx = self._build_risk_context(
                        symbol=symbol,
                        account=account,
                        price=price,
                        bot=bot,
                        spread_bps=float(intel_context.spread_pct or 0) * 10000.0 if intel_context.spread_pct else None,
                        volatility_pct=vol_pct,
                    )
                    exec_result = self.executor.execute_decision(
                        intel_decision, self.bot_id, symbol, dry_run,
                        risk_context=risk_ctx,
                    )
                    
                    if exec_result.get("errors"):
                        for err in exec_result["errors"]:
                            # Don't log market closed as ERROR - it's expected
                            if "Market is closed" in str(err):
                                self._set(f"Market closed: {err}", "INFO", "SYSTEM")
                            else:
                                self._set(f"Execution error: {err}", "ERROR", "EXECUTOR")
                    
                    if exec_result.get("orders_placed"):
                        for order in exec_result["orders_placed"]:
                            self._set(f"Order placed: {order.get('side')} {order.get('type')} @ {order.get('price')}", "INFO", "EXECUTOR")
                        # Update scale-in state after ENTER or SCALE_IN
                        if decision.action == "ENTER":
                            self.state.scale_in_tranche_index = 1
                            self.state.scale_in_last_add_ts = int(time.time())
                        elif decision.action == "SCALE_IN":
                            self.state.scale_in_tranche_index = int(decision.order.get("scale_in_tranche", 0) or 0) + 1
                            self.state.scale_in_last_add_ts = int(time.time())
                    
                    # Update intelligence decision with execution results
                    try:
                        import json
                        # Update the logged decision with execution results
                        # (In production, you'd update the DB record)
                    except Exception:
                        pass
                
                # Legacy execution path removed (replaced by self.executor.execute_decision)

                # Deal closure detection (same as classic)
                if not dry_run:
                    try:
                        _, base_total = self._balance_free_total(base)
                        if buy_amt > 0 and base_total <= 0.0:
                            realized = float(sell_proceeds - buy_cost)
                            avg_exit = (sell_proceeds / sell_amt) if sell_amt > 0 else None
                            self._set(f"Deal closed. Realized PnL (est): {realized:.2f} {quote}.", "INFO", "ORDER")
                            self.state.scale_in_tranche_index = 0
                            self.state.scale_in_last_add_ts = None
                            od = latest_open_deal(self.bot_id)
                            if od:
                                hold_sec = int(time.time()) - int(self.state.deal_opened_at or int(time.time()))
                                close_deal(
                                    int(od["id"]),
                                    float(avg_entry) if avg_entry is not None else None,
                                    float(avg_exit) if avg_exit is not None else None,
                                    float(buy_amt),
                                    float(realized),
                                    entry_regime=self.state.entry_regime,
                                    exit_regime=self.state.regime_label,
                                    entry_strategy=self.state.entry_strategy,
                                    exit_strategy=self.state.active_strategy,
                                    mae=self.state.mae,
                                    mfe=self.state.mfe,
                                    hold_sec=hold_sec,
                                    safety_count=self.state.safety_used,
                                )
                                try:
                                    strat = self.state.entry_strategy or self.state.active_strategy or "unknown"
                                    add_strategy_trade(self.bot_id, strat, float(realized))
                                except Exception:
                                    pass
                            if int(bot.get("auto_restart", 0)) == 1 and not self._stop.is_set():
                                new_deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                                with self._lock:
                                    self.state.deal_id = new_deal_id
                                    self.state.deal_opened_at = int(time.time())
                                    self.state.safety_used = 0
                                    self.state.spent_quote = 0.0
                                    self.state.tp_order_id = None
                                self._set("Auto-restart: opened new deal.", "INFO", "SYSTEM")
                                continue
                            with self._lock:
                                self.state.running = False
                                self.state.last_event = "Deal closed. Bot stopped."
                            return
                    except Exception:
                        pass

                time.sleep(poll)

        except Exception as e:
            with self._lock:
                self.state.errors += 1
                self.state.running = False
                self.state.last_event = f"Fatal error: {type(e).__name__}: {e}"
            self._log(f"Fatal error: {type(e).__name__}: {e}", "ERROR", "DATA")
        finally:
            with self._lock:
                self.state.running = False
            self._stopping = False

    # -----------------
    # Main loop (DCA)
    # -----------------
    def _run_loop(self) -> None:
        backoff = float(BACKOFF_MIN_SEC)
        tp_order_id: Optional[str] = None

        try:
            bot = get_bot(self.bot_id)
            if not bot:
                self._set("Bot config not found.", "ERROR", "DATA")
                with self._lock:
                    self.state.running = False
                return
            mode = str(bot.get("strategy_mode", "classic")).lower()
            if mode in ("smart", "smart_dca", "classic_dca", "grid", "trend", "trend_follow", "range_mean_reversion", "high_vol_defensive", "breakout", "auto", "router"):
                self._run_loop_multi()
                return

            raw_symbol = str(bot["symbol"])
            dry_run = bool(bot.get("dry_run", 1))

            # Live trading guardrail
            if not dry_run and not ALLOW_LIVE_TRADING:
                self._set(
                    "LIVE trading blocked by server guardrail. Set ALLOW_LIVE_TRADING=1 in .env to allow real orders.",
                    "ERROR",
                    "RISK",
                )
                with self._lock:
                    self.state.running = False
                return
            if int(bot.get("enabled", 0)) == 0:
                with self._lock:
                    self.state.running = False
                self._set("Bot disabled via config.", "INFO", "SYSTEM")
                return

            raw_symbol = str(bot.get("symbol") or "")
            
            # --- ROUTING FIX ---
            market_type = classify_symbol(raw_symbol)
            is_stock = (market_type == "stock")
            if bot.get("market_type") == "stocks":
                is_stock = True
                
            mk = None
            if is_stock:
                if not self.manager:
                     self._set("BotManager missing for stock bot", "ERROR", "SYSTEM")
                     with self._lock: self.state.running = False
                     return
                try:
                    # Pass detected market_type to ensure correct client is used
                    c = self.manager.get_client_for_bot(bot, override_market_type="stock")
                    self.kc = AlpacaAdapter(c)
                    # Ensure market entry exists for validation
                    self.kc.ensure_market(raw_symbol)
                    symbol = raw_symbol
                    # Mock market data for legacy code compatibility
                    mk = {"base": raw_symbol, "quote": "USD"} 
                    self._set(f"Stock bot initialized for {symbol} (Alpaca)", "INFO", "SYSTEM")
                except Exception as e:
                     self._set(f"Alpaca init failed for {raw_symbol}: {type(e).__name__}: {e}", "ERROR", "SYSTEM")
                     with self._lock: self.state.running = False
                     return
            else:
                # KRAKEN LEGACY PATH
                markets = self.kc.load_markets()
                symbol = _try_resolve_symbol(markets, raw_symbol)
                if not symbol:
                    self._set(f"Symbol not found on Kraken: {raw_symbol}", "ERROR", "DATA")
                    with self._lock:
                        self.state.running = False
                    return
                mk = markets[symbol]
                
            base = str(mk.get("base"))
            quote = str(mk.get("quote"))
            # -------------------

            max_spend = _safe_float(bot.get("max_spend_quote", 0.0), 0.0)
            base_quote_raw = _safe_float(bot.get("base_quote", 0.0))
            
            # CRITICAL: Cap base_quote at 15% of max_spend or $100, whichever is smaller
            # This prevents spending all money at once
            if max_spend > 0:
                safe_base_quote = min(base_quote_raw, max(5.0, max_spend * 0.15), 100.0)
            else:
                # If no max_spend set, calculate from base + safety orders
                safety_quote = _safe_float(bot.get("safety_quote", 0.0))
                max_safety = _safe_int(bot.get("max_safety", 0))
                estimated_max = base_quote_raw + (safety_quote * max_safety)
                safe_base_quote = min(base_quote_raw, max(5.0, estimated_max * 0.15), 100.0)
            
            if safe_base_quote < base_quote_raw:
                max_spend_display = max_spend if max_spend > 0 else (base_quote_raw + (_safe_float(bot.get("safety_quote", 0.0)) * _safe_int(bot.get("max_safety", 0))))
                self._set(
                    f"Base order capped at ${safe_base_quote:.2f} (15% of ${max_spend_display:.2f} budget) for DCA safety.",
                    "INFO",
                    "RISK"
                )
            
            # Auto-scaling: apply allocation from capital_allocator (win/loss streaks)
            alloc_mult = 1.0
            if self.manager:
                try:
                    pt = self.manager.get_portfolio_total()
                    if pt > 0:
                        alloc = self.manager.get_allocation_for_bot(self.bot_id, bot, pt)
                        alloc_mult = float(alloc.get("allocation_mult", 1.0))
                        if alloc_mult != 1.0:
                            self._set(f"Capital allocation: {alloc.get('reason', '')} (mult={alloc_mult:.2f})", "INFO", "RISK")
                            safe_base_quote = safe_base_quote * alloc_mult
                except Exception:
                    pass
            cfg = DcaConfig(
                base_quote=safe_base_quote,  # Use capped value
                safety_quote=_safe_float(bot.get("safety_quote", 0.0)),
                max_safety=_safe_int(bot.get("max_safety", 0)),
                first_dev=_safe_float(bot.get("first_dev", 0.01)),
                step_mult=_safe_float(bot.get("step_mult", 1.2)),
                tp=_safe_float(bot.get("tp", 0.01)),
                trend_filter=bool(bot.get("trend_filter", 0)),
                trend_sma=_safe_int(bot.get("trend_sma", 200)),
            )
            # Store max_spend for use in base buy logic
            max_spend_for_base = max_spend
            effective_max_spend = max_spend if max_spend > 0 else (cfg.base_quote + cfg.safety_quote * cfg.max_safety)
            with self._lock:
                self.state.active_strategy = "classic_dca"
                self.state.risk_state = None
            poll = max(1, _safe_int(bot.get("poll_seconds", 10), 10))

            self._set(f"Loaded config for {raw_symbol}. Dry run={dry_run}.", "INFO", "STRATEGY")

            # Redundant logic removed (handled above)
            pass

            # Resume-safe: use existing open deal if present, otherwise create one.
            deal = latest_open_deal(self.bot_id)
            if deal:
                deal_id = int(deal["id"])
                deal_opened_at = int(deal.get("opened_at") or int(time.time()))
                self._set(f"Resuming existing open deal (deal_id={deal_id}).", "INFO", "STRATEGY")
            else:
                deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                deal_opened_at = int(time.time())
                self._set(f"Opened new deal (deal_id={deal_id}).", "INFO", "STRATEGY")

            with self._lock:
                self.state.deal_id = deal_id
                self.state.deal_opened_at = deal_opened_at
                self.state.safety_used = 0
                self.state.spent_quote = 0.0
                self.state.tp_order_id = None
                self.state.partial_initial_position = None
                self.state.partial_levels_hit = []

            self._dry_run_safety_used = 0
            self._last_safety_buy_ts = 0.0

            # Trend gating BEFORE first buy, only if no buys yet
            if cfg.trend_filter and not self._stop.is_set():
                avg_entry, buy_amt, buy_cost, _, _ = self._deal_metrics_from_trades(symbol, deal_opened_at)
                if buy_amt <= 0:
                    self._set(f"Trend filter enabled (SMA{cfg.trend_sma}). Waiting for entry condition…", "INFO", "STRATEGY")
                    while not self._stop.is_set():
                        ohlcv = self.kc.fetch_ohlcv(symbol, timeframe="15m", limit=max(cfg.trend_sma + 10, 260))
                        closes = [float(c[4]) for c in ohlcv]
                        price_now = float(closes[-1])
                        with self._lock:
                            self.state.last_price = price_now
                        if trend_ok(closes, cfg.trend_sma):
                            self._set("Trend filter passed. Entering now.", "INFO", "STRATEGY")
                            break
                        wait_msg = f"Trend filter blocked entry (below SMA{cfg.trend_sma}). Waiting…"
                        if self._last_wait_reason != wait_msg:
                            self._last_wait_reason = wait_msg
                            self._set(wait_msg, "INFO", "STRATEGY")
                        time.sleep(poll)

            # Base buy (only if deal has no buys yet)
            avg_entry, buy_amt, buy_cost, _, _ = self._deal_metrics_from_trades(symbol, deal_opened_at)
            base_already_placed_dry = dry_run and (float(self.state.spent_quote or 0.0) > 0.0)
            # Resume guard: if we already hold base asset, treat as having bought (avoids double-buy after restart)
            free_base, total_base = self._balance_free_total(base) if base else (0.0, 0.0)
            already_has_position = (total_base or 0.0) > 0.0
            # Prevent double-buy: skip if we have an open BUY order (e.g. limit pending or market in flight)
            has_open_buy_order = False
            if not dry_run and hasattr(self.kc, "fetch_open_orders"):
                try:
                    opens = self.kc.fetch_open_orders(symbol) or []
                    has_open_buy_order = any((o.get("side") or "").lower() == "buy" for o in opens)
                    if has_open_buy_order:
                        self._set("Open buy order exists; waiting for fill before placing another.", "INFO", "ORDER")
                except Exception:
                    pass
            if already_has_position and buy_amt <= 0:
                # Restore state from position so main loop works correctly
                try:
                    px = float(self.kc.fetch_ticker_last(symbol)) if symbol else 0.0
                    est_cost = (total_base or 0.0) * (px if px > 0 else 1.0)
                    with self._lock:
                        self.state.spent_quote = est_cost
                        self.state.base_pos = total_base
                except Exception:
                    pass

            if buy_amt <= 0.0 and not base_already_placed_dry and not already_has_position and not has_open_buy_order and not self._stop.is_set():
                # CRITICAL: Ensure base_quote is reasonable (not the full investment)
                # For DCA, base should be 10-20% of max_spend, not 100%
                # Use max_spend_for_base which was calculated above
                max_spend_val = max_spend_for_base if max_spend_for_base > 0 else (cfg.base_quote + (cfg.safety_quote * cfg.max_safety))
                
                # Cap base_quote at 15% of max_spend or $100, whichever is smaller
                safe_base_quote = min(cfg.base_quote, max(5.0, max_spend_val * 0.15), 100.0)
                
                if safe_base_quote < cfg.base_quote:
                    self._set(
                        f"Base order capped at ${safe_base_quote:.2f} (15% of ${max_spend_val:.2f} budget) for DCA strategy.",
                        "INFO",
                        "RISK"
                    )
                
                # Check if market is closed for stock bots BEFORE trying to fetch price
                market_closed = False
                if is_stock and isinstance(self.kc, AlpacaAdapter):
                    try:
                        if not self.kc.is_market_open():
                            market_closed = True
                            # Try to get last known price for display
                            try:
                                last_price = float(self.kc.fetch_ticker_last(symbol))
                                if last_price > 0:
                                    with self._lock:
                                        self.state.last_price = last_price
                                    self._set(f"Market closed. Last price: ${last_price:.2f}. Waiting for market to open.", "INFO", "SYSTEM")
                                else:
                                    self._set("Market closed. Waiting for market to open.", "INFO", "SYSTEM")
                            except Exception:
                                self._set("Market closed. Waiting for market to open.", "INFO", "SYSTEM")
                            self._heartbeat()
                            time.sleep(min(60.0, poll * 3))
                            # Skip order placement when market closed - break out of this if block
                            base_px = None
                            base_amt = None
                    except Exception:
                        # If market check fails, continue with price fetch
                        pass
                
                # Only fetch price if market is not closed
                if not market_closed:
                    base_px = None
                    base_amt = None
                    # ATR-based position sizing: reduce size when volatility is high
                    effective_base = safe_base_quote
                    scaled_base, atr_reason = _atr_scaled_quote(safe_base_quote, symbol, self.kc)
                    if atr_reason:
                        effective_base = scaled_base
                        self._set(f"Position sizing: {atr_reason}", "INFO", "RISK")
                    try:
                        base_px = float(self.kc.fetch_ticker_last(symbol))
                        if base_px and base_px > 0:
                            base_amt = float(effective_base) / base_px
                        else:
                            # Price fetch returned 0 or None
                            base_px = None
                            base_amt = None
                    except Exception as e:
                        base_px = None
                        base_amt = None
                        # Log the actual error for debugging
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Price fetch failed for {symbol}: {type(e).__name__}: {e}")
                    
                # Correlation check: block if portfolio already over-exposed to correlated assets
                corr_ok, corr_reason = _check_correlation_allowed(symbol, self.bot_id, self.kc)
                if not corr_ok:
                    self._set(
                        f"Correlation guard: {corr_reason}. Skipping base buy.",
                        "INFO",
                        "RISK"
                    )
                    self._heartbeat()
                    time.sleep(poll)
                    # Fall through to main loop without placing
                # CRITICAL: Validate order size before placing
                elif not base_amt or base_amt <= 0 or not base_px or base_px <= 0:
                    # Use WARN instead of ERROR - this is expected when price unavailable
                    self._set(f"Waiting for valid price: amount={base_amt}, price={base_px}. Retrying...", "WARN", "SYSTEM")
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    # Skip order placement, continue to main loop
                else:
                    # Valid order - proceed with placement (use effective_base = ATR-scaled amount)
                    if dry_run:
                        if base_px and base_amt:
                            self._set(
                                f"[DRY RUN] Placing base order. Price: {base_px:.2f} {quote} "
                                f"Size: {effective_base:.8f} {quote} ({base_amt:.8f} {base}).",
                                "INFO",
                                "ORDER",
                            )
                            self._notify_discord(
                                f"[DRY RUN] {self._bot_label()} base buy {base_amt:.8f} {base} @ {base_px:.2f} {quote}"
                            )
                        else:
                            self._set(f"[DRY RUN] Placing base order. Size: {effective_base:.8f} {quote}.", "INFO", "ORDER")
                            self._notify_discord(f"[DRY RUN] {self._bot_label()} base buy {effective_base:.2f} {quote}")
                        with self._lock:
                            self.state.spent_quote = float(effective_base)
                    else:
                        free_quote, _ = self._balance_free_total(quote)
                        while free_quote < effective_base and not self._stop.is_set():
                            self._set(f"Insufficient {quote} for base order. Available {free_quote:.2f}. Waiting…", "WARN", "RISK")
                            self._heartbeat()
                            time.sleep(poll)
                            if not get_bot(self.bot_id):
                                return
                            free_quote, _ = self._balance_free_total(quote)

                        if base_px and base_amt:
                            self._set(
                                f"Placing base order. Price: {base_px:.2f} {quote} "
                                f"Size: {effective_base:.8f} {quote} ({base_amt:.8f} {base}).",
                                "INFO",
                                "ORDER",
                            )
                            self._notify_discord(
                                f"🟢 {self._bot_label()} base buy {base_amt:.8f} {base} @ {base_px:.2f} {quote}",
                                trade_event=True,
                            )
                        else:
                            self._set(f"Placing base order. Size: {effective_base:.8f} {quote}.", "INFO", "ORDER")
                            self._notify_discord(f"🟢 {self._bot_label()} base buy {effective_base:.2f} {quote}", trade_event=True)
                        
                        # CRITICAL: Use effective_base (ATR-scaled)
                        self._ensure_trading_allowed()
                        is_major = symbol in ("XBT/USD", "BTC/USD", "ETH/USD")
                        allowed, eff_size, ob_reason = _check_order_book_market(self.kc, symbol, "buy", effective_base, is_major)
                        if not allowed:
                            self._set(f"Order book block: {ob_reason}", "WARN", "RISK")
                        else:
                            if eff_size < effective_base and ob_reason:
                                self._set(f"Order book: {ob_reason}", "INFO", "ORDER")
                            self.kc.create_market_buy_quote(symbol, float(eff_size))

            # Ladder anchor:
            # - Use the first observed price after start (stable for DCA ladder),
            #   but if we already have an avg entry, anchor to that for consistency.
            try:
                px_now = float(self.kc.fetch_ticker_last(symbol))
            except Exception:
                px_now = 0.0
            entry_ref = float(avg_entry) if (avg_entry is not None and avg_entry > 0) else (px_now if px_now > 0 else 1.0)
            levels = safety_levels(entry_ref, cfg)

            # -------------------------
            # Main control loop
            # -------------------------
            while not self._stop.is_set():
                # Hot-reload bot fields each cycle
                bot = get_bot(self.bot_id)
                if not bot:
                    self._set("Bot removed. Stopping.", "ERROR", "DATA")
                    break

                dry_run = bool(bot.get("dry_run", 1))
                if not dry_run and not ALLOW_LIVE_TRADING:
                    self._set(
                        "LIVE trading blocked by server guardrail. Set ALLOW_LIVE_TRADING=1 in .env to allow real orders.",
                        "ERROR",
                        "RISK",
                    )
                    break

                max_spend = _safe_float(bot.get("max_spend_quote", 0.0), 0.0)
                poll = max(1, _safe_int(bot.get("poll_seconds", 10), 10))

                if self._global_pause_on():
                    self._set("Global pause active. Waiting…", "INFO", "RISK")
                    # Update heartbeat before sleep to prevent watchdog restart
                    self._heartbeat()
                    time.sleep(poll)
                    continue

                # Kill switch: too many runtime errors
                with self._lock:
                    if self.state.errors >= MAX_ERRORS_BEFORE_HALT:
                        self._set(f"Bot halted: too many errors ({self.state.errors}).", "ERROR", "RISK")
                        break

                # Kill switch: daily realized loss limit
                try:
                    ps = pnl_summary(_today_start_ts_local())
                    realized_today = _extract_realized_pnl(ps)
                    if realized_today <= -abs(float(MAX_DAILY_LOSS_QUOTE)):
                        self._set(
                            f"Bot halted: daily loss limit reached (realized {realized_today:.2f} {quote}).",
                            "ERROR",
                            "RISK",
                        )
                        break
                except Exception:
                    pass

                # Current price - ALWAYS fetch live, no caching
                # For stocks, check market hours first
                market_closed = False
                if is_stock and isinstance(self.kc, AlpacaAdapter):
                    try:
                        if not self.kc.is_market_open():
                            market_closed = True
                            # Get next market open time for status message
                            try:
                                open_time, close_time = self.kc.client.get_market_hours()
                                if open_time:
                                    from datetime import datetime
                                    try:
                                        # Parse ISO format with timezone
                                        next_open = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
                                        next_open_str = next_open.strftime("%Y-%m-%d %H:%M:%S %Z")
                                        self._status(f"Market closed. Waiting for market to open at {next_open_str}.")
                                    except Exception:
                                        self._status(f"Market closed. Waiting for market to open (next: {open_time}).")
                                else:
                                    self._status("Market closed. Waiting for market to open.")
                            except Exception:
                                self._status("Market closed. Waiting for market to open.")
                            
                            # Try to get last known price (from daily bar) for display
                            try:
                                # fetch_ticker_last should return daily bar close when market closed
                                price = float(self.kc.fetch_ticker_last(symbol))
                                if price > 0:
                                    with self._lock:
                                        self.state.last_price = price
                                    # Update status to show current price even when closed
                                    self._status(f"Market closed. Last price: ${price:.2f}. Waiting for market to open.")
                                else:
                                    # No price available - use cached if exists
                                    with self._lock:
                                        price = self.state.last_price if self.state.last_price and self.state.last_price > 0 else 0.0
                                    if price <= 0:
                                        self._status("Market closed. Waiting for market to open (price unavailable).")
                            except Exception:
                                with self._lock:
                                    price = self.state.last_price if self.state.last_price and self.state.last_price > 0 else 0.0
                            
                            # Update heartbeat before sleep to prevent watchdog restart
                            self._heartbeat()
                            # Wait in smaller chunks to keep heartbeat fresh (60-120s total, but chunked)
                            wait_time = min(60.0, poll * 3)  # Longer wait when market closed
                            time.sleep(wait_time)
                            # Continue loop - orders will be blocked by executor when market closed
                            continue
                    except Exception:
                        # If market check fails, continue with normal price fetch
                        pass
                
                # Fetch live price if market is open (or if we don't know market status)
                if not market_closed:
                    # Reduced retries - faster failure for better UX
                    price_fetch_attempts = 0
                    max_price_attempts = 3  # Reduced from 5
                    price = 0.0
                    last_error = None
                    last_error_type = None
                    
                    while price_fetch_attempts < max_price_attempts:
                        try:
                            price = float(self.kc.fetch_ticker_last(symbol))
                            # CRITICAL: Reject zero or invalid prices
                            if price > 0 and isinstance(price, (int, float)):
                                import math
                                if not (math.isnan(price) or math.isinf(price)):
                                    break  # Valid price found
                            price = 0.0
                        except Exception as e:
                            price = 0.0
                            last_error = str(e)[:200]  # Truncate long errors
                            last_error_type = type(e).__name__
                        
                        price_fetch_attempts += 1
                        if price_fetch_attempts < max_price_attempts:
                            # Progressive backoff
                            time.sleep(min(1.0 * price_fetch_attempts, 3.0))
                    
                    # If still no valid price after retries, log specific error and continue
                    if price <= 0:
                        error_msg = f"Price fetch failed after {max_price_attempts} attempts: symbol={symbol}"
                        if last_error:
                            error_msg += f", error={last_error_type}: {last_error}"
                        if is_stock:
                            # Check if it's an authentication issue
                            if last_error_type and ("401" in last_error or "403" in last_error or "Unauthorized" in last_error):
                                error_msg += f" (Alpaca API authentication failed - check API keys)"
                            elif last_error_type and ("404" in last_error or "Not Found" in last_error):
                                error_msg += f" (Symbol {symbol} not found - check symbol validity)"
                            else:
                                error_msg += f" (Stock via Alpaca - check: 1) API keys configured, 2) Market is open, 3) Symbol is valid)"
                        # Log as WARN, not ERROR, to reduce Discord spam
                        self._set(error_msg + ". Will retry on next cycle.", "WARN", "SYSTEM")
                        # Use cached price if available
                        with self._lock:
                            if self.state.last_price and self.state.last_price > 0:
                                price = self.state.last_price
                                self._set(f"Using cached price: ${price:.2f}", "INFO", "SYSTEM")
                            else:
                                self._heartbeat()
                                time.sleep(min(30.0, poll * 2))
                                continue

                backoff = float(BACKOFF_MIN_SEC)
                with self._lock:
                    self.state.last_price = price
                self._heartbeat()

                deal_opened_at = int(self.state.deal_opened_at or int(time.time()))
                avg_entry, buy_amt, buy_cost, sell_amt, sell_proceeds = self._deal_metrics_from_trades(symbol, deal_opened_at)

                # Position (live uses balances)
                pos_free = 0.0
                pos_total = 0.0
                if not dry_run:
                    try:
                        pos_free, pos_total = self._balance_free_total(base)
                    except Exception:
                        pos_free, pos_total = 0.0, 0.0

                # Update runtime state
                with self._lock:
                    self.state.avg_entry = float(avg_entry) if (avg_entry is not None) else None
                    self.state.base_pos = float(pos_total) if not dry_run else float(max(buy_amt - sell_amt, 0.0))
                    # spent_quote: prefer buy_cost if we have it
                    if buy_cost > 0:
                        self.state.spent_quote = float(buy_cost)

                # When no position yet, refresh levels from current price (avoids wrong 1.0 fallback when initial fetch failed)
                if avg_entry is None and price > 0:
                    entry_ref = price
                    levels = safety_levels(entry_ref, cfg)

                # TP target (adaptive: scale by volatility)
                effective_tp, tp_reason = _adaptive_tp(float(cfg.tp), symbol, self.kc)
                if tp_reason:
                    self._set(tp_reason, "INFO", "STRATEGY")
                tp_price = float(avg_entry) * (1.0 + effective_tp) if (avg_entry is not None and avg_entry > 0) else None
                with self._lock:
                    self.state.tp_price = tp_price

                # Drawdown guard (based on account + position value)
                try:
                    account = self._account_snapshot_simple(quote, price, float(self.state.base_pos))
                    equity_now = account.total_usd + account.positions_usd
                    if equity_now > self._equity_peak:
                        self._equity_peak = equity_now
                    max_dd = float(os.getenv("MAX_DRAWDOWN_PCT", "0.20"))
                    if self._equity_peak > 0 and (self._equity_peak - equity_now) / self._equity_peak >= max_dd:
                        self._set("Bot halted: drawdown limit reached.", "ERROR", "RISK")
                        break
                except Exception:
                    pass

                # TP management (live only): place TP for FREE base so we do not oversell
                if not dry_run and tp_price is not None and pos_free > 0:
                    if tp_order_id is None:
                        try:
                            self._ensure_trading_allowed()
                            o = self.kc.create_limit_sell_base(symbol, pos_free, tp_price)
                            tp_order_id = o.get("id")
                            with self._lock:
                                self.state.tp_order_id = tp_order_id
                            size_quote = float(pos_free) * float(tp_price)
                            pct = effective_tp * 100.0
                            self._set(
                                f"Placing TakeProfit trade. Price: {tp_price:.2f} {quote} "
                                f"Size: {size_quote:.8f} {quote} ({pos_free:.8f} {base}), "
                                f"the price should rise for {pct:.2f}% to close the trade.",
                                "INFO",
                                "ORDER",
                            )
                            self._notify_discord(
                                f"🟣 {self._bot_label()} TP sell {pos_free:.8f} {base} @ {tp_price:.2f} {quote}",
                                trade_event=True,
                            )
                        except Exception as e:
                            self._set(f"Failed to place TP order: {type(e).__name__}: {e}", "ERROR", "ORDER")
                            with self._lock:
                                self.state.errors += 1
                    else:
                        # Verify TP still open; if not, reset so it can be re-placed
                        try:
                            open_orders = self.kc.fetch_open_orders(symbol)
                            still = any(str(o.get("id")) == str(tp_order_id) for o in (open_orders or []))
                            if not still:
                                self._set("TakeProfit trade cancelled or filled.", "INFO", "ORDER")
                                self._notify_discord(f"🟠 {self._bot_label()} take profit order closed.", trade_event=True)
                                tp_order_id = None
                                with self._lock:
                                    self.state.tp_order_id = None
                        except Exception:
                            pass

                # Safety used = number of buy trades after the first buy (or dry-run simulated count)
                safety_used_est = 0
                if dry_run:
                    safety_used_est = int(self._dry_run_safety_used)
                else:
                    try:
                        trades = self._fetch_trades_since(symbol, int(deal_opened_at) * 1000, limit=500)
                        buy_count = sum(1 for t in trades if (t.get("side") or "").lower() == "buy")
                        safety_used_est = max(0, int(buy_count) - 1)
                    except Exception:
                        safety_used_est = 0

                with self._lock:
                    self.state.safety_used = int(safety_used_est)

                # Effective budget cap: never "allow unlimited" when max_spend is 0
                eff_max = effective_max_spend

                # Safety buy logic
                if safety_used_est < cfg.max_safety:
                    # ATR-scale safety order size (same as base order)
                    effective_safety_quote, _ = _atr_scaled_quote(float(cfg.safety_quote), symbol, self.kc)
                    spent_now = float(buy_cost) if buy_cost > 0 else float(self.state.spent_quote)
                    can_spend = (spent_now + float(effective_safety_quote)) <= float(eff_max) + 1e-9
                    trigger = levels[safety_used_est] if safety_used_est < len(levels) else None

                    # Min cooldown between safety buys (avoid stacking all in seconds)
                    _safety_cooldown_sec = max(0, int(os.getenv("SAFETY_BUY_COOLDOWN_SEC", "60")))
                    now_ts = time.time()
                    cooldown_ok = safety_used_est == 0 or (now_ts - self._last_safety_buy_ts) >= _safety_cooldown_sec

                    if can_spend and cooldown_ok and trigger is not None and price <= float(trigger):
                        # CRITICAL: Validate price before placing order
                        if price <= 0:
                            self._set(
                                f"Safety buy blocked: invalid price {price:.2f} {quote}. Retrying...",
                                "ERROR",
                                "DATA",
                            )
                            time.sleep(poll)
                            continue
                        
                        safety_index = safety_used_est + 1
                        safety_of = cfg.max_safety
                        safety_base_amt = (float(effective_safety_quote) / float(price)) if price > 0 else 0.0
                        
                        # CRITICAL: Validate calculated amount
                        if safety_base_amt <= 0:
                            self._set(
                                f"Safety buy blocked: calculated amount is 0 (price={price:.2f}, quote={effective_safety_quote:.2f}).",
                                "ERROR",
                                "RISK",
                            )
                            time.sleep(poll)
                            continue
                        
                        if dry_run:
                            self._set(
                                f"[DRY RUN] Placing averaging order ({safety_index} out of {safety_of}). "
                                f"Price: {price:.2f} {quote} Size: {effective_safety_quote:.8f} {quote} "
                                f"({safety_base_amt:.8f} {base}).",
                                "INFO",
                                "ORDER",
                            )
                            self._notify_discord(
                                f"[DRY RUN] {self._bot_label()} safety buy {safety_base_amt:.8f} {base} @ {price:.2f} {quote}"
                            )
                            self._dry_run_safety_used = safety_used_est + 1
                            with self._lock:
                                self.state.safety_used = self._dry_run_safety_used
                                self.state.spent_quote = float(self.state.spent_quote or 0.0) + float(effective_safety_quote)
                            self._last_safety_buy_ts = time.time()
                        else:
                            try:
                                free_quote, _ = self._balance_free_total(quote)
                                if free_quote < effective_safety_quote:
                                    self._set(
                                        f"Safety buy blocked: insufficient {quote}. Available {free_quote:.2f}.",
                                        "ERROR",
                                        "RISK",
                                    )
                                else:
                                    # Optional trend filter on safety buys
                                    if cfg.trend_filter:
                                        ohlcv = self.kc.fetch_ohlcv(
                                            symbol,
                                            timeframe="15m",
                                            limit=max(cfg.trend_sma + 10, 260),
                                        )
                                        closes = [float(c[4]) for c in ohlcv]
                                        if not trend_ok(closes, cfg.trend_sma):
                                            self._set(
                                                f"Safety buy blocked by trend filter (below SMA{cfg.trend_sma}).",
                                                "INFO",
                                                "STRATEGY",
                                            )
                                            time.sleep(poll)
                                            continue

                                    self._set(
                                        f"Placing averaging order ({safety_index} out of {safety_of}). "
                                        f"Price: {price:.2f} {quote} Size: {effective_safety_quote:.8f} {quote} "
                                        f"({safety_base_amt:.8f} {base}).",
                                        "INFO",
                                        "ORDER",
                                    )
                                    self._notify_discord(
                                        f"🟢 {self._bot_label()} safety buy {safety_base_amt:.8f} {base} @ {price:.2f} {quote}",
                                        trade_event=True,
                                    )
                                    self._ensure_trading_allowed()
                                    is_major = symbol in ("XBT/USD", "BTC/USD", "ETH/USD")
                                    allowed, eff_size, ob_reason = _check_order_book_market(self.kc, symbol, "buy", float(effective_safety_quote), is_major)
                                    if not allowed:
                                        self._set(f"Order book block: {ob_reason}", "WARN", "RISK")
                                    else:
                                        self.kc.create_market_buy_quote(symbol, float(eff_size))
                                    self._last_safety_buy_ts = time.time()

                                    # Cancel/replace TP on next loop (only our TP)
                                    if tp_order_id:
                                        size_quote = float(pos_free) * float(tp_price or 0.0)
                                        self._set(
                                            f"Cancelling TakeProfit trade. Price: {float(tp_price or 0.0):.2f} {quote} "
                                            f"Size: {size_quote:.8f} {quote} ({pos_free:.8f} {base}).",
                                            "INFO",
                                            "ORDER",
                                        )
                                        self._notify_discord(f"🟠 {self._bot_label()} cancelling take profit order.")
                                        self._cancel_order_safe(symbol, tp_order_id)
                                        tp_order_id = None
                                        with self._lock:
                                            self.state.tp_order_id = None
                            except Exception as e:
                                self._set(f"Safety buy failed: {type(e).__name__}: {e}", "ERROR", "ORDER")
                                with self._lock:
                                    self.state.errors += 1

                # Deal closure detection (live only):
                # If base total is ~0 and we previously bought something, treat as closed.
                if not dry_run:
                    try:
                        _, base_total = self._balance_free_total(base)
                        if buy_amt > 0 and base_total <= 0.0:
                            # Estimated realized from trade totals in the deal window
                            realized = float(sell_proceeds - buy_cost)

                            # Estimate avg exit (if sells exist)
                            avg_exit = (sell_proceeds / sell_amt) if sell_amt > 0 else None

                            self._set(f"Deal closed. Realized PnL (est): {realized:.2f} {quote}.", "INFO", "ORDER")
                            self._notify_discord(
                                f"✅ {self._bot_label()} deal closed. Realized: {realized:.2f} {quote}",
                                trade_event=True,
                            )

                            od = latest_open_deal(self.bot_id)
                            if od:
                                hold_sec = int(time.time()) - int(self.state.deal_opened_at or int(time.time()))
                                close_deal(
                                    int(od["id"]),
                                    float(avg_entry) if avg_entry is not None else None,
                                    float(avg_exit) if avg_exit is not None else None,
                                    float(buy_amt),
                                    float(realized),
                                    entry_regime=self.state.entry_regime,
                                    exit_regime=self.state.regime_label,
                                    entry_strategy=self.state.entry_strategy,
                                    exit_strategy=self.state.active_strategy,
                                    mae=self.state.mae,
                                    mfe=self.state.mfe,
                                    hold_sec=hold_sec,
                                    safety_count=self.state.safety_used,
                                )
                                try:
                                    strat = self.state.entry_strategy or self.state.active_strategy or "classic_dca"
                                    notional = float(buy_cost) if buy_cost and buy_cost > 0 else 1.0
                                    pnl_pct = (float(realized) / notional) if notional > 0 else None
                                    add_strategy_trade(
                                        self.bot_id,
                                        strat,
                                        float(realized),
                                        symbol=symbol,
                                        regime=self.state.regime_label,
                                        pnl_pct=pnl_pct,
                                    )
                                except Exception:
                                    pass

                            if int(bot.get("auto_restart", 0)) == 1 and not self._stop.is_set():
                                new_deal_id = open_deal(self.bot_id, symbol, state="OPEN")
                                with self._lock:
                                    self.state.deal_id = new_deal_id
                                    self.state.deal_opened_at = int(time.time())
                                    self.state.safety_used = 0
                                    self.state.spent_quote = 0.0
                                    self.state.tp_order_id = None
                                self._dry_run_safety_used = 0
                                self._last_safety_buy_ts = 0.0
                                self._set("Auto-restart: opened new deal.", "INFO", "SYSTEM")
                                continue

                            with self._lock:
                                self.state.running = False
                                self.state.last_event = "Deal closed. Bot stopped."
                            return
                    except Exception:
                        pass

                # Status: show next action without spamming logs
                try:
                    next_trigger = levels[safety_used_est] if safety_used_est < len(levels) else None
                    if tp_price and avg_entry:
                        if price < tp_price:
                            pct = effective_tp * 100.0
                            self._status(f"Waiting to sell at {tp_price:.2f} {quote} (+{pct:.2f}%). Current: {price:.2f} {quote}.")
                        else:
                            self._status(f"TP reached. Waiting for fill… Current: {price:.2f} {quote}.")
                    elif next_trigger and price > 0:
                        # Sanity: only show trigger if it's realistic (within 50% of current price)
                        if next_trigger >= price * 0.5:
                            self._status(f"Waiting to buy at {float(next_trigger):.2f} {quote}. Current: {price:.2f} {quote}.")
                        else:
                            self._status(f"Monitoring. Current price: {price:.2f} {quote}.")
                    elif price > 0:
                        # Fallback: just show current price if no trigger
                        self._status(f"Monitoring. Current price: {price:.2f} {quote}.")
                except Exception:
                    pass

                # Update heartbeat before sleep to prevent watchdog from restarting
                self._heartbeat()
                time.sleep(poll)

        except Exception as e:
            with self._lock:
                self.state.errors += 1
                self.state.running = False
                self.state.last_event = f"Fatal error: {type(e).__name__}: {e}"
            self._log(f"Fatal error: {type(e).__name__}: {e}", "ERROR", "DATA")

        finally:
            with self._lock:
                self.state.running = False
            self._stopping = False


class BotManager:
    def __init__(self, kc: KrakenClient, alpaca_paper: Optional[AlpacaClient] = None, alpaca_live: Optional[AlpacaClient] = None):
        self.kc = kc
        self.alpaca_paper = alpaca_paper
        self.alpaca_live = alpaca_live
        self._bots: Dict[int, BotRunner] = {}
        self._lock = threading.Lock()
        self._md_lock = threading.Lock()
        self._md_cache: Dict[str, Dict[str, Any]] = {}
        try:
            from market_data import MarketDataRouter
            self._md_router = MarketDataRouter(
                kraken_client=kc,
                alpaca_paper=alpaca_paper,
                alpaca_live=alpaca_live,
            )
        except Exception as e:
            logger.warning("MarketDataRouter init failed, using legacy ohlcv: %s", e)
            self._md_router = None

    def subscribe_all_symbols(self) -> None:
        """Subscribe to WebSocket for all stock bot symbols (UnifiedAlpacaClient)."""
        client = self.alpaca_live or self.alpaca_paper
        if not client or not hasattr(client, "subscribe_to_symbols"):
            return
        try:
            from db import list_bots
            from symbol_classifier import is_stock_symbol
            bots = list_bots() or []
            symbols = []
            for b in bots:
                sym = str(b.get("symbol") or "").strip()
                if not sym:
                    continue
                mt = str(b.get("market_type") or "crypto").lower()
                if mt in ("stock", "stocks") or is_stock_symbol(sym):
                    symbols.append(sym)
            symbols = list(dict.fromkeys(symbols))[:100]
            if symbols:
                client.subscribe_to_symbols(symbols)
                logger.info("Subscribed to %d stock symbols for WebSocket", len(symbols))
        except Exception as e:
            logger.warning("subscribe_all_symbols failed: %s", e)

    def get_client_for_bot(self, bot_row: Dict[str, Any], override_market_type: Optional[str] = None):
        """Get the appropriate trading client for a bot based on market_type and alpaca_mode.
        
        Args:
            bot_row: Bot configuration dict
            override_market_type: Optional override for market_type (use "stock" or "stocks")
        """
        # Use override if provided, otherwise use bot's market_type
        market_type = override_market_type or bot_row.get("market_type", "crypto")
        
        # Accept both "stock" and "stocks" for consistency
        is_stock = market_type in ("stock", "stocks")
        
        if is_stock:
            alpaca_mode = bot_row.get("alpaca_mode", "paper")
            if alpaca_mode == "live":
                if not self.alpaca_live:
                    raise ValueError(
                        "Alpaca live trading client not initialized. "
                        "Set LIVE_TRADING_ENABLED=1 in .env and add Alpaca live API keys, then restart the app."
                    )
                return self.alpaca_live
            else:
                if not self.alpaca_paper:
                    raise ValueError("Alpaca paper trading client not initialized")
                return self.alpaca_paper
        else:
            # Default to Kraken for crypto
            return self.kc

    def ohlcv_cached(self, symbol: str, timeframe: str, limit: int = 200, ttl_sec: int = 20, market_type: Optional[str] = None) -> List[List[float]]:
        """Fetch OHLCV from correct client. Request buffer for validation; Yahoo fallback for stocks."""
        min_acceptable = 20
        req_limit = max(limit + 50, min_acceptable + 30)
        if self._md_router:
            try:
                data = self._md_router.get_candles(symbol, timeframe, req_limit, market_type, use_cache=True)
                if data and len(data) >= min_acceptable:
                    return data[-limit:] if limit < len(data) else data
                if data and len(data) < min_acceptable and is_stock_symbol(symbol):
                    data = self._ohlcv_yahoo_fallback(symbol, timeframe, req_limit, data)
                    if data and len(data) >= min_acceptable:
                        return data[-limit:] if limit < len(data) else data
                if is_stock_symbol(symbol) and (not data or len(data) < min_acceptable):
                    try:
                        from phase2_data_fetcher import fetch_recent_candles
                        data = fetch_recent_candles(symbol, timeframe, req_limit)
                        if data and len(data) >= min_acceptable:
                            return data[-limit:] if limit < len(data) else data
                    except Exception as e2:
                        logger.debug("ohlcv_cached phase2 Yahoo fallback %s: %s", symbol, e2)
                logger.warning(
                    "ohlcv_cached router: symbol=%s tf=%s got=%d need=%d",
                    symbol, timeframe, len(data) if data else 0, min_acceptable,
                )
            except Exception as e:
                logger.warning("ohlcv_cached router failed, falling back to legacy: %s", e)
        key = f"{symbol}|{timeframe}|{limit}"
        now = time.time()
        with self._md_lock:
            entry = self._md_cache.get(key)
            if entry and (now - float(entry.get("ts", 0))) <= ttl_sec:
                cached = entry.get("data") or []
                if cached and len(cached) >= min_acceptable:
                    return cached[-limit:] if limit < len(cached) else cached
        client = self.kc
        if is_stock_symbol(symbol) and (self.alpaca_paper or self.alpaca_live):
            alpaca = self.alpaca_live or self.alpaca_paper
            client = AlpacaAdapter(alpaca)
            if symbol:
                client.ensure_market(symbol)
        for attempt in range(3):
            try:
                data = client.fetch_ohlcv(symbol, timeframe=timeframe, limit=req_limit)
                if data and len(data) >= min_acceptable:
                    break
            except Exception as e:
                logger.warning("ohlcv_cached fetch attempt %d symbol=%s tf=%s: %s", attempt + 1, symbol, timeframe, e)
                data = []
            if is_stock_symbol(symbol) and (not data or len(data) < min_acceptable):
                data = self._ohlcv_yahoo_fallback(symbol, timeframe, req_limit, data)
            if data and len(data) >= min_acceptable:
                break
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
        with self._md_lock:
            self._md_cache[key] = {"ts": now, "data": data or []}
        out = (data or [])[-limit:] if data and limit < len(data) else (data or [])
        return out

    def _ohlcv_yahoo_fallback(self, symbol: str, timeframe: str, limit: int, existing: Optional[List] = None) -> List[List[float]]:
        """Yahoo Finance fallback for stocks when Alpaca returns empty or insufficient data."""
        if existing and len(existing) >= 20:
            return existing
        try:
            from phase2_data_fetcher import _fetch_candles_yahoo
            for try_limit in [max(limit, 100), 100, 80]:
                yahoo = _fetch_candles_yahoo(symbol, timeframe, try_limit)
                if yahoo and len(yahoo) >= 20:
                    logger.info("ohlcv_cached Yahoo fallback: %s %s -> %d candles", symbol, timeframe, len(yahoo))
                    return yahoo
                time.sleep(0.2)
        except Exception as e:
            logger.debug("Yahoo fallback %s %s: %s", symbol, timeframe, e)
        return existing or []

    def get_portfolio_total(self) -> float:
        """Aggregate portfolio value from all connected accounts (Kraken + Alpaca)."""
        total = 0.0
        try:
            if self.kc and hasattr(self.kc, "fetch_balance"):
                bal = self.kc.fetch_balance()
                free = (bal.get("free") or {}) or {}
                total += float(free.get("USD") or free.get("ZUSD") or 0) or 0
                total += float(free.get("USDT") or 0) or 0
                for asset, amt in (bal.get("total") or {}).items():
                    if asset in ("USD", "ZUSD", "USDT"):
                        continue
                    try:
                        if asset == "XXBT":
                            total += float(amt or 0) * 43000
                        else:
                            total += float(amt or 0)
                    except Exception:
                        pass
        except Exception as e:
            logger.debug("Kraken balance for portfolio_total failed: %s", e)
        try:
            for client in [getattr(self, "alpaca_paper", None), getattr(self, "alpaca_live", None)]:
                if client and hasattr(client, "get_account"):
                    acct = client.get_account()
                    total += float(acct.get("portfolio_value") or acct.get("equity") or 0) or 0
        except Exception as e:
            logger.debug("Alpaca balance for portfolio_total failed: %s", e)
        return max(0.0, total)

    def get_allocation_for_bot(
        self,
        bot_id: int,
        bot: Dict[str, Any],
        portfolio_total: float,
    ) -> Dict[str, Any]:
        """Get capital allocation for a bot (auto-scaling from win/loss streaks)."""
        try:
            from capital_allocator import get_allocation_for_bot as _get_alloc
            from capital_allocator import BotAllocationInput
            from db import get_bot_recent_streak, bot_performance_stats, bot_deal_stats
            streak = get_bot_recent_streak(bot_id, 5)
            perf = bot_performance_stats(bot_id)
            stats = bot_deal_stats(bot_id)
            inp = BotAllocationInput(
                bot_id=bot_id,
                symbol=str(bot.get("symbol", "")),
                base_quote=float(bot.get("base_quote", 0)),
                max_spend_quote=float(bot.get("max_spend_quote", 0)),
                streak=streak,
                win_rate=float(perf.get("win_rate", 0.5)),
                realized_total=float(stats.get("realized_total", 0)),
                strategy_mode=str(bot.get("strategy_mode", "classic")),
            )
            res = _get_alloc(bot_id, portfolio_total, inp)
            return {
                "allocation_mult": res.allocation_mult,
                "effective_base_quote": res.effective_base_quote,
                "effective_max_spend": res.effective_max_spend,
                "reason": res.reason,
            }
        except Exception as e:
            logger.debug("get_allocation_for_bot failed: %s", e)
            return {"allocation_mult": 1.0, "effective_base_quote": float(bot.get("base_quote", 0)), "effective_max_spend": float(bot.get("max_spend_quote", 0)), "reason": "fallback"}

    def total_exposure_usd(self) -> float:
        total = 0.0
        with self._lock:
            runners = list(self._bots.values())
        for r in runners:
            snap = r.snapshot()
            try:
                if snap.get("last_price") is None:
                    continue
                total += float(snap.get("base_pos") or 0.0) * float(snap.get("last_price") or 0.0)
            except Exception:
                continue
        return float(total)

    def open_positions_count(self) -> int:
        cnt = 0
        with self._lock:
            runners = list(self._bots.values())
        for r in runners:
            snap = r.snapshot()
            try:
                if float(snap.get("base_pos") or 0.0) > 0:
                    cnt += 1
            except Exception:
                continue
        return int(cnt)

    def get_runner(self, bot_id: int) -> BotRunner:
        """Get or create BotRunner with correct client based on bot's market_type."""
        bot_id = int(bot_id)
        with self._lock:
            if bot_id not in self._bots:
                # Load bot to determine market_type
                bot = get_bot(bot_id)
                if not bot:
                    # Fallback to Kraken if bot not found
                    client = self.kc
                else:
                    # Get correct client based on market_type OR symbol classification
                    symbol = str(bot.get("symbol", ""))
                    stored_market_type = bot.get("market_type", "crypto")
                    detected_market_type = classify_symbol(symbol) if symbol else "crypto"
                    # Use either stored "stocks" or detected "stock"
                    is_stock = stored_market_type in ("stock", "stocks") or detected_market_type == "stock"
                    
                    if is_stock:
                        # Get Alpaca client and wrap in adapter
                        real_client = self.get_client_for_bot(bot, override_market_type="stock")
                        client = AlpacaAdapter(real_client)
                        # Ensure market is set up for the symbol
                        if symbol:
                            client.ensure_market(symbol)
                    else:
                        # Crypto - use Kraken
                        client = self.kc
                
                self._bots[bot_id] = BotRunner(bot_id, client, self)
            return self._bots[bot_id]

    def start(self, bot_id: int) -> str:
        return self.get_runner(bot_id).start()

    def stop(self, bot_id: int) -> str:
        return self.get_runner(bot_id).stop()

    def snapshot(self, bot_id: int) -> Dict[str, Any]:
        return self.get_runner(bot_id).snapshot()
