"""
Phase 1 Intelligence: Quick Wins
- Trailing Stop Loss
- Cooldown After Stop Loss
- Volatility-Based TP Scaling
- BTC Correlation Guard
- Time-Based Filters
- Adaptive Volume & Spread Guards
"""
import time
from datetime import datetime, timezone, time as dt_time
from typing import Optional, Tuple, Dict, Any
import logging

from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

# US equities: open 09:30 ET, close 16:00 ET. Skip first 30 min (09:30–10:00) and last 30 min (15:30–16:00).
MARKET_OPEN_ET = dt_time(9, 30)
MARKET_CLOSE_ET = dt_time(16, 0)
SKIP_FIRST_END_ET = dt_time(10, 0)   # end of "skip first 30 min" window
SKIP_LAST_START_ET = dt_time(15, 30)  # start of "skip last 30 min" window
# Extended hours: pre-market 4:00-9:30 ET, after-hours 16:00-20:00 ET
PRE_MARKET_START_ET = dt_time(4, 0)
AFTER_HOURS_END_ET = dt_time(20, 0)


def ny_time_now() -> datetime:
    """Return current time in America/New_York timezone."""
    return datetime.now(UTC).astimezone(ET)


def is_regular_market_hours_ny(dt: Optional[datetime] = None) -> bool:
    """True only Mon–Fri, 9:30–16:00 NY time."""
    if dt is None:
        dt = ny_time_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC).astimezone(ET)
    else:
        dt = dt.astimezone(ET)
    wd = dt.weekday()
    if wd >= 5:  # Sat, Sun
        return False
    t = dt.time()
    return MARKET_OPEN_ET <= t < MARKET_CLOSE_ET


def is_within_open_close_avoid_window(dt: Optional[datetime] = None, minutes: int = 30) -> bool:
    """
    True if within first or last N minutes of market session (high volatility).
    - First 30 min: 9:30–10:00 ET
    - Last 30 min: 15:30–16:00 ET
    """
    if dt is None:
        dt = ny_time_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC).astimezone(ET)
    else:
        dt = dt.astimezone(ET)
    t = dt.time()
    # First N minutes after open
    if t >= MARKET_OPEN_ET and t < SKIP_FIRST_END_ET:
        return True
    # Last N minutes before close
    if t >= SKIP_LAST_START_ET and t < MARKET_CLOSE_ET:
        return True
    return False


def is_extended_hours_ny(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Detect if currently in pre-market (4:00-9:30 ET) or after-hours (16:00-20:00 ET).
    Returns: (is_extended, "pre_market"|"after_hours"|"regular"|"closed")
    """
    if dt is None:
        dt = ny_time_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC).astimezone(ET)
    else:
        dt = dt.astimezone(ET)
    wd = dt.weekday()
    if wd >= 5:
        return False, "closed"
    t = dt.time()
    if PRE_MARKET_START_ET <= t < MARKET_OPEN_ET:
        return True, "pre_market"
    if MARKET_CLOSE_ET <= t < AFTER_HOURS_END_ET:
        return True, "after_hours"
    if MARKET_OPEN_ET <= t < MARKET_CLOSE_ET:
        return False, "regular"
    return False, "closed"


def extended_hours_spread_multiplier(is_extended: bool) -> float:
    """Wider spread threshold in extended hours (low liquidity). Default 1.5x."""
    return 1.5 if is_extended else 1.0


# EOD close window: last 15 min of regular session (15:45-16:00 ET)
EOD_CLOSE_START_ET = dt_time(15, 45)


def should_auto_close_eod(dt: Optional[datetime] = None) -> bool:
    """True if within EOD close window (last 15 min of regular session). For AUTO_CLOSE_EOD bots."""
    if dt is None:
        dt = ny_time_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC).astimezone(ET)
    else:
        dt = dt.astimezone(ET)
    wd = dt.weekday()
    if wd >= 5:
        return False
    t = dt.time()
    return EOD_CLOSE_START_ET <= t < MARKET_CLOSE_ET


def is_us_equities_trade_window_ok(
    now_utc: Optional[datetime] = None,
    skip_first_30min: bool = True,
    skip_last_30min: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Pure, unit-testable helper. US equities only.
    Converts now to America/New_York; checks weekend, market hours, skip windows.

    Args:
        now_utc: Current time. Naive treated as UTC. None -> datetime.now(timezone.utc).
        skip_first_30min: Skip 09:30–10:00 ET.
        skip_last_30min: Skip 15:30–16:00 ET.

    Returns:
        (is_trade_window_ok, reason). reason is None when ok.
    """
    if now_utc is None:
        now_utc = datetime.now(UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    et = now_utc.astimezone(ET)
    wd = et.weekday()  # 0 Mon .. 6 Sun
    if wd >= 5:
        return False, "Market closed (weekend)"
    t = et.time()
    if t < MARKET_OPEN_ET or t >= MARKET_CLOSE_ET:
        return False, "Market closed"
    if skip_first_30min and MARKET_OPEN_ET <= t < SKIP_FIRST_END_ET:
        return False, "Market just opened (first 30min volatility)"
    if skip_last_30min and SKIP_LAST_START_ET <= t < MARKET_CLOSE_ET:
        return False, "Market closing soon (last 30min volatility)"
    return True, None


class TrailingStopLoss:
    """
    Trailing Stop Loss - automatically moves stop loss up as price increases
    to lock in profits without manual intervention.
    """
    
    def __init__(
        self,
        activation_pct: float = 0.02,  # Activate after 2% profit
        trail_pct: float = 0.01         # Trail by 1%
    ):
        self.activation_pct = activation_pct
        self.trail_pct = trail_pct
    
    def update(
        self,
        current_price: float,
        entry_price: float,
        highest_price: Optional[float] = None,
        is_active: bool = False
    ) -> Tuple[bool, Optional[float], Optional[float], Optional[str]]:
        """
        Update trailing stop and check if stop hit.
        
        Returns:
            (should_exit, new_highest, new_stop_price, reason)
        """
        if not entry_price or entry_price <= 0:
            return False, highest_price, None, None
        
        # Calculate current profit
        profit_pct = (current_price - entry_price) / entry_price
        
        # Update highest price
        if highest_price is None or current_price > highest_price:
            highest_price = current_price
        
        # Check if we should activate trailing stop
        if profit_pct >= self.activation_pct:
            is_active = True
            
            # Calculate trailing stop price
            stop_price = highest_price * (1 - self.trail_pct)
            
            # Check if current price hit the stop
            if current_price <= stop_price:
                reason = f"Trailing stop hit at ${current_price:.6f} (stop: ${stop_price:.6f}, entry: ${entry_price:.6f}, profit: {profit_pct*100:.2f}%)"
                return True, highest_price, stop_price, reason
            
            return False, highest_price, stop_price, None
        
        return False, highest_price, None, None


class CooldownManager:
    """
    Manages cooldown periods after stop losses to prevent revenge trading.
    """
    
    def __init__(self, cooldown_sec: int = 3600):
        self.cooldown_sec = cooldown_sec
    
    def check_cooldown(self, last_stop_loss_at: Optional[int]) -> Tuple[bool, Optional[str]]:
        """
        Check if bot is in cooldown period.
        
        Returns:
            (can_trade, reason)
        """
        if not last_stop_loss_at:
            return True, None
        
        time_since = time.time() - last_stop_loss_at
        
        if time_since < self.cooldown_sec:
            remaining_sec = self.cooldown_sec - time_since
            remaining_min = int(remaining_sec / 60)
            reason = f"Cooldown active: {remaining_min}min remaining after stop loss"
            return False, reason
        
        return True, None


class VolatilityTPScaler:
    """
    Scales take-profit targets based on current market volatility.
    Higher volatility = larger TP to capture bigger moves.
    """
    
    def __init__(self, base_tp: float, volatility_mult: float = 1.5):
        self.base_tp = base_tp
        self.volatility_mult = volatility_mult
    
    def calculate_adaptive_tp(self, atr_pct: Optional[float]) -> float:
        """
        Calculate adaptive TP based on ATR (volatility measure).
        
        Args:
            atr_pct: ATR as percentage of price (e.g., 0.05 = 5%)
        
        Returns:
            Adjusted TP percentage
        """
        if not atr_pct or atr_pct <= 0:
            return self.base_tp
        
        # Low volatility (<3%): reduce TP to be more realistic
        if atr_pct < 0.03:
            return self.base_tp * 0.5  # 1% TP if base was 2%
        
        # Medium volatility (3-7%): use base TP
        elif atr_pct < 0.07:
            return self.base_tp
        
        # High volatility (>7%): increase TP to capture bigger moves
        else:
            return self.base_tp * self.volatility_mult  # 3% TP if base was 2%


class BTCCorrelationGuard:
    """
    Monitors BTC health and pauses altcoin trading during BTC dumps.
    Prevents cascade liquidations when market leader crashes.
    """
    
    def __init__(
        self,
        dump_threshold: float = 0.05,  # -5% or worse
        weak_threshold: float = 0.03    # -3% to -5%
    ):
        self.dump_threshold = dump_threshold
        self.weak_threshold = weak_threshold
    
    def check_btc_health(
        self,
        btc_1h_change: Optional[float],
        is_altcoin: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if BTC is healthy enough to trade altcoins.
        
        Args:
            btc_1h_change: BTC price change over last hour (e.g., -0.05 = -5%)
            is_altcoin: True if checking for altcoin, False for BTC itself
        
        Returns:
            (can_trade, reason)
        """
        # Always allow BTC trading
        if not is_altcoin:
            return True, None
        
        # No data available - be cautious
        if btc_1h_change is None:
            return True, None  # Allow trade but log warning
        
        # Severe dump: pause all altcoin trading
        if btc_1h_change <= -self.dump_threshold:
            reason = f"BTC dumping {btc_1h_change*100:.1f}% (>5%), pausing alts"
            return False, reason
        
        # Moderate dump: reduce risk
        if btc_1h_change <= -self.weak_threshold:
            reason = f"BTC weak {btc_1h_change*100:.1f}% (-3% to -5%), reducing risk"
            return False, reason
        
        return True, None


class TimeFilter:
    """
    Filters out bad trading times (high volatility periods, low liquidity).
    Uses America/New_York for US equities; skip first/last 30 min.
    """
    
    def __init__(
        self,
        skip_first_30min: bool = True,
        skip_last_30min: bool = True
    ):
        self.skip_first_30min = skip_first_30min
        self.skip_last_30min = skip_last_30min
    
    def should_trade(
        self,
        market_type: str,
        current_time: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if current time is good for trading.
        - If outside market hours -> skip trading actions for stocks.
        - If within first/last 30 minutes -> skip new entries (avoid window).

        Args:
            market_type: "crypto" or "stocks"
            current_time: Current time (defaults to now UTC). Naive treated as UTC.

        Returns:
            (can_trade, reason). reason is None when ok.
        """
        if current_time is None:
            current_time = datetime.now(UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)
        et = current_time.astimezone(ET)

        if market_type == "crypto":
            return True, None

        if market_type in ("stocks", "stock"):
            # Outside market hours -> skip
            if not is_regular_market_hours_ny(et):
                return False, "Market closed"
            # Within first/last 30 min -> skip entries
            if self.skip_first_30min or self.skip_last_30min:
                if is_within_open_close_avoid_window(et, minutes=30):
                    return False, "Market volatility window (first/last 30min)"
            return True, None

        return True, None


class VolumeFilter:
    """
    Ensures sufficient volume before trading to avoid liquidity traps.
    """
    
    def __init__(self, min_volume_ratio: float = 1.5):
        self.min_volume_ratio = min_volume_ratio
    
    def check_volume(
        self,
        current_volume: Optional[float],
        avg_volume_20: Optional[float]
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if current volume is sufficient.
        
        Args:
            current_volume: Current period volume
            avg_volume_20: 20-period average volume
        
        Returns:
            (sufficient_volume, reason)
        """
        if current_volume is None or avg_volume_20 is None:
            return True, None  # No data, allow trade (logged elsewhere)
        
        if avg_volume_20 <= 0:
            return True, None  # Avoid division by zero
        
        ratio = current_volume / avg_volume_20
        
        if ratio < self.min_volume_ratio:
            reason = f"Low volume: {ratio:.1f}x average (need {self.min_volume_ratio:.1f}x)"
            return False, reason
        
        return True, None


class AdaptiveSpreadGuard:
    """
    Dynamically adjusts maximum acceptable spread based on market volatility.
    Tighter spreads in calm markets, wider tolerance in volatile markets.
    """
    
    def __init__(self, base_spread_pct: float = 0.003):
        self.base_spread_pct = base_spread_pct
    
    def get_max_spread(self, volatility_pct: Optional[float]) -> float:
        """
        Calculate maximum acceptable spread based on volatility.
        
        Args:
            volatility_pct: ATR as percentage (e.g., 0.05 = 5%)
        
        Returns:
            Maximum acceptable spread percentage
        """
        if not volatility_pct or volatility_pct <= 0:
            return self.base_spread_pct  # Default 0.3%
        
        # Low volatility (<5%): tight spread
        if volatility_pct < 0.05:
            return self.base_spread_pct  # 0.3%
        
        # Medium volatility (5-10%): moderate spread
        elif volatility_pct < 0.10:
            return 0.006  # 0.6%
        
        # High volatility (>10%): wide spread
        else:
            return 0.010  # 1.0%
    
    def check_spread(
        self,
        bid: Optional[float],
        ask: Optional[float],
        volatility_pct: Optional[float]
    ) -> Tuple[bool, Optional[str], float]:
        """
        Check if spread is acceptable.
        
        Returns:
            (spread_ok, reason, max_spread)
        """
        if not bid or not ask or bid <= 0 or ask <= 0:
            return True, None, self.base_spread_pct
        
        spread_pct = (ask - bid) / bid
        max_spread = self.get_max_spread(volatility_pct)
        
        if spread_pct > max_spread:
            reason = f"Spread too wide: {spread_pct*100:.2f}% (max: {max_spread*100:.2f}%)"
            return False, reason, max_spread
        
        return True, None, max_spread


# =============================================================================
# Phase 1 Intelligence Coordinator
# =============================================================================

class Phase1Intelligence:
    """
    Coordinates all Phase 1 quick-win intelligence features.
    """
    
    def __init__(self, bot_config: Dict[str, Any]):
        """
        Initialize Phase 1 intelligence with bot configuration.
        
        Args:
            bot_config: Bot configuration dict from database
        """
        self.config = bot_config
        
        # Initialize components - scalp (5%/0.2%), long-term (15%/5%), or default (2%/1%)
        day_trading = bool(bot_config.get('day_trading_mode', 0))
        long_term = bool(bot_config.get('long_term_mode', 0))
        if day_trading:
            act_pct = float(bot_config.get('scalp_trailing_activation_pct', 0.05))
            trail_pct = float(bot_config.get('scalp_trailing_distance_pct', 0.002))
        elif long_term:
            act_pct = float(bot_config.get('long_term_trailing_pct', 0.15)) * 0.5  # activation at 7.5%
            trail_pct = float(bot_config.get('long_term_trailing_pct', 0.15)) * 0.33  # trail 5%
        else:
            act_pct = float(bot_config.get('trailing_activation_pct', 0.02))
            trail_pct = float(bot_config.get('trailing_distance_pct', 0.01))
        self.trailing_stop = TrailingStopLoss(
            activation_pct=act_pct,
            trail_pct=trail_pct
        ) if bot_config.get('trailing_stop_enabled', 1) else None
        
        self.cooldown_mgr = CooldownManager(
            cooldown_sec=bot_config.get('stop_loss_cooldown_sec', 3600)
        )
        
        self.tp_scaler = VolatilityTPScaler(
            base_tp=bot_config.get('tp', 0.02),
            volatility_mult=bot_config.get('tp_volatility_mult', 1.5)
        ) if bot_config.get('adaptive_tp_enabled', 1) else None
        
        self.btc_guard = BTCCorrelationGuard(
            dump_threshold=bot_config.get('btc_dump_threshold_pct', 0.05)
        ) if bot_config.get('btc_correlation_guard', 1) else None
        
        self.time_filter = TimeFilter(
            skip_first_30min=bot_config.get('skip_first_30min', 1),
            skip_last_30min=bot_config.get('skip_last_30min', 1)
        ) if bot_config.get('time_filter_enabled', 1) else None
        
        self.volume_filter = VolumeFilter(
            min_volume_ratio=bot_config.get('min_volume_ratio', 1.5)
        )
        
        self.spread_guard = AdaptiveSpreadGuard(
            base_spread_pct=bot_config.get('spread_guard_pct', 0.003)
        ) if bot_config.get('adaptive_spread_enabled', 1) else None
    
    def check_entry_allowed(
        self,
        market_data: Dict[str, Any],
        btc_1h_change: Optional[float] = None
    ) -> Tuple[bool, list]:
        """
        Check if entry is allowed based on all Phase 1 filters.
        
        Args:
            market_data: Dict with keys: symbol, price, bid, ask, volume, avg_volume, atr_pct
            btc_1h_change: BTC 1-hour price change percentage
        
        Returns:
            (allowed, reasons) - reasons list contains blocking reasons
        """
        reasons = []
        
        # 1. Cooldown check
        last_stop_loss = self.config.get('last_stop_loss_at')
        can_trade, reason = self.cooldown_mgr.check_cooldown(last_stop_loss)
        if not can_trade:
            reasons.append(reason)
        
        # 2. Time filter
        if self.time_filter:
            can_trade, reason = self.time_filter.should_trade(
                self.config.get('market_type', 'crypto')
            )
            if not can_trade:
                reasons.append(reason)
        
        # 3. BTC correlation guard
        if self.btc_guard:
            symbol = market_data.get('symbol', '')
            is_altcoin = 'BTC' not in symbol
            can_trade, reason = self.btc_guard.check_btc_health(btc_1h_change, is_altcoin)
            if not can_trade:
                reasons.append(reason)
        
        # 4. Volume filter
        can_trade, reason = self.volume_filter.check_volume(
            market_data.get('volume'),
            market_data.get('avg_volume')
        )
        if not can_trade:
            reasons.append(reason)
        
        # 5. Spread guard
        if self.spread_guard:
            can_trade, reason, _ = self.spread_guard.check_spread(
                market_data.get('bid'),
                market_data.get('ask'),
                market_data.get('atr_pct')
            )
            if not can_trade:
                reasons.append(reason)
        
        # 6. Event awareness: avoid entry day before earnings/Fed
        try:
            from event_calendar import should_avoid_entry
            symbol = market_data.get('symbol', '')
            days_ahead = int(self.config.get('event_avoid_days_ahead', 2))
            avoid, event_reason = should_avoid_entry(symbol=symbol, days_ahead=days_ahead)
            if avoid and event_reason:
                reasons.append(event_reason)
        except Exception:
            pass
        
        allowed = len(reasons) == 0
        return allowed, reasons
    
    def get_adaptive_tp(self, atr_pct: Optional[float]) -> float:
        """
        Get adaptive take-profit percentage based on volatility.
        """
        if self.tp_scaler:
            return self.tp_scaler.calculate_adaptive_tp(atr_pct)
        return self.config.get('tp', 0.02)
    
    def update_trailing_stop(
        self,
        deal: Dict[str, Any],
        current_price: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Update trailing stop for an open deal.
        
        Returns:
            (should_exit, updated_deal_data)
        """
        if not self.trailing_stop:
            return False, {}
        
        should_exit, highest, stop_price, reason = self.trailing_stop.update(
            current_price=current_price,
            entry_price=deal.get('entry_avg'),
            highest_price=deal.get('highest_price'),
            is_active=deal.get('trailing_stop_active', False)
        )
        
        updated_data = {
            'highest_price': highest,
            'trailing_stop_price': stop_price,
            'trailing_stop_active': 1 if stop_price else 0
        }
        
        if should_exit:
            logger.info(f"[TrailingStop] {reason}")
        
        return should_exit, updated_data
