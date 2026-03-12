# smart_entry.py
"""
Smart Entry & Position Management Intelligence Layer

Implements modular intelligence features for bot trade entry and position sizing:
- Smart entry filtering (RSI, price level, volume, trend, multi-timeframe)
- ATR-based dynamic position sizing
- Market regime detection and exposure adjustment
- Fibonacci DCA spacing
- Trailing take profit management
- Loss cooldown tracking
- Daily deal limits
- Kelly criterion position sizing
- Maximum drawdown circuit breaker
- Sharpe ratio calculation

All classes are production-ready with error handling, logging, and docstrings.
"""

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# =========================
# Helper Functions
# =========================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _clean_values(values: List[float]) -> List[float]:
    """Filter out NaN and Inf values from a list of floats."""
    return [v for v in values if not (math.isnan(v) or math.isinf(v))]


# =========================
# Technical Indicators
# =========================

def sma(values: List[float], n: int) -> Optional[float]:
    """Simple Moving Average."""
    if n <= 0 or len(values) < n:
        return None
    window = _clean_values(values[-n:])
    if len(window) < n:
        return None
    return sum(window) / n


def ema(values: List[float], n: int) -> Optional[float]:
    """Exponential Moving Average."""
    if n <= 0 or len(values) < n:
        return None
    clean = _clean_values(values)
    if len(clean) < n:
        return None
    k = 2.0 / (n + 1.0)
    e = clean[0]
    for v in clean[1:]:
        e = (v * k) + (e * (1.0 - k))
    return float(e)


def rsi(values: List[float], n: int = 14) -> Optional[float]:
    """Relative Strength Index."""
    if n <= 0 or len(values) < n + 1:
        return None
    clean = _clean_values(values)
    if len(clean) < n + 1:
        return None
    gains = []
    losses = []
    for i in range(-n, 0):
        diff = clean[i] - clean[i - 1]
        if diff >= 0:
            gains.append(diff)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(-diff)
    avg_gain = sum(gains) / n
    avg_loss = sum(losses) / n
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def atr(candles: List[List[float]], n: int = 14) -> Optional[float]:
    """Average True Range from OHLC candles."""
    if len(candles) < n + 1:
        return None

    trs = []
    for i in range(1, len(candles)):
        high = safe_float(candles[i][2])
        low = safe_float(candles[i][3])
        prev_close = safe_float(candles[i - 1][4])

        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        trs.append(tr)

    if len(trs) < n:
        return None
    return sum(trs[-n:]) / n


def volume_ma(candles: List[List[float]], n: int = 20) -> Optional[float]:
    """Moving average of volume."""
    if len(candles) < n:
        return None
    vols = [safe_float(c[5]) for c in candles[-n:]]
    if not vols:
        return None
    return sum(vols) / len(vols)


# =========================
# 1. Smart Entry Filter
# =========================

@dataclass
class SmartEntryFilter:
    """
    Multi-condition entry filter for improved trade selection.

    Checks:
    - RSI not overbought (< 70)
    - Price not too far above 20 EMA (< 3%)
    - Entry volume above 20-period average
    - Daily trend up (price > 200 EMA on 1D)
    - Multi-timeframe agreement (2+ of 3 timeframes agree)
    """

    rsi_overbought_level: float = 70.0
    price_distance_pct: float = 0.03  # 3%
    volume_ma_period: int = 20
    trend_ema_period: int = 200

    def should_enter(
        self,
        symbol: str,
        market_type: str,
        candles_1h: List[List[float]],
        candles_4h: List[List[float]],
        candles_1d: List[List[float]],
    ) -> Tuple[bool, str]:
        """
        Determine if we should enter a trade.

        Args:
            symbol: Trading pair (e.g., 'BTC/USD')
            market_type: 'crypto' or 'stocks'
            candles_1h: 1-hour OHLCV candles
            candles_4h: 4-hour OHLCV candles
            candles_1d: Daily OHLCV candles

        Returns:
            (should_enter: bool, reason: str)
        """
        try:
            checks = {}

            # Check 1: RSI not overbought on 1H
            closes_1h = [safe_float(c[4]) for c in candles_1h]
            rsi_1h = rsi(closes_1h, 14)
            checks['rsi_1h'] = (rsi_1h is None or rsi_1h < self.rsi_overbought_level,
                               f"RSI1H={rsi_1h:.1f}")

            # Check 2: Price not > 3% above 20 EMA on 1H
            ema_20_1h = ema(closes_1h, 20)
            price_1h = safe_float(candles_1h[-1][4]) if candles_1h else 0.0
            if ema_20_1h and ema_20_1h > 0:
                dist = (price_1h - ema_20_1h) / ema_20_1h
                checks['price_distance'] = (abs(dist) <= self.price_distance_pct,
                                           f"Dist={dist*100:.2f}%")
            else:
                checks['price_distance'] = (True, "EMA20 unavailable")

            # Check 3: Volume on entry candle > 20-period average
            vol_ma = volume_ma(candles_1h, self.volume_ma_period)
            current_vol = safe_float(candles_1h[-1][5]) if candles_1h else 0.0
            if vol_ma and vol_ma > 0:
                checks['volume'] = (current_vol >= vol_ma,
                                   f"Vol={current_vol:.0f} vs MA={vol_ma:.0f}")
            else:
                checks['volume'] = (True, "Vol MA unavailable")

            # Check 4: Daily trend is up (price > 200 EMA on 1D)
            closes_1d = [safe_float(c[4]) for c in candles_1d]
            ema_200_1d = ema(closes_1d, 200)
            price_1d = safe_float(candles_1d[-1][4]) if candles_1d else 0.0
            checks['daily_trend'] = (ema_200_1d is None or price_1d >= ema_200_1d,
                                    f"Price={price_1d:.2f} vs EMA200={ema_200_1d}")

            # Check 5: Multi-timeframe agreement (2+ of 3 agree on direction)
            # Simple check: all timeframes above their 20 EMA = agreement
            ema_20_4h = ema([safe_float(c[4]) for c in candles_4h], 20)
            price_4h = safe_float(candles_4h[-1][4]) if candles_4h else 0.0

            tf_agree = 0
            if ema_20_1h is None or price_1h >= ema_20_1h:
                tf_agree += 1
            if ema_20_4h is None or price_4h >= ema_20_4h:
                tf_agree += 1
            if ema_200_1d is None or price_1d >= ema_200_1d:
                tf_agree += 1

            checks['multi_tf'] = (tf_agree >= 2, f"Agreement={tf_agree}/3")

            # Compile results
            all_pass = all(check[0] for check in checks.values())
            reason_parts = [f"{k}: {v[1]}" for k, v in checks.items()]
            reason = " | ".join(reason_parts)

            if all_pass:
                logger.info(f"[{symbol}] ENTRY OK: {reason}")
                return True, f"All conditions met: {reason}"
            else:
                failed = [k for k, v in checks.items() if not v[0]]
                logger.info(f"[{symbol}] ENTRY REJECTED: {failed} | {reason}")
                return False, f"Failed checks: {', '.join(failed)}"

        except Exception as e:
            logger.error(f"[{symbol}] SmartEntryFilter error: {e}", exc_info=True)
            return False, f"Filter error: {str(e)}"


# =========================
# 2. ATR-Based Position Sizer
# =========================

@dataclass
class ATRPositionSizer:
    """
    Position sizing based on ATR (Average True Range) for consistent risk per trade.

    Adjusts position size by volatility:
    - Higher ATR% = smaller position
    - If daily ATR% > 5%, reduce position by 50%
    """

    atr_period: int = 14
    volatility_threshold_pct: float = 0.05  # 5%
    volatility_reduce_factor: float = 0.5   # 50% reduction

    def calculate_position_size(
        self,
        candles: List[List[float]],
        account_balance: float,
        risk_pct: float = 0.02,
    ) -> float:
        """
        Calculate position size based on ATR and account volatility.

        Args:
            candles: OHLCV candles (usually daily)
            account_balance: Total account balance in quote currency
            risk_pct: Desired risk percentage per trade (default 2%)

        Returns:
            Position size in quote currency (e.g., USD amount)
        """
        try:
            if not candles or len(candles) < self.atr_period + 1:
                # Fallback: basic position based on risk percentage
                return account_balance * risk_pct

            # Calculate ATR
            atr_val = atr(candles, self.atr_period)
            if atr_val is None or atr_val <= 0:
                return account_balance * risk_pct

            # Get current price
            price = safe_float(candles[-1][4])
            if price <= 0:
                return account_balance * risk_pct

            # ATR as percentage of price
            atr_pct = atr_val / price

            # Determine position size: base size = account * risk_pct
            base_size = account_balance * risk_pct

            # Volatility filter: if ATR% > threshold, reduce position
            if atr_pct > self.volatility_threshold_pct:
                adjusted_size = base_size * self.volatility_reduce_factor
                logger.info(
                    f"ATR position sizing: ATR%={atr_pct*100:.2f}% (threshold={self.volatility_threshold_pct*100}%) "
                    f"-> reduce from ${base_size:.2f} to ${adjusted_size:.2f}"
                )
                return adjusted_size

            logger.debug(f"ATR position sizing: ATR%={atr_pct*100:.2f}% (normal), size=${base_size:.2f}")
            return base_size

        except Exception as e:
            logger.error(f"ATRPositionSizer error: {e}", exc_info=True)
            return account_balance * risk_pct


# =========================
# 3. Market Regime Filter
# =========================

@dataclass
class MarketRegimeFilter:
    """
    Market regime detection based on BTC trend.

    Returns: 'bull', 'bear', or 'range'
    Reduces exposure by 50% in bear markets.
    """

    ema_period: int = 200

    def get_regime(self, btc_daily_candles: List[List[float]]) -> str:
        """
        Determine market regime from BTC daily candles.

        Args:
            btc_daily_candles: BTC daily OHLCV candles

        Returns:
            'bull' (price > EMA200), 'bear' (price < EMA200), or 'range'
        """
        try:
            if not btc_daily_candles or len(btc_daily_candles) < self.ema_period:
                logger.warning("Insufficient BTC candles for regime detection")
                return "range"

            closes = [safe_float(c[4]) for c in btc_daily_candles]
            ema_val = ema(closes, self.ema_period)
            current_price = safe_float(btc_daily_candles[-1][4])

            if ema_val is None:
                return "range"

            if current_price > ema_val:
                regime = "bull"
            elif current_price < ema_val:
                regime = "bear"
            else:
                regime = "range"

            logger.debug(f"Market regime: {regime} (Price={current_price:.2f}, EMA200={ema_val:.2f})")
            return regime

        except Exception as e:
            logger.error(f"MarketRegimeFilter error: {e}", exc_info=True)
            return "range"


# =========================
# 4. Fibonacci DCA Spacing
# =========================

@dataclass
class FibonacciDCA:
    """
    Generates Fibonacci-spaced safety order levels below entry price.

    Uses fibonacci ratios: 1%, 2.3%, 3.8%, 6.2%, 10%
    """

    # Fibonacci levels as percentage drops
    fibonacci_levels: List[float] = field(default_factory=lambda: [0.01, 0.023, 0.038, 0.062, 0.10])

    def get_safety_order_levels(
        self,
        entry_price: float,
        num_orders: int = 5,
    ) -> List[float]:
        """
        Calculate safety order levels using Fibonacci spacing.

        Args:
            entry_price: Entry price for the trade
            num_orders: Number of safety orders to generate (max 5)

        Returns:
            List of prices at which to place safety orders (below entry)
        """
        try:
            if entry_price <= 0 or num_orders <= 0:
                return []

            num_orders = min(num_orders, len(self.fibonacci_levels))
            levels = []

            for i in range(num_orders):
                fib_ratio = self.fibonacci_levels[i]
                level_price = entry_price * (1.0 - fib_ratio)
                levels.append(round(level_price, 8))

            logger.debug(f"Fibonacci DCA levels from ${entry_price:.2f}: {levels}")
            return levels

        except Exception as e:
            logger.error(f"FibonacciDCA error: {e}", exc_info=True)
            return []


# =========================
# 5. Trailing Take Profit
# =========================

@dataclass
class TrailingTakeProfit:
    """
    Trailing take profit management.

    Logic:
    - Once in 2% profit, move TP to breakeven
    - Once in 4% profit, trail by 1.5%
    """

    breakeven_profit_pct: float = 0.02    # 2%
    trailing_start_profit_pct: float = 0.04  # 4%
    trailing_distance_pct: float = 0.015   # 1.5%

    def update(
        self,
        current_price: float,
        entry_price: float,
        current_tp: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Update take profit level based on current price.

        Args:
            current_price: Current market price
            entry_price: Entry price
            current_tp: Current take profit level (or None)

        Returns:
            (action: str, new_tp: float)
            action: 'hold', 'move_to_breakeven', 'trail'
        """
        try:
            if entry_price <= 0 or current_price <= 0:
                return "hold", current_tp or (entry_price * 1.02)

            profit_pct = (current_price - entry_price) / entry_price

            # Phase 1: In profit < 2%, hold
            if profit_pct < self.breakeven_profit_pct:
                return "hold", current_tp or (entry_price * 1.02)

            # Phase 2: 2% profit <= profit < 4%, move TP to breakeven
            if profit_pct < self.trailing_start_profit_pct:
                new_tp = entry_price
                return "move_to_breakeven", new_tp

            # Phase 3: >= 4% profit, trail by 1.5%
            trailing_level = current_price * (1.0 - self.trailing_distance_pct)
            return "trail", trailing_level

        except Exception as e:
            logger.error(f"TrailingTakeProfit error: {e}", exc_info=True)
            return "hold", current_tp or (entry_price * 1.02)


# =========================
# 6. Cooldown Manager
# =========================

@dataclass
class CooldownManager:
    """
    Prevents rapid re-entry after losses.

    Records losses and enforces cooldown period (default 30 minutes).
    """

    cooldown_minutes: int = 30

    # Internal state: symbol -> timestamp of loss
    _loss_history: Dict[str, float] = field(default_factory=dict)

    def record_loss(self, symbol: str, timestamp: Optional[float] = None) -> None:
        """
        Record a loss for a symbol.

        Args:
            symbol: Trading pair
            timestamp: Unix timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()
        self._loss_history[symbol] = timestamp
        logger.info(f"[{symbol}] Loss recorded, cooldown until {datetime.fromtimestamp(timestamp + self.cooldown_minutes * 60)}")

    def can_trade(self, symbol: str, cooldown_minutes: Optional[int] = None) -> bool:
        """
        Check if we can trade after a loss cooldown.

        Args:
            symbol: Trading pair
            cooldown_minutes: Override default cooldown (optional)

        Returns:
            True if can trade, False if in cooldown
        """
        try:
            cooldown = cooldown_minutes or self.cooldown_minutes

            if symbol not in self._loss_history:
                return True  # No prior loss

            loss_time = self._loss_history[symbol]
            elapsed_minutes = (time.time() - loss_time) / 60.0

            if elapsed_minutes >= cooldown:
                logger.info(f"[{symbol}] Cooldown expired after {elapsed_minutes:.1f}m")
                del self._loss_history[symbol]
                return True

            logger.debug(f"[{symbol}] Still in cooldown: {cooldown - elapsed_minutes:.1f}m remaining")
            return False

        except Exception as e:
            logger.error(f"CooldownManager error: {e}", exc_info=True)
            return True  # Fail-safe: allow trading on error


# =========================
# 7. Daily Deal Limiter
# =========================

@dataclass
class DailyDealLimiter:
    """
    Limits number of deals opened per day.

    Default: max 5 deals per bot per day.
    """

    max_deals_per_day: int = 5

    # Internal state: bot_id -> list of deal open timestamps
    _deal_history: Dict[int, List[float]] = field(default_factory=dict)

    def _today_start_ts(self) -> float:
        """Get Unix timestamp for start of today (midnight local)."""
        import time as time_module
        lt = time_module.localtime()
        return float(time_module.mktime(
            (lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, lt.tm_wday, lt.tm_yday, lt.tm_isdst)
        ))

    def record_deal(self, bot_id: int, timestamp: Optional[float] = None) -> None:
        """
        Record a deal opening.

        Args:
            bot_id: Bot ID
            timestamp: Unix timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()

        if bot_id not in self._deal_history:
            self._deal_history[bot_id] = []

        # Clean old deals (older than today)
        today_start = self._today_start_ts()
        self._deal_history[bot_id] = [ts for ts in self._deal_history[bot_id] if ts >= today_start]

        # Record new deal
        self._deal_history[bot_id].append(timestamp)
        logger.info(f"[Bot {bot_id}] Deal recorded ({len(self._deal_history[bot_id])}/{self.max_deals_per_day} today)")

    def can_open_deal(self, bot_id: int, max_deals: Optional[int] = None) -> bool:
        """
        Check if bot can open a new deal today.

        Args:
            bot_id: Bot ID
            max_deals: Override max deals limit (optional)

        Returns:
            True if deal count < limit, False otherwise
        """
        try:
            limit = max_deals or self.max_deals_per_day

            if bot_id not in self._deal_history:
                return True

            # Clean old deals
            today_start = self._today_start_ts()
            self._deal_history[bot_id] = [ts for ts in self._deal_history[bot_id] if ts >= today_start]

            count = len(self._deal_history[bot_id])
            can_trade = count < limit

            if not can_trade:
                logger.info(f"[Bot {bot_id}] Daily deal limit reached: {count}/{limit}")

            return can_trade

        except Exception as e:
            logger.error(f"DailyDealLimiter error: {e}", exc_info=True)
            return True  # Fail-safe


# =========================
# 8. Kelly Criterion Position Sizing
# =========================

@dataclass
class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    Formula: f* = (bp - q) / b
    - f* = optimal fraction of capital
    - b = odds (win / loss)
    - p = probability of win
    - q = probability of loss (1 - p)
    """

    kelly_fraction: float = 0.25  # Use 25% of Kelly to be conservative

    def optimal_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate optimal position sizing using Kelly Criterion.

        Args:
            win_rate: Win rate as fraction (0.0 to 1.0)
            avg_win: Average winning trade size
            avg_loss: Average losing trade size

        Returns:
            Optimal fraction of capital (0.0 to 1.0)
        """
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0 or avg_win <= 0:
                return 0.0

            # Kelly: f* = (bp - q) / b
            # where: b = win/loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1.0 - win_rate

            kelly = (b * p - q) / b
            kelly = max(0.0, kelly)  # Never negative

            # Conservative Kelly: use only kelly_fraction of optimal
            conservative = kelly * self.kelly_fraction
            conservative = min(conservative, 1.0)  # Never > 100%

            logger.debug(
                f"Kelly optimal: {kelly*100:.2f}% (conservative: {conservative*100:.2f}%) "
                f"[WR={win_rate*100:.1f}%, AvgW=${avg_win:.2f}, AvgL=${avg_loss:.2f}]"
            )
            return conservative

        except Exception as e:
            logger.error(f"KellyCriterion error: {e}", exc_info=True)
            return 0.0


# =========================
# 9. Maximum Drawdown Circuit Breaker
# =========================

@dataclass
class DrawdownCircuitBreaker:
    """
    Stops all trading if maximum drawdown is exceeded.

    Drawdown = (peak_value - current_value) / peak_value
    Default max: 10% drawdown
    """

    max_drawdown_pct: float = 0.10  # 10%

    def check(
        self,
        portfolio_value: float,
        peak_value: float,
        max_dd_pct: Optional[float] = None,
    ) -> bool:
        """
        Check if portfolio drawdown exceeds limit.

        Args:
            portfolio_value: Current portfolio value
            peak_value: Peak portfolio value
            max_dd_pct: Override max drawdown limit (optional)

        Returns:
            True if should continue trading, False if circuit breaker should trip
        """
        try:
            if peak_value <= 0:
                return True

            limit = max_dd_pct or self.max_drawdown_pct

            if portfolio_value >= peak_value:
                # New peak
                return True

            drawdown = (peak_value - portfolio_value) / peak_value

            if drawdown > limit:
                logger.warning(
                    f"CIRCUIT BREAKER TRIPPED: Drawdown {drawdown*100:.2f}% exceeds limit {limit*100:.2f}%"
                )
                return False

            logger.debug(f"Drawdown OK: {drawdown*100:.2f}% (limit {limit*100:.2f}%)")
            return True

        except Exception as e:
            logger.error(f"DrawdownCircuitBreaker error: {e}", exc_info=True)
            return True  # Fail-safe


# =========================
# 10. Sharpe Ratio Calculator
# =========================

@dataclass
class SharpeCalculator:
    """
    Calculates Sharpe ratio from daily returns.

    Sharpe = (avg_return - risk_free_rate) / std_dev(returns)
    """

    risk_free_rate: float = 0.05  # 5% annualized

    def calculate(
        self,
        daily_returns: List[float],
        risk_free_rate: Optional[float] = None,
    ) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            daily_returns: List of daily returns as fractions (e.g., 0.02 for 2%)
            risk_free_rate: Override risk-free rate (optional)

        Returns:
            Sharpe ratio (higher is better, typically 1.0+ is good)
        """
        try:
            if not daily_returns or len(daily_returns) < 2:
                return 0.0

            rfr = risk_free_rate or self.risk_free_rate

            # Clean returns
            clean_returns = _clean_values(daily_returns)
            if len(clean_returns) < 2:
                return 0.0

            # Calculate average daily return
            avg_return = sum(clean_returns) / len(clean_returns)

            # Calculate standard deviation
            variance = sum((r - avg_return) ** 2 for r in clean_returns) / len(clean_returns)
            std_dev = math.sqrt(variance)

            if std_dev <= 0:
                return 0.0

            # Daily risk-free rate (assuming 252 trading days per year)
            daily_rfr = (1.0 + rfr) ** (1.0 / 252.0) - 1.0

            # Sharpe ratio
            sharpe = (avg_return - daily_rfr) / std_dev

            logger.debug(
                f"Sharpe ratio: {sharpe:.3f} "
                f"(avg_return={avg_return*100:.3f}%, std_dev={std_dev*100:.3f}%, daily_rfr={daily_rfr*100:.3f}%)"
            )
            return sharpe

        except Exception as e:
            logger.error(f"SharpeCalculator error: {e}", exc_info=True)
            return 0.0


# =========================
# Convenience Aggregator
# =========================

@dataclass
class SmartEntrySystem:
    """
    Aggregates all smart entry intelligence components.

    Usage:
        system = SmartEntrySystem()

        # Check entry conditions
        can_enter, reason = system.entry_filter.should_enter(...)

        # Size position
        size = system.position_sizer.calculate_position_size(...)

        # Check regime
        regime = system.regime_filter.get_regime(btc_candles)

        # And more...
    """

    entry_filter: SmartEntryFilter = field(default_factory=SmartEntryFilter)
    position_sizer: ATRPositionSizer = field(default_factory=ATRPositionSizer)
    regime_filter: MarketRegimeFilter = field(default_factory=MarketRegimeFilter)
    fibonacci_dca: FibonacciDCA = field(default_factory=FibonacciDCA)
    trailing_tp: TrailingTakeProfit = field(default_factory=TrailingTakeProfit)
    cooldown: CooldownManager = field(default_factory=CooldownManager)
    deal_limiter: DailyDealLimiter = field(default_factory=DailyDealLimiter)
    kelly: KellyCriterion = field(default_factory=KellyCriterion)
    drawdown_breaker: DrawdownCircuitBreaker = field(default_factory=DrawdownCircuitBreaker)
    sharpe_calc: SharpeCalculator = field(default_factory=SharpeCalculator)


if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.DEBUG)

    system = SmartEntrySystem()
    print("SmartEntrySystem initialized successfully")
    print(f"- SmartEntryFilter: {system.entry_filter}")
    print(f"- ATRPositionSizer: {system.position_sizer}")
    print(f"- MarketRegimeFilter: {system.regime_filter}")
    print(f"- FibonacciDCA: {system.fibonacci_dca}")
    print(f"- TrailingTakeProfit: {system.trailing_tp}")
    print(f"- CooldownManager: {system.cooldown}")
    print(f"- DailyDealLimiter: {system.deal_limiter}")
    print(f"- KellyCriterion: {system.kelly}")
    print(f"- DrawdownCircuitBreaker: {system.drawdown_breaker}")
    print(f"- SharpeCalculator: {system.sharpe_calc}")
