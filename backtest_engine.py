"""
Backtest Engine for Eirin Bot
Professional-grade backtesting with multiple strategies, indicators, and analytics.
"""

import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json


# =========================================================
# Dataclasses for Results
# =========================================================

@dataclass
class Trade:
    """Single completed trade."""
    entry_time: int
    exit_time: int
    entry_price: float
    exit_price: float
    pnl_usd: float
    pnl_pct: float
    side: str  # "long" or "short"
    duration_hours: float
    position_size: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BacktestResult:
    """Complete backtest results."""
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    total_return_pct: float
    total_return_usd: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_trade_duration_hours: float
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'strategy': self.strategy,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_return_pct': self.total_return_pct,
            'total_return_usd': self.total_return_usd,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win_pct': self.avg_win_pct,
            'avg_loss_pct': self.avg_loss_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'profit_factor': self.profit_factor,
            'avg_trade_duration_hours': self.avg_trade_duration_hours,
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'monthly_returns': self.monthly_returns,
        }


# =========================================================
# Indicator Calculations
# =========================================================

def calculate_rsi(closes: List[float], period: int = 14) -> List[float]:
    """Calculate Relative Strength Index."""
    if len(closes) < period + 1:
        return [None] * len(closes)

    rsis = [None] * len(closes)
    for i in range(period, len(closes)):
        deltas = [closes[j] - closes[j-1] for j in range(i - period + 1, i + 1)]
        gains = sum(d for d in deltas if d > 0)
        losses = sum(-d for d in deltas if d < 0)
        avg_gain = gains / period
        avg_loss = losses / period

        if avg_loss == 0:
            rsis[i] = 100 if avg_gain > 0 else 50
        else:
            rs = avg_gain / avg_loss
            rsis[i] = 100 - (100 / (1 + rs))

    return rsis


def calculate_ema(closes: List[float], period: int = 20) -> List[float]:
    """Calculate Exponential Moving Average."""
    if len(closes) == 0:
        return []

    emas = []
    multiplier = 2.0 / (period + 1)
    ema = closes[0]
    emas.append(ema)

    for i in range(1, len(closes)):
        ema = closes[i] * multiplier + ema * (1 - multiplier)
        emas.append(ema)

    return emas


def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    """Calculate Average True Range."""
    if len(closes) < period:
        return [None] * len(closes)

    true_ranges = []
    for i in range(len(closes)):
        if i == 0:
            true_range = highs[i] - lows[i]
        else:
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_range = max(tr1, tr2, tr3)
        true_ranges.append(true_range)

    atrs = [None] * period
    atr = sum(true_ranges[:period]) / period
    atrs.append(atr)

    for i in range(period + 1, len(true_ranges)):
        atr = (atr * (period - 1) + true_ranges[i]) / period
        atrs.append(atr)

    return atrs


def calculate_bollinger_bands(closes: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
    """Calculate Bollinger Bands (middle, upper, lower)."""
    if len(closes) < period:
        return [None] * len(closes), [None] * len(closes), [None] * len(closes)

    middles = []
    uppers = []
    lowers = []

    for i in range(len(closes)):
        if i < period - 1:
            middles.append(None)
            uppers.append(None)
            lowers.append(None)
        else:
            window = closes[i - period + 1:i + 1]
            middle = sum(window) / period
            variance = sum((x - middle) ** 2 for x in window) / period
            std = math.sqrt(variance)
            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)

            middles.append(middle)
            uppers.append(upper)
            lowers.append(lower)

    return middles, uppers, lowers


def calculate_macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
    """Calculate MACD (macd line, signal line, histogram)."""
    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)

    macd_line = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    signal_line = calculate_ema([x for x in macd_line if x is not None], signal) if any(x is not None for x in macd_line) else [None] * len(closes)

    # Pad signal_line to match length
    padded_signal = [None] * (len(closes) - len(signal_line)) + signal_line

    histogram = [None] * len(closes)
    for i in range(len(closes)):
        if macd_line[i] is not None and padded_signal[i] is not None:
            histogram[i] = macd_line[i] - padded_signal[i]

    return macd_line, padded_signal, histogram


# =========================================================
# BacktestEngine
# =========================================================

class BacktestEngine:
    """Professional backtesting engine with multiple strategies."""

    def __init__(self, symbol: str, candles: List[Dict[str, float]], strategy_params: Dict[str, Any]):
        """
        Initialize backtest engine.

        Args:
            symbol: Trading pair (e.g., "BTC/USD")
            candles: List of {time, open, high, low, close, volume}
            strategy_params: Dict with strategy config
        """
        self.symbol = symbol
        self.candles = candles
        self.params = strategy_params

        # Extract strategy name
        self.strategy = strategy_params.get("strategy", "dca").lower()

        # Default params
        self.initial_capital = 1000.0
        self.trading_fee_pct = 0.1  # 0.1% per order
        self.slippage_pct = 0.05   # 0.05% slippage

        # Strategy params
        self.base_quote = float(strategy_params.get("base_quote", 100.0))
        self.tp_pct = float(strategy_params.get("tp_pct", 1.5))
        self.sl_pct = float(strategy_params.get("sl_pct", 0.0))
        self.max_safety = int(strategy_params.get("max_safety", 5))
        self.safety_quote = float(strategy_params.get("safety_quote", 25.0))
        self.first_dev = float(strategy_params.get("first_dev", 5.0))
        self.step_mult = float(strategy_params.get("step_mult", 1.5))

        # Indicator params
        self.rsi_period = int(strategy_params.get("rsi_period", 14))
        self.ema_fast = int(strategy_params.get("ema_fast", 20))
        self.ema_slow = int(strategy_params.get("ema_slow", 50))
        self.trailing_tp = bool(strategy_params.get("trailing_tp", False))

        # Indicators (calculated on demand)
        self.rsis = []
        self.ema_fast_vals = []
        self.ema_slow_vals = []
        self.atrs = []
        self.bb_middles = []
        self.bb_uppers = []
        self.bb_lowers = []
        self.macds = []
        self.macd_signals = []

    def _extract_candle_data(self) -> Tuple[List[int], List[float], List[float], List[float], List[float], List[float]]:
        """Extract OHLCV data from candles."""
        times = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []

        for c in self.candles:
            times.append(int(c.get("time", 0)))
            opens.append(float(c.get("open", 0)))
            highs.append(float(c.get("high", 0)))
            lows.append(float(c.get("low", 0)))
            closes.append(float(c.get("close", 0)))
            volumes.append(float(c.get("volume", 0)))

        return times, opens, highs, lows, closes, volumes

    def _calculate_indicators(self):
        """Calculate all technical indicators."""
        times, opens, highs, lows, closes, volumes = self._extract_candle_data()

        self.rsis = calculate_rsi(closes, self.rsi_period)
        self.ema_fast_vals = calculate_ema(closes, self.ema_fast)
        self.ema_slow_vals = calculate_ema(closes, self.ema_slow)
        self.atrs = calculate_atr(highs, lows, closes, 14)
        self.bb_middles, self.bb_uppers, self.bb_lowers = calculate_bollinger_bands(closes, 20, 2.0)
        self.macds, self.macd_signals, _ = calculate_macd(closes, 12, 26, 9)

    def run(self) -> BacktestResult:
        """Run the backtest simulation."""
        self._calculate_indicators()

        if self.strategy == "dca":
            trades = self._simulate_dca()
        elif self.strategy == "trend_follow":
            trades = self._simulate_trend_follow()
        elif self.strategy == "mean_reversion":
            trades = self._simulate_mean_reversion()
        elif self.strategy == "momentum":
            trades = self._simulate_momentum()
        elif self.strategy == "grid":
            trades = self._simulate_grid()
        else:
            trades = []

        return self._calculate_results(trades)

    def _simulate_dca(self) -> List[Trade]:
        """Simulate Dollar Cost Averaging with safety orders."""
        trades = []
        times, opens, highs, lows, closes, volumes = self._extract_candle_data()

        cash = self.initial_capital
        position = 0.0
        avg_entry = None
        safety_count = 0
        entry_price = None
        entry_time = None
        safety_orders = []

        for i in range(len(closes)):
            if i < 50:  # Warm-up period
                continue

            price = closes[i]
            rsi = self.rsis[i] if i < len(self.rsis) else None

            # Entry conditions: RSI < 35 (oversold) and EMA cross
            if position == 0 and rsi is not None:
                ema_fast = self.ema_fast_vals[i] if i < len(self.ema_fast_vals) else None
                ema_slow = self.ema_slow_vals[i] if i < len(self.ema_slow_vals) else None

                if ema_fast and ema_slow and rsi < 35 and ema_fast < ema_slow:
                    # Entry signal
                    cost = self.base_quote * (1 + self.slippage_pct / 100.0)
                    fee = cost * (self.trading_fee_pct / 100.0)
                    total_cost = cost + fee

                    if cash >= total_cost and price > 0:
                        size = (self.base_quote / price)
                        cash -= total_cost
                        position += size
                        avg_entry = price
                        entry_price = price
                        entry_time = times[i]
                        safety_count = 0
                        safety_orders = []

            # Safety orders: on 5%, 10%, 15% below average entry
            elif position > 0 and safety_count < self.max_safety:
                deviation_pct = (avg_entry - price) / avg_entry * 100.0
                safety_levels = [
                    self.first_dev * (self.step_mult ** j) for j in range(self.max_safety)
                ]

                for j, level in enumerate(safety_levels):
                    if j == safety_count and deviation_pct >= level:
                        cost = self.safety_quote * (1 + self.slippage_pct / 100.0)
                        fee = cost * (self.trading_fee_pct / 100.0)
                        total_cost = cost + fee

                        if cash >= total_cost and price > 0:
                            size = (self.safety_quote / price)
                            total_spent = (position * avg_entry) + total_cost
                            position += size
                            avg_entry = total_spent / position
                            cash -= total_cost
                            safety_count += 1
                            safety_orders.append({"level": level, "price": price})

            # Take profit: TP% above average entry
            if position > 0 and entry_price and entry_time:
                tp_target = avg_entry * (1 + self.tp_pct / 100.0)

                if price >= tp_target or (self.sl_pct > 0 and price <= avg_entry * (1 - self.sl_pct / 100.0)):
                    # Exit position
                    exit_price = price * (1 - self.slippage_pct / 100.0)
                    proceeds = position * exit_price
                    fee = proceeds * (self.trading_fee_pct / 100.0)
                    cash += proceeds - fee

                    pnl_usd = proceeds - fee - (position * avg_entry)
                    pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0 if position * avg_entry > 0 else 0.0

                    duration_hours = (times[i] - entry_time) / 3600.0 if entry_time else 0.0

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=times[i],
                        entry_price=avg_entry,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        side="long",
                        duration_hours=duration_hours,
                        position_size=position,
                    ))

                    position = 0.0
                    avg_entry = None
                    entry_price = None
                    entry_time = None
                    safety_count = 0
                    safety_orders = []

        # Close any open position at end
        if position > 0 and entry_time:
            exit_price = closes[-1]
            proceeds = position * exit_price
            fee = proceeds * (self.trading_fee_pct / 100.0)
            pnl_usd = proceeds - fee - (position * avg_entry)
            pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0 if position * avg_entry > 0 else 0.0
            duration_hours = (times[-1] - entry_time) / 3600.0

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=times[-1],
                entry_price=avg_entry,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                side="long",
                duration_hours=duration_hours,
                position_size=position,
            ))

        return trades

    def _simulate_trend_follow(self) -> List[Trade]:
        """Simulate trend following strategy with EMA crossover."""
        trades = []
        times, opens, highs, lows, closes, volumes = self._extract_candle_data()

        cash = self.initial_capital
        position = 0.0
        avg_entry = None
        entry_time = None

        for i in range(len(closes)):
            if i < self.ema_slow:
                continue

            price = closes[i]
            ema_fast = self.ema_fast_vals[i]
            ema_slow = self.ema_slow_vals[i]

            # Entry: fast EMA crosses above slow EMA
            if position == 0 and ema_fast and ema_slow:
                if i > 0 and self.ema_fast_vals[i-1] <= self.ema_slow_vals[i-1] and ema_fast > ema_slow:
                    cost = self.base_quote * (1 + self.slippage_pct / 100.0)
                    fee = cost * (self.trading_fee_pct / 100.0)
                    total_cost = cost + fee

                    if cash >= total_cost and price > 0:
                        size = self.base_quote / price
                        cash -= total_cost
                        position = size
                        avg_entry = price
                        entry_time = times[i]

            # Exit: fast EMA crosses below slow EMA or TP/SL hit
            elif position > 0 and entry_time:
                tp_target = avg_entry * (1 + self.tp_pct / 100.0)
                sl_target = avg_entry * (1 - self.sl_pct / 100.0) if self.sl_pct > 0 else 0

                exit_signal = (
                    (i > 0 and self.ema_fast_vals[i-1] >= self.ema_slow_vals[i-1] and ema_fast < ema_slow) or
                    price >= tp_target or
                    (self.sl_pct > 0 and price <= sl_target)
                )

                if exit_signal:
                    exit_price = price * (1 - self.slippage_pct / 100.0)
                    proceeds = position * exit_price
                    fee = proceeds * (self.trading_fee_pct / 100.0)
                    cash += proceeds - fee

                    pnl_usd = proceeds - fee - (position * avg_entry)
                    pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0 if position * avg_entry > 0 else 0.0
                    duration_hours = (times[i] - entry_time) / 3600.0

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=times[i],
                        entry_price=avg_entry,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        side="long",
                        duration_hours=duration_hours,
                        position_size=position,
                    ))

                    position = 0.0
                    avg_entry = None
                    entry_time = None

        # Close open position
        if position > 0 and entry_time:
            exit_price = closes[-1]
            proceeds = position * exit_price
            fee = proceeds * (self.trading_fee_pct / 100.0)
            pnl_usd = proceeds - fee - (position * avg_entry)
            pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0
            duration_hours = (times[-1] - entry_time) / 3600.0

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=times[-1],
                entry_price=avg_entry,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                side="long",
                duration_hours=duration_hours,
                position_size=position,
            ))

        return trades

    def _simulate_mean_reversion(self) -> List[Trade]:
        """Simulate mean reversion strategy with Bollinger Bands."""
        trades = []
        times, opens, highs, lows, closes, volumes = self._extract_candle_data()

        cash = self.initial_capital
        position = 0.0
        avg_entry = None
        entry_time = None

        for i in range(len(closes)):
            if i < 50:
                continue

            price = closes[i]
            bb_lower = self.bb_lowers[i]
            bb_middle = self.bb_middles[i]
            bb_upper = self.bb_uppers[i]

            # Entry: price below lower Bollinger Band
            if position == 0 and bb_lower and price < bb_lower:
                cost = self.base_quote * (1 + self.slippage_pct / 100.0)
                fee = cost * (self.trading_fee_pct / 100.0)
                total_cost = cost + fee

                if cash >= total_cost and price > 0:
                    size = self.base_quote / price
                    cash -= total_cost
                    position = size
                    avg_entry = price
                    entry_time = times[i]

            # Exit: price above middle or TP/SL
            elif position > 0 and entry_time:
                tp_target = avg_entry * (1 + self.tp_pct / 100.0)
                sl_target = avg_entry * (1 - self.sl_pct / 100.0) if self.sl_pct > 0 else 0

                exit_signal = (
                    (bb_middle and price >= bb_middle) or
                    price >= tp_target or
                    (self.sl_pct > 0 and price <= sl_target)
                )

                if exit_signal:
                    exit_price = price * (1 - self.slippage_pct / 100.0)
                    proceeds = position * exit_price
                    fee = proceeds * (self.trading_fee_pct / 100.0)
                    cash += proceeds - fee

                    pnl_usd = proceeds - fee - (position * avg_entry)
                    pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0
                    duration_hours = (times[i] - entry_time) / 3600.0

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=times[i],
                        entry_price=avg_entry,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        side="long",
                        duration_hours=duration_hours,
                        position_size=position,
                    ))

                    position = 0.0
                    avg_entry = None
                    entry_time = None

        if position > 0 and entry_time:
            exit_price = closes[-1]
            proceeds = position * exit_price
            fee = proceeds * (self.trading_fee_pct / 100.0)
            pnl_usd = proceeds - fee - (position * avg_entry)
            pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0
            duration_hours = (times[-1] - entry_time) / 3600.0

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=times[-1],
                entry_price=avg_entry,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                side="long",
                duration_hours=duration_hours,
                position_size=position,
            ))

        return trades

    def _simulate_momentum(self) -> List[Trade]:
        """Simulate momentum strategy with breakout."""
        trades = []
        times, opens, highs, lows, closes, volumes = self._extract_candle_data()

        cash = self.initial_capital
        position = 0.0
        avg_entry = None
        entry_time = None
        lookback = 20

        for i in range(len(closes)):
            if i < lookback:
                continue

            price = closes[i]
            highest_high = max(highs[i-lookback:i])

            # Entry: breakout above highest high
            if position == 0 and price >= highest_high:
                cost = self.base_quote * (1 + self.slippage_pct / 100.0)
                fee = cost * (self.trading_fee_pct / 100.0)
                total_cost = cost + fee

                if cash >= total_cost and price > 0:
                    size = self.base_quote / price
                    cash -= total_cost
                    position = size
                    avg_entry = price
                    entry_time = times[i]

            # Exit: TP or SL
            elif position > 0 and entry_time:
                tp_target = avg_entry * (1 + self.tp_pct / 100.0)
                sl_target = avg_entry * (1 - self.sl_pct / 100.0) if self.sl_pct > 0 else 0

                exit_signal = (
                    price >= tp_target or
                    (self.sl_pct > 0 and price <= sl_target)
                )

                if exit_signal:
                    exit_price = price * (1 - self.slippage_pct / 100.0)
                    proceeds = position * exit_price
                    fee = proceeds * (self.trading_fee_pct / 100.0)
                    cash += proceeds - fee

                    pnl_usd = proceeds - fee - (position * avg_entry)
                    pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0
                    duration_hours = (times[i] - entry_time) / 3600.0

                    trades.append(Trade(
                        entry_time=entry_time,
                        exit_time=times[i],
                        entry_price=avg_entry,
                        exit_price=exit_price,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        side="long",
                        duration_hours=duration_hours,
                        position_size=position,
                    ))

                    position = 0.0
                    avg_entry = None
                    entry_time = None

        if position > 0 and entry_time:
            exit_price = closes[-1]
            proceeds = position * exit_price
            fee = proceeds * (self.trading_fee_pct / 100.0)
            pnl_usd = proceeds - fee - (position * avg_entry)
            pnl_pct = (pnl_usd / (position * avg_entry)) * 100.0
            duration_hours = (times[-1] - entry_time) / 3600.0

            trades.append(Trade(
                entry_time=entry_time,
                exit_time=times[-1],
                entry_price=avg_entry,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                side="long",
                duration_hours=duration_hours,
                position_size=position,
            ))

        return trades

    def _simulate_grid(self) -> List[Trade]:
        """Simulate grid trading strategy."""
        trades = []
        times, opens, highs, lows, closes, volumes = self._extract_candle_data()

        cash = self.initial_capital
        grid_levels = {}  # {price_level: position_size}
        all_positions = []  # Track all positions for closing

        grid_low = min(closes[:100])
        grid_high = max(closes[:100])
        num_levels = 10
        level_size = (grid_high - grid_low) / num_levels

        for i in range(len(closes)):
            price = closes[i]

            # Buy at grid levels below current price
            for level in range(int(grid_low), int(grid_high), int(level_size)):
                if level not in grid_levels and price <= level:
                    cost = self.base_quote * (1 + self.slippage_pct / 100.0)
                    fee = cost * (self.trading_fee_pct / 100.0)
                    total_cost = cost + fee

                    if cash >= total_cost and price > 0:
                        size = self.base_quote / price
                        cash -= total_cost
                        grid_levels[level] = size
                        all_positions.append({
                            'entry_time': times[i],
                            'entry_price': price,
                            'size': size,
                            'level': level,
                        })

            # Sell at grid levels above current price
            levels_to_close = [l for l in grid_levels if l < price]
            for level in levels_to_close:
                size = grid_levels[level]
                exit_price = price * (1 - self.slippage_pct / 100.0)
                proceeds = size * exit_price
                fee = proceeds * (self.trading_fee_pct / 100.0)
                cash += proceeds - fee

                entry_price = level
                pnl_usd = proceeds - fee - (size * entry_price)
                pnl_pct = (pnl_usd / (size * entry_price)) * 100.0
                duration_hours = (times[i] - all_positions[0]['entry_time']) / 3600.0 if all_positions else 0

                trades.append(Trade(
                    entry_time=times[i-1] if i > 0 else times[0],
                    exit_time=times[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    pnl_pct=pnl_pct,
                    side="long",
                    duration_hours=duration_hours,
                    position_size=size,
                ))

                del grid_levels[level]

        # Close remaining positions
        for level, size in grid_levels.items():
            exit_price = closes[-1]
            proceeds = size * exit_price
            fee = proceeds * (self.trading_fee_pct / 100.0)

            entry_price = level
            pnl_usd = proceeds - fee - (size * entry_price)
            pnl_pct = (pnl_usd / (size * entry_price)) * 100.0
            duration_hours = (times[-1] - times[0]) / 3600.0

            trades.append(Trade(
                entry_time=times[0],
                exit_time=times[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_usd=pnl_usd,
                pnl_pct=pnl_pct,
                side="long",
                duration_hours=duration_hours,
                position_size=size,
            ))

        return trades

    def _calculate_results(self, trades: List[Trade]) -> BacktestResult:
        """Calculate backtest result metrics."""
        times, opens, highs, lows, closes, volumes = self._extract_candle_data()

        # Basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl_usd > 0])
        losing_trades = len([t for t in trades if t.pnl_usd < 0])

        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100.0
            avg_win_pct = sum(t.pnl_pct for t in trades if t.pnl_usd > 0) / max(1, winning_trades)
            avg_loss_pct = sum(t.pnl_pct for t in trades if t.pnl_usd < 0) / max(1, losing_trades)
            avg_trade_duration = sum(t.duration_hours for t in trades) / total_trades
        else:
            win_rate = 0.0
            avg_win_pct = 0.0
            avg_loss_pct = 0.0
            avg_trade_duration = 0.0

        # P&L
        total_pnl_usd = sum(t.pnl_usd for t in trades)
        total_return_pct = (total_pnl_usd / self.initial_capital) * 100.0

        # Profit factor
        wins = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
        losses = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
        profit_factor = wins / losses if losses > 0 else (1.0 if wins > 0 else 0.0)

        # Equity curve
        equity_curve = []
        equity = self.initial_capital
        trade_dict = {t.exit_time: t for t in trades}

        for i, time in enumerate(times):
            if time in trade_dict:
                t = trade_dict[time]
                equity += t.pnl_usd

            equity_curve.append({
                'time': time,
                'value': equity,
            })

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for ec in equity_curve:
            if ec['value'] > peak:
                peak = ec['value']
            dd = (peak - ec['value']) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe and Sortino ratios
        returns = []
        for i in range(1, len(equity_curve)):
            prev = equity_curve[i-1]['value']
            cur = equity_curve[i]['value']
            if prev > 0:
                returns.append((cur - prev) / prev)

        if len(returns) > 1:
            mean_ret = sum(returns) / len(returns)
            std_ret = statistics.pstdev(returns)
            sharpe_ratio = (mean_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0.0

            neg_returns = [r for r in returns if r < 0]
            if len(neg_returns) > 0:
                neg_std = statistics.pstdev(neg_returns)
                sortino_ratio = (mean_ret / neg_std * math.sqrt(252)) if neg_std > 0 else 0.0
            else:
                sortino_ratio = sharpe_ratio
        else:
            sharpe_ratio = 0.0
            sortino_ratio = 0.0

        # Monthly returns
        monthly_returns = {}
        if len(times) > 0:
            for t in trades:
                month_key = datetime.fromtimestamp(t.exit_time).strftime("%Y-%m")
                monthly_returns[month_key] = monthly_returns.get(month_key, 0.0) + t.pnl_pct

        # Date range
        start_date = datetime.fromtimestamp(times[0]).strftime("%Y-%m-%d") if times else ""
        end_date = datetime.fromtimestamp(times[-1]).strftime("%Y-%m-%d") if times else ""

        return BacktestResult(
            symbol=self.symbol,
            strategy=self.strategy,
            start_date=start_date,
            end_date=end_date,
            total_return_pct=total_return_pct,
            total_return_usd=total_pnl_usd,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,
            max_drawdown_pct=max_dd * 100.0,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            avg_trade_duration_hours=avg_trade_duration,
            equity_curve=equity_curve,
            trades=[t.to_dict() for t in trades],
            monthly_returns=monthly_returns,
        )


    def walk_forward_test(
        self,
        candles: List[dict],
        strategy: str = "dca",
        initial_capital: float = 10000.0,
        is_ratio: float = 0.7,
        n_windows: int = 5,
        param_grid: Optional[Dict[str, List]] = None,
    ) -> dict:
        """
        Walk-Forward Analysis to prevent curve fitting.

        Splits data into rolling windows of in-sample (training) and
        out-of-sample (testing) periods. Optimizes parameters on IS,
        validates on OOS.

        Args:
            candles: Full historical OHLCV data
            strategy: Strategy name
            initial_capital: Starting capital per window
            is_ratio: Ratio of in-sample to total window size
            n_windows: Number of rolling windows
            param_grid: Parameter grid for optimization (uses defaults if None)

        Returns:
            {
                "strategy": str,
                "n_windows": int,
                "is_ratio": float,
                "windows": [
                    {
                        "window_idx": int,
                        "is_start": int, "is_end": int,
                        "oos_start": int, "oos_end": int,
                        "best_params": dict,
                        "is_return_pct": float,
                        "oos_return_pct": float,
                        "is_sharpe": float,
                        "oos_sharpe": float,
                        "is_max_drawdown": float,
                        "oos_max_drawdown": float,
                        "is_trades": int,
                        "oos_trades": int,
                    }
                ],
                "aggregate": {
                    "avg_oos_return": float,
                    "avg_oos_sharpe": float,
                    "avg_oos_drawdown": float,
                    "total_oos_trades": int,
                    "is_vs_oos_degradation": float,
                    "robustness_score": float,
                },
                "conclusion": str,
            }
        """
        from itertools import product

        # Set default param grid if not provided
        if param_grid is None:
            param_grid = {
                "dca": {"base_quote": [50, 100, 150], "tp_pct": [1.0, 1.5, 2.0]},
                "trend_follow": {"ema_fast": [9, 12, 20], "ema_slow": [26, 50, 100]},
                "mean_reversion": {"rsi_period": [10, 14, 20]},
                "momentum": {"tp_pct": [1.5, 2.0, 2.5]},
                "grid": {"base_quote": [75, 100, 125]},
            }.get(strategy, {"base_quote": [75, 100, 125]})

        total_candles = len(candles)
        if total_candles < 100:
            return {
                "error": "Insufficient data (need at least 100 candles)",
                "strategy": strategy,
                "n_windows": n_windows,
            }

        # Calculate window sizes for rolling windows
        step_size = total_candles // (n_windows + 1)
        is_size = int(step_size * (1 + is_ratio))
        oos_size = step_size

        windows_data = []

        # Walk forward through each window
        for window_idx in range(n_windows):
            is_start = window_idx * step_size
            is_end = is_start + is_size
            oos_start = is_end
            oos_end = oos_start + oos_size

            # Boundary checks
            if oos_end > total_candles:
                oos_end = total_candles
            if is_end > total_candles:
                break

            is_candles = candles[is_start:is_end]
            oos_candles = candles[oos_start:oos_end]

            if len(is_candles) < 50 or len(oos_candles) < 20:
                continue

            # Step 1: Optimize on IS data
            best_params = None
            best_sharpe = -float('inf')

            keys = list(param_grid.keys())
            value_lists = [param_grid[k] for k in keys]

            for values in product(*value_lists):
                param_combo = dict(zip(keys, values))
                param_combo["strategy"] = strategy
                param_combo["base_quote"] = param_combo.get("base_quote", 100.0)
                param_combo["tp_pct"] = param_combo.get("tp_pct", 1.5)
                param_combo["sl_pct"] = param_combo.get("sl_pct", 0.0)
                param_combo["ema_fast"] = param_combo.get("ema_fast", 20)
                param_combo["ema_slow"] = param_combo.get("ema_slow", 50)
                param_combo["rsi_period"] = param_combo.get("rsi_period", 14)

                is_engine = BacktestEngine(self.symbol, is_candles, param_combo)
                is_result = is_engine.run()

                if is_result.sharpe_ratio > best_sharpe:
                    best_sharpe = is_result.sharpe_ratio
                    best_params = param_combo.copy()

            if best_params is None:
                continue

            # Step 2: Test on OOS data with best params
            oos_engine = BacktestEngine(self.symbol, oos_candles, best_params)
            oos_result = oos_engine.run()

            # Step 3: Re-run on IS with best params for consistency
            is_engine_final = BacktestEngine(self.symbol, is_candles, best_params)
            is_result_final = is_engine_final.run()

            windows_data.append({
                "window_idx": window_idx,
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "best_params": {k: v for k, v in best_params.items() if k != "strategy"},
                "is_return_pct": is_result_final.total_return_pct,
                "oos_return_pct": oos_result.total_return_pct,
                "is_sharpe": is_result_final.sharpe_ratio,
                "oos_sharpe": oos_result.sharpe_ratio,
                "is_max_drawdown": is_result_final.max_drawdown_pct,
                "oos_max_drawdown": oos_result.max_drawdown_pct,
                "is_trades": is_result_final.total_trades,
                "oos_trades": oos_result.total_trades,
            })

        # Calculate aggregate statistics
        if not windows_data:
            return {
                "error": "No valid windows generated",
                "strategy": strategy,
                "n_windows": n_windows,
            }

        oos_returns = [w["oos_return_pct"] for w in windows_data]
        oos_sharpes = [w["oos_sharpe"] for w in windows_data]
        oos_drawdowns = [w["oos_max_drawdown"] for w in windows_data]
        is_returns = [w["is_return_pct"] for w in windows_data]

        avg_oos_return = sum(oos_returns) / len(oos_returns) if oos_returns else 0.0
        avg_oos_sharpe = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0.0
        avg_oos_drawdown = sum(oos_drawdowns) / len(oos_drawdowns) if oos_drawdowns else 0.0
        avg_is_return = sum(is_returns) / len(is_returns) if is_returns else 0.0
        total_oos_trades = sum(w["oos_trades"] for w in windows_data)

        # Degradation: OOS return vs IS return (lower = more curve fitting)
        is_vs_oos_degradation = avg_oos_return / avg_is_return if avg_is_return > 0 else 0.0

        # Robustness score: based on consistency and low drawdown
        # Score ranges 0-1, higher = more robust
        return_consistency = 1.0 - (statistics.pstdev(oos_returns) / (abs(avg_oos_return) + 1.0)) if len(oos_returns) > 1 else 0.5
        return_consistency = max(0.0, min(1.0, return_consistency))

        drawdown_score = max(0.0, 1.0 - (avg_oos_drawdown / 50.0))  # Penalize large drawdowns

        robustness_score = (return_consistency * 0.6 + drawdown_score * 0.4)
        robustness_score = max(0.0, min(1.0, robustness_score))

        # Generate conclusion
        conclusion_parts = []
        if robustness_score > 0.7:
            conclusion_parts.append("Strategy shows good robustness.")
        elif robustness_score > 0.5:
            conclusion_parts.append("Strategy shows moderate robustness.")
        else:
            conclusion_parts.append("Strategy robustness is low - may be over-fitted.")

        if is_vs_oos_degradation > 0.8:
            conclusion_parts.append("OOS performance closely matches IS (low overfitting).")
        elif is_vs_oos_degradation > 0.5:
            conclusion_parts.append("Some degradation from IS to OOS performance detected.")
        else:
            conclusion_parts.append("Significant degradation from IS to OOS - likely over-fitted.")

        if avg_oos_sharpe > 1.0:
            conclusion_parts.append("Risk-adjusted returns are strong.")
        elif avg_oos_sharpe > 0.5:
            conclusion_parts.append("Risk-adjusted returns are moderate.")

        conclusion = " ".join(conclusion_parts)

        return {
            "strategy": strategy,
            "n_windows": n_windows,
            "is_ratio": is_ratio,
            "windows": windows_data,
            "aggregate": {
                "avg_oos_return": avg_oos_return,
                "avg_oos_sharpe": avg_oos_sharpe,
                "avg_oos_drawdown": avg_oos_drawdown,
                "total_oos_trades": total_oos_trades,
                "is_vs_oos_degradation": is_vs_oos_degradation,
                "robustness_score": robustness_score,
            },
            "conclusion": conclusion,
        }


def optimize_parameters(symbol: str, candles: List[Dict[str, float]], param_grid: Dict[str, List[Any]], strategy: str = "dca") -> List[Dict[str, Any]]:
    """Run backtest across all parameter combinations."""
    from itertools import product

    results = []
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    for values in product(*value_lists):
        params = dict(zip(keys, values))
        params["strategy"] = strategy

        engine = BacktestEngine(symbol, candles, params)
        result = engine.run()

        results.append({
            'params': params,
            'result': result.to_dict(),
            'sharpe_ratio': result.sharpe_ratio,
        })

    # Sort by Sharpe ratio descending
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    return results


def monte_carlo_simulation(trades: List[Trade], n_simulations: int = 1000) -> Dict[str, float]:
    """Run Monte Carlo simulation on trade sequence."""
    import random

    if not trades:
        return {
            'median_return': 0.0,
            'worst_return': 0.0,
            'best_return': 0.0,
            'p5_return': 0.0,
            'p95_return': 0.0,
            'worst_drawdown': 0.0,
            'median_drawdown': 0.0,
        }

    total_returns = []
    max_drawdowns = []

    for _ in range(n_simulations):
        shuffled_trades = random.sample(trades, len(trades))
        equity = 1000.0
        peak = equity
        max_dd = 0.0

        for trade in shuffled_trades:
            equity += trade.pnl_usd
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        total_returns.append(equity - 1000.0)
        max_drawdowns.append(max_dd * 100.0)

    total_returns.sort()
    max_drawdowns.sort()

    return {
        'median_return': total_returns[len(total_returns) // 2],
        'worst_return': min(total_returns),
        'best_return': max(total_returns),
        'p5_return': total_returns[int(len(total_returns) * 0.05)],
        'p95_return': total_returns[int(len(total_returns) * 0.95)],
        'worst_drawdown': max(max_drawdowns),
        'median_drawdown': max_drawdowns[len(max_drawdowns) // 2],
    }
