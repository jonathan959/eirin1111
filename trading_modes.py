"""
Trading mode configurations - timeframe-specific risk/reward parameters.
Each mode: scalp, day_trade, swing_trade, position_trade, long_term_hold.
"""
from typing import Any, Dict

TRADING_MODE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "scalp": {
        "name": "Scalping",
        "description": "Intraday quick profits, hold minutes to hours",
        "timeframes": ["1m", "5m", "15m"],
        "stop_loss_pct": -0.5,
        "take_profit_pct": 0.8,
        "trailing_activation_pct": 10,
        "trailing_distance_pct": 0.2,
        "max_hold_minutes": 60,
        "partial_exit_enabled": True,
        "partial_exit_levels": [0.6, 1.0],
        "partial_exit_amounts": [0.6, 0.4],
        "regime_exit_enabled": True,
        "use_limit_orders": True,
        "min_spread_bps": 5,
        "expected_hold_minutes": 30,
        "expected_return_pct": 0.6,
        "ideal_for": ["High liquidity crypto", "Major forex pairs"],
        "not_for": ["Low volume stocks", "Wide spread assets"],
    },
    "day_trade": {
        "name": "Day Trading",
        "description": "Intraday positions, close all by EOD",
        "timeframes": ["5m", "15m", "1h"],
        "stop_loss_pct": -2.0,
        "take_profit_pct": 4.0,
        "trailing_activation_pct": 30,
        "trailing_distance_pct": 1.0,
        "max_hold_minutes": 360,
        "partial_exit_enabled": True,
        "partial_exit_levels": [1.0, 1.5],
        "partial_exit_amounts": [0.5, 0.5],
        "regime_exit_enabled": True,
        "close_eod": True,
        "avoid_first_30min": True,
        "avoid_last_30min": True,
        "expected_hold_minutes": 180,
        "expected_return_pct": 3.0,
        "ideal_for": ["Liquid stocks", "Major crypto pairs"],
        "not_for": ["Earnings day", "Low volatility periods"],
    },
    "swing_trade": {
        "name": "Swing Trading",
        "description": "Hold days to weeks, technical-based exits",
        "timeframes": ["1h", "4h", "1d"],
        "stop_loss_pct": -5.0,
        "take_profit_pct": 10.0,
        "trailing_activation_pct": 50,
        "trailing_distance_pct": 2.5,
        "max_hold_days": 30,
        "partial_exit_enabled": True,
        "partial_exit_levels": [1.0, 1.5],
        "partial_exit_amounts": [0.5, 0.5],
        "regime_exit_enabled": True,
        "breakeven_stop_enabled": True,
        "expected_hold_days": 10,
        "expected_return_pct": 8.0,
        "ideal_for": ["Most stocks", "Most crypto", "Clear trends"],
        "not_for": ["Extreme volatility", "News-driven chaos"],
    },
    "position_trade": {
        "name": "Position Trading",
        "description": "Hold weeks to months, fundamental-aware",
        "timeframes": ["4h", "1d", "1w"],
        "stop_loss_pct": -12.0,
        "take_profit_pct": 25.0,
        "trailing_activation_pct": 100,
        "trailing_distance_pct": 5.0,
        "max_hold_days": 180,
        "partial_exit_enabled": True,
        "partial_exit_levels": [1.0, 2.0],
        "partial_exit_amounts": [0.25, 0.75],
        "regime_exit_enabled": False,
        "major_support_exit_only": True,
        "fundamental_monitoring": True,
        "expected_hold_days": 60,
        "expected_return_pct": 20.0,
        "ideal_for": ["Quality stocks", "BTC/ETH", "Strong fundamentals"],
        "not_for": ["Speculative plays", "News-driven stocks"],
    },
    "long_term_hold": {
        "name": "Long-Term Hold",
        "description": "Buy and hold for years, fundamental exits only",
        "timeframes": ["1d", "1w", "1M"],
        "stop_loss_pct": -25.0,
        "take_profit_pct": 100.0,
        "trailing_activation_pct": 200,
        "trailing_distance_pct": 10.0,
        "max_hold_days": 730,
        "partial_exit_enabled": False,
        "regime_exit_enabled": False,
        "fundamental_exit_only": True,
        "auto_dip_buy": True,
        "dip_buy_trigger_pct": -10.0,
        "dip_buy_max_adds": 3,
        "rebalance_enabled": True,
        "tax_aware": True,
        "fundamental_exit_triggers": [
            "earnings_miss_2_quarters",
            "revenue_decline_20pct",
            "debt_spike_50pct",
            "management_change_negative",
            "regulatory_ban",
            "protocol_hack",
        ],
        "expected_hold_days": 365,
        "expected_return_pct": 50.0,
        "ideal_for": ["S&P 500", "BTC", "ETH", "AAPL", "MSFT", "Blue chips"],
        "not_for": ["Speculative altcoins", "Penny stocks", "Meme stocks"],
    },
}


def get_mode_config(trading_mode: str, conviction_level: int = 5) -> Dict[str, Any]:
    """
    Get configuration for trading mode, adjusted by conviction (1-10).
    Higher conviction = wider stops, larger size.
    """
    base = TRADING_MODE_CONFIGS.get(trading_mode, TRADING_MODE_CONFIGS["swing_trade"]).copy()
    mult = conviction_level / 5.0
    base["stop_loss_pct"] = base.get("stop_loss_pct", -5.0) * mult
    base["take_profit_pct"] = base.get("take_profit_pct", 10.0) * mult
    base["conviction_size_multiplier"] = mult
    return base


def suggest_mode_for_asset(symbol: str, horizon: str, volatility: float) -> str:
    """Auto-suggest optimal trading mode based on asset characteristics."""
    blue_chips = ["BTC/USD", "ETH/USD", "AAPL", "MSFT", "GOOGL", "SPY", "QQQ"]
    is_blue = symbol in blue_chips
    if horizon == "long_term":
        return "long_term_hold" if is_blue else "position_trade"
    if horizon == "short_term":
        return "day_trade" if volatility > 3.0 else "swing_trade"
    return "swing_trade"
