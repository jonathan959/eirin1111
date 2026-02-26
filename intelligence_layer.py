# intelligence_layer.py
"""
Intelligence Layer - Deterministic decision stack for trading decisions.

This module implements a 7-layer decision hierarchy that sits between market data
and order placement, becoming the single source of truth for "should we trade" and
"how to trade".

Decision Hierarchy (enforced in order):
1. Data Validity Gate (hard block)
2. Market Safety Gate (hard/soft block)
3. Regime Detection (multi-timeframe + confidence + hysteresis)
4. Strategy Routing (posture selection)
5. Position Sizing (risk budget -> order plan)
6. Execution Policy (how to place orders)
7. Trade Management (exits + risk reduction)

All strategies MUST return proposed actions only. Only the centralized executor
is allowed to place/cancel orders after Intelligence approval.
"""

import os
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Safety: handle Enum/string in logs and metrics
def _safe_enum_val(x: Any) -> str:
    try:
        return str(x.value) if hasattr(x, "value") else str(x)
    except Exception:
        return str(x)

from strategies import (
    detect_regime,
    RegimeResult,
    _atr,
    ema,
    ema_series,
    rolling_return,
    rsi,
    clamp,
    safe_float,
)
from db import get_strategy_perf, get_expected_edge

# Phase 1 & Phase 2 Intelligence imports
try:
    from phase1_intelligence import (
        Phase1Intelligence,
        TrailingStopLoss,
        CooldownManager,
        VolatilityTPScaler,
        BTCCorrelationGuard,
        TimeFilter,
        VolumeFilter,
        AdaptiveSpreadGuard,
    )
    PHASE1_AVAILABLE = True
except ImportError:
    PHASE1_AVAILABLE = False
    logger.warning("Phase 1 intelligence modules not available")

_PHASE2_IMPORT_OK = False
if os.getenv("DISABLE_PHASE2", "0").strip().lower() not in ("1", "true", "yes", "y", "on"):
    try:
        from multi_timeframe import MultiTimeframeAnalyzer
        from kelly_criterion import KellyPositionSizer, DynamicPositionSizer
        from sentiment_analyzer import SentimentAnalyzer
        from portfolio_correlation import PortfolioCorrelationAnalyzer
        from order_book_analyzer import OrderBookAnalyzer, SmartOrderRouter
        _PHASE2_IMPORT_OK = True
    except ImportError:
        logger.warning("Phase 2 intelligence modules not available")
PHASE2_AVAILABLE = _PHASE2_IMPORT_OK

PHASE3_AVAILABLE = False
if os.getenv("DISABLE_PHASE3", "0").strip().lower() not in ("1", "true", "yes", "y", "on"):
    try:
        from ml_predictor import create_ml_predictor
        PHASE3_AVAILABLE = True
    except ImportError:
        PHASE3_AVAILABLE = False
        logger.warning("Phase 3 intelligence modules (ML) not available")

# Phase 2 Advanced (ML regime, vol forecast, order flow, liquidity, adaptive params)
ENABLE_ML_REGIME = os.getenv("ENABLE_ML_REGIME_DETECTION", "0").strip().lower() in ("1", "true", "yes", "y", "on")
ENABLE_VOL_FORECAST = os.getenv("ENABLE_VOLATILITY_FORECASTING", "0").strip().lower() in ("1", "true", "yes", "y", "on")
ENABLE_ORDER_FLOW = os.getenv("ENABLE_ORDER_FLOW_ANALYSIS", "0").strip().lower() in ("1", "true", "yes", "y", "on")

# Phase 4 Ultimate Intelligence - all 13 features
ENABLE_PHASE4 = os.getenv("ENABLE_PHASE4_ULTIMATE", "1").strip().lower() in ("1", "true", "yes", "y", "on")

_PHASE4_IMPORTS: Dict[str, Any] = {}
if ENABLE_PHASE4:
    def _phase4_import(name: str, mod: str, attrs: List[str], key: Optional[str] = None) -> None:
        try:
            m = __import__(mod, fromlist=attrs)
            for a in attrs:
                k = key or a
                _PHASE4_IMPORTS[k] = getattr(m, a)
        except Exception as e:
            logger.debug("Phase 4 %s: %s", mod, e)

    _phase4_import("CorrelationTrader", "correlation_trading", ["CorrelationTrader"])
    _phase4_import("SeasonalityAnalyzer", "seasonality", ["SeasonalityAnalyzer"])
    _phase4_import("OptionsFlowAnalyzer", "options_flow", ["OptionsFlowAnalyzer"])
    _phase4_import("EarningsMomentumTrader", "earnings_momentum", ["EarningsMomentumTrader"])
    _phase4_import("ZScoreTrader", "zscore_trading", ["ZScoreTrader"])
    _phase4_import("MomentumRanker", "momentum_ranking", ["MomentumRanker"])
    _phase4_import("AlternativeDataIntegrator", "alternative_data", ["AlternativeDataIntegrator", "calculate_alternative_data_score"])
    # MLEnsemble may fail if numpy/sklearn/xgboost not installed
    _phase4_import("get_ml_ensemble", "ml_ensemble", ["get_ml_ensemble"])
    if "get_ml_ensemble" not in _PHASE4_IMPORTS:
        _phase4_import("MLEnsemble", "ml_ensemble", ["MLEnsemble"])
    # RL agent (optional deps: gymnasium, stable-baselines3)
    try:
        from rl_agent import RLTradingAgent, ENABLE_RL_AGENT
        _PHASE4_IMPORTS["RLTradingAgent"] = RLTradingAgent
        _PHASE4_IMPORTS["ENABLE_RL_AGENT"] = ENABLE_RL_AGENT
    except Exception:
        _PHASE4_IMPORTS["RLTradingAgent"] = None
        _PHASE4_IMPORTS["ENABLE_RL_AGENT"] = False
    # High-frequency order book
    try:
        from high_frequency import OrderBookAnalyzer as HFOrderBookAnalyzer
        _PHASE4_IMPORTS["HFOrderBookAnalyzer"] = HFOrderBookAnalyzer
    except Exception:
        _PHASE4_IMPORTS["HFOrderBookAnalyzer"] = None

# =========================
# Enums and Constants
# =========================

class AllowedAction(Enum):
    """Allowed trading actions after intelligence evaluation."""
    NO_TRADE = "NO_TRADE"  # Hard block - do nothing except manage existing positions
    MANAGE_ONLY = "MANAGE_ONLY"  # Soft block - only manage exits, no new entries
    TRADE_ALLOWED = "TRADE_ALLOWED"  # Full trading allowed


class RegimeType(Enum):
    """Market regime classifications (5-level primary + legacy)."""
    STRONG_BULL = "STRONG_BULL"
    WEAK_BULL = "WEAK_BULL"
    RANGE = "RANGE"
    WEAK_BEAR = "WEAK_BEAR"
    STRONG_BEAR = "STRONG_BEAR"
    HIGH_VOL_DEFENSIVE = "HIGH_VOL_DEFENSIVE"  # High volatility, defensive mode
    BREAKOUT = "BREAKOUT"  # Breakout pattern detected
    RISK_OFF = "RISK_OFF"  # Risk-off market conditions
    # Legacy compatibility
    BULL = "BULL"
    BEAR = "BEAR"


# =========================
# Data Structures
# =========================

@dataclass
class DataValidityResult:
    """Result from Data Validity Gate."""
    data_ok: bool
    reasons: List[str] = field(default_factory=list)
    stale_price: bool = False
    missing_candles: bool = False
    extreme_spread: bool = False
    illiquid: bool = False
    exchange_error: bool = False


@dataclass
class MarketSafetyResult:
    """Result from Market Safety Gate."""
    allowed_actions: AllowedAction
    risk_budget: float  # Available risk budget in quote currency
    reasons: List[str] = field(default_factory=list)
    global_pause: bool = False
    kill_switch: bool = False
    daily_loss_limit_hit: bool = False
    drawdown_limit_hit: bool = False
    exposure_limit_hit: bool = False
    btc_risk_off: bool = False


@dataclass
class RegimeDetectionResult:
    """Result from Regime Detection."""
    regime: RegimeType
    confidence: float  # 0.0 to 1.0
    ttl_seconds: int  # Time-to-live for this regime classification
    hysteresis_state: Dict[str, Any] = field(default_factory=dict)
    multi_timeframe_regimes: Dict[str, RegimeResult] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyRoutingResult:
    """Result from Strategy Routing."""
    strategy_mode: str  # e.g., "trend_follow", "range_mean_reversion", "smart_dca"
    entry_style: str  # "aggressive", "moderate", "defensive", "probe_only"
    exit_style: str  # "trailing", "fixed_tp", "time_based", "regime_based"
    reasons: List[str] = field(default_factory=list)


@dataclass
class PositionSizingResult:
    """Result from Position Sizing."""
    base_size: float  # Base order size in quote currency
    ladder_steps: int  # Number of DCA ladder steps
    ladder_spacing_pct: float  # Spacing between ladder steps as % of price
    max_adds: int  # Maximum number of additional positions
    volatility_adjusted: bool = False
    btc_risk_adjusted: bool = False
    reasons: List[str] = field(default_factory=list)


@dataclass
class ExecutionPolicyResult:
    """Result from Execution Policy."""
    order_type: str  # "limit", "market", "post_only"
    limit_price: Optional[float] = None
    max_slippage_pct: float = 0.001  # 0.1% default
    post_only: bool = False
    cancel_replace_rules: Dict[str, Any] = field(default_factory=dict)
    min_cooldown_seconds: int = 0
    reasons: List[str] = field(default_factory=list)
    urgency: str = "medium"  # "high" | "medium" | "low" - affects limit placement & patience


@dataclass
class TradeManagementResult:
    """Result from Trade Management."""
    manage_actions: List[Dict[str, Any]] = field(default_factory=list)
    partial_tp: Optional[Dict[str, Any]] = None
    trailing_stop: Optional[Dict[str, Any]] = None
    time_stop: Optional[Dict[str, Any]] = None
    break_even: Optional[Dict[str, Any]] = None
    forced_de_risk: Optional[Dict[str, Any]] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class IntelligenceContext:
    """Complete context for intelligence evaluation."""
    # Market data (multi-timeframe)
    symbol: str
    last_price: float
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    spread_pct: Optional[float] = None
    candles_1h: List[List[float]] = field(default_factory=list)
    candles_4h: List[List[float]] = field(default_factory=list)
    candles_1d: List[List[float]] = field(default_factory=list)
    candles_1w: List[List[float]] = field(default_factory=list)
    
    # Account state
    balances: Dict[str, float] = field(default_factory=dict)
    free_quote: float = 0.0
    total_quote: float = 0.0
    
    # Position state
    open_positions: List[Dict[str, Any]] = field(default_factory=list)
    current_position_size: float = 0.0
    avg_entry_price: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    # Bot config
    bot_config: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True
    
    # Portfolio state
    portfolio_total_usd: float = 0.0
    portfolio_exposure_pct: float = 0.0
    daily_realized_pnl: float = 0.0
    portfolio_drawdown: float = 0.0
    
    # BTC context
    btc_context: Dict[str, Any] = field(default_factory=dict)
    
    # Market-wide context
    market_breadth: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    now_ts: int = 0
    last_price_ts: int = 0
    last_candle_ts: int = 0
    
    # Exchange state
    exchange_errors: int = 0
    rate_limit_storm: bool = False


@dataclass
class IntelligenceDecision:
    """Complete intelligence decision output."""
    # Layer 1: Data Validity
    data_validity: DataValidityResult
    
    # Layer 2: Market Safety
    market_safety: MarketSafetyResult
    
    # Layer 3: Regime Detection
    regime_detection: RegimeDetectionResult
    
    # Layer 4: Strategy Routing
    strategy_routing: StrategyRoutingResult
    
    # Layer 5: Position Sizing
    position_sizing: PositionSizingResult
    
    # Layer 6: Execution Policy
    execution_policy: ExecutionPolicyResult
    
    # Layer 7: Trade Management
    trade_management: TradeManagementResult
    
    # Final decision
    allowed_actions: AllowedAction
    final_action: str  # "NO_TRADE", "MANAGE_ONLY", "ENTER", "ADD", "EXIT", etc.
    final_reason: str
    
    # Proposed orders (if any)
    proposed_orders: List[Dict[str, Any]] = field(default_factory=list)
    
    # Debug/metadata
    debug: Dict[str, Any] = field(default_factory=dict)
    timestamp: int = 0


# =========================
# Intelligence Layer Implementation
# =========================

class IntelligenceLayer:
    """
    Intelligence Layer - Single source of truth for trading decisions.
    
    Usage:
        layer = IntelligenceLayer()
        context = IntelligenceContext(...)
        decision = layer.evaluate(context)
        
        if decision.allowed_actions == AllowedAction.TRADE_ALLOWED:
            # Execute proposed orders through centralized executor
            pass
    """
    
    def __init__(self):
        # Configuration
        self.max_spread_pct = float(os.getenv("INTEL_MAX_SPREAD_PCT", "0.005"))  # 0.5%
        self.stale_price_seconds = int(os.getenv("INTEL_STALE_PRICE_SEC", "60"))
        self.min_candles_required = int(os.getenv("INTEL_MIN_CANDLES", "50"))
        self.min_candles_stocks = int(os.getenv("INTEL_MIN_CANDLES_STOCKS", "20"))  # Lower for stocks (IPOs, new listings)
        
        self.max_drawdown_pct = float(os.getenv("INTEL_MAX_DRAWDOWN_PCT", "0.20"))  # 20%
        
        self.regime_confidence_threshold = float(os.getenv("INTEL_REGIME_CONFIDENCE_THRESHOLD", "0.40"))
        self.regime_hysteresis_seconds = int(os.getenv("INTEL_REGIME_HYSTERESIS_SEC", "300"))  # 5 min
        self.base_risk_pct = float(os.getenv("INTEL_BASE_RISK_PCT", "0.02"))  # 2% of equity
        # Default hard caps (safe defaults unless overridden)
        self.max_daily_loss_pct = float(os.getenv("INTEL_MAX_DAILY_LOSS_PCT", "0.05"))  # 5%
        self.max_exposure_pct = float(os.getenv("INTEL_MAX_EXPOSURE_PCT", "0.20"))  # 20%
        self.per_symbol_exposure_pct = float(os.getenv("INTEL_PER_SYMBOL_EXPOSURE_PCT", "0.15"))  # 15%
        self.max_consec_losses = int(os.getenv("INTEL_MAX_CONSEC_LOSSES", "3"))
        
        self.max_slippage_pct = float(os.getenv("INTEL_MAX_SLIPPAGE_PCT", "0.001"))  # 0.1%
        
        # Regime score weights (configurable via env)
        self.score_strong_bull = (float(os.getenv("SCORE_STRONG_BULL", "75")), float(os.getenv("SCORE_STRONG_BULL_CONF", "15")))
        self.score_weak_bull = (float(os.getenv("SCORE_WEAK_BULL", "60")), float(os.getenv("SCORE_WEAK_BULL_CONF", "10")))
        self.score_breakout = (float(os.getenv("SCORE_BREAKOUT", "75")), float(os.getenv("SCORE_BREAKOUT_CONF", "15")))
        self.score_range = (float(os.getenv("SCORE_RANGE", "45")), float(os.getenv("SCORE_RANGE_CONF", "10")))
        self.score_high_vol = float(os.getenv("SCORE_HIGH_VOL", "30"))
        self.score_weak_bear = float(os.getenv("SCORE_WEAK_BEAR", "30"))
        self.score_strong_bear = float(os.getenv("SCORE_STRONG_BEAR", "25"))
        self.score_risk_off = float(os.getenv("SCORE_RISK_OFF", "10"))
        
        # Hysteresis state cache (per symbol)
        self._hysteresis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Phase 2: Multi-timeframe analyzer (initialized per-symbol as needed)
        self._multi_tf_analyzers: Dict[str, MultiTimeframeAnalyzer] = {}
        
        # Phase 2: Kelly position sizer (initialized per-bot as needed)
        self._kelly_sizers: Dict[int, DynamicPositionSizer] = {}
        
        # Phase 2: Sentiment analyzer (shared instance)
        self._sentiment_analyzer = None
        if PHASE2_AVAILABLE:
            try:
                self._sentiment_analyzer = SentimentAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize SentimentAnalyzer: {e}")
        
        # Phase 2: Portfolio correlation analyzer (shared instance)
        self._correlation_analyzer = None
        if PHASE2_AVAILABLE:
            try:
                self._correlation_analyzer = PortfolioCorrelationAnalyzer()
            except Exception as e:
                logger.warning(f"Failed to initialize PortfolioCorrelationAnalyzer: {e}")
        
        # Phase 2: Order book analyzer (initialized per-symbol as needed)
        self._orderbook_analyzers: Dict[str, OrderBookAnalyzer] = {}
        
        # Phase 3: ML predictor (initialized per-symbol as needed)
        self._ml_predictors: Dict[str, Any] = {}
        self._strategy_state: Dict[str, Dict[str, Any]] = {}
    
    def evaluate(self, context: IntelligenceContext) -> IntelligenceDecision:
        """
        Main evaluation function. Runs all 7 layers in order.
        
        Returns complete IntelligenceDecision with all layer results.
        """
        if context.now_ts == 0:
            context.now_ts = int(time.time())
        
        # Layer 1: Data Validity Gate
        data_validity = self._check_data_validity(context)
        if not data_validity.data_ok:
            # Hard block - return early with NO_TRADE
            detailed_reason = f"Data invalid: {'; '.join(data_validity.reasons)}"
            return self._create_no_trade_decision(
                context, data_validity, reason=detailed_reason
            )
        
        # Layer 2: Market Safety Gate
        market_safety = self._check_market_safety(context)
        if market_safety.allowed_actions == AllowedAction.NO_TRADE:
            return self._create_no_trade_decision(
                context, data_validity, market_safety, "Market safety check failed"
            )
        
        # Layer 3: Regime Detection
        regime_detection = self._detect_regime(context)
        
        # Layer 4: Strategy Routing
        strategy_routing = self._route_strategy(context, regime_detection)
        
        # Layer 5: Position Sizing
        position_sizing = self._calculate_position_sizing(
            context, regime_detection, strategy_routing, market_safety
        )
        
        # Layer 6: Execution Policy
        execution_policy = self._determine_execution_policy(context, regime_detection)
        
        # Layer 7: Trade Management (always runs)
        trade_management = self._manage_trades(context, regime_detection)
        
        # Determine final action
        allowed_actions = market_safety.allowed_actions
        if allowed_actions == AllowedAction.MANAGE_ONLY:
            final_action = "MANAGE_ONLY"
            final_reason = "Market safety: manage-only mode"
        elif allowed_actions == AllowedAction.TRADE_ALLOWED:
            # Check if we should enter, add, or exit based on strategy routing
            if context.current_position_size == 0:
                final_action = "ENTER" if strategy_routing.entry_style != "probe_only" else "PROBE"
            else:
                final_action = "ADD"  # Strategy will determine if we should add
            final_reason = f"Trading allowed: {strategy_routing.strategy_mode}"

            # Gate ENTER/ADD on regime confidence (per-bot override via bot_config)
            conf_thresh = self.regime_confidence_threshold
            try:
                bot_thresh = context.bot_config.get("regime_confidence_threshold")
                if bot_thresh is not None and float(bot_thresh) >= 0:
                    conf_thresh = float(bot_thresh)
            except (TypeError, ValueError):
                pass
            if final_action in ("ENTER", "ADD", "PROBE") and regime_detection.confidence < conf_thresh:
                final_action = "HOLD"
                final_reason = f"Regime confidence {regime_detection.confidence:.2f} below threshold {conf_thresh:.2f}"

            # Per-position drawdown limit: force EXIT when unrealized loss >= bot max_drawdown_pct
            if final_action in ("ENTER", "ADD", "HOLD", "PROBE") and context.current_position_size > 0 and context.avg_entry_price and context.avg_entry_price > 0 and context.last_price > 0:
                try:
                    max_dd = float(context.bot_config.get("max_drawdown_pct") or 0)
                    if max_dd > 0:
                        unrealized_dd = (context.avg_entry_price - context.last_price) / context.avg_entry_price
                        if unrealized_dd >= max_dd:
                            final_action = "EXIT"
                            final_reason = f"Per-position drawdown limit: {unrealized_dd*100:.2f}% >= {max_dd*100:.2f}%"
                except (TypeError, ValueError):
                    pass
        else:
            final_action = "NO_TRADE"
            final_reason = "Trading not allowed"
        
        # Build proposed orders (strategies will add to this)
        proposed_orders = []
        
        return IntelligenceDecision(
            data_validity=data_validity,
            market_safety=market_safety,
            regime_detection=regime_detection,
            strategy_routing=strategy_routing,
            position_sizing=position_sizing,
            execution_policy=execution_policy,
            trade_management=trade_management,
            allowed_actions=allowed_actions,
            final_action=final_action,
            final_reason=final_reason,
            proposed_orders=proposed_orders,
            debug={
                "symbol": context.symbol,
                "price": context.last_price,
                "regime": regime_detection.regime.value if hasattr(regime_detection.regime, "value") else str(regime_detection.regime),
                "confidence": regime_detection.confidence,
                "confidence_score": int(clamp(regime_detection.confidence * 100.0, 0.0, 100.0)),
                "regime_scores": regime_detection.scores or {},
                "data_health": "ok" if data_validity.data_ok else ("invalid: " + "; ".join(data_validity.reasons or ["unknown"])),
            },
            timestamp=context.now_ts,
        )
    
    # =========================
    # Layer 1: Data Validity Gate
    # =========================
    
    def _check_data_validity(self, context: IntelligenceContext) -> DataValidityResult:
        """Check if market data is valid and fresh."""
        reasons = []
        stale_price = False
        missing_candles = False
        extreme_spread = False
        illiquid = False
        exchange_error = False
        
        # Check price staleness
        if context.last_price_ts > 0:
            age = context.now_ts - context.last_price_ts
            if age > self.stale_price_seconds:
                stale_price = True
                reasons.append(f"Price stale: {age}s old")
        
        # Check for valid price
        if context.last_price <= 0:
            reasons.append("Invalid price: <= 0")
            return DataValidityResult(
                data_ok=False,
                reasons=reasons,
                stale_price=stale_price,
            )
        
        # Check candles (stocks can use lower threshold via INTEL_MIN_CANDLES_STOCKS)
        is_stock = (context.market_breadth or {}).get("is_stock") or (len(context.symbol) < 6 and "/" not in context.symbol)
        min_candles = self.min_candles_stocks if is_stock else self.min_candles_required
        all_candles = [
            context.candles_1h,
            context.candles_4h,
            context.candles_1d,
        ]
        mb = context.market_breadth or {}
        provider = mb.get("provider", "unknown")
        sym_norm = mb.get("symbol_normalized", context.symbol)
        limits = mb.get("limits_requested") or {}
        data_error = mb.get("data_error")
        if data_error:
            reasons.append(str(data_error))
        for tf_name, candles in zip(["1h", "4h", "1d"], all_candles):
            count = len(candles) if candles else 0
            limit_req = limits.get(tf_name, "?")
            if not candles or count < min_candles:
                missing_candles = True
                reasons.append(
                    f"Missing/incomplete {tf_name} candles: {count} < {min_candles} "
                    f"(market_type={mb.get('market_type','?')} provider={provider} symbol={sym_norm} limit={limit_req})"
                )
        
        # Check spread
        if context.spread_pct is not None:
            if context.spread_pct > self.max_spread_pct:
                extreme_spread = True
                reasons.append(f"Spread too wide: {context.spread_pct*100:.2f}% > {self.max_spread_pct*100:.2f}%")
        
        # Check exchange errors
        if context.exchange_errors > 5:
            exchange_error = True
            reasons.append(f"Too many exchange errors: {context.exchange_errors}")
        
        if context.rate_limit_storm:
            exchange_error = True
            reasons.append("Rate limit storm detected")
        
        data_ok = not (stale_price or missing_candles or extreme_spread or exchange_error)
        
        return DataValidityResult(
            data_ok=data_ok,
            reasons=reasons,
            stale_price=stale_price,
            missing_candles=missing_candles,
            extreme_spread=extreme_spread,
            exchange_error=exchange_error,
        )
    
    # =========================
    # Layer 2: Market Safety Gate
    # =========================
    
    def _check_market_safety(self, context: IntelligenceContext) -> MarketSafetyResult:
        """Check global safety gates and risk limits."""
        reasons = []
        allowed_actions = AllowedAction.TRADE_ALLOWED
        
        # Check global pause (from settings/DB)
        global_pause = bool(context.bot_config.get("global_pause", False))
        if global_pause:
            allowed_actions = AllowedAction.NO_TRADE
            reasons.append("Global pause active")
            return MarketSafetyResult(
                allowed_actions=allowed_actions,
                risk_budget=0.0,
                reasons=reasons,
                global_pause=True,
            )
        
        # Check kill switch
        kill_switch = bool(context.bot_config.get("kill_switch", False))
        if kill_switch:
            allowed_actions = AllowedAction.NO_TRADE
            reasons.append("Kill switch active")
            return MarketSafetyResult(
                allowed_actions=allowed_actions,
                risk_budget=0.0,
                reasons=reasons,
                kill_switch=True,
            )

        # Consecutive loss circuit breaker
        try:
            consec = int(context.bot_config.get("consecutive_losses", 0))
            if consec >= self.max_consec_losses:
                allowed_actions = AllowedAction.NO_TRADE
                reasons.append(f"Consecutive loss breaker: {consec} losses")
                return MarketSafetyResult(
                    allowed_actions=allowed_actions,
                    risk_budget=0.0,
                    reasons=reasons,
                )
        except Exception:
            pass

        # API failure safe-stop
        if context.exchange_errors >= 3 or context.rate_limit_storm:
            allowed_actions = AllowedAction.NO_TRADE
            reasons.append("API failure/rate limit storm: safe-stop")
            return MarketSafetyResult(
                allowed_actions=allowed_actions,
                risk_budget=0.0,
                reasons=reasons,
            )
        
        # Check daily loss limit
        if context.portfolio_total_usd > 0:
            daily_loss_pct = abs(context.daily_realized_pnl) / context.portfolio_total_usd
            if daily_loss_pct >= self.max_daily_loss_pct:
                allowed_actions = AllowedAction.NO_TRADE
                reasons.append(f"Daily loss limit hit: {daily_loss_pct*100:.2f}% >= {self.max_daily_loss_pct*100:.2f}%")
                return MarketSafetyResult(
                    allowed_actions=allowed_actions,
                    risk_budget=0.0,
                    reasons=reasons,
                    daily_loss_limit_hit=True,
                )
        
        # Check drawdown limit (kill switch). Use bot-level max_drawdown_pct when set and > 0; else layer default. 0 = disabled.
        max_dd_limit = self.max_drawdown_pct
        try:
            bot_dd = float(context.bot_config.get("max_drawdown_pct") or 0)
            if bot_dd > 0:
                max_dd_limit = bot_dd
        except (TypeError, ValueError):
            pass
        if max_dd_limit > 0 and context.portfolio_drawdown >= max_dd_limit:
            allowed_actions = AllowedAction.NO_TRADE
            reasons.append(f"Max drawdown kill-switch: {context.portfolio_drawdown*100:.2f}% >= {max_dd_limit*100:.2f}%")
        
        # Check exposure limits
        if context.portfolio_exposure_pct >= self.max_exposure_pct:
            allowed_actions = AllowedAction.MANAGE_ONLY
            reasons.append(f"Portfolio exposure limit hit: {context.portfolio_exposure_pct*100:.2f}% >= {self.max_exposure_pct*100:.2f}%")
        
        # Check per-symbol exposure
        if context.current_position_size > 0 and context.portfolio_total_usd > 0:
            symbol_exposure_pct = (context.current_position_size * context.last_price) / context.portfolio_total_usd
            if symbol_exposure_pct >= self.per_symbol_exposure_pct:
                allowed_actions = AllowedAction.MANAGE_ONLY
                reasons.append(f"Per-symbol exposure limit hit: {symbol_exposure_pct*100:.2f}% >= {self.per_symbol_exposure_pct*100:.2f}%")
        
        # Check BTC risk-off context (position sizing already reduces size by 50% at Layer 5)
        btc_risk_off = bool(context.btc_context.get("risk_off", False))
        if btc_risk_off and context.symbol != "XBT/USD" and context.symbol != "BTC/USD":
            reasons.append("BTC risk-off context: altcoin position size will be reduced")

        # Volatility spike filter (pause new entries)
        try:
            atr_val = _atr(context.candles_1d or context.candles_4h or [], 14)
            if atr_val and context.last_price > 0:
                vol_pct = atr_val / context.last_price
                if vol_pct >= 0.15 and allowed_actions == AllowedAction.TRADE_ALLOWED:
                    allowed_actions = AllowedAction.MANAGE_ONLY
                    reasons.append("Volatility spike: manage-only")
        except Exception:
            pass
        
        # ============================================================
        # PHASE 1 INTEGRATION: Additional Safety Filters
        # ============================================================
        if PHASE1_AVAILABLE and allowed_actions != AllowedAction.NO_TRADE:
            try:
                # Initialize Phase 1 intelligence for this bot
                phase1 = Phase1Intelligence(context.bot_config)
                
                # Check Phase 1 entry filters
                market_data = {
                    'symbol': context.symbol,
                    'price': context.last_price,
                    'bid': context.bid_price,
                    'ask': context.ask_price,
                    'volume': context.bot_config.get('volume', 0),
                    'avg_volume': context.bot_config.get('avg_volume', 0),
                    'atr_pct': context.bot_config.get('atr_pct', 0),
                }
                
                btc_1h_change = context.btc_context.get('1h_change_pct')
                entry_allowed, phase1_reasons = phase1.check_entry_allowed(market_data, btc_1h_change)
                
                if not entry_allowed:
                    if allowed_actions == AllowedAction.TRADE_ALLOWED:
                        allowed_actions = AllowedAction.MANAGE_ONLY
                    reasons.extend(phase1_reasons)
                
                # Check cooldown after stop loss
                last_stop_loss = context.bot_config.get('last_stop_loss_at')
                can_trade, cooldown_reason = phase1.cooldown_mgr.check_cooldown(last_stop_loss)
                if not can_trade:
                    if allowed_actions == AllowedAction.TRADE_ALLOWED:
                        allowed_actions = AllowedAction.MANAGE_ONLY
                    reasons.append(cooldown_reason)
                    
            except Exception as e:
                logger.warning(f"Phase 1 safety check failed: {e}")
        
        # ============================================================
        # PHASE 2 INTEGRATION: Sentiment Analysis
        # ============================================================
        if PHASE2_AVAILABLE and self._sentiment_analyzer and allowed_actions != AllowedAction.NO_TRADE:
            try:
                sentiment_ok, sentiment_reason = self._sentiment_analyzer.should_trade(context.symbol)
                if not sentiment_ok:
                    if allowed_actions == AllowedAction.TRADE_ALLOWED:
                        allowed_actions = AllowedAction.MANAGE_ONLY
                    reasons.append(f"Sentiment filter: {sentiment_reason}")
                break_alert = self._sentiment_analyzer.get_breaking_news_alert(context.symbol)
                if break_alert:
                    if allowed_actions == AllowedAction.TRADE_ALLOWED:
                        allowed_actions = AllowedAction.MANAGE_ONLY
                    reasons.append(f"Breaking news: {break_alert[:80]}")
            except Exception as e:
                logger.warning(f"Sentiment check failed: {e}")

        # ============================================================
        # PHASE 2 INTEGRATION: Portfolio Correlation Guard
        # ============================================================
        if PHASE2_AVAILABLE and self._correlation_analyzer and allowed_actions != AllowedAction.NO_TRADE:
            try:
                price_history = (context.market_breadth or {}).get("price_history") or {}
                active_bots = (context.market_breadth or {}).get("active_bots") or (context.market_breadth or {}).get("bots") or []

                # Normalize active_bots to list of dicts with symbol keys
                if active_bots and isinstance(active_bots, list) and not isinstance(active_bots[0], dict):
                    active_bots = [{"symbol": s} for s in active_bots]

                if price_history and active_bots:
                    diversification_score, high_corr_pairs, details = self._correlation_analyzer.analyze_portfolio(
                        active_bots=active_bots,
                        price_history=price_history,
                    )
                    context.market_breadth["diversification_score"] = diversification_score
                    context.market_breadth["high_corr_pairs"] = high_corr_pairs
                    context.market_breadth["correlation_details"] = details

                    if high_corr_pairs:
                        if allowed_actions == AllowedAction.TRADE_ALLOWED:
                            allowed_actions = AllowedAction.MANAGE_ONLY
                        top_pair = high_corr_pairs[0]
                        reasons.append(
                            f"Portfolio correlation high: {top_pair.get('symbol1')} vs {top_pair.get('symbol2')} "
                            f"({float(top_pair.get('correlation') or 0.0):.2f})"
                        )
                    elif diversification_score < 0.4:
                        if allowed_actions == AllowedAction.TRADE_ALLOWED:
                            allowed_actions = AllowedAction.MANAGE_ONLY
                        reasons.append(f"Low diversification score: {diversification_score:.2f}")
            except Exception as e:
                logger.warning(f"Portfolio correlation check failed: {e}")
        
        # Calculate risk budget
        risk_budget = context.free_quote
        if context.portfolio_total_usd > 0:
            max_risk = context.portfolio_total_usd * self.max_exposure_pct
            current_exposure = context.portfolio_total_usd * context.portfolio_exposure_pct
            risk_budget = min(risk_budget, max_risk - current_exposure)
        
        risk_budget = max(0.0, risk_budget)

        # Correlation-adjusted exposure (clustered symbols)
        try:
            details = (context.market_breadth or {}).get("correlation_details") or []
            for d in details:
                s1 = str(d.get("symbol1") or "")
                s2 = str(d.get("symbol2") or "")
                corr = float(d.get("correlation") or 0.0)
                if corr >= 0.8 and (context.symbol in (s1, s2)):
                    risk_budget *= 0.5
                    reasons.append(f"High correlation cluster ({corr:.2f}): risk budget reduced")
                    break
        except Exception:
            pass
        
        return MarketSafetyResult(
            allowed_actions=allowed_actions,
            risk_budget=risk_budget,
            reasons=reasons,
            btc_risk_off=btc_risk_off,
        )
    
    # =========================
    # Layer 3: Regime Detection
    # =========================

    def _vwap(self, candles: List[List[float]], window: int = 50) -> Optional[float]:
        if not candles:
            return None
        segment = candles[-window:] if len(candles) >= window else candles
        num = 0.0
        den = 0.0
        for c in segment:
            if len(c) < 6:
                continue
            h = float(c[2])
            l = float(c[3])
            close = float(c[4])
            v = float(c[5]) if len(c) >= 6 else 0.0
            price = (h + l + close) / 3.0
            num += price * v
            den += v
        if den <= 0:
            return None
        return num / den

    def _ema_slope_pct(self, closes: List[float], period: int = 20) -> float:
        if not closes or len(closes) < max(30, period + 5):
            return 0.0
        ema_vals = ema_series(closes, period)
        if not ema_vals or len(ema_vals) < 5:
            return 0.0
        recent = ema_vals[-10:]
        slope_val = 0.0
        try:
            slope_val = (recent[-1] - recent[0]) / max(1e-9, recent[0])
        except Exception:
            slope_val = 0.0
        return float(slope_val)

    def _atr_percentile(self, candles_1d: List[List[float]], window: int = 90) -> float:
        if not candles_1d or len(candles_1d) < max(window, 20):
            return 0.5
        atr_vals: List[float] = []
        for i in range(20, min(len(candles_1d), window + 20)):
            segment = candles_1d[max(0, i - 14):i + 1]
            atr_val = _atr(segment, 14)
            if atr_val:
                atr_vals.append(float(atr_val))
        if not atr_vals:
            return 0.5
        current_atr = atr_vals[-1]
        less = sum(1 for v in atr_vals if v <= current_atr)
        return float(less) / float(len(atr_vals))

    def _momentum_decay(self, closes: List[float]) -> float:
        if not closes or len(closes) < 60:
            return 0.0
        short = rolling_return(closes, 12) or 0.0
        long = rolling_return(closes, 48) or 0.0
        return float(short - long)

    def _mtf_alignment(self, mtf: Dict[str, RegimeResult]) -> float:
        if not mtf:
            return 0.0
        dirs = []
        for reg in mtf.values():
            scores = reg.scores or {}
            up = float(scores.get("uptrend_score", 0.0))
            down = float(scores.get("downtrend_score", 0.0))
            if up >= 0.55 and up > down:
                dirs.append(1)
            elif down >= 0.55 and down > up:
                dirs.append(-1)
            else:
                dirs.append(0)
        if not dirs:
            return 0.0
        return float(sum(dirs)) / float(len(dirs))
    
    def _detect_regime(self, context: IntelligenceContext) -> RegimeDetectionResult:
        """Detect market regime using multi-timeframe analysis."""
        multi_tf_regimes = {}
        all_scores = {}
        
        # ============================================================
        # PHASE 2 INTEGRATION: Multi-Timeframe Analyzer
        # ============================================================
        use_phase2_mtf = PHASE2_AVAILABLE and all([
            context.candles_1h and len(context.candles_1h) >= 50,
            context.candles_4h and len(context.candles_4h) >= 50,
            context.candles_1d and len(context.candles_1d) >= 50,
            context.candles_1w and len(context.candles_1w) >= 20,
        ])
        
        if use_phase2_mtf:
            try:
                # Get or create multi-timeframe analyzer for this symbol
                if context.symbol not in self._multi_tf_analyzers:
                    self._multi_tf_analyzers[context.symbol] = MultiTimeframeAnalyzer(
                        require_agreement=3,
                        weighted_threshold=0.6
                    )
                
                analyzer = self._multi_tf_analyzers[context.symbol]
                
                # Analyze with Phase 2 multi-timeframe
                signal, mtf_details = analyzer.analyze(
                    candles_1h=context.candles_1h,
                    candles_4h=context.candles_4h,
                    candles_1d=context.candles_1d,
                    candles_1w=context.candles_1w,
                )
                
                # Map Phase 2 signal to our RegimeType (5-level)
                from multi_timeframe import Signal
                if signal == Signal.STRONG_BULLISH:
                    dominant_regime = RegimeType.STRONG_BULL
                    confidence = mtf_details.get('confidence', 0.8)
                elif signal == Signal.BULLISH:
                    dominant_regime = RegimeType.WEAK_BULL
                    confidence = mtf_details.get('confidence', 0.6)
                elif signal == Signal.STRONG_BEARISH:
                    dominant_regime = RegimeType.STRONG_BEAR
                    confidence = mtf_details.get('confidence', 0.8)
                elif signal == Signal.BEARISH:
                    dominant_regime = RegimeType.WEAK_BEAR
                    confidence = mtf_details.get('confidence', 0.6)
                else:
                    dominant_regime = RegimeType.RANGE
                    confidence = 0.5
                
                # Store multi-timeframe details
                all_scores.update({
                    'mtf_agreement_count': mtf_details.get('agreement_count', 0),
                    'mtf_weighted_score': mtf_details.get('weighted_score', 0.0),
                    'mtf_overall': signal.value if hasattr(signal, "value") else str(signal),
                })
                
                # Use Phase 2 result directly (skip basic regime detection)
                ttl_seconds = int(self.regime_hysteresis_seconds * (0.5 + confidence))
                
                return RegimeDetectionResult(
                    regime=dominant_regime,
                    confidence=confidence,
                    ttl_seconds=ttl_seconds,
                    hysteresis_state=self._hysteresis_cache.get(context.symbol, {}),
                    multi_timeframe_regimes={},  # Phase 2 handles this internally
                    scores=all_scores,
                )
            except Exception as e:
                logger.warning(f"Phase 2 multi-timeframe analysis failed, falling back to basic: {e}")
        
        # Fallback to basic regime detection
        # Detect regime on each timeframe
        for tf_name, candles in [
            ("1h", context.candles_1h),
            ("4h", context.candles_4h),
            ("1d", context.candles_1d),
            ("1w", context.candles_1w),
        ]:
            if candles and len(candles) >= 50:
                regime_result = detect_regime(candles)
                multi_tf_regimes[tf_name] = regime_result
                if regime_result.scores:
                    for key, val in regime_result.scores.items():
                        all_scores[f"{tf_name}_{key}"] = val
        
        # Determine dominant regime (weighted by timeframe importance)
        # 1D and 4H are most important
        regime_weights = {"1d": 0.4, "4h": 0.3, "1h": 0.2, "1w": 0.1}
        regime_scores = {}
        
        for tf_name, weight in regime_weights.items():
            if tf_name in multi_tf_regimes:
                reg = multi_tf_regimes[tf_name]
                regime_label = reg.regime
                
                # Map to our RegimeType enum
                if regime_label == "TREND_UP":
                    regime_type = RegimeType.WEAK_BULL
                elif regime_label == "TREND_DOWN":
                    regime_type = RegimeType.WEAK_BEAR
                elif regime_label == "RANGING":
                    regime_type = RegimeType.RANGE
                elif regime_label == "HIGH_VOL_RISK":
                    regime_type = RegimeType.HIGH_VOL_DEFENSIVE
                elif regime_label == "RISK_OFF":
                    regime_type = RegimeType.RISK_OFF
                else:
                    regime_type = RegimeType.RANGE
                
                if regime_type not in regime_scores:
                    regime_scores[regime_type] = 0.0
                regime_scores[regime_type] += weight * (reg.confidence or 0.5)
        
        # Composite 5-level regime scoring
        closes = []
        if context.candles_4h:
            closes = [float(c[4]) for c in context.candles_4h if len(c) >= 5]
        elif context.candles_1d:
            closes = [float(c[4]) for c in context.candles_1d if len(c) >= 5]
        price = float(closes[-1]) if closes else float(context.last_price or 0.0)
        ema_slope_pct = self._ema_slope_pct(closes)
        vwap_val = self._vwap(context.candles_4h or context.candles_1d or [])
        vwap_dist = ((price - vwap_val) / vwap_val) if vwap_val and vwap_val > 0 else 0.0
        atr_pctile = self._atr_percentile(context.candles_1d or [])
        mom_decay = self._momentum_decay(closes)
        alignment = self._mtf_alignment(multi_tf_regimes)
        align_norm = (alignment + 1.0) / 2.0

        def _pos(x: float) -> float:
            return max(0.0, x)

        bull_strength = clamp(
            align_norm
            + _pos(ema_slope_pct * 5.0)
            + _pos(vwap_dist * 3.0)
            + _pos(mom_decay * 4.0)
            - _pos(atr_pctile - 0.7),
            0.0,
            1.0,
        )
        bear_strength = clamp(
            (1.0 - align_norm)
            + _pos(-ema_slope_pct * 5.0)
            + _pos(-vwap_dist * 3.0)
            + _pos(-mom_decay * 4.0)
            + _pos(atr_pctile - 0.7),
            0.0,
            1.0,
        )
        range_strength = clamp(1.0 - max(bull_strength, bear_strength), 0.0, 1.0)

        all_scores.update(
            {
                "mtf_alignment": alignment,
                "ema_slope_pct": ema_slope_pct,
                "vwap_dist": vwap_dist,
                "atr_percentile": atr_pctile,
                "momentum_decay": mom_decay,
                "bull_strength": bull_strength,
                "bear_strength": bear_strength,
                "range_strength": range_strength,
            }
        )

        if bull_strength >= 0.7:
            dominant_regime = RegimeType.STRONG_BULL
            confidence = bull_strength
        elif bull_strength >= 0.55:
            dominant_regime = RegimeType.WEAK_BULL
            confidence = bull_strength
        elif bear_strength >= 0.7:
            dominant_regime = RegimeType.STRONG_BEAR
            confidence = bear_strength
        elif bear_strength >= 0.55:
            dominant_regime = RegimeType.WEAK_BEAR
            confidence = bear_strength
        else:
            dominant_regime = RegimeType.RANGE
            confidence = max(range_strength, 0.3)
        
        # Apply hysteresis
        cache_key = context.symbol
        if cache_key not in self._hysteresis_cache:
            self._hysteresis_cache[cache_key] = {
                "regime": dominant_regime,
                "confidence": confidence,
                "last_change_ts": context.now_ts,
            }
        else:
            cached = self._hysteresis_cache[cache_key]
            last_regime = cached["regime"]
            last_change_ts = cached.get("last_change_ts", 0)
            
            # Only change regime if confidence is high AND enough time has passed
            if dominant_regime != last_regime:
                time_since_change = context.now_ts - last_change_ts
                if confidence >= self.regime_confidence_threshold and time_since_change >= self.regime_hysteresis_seconds:
                    # Regime change allowed
                    self._hysteresis_cache[cache_key] = {
                        "regime": dominant_regime,
                        "confidence": confidence,
                        "last_change_ts": context.now_ts,
                    }
                else:
                    # Keep old regime (hysteresis)
                    dominant_regime = last_regime
                    confidence = cached.get("confidence", 0.5)
        
        # TTL based on confidence (higher confidence = longer TTL)
        ttl_seconds = int(self.regime_hysteresis_seconds * (0.5 + confidence))

        # Phase 2 Advanced: ML regime blending
        if ENABLE_ML_REGIME and context.candles_1d and len(context.candles_1d) >= 60:
            try:
                from ml_regime_detector import MLRegimeDetector
                ml = MLRegimeDetector()
                ml_result = ml.predict_regime(context.candles_1d)
                if ml_result.get("confidence", 0) > 0.75:
                    ml_regime = str(ml_result.get("regime", "RANGE")).upper()
                    ml_to_enum = {"STRONG_BULL": RegimeType.STRONG_BULL, "WEAK_BULL": RegimeType.WEAK_BULL, "RANGE": RegimeType.RANGE,
                                 "WEAK_BEAR": RegimeType.WEAK_BEAR, "STRONG_BEAR": RegimeType.STRONG_BEAR, "HIGH_VOL": RegimeType.HIGH_VOL_DEFENSIVE}
                    if ml_regime in ml_to_enum:
                        dominant_regime = ml_to_enum[ml_regime]
                        confidence = min(0.95, (confidence + ml_result["confidence"]) / 2)
                        all_scores["ml_regime"] = ml_regime
                        all_scores["ml_confidence"] = ml_result["confidence"]
                elif ml_result.get("regime") == str(dominant_regime.value if hasattr(dominant_regime, "value") else dominant_regime):
                    confidence = min(0.95, (confidence + ml_result.get("confidence", 0.5)) / 2)
                    all_scores["ml_agree"] = True
            except Exception as e:
                logger.debug("ML regime blend skipped: %s", e)
        
        return RegimeDetectionResult(
            regime=dominant_regime,
            confidence=confidence,
            ttl_seconds=ttl_seconds,
            hysteresis_state=self._hysteresis_cache.get(cache_key, {}),
            multi_timeframe_regimes=multi_tf_regimes,
            scores=all_scores,
        )
    
    # =========================
    # Layer 4: Strategy Routing
    # =========================
    
    def _route_strategy(self, context: IntelligenceContext, regime: RegimeDetectionResult) -> StrategyRoutingResult:
        """
        Route to appropriate strategy based on regime.
        C1: Strategy selection uses candles from same provider (MarketDataRouter per symbol:
        Kraken for crypto, Alpaca for stocks). All timeframes (1h, 4h, 1d) come from one source.
        """
        regime_type = regime.regime
        confidence = regime.confidence
        reasons = []
        
        # Check if forced strategy in bot config
        forced_strategy = context.bot_config.get("forced_strategy", "").strip()
        if forced_strategy:
            return StrategyRoutingResult(
                strategy_mode=forced_strategy,
                entry_style="moderate",
                exit_style="fixed_tp",
                reasons=[f"Forced strategy: {forced_strategy}"],
            )
        
        # ============================================================
        # PHASE 3 INTEGRATION: ML Prediction for Strategy Routing
        # ============================================================
        ml_signal = None
        if PHASE3_AVAILABLE:
            try:
                # Get or create ML predictor for this symbol
                if context.symbol not in self._ml_predictors:
                    self._ml_predictors[context.symbol] = create_ml_predictor(use_advanced=False)
                
                predictor = self._ml_predictors[context.symbol]
                
                # Get best available candles for prediction
                prediction_candles = context.candles_1d or context.candles_4h or context.candles_1h
                if prediction_candles:
                    ml_prediction = predictor.predict(
                        candles=prediction_candles,
                        current_price=context.last_price,
                        symbol=context.symbol
                    )
                    
                    if ml_prediction:
                        ml_signal = ml_prediction.direction
                        reasons.append(f"ML prediction: {ml_prediction.direction} ({ml_prediction.confidence:.0%} confidence)")
                        try:
                            from ml_prediction_tracker import log_prediction
                            regime_str = regime_type.value if hasattr(regime_type, "value") else str(regime_type)
                            log_prediction(
                                symbol=context.symbol,
                                predicted_direction=ml_prediction.direction,
                                confidence=ml_prediction.confidence,
                                predicted_price=ml_prediction.price_prediction,
                                price_at_prediction=context.last_price,
                                model_used=ml_prediction.model_used,
                                regime_at_prediction=regime_str,
                            )
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
        
        # Route based on regime + ML signal (5-level)
        if regime_type in (RegimeType.STRONG_BULL, RegimeType.WEAK_BULL, RegimeType.BULL) and confidence >= 0.6:
            strategy_mode = "trend_follow"
            entry_style = "aggressive" if confidence >= 0.8 else "moderate"
            exit_style = "trailing"
            reasons.append("Bull regime: trend-follow enabled")
        elif regime_type == RegimeType.RANGE and confidence >= 0.6:
            strategy_mode = "range_mean_reversion"
            entry_style = "moderate"
            exit_style = "fixed_tp"
            reasons.append("Range-bound market detected")
        elif regime_type in (RegimeType.WEAK_BEAR, RegimeType.STRONG_BEAR, RegimeType.BEAR, RegimeType.HIGH_VOL_DEFENSIVE, RegimeType.RISK_OFF):
            strategy_mode = "high_vol_defensive"
            entry_style = "probe_only"
            exit_style = "regime_based"
            reasons.append("High volatility or risk-off regime")
        else:
            # Default
            strategy_mode = "smart_dca"
            entry_style = "moderate"
            exit_style = "fixed_tp"
            reasons.append("Default routing")
        
        # Override with ML signal if strong
        if ml_signal == "UP" and regime_type not in (RegimeType.STRONG_BEAR, RegimeType.WEAK_BEAR, RegimeType.BEAR):
            if strategy_mode == "smart_dca":
                strategy_mode = "trend_follow"
                reasons.append("ML UP signal: upgrading to trend_follow")
        elif ml_signal == "DOWN" and regime_type in (RegimeType.STRONG_BULL, RegimeType.WEAK_BULL, RegimeType.BULL):
            strategy_mode = "high_vol_defensive"
            reasons.append("ML DOWN signal conflicts with bull regime: defensive mode")

        # Volatility override: adapt to real-time vol shifts
        atr_pct = 0.0
        try:
            atr_val = _atr(context.candles_1d or context.candles_4h or [], 14)
            if atr_val and context.last_price > 0:
                atr_pct = atr_val / context.last_price
        except Exception:
            pass
        if atr_pct >= 0.08 and strategy_mode != "high_vol_defensive":
            strategy_mode = "high_vol_defensive"
            reasons.append("High volatility override (ATR% >= 8%)")
        elif atr_pct >= 0.05 and strategy_mode in ("range_mean_reversion", "trend_follow"):
            strategy_mode = "high_vol_defensive"
            reasons.append("Elevated volatility: defensive override")

        # BTC regime override for alts
        btc_regime = context.btc_context.get("regime") if context.btc_context else None
        if btc_regime in ("STRONG_BEAR", "WEAK_BEAR", "BEAR", RegimeType.STRONG_BEAR, RegimeType.WEAK_BEAR, RegimeType.BEAR):
            if context.symbol not in ("XBT/USD", "BTC/USD"):
                strategy_mode = "high_vol_defensive"
                reasons.append("BTC strong/weak bear: defensive override for alts")
        
        return StrategyRoutingResult(
            strategy_mode=strategy_mode,
            entry_style=entry_style,
            exit_style=exit_style,
            reasons=reasons,
        )
    
    # =========================
    # Layer 5: Position Sizing
    # =========================

    def _strategy_perf_scale(self, bot_id: int, strategy_mode: str, symbol: str, regime: str) -> Tuple[float, List[str]]:
        reasons: List[str] = []
        if not bot_id or not strategy_mode:
            return 1.0, reasons
        try:
            perf = get_expected_edge(symbol, regime, strategy_mode, window=100, prior_weight=50)
            trades = int(perf.get("trades", 0))
            edge = float(perf.get("expected_edge", 0.0))
            # Anti-overfitting: require >=100 trades before disabling, with hysteresis
            desired_state = "normal"
            if trades >= 100 and edge <= 0:
                if strategy_mode == "high_vol_defensive":
                    reasons.append("Defensive strategy protected from full disable")
                    desired_state = "reduced"
                else:
                    desired_state = "disabled"
            elif trades >= 50 and edge <= 0:
                desired_state = "reduced"

            key = f"{bot_id}:{symbol}:{regime}:{strategy_mode}"
            state = self._strategy_state.get(key) or {"state": "normal", "ts": 0}
            now = int(time.time())
            if desired_state != state["state"] and (now - int(state.get("ts", 0))) < 600:
                desired_state = state["state"]
            else:
                self._strategy_state[key] = {"state": desired_state, "ts": now}

            if desired_state == "disabled":
                reasons.append(f"Strategy underperforming (edge={edge:.4f}): disabled")
                return 0.0, reasons
            if desired_state == "reduced":
                reasons.append(f"Strategy underperforming (edge={edge:.4f}): size reduced")
                return 0.5, reasons
        except Exception:
            pass
        return 1.0, reasons
    
    def _calculate_position_sizing(
        self,
        context: IntelligenceContext,
        regime: RegimeDetectionResult,
        strategy: StrategyRoutingResult,
        safety: MarketSafetyResult,
    ) -> PositionSizingResult:
        """Calculate position sizes based on adaptive risk sizing."""
        reasons = []
        
        # ============================================================
        # PHASE 2 INTEGRATION: Kelly Criterion Position Sizing
        # ============================================================
        use_kelly = PHASE2_AVAILABLE and context.bot_config.get("use_kelly_sizing", False)
        bot_id = context.bot_config.get("id") or 0
        
        equity = context.portfolio_total_usd if context.portfolio_total_usd > 0 else safety.risk_budget
        base_size = max(0.0, equity * self.base_risk_pct)

        if use_kelly and bot_id > 0:
            try:
                # Precompute volatility for market_data (used below)
                volatility_pct = 0.02
                if context.candles_1d and len(context.candles_1d) >= 14:
                    try:
                        atr_val = _atr(context.candles_1d, 14)
                        if atr_val and context.last_price > 0:
                            volatility_pct = atr_val / context.last_price
                    except Exception:
                        pass
                # Create Kelly sizer with current bot config
                kelly_frac = context.bot_config.get("kelly_fraction", 0.25)
                max_pos_pct = context.bot_config.get("max_position_pct", 0.10)
                kelly_base = KellyPositionSizer(
                    kelly_fraction=kelly_frac,
                    max_position_pct=max_pos_pct,
                )
                kelly_sizer = DynamicPositionSizer(kelly_sizer=kelly_base)
                # Bot stats from DB (populated by bot_manager)
                bot_stats = {
                    "total_trades": context.bot_config.get("total_trades", 0),
                    "winning_trades": context.bot_config.get("winning_trades", 0),
                    "avg_profit_pct": context.bot_config.get("avg_profit_pct", 0.02),
                    "avg_loss_pct": context.bot_config.get("avg_loss_pct", 0.01),
                }
                market_data = {
                    "volatility_pct": volatility_pct,
                    "signal_confidence": regime.confidence,
                    "current_drawdown_pct": context.portfolio_drawdown or 0.0,
                }
                available_capital = safety.risk_budget
                kelly_size, _, _ = kelly_sizer.calculate_position_size(
                    available_capital, bot_stats, market_data
                )
                if kelly_size > 0:
                    base_size = min(base_size, kelly_size)
                    reasons.append(f"Kelly sizing cap: ${base_size:.2f}")
            except Exception as e:
                logger.warning(f"Kelly sizing failed, using base risk sizing: {e}")
        
        # Calculate volatility (ATR)
        volatility_pct = 0.02  # Default 2%
        if context.candles_1d and len(context.candles_1d) >= 14:
            try:
                atr_val = _atr(context.candles_1d, 14)
                if atr_val and context.last_price > 0:
                    volatility_pct = atr_val / context.last_price
            except Exception:
                pass
        
        # Adjust size based on volatility (moderate reduction to stay in the game)
        volatility_adjusted = False
        if volatility_pct > 0.08:  # Very high vol (>8%)
            base_size *= 0.5
            volatility_adjusted = True
            reasons.append(f"Very high volatility ({volatility_pct*100:.2f}%): size halved")
        elif volatility_pct > 0.05:  # High vol (>5%)
            base_size *= 0.75
            volatility_adjusted = True
            reasons.append(f"High volatility ({volatility_pct*100:.2f}%): size reduced 25%")
        
        # Correlation + BTC risk-off adjustments
        btc_risk_adjusted = False
        btc_corr = float(context.btc_context.get("corr", 0.0) or 0.0) if context.btc_context else 0.0
        if btc_corr >= 0.6:
            scale = max(0.5, 1.0 - (btc_corr - 0.6) * 0.5)
            base_size *= scale
            reasons.append(f"BTC correlation {btc_corr:.2f}: size scaled x{scale:.2f}")
        if context.btc_context.get("risk_off", False) and context.symbol not in ["XBT/USD", "BTC/USD"]:
            base_size *= 0.5
            btc_risk_adjusted = True
            reasons.append("BTC risk-off: altcoin size reduced")

        # Drawdown adjustment
        if context.portfolio_drawdown >= 0.10:
            dd_scale = max(0.4, 1.0 - context.portfolio_drawdown * 1.5)
            base_size *= dd_scale
            reasons.append(f"Drawdown {context.portfolio_drawdown*100:.1f}%: size scaled x{dd_scale:.2f}")

        # Earnings proximity (stocks)
        earnings_days = context.bot_config.get("earnings_days")
        if isinstance(earnings_days, (int, float)) and earnings_days >= 0:
            if earnings_days <= 5:
                base_size *= 0.5
                reasons.append(f"Earnings in {int(earnings_days)}d: size reduced")

        # Strategy performance scaling
        perf_scale, perf_reasons = self._strategy_perf_scale(
            bot_id,
            strategy.strategy_mode,
            context.symbol,
            _safe_enum_val(regime.regime),
        )
        if perf_scale <= 0:
            base_size = 0.0
        else:
            base_size *= perf_scale
        if perf_reasons:
            reasons.extend(perf_reasons)
        
        # Hard caps: per-symbol + total exposure + daily loss
        if context.portfolio_total_usd > 0:
            per_symbol_cap = context.portfolio_total_usd * self.per_symbol_exposure_pct
            base_size = min(base_size, per_symbol_cap)
        if context.portfolio_exposure_pct >= self.max_exposure_pct:
            base_size = 0.0
            reasons.append("Total exposure cap hit")
        if context.portfolio_total_usd > 0:
            daily_loss_pct = abs(context.daily_realized_pnl) / context.portfolio_total_usd
            if context.daily_realized_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
                base_size = 0.0
                reasons.append("Daily loss cap hit")

        # Cap by risk budget
        base_size = min(base_size, max(0.0, safety.risk_budget))

        # Phase 2 Advanced: Volatility forecast position adjustment
        if ENABLE_VOL_FORECAST and base_size > 0 and context.candles_1d and len(context.candles_1d) >= 100:
            try:
                from volatility_forecaster import VolatilityForecaster
                import numpy as np
                arr = np.array(context.candles_1d)
                close = arr[:, 4].astype(float)
                ret = np.diff(close) / np.maximum(close[:-1], 1e-9)
                vf = VolatilityForecaster()
                vol_fc = vf.forecast_volatility(context.symbol, list(ret))
                adj = vf.adjust_position_size_for_volatility_forecast(base_size, vol_fc)
                base_size = adj["adjusted_size"]
                reasons.append(adj.get("reasoning", "Vol forecast adj")[:80])
            except Exception as e:
                logger.debug("Vol forecast sizing skipped: %s", e)

        # Phase 2 Advanced: Order flow position adjustment
        if ENABLE_ORDER_FLOW and base_size > 0:
            try:
                from order_flow_analyzer import OrderFlowAnalyzer
                mkt = "stock" if (len(context.symbol) < 6 and "/" not in context.symbol) else "crypto"
                ofa = OrderFlowAnalyzer()
                of_result = ofa.analyze_order_flow(context.symbol, mkt)
                sig = of_result.get("smart_money_signal", "neutral")
                strength = float(of_result.get("signal_strength", 0.5))
                if sig == "distribution" and strength > 0.7:
                    base_size *= 0.5
                    reasons.append("Order flow: smart money exiting - size reduced")
                elif sig == "accumulation" and strength > 0.7:
                    base_size *= 1.2
                    reasons.append("Order flow: smart money accumulating - size increased")
            except Exception as e:
                logger.debug("Order flow sizing skipped: %s", e)
        
        # DCA ladder configuration
        ladder_steps = int(context.bot_config.get("max_safety", 3))
        ladder_spacing_pct = float(context.bot_config.get("first_dev", 0.01))  # 1% default
        
        # Adjust ladder spacing for volatility
        if volatility_pct > 0:
            ladder_spacing_pct = max(ladder_spacing_pct, volatility_pct * 0.5)
            ladder_spacing_pct = min(ladder_spacing_pct, volatility_pct * 2.0)
        
        # Reduce ladder in defensive regimes
        if regime.regime in (RegimeType.WEAK_BEAR, RegimeType.STRONG_BEAR, RegimeType.HIGH_VOL_DEFENSIVE, RegimeType.RISK_OFF):
            ladder_steps = min(ladder_steps, 1)
        max_adds = ladder_steps
        
        # Guard against NaN/Inf from upstream calculations
        import math
        if math.isnan(base_size) or math.isinf(base_size):
            base_size = 0.0
            reasons.append("Position size reset: NaN/Inf detected")

        # Minimum position size floor to prevent near-zero orders
        min_position = float(os.getenv("INTEL_MIN_POSITION_SIZE", "2.0"))
        if 0 < base_size < min_position:
            base_size = min_position
            reasons.append(f"Position size floored to ${min_position:.0f} minimum")

        return PositionSizingResult(
            base_size=base_size,
            ladder_steps=ladder_steps,
            ladder_spacing_pct=ladder_spacing_pct,
            max_adds=max_adds,
            volatility_adjusted=volatility_adjusted,
            btc_risk_adjusted=btc_risk_adjusted,
            reasons=reasons,
        )
    
    # =========================
    # Layer 6: Execution Policy
    # =========================
    
    def _determine_execution_policy(
        self, context: IntelligenceContext, regime: RegimeDetectionResult
    ) -> ExecutionPolicyResult:
        """Determine how to execute orders (limit vs market, slippage, etc.)."""
        reasons = []
        
        # Default to post-only limit orders
        order_type = "limit"
        limit_price = context.last_price
        post_only = True
        
        # ============================================================
        # PHASE 2 INTEGRATION: Order Book Analysis
        # ============================================================
        if PHASE2_AVAILABLE and context.bid_price and context.ask_price:
            try:
                # Get or create order book analyzer for this symbol
                if context.symbol not in self._orderbook_analyzers:
                    self._orderbook_analyzers[context.symbol] = OrderBookAnalyzer()
                
                analyzer = self._orderbook_analyzers[context.symbol]
                
                # Analyze order book (would need actual order book data from exchange)
                # For now, use bid/ask spread analysis
                if context.bid_price and context.ask_price:
                    mid_price = (context.bid_price + context.ask_price) / 2
                    spread = context.ask_price - context.bid_price
                    spread_pct = spread / mid_price if mid_price > 0 else 0
                    
                    # Estimate slippage
                    estimated_slippage = analyzer.estimate_slippage(
                        order_size=context.bot_config.get('base_quote', 25.0) / context.last_price,
                        side='buy',
                        order_book_data=None  # Would need actual order book
                    )
                    
                    if estimated_slippage and estimated_slippage > self.max_slippage_pct:
                        # Slippage too high - use limit with post-only
                        order_type = "limit"
                        post_only = True
                        reasons.append(f"High estimated slippage ({estimated_slippage*100:.3f}%): using limit with post-only")
                    elif spread_pct < 0.001 and regime.regime in (RegimeType.STRONG_BULL, RegimeType.WEAK_BULL):
                        # Tight spread - can use market if urgent
                        order_type = "market"
                        post_only = False
                        reasons.append("Tight spread in bull regime: using market order")
                    else:
                        order_type = "limit"
                        reasons.append(f"Spread {spread_pct*100:.2f}%: using limit order")
            except Exception as e:
                logger.warning(f"Order book analysis failed: {e}")
        
        # Fallback to basic logic
        if not reasons:
            # Use market if spread is very tight (< 0.1%)
            if context.spread_pct is not None and context.spread_pct < 0.001 and regime.regime in (RegimeType.STRONG_BULL, RegimeType.WEAK_BULL):
                order_type = "market"
                post_only = False
                reasons.append("Tight spread: using market order")
            elif context.spread_pct is not None and context.spread_pct > 0.002:
                # Wide spread: use limit with post-only
                order_type = "limit"
                post_only = True
                reasons.append(f"Wide spread ({context.spread_pct*100:.2f}%): using limit with post-only")
        
        # Max slippage guard
        max_slippage_pct = self.max_slippage_pct
        
        # Cooldown between actions
        min_cooldown_seconds = 2  # Default 2 seconds
        
        # Urgency-based routing: high=breakout/stop, medium=limit+buffer, low=DCA patient
        urgency = "medium"
        if order_type == "market":
            urgency = "high"
        elif regime.regime in (RegimeType.BREAKOUT, RegimeType.STRONG_BULL) and order_type == "limit":
            urgency = "medium"
        elif regime.regime in (RegimeType.RANGE,) or "DCA" in str(context.bot_config.get("strategy_mode", "")):
            urgency = "low"
        
        return ExecutionPolicyResult(
            order_type=order_type,
            limit_price=limit_price,
            max_slippage_pct=max_slippage_pct,
            post_only=post_only,
            min_cooldown_seconds=min_cooldown_seconds,
            reasons=reasons,
            urgency=urgency,
        )
    
    # =========================
    # Layer 7: Trade Management
    # =========================
    
    def _manage_trades(
        self, context: IntelligenceContext, regime: RegimeDetectionResult
    ) -> TradeManagementResult:
        """Manage existing positions (exits, trailing stops, etc.)."""
        manage_actions = []
        reasons = []
        
        if context.current_position_size == 0 or context.avg_entry_price is None:
            # No position to manage
            return TradeManagementResult(
                manage_actions=manage_actions,
                reasons=["No open position to manage"],
            )

        # EXIT_BEFORE_EARNINGS: reduce/exit 2 days before earnings (stocks)
        if (context.market_breadth or {}).get("is_stock"):
            try:
                from earnings_calendar import should_reduce_before_earnings
                should_reduce, reason = should_reduce_before_earnings(context.symbol)
                if should_reduce and reason:
                    return TradeManagementResult(
                        manage_actions=[{"action": "exit_all", "reason": reason, "price": context.last_price}],
                        forced_de_risk={"reason": reason, "priority": "high"},
                        reasons=[reason],
                    )
            except Exception:
                pass

        # Calculate current PnL
        if context.avg_entry_price > 0:
            pnl_pct = (context.last_price - context.avg_entry_price) / context.avg_entry_price
        else:
            pnl_pct = 0.0

        # Mode config for stop/TP (Master Upgrade Part 1)
        mode_config = {}
        trading_mode = context.bot_config.get("trading_mode") or ""
        if trading_mode:
            try:
                from trading_modes import get_mode_config
                conviction = int(context.bot_config.get("conviction_level") or 5)
                mode_config = get_mode_config(trading_mode, conviction)
            except Exception:
                pass

        # Fixed stop-loss check (mode-specific)
        stop_loss_pct = float(mode_config.get("stop_loss_pct", -5.0)) / 100.0
        if pnl_pct <= stop_loss_pct and context.avg_entry_price > 0:
            manage_actions.append({
                "action": "legacy_stop_loss",
                "reason": f"Stop loss hit ({pnl_pct*100:.1f}%)",
                "price": context.last_price,
            })
            reasons.append(f"Stop loss triggered at {pnl_pct*100:.1f}%")
            return TradeManagementResult(
                manage_actions=manage_actions,
                partial_tp=None,
                trailing_stop=None,
                break_even=None,
                reasons=reasons,
            )

        # ============================================================
        # PHASE 1 INTEGRATION: Adaptive TP Scaling
        # ============================================================
        base_tp = float(context.bot_config.get("tp", 0.01))  # Default 1%
        tp_from_mode = mode_config.get("take_profit_pct")
        if tp_from_mode is not None:
            base_tp = float(tp_from_mode) / 100.0
        tp_pct = base_tp

        if PHASE1_AVAILABLE and context.bot_config.get('adaptive_tp_enabled', 1):
            try:
                phase1 = Phase1Intelligence(context.bot_config)
                
                # Calculate ATR percentage for volatility
                atr_pct = None
                if context.candles_1d and len(context.candles_1d) >= 14:
                    try:
                        atr_val = _atr(context.candles_1d, 14)
                        if atr_val and context.last_price > 0:
                            atr_pct = atr_val / context.last_price
                    except Exception:
                        pass
                
                # Get adaptive TP
                tp_pct = phase1.get_adaptive_tp(atr_pct)
                if tp_pct != base_tp:
                    reasons.append(f"Adaptive TP: {tp_pct*100:.2f}% (base: {base_tp*100:.2f}%)")
            except Exception as e:
                logger.warning(f"Adaptive TP calculation failed: {e}")
        
        # Partial take-profit (if > 2x TP target)
        if pnl_pct >= tp_pct * 2:  # 2x TP target
            partial_tp = {
                "action": "partial_tp",
                "size_pct": 0.5,  # Take 50% profit
                "price": context.last_price,
            }
            manage_actions.append(partial_tp)
            reasons.append(f"Partial TP: {pnl_pct*100:.2f}% profit")
        
        # ============================================================
        # PHASE 1 INTEGRATION: Trailing Stop Loss
        # ============================================================
        trailing_stop = None
        if PHASE1_AVAILABLE and context.bot_config.get('trailing_stop_enabled', 1):
            try:
                phase1 = Phase1Intelligence(context.bot_config)
                
                # Get deal data (would need to fetch from DB in real implementation)
                deal_data = {
                    'entry_avg': context.avg_entry_price,
                    'highest_price': context.bot_config.get('highest_price'),
                    'trailing_stop_active': context.bot_config.get('trailing_stop_active', False),
                }
                
                should_exit, updated_data = phase1.update_trailing_stop(
                    deal=deal_data,
                    current_price=context.last_price
                )
                
                if should_exit:
                    # Trailing stop hit - exit position
                    manage_actions.append({
                        "action": "exit_all",
                        "reason": updated_data.get('reason', "Trailing stop hit"),
                        "price": context.last_price,
                    })
                    reasons.append("Trailing stop triggered exit")
                elif updated_data.get('trailing_stop_price'):
                    # Update trailing stop price
                    trailing_stop = {
                        "action": "trailing_stop",
                        "price": updated_data['trailing_stop_price'],
                        "highest_price": updated_data.get('highest_price'),
                        "distance_pct": abs((updated_data['trailing_stop_price'] - context.last_price) / context.last_price),
                    }
                    manage_actions.append(trailing_stop)
                    reasons.append(f"Trailing stop active at ${updated_data['trailing_stop_price']:.6f}")
            except Exception as e:
                logger.warning(f"Phase 1 trailing stop failed: {e}")
        
        # Fallback to basic trailing stop if Phase 1 not available
        if not trailing_stop and pnl_pct > 0 and regime.regime in (RegimeType.STRONG_BULL, RegimeType.WEAK_BULL, RegimeType.BULL):
            # Calculate trailing stop based on ATR
            if context.candles_1d and len(context.candles_1d) >= 14:
                try:
                    atr_val = _atr(context.candles_1d, 14)
                    if atr_val and context.last_price > 0:
                        trail_distance = atr_val * 2.0  # 2x ATR
                        trail_price = context.last_price - trail_distance
                        if trail_price > context.avg_entry_price:  # Above break-even
                            trailing_stop = {
                                "action": "trailing_stop",
                                "price": trail_price,
                                "distance_pct": (trail_distance / context.last_price),
                            }
                            manage_actions.append(trailing_stop)
                            reasons.append("Trailing stop active (basic)")
                except Exception:
                    pass
        
        # Break-even after partial TP
        if pnl_pct >= tp_pct and context.avg_entry_price:
            break_even = {
                "action": "break_even",
                "price": context.avg_entry_price,
            }
            manage_actions.append(break_even)
            reasons.append("Break-even stop set")
        
        # Time-based stop (if position held too long without profit)
        # This would require position open time, which we'd need to add to context
        
        return TradeManagementResult(
            manage_actions=manage_actions,
            partial_tp=manage_actions[0] if manage_actions and manage_actions[0].get("action") == "partial_tp" else None,
            trailing_stop=next((a for a in manage_actions if a.get("action") == "trailing_stop"), None),
            break_even=next((a for a in manage_actions if a.get("action") == "break_even"), None),
            reasons=reasons,
        )
    
    # =========================
    # Helper Methods
    # =========================

    def generate_recommendation(self, context: IntelligenceContext, horizon: str) -> Dict[str, Any]:
        """
        Generate a recommendation suitable for the worker API / frontend.
        Phase 2 Upgrade: Multi-factor scoring, deep explainability, and guardrails.
        """
        # Run full evaluation
        decision = self.evaluate(context)
        
        # Initialize
        score = 0.0
        eligible = False
        reasons = []
        risk_flags = []
        long_term_downtrend_no_proof = False
        
        # --- 1. Eligibility Check (Hard Gates) ---
        # C3: Data validity gate — exclude symbols with data failures
        if decision.data_validity.data_ok:
            eligible = True
        else:
            risk_flags.append("DATA_INVALID")
            reasons.extend(decision.data_validity.reasons or ["Data invalid"])
            
        if decision.market_safety.allowed_actions == AllowedAction.NO_TRADE:
            eligible = False
            reasons.extend(decision.market_safety.reasons)
            if decision.market_safety.btc_risk_off:
                 risk_flags.append("btc_risk_off")

        # --- 2. Scoring Model ---
        _ml_metrics = {}
        if eligible:
            # A. Regime Foundation (0-100 base)
            regime = decision.regime_detection.regime
            conf = decision.regime_detection.confidence
            
            base, conf_mult = self.score_strong_bull
            if regime in (RegimeType.STRONG_BULL, RegimeType.BULL):
                score = base + (conf * conf_mult)
                reasons.append("Strong bull regime detected")
            elif regime == RegimeType.WEAK_BULL:
                base, conf_mult = self.score_weak_bull
                score = base + (conf * conf_mult)
                reasons.append("Weak bull regime")
            elif regime == RegimeType.BREAKOUT:
                base, conf_mult = self.score_breakout
                score = base + (conf * conf_mult)
                reasons.append("Breakout pattern confirmed")
            elif regime == RegimeType.RANGE:
                base, conf_mult = self.score_range
                score = base + (conf * conf_mult)
                reasons.append("Ranging market")
            elif regime == RegimeType.HIGH_VOL_DEFENSIVE:
                score = self.score_high_vol
                risk_flags.append("High volatility regime")
            elif regime == RegimeType.WEAK_BEAR:
                score = self.score_weak_bear
                reasons.append("Weak bear regime")
            elif regime in (RegimeType.STRONG_BEAR, RegimeType.BEAR):
                score = self.score_strong_bear
                reasons.append("Bearish regime")
            elif regime == RegimeType.RISK_OFF:
                score = self.score_risk_off
                risk_flags.append("Risk-off conditions")

            # B. Trend Alignment (Bonus, cap +12 total)
            # Check EMA alignment - cap combined trend bonuses to avoid all 100s
            closes_1d = [c[4] for c in context.candles_1d] if context.candles_1d else []
            closes_1w = [c[4] for c in context.candles_1w] if context.candles_1w else []
            
            ema50_1d = ema(closes_1d, 50)
            ema200_1d = ema(closes_1d, 200)
            ema50_1w = ema(closes_1w, 50)
            
            trend_bonus = 0.0
            if ema50_1d and ema200_1d and ema50_1d > ema200_1d:
                trend_bonus += 6.0
                reasons.append("Daily Golden Cross area")
            if ema50_1w and context.last_price > ema50_1w:
                trend_bonus += 6.0
                reasons.append("Price above Weekly EMA50")
            score += min(trend_bonus, 12.0)  # Cap trend at +12

            # B1b. Horizon-aware bonus: short = recent momentum; long = established trend (200 SMA, weekly)
            horizon_str = (str(horizon or "short")).strip().lower()
            if horizon_str == "short" and regime in (RegimeType.BREAKOUT, RegimeType.STRONG_BULL, RegimeType.BULL):
                ret_5d = rolling_return(closes_1d, 5) if closes_1d and len(closes_1d) >= 6 else None
                if ret_5d is not None and ret_5d > 0.01:
                    score += 3.0
                    reasons.append("Short-term: strong 5d momentum")
            elif horizon_str == "long":
                ema200_1d = ema(closes_1d, 200) if closes_1d else None
                if ema200_1d and context.last_price > ema200_1d and ema50_1w and context.last_price > ema50_1w:
                    score += 3.0
                    reasons.append("Long-term: above 200 SMA and weekly trend")

            # B2. Pattern Recognition (swing: H&S, double top/bottom, flags, triangles)
            try:
                from pattern_recognition import detect_patterns, pattern_score_boost
                pattern_candles = context.candles_1d or context.candles_4h or []
                if len(pattern_candles) >= 20:
                    patterns = detect_patterns(pattern_candles)
                    direction = "bullish" if regime in (RegimeType.STRONG_BULL, RegimeType.BULL, RegimeType.WEAK_BULL, RegimeType.BREAKOUT) else "bearish"
                    pattern_boost = pattern_score_boost(patterns, direction)
                    if pattern_boost > 0:
                        score += pattern_boost
                        reasons.append(f"Pattern support (+{pattern_boost:.0f})")
            except Exception:
                pass

            # B3. Alternative Data (social, news, insider, options, on-chain) - 10% weight
            try:
                from alternative_data import calculate_alternative_data_score
                alt = calculate_alternative_data_score(context.symbol)
                if alt.sources_used and alt.confidence > 0:
                    contrib = (alt.score - 50) / 5.0  # -10 to +10 points
                    score += contrib
                    reasons.append(f"Alt data: {','.join(alt.sources_used)}")
                    if alt.short_squeeze_candidate:
                        score += 2.0
                        risk_flags.append("Short squeeze candidate")
                    if alt.high_short_interest_risk:
                        risk_flags.append("High short interest")
                    if alt.alerts:
                        risk_flags.extend(alt.alerts[:2])
            except Exception:
                pass

            # B4. ML Prediction (15% weight if accuracy >65%, 0% if <55%, ensemble agreement filter)
            try:
                from ml_prediction_tracker import get_ml_score_for_recommendation, ENABLE_ML_PREDICTIONS
                if ENABLE_ML_PREDICTIONS:
                    candles_ml = context.candles_1d or context.candles_4h or []
                    regime_str = decision.regime_detection.regime.value if hasattr(decision.regime_detection.regime, "value") else str(decision.regime_detection.regime)
                    ml_delta, ml_conviction, ml_details = get_ml_score_for_recommendation(
                        context.symbol, candles_ml, context.last_price, regime_str
                    )
                    if ml_delta != 0:
                        score += ml_delta
                        reasons.append(f"ML {ml_details.get('direction', '')} ({ml_conviction})")
                        if ml_conviction == "high_conviction":
                            reasons.append("ML ensemble agreement")
                    _ml_metrics = {"ml_accuracy": ml_details.get("accuracy"), "ml_conviction": ml_conviction} if ml_details else {}
                else:
                    _ml_metrics = {}
            except Exception:
                _ml_metrics = {}

            # B5. Style-based bonus (RECO_STYLE: momentum | value | mean_reversion | breakout | balanced)
            _style = (os.getenv("RECO_STYLE", "balanced") or "balanced").strip().lower()
            rsi_val_style = rsi(closes_1d or [], 14) if closes_1d else None
            ret_5d_style = rolling_return(closes_1d, 5) if closes_1d and len(closes_1d) >= 6 else None
            if _style == "momentum" and regime in (RegimeType.STRONG_BULL, RegimeType.BULL, RegimeType.WEAK_BULL):
                if ret_5d_style is not None and ret_5d_style > 0.02:
                    score += 4.0
                    reasons.append("Style(momentum): strong short-term momentum")
            elif _style == "value" and regime in (RegimeType.RANGE, RegimeType.WEAK_BULL):
                if rsi_val_style is not None and rsi_val_style <= 35:
                    score += 4.0
                    reasons.append("Style(value): oversold / dip setup")
            elif _style == "mean_reversion" and regime == RegimeType.RANGE:
                if rsi_val_style is not None and rsi_val_style <= 40:
                    score += 3.0
                    reasons.append("Style(mean_reversion): range + oversold")
            elif _style == "breakout" and regime == RegimeType.BREAKOUT:
                score += 4.0
                reasons.append("Style(breakout): breakout regime")
            # balanced: no extra style bonus

            # C. Momentum & Returns (Bonus +10 / Penalty -15, cap +10)
            # Calc 30d return - reduced max to spread scores
            momentum_bonus = 0.0
            if len(closes_1d) >= 30:
                ret_30 = (closes_1d[-1] - closes_1d[-30]) / closes_1d[-30]
                if ret_30 > 0.20: momentum_bonus += 8.0   # Strong momentum
                elif ret_30 > 0.05: momentum_bonus += 4.0
                elif ret_30 < -0.10: momentum_bonus -= 5.0
                elif ret_30 < -0.25: momentum_bonus -= 10.0
            score += momentum_bonus

            # C1b. Multi-timeframe momentum (5d + 10d weighted for better signal quality)
            mtf_bonus = 0.0
            if len(closes_1d) >= 11:
                denom_5 = float(closes_1d[-6]) if closes_1d[-6] != 0 else 1.0
                denom_10 = float(closes_1d[-11]) if closes_1d[-11] != 0 else 1.0
                ret_5d_mtf = (closes_1d[-1] - closes_1d[-6]) / denom_5
                ret_10d_mtf = (closes_1d[-1] - closes_1d[-11]) / denom_10
                if ret_5d_mtf > 0 and ret_10d_mtf > 0:
                    mtf_bonus = min(6.0, (ret_5d_mtf * 40 + ret_10d_mtf * 20))
                    if mtf_bonus >= 2.0:
                        reasons.append(f"Multi-TF momentum: 5d +{ret_5d_mtf:.1%}, 10d +{ret_10d_mtf:.1%}")
                elif ret_5d_mtf < -0.03 and ret_10d_mtf < -0.05:
                    mtf_bonus = max(-8.0, (ret_5d_mtf * 30 + ret_10d_mtf * 20))
                    risk_flags.append(f"Declining momentum: 5d {ret_5d_mtf:.1%}, 10d {ret_10d_mtf:.1%}")
            score += mtf_bonus

            # C1c. Volume confirmation (above-average volume confirms move)
            vol_bonus = 0.0
            if context.candles_1d and len(context.candles_1d) >= 21:
                recent_vol = float(context.candles_1d[-1][5]) if len(context.candles_1d[-1]) > 5 else 0
                avg_vol = sum(float(c[5]) for c in context.candles_1d[-21:-1] if len(c) > 5) / 20.0
                if avg_vol > 0 and recent_vol > 0:
                    vol_ratio = recent_vol / avg_vol
                    if vol_ratio >= 2.0:
                        vol_bonus = 5.0
                        reasons.append(f"Volume surge {vol_ratio:.1f}x average")
                    elif vol_ratio >= 1.3:
                        vol_bonus = 2.0
                        reasons.append(f"Above-average volume ({vol_ratio:.1f}x)")
                    elif vol_ratio < 0.5:
                        vol_bonus = -3.0
                        risk_flags.append("Very low volume (weak conviction)")
            score += vol_bonus

            # C1d. Trend strength via ADX (strong trends are more reliable)
            try:
                from strategies import adx as _adx_fn
                adx_val = _adx_fn(context.candles_1d, 14)
                if adx_val is not None:
                    if adx_val >= 30:
                        score += 4.0
                        reasons.append(f"Strong trend (ADX {adx_val:.0f})")
                    elif adx_val >= 22:
                        score += 2.0
                    elif adx_val < 15:
                        score -= 2.0
                        reasons.append(f"Weak/choppy market (ADX {adx_val:.0f})")
            except Exception:
                pass

            # C1e. Price distance from key MAs (mean reversion / extended move signal)
            if ema50_1d and context.last_price > 0:
                dist_ema50 = (context.last_price - ema50_1d) / ema50_1d
                if dist_ema50 > 0.15:
                    score -= 4.0
                    risk_flags.append(f"Extended above EMA50 ({dist_ema50:.1%})")
                elif dist_ema50 < -0.10 and regime in (RegimeType.RANGE, RegimeType.WEAK_BULL):
                    score += 4.0
                    reasons.append(f"Mean reversion: {abs(dist_ema50):.1%} below EMA50")

            # C2. RSI Oversold Bonus (mean-reversion opportunity in range/bull)
            rsi_val = rsi(closes_1d or [], 14)
            if rsi_val is not None:
                if rsi_val <= 30 and regime in (RegimeType.RANGE, RegimeType.WEAK_BULL, RegimeType.WEAK_BEAR):
                    score += 6.0
                    reasons.append(f"RSI oversold ({rsi_val:.0f}) — mean-reversion opportunity")
                elif rsi_val <= 35 and regime in (RegimeType.RANGE, RegimeType.WEAK_BULL):
                    score += 3.0
                    reasons.append(f"RSI approaching oversold ({rsi_val:.0f})")
                elif rsi_val >= 80:
                    score -= 6.0
                    risk_flags.append(f"RSI overbought ({rsi_val:.0f}) — extended, risk of pullback")
                elif rsi_val >= 70 and regime not in (RegimeType.STRONG_BULL, RegimeType.BREAKOUT):
                    score -= 3.0
                    risk_flags.append(f"RSI overbought ({rsi_val:.0f})")
            
            # D. Drawdown Penalty (Max -20)
            # Don't buy bags
            if len(closes_1d) > 200:
                peak_1y = max(closes_1d[-200:])
                if peak_1y > 0:
                    dd = (peak_1y - closes_1d[-1]) / peak_1y
                    if dd > 0.60: 
                        score -= 20.0
                        risk_flags.append("Deep Drawdown >60%")
                    elif dd > 0.40:
                        score -= 10.0

            # D2. Sustained long-term downtrend - only recommend with proof
            # Downtrend assets need clear reversal evidence (RSI oversold + short-term bounce)
            long_term_downtrend_no_proof = False
            if len(closes_1d) >= 60:
                first_price = float(closes_1d[0])
                last_price_d = float(closes_1d[-1])
                if first_price > 0:
                    long_term_ret = (last_price_d - first_price) / first_price
                    denom = float(closes_1d[-5]) if len(closes_1d) >= 5 else 0.0
                    ret_5d = (closes_1d[-1] - closes_1d[-5]) / denom if denom > 0 else 0.0
                    rsi_for_dd = rsi(closes_1d or [], 14)
                    regime_supports_reversal = regime in (RegimeType.RANGE, RegimeType.WEAK_BULL)

                    if long_term_ret < -0.50:
                        # Hard exclude: never recommend assets down 50%+ over history
                        score -= 20.0
                        risk_flags.append("Sustained downtrend (long-term -50%+)")
                        long_term_downtrend_no_proof = True
                    elif long_term_ret < -0.30:
                        # Only recommend if we have proof of potential reversal
                        has_proof = (
                            (rsi_for_dd is not None and rsi_for_dd <= 35 and ret_5d > 0)  # oversold + bounce
                            or (regime_supports_reversal and ret_5d > 0.08)  # mean-reversion setup
                        )
                        if has_proof:
                            score -= 8.0
                            reasons.append("Reversal setup (downtrend with oversold/early bounce)")
                        else:
                            score -= 18.0
                            risk_flags.append("Long-term downtrend (-30%+), no reversal proof")
                            long_term_downtrend_no_proof = True

            # E. Volatility Guardrails (Max -20)
            atr_val = _atr(context.candles_1d, 14)
            atr_pct = (atr_val / context.last_price) if atr_val and context.last_price > 0 else 0.0
            
            if atr_pct > 0.15: # >15% daily moves is meme territory
                score -= 20.0
                risk_flags.append("Extreme Volatility")
            elif atr_pct > 0.08:
                score -= 5.0
                risk_flags.append("High Volatility")
            elif atr_pct < 0.005:
                score -= 10.0 # Stablecoin or dead
                reasons.append("Low volatility")

            # F. Stock-Specific Scoring (sector, liquidity tier, earnings, sector ETF, market cap, IPO)
            mb = context.market_breadth or {}
            if mb.get("is_stock"):
                sector = mb.get("sector")
                liquidity_tier = mb.get("liquidity_tier", "unknown")
                # Sector ETF downtrend: block entry
                if mb.get("sector_etf_ok") is False:
                    score -= 20.0
                    risk_flags.append(mb.get("sector_etf_reason") or "Sector ETF downtrend")
                # Market cap filter
                if mb.get("market_cap_ok") is False:
                    score -= 25.0
                    risk_flags.append(mb.get("market_cap_reason") or "Market cap filter")
                # Recent IPO: high volatility, limited history
                if mb.get("recent_ipo"):
                    score -= 15.0
                    risk_flags.append(f"Recent IPO (<{mb.get('ipo_days', 90)} days)")
                # Analyst rating contribution
                ac = mb.get("analyst_score_contrib")
                if ac is not None and ac != 0:
                    score += ac
                    if ac > 0:
                        reasons.append(f"Analyst upgrade (+{ac:.0f})")
                    else:
                        risk_flags.append(f"Analyst downgrade ({ac:.0f})")
                # Liquidity tier bonus: mega/large cap more stable
                if liquidity_tier == "mega":
                    score += 3.0
                    reasons.append("Mega-cap liquidity")
                elif liquidity_tier == "large":
                    score += 2.0
                    reasons.append("Large-cap liquidity")
                elif liquidity_tier == "small":
                    score -= 5.0
                    risk_flags.append("Small-cap liquidity")
                # Sector: defensive sectors get small stability bonus in risk-off
                if sector and context.btc_context.get("risk_off"):
                    if sector in ("Healthcare", "Consumer Defensive"):
                        score += 2.0
                        reasons.append(f"{sector} defensive sector")
                # earnings_days: from bot_config or market_breadth
                earnings_days = context.bot_config.get("earnings_days")
                if earnings_days is None:
                    earnings_days = mb.get("earnings_days")
                if isinstance(earnings_days, (int, float)) and 0 <= earnings_days <= 5:
                    score -= 10.0
                    risk_flags.append(f"Earnings in {int(earnings_days)}d")
            
            # G. Macro Context (Risk Off)
            if context.btc_context.get("risk_off") and context.symbol not in ("BTC/USD", "XBT/USD", "USDT/USD"):
                score -= 20.0
                risk_flags.append("Macro Risk-Off")

            # H. Earnings proximity filter (stocks)
            earnings_days = context.bot_config.get("earnings_days") or (context.market_breadth or {}).get("earnings_days")
            if isinstance(earnings_days, (int, float)) and earnings_days >= 0:
                if earnings_days <= 5:
                    score -= 10.0
                    risk_flags.append(f"Earnings in {int(earnings_days)}d")

        # --- 3. Final Clamping & Formatting ---
        # Cap at 95.0 to preserve differentiation at top (avoids "all 100" clustering)
        score = clamp(score, 0.0, 95.0)
        score = round(score, 1)

        # Min score for eligibility: profile-based (conservative=stricter, aggressive=looser)
        _profile = (os.getenv("RECO_PROFILE", "balanced") or "balanced").strip().lower()
        _min_score = {"conservative": 45.0, "balanced": 40.0, "aggressive": 35.0}.get(_profile, 40.0)
        if score < _min_score and not context.dry_run:
            eligible = False
            reasons.append(f"Score {score:.1f} below threshold ({_profile})")
        if long_term_downtrend_no_proof and not context.dry_run:
            eligible = False
            reasons.append("Long-term downtrend; recommend only with reversal proof")

        # Deduplicate strings
        reasons = list(dict.fromkeys(reasons))
        risk_flags = list(dict.fromkeys(risk_flags))
        
        # Strategy Hint logic
        strategy_hint = decision.strategy_routing.strategy_mode or "smart_dca"
        confidence_score = int(clamp(decision.regime_detection.confidence * 100.0, 0.0, 100.0))
        
        # Formatting metrics
        metrics = {
            "symbol": context.symbol,
            "price": context.last_price,
            "score": score,
            "regime": decision.regime_detection.regime.value if hasattr(decision.regime_detection.regime, "value") else str(decision.regime_detection.regime),
            "confidence_score": confidence_score,
            "regime_scores": decision.regime_detection.scores or {},
            "strategy": strategy_hint,
            "allowed_action": decision.allowed_actions.value if hasattr(decision.allowed_actions, "value") else str(decision.allowed_actions),
            "position_size": float(decision.position_sizing.base_size),
            "risk_budget": decision.market_safety.risk_budget,
            "daily_realized_pnl": context.daily_realized_pnl,
            "portfolio_exposure_pct": context.portfolio_exposure_pct,
            "volatility": atr_pct if 'atr_pct' in locals() else 0.0,
            "market_type": "stocks" if len(context.symbol) < 6 and "/" not in context.symbol else "crypto", # Simple heuristic
            "weekly_trend": "UP" if (locals().get("ema50_1w") and context.last_price > locals().get("ema50_1w")) else "DOWN"
        }
        if (context.market_breadth or {}).get("is_stock"):
            metrics["sector"] = (context.market_breadth or {}).get("sector")
            metrics["liquidity_tier"] = (context.market_breadth or {}).get("liquidity_tier")
        for k, v in _ml_metrics.items():
            if v is not None:
                metrics[k] = v
        try:
            from alternative_data import calculate_alternative_data_score
            alt = calculate_alternative_data_score(context.symbol)
            if alt.sources_used:
                metrics["alt_data_score"] = alt.score
                metrics["alt_data_sources"] = ",".join(alt.sources_used)
                if alt.short_squeeze_candidate:
                    metrics["short_squeeze_candidate"] = True
                if alt.high_short_interest_risk:
                    metrics["high_short_interest"] = True
            # Short squeeze alert (stocks)
            if (context.market_breadth or {}).get("is_stock"):
                try:
                    from short_interest_monitor import short_squeeze_alert
                    alert = short_squeeze_alert(context.symbol, score)
                    if alert:
                        metrics["short_squeeze_alert"] = alert
                except Exception:
                    pass
        except Exception:
            pass
        
        # Construct Regime JSON for frontend
        regime_json = {
            "label": decision.regime_detection.regime.value if hasattr(decision.regime_detection.regime, "value") else str(decision.regime_detection.regime),
            "conf": decision.regime_detection.confidence,
            "scores": decision.regime_detection.scores or {}
        }

        # Adaptive scoring: apply calibrated weights from recommendation outcomes
        try:
            from adaptive_scorer import apply_adaptive_score
            regime_str = regime_json.get("label", "") or str(decision.regime_detection.regime)
            score = apply_adaptive_score(float(score), regime_str)
            metrics["score"] = score
        except Exception:
            pass

        return {
            "symbol": context.symbol,
            "horizon": horizon,
            "score": score,
            "eligible": eligible,
            "data_ok": decision.data_validity.data_ok,
            "research_only": False,
            "regime_json": json.dumps(regime_json), # Returning JSON string for worker_api to store
            "metrics_json": json.dumps(metrics),      # Returning JSON string
            "reasons_json": json.dumps(reasons[:5]),
            "risk_flags_json": json.dumps(risk_flags),
            "created_ts": int(time.time()),
            # Legacy fields for compat if needed, but we prefer the JSON fields above
            "regime": regime_json, 
            "metrics": metrics,
            "reasons": reasons,
            "risk_flags": risk_flags,
            "strategy": strategy_hint,
            "updated_ts": int(time.time()),
            "decision_debug": decision.debug, 
        }
    
    def _create_no_trade_decision(
        self,
        context: IntelligenceContext,
        data_validity: DataValidityResult,
        market_safety: Optional[MarketSafetyResult] = None,
        reason: str = "Trading not allowed",
    ) -> IntelligenceDecision:
        """Create a NO_TRADE decision with minimal processing."""
        if market_safety is None:
            market_safety = MarketSafetyResult(
                allowed_actions=AllowedAction.NO_TRADE,
                risk_budget=0.0,
                reasons=[reason],
            )
        
        # Still run regime detection for context
        regime_detection = self._detect_regime(context)
        
        # Still run trade management
        trade_management = self._manage_trades(context, regime_detection)
        
        return IntelligenceDecision(
            data_validity=data_validity,
            market_safety=market_safety,
            regime_detection=regime_detection,
            strategy_routing=StrategyRoutingResult(
                strategy_mode="none",
                entry_style="none",
                exit_style="none",
                reasons=["No trade allowed"],
            ),
            position_sizing=PositionSizingResult(
                base_size=0.0,
                ladder_steps=0,
                ladder_spacing_pct=0.0,
                max_adds=0,
            ),
            execution_policy=ExecutionPolicyResult(
                order_type="none",
                max_slippage_pct=0.0,
            ),
            trade_management=trade_management,
            allowed_actions=AllowedAction.NO_TRADE,
            final_action="NO_TRADE",
            final_reason=reason,
            timestamp=context.now_ts,
        )


# =========================
# UltimateIntelligenceLayer - Phase 4 Master Class
# =========================


class UltimateIntelligenceLayer(IntelligenceLayer):
    """
    Intelligence layer with ALL 13 Phase 4 features.
    Integrates: SentimentAnalyzer, KellyPositionSizer, PatternRecognizer,
    CorrelationTrader, SeasonalityAnalyzer, MLEnsemble, OptionsFlowAnalyzer,
    EarningsMomentumTrader, ZScoreTrader, MomentumRanker, RLTradingAgent (when enabled),
    OrderBookAnalyzer (high_frequency), AlternativeDataIntegrator.
    """

    def __init__(self) -> None:
        super().__init__()
        self._phase4_signals: Dict[str, Any] = {}

        # Phase 4A
        _ct = _PHASE4_IMPORTS.get("CorrelationTrader")
        self._correlation_trader = _ct() if _ct else None
        _sa = _PHASE4_IMPORTS.get("SeasonalityAnalyzer")
        self._seasonality_analyzer = _sa() if _sa else None

        # Phase 4B
        self._ml_ensemble = None
        get_ml = _PHASE4_IMPORTS.get("get_ml_ensemble")
        if get_ml and callable(get_ml):
            try:
                self._ml_ensemble = get_ml()
            except Exception:
                pass
        if self._ml_ensemble is None and _PHASE4_IMPORTS.get("MLEnsemble"):
            try:
                self._ml_ensemble = _PHASE4_IMPORTS["MLEnsemble"]()
            except Exception:
                pass

        self._options_analyzer = _PHASE4_IMPORTS.get("OptionsFlowAnalyzer")
        if self._options_analyzer:
            self._options_analyzer = self._options_analyzer()

        self._earnings_trader = _PHASE4_IMPORTS.get("EarningsMomentumTrader")
        if self._earnings_trader:
            self._earnings_trader = self._earnings_trader()

        self._zscore_trader = _PHASE4_IMPORTS.get("ZScoreTrader")
        if self._zscore_trader:
            self._zscore_trader = self._zscore_trader()

        self._momentum_ranker = _PHASE4_IMPORTS.get("MomentumRanker")
        if self._momentum_ranker:
            self._momentum_ranker = self._momentum_ranker()

        # Phase 4C
        self._rl_agent = None
        if _PHASE4_IMPORTS.get("ENABLE_RL_AGENT") and _PHASE4_IMPORTS.get("RLTradingAgent"):
            self._rl_agent = _PHASE4_IMPORTS["RLTradingAgent"]()

        self._hf_orderbook = _PHASE4_IMPORTS.get("HFOrderBookAnalyzer")
        if self._hf_orderbook:
            self._hf_orderbook = self._hf_orderbook()

        self._alt_data = _PHASE4_IMPORTS.get("AlternativeDataIntegrator")
        if self._alt_data:
            self._alt_data = self._alt_data()

        logger.info(
            "UltimateIntelligenceLayer initialized: corr=%s season=%s ml=%s options=%s "
            "earnings=%s zscore=%s mom=%s rl=%s hf=%s alt=%s",
            self._correlation_trader is not None,
            self._seasonality_analyzer is not None,
            self._ml_ensemble is not None,
            self._options_analyzer is not None,
            self._earnings_trader is not None,
            self._zscore_trader is not None,
            self._momentum_ranker is not None,
            self._rl_agent is not None,
            self._hf_orderbook is not None,
            self._alt_data is not None,
        )

    def evaluate_with_ultimate_intelligence(self, context: IntelligenceContext) -> IntelligenceDecision:
        """
        Full evaluation with ALL Phase 4 intelligence layers.
        Runs base evaluate() then augments with Phase 4 signals.
        """
        decision = self.evaluate(context)
        phase4_debug: Dict[str, Any] = {}

        # Helper: market type
        is_stock = (context.market_breadth or {}).get("is_stock") or (
            len(context.symbol) < 6 and "/" not in context.symbol
        )
        market_type = "stock" if is_stock else "crypto"

        # 1. Correlation Trader
        if self._correlation_trader:
            try:
                opps = self._correlation_trader.find_opportunities()
                for o in opps:
                    if o.get("symbol") == context.symbol:
                        phase4_debug["correlation"] = o
                        decision.strategy_routing.reasons.append(f"Correlation: {o.get('reasoning', '')[:60]}")
                        break
            except Exception as e:
                logger.debug("CorrelationTrader: %s", e)

        # 2. Seasonality
        if self._seasonality_analyzer:
            try:
                seasonal = self._seasonality_analyzer.get_seasonal_bias(context.symbol, market_type)
                phase4_debug["seasonality"] = seasonal
                if seasonal.get("strength", 0) > 0.3:
                    decision.strategy_routing.reasons.append(
                        f"Seasonality: {seasonal.get('reasoning', '')[:60]}"
                    )
            except Exception as e:
                logger.debug("SeasonalityAnalyzer: %s", e)

        # 3. ML Ensemble
        if self._ml_ensemble:
            try:
                candles = context.candles_1d or context.candles_4h or context.candles_1h
                if candles and len(candles) >= 50:
                    atr_val = _atr(candles, 14)
                    vol = (atr_val / context.last_price) if atr_val and context.last_price else 0.02
                    pred = self._ml_ensemble.predict(candles, current_volatility=vol)
                    phase4_debug["ml_ensemble"] = {
                        "direction": pred.direction,
                        "confidence": pred.confidence,
                        "prob_up": pred.probability_up,
                        "prob_down": pred.probability_down,
                    }
                    if pred.should_trade(0.55):
                        decision.strategy_routing.reasons.append(
                            f"ML ensemble: {pred.direction} ({pred.confidence:.0%})"
                        )
            except Exception as e:
                logger.debug("MLEnsemble: %s", e)

        # 4. Options Flow (stocks)
        if self._options_analyzer and is_stock:
            try:
                opts = self._options_analyzer.get_options_signals(context.symbol)
                phase4_debug["options_flow"] = opts
                if opts.get("signal") != "neutral":
                    decision.strategy_routing.reasons.append(
                        f"Options: {opts.get('signal')} - {opts.get('reasoning', '')[:40]}"
                    )
            except Exception as e:
                logger.debug("OptionsFlowAnalyzer: %s", e)

        # 5. Earnings Momentum (stocks)
        if self._earnings_trader and is_stock:
            try:
                earn = self._earnings_trader.get_earnings_signal(context.symbol)
                phase4_debug["earnings_momentum"] = earn
                if earn.get("confidence", 0) > 0.6 and earn.get("signal") != "neutral":
                    decision.strategy_routing.reasons.append(
                        f"Earnings drift: {earn.get('signal')} ({earn.get('reasoning', '')[:40]})"
                    )
            except Exception as e:
                logger.debug("EarningsMomentumTrader: %s", e)

        # 6. Z-Score (mean reversion)
        if self._zscore_trader:
            try:
                candles = context.candles_1d or context.candles_4h or context.candles_1h
                if candles and len(candles) >= 25:
                    zsig = self._zscore_trader.get_zscore_signal(candles, context.symbol)
                    phase4_debug["zscore"] = zsig
                    if zsig.get("confidence", 0) > 0.6 and zsig.get("signal") != "neutral":
                        decision.strategy_routing.reasons.append(
                            f"Z-score: {zsig.get('signal')} {zsig.get('reasoning', '')[:40]}"
                        )
            except Exception as e:
                logger.debug("ZScoreTrader: %s", e)

        # 7. Momentum Ranker (for universe context)
        if self._momentum_ranker:
            try:
                mom = self._momentum_ranker.calculate_momentum_score(context.symbol)
                phase4_debug["momentum"] = mom
                if mom.get("score", 50) > 70:
                    decision.strategy_routing.reasons.append(
                        f"Momentum score {mom.get('score')} (strong)"
                    )
                elif mom.get("score", 50) < 30:
                    decision.strategy_routing.reasons.append(
                        f"Momentum score {mom.get('score')} (weak)"
                    )
            except Exception as e:
                logger.debug("MomentumRanker: %s", e)

        # 8. RL Agent (when enabled)
        if self._rl_agent:
            try:
                candles = context.candles_1d or context.candles_4h or context.candles_1h
                if candles:
                    rl_sig = self._rl_agent.get_signal_from_candles(candles)
                    phase4_debug["rl_agent"] = rl_sig
                    if rl_sig.get("action") != "hold" and rl_sig.get("confidence", 0) > 0.6:
                        decision.strategy_routing.reasons.append(
                            f"RL: {rl_sig.get('action')} (conf {rl_sig.get('confidence', 0):.0%})"
                        )
            except Exception as e:
                logger.debug("RLTradingAgent: %s", e)

        # 9. High-Frequency Order Book
        if self._hf_orderbook:
            try:
                ob_sig = self._hf_orderbook.analyze_order_book(context.symbol)
                phase4_debug["order_book_microstructure"] = ob_sig
                if ob_sig.get("signal") != "neutral" and ob_sig.get("confidence", 0) > 0.6:
                    decision.strategy_routing.reasons.append(
                        f"OB imbalance: {ob_sig.get('signal')}"
                    )
            except Exception as e:
                logger.debug("HFOrderBookAnalyzer: %s", e)

        # 10. Alternative Data
        if self._alt_data:
            try:
                alt = self._alt_data.get_alternative_data_signal(context.symbol)
                phase4_debug["alternative_data"] = {
                    "score": alt.score,
                    "confidence": alt.confidence,
                    "sources": alt.sources_used,
                    "short_squeeze": alt.short_squeeze_candidate,
                }
                if alt.confidence > 0.5 and alt.score > 60:
                    decision.strategy_routing.reasons.append(
                        f"Alt data score {alt.score:.0f} ({','.join(alt.sources_used[:2])})"
                    )
                if alt.short_squeeze_candidate:
                    decision.strategy_routing.reasons.append("Short squeeze candidate")
            except Exception as e:
                logger.debug("AlternativeDataIntegrator: %s", e)

        # Sentiment, Kelly, Pattern Recognition are already integrated in base IntelligenceLayer
        # (sentiment in market_safety, kelly in position_sizing, pattern in generate_recommendation)
        # Attach phase4 debug to decision
        decision.debug["phase4_signals"] = phase4_debug

        return decision
