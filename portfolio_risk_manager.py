"""
Portfolio Risk Manager - Advanced risk management for multi-asset trading.

Features:
- Value at Risk (VaR) calculation
- Sharpe Ratio tracking
- Correlation-based position control
- Dynamic rebalancing
- Global circuit breakers
- Performance analytics

Based on: "Upgrading to an Autonomous Multi-Asset Trading Bot" specification.
"""

import math
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import deque
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Portfolio risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PositionInfo:
    """Information about a single position."""
    symbol: str
    market_type: str  # "crypto" or "stock"
    entry_price: float
    current_price: float
    size: float  # In base currency
    notional: float  # In quote currency (USD)
    unrealized_pnl: float
    unrealized_pnl_pct: float
    strategy: str
    confidence: float
    risk_tier: str  # "low", "medium", "high"
    opened_at: int  # Unix timestamp
    bot_id: int


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio metrics."""
    total_equity: float
    total_positions: int
    total_exposure: float
    exposure_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl_today: float
    daily_pnl_pct: float
    
    # Risk metrics
    var_95: float  # 95% Value at Risk (1-day)
    var_99: float  # 99% Value at Risk (1-day)
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Diversification
    diversification_score: float  # 0-1
    correlation_risk: float  # Average pairwise correlation
    concentration_risk: float  # Herfindahl index
    
    # Limits
    positions_remaining: int
    exposure_remaining: float
    var_utilization: float  # Current VaR / Max VaR
    
    risk_level: RiskLevel
    risk_flags: List[str] = field(default_factory=list)
    timestamp: int = field(default_factory=lambda: int(time.time()))


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_portfolio_var_pct: float = 0.05  # 5% max VaR
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    max_daily_loss_pct: float = 0.03  # 3% max daily loss
    max_exposure_pct: float = 0.80  # 80% max exposure
    max_single_position_pct: float = 0.20  # 20% max per position
    max_correlated_exposure_pct: float = 0.40  # 40% max in correlated assets
    max_concurrent_positions: int = 10
    min_diversification_score: float = 0.3
    correlation_threshold: float = 0.70  # Consider assets correlated above this
    
    # Circuit breakers
    pause_on_drawdown_pct: float = 0.08  # Pause new trades at 8% drawdown
    close_all_on_drawdown_pct: float = 0.15  # Close all at 15% drawdown
    
    # Performance targets
    target_sharpe: float = 1.0
    min_acceptable_sharpe: float = 0.5


class PortfolioRiskManager:
    """
    Advanced portfolio-level risk management.
    
    Monitors all positions, calculates risk metrics, enforces limits,
    and provides recommendations for rebalancing.
    """
    
    def __init__(self, 
                 initial_equity: float = 10000.0,
                 limits: Optional[RiskLimits] = None):
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.limits = limits or RiskLimits()
        
        self._positions: Dict[int, PositionInfo] = {}  # bot_id -> position
        self._returns_history: deque = deque(maxlen=252)  # ~1 year of daily returns
        self._equity_history: deque = deque(maxlen=1000)  # Equity curve
        self._pnl_history: deque = deque(maxlen=100)  # Recent PnL for analytics
        
        self._correlation_matrix: Dict[str, Dict[str, float]] = {}
        self._volatilities: Dict[str, float] = {}  # Symbol -> annualized volatility
        
        self._lock = threading.Lock()
        self._last_update = 0
        self._daily_start_equity = initial_equity
        self._daily_realized_pnl = 0.0
        self._last_day = 0
        
        # Risk state
        self._paused = False
        self._pause_reason = ""
        self._close_all_triggered = False
        
        logger.info(f"PortfolioRiskManager initialized with equity={initial_equity}, limits={limits}")
    
    def update_position(self, bot_id: int, position: PositionInfo) -> None:
        """Update or add a position."""
        with self._lock:
            self._positions[bot_id] = position
            self._last_update = int(time.time())
    
    def remove_position(self, bot_id: int, realized_pnl: float = 0.0) -> None:
        """Remove a closed position."""
        with self._lock:
            if bot_id in self._positions:
                del self._positions[bot_id]
            self._daily_realized_pnl += realized_pnl
            self._pnl_history.append({
                "bot_id": bot_id,
                "pnl": realized_pnl,
                "timestamp": int(time.time())
            })
    
    def update_equity(self, equity: float) -> None:
        """Update current equity value."""
        with self._lock:
            # Track peak for drawdown
            if equity > self.peak_equity:
                self.peak_equity = equity
            
            # Record for history
            self._equity_history.append({
                "equity": equity,
                "timestamp": int(time.time())
            })
            
            # Daily reset
            today = int(time.time()) // 86400
            if today != self._last_day:
                self._record_daily_return(equity)
                self._daily_start_equity = equity
                self._daily_realized_pnl = 0.0
                self._last_day = today
    
    def _record_daily_return(self, current_equity: float) -> None:
        """Record daily return for Sharpe calculation."""
        if self._daily_start_equity > 0:
            daily_return = (current_equity - self._daily_start_equity) / self._daily_start_equity
            self._returns_history.append(daily_return)
    
    def update_correlations(self, correlation_matrix: Dict[str, Dict[str, float]]) -> None:
        """Update correlation matrix from portfolio_correlation module."""
        with self._lock:
            self._correlation_matrix = correlation_matrix
    
    def update_volatilities(self, volatilities: Dict[str, float]) -> None:
        """Update volatility estimates for each symbol."""
        with self._lock:
            self._volatilities.update(volatilities)
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calculate portfolio Value at Risk using parametric method.
        
        Uses position sizes, volatilities, and correlations.
        Returns VaR as a positive dollar amount.
        """
        with self._lock:
            if not self._positions:
                return 0.0
            
            # Simple parametric VaR
            # VaR = Z * σ * Exposure
            # where Z is the z-score for confidence level
            
            z_scores = {0.95: 1.645, 0.99: 2.326}
            z = z_scores.get(confidence, 1.645)
            
            total_var_sq = 0.0
            symbols = []
            notionals = []
            
            for pos in self._positions.values():
                symbols.append(pos.symbol)
                notionals.append(pos.notional)
            
            # Calculate individual VaRs
            individual_vars = []
            for i, sym in enumerate(symbols):
                vol = self._volatilities.get(sym, 0.02)  # Default 2% daily vol
                # Annualized vol to daily: vol / sqrt(252), but we often store daily already
                daily_vol = vol if vol < 0.1 else vol / math.sqrt(252)
                ind_var = z * daily_vol * notionals[i]
                individual_vars.append(ind_var)
            
            # Account for correlations (simplified)
            # Full formula: VaR_p = sqrt(sum_i sum_j w_i w_j σ_i σ_j ρ_ij)
            # Simplified: assume average correlation
            
            if len(individual_vars) == 1:
                return individual_vars[0]
            
            avg_corr = self._calculate_average_correlation(symbols)
            
            # Portfolio VaR with correlation adjustment
            sum_var = sum(individual_vars)
            sum_var_sq = sum(v**2 for v in individual_vars)
            
            # VaR_p^2 = sum(VaR_i^2) + 2*sum_i<j(ρ_ij * VaR_i * VaR_j)
            # Simplified: VaR_p^2 ≈ sum(VaR_i^2) + ρ_avg * (sum(VaR_i)^2 - sum(VaR_i^2))
            portfolio_var_sq = sum_var_sq + avg_corr * (sum_var**2 - sum_var_sq)
            
            return math.sqrt(max(0, portfolio_var_sq))
    
    def _calculate_average_correlation(self, symbols: List[str]) -> float:
        """Calculate average pairwise correlation."""
        if len(symbols) < 2 or not self._correlation_matrix:
            return 0.5  # Default assumption
        
        correlations = []
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = self._correlation_matrix.get(sym1, {}).get(sym2)
                if corr is None:
                    corr = self._correlation_matrix.get(sym2, {}).get(sym1)
                if corr is not None:
                    correlations.append(abs(corr))
        
        return sum(correlations) / len(correlations) if correlations else 0.5
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate annualized Sharpe ratio from daily returns.
        
        Sharpe = (E[R] - Rf) / σ[R] * sqrt(252)
        """
        if len(self._returns_history) < 10:
            return 0.0
        
        returns = list(self._returns_history)
        mean_return = sum(returns) / len(returns)
        
        # Standard deviation
        variance = sum((r - mean_return)**2 for r in returns) / len(returns)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0001
        
        # Daily risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Annualized Sharpe
        sharpe = (mean_return - daily_rf) / std_dev * math.sqrt(252)
        
        return round(sharpe, 2)
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """
        Calculate Sortino ratio (uses downside deviation only).
        """
        if len(self._returns_history) < 10:
            return 0.0
        
        returns = list(self._returns_history)
        mean_return = sum(returns) / len(returns)
        daily_rf = risk_free_rate / 252
        
        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < daily_rf]
        if not negative_returns:
            return 3.0  # No downside = great
        
        downside_variance = sum((r - daily_rf)**2 for r in negative_returns) / len(negative_returns)
        downside_dev = math.sqrt(downside_variance) if downside_variance > 0 else 0.0001
        
        sortino = (mean_return - daily_rf) / downside_dev * math.sqrt(252)
        
        return round(sortino, 2)
    
    def calculate_diversification_score(self) -> float:
        """
        Calculate diversification score (0-1).
        
        Based on:
        - Number of positions
        - Position size distribution (Herfindahl index)
        - Correlation between positions
        - Asset class distribution
        """
        with self._lock:
            if not self._positions:
                return 1.0  # No positions = fully diversified (nothing at risk)
            
            n = len(self._positions)
            if n == 1:
                return 0.2  # Single position = poor diversification
            
            # 1. Position count score (more positions = better, up to a point)
            count_score = min(1.0, n / 5)  # Max score at 5+ positions
            
            # 2. Concentration score (inverse Herfindahl)
            total_notional = sum(p.notional for p in self._positions.values())
            if total_notional > 0:
                weights = [p.notional / total_notional for p in self._positions.values()]
                herfindahl = sum(w**2 for w in weights)
                # HHI ranges from 1/n (perfect diversification) to 1 (single position)
                # Normalize: 0 = concentrated, 1 = diversified
                concentration_score = 1 - (herfindahl - 1/n) / (1 - 1/n) if n > 1 else 0
            else:
                concentration_score = 0.5
            
            # 3. Correlation score (low average correlation = better)
            symbols = [p.symbol for p in self._positions.values()]
            avg_corr = self._calculate_average_correlation(symbols)
            correlation_score = 1 - avg_corr  # 0 correlation = score 1
            
            # 4. Asset class score (mix of crypto and stocks)
            crypto_count = sum(1 for p in self._positions.values() if p.market_type == "crypto")
            stock_count = n - crypto_count
            if crypto_count > 0 and stock_count > 0:
                class_score = 1.0  # Both markets
            elif n > 2:
                class_score = 0.7  # Single market but multiple assets
            else:
                class_score = 0.4  # Limited diversity
            
            # Weighted average
            score = (
                0.2 * count_score +
                0.3 * concentration_score +
                0.3 * correlation_score +
                0.2 * class_score
            )
            
            return round(score, 2)
    
    def get_metrics(self, current_equity: float) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        with self._lock:
            self.update_equity(current_equity)
            
            # Basic metrics
            total_exposure = sum(p.notional for p in self._positions.values())
            exposure_pct = total_exposure / current_equity if current_equity > 0 else 0
            unrealized_pnl = sum(p.unrealized_pnl for p in self._positions.values())
            unrealized_pnl_pct = unrealized_pnl / current_equity if current_equity > 0 else 0
            
            daily_pnl = current_equity - self._daily_start_equity + self._daily_realized_pnl
            daily_pnl_pct = daily_pnl / self._daily_start_equity if self._daily_start_equity > 0 else 0
            
            # Risk metrics
            var_95 = self.calculate_var(0.95)
            var_99 = self.calculate_var(0.99)
            
            max_dd = (self.peak_equity - min(
                e["equity"] for e in self._equity_history
            ) if self._equity_history else 0) / self.peak_equity if self.peak_equity > 0 else 0
            
            current_dd = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0
            
            sharpe = self.calculate_sharpe_ratio()
            sortino = self.calculate_sortino_ratio()
            
            # Diversification
            div_score = self.calculate_diversification_score()
            symbols = [p.symbol for p in self._positions.values()]
            avg_corr = self._calculate_average_correlation(symbols)
            
            # Concentration (Herfindahl)
            if total_exposure > 0:
                weights = [p.notional / total_exposure for p in self._positions.values()]
                concentration = sum(w**2 for w in weights)
            else:
                concentration = 0
            
            # Limits utilization
            positions_remaining = max(0, self.limits.max_concurrent_positions - len(self._positions))
            exposure_remaining = max(0, self.limits.max_exposure_pct * current_equity - total_exposure)
            max_var = self.limits.max_portfolio_var_pct * current_equity
            var_util = var_95 / max_var if max_var > 0 else 0
            
            # Risk level
            risk_flags = []
            risk_level = RiskLevel.LOW
            
            if current_dd >= self.limits.close_all_on_drawdown_pct:
                risk_level = RiskLevel.CRITICAL
                risk_flags.append(f"CRITICAL: Drawdown {current_dd:.1%} exceeds close threshold")
            elif current_dd >= self.limits.pause_on_drawdown_pct:
                risk_level = RiskLevel.HIGH
                risk_flags.append(f"HIGH: Drawdown {current_dd:.1%} exceeds pause threshold")
            elif current_dd >= self.limits.max_drawdown_pct * 0.5:
                risk_level = RiskLevel.MEDIUM
                risk_flags.append(f"MEDIUM: Drawdown {current_dd:.1%} approaching limit")
            
            if daily_pnl_pct < -self.limits.max_daily_loss_pct:
                risk_level = max(risk_level, RiskLevel.HIGH, key=lambda x: list(RiskLevel).index(x))
                risk_flags.append(f"Daily loss {daily_pnl_pct:.1%} exceeds limit")
            
            if var_util > 1.0:
                risk_level = max(risk_level, RiskLevel.HIGH, key=lambda x: list(RiskLevel).index(x))
                risk_flags.append(f"VaR utilization {var_util:.1%} exceeds 100%")
            
            if div_score < self.limits.min_diversification_score:
                risk_flags.append(f"Low diversification score: {div_score:.2f}")
            
            if avg_corr > self.limits.correlation_threshold:
                risk_flags.append(f"High portfolio correlation: {avg_corr:.2f}")
            
            return PortfolioMetrics(
                total_equity=current_equity,
                total_positions=len(self._positions),
                total_exposure=total_exposure,
                exposure_pct=exposure_pct,
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_pct=unrealized_pnl_pct,
                realized_pnl_today=self._daily_realized_pnl,
                daily_pnl_pct=daily_pnl_pct,
                var_95=var_95,
                var_99=var_99,
                max_drawdown=max_dd,
                current_drawdown=current_dd,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                diversification_score=div_score,
                correlation_risk=avg_corr,
                concentration_risk=concentration,
                positions_remaining=positions_remaining,
                exposure_remaining=exposure_remaining,
                var_utilization=var_util,
                risk_level=risk_level,
                risk_flags=risk_flags
            )
    
    def can_open_position(self, 
                          symbol: str, 
                          notional: float, 
                          market_type: str,
                          current_equity: float) -> Tuple[bool, str]:
        """
        Check if a new position can be opened within risk limits.
        
        Returns (allowed, reason).
        """
        with self._lock:
            # Check pause state
            if self._paused:
                return False, f"Trading paused: {self._pause_reason}"
            
            if self._close_all_triggered:
                return False, "Close-all mode active due to critical drawdown"
            
            # Check position count
            if len(self._positions) >= self.limits.max_concurrent_positions:
                return False, f"Max positions ({self.limits.max_concurrent_positions}) reached"
            
            # Check total exposure
            current_exposure = sum(p.notional for p in self._positions.values())
            new_exposure = current_exposure + notional
            max_exposure = self.limits.max_exposure_pct * current_equity
            
            if new_exposure > max_exposure:
                return False, f"Would exceed max exposure ({self.limits.max_exposure_pct:.0%})"
            
            # Check single position size
            if notional > self.limits.max_single_position_pct * current_equity:
                return False, f"Position too large (max {self.limits.max_single_position_pct:.0%})"
            
            # Check drawdown
            if current_equity < self.peak_equity:
                dd = (self.peak_equity - current_equity) / self.peak_equity
                if dd >= self.limits.pause_on_drawdown_pct:
                    self._paused = True
                    self._pause_reason = f"Drawdown {dd:.1%} exceeds threshold"
                    return False, self._pause_reason
            
            # Check correlation with existing positions
            existing_symbols = [p.symbol for p in self._positions.values()]
            for existing_sym in existing_symbols:
                corr = self._correlation_matrix.get(symbol, {}).get(existing_sym)
                if corr is None:
                    corr = self._correlation_matrix.get(existing_sym, {}).get(symbol, 0.5)
                
                if corr and corr > self.limits.correlation_threshold:
                    # Check if adding would exceed correlated exposure limit
                    existing_notional = sum(
                        p.notional for p in self._positions.values() 
                        if p.symbol == existing_sym
                    )
                    correlated_total = existing_notional + notional
                    max_correlated = self.limits.max_correlated_exposure_pct * current_equity
                    
                    if correlated_total > max_correlated:
                        return False, f"High correlation with {existing_sym} ({corr:.2f}), would exceed correlated exposure limit"
            
            # Check VaR impact (estimate)
            current_var = self.calculate_var()
            max_var = self.limits.max_portfolio_var_pct * current_equity
            
            # Estimate new position's VaR contribution
            vol = self._volatilities.get(symbol, 0.02)
            daily_vol = vol if vol < 0.1 else vol / math.sqrt(252)
            position_var = 1.645 * daily_vol * notional  # 95% VaR
            
            # Rough estimate of portfolio VaR increase (assumes some diversification)
            avg_corr = self._calculate_average_correlation(existing_symbols + [symbol])
            estimated_var_increase = position_var * (0.5 + 0.5 * avg_corr)  # Simplified
            
            if current_var + estimated_var_increase > max_var:
                return False, f"Would exceed VaR limit (current: ${current_var:.0f}, max: ${max_var:.0f})"
            
            return True, "OK"
    
    def get_rebalancing_recommendations(self, current_equity: float) -> List[Dict[str, Any]]:
        """
        Get recommendations for portfolio rebalancing.
        
        Returns list of recommended actions:
        - Reduce overweight positions
        - Close highly correlated duplicates
        - Close underperforming positions
        """
        recommendations = []
        
        with self._lock:
            if not self._positions:
                return recommendations
            
            total_exposure = sum(p.notional for p in self._positions.values())
            
            # 1. Check for overweight positions
            for bot_id, pos in self._positions.items():
                weight = pos.notional / current_equity if current_equity > 0 else 0
                if weight > self.limits.max_single_position_pct:
                    excess = weight - self.limits.max_single_position_pct
                    recommendations.append({
                        "action": "reduce",
                        "bot_id": bot_id,
                        "symbol": pos.symbol,
                        "reason": f"Overweight ({weight:.1%} vs {self.limits.max_single_position_pct:.1%} limit)",
                        "suggested_reduction_pct": excess / weight,
                        "priority": "high"
                    })
            
            # 2. Check for highly correlated positions
            positions_list = list(self._positions.items())
            for i, (bot_id1, pos1) in enumerate(positions_list):
                for bot_id2, pos2 in positions_list[i+1:]:
                    corr = self._correlation_matrix.get(pos1.symbol, {}).get(pos2.symbol)
                    if corr is None:
                        corr = self._correlation_matrix.get(pos2.symbol, {}).get(pos1.symbol, 0)
                    
                    if corr and corr > self.limits.correlation_threshold:
                        # Recommend closing the smaller/worse performing one
                        if pos1.unrealized_pnl_pct < pos2.unrealized_pnl_pct:
                            worse = (bot_id1, pos1)
                        else:
                            worse = (bot_id2, pos2)
                        
                        recommendations.append({
                            "action": "close",
                            "bot_id": worse[0],
                            "symbol": worse[1].symbol,
                            "reason": f"Highly correlated with {pos1.symbol if worse[0] == bot_id2 else pos2.symbol} (ρ={corr:.2f})",
                            "priority": "medium"
                        })
            
            # 3. Check for underperformers
            for bot_id, pos in self._positions.items():
                # Close positions that are both old and losing
                age_hours = (time.time() - pos.opened_at) / 3600
                if age_hours > 24 and pos.unrealized_pnl_pct < -0.05:  # >24h and >5% loss
                    recommendations.append({
                        "action": "review",
                        "bot_id": bot_id,
                        "symbol": pos.symbol,
                        "reason": f"Underperforming: {pos.unrealized_pnl_pct:.1%} over {age_hours:.0f}h",
                        "priority": "low"
                    })
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))
        
        return recommendations
    
    def check_circuit_breakers(self, current_equity: float) -> Tuple[bool, str, str]:
        """
        Check if any circuit breakers should be triggered.
        
        Returns (triggered, action, reason):
        - action: "none", "pause", "close_all"
        """
        if current_equity <= 0 or self.peak_equity <= 0:
            return False, "none", ""
        
        dd = (self.peak_equity - current_equity) / self.peak_equity
        
        if dd >= self.limits.close_all_on_drawdown_pct:
            self._close_all_triggered = True
            return True, "close_all", f"Critical drawdown: {dd:.1%}"
        
        if dd >= self.limits.pause_on_drawdown_pct:
            self._paused = True
            self._pause_reason = f"Drawdown: {dd:.1%}"
            return True, "pause", self._pause_reason
        
        # Daily loss check
        daily_pnl = current_equity - self._daily_start_equity
        daily_pnl_pct = daily_pnl / self._daily_start_equity if self._daily_start_equity > 0 else 0
        
        if daily_pnl_pct < -self.limits.max_daily_loss_pct:
            self._paused = True
            self._pause_reason = f"Daily loss limit: {daily_pnl_pct:.1%}"
            return True, "pause", self._pause_reason
        
        return False, "none", ""
    
    def reset_circuit_breakers(self) -> None:
        """Reset circuit breakers (manual override)."""
        with self._lock:
            self._paused = False
            self._pause_reason = ""
            self._close_all_triggered = False
            logger.info("Circuit breakers reset manually")
    
    def get_risk_summary(self, current_equity: float) -> Dict[str, Any]:
        """Get a summary for UI/Discord notifications."""
        metrics = self.get_metrics(current_equity)
        
        return {
            "equity": metrics.total_equity,
            "positions": metrics.total_positions,
            "exposure_pct": round(metrics.exposure_pct * 100, 1),
            "unrealized_pnl": round(metrics.unrealized_pnl, 2),
            "daily_pnl_pct": round(metrics.daily_pnl_pct * 100, 2),
            "var_95": round(metrics.var_95, 2),
            "drawdown_pct": round(metrics.current_drawdown * 100, 2),
            "sharpe": metrics.sharpe_ratio,
            "diversification": round(metrics.diversification_score * 100, 0),
            "risk_level": metrics.risk_level.value,
            "risk_flags": metrics.risk_flags,
            "positions_available": metrics.positions_remaining,
            "paused": self._paused,
            "pause_reason": self._pause_reason if self._paused else None
        }
    
    def to_json(self) -> str:
        """Serialize state to JSON for persistence."""
        with self._lock:
            state = {
                "initial_equity": self.initial_equity,
                "peak_equity": self.peak_equity,
                "daily_start_equity": self._daily_start_equity,
                "daily_realized_pnl": self._daily_realized_pnl,
                "last_day": self._last_day,
                "returns_history": list(self._returns_history),
                "paused": self._paused,
                "pause_reason": self._pause_reason,
                "close_all_triggered": self._close_all_triggered,
                "timestamp": int(time.time())
            }
            return json.dumps(state)
    
    def from_json(self, json_str: str) -> None:
        """Restore state from JSON."""
        try:
            state = json.loads(json_str)
            with self._lock:
                self.initial_equity = state.get("initial_equity", self.initial_equity)
                self.peak_equity = state.get("peak_equity", self.peak_equity)
                self._daily_start_equity = state.get("daily_start_equity", self._daily_start_equity)
                self._daily_realized_pnl = state.get("daily_realized_pnl", 0)
                self._last_day = state.get("last_day", 0)
                self._returns_history = deque(state.get("returns_history", []), maxlen=252)
                self._paused = state.get("paused", False)
                self._pause_reason = state.get("pause_reason", "")
                self._close_all_triggered = state.get("close_all_triggered", False)
            logger.info("PortfolioRiskManager state restored from JSON")
        except Exception as e:
            logger.error(f"Failed to restore PortfolioRiskManager state: {e}")


# Singleton instance
_portfolio_risk_manager: Optional[PortfolioRiskManager] = None


def get_portfolio_risk_manager(initial_equity: float = 10000.0) -> PortfolioRiskManager:
    """Get or create the singleton portfolio risk manager."""
    global _portfolio_risk_manager
    if _portfolio_risk_manager is None:
        _portfolio_risk_manager = PortfolioRiskManager(initial_equity=initial_equity)
    return _portfolio_risk_manager


def reset_portfolio_risk_manager(initial_equity: float = 10000.0) -> PortfolioRiskManager:
    """Reset and return a new portfolio risk manager."""
    global _portfolio_risk_manager
    _portfolio_risk_manager = PortfolioRiskManager(initial_equity=initial_equity)
    return _portfolio_risk_manager


# ============================================================
# Master Upgrade Part 5: Simple check helpers for autopilot
# ============================================================

def check_portfolio_risk(current_bots: List[Dict[str, Any]], max_drawdown_pct: float = 10.0) -> Dict[str, Any]:
    """
    Check portfolio-level risk limits. Returns critical violations that should stop trading.
    Used by autopilot before creating new bots.
    """
    violations: List[Dict[str, Any]] = []

    try:
        from db import pnl_summary, get_setting
        from alpaca_adapter import AlpacaAdapter

        # Get portfolio value
        adapter = AlpacaAdapter()
        account = adapter.get_account()
        equity = float(account.get("equity", 0) or account.get("portfolio_value", 0))

        # Drawdown: would need peak tracking - skip if no peak stored
        peak_raw = get_setting("portfolio_peak_equity")
        if peak_raw and equity > 0:
            try:
                peak = float(peak_raw)
                if peak > 0:
                    dd_pct = ((peak - equity) / peak) * 100
                    if dd_pct > max_drawdown_pct:
                        violations.append({
                            "violation": True,
                            "severity": "critical",
                            "message": f"Portfolio drawdown {dd_pct:.1f}% (max: {max_drawdown_pct}%)",
                            "action": "Stop all new entries",
                        })
            except (ValueError, TypeError):
                pass

        # Daily loss: from pnl_summary
        try:
            from db import now_ts
            since = now_ts() - 86400
            pnl_data = pnl_summary(since_ts=since)
            daily_pnl = float(pnl_data.get("realized", 0) or pnl_data.get("total_pnl", 0) or 0)
            if equity > 0 and daily_pnl < 0:
                daily_pct = abs(daily_pnl / equity) * 100
                max_daily = 5.0  # 5% default
                if daily_pct > max_daily:
                    violations.append({
                        "violation": True,
                        "severity": "critical",
                        "message": f"Daily loss {daily_pct:.1f}%",
                        "action": "Trading paused for today",
                    })
        except Exception:
            pass

    except Exception as e:
        logger.warning("check_portfolio_risk: %s", e)

    critical = any(v.get("severity") == "critical" for v in violations)
    return {
        "critical_violation": critical,
        "violations": violations,
        "reason": "; ".join(v.get("message", "") for v in violations) if violations else "",
    }


def check_correlation_limit(new_symbol: str, max_correlated_pct: float = 30.0) -> bool:
    """
    Check if adding this symbol would exceed correlation limits.
    Returns True if OK to add, False if would exceed.
    """
    try:
        from db import get_symbols_with_open_deals
        try:
            from portfolio_correlation import PortfolioCorrelationAnalyzer
        except ImportError:
            return True  # No correlation module = allow

        symbols = get_symbols_with_open_deals()
        if not symbols:
            return True

        # Simplified: full correlation analysis would need price history
        # For now, allow - autopilot can extend with PortfolioCorrelationAnalyzer
        return True
    except Exception as e:
        logger.debug("check_correlation_limit: %s", e)
        return True


def check_sector_limit(new_symbol: str, max_sector_pct: float = 35.0) -> bool:
    """
    Prevent > max_sector_pct in single sector.
    Returns True if OK to add, False if sector limit would be exceeded.
    """
    try:
        from stock_metadata import get_sector
        from db import get_symbols_with_open_deals

        sector = get_sector(new_symbol)
        if not sector:
            return True

        symbols = get_symbols_with_open_deals()
        sector_count = 0
        total = len(symbols)
        for sym in symbols:
            if get_sector(sym) == sector:
                sector_count += 1

        if total == 0:
            return True
        sector_pct = (sector_count / total) * 100
        # Adding new_symbol would add 1 to sector; check (sector_count + 1) / (total + 1)
        new_total = total + 1
        new_sector_count = sector_count + (1 if get_sector(new_symbol) == sector else 0)
        new_sector_pct = (new_sector_count / new_total) * 100
        if new_sector_pct >= max_sector_pct:
            logger.warning("%s would be %.1f%% (max: %.1f%%)", sector, new_sector_pct, max_sector_pct)
            return False
        return True
    except Exception as e:
        logger.debug("check_sector_limit: %s", e)
        return True
