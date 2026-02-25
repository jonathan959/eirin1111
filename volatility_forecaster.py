"""
GARCH volatility forecasting (Phase 2 Advanced Intelligence).
Predicts future volatility to adjust position sizes proactively.
"""
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _to_returns_series(candles: Any) -> pd.Series:
    if isinstance(candles, pd.DataFrame):
        close = candles["close"] if "close" in candles.columns else candles.iloc[:, 4]
    elif isinstance(candles, (list, np.ndarray)):
        arr = np.array(candles)
        if arr.ndim == 2 and arr.shape[1] >= 5:
            close = pd.Series(arr[:, 4].astype(float))
        else:
            return pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float)
    return close.pct_change().dropna()


class VolatilityForecaster:
    """
    Forecasts future volatility using GARCH models.
    Adjusts position sizing before volatility events.
    """

    def __init__(self):
        self.forecast_horizon = int(__import__("os").getenv("VOL_FORECAST_HORIZON_DAYS", "5"))

    def forecast_volatility(
        self,
        symbol: str,
        returns: Union[pd.Series, List[float], None],
        horizon_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Forecast volatility for next N days.
        returns: historical daily % changes (or pass candles and we derive).
        """
        if returns is None:
            return self._fallback_forecast(pd.Series(dtype=float))
        if isinstance(returns, list):
            returns = pd.Series(returns)
        returns = returns.dropna()
        horizon = horizon_days or self.forecast_horizon

        if len(returns) < 100:
            return self._fallback_forecast(returns)

        try:
            from arch import arch_model

            # GARCH(1,1) - percentage for numerical stability
            ret_pct = returns * 100
            model = arch_model(
                ret_pct,
                vol="Garch",
                p=1,
                q=1,
                mean="Zero",
                rescale=False,
            )
            fitted = model.fit(disp="off", show_warning=False)

            forecast = fitted.forecast(horizon=horizon, reindex=False)
            # forecast.variance shape (1, horizon)
            var_vals = np.asarray(forecast.variance).flatten()
            forecasted_vol = np.sqrt(var_vals) / 100  # back from pct
            forecasted_vol_annual = forecasted_vol * np.sqrt(252)

            cond_vol = fitted.conditional_volatility
            current_vol = float(np.sqrt(cond_vol.iloc[-1] ** 2 / 10000) * np.sqrt(252))
            if not np.isfinite(current_vol) or current_vol <= 0:
                current_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 0 else 0.20

            first_f = float(forecasted_vol_annual[0]) if len(forecasted_vol_annual) > 0 else current_vol
            last_f = float(forecasted_vol_annual[-1]) if len(forecasted_vol_annual) > 0 else current_vol

            if last_f > first_f * 1.1:
                trend = "increasing"
            elif last_f < first_f * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"

            hist_vol = returns.rolling(20).std() * np.sqrt(252)
            valid = hist_vol.dropna()
            current_percentile = float((valid < current_vol).mean()) if len(valid) > 0 else 0.5

            if current_vol < 0.15:
                regime = "low_vol"
            elif current_vol < 0.30:
                regime = "normal"
            elif current_vol < 0.50:
                regime = "high_vol"
            else:
                regime = "extreme_vol"

            return {
                "current_volatility": current_vol,
                "forecasted_volatility": [float(v) for v in forecasted_vol_annual],
                "volatility_trend": trend,
                "volatility_percentile": current_percentile,
                "regime": regime,
                "forecast_horizon_days": horizon,
                "avg_forecast": float(np.mean(forecasted_vol_annual)) if len(forecasted_vol_annual) > 0 else current_vol,
            }
        except Exception as e:
            logger.debug("GARCH forecast failed for %s: %s", symbol, e)
            return self._fallback_forecast(returns)

    def _fallback_forecast(self, returns: pd.Series) -> Dict[str, Any]:
        if returns.empty or len(returns) < 5:
            vol = 0.20
        else:
            vol = float(returns.rolling(20).std().iloc[-1] * np.sqrt(252))
            if not np.isfinite(vol) or vol <= 0:
                vol = 0.20
        return {
            "current_volatility": vol,
            "forecasted_volatility": [vol] * 5,
            "volatility_trend": "stable",
            "volatility_percentile": 0.5,
            "regime": "normal",
            "forecast_horizon_days": 5,
            "avg_forecast": vol,
        }

    def adjust_position_size_for_volatility_forecast(
        self,
        base_size: float,
        vol_forecast: Dict[str, Any],
        target_vol: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Adjust position size based on forecasted volatility.
        Maintain constant portfolio volatility.
        """
        target = target_vol or float(__import__("os").getenv("VOL_TARGET_ANNUAL", "0.20"))
        avg_f = vol_forecast.get("avg_forecast") or vol_forecast.get("current_volatility", 0.20)
        current_vol = vol_forecast.get("current_volatility", avg_f)
        trend = vol_forecast.get("volatility_trend", "stable")

        vol_mult = target / avg_f if avg_f > 1e-9 else 1.0
        vol_mult = max(0.3, min(2.0, vol_mult))

        if trend == "increasing":
            trend_adj = 0.9
            trend_reason = "volatility increasing"
        elif trend == "decreasing":
            trend_adj = 1.1
            trend_reason = "volatility decreasing"
        else:
            trend_adj = 1.0
            trend_reason = "volatility stable"

        final_mult = vol_mult * trend_adj
        adjusted_size = base_size * final_mult

        return {
            "adjusted_size": adjusted_size,
            "size_multiplier": final_mult,
            "reasoning": f"Forecast vol: {avg_f:.1%} (target: {target:.1%}), {trend_reason} → mult: {final_mult:.2f}x",
            "vol_forecast": avg_f,
            "target_vol": target,
        }


def calculate_position_size_with_vol_forecast(
    base_size: float,
    symbol: str,
    returns: Union[pd.Series, List[float], None],
    target_vol: float = 0.20,
) -> float:
    """Enhanced position sizing with volatility forecasting."""
    forecaster = VolatilityForecaster()
    vol_forecast = forecaster.forecast_volatility(symbol, returns)
    adj = forecaster.adjust_position_size_for_volatility_forecast(
        base_size=base_size,
        vol_forecast=vol_forecast,
        target_vol=target_vol,
    )
    logger.debug("%s position sizing: %s", symbol, adj.get("reasoning", ""))
    return adj["adjusted_size"]
