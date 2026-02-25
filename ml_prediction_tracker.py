"""
ML Prediction Tracking - log predictions, update outcomes, calculate accuracy.
"""
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLE_ML_PREDICTIONS = os.getenv("ENABLE_ML_PREDICTIONS", "0").strip().lower() in ("1", "true", "yes")
ML_MIN_ACCURACY = float(os.getenv("ML_MIN_ACCURACY", "0.55"))
ML_CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.70"))
ML_RETRAIN_FREQUENCY = int(os.getenv("ML_RETRAIN_FREQUENCY", "7"))  # days


def log_prediction(
    symbol: str,
    predicted_direction: str,
    confidence: float,
    predicted_price: Optional[float] = None,
    price_at_prediction: Optional[float] = None,
    model_used: str = "ensemble",
    model_version: Optional[str] = None,
    regime_at_prediction: Optional[str] = None,
) -> Optional[int]:
    """Log ML prediction to DB. Returns prediction id or None."""
    if not ENABLE_ML_PREDICTIONS:
        return None
    try:
        from db import save_ml_prediction
        return save_ml_prediction(
            symbol=symbol,
            prediction_date=int(time.time()),
            predicted_direction=predicted_direction,
            predicted_price=predicted_price,
            confidence=confidence,
            price_at_prediction=price_at_prediction,
            model_version=model_version,
            model_used=model_used,
            regime_at_prediction=regime_at_prediction,
        )
    except Exception as e:
        logger.debug("Log prediction failed: %s", e)
        return None


def get_ml_accuracy_for_scoring(days_back: int = 30) -> Tuple[float, float]:
    """
    Get ML accuracy for recommendation score weight.
    Returns: (accuracy 0-1, weight 0-0.15).
    Weight: 15% if accuracy >65%, 0% if <55%, linear between.
    """
    try:
        from db import get_ml_model_accuracy
        stats = get_ml_model_accuracy(days_back=days_back)
        acc = stats.get("accuracy", 0.5)
        if stats.get("total", 0) < 20:
            return acc, 0.0
        if acc >= 0.65:
            return acc, 0.15
        if acc < 0.55:
            return acc, 0.0
        weight = 0.15 * (acc - 0.55) / 0.10
        return acc, weight
    except Exception as e:
        logger.debug("ML accuracy fetch failed: %s", e)
        return 0.5, 0.0


def get_ensemble_agreement(model_preds: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Check if all models agree. High conviction = all same direction + confidence >70%.
    Returns: (high_conviction, direction).
    """
    if not model_preds:
        return False, "NEUTRAL"
    dirs = [p.get("direction") for p in model_preds.values() if p]
    if not dirs:
        return False, "NEUTRAL"
    if len(set(dirs)) > 1:
        return False, "NEUTRAL"
    confs = [p.get("prob_up", 0.5) if p.get("direction") == "UP" else (1 - p.get("prob_down", 0.5)) for p in model_preds.values() if p]
    if not confs:
        return False, dirs[0]
    min_conf = min(confs)
    high_conviction = min_conf >= ML_CONFIDENCE_THRESHOLD
    return high_conviction, dirs[0]


def get_ml_score_for_recommendation(
    symbol: str,
    candles: List[List[float]],
    current_price: float,
    regime: Optional[str] = None,
) -> Tuple[float, str, Dict[str, Any]]:
    """
    Get ML contribution to recommendation score.
    Returns: (score_delta -15 to +15, conviction, details).
    """
    if not ENABLE_ML_PREDICTIONS:
        return 0.0, "none", {}
    try:
        from db import get_ml_model_accuracy
        acc, weight = get_ml_accuracy_for_scoring(30)
        if weight <= 0:
            return 0.0, "low_accuracy", {"accuracy": acc, "weight": 0}
        try:
            from ml_ensemble import get_ml_ensemble
            ensemble = get_ml_ensemble()
            if not ensemble._is_trained:
                return 0.0, "untrained", {}
            atr_val = 0.02
            if candles and len(candles) >= 14:
                highs = [c[2] for c in candles[-14:]]
                lows = [c[3] for c in candles[-14:]]
                closes = [c[4] for c in candles[-14:]]
                if highs and lows and closes:
                    trs = []
                    for i in range(1, len(highs)):
                        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                        trs.append(tr)
                    atr_val = sum(trs) / len(trs) / current_price if trs and current_price > 0 else 0.02
            pred = ensemble.predict(candles, current_volatility=atr_val)
            agree, direction = get_ensemble_agreement(pred.model_predictions)
            if direction == "NEUTRAL":
                return 0.0, "neutral", {"accuracy": acc, "direction": direction}
            if not agree:
                contrib = 5.0 if direction == "UP" else -5.0
                return contrib * weight, "low_conviction", {"accuracy": acc, "direction": direction, "agreement": False}
            contrib = 15.0 if direction == "UP" else -15.0
            return contrib * weight, "high_conviction", {"accuracy": acc, "direction": direction, "agreement": True, "confidence": pred.confidence}
        except Exception as e:
            logger.debug("ML ensemble predict failed: %s", e)
            try:
                from ml_predictor import create_ml_predictor
                predictor = create_ml_predictor(use_advanced=False)
                res = predictor.predict(candles, current_price, symbol)
                if res and res.direction != "NEUTRAL":
                    contrib = 10.0 if res.direction == "UP" else -10.0
                    return contrib * weight, "fallback", {"accuracy": acc, "direction": res.direction, "model": res.model_used}
            except Exception:
                pass
        return 0.0, "unavailable", {"accuracy": acc}
    except Exception as e:
        logger.debug("ML score failed: %s", e)
        return 0.0, "error", {}


def update_outcomes_job(price_fetcher: Optional[Any] = None) -> Dict[str, int]:
    """
    Background job: update predictions with actual outcomes after 7/30 days.
    price_fetcher(symbol, timestamp) -> price or None. If None, outcomes not updated.
    """
    try:
        from db import get_ml_predictions, update_ml_prediction_outcome
        updated_7 = 0
        updated_30 = 0
        cutoff_7 = int(time.time()) - 8 * 86400
        cutoff_30 = int(time.time()) - 31 * 86400
        preds = get_ml_predictions(days_back=0, limit=500)
        for p in preds:
            pid = p.get("id")
            pred_ts = p.get("prediction_date") or p.get("recorded_at") or 0
            price_at = p.get("price_at_prediction")
            if not price_at or price_at <= 0:
                continue
            if price_fetcher and p.get("actual_outcome_7d") is None and pred_ts <= cutoff_7:
                try:
                    price_7d = price_fetcher(p["symbol"], pred_ts + 7 * 86400)
                    if price_7d and price_7d > 0:
                        ret_7d = (price_7d - price_at) / price_at
                        update_ml_prediction_outcome(pid, actual_outcome_7d=ret_7d)
                        updated_7 += 1
                except Exception:
                    pass
            if price_fetcher and p.get("actual_outcome_30d") is None and pred_ts <= cutoff_30:
                try:
                    price_30d = price_fetcher(p["symbol"], pred_ts + 30 * 86400)
                    if price_30d and price_30d > 0:
                        ret_30d = (price_30d - price_at) / price_at
                        update_ml_prediction_outcome(pid, actual_outcome_30d=ret_30d)
                        updated_30 += 1
                except Exception:
                    pass
        return {"updated_7d": updated_7, "updated_30d": updated_30}
    except Exception as e:
        logger.warning("Update outcomes job failed: %s", e)
        return {"updated_7d": 0, "updated_30d": 0}
