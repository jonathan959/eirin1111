"""
Real-time liquidity analysis and execution optimization (Phase 2 Advanced Intelligence).
Predicts slippage, optimizes order timing, detects liquidity zones.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_ob(ob: Dict[str, Any]) -> Dict[str, List]:
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    out_b = []
    out_a = []
    for b in bids[:50]:
        p, s = (b[0], b[1]) if len(b) >= 2 else (0, 0)
        out_b.append({"price": float(p), "size": float(s)})
    for a in asks[:50]:
        p, s = (a[0], a[1]) if len(a) >= 2 else (0, 0)
        out_a.append({"price": float(p), "size": float(s)})
    return {"bids": out_b, "asks": out_a}


class LiquidityOptimizer:
    """
    Analyzes market liquidity in real-time.
    Optimizes order execution to minimize slippage.
    """

    def analyze_liquidity(
        self,
        symbol: str,
        order_size: float,
        side: str,
    ) -> Dict[str, Any]:
        """
        Analyze current market liquidity for proposed order.
        order_size: in base currency units.
        Returns liquidity_score, predicted_slippage_pct, execution_recommendation.
        """
        try:
            from phase2_data_fetcher import fetch_order_book, fetch_recent_candles

            ob = _normalize_ob(fetch_order_book(symbol, depth=50))
            bids = ob["bids"]
            asks = ob["asks"]

            if not bids or not asks:
                return self._fallback_result("No order book")

            best_bid = bids[0]["price"]
            best_ask = asks[0]["price"]
            mid_price = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else best_ask or best_bid

            depth = self._calculate_order_book_depth(ob, mid_price, side)
            predicted_slippage = self._predict_slippage(ob, order_size, side, mid_price)
            vol_analysis = self._analyze_volume_timing(symbol, fetch_recent_candles)
            liquidity_score = self._calculate_liquidity_score(depth, predicted_slippage, vol_analysis)
            exec_rec = self._recommend_execution_strategy(order_size, liquidity_score, predicted_slippage, vol_analysis)

            return {
                "liquidity_score": liquidity_score,
                "predicted_slippage_pct": predicted_slippage,
                "liquidity_tier": self._classify_liquidity_tier(liquidity_score),
                "order_book_depth": depth.get("available_liquidity", 0),
                "execution_recommendation": exec_rec.get("strategy", "limit_order"),
                "optimal_timing": exec_rec.get("timing", "immediate"),
                "split_recommendation": exec_rec.get("split"),
                "details": {"depth_analysis": depth, "volume_analysis": vol_analysis},
            }
        except Exception as e:
            logger.debug("analyze_liquidity: %s", e)
            return self._fallback_result(str(e))

    def _fallback_result(self, reason: str) -> Dict[str, Any]:
        return {
            "liquidity_score": 0.5,
            "predicted_slippage_pct": 0.15,
            "liquidity_tier": "medium",
            "order_book_depth": 0,
            "execution_recommendation": "limit_order",
            "optimal_timing": "immediate",
            "split_recommendation": None,
            "details": {},
        }

    def _calculate_order_book_depth(
        self,
        ob: Dict[str, List[Dict]],
        mid_price: float,
        side: str,
    ) -> Dict[str, float]:
        book_side = ob["asks"] if side == "buy" else ob["bids"]
        threshold = mid_price * 1.005 if side == "buy" else mid_price * 0.995
        available = 0.0
        levels = 0
        for level in book_side:
            p = level["price"]
            s = level["size"]
            if side == "buy" and p <= threshold:
                available += s * p
                levels += 1
            elif side == "sell" and p >= threshold:
                available += s * p
                levels += 1
        return {
            "available_liquidity": available,
            "levels_within_threshold": levels,
            "avg_level_size": available / levels if levels > 0 else 0,
        }

    def _predict_slippage(
        self,
        ob: Dict[str, List[Dict]],
        order_size: float,
        side: str,
        mid_price: float,
    ) -> float:
        book_side = ob["asks"] if side == "buy" else ob["bids"]
        remaining = order_size
        total_cost = 0.0
        for level in book_side:
            if remaining <= 0:
                break
            fill_size = min(remaining, level["size"])
            total_cost += fill_size * level["price"]
            remaining -= fill_size

        if remaining > 0:
            return 5.0

        if order_size <= 0:
            return 0.0
        avg_fill = total_cost / order_size
        slippage_pct = abs((avg_fill - mid_price) / mid_price) * 100
        return slippage_pct

    def _analyze_volume_timing(
        self,
        symbol: str,
        fetch_fn,
    ) -> Dict[str, Any]:
        try:
            candles = fetch_fn(symbol, timeframe="1h", periods=24)
            if not candles or len(candles) < 12:
                return {"volume_ratio": 1.0, "volume_timing": "average_volume", "timing_quality": "fair"}

            import numpy as np
            arr = np.array(candles)
            vol = arr[:, 5].astype(float) if arr.shape[1] >= 6 else np.ones(len(arr))
            current_hour = datetime.utcnow().hour
            # Simple: compare last 2h vs overall
            recent_vol = vol[-2:].mean() if len(vol) >= 2 else vol[-1]
            overall = vol.mean()
            ratio = recent_vol / overall if overall > 0 else 1.0

            if ratio > 1.3:
                return {"volume_ratio": float(ratio), "volume_timing": "high_volume_period", "timing_quality": "excellent"}
            elif ratio > 1.1:
                return {"volume_ratio": float(ratio), "volume_timing": "above_average_volume", "timing_quality": "good"}
            elif ratio < 0.7:
                return {"volume_ratio": float(ratio), "volume_timing": "low_volume_period", "timing_quality": "poor"}
            return {"volume_ratio": float(ratio), "volume_timing": "average_volume", "timing_quality": "fair"}
        except Exception:
            return {"volume_ratio": 1.0, "volume_timing": "average_volume", "timing_quality": "fair"}

    def _calculate_liquidity_score(
        self,
        depth: Dict[str, float],
        slippage: float,
        vol_analysis: Dict[str, Any],
    ) -> float:
        levels = depth.get("levels_within_threshold", 0)
        depth_score = min(1.0, levels / 20)
        slippage_score = max(0, 1.0 - (slippage / 2.0))
        vol_score = {"excellent": 1.0, "good": 0.8, "fair": 0.6, "poor": 0.3}.get(vol_analysis.get("timing_quality", "fair"), 0.5)
        return 0.4 * depth_score + 0.4 * slippage_score + 0.2 * vol_score

    def _classify_liquidity_tier(self, score: float) -> str:
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        return "low"

    def _recommend_execution_strategy(
        self,
        order_size: float,
        liquidity_score: float,
        predicted_slippage: float,
        vol_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        if order_size < 1000 and liquidity_score > 0.7:
            return {"strategy": "market_order", "timing": "immediate", "reasoning": "Small order, good liquidity"}
        elif order_size < 10000 and liquidity_score > 0.8:
            return {"strategy": "limit_order", "timing": "immediate", "reasoning": "Medium order, excellent liquidity"}
        elif order_size >= 10000 or liquidity_score < 0.5:
            num_chunks = min(5, max(2, int(order_size / 2000)))
            return {
                "strategy": "twap",
                "timing": "split_over_time",
                "split": {"chunks": num_chunks, "interval_seconds": 300},
                "reasoning": f"Large order - split into {num_chunks} chunks",
            }
        elif vol_analysis.get("timing_quality") == "poor":
            return {"strategy": "limit_order", "timing": "wait_high_volume", "reasoning": "Low volume period"}
        return {"strategy": "limit_order", "timing": "immediate", "reasoning": "Standard execution"}
