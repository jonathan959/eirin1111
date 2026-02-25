"""
Order flow analysis (Phase 2 Advanced Intelligence).
Detects institutional (smart money) activity: large orders, accumulation/distribution.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _normalize_ob(ob: Dict[str, Any]) -> Dict[str, List]:
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    # CCXT/Kraken: [[price, size], ...]
    out_b = []
    out_a = []
    for b in bids[:20]:
        p, s = (b[0], b[1]) if len(b) >= 2 else (0, 0)
        out_b.append({"price": float(p), "size": float(s)})
    for a in asks[:20]:
        p, s = (a[0], a[1]) if len(a) >= 2 else (0, 0)
        out_a.append({"price": float(p), "size": float(s)})
    return {"bids": out_b, "asks": out_a}


class OrderFlowAnalyzer:
    """
    Detects institutional order flow patterns.
    Smart money indicators: volume profile, order book imbalance, large orders.
    """

    def analyze_order_flow(self, symbol: str, market_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze order flow for institutional signals.
        Returns smart_money_signal, signal_strength, recommendation.
        """
        if market_type is None:
            market_type = "stock" if (len(symbol) < 6 and "/" not in symbol) else "crypto"

        signals: List[tuple] = []

        try:
            from phase2_data_fetcher import fetch_recent_candles, fetch_order_book, fetch_recent_trades

            vol_sig = self._analyze_volume_profile(symbol, fetch_recent_candles)
            signals.append(("volume", vol_sig))

            ob_sig = self._analyze_order_book_imbalance(symbol, fetch_order_book)
            signals.append(("order_book", ob_sig))

            large_sig = self._detect_large_orders(symbol, fetch_recent_trades)
            signals.append(("large_orders", large_sig))

            if market_type == "crypto":
                whale_sig = self._detect_whale_activity(symbol)
                signals.append(("whale", whale_sig))
            else:
                signals.append(("dark_pool", {"direction": "neutral", "dark_pool_ratio": 0.0}))

        except Exception as e:
            logger.debug("analyze_order_flow: %s", e)
            return {
                "smart_money_signal": "neutral",
                "signal_strength": 0.5,
                "signals_breakdown": {},
                "recommendation": "Data unavailable",
            }

        bullish = sum(1 for _, s in signals if s.get("direction") == "bullish")
        bearish = sum(1 for _, s in signals if s.get("direction") == "bearish")
        total = max(1, len(signals))

        if bullish > bearish * 1.5:
            smart_money_signal = "accumulation"
            signal_strength = bullish / total
        elif bearish > bullish * 1.5:
            smart_money_signal = "distribution"
            signal_strength = bearish / total
        else:
            smart_money_signal = "neutral"
            signal_strength = 0.5

        if smart_money_signal == "accumulation" and signal_strength > 0.6:
            recommendation = "Follow smart money - increase position"
        elif smart_money_signal == "distribution" and signal_strength > 0.6:
            recommendation = "Smart money exiting - reduce position or avoid"
        else:
            recommendation = "Mixed signals - proceed with caution"

        return {
            "smart_money_signal": smart_money_signal,
            "signal_strength": float(signal_strength),
            "signals_breakdown": {n: s for n, s in signals},
            "recommendation": recommendation,
        }

    def _analyze_volume_profile(
        self,
        symbol: str,
        fetch_fn,
    ) -> Dict[str, Any]:
        try:
            candles = fetch_fn(symbol, timeframe="1h", periods=100)
            if not candles or len(candles) < 20:
                return {"direction": "neutral", "pattern": "unknown", "volume_ratio": 1.0, "strength": 0}

            import numpy as np
            arr = np.array(candles)
            close = arr[:, 4].astype(float)
            ret = np.diff(close)
            up = ret > 0
            vol = arr[1:, 5].astype(float) if arr.shape[1] >= 6 else np.ones(len(ret))
            up_vol = np.where(up, vol, 0).sum()
            down_vol = np.where(~up, vol, 0).sum()
            ratio = up_vol / down_vol if down_vol > 0 else 1.0

            if ratio > 1.3:
                return {"direction": "bullish", "pattern": "accumulation", "volume_ratio": float(ratio), "strength": min(1.0, (ratio - 1) / 2)}
            elif ratio < 0.7:
                return {"direction": "bearish", "pattern": "distribution", "volume_ratio": float(ratio), "strength": min(1.0, (1 - ratio) / 2)}
            return {"direction": "neutral", "pattern": "balanced", "volume_ratio": float(ratio), "strength": 0}
        except Exception:
            return {"direction": "neutral", "pattern": "unknown", "volume_ratio": 1.0, "strength": 0}

    def _analyze_order_book_imbalance(
        self,
        symbol: str,
        fetch_fn,
    ) -> Dict[str, Any]:
        try:
            ob = _normalize_ob(fetch_fn(symbol))
            bids = ob["bids"]
            asks = ob["asks"]
            total_bid = sum(b["size"] * b["price"] for b in bids[:10])
            total_ask = sum(a["size"] * a["price"] for a in asks[:10])

            if total_ask <= 0:
                ratio = 1.0
            else:
                ratio = total_bid / total_ask

            if ratio > 1.2:
                return {"direction": "bullish", "pressure": "buying_pressure", "imbalance_ratio": float(ratio), "total_bid_volume": total_bid, "total_ask_volume": total_ask}
            elif ratio < 0.8:
                return {"direction": "bearish", "pressure": "selling_pressure", "imbalance_ratio": float(ratio), "total_bid_volume": total_bid, "total_ask_volume": total_ask}
            return {"direction": "neutral", "pressure": "balanced", "imbalance_ratio": float(ratio), "total_bid_volume": total_bid, "total_ask_volume": total_ask}
        except Exception:
            return {"direction": "neutral", "pressure": "unknown", "imbalance_ratio": 1.0}

    def _detect_large_orders(
        self,
        symbol: str,
        fetch_fn,
    ) -> Dict[str, Any]:
        try:
            trades = fetch_fn(symbol, limit=500)
            if not trades:
                return {"direction": "neutral", "large_orders_detected": False}

            import numpy as np
            sizes = np.array([float(t.get("size", 0) or 0) for t in trades])
            avg = sizes.mean()
            std = sizes.std()
            if std < 1e-9:
                return {"direction": "neutral", "large_orders_detected": False}
            threshold = avg + 3 * std
            large = [t for t in trades if float(t.get("size", 0) or 0) > threshold]

            if not large:
                return {"direction": "neutral", "large_orders_detected": False}

            buy_vol = sum(float(t.get("size", 0) or 0) for t in large if str(t.get("side", "")).lower() == "buy")
            sell_vol = sum(float(t.get("size", 0) or 0) for t in large if str(t.get("side", "")).lower() == "sell")

            if buy_vol > sell_vol * 1.5:
                return {"direction": "bullish", "pattern": "whale_accumulation", "large_orders_detected": True, "large_buy_volume": buy_vol, "large_sell_volume": sell_vol, "num_large_trades": len(large)}
            elif sell_vol > buy_vol * 1.5:
                return {"direction": "bearish", "pattern": "whale_distribution", "large_orders_detected": True, "large_buy_volume": buy_vol, "large_sell_volume": sell_vol, "num_large_trades": len(large)}
            return {"direction": "neutral", "pattern": "mixed", "large_orders_detected": True}
        except Exception:
            return {"direction": "neutral", "large_orders_detected": False}

    def _detect_whale_activity(self, symbol: str) -> Dict[str, Any]:
        # Placeholder - would integrate Glassnode/Nansen for crypto
        return {"direction": "neutral", "activity": "unknown"}
