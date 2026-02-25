"""
Phase 2 Intelligence: Order Book Depth Analysis
Analyzes order book to estimate liquidity and price impact
"""
import time
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class OrderBookAnalyzer:
    """
    Analyzes order book depth to:
    1. Estimate slippage for trade sizes
    2. Check liquidity before large trades
    3. Identify support/resistance levels
    4. Detect order book imbalances (buy/sell pressure)
    """
    
    def __init__(
        self,
        max_slippage_pct: float = 0.01,  # 1% max acceptable slippage
        min_liquidity_mult: float = 3.0   # Need 3x trade size in liquidity
    ):
        self.max_slippage_pct = max_slippage_pct
        self.min_liquidity_mult = min_liquidity_mult
    
    def analyze_liquidity(
        self,
        order_book: Dict[str, Any],
        side: str,  # "buy" or "sell"
        size_quote: float  # Trade size in quote currency (USD)
    ) -> Tuple[bool, float, str, Dict[str, Any]]:
        """
        Analyze order book liquidity for a trade.
        
        Args:
            order_book: Order book dict with 'bids' and 'asks'
                        Each is list of [price, volume] pairs
            side: "buy" (check asks) or "sell" (check bids)
            size_quote: Trade size in quote currency
        
        Returns:
            (liquidity_ok, estimated_slippage_pct, reason, details)
        """
        if side.lower() == "buy":
            levels = order_book.get('asks', [])
            direction = "asks"
        else:
            levels = order_book.get('bids', [])
            direction = "bids"
        
        if not levels:
            return False, 0.0, f"Empty order book ({direction})", {}
        
        # Calculate how deep we need to go to fill order
        best_price = levels[0][0]
        cumulative_volume_quote = 0.0
        cumulative_volume_base = 0.0
        levels_needed = 0
        worst_price = best_price
        
        for price, volume in levels[:50]:  # Check top 50 levels
            order_value_quote = price * volume
            cumulative_volume_quote += order_value_quote
            cumulative_volume_base += volume
            levels_needed += 1
            worst_price = price
            
            if cumulative_volume_quote >= size_quote:
                break
        
        # Calculate slippage
        if best_price > 0:
            slippage_pct = abs(worst_price - best_price) / best_price
        else:
            slippage_pct = 0.0
        
        # Check if we have enough liquidity
        liquidity_ratio = cumulative_volume_quote / size_quote if size_quote > 0 else 0.0
        
        # Determine if liquidity is acceptable
        if cumulative_volume_quote < size_quote:
            reason = f"Insufficient liquidity: ${cumulative_volume_quote:.2f} available for ${size_quote:.2f} order"
            liquidity_ok = False
        elif slippage_pct > self.max_slippage_pct:
            reason = f"High slippage: {slippage_pct*100:.2f}% (max: {self.max_slippage_pct*100:.1f}%)"
            liquidity_ok = False
        elif liquidity_ratio < self.min_liquidity_mult:
            reason = f"Low liquidity depth: {liquidity_ratio:.1f}x trade size (need {self.min_liquidity_mult:.1f}x)"
            liquidity_ok = False
        else:
            reason = f"Good liquidity: {slippage_pct*100:.2f}% slippage, {liquidity_ratio:.1f}x depth"
            liquidity_ok = True
        
        details = {
            'best_price': best_price,
            'worst_price': worst_price,
            'slippage_pct': slippage_pct,
            'levels_needed': levels_needed,
            'cumulative_volume_quote': cumulative_volume_quote,
            'cumulative_volume_base': cumulative_volume_base,
            'liquidity_ratio': liquidity_ratio,
            'trade_size_quote': size_quote
        }
        
        return liquidity_ok, slippage_pct, reason, details

    def estimate_slippage(
        self,
        order_size: float,
        side: str,
        order_book_data: Optional[Dict[str, Any]] = None,
        size_quote: Optional[float] = None,
    ) -> Optional[float]:
        """
        Estimate slippage for a given order. Returns slippage as decimal (e.g. 0.004 = 0.4%).
        order_size: size in base currency. If size_quote not provided, uses best bid/ask * order_size.
        Returns None if order_book_data is None or empty.
        """
        if not order_book_data:
            return None
        levels = order_book_data.get("asks" if side.lower() == "buy" else "bids", [])
        if not levels:
            return None
        best = float(levels[0][0])
        sz = size_quote if size_quote is not None and size_quote > 0 else order_size * best
        _, slippage_pct, _, _ = self.analyze_liquidity(order_book_data, side, sz)
        return float(slippage_pct)

    def analyze_imbalance(
        self,
        order_book: Dict[str, Any],
        depth: int = 10
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Analyze order book imbalance (buy vs sell pressure).
        
        Args:
            order_book: Order book with 'bids' and 'asks'
            depth: Number of levels to analyze
        
        Returns:
            (imbalance_ratio, interpretation, details)
            imbalance_ratio: >1.0 = buy pressure, <1.0 = sell pressure
        """
        bids = order_book.get('bids', [])[:depth]
        asks = order_book.get('asks', [])[:depth]
        
        if not bids or not asks:
            return 1.0, "Neutral (no order book data)", {}
        
        # Calculate total volume on each side
        bid_volume = sum(price * volume for price, volume in bids)
        ask_volume = sum(price * volume for price, volume in asks)
        
        if ask_volume == 0:
            return float('inf'), "Extreme buy pressure (no asks)", {'bid_volume': bid_volume, 'ask_volume': 0}
        
        # Imbalance ratio
        imbalance_ratio = bid_volume / ask_volume
        
        # Interpret
        if imbalance_ratio > 1.5:
            interpretation = "Strong buy pressure"
        elif imbalance_ratio > 1.1:
            interpretation = "Moderate buy pressure"
        elif imbalance_ratio > 0.9:
            interpretation = "Neutral"
        elif imbalance_ratio > 0.66:
            interpretation = "Moderate sell pressure"
        else:
            interpretation = "Strong sell pressure"
        
        details = {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance_ratio': imbalance_ratio,
            'bid_levels': len(bids),
            'ask_levels': len(asks)
        }
        
        return imbalance_ratio, interpretation, details
    
    def find_support_resistance(
        self,
        order_book: Dict[str, Any],
        current_price: float,
        min_volume_threshold: float = 10000.0  # Min $10k volume to be significant
    ) -> Tuple[List[float], List[float], Dict[str, Any]]:
        """
        Identify support and resistance levels from order book.
        
        Args:
            order_book: Order book with 'bids' and 'asks'
            current_price: Current market price
            min_volume_threshold: Minimum volume to consider significant
        
        Returns:
            (support_levels, resistance_levels, details)
        """
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        # Find support levels (large bids below current price)
        support_levels = []
        for price, volume in bids:
            if price < current_price:
                volume_quote = price * volume
                if volume_quote >= min_volume_threshold:
                    support_levels.append(price)
        
        # Find resistance levels (large asks above current price)
        resistance_levels = []
        for price, volume in asks:
            if price > current_price:
                volume_quote = price * volume
                if volume_quote >= min_volume_threshold:
                    resistance_levels.append(price)
        
        details = {
            'current_price': current_price,
            'support_count': len(support_levels),
            'resistance_count': len(resistance_levels),
            'min_volume_threshold': min_volume_threshold
        }
        
        return support_levels, resistance_levels, details

    def detect_bid_ask_walls(
        self,
        order_book: Dict[str, Any],
        current_price: float,
        min_wall_size_pct: float = 0.02,
        depth_levels: int = 20,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Detect large bid/ask walls (support/resistance).
        min_wall_size_pct: wall must be at least this % of total side volume to count.
        Returns: (bid_walls, ask_walls, details)
        """
        bids = order_book.get("bids", [])[:depth_levels]
        asks = order_book.get("asks", [])[:depth_levels]
        bid_walls = []
        ask_walls = []
        total_bid_vol = sum(p * v for p, v in bids)
        total_ask_vol = sum(p * v for p, v in asks)
        for price, volume in bids:
            pv = price * volume
            if total_bid_vol > 0 and pv / total_bid_vol >= min_wall_size_pct:
                bid_walls.append({"price": price, "volume": volume, "volume_quote": pv, "pct_of_side": pv / total_bid_vol})
        for price, volume in asks:
            pv = price * volume
            if total_ask_vol > 0 and pv / total_ask_vol >= min_wall_size_pct:
                ask_walls.append({"price": price, "volume": volume, "volume_quote": pv, "pct_of_side": pv / total_ask_vol})
        details = {"total_bid_vol": total_bid_vol, "total_ask_vol": total_ask_vol, "current_price": current_price}
        return bid_walls, ask_walls, details

    def detect_spoofing(
        self,
        order_book: Dict[str, Any],
        recent_snapshots: Optional[List[Dict[str, Any]]] = None,
        volatility_threshold: float = 0.3,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Detect potential spoofing (large orders placed then cancelled quickly).
        recent_snapshots: list of prior order books with timestamps; if large walls disappear between snaps, possible spoofing.
        Returns: (is_suspicious, reason, details)
        """
        if not recent_snapshots or len(recent_snapshots) < 2:
            return False, "Insufficient history for spoofing check", {}
        bids_now = {float(p): float(v) for p, v in order_book.get("bids", [])[:10]}
        asks_now = {float(p): float(v) for p, v in order_book.get("asks", [])[:10]}
        prev = recent_snapshots[-1]
        bids_prev = {float(p): float(v) for p, v in prev.get("bids", [])[:10]}
        asks_prev = {float(p): float(v) for p, v in prev.get("asks", [])[:10]}
        disappeared_bids = [(p, bids_prev[p]) for p in bids_prev if p not in bids_now or bids_now.get(p, 0) < bids_prev[p] * (1 - volatility_threshold)]
        disappeared_asks = [(p, asks_prev[p]) for p in asks_prev if p not in asks_now or asks_now.get(p, 0) < asks_prev[p] * (1 - volatility_threshold)]
        large_drops = [x for x in disappeared_bids + disappeared_asks if x[1] > 0]
        if len(large_drops) >= 2:
            return True, "Large walls disappeared between snapshots (possible spoofing)", {"disappeared": large_drops}
        return False, "No spoofing indicators", {}

    def detect_layering(
        self,
        order_book: Dict[str, Any],
        current_price: float,
        min_levels: int = 4,
        similarity_pct: float = 0.02,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Detect layering: multiple orders creating fake support/resistance.
        Look for evenly spaced levels with similar size - characteristic of layered spoofing.
        Returns: (is_suspicious, reason, details)
        """
        bids = order_book.get("bids", [])[:20]
        asks = order_book.get("asks", [])[:20]
        if len(bids) < min_levels or len(asks) < min_levels:
            return False, "Insufficient depth for layering check", {}
        # Check bid side: similar volumes at evenly spaced prices?
        bid_prices = [float(p) for p, _ in bids]
        bid_vols = [float(v) for _, v in bids]
        avg_vol = sum(bid_vols) / len(bid_vols) if bid_vols else 0
        vol_std = (sum((v - avg_vol) ** 2 for v in bid_vols) / len(bid_vols)) ** 0.5 if len(bid_vols) > 1 else 0
        # Low variance in volume + evenly spaced = layering
        price_gaps = [bid_prices[i] - bid_prices[i + 1] for i in range(len(bid_prices) - 1)]
        avg_gap = sum(price_gaps) / len(price_gaps) if price_gaps else 0
        gap_cv = (sum((g - avg_gap) ** 2 for g in price_gaps) / len(price_gaps)) ** 0.5 / avg_gap if avg_gap > 0 else 0
        if vol_std < avg_vol * 0.3 and gap_cv < 0.3:
            return True, "Evenly layered bid levels (possible layering)", {"side": "bid", "levels": len(bids)}
        # Same for asks
        ask_prices = [float(p) for p, _ in asks]
        ask_vols = [float(v) for _, v in asks]
        avg_vol_a = sum(ask_vols) / len(ask_vols) if ask_vols else 0
        vol_std_a = (sum((v - avg_vol_a) ** 2 for v in ask_vols) / len(ask_vols)) ** 0.5 if len(ask_vols) > 1 else 0
        price_gaps_a = [ask_prices[i + 1] - ask_prices[i] for i in range(len(ask_prices) - 1)]
        avg_gap_a = sum(price_gaps_a) / len(price_gaps_a) if price_gaps_a else 0
        gap_cv_a = (sum((g - avg_gap_a) ** 2 for g in price_gaps_a) / len(price_gaps_a)) ** 0.5 / avg_gap_a if avg_gap_a > 0 else 0
        if vol_std_a < avg_vol_a * 0.3 and gap_cv_a < 0.3:
            return True, "Evenly layered ask levels (possible layering)", {"side": "ask", "levels": len(asks)}
        return False, "No layering indicators", {}

    def get_manipulated_levels(
        self,
        order_book: Dict[str, Any],
        recent_snapshots: Optional[List[Dict[str, Any]]] = None,
        current_price: float = 0.0,
    ) -> Tuple[List[float], List[float], Dict[str, Any]]:
        """
        Return price levels to avoid (manipulated). Don't trade into these.
        Returns: (avoid_buy_at_levels, avoid_sell_at_levels, details)
        """
        avoid_buy = []
        avoid_sell = []
        details = {"spoofing": False, "layering": False}
        spoof, _, spoof_d = self.detect_spoofing(order_book, recent_snapshots)
        if spoof and spoof_d.get("disappeared"):
            for p, _ in spoof_d["disappeared"]:
                avoid_buy.append(p)
                avoid_sell.append(p)
            details["spoofing"] = True
        lay, _, lay_d = self.detect_layering(order_book, current_price)
        if lay and lay_d.get("side") == "bid":
            bids = order_book.get("bids", [])[:10]
            for p, _ in bids:
                avoid_buy.append(float(p))
            details["layering"] = True
        elif lay and lay_d.get("side") == "ask":
            asks = order_book.get("asks", [])[:10]
            for p, _ in asks:
                avoid_sell.append(float(p))
            details["layering"] = True
        if spoof or lay:
            logger.info("Order book manipulation detected: spoofing=%s layering=%s - avoid levels logged", spoof, lay)
        return avoid_buy, avoid_sell, details


class SmartOrderRouter:
    """
    Routes orders intelligently based on order book analysis.
    Can split large orders to minimize slippage.
    """
    
    def __init__(self, analyzer: Optional[OrderBookAnalyzer] = None):
        self.analyzer = analyzer or OrderBookAnalyzer()
    
    def route_order(
        self,
        order_book: Dict[str, Any],
        side: str,
        size_quote: float,
        max_slippage_pct: float = 0.01
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Determine best way to execute order.
        
        Returns:
            (strategy, order_chunks, details)
            strategy: "market", "limit", "twap", "iceberg"
            order_chunks: List of order chunks [{price, size}, ...]
        """
        # Analyze liquidity
        liquidity_ok, slippage, reason, liq_details = self.analyzer.analyze_liquidity(
            order_book,
            side,
            size_quote
        )
        
        # If liquidity is good, execute as single market order
        if liquidity_ok and slippage <= max_slippage_pct * 0.5:
            return "market", [{'size_quote': size_quote}], {
                'reason': f"Sufficient liquidity for market order ({slippage*100:.2f}% slippage)",
                'liquidity_analysis': liq_details
            }
        
        # If moderate slippage, use limit order at best price
        if liquidity_ok:
            best_price = liq_details['best_price']
            return "limit", [{'price': best_price, 'size_quote': size_quote}], {
                'reason': f"Using limit order to avoid {slippage*100:.2f}% slippage",
                'liquidity_analysis': liq_details
            }
        
        # If poor liquidity, recommend splitting order (TWAP - Time Weighted Average Price)
        num_chunks = 5  # Split into 5 orders
        chunk_size = size_quote / num_chunks
        
        chunks = []
        for i in range(num_chunks):
            chunks.append({
                'size_quote': chunk_size,
                'delay_minutes': i * 2  # 2 minutes between orders
            })
        
        return "twap", chunks, {
            'reason': f"Poor liquidity - splitting into {num_chunks} orders to reduce impact",
            'liquidity_analysis': liq_details
        }


# =============================================================================
# Testing / Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example order book
    order_book = {
        'bids': [  # Buy orders (price, volume)
            [100.00, 5.0],
            [99.95, 3.0],
            [99.90, 10.0],
            [99.80, 20.0],
            [99.50, 50.0]
        ],
        'asks': [  # Sell orders (price, volume)
            [100.05, 4.0],
            [100.10, 6.0],
            [100.20, 8.0],
            [100.50, 15.0],
            [101.00, 30.0]
        ]
    }
    
    analyzer = OrderBookAnalyzer()
    
    # Test 1: Small trade (good liquidity expected)
    print("=== Test 1: Small Trade ($500) ===")
    ok, slippage, reason, details = analyzer.analyze_liquidity(
        order_book,
        side="buy",
        size_quote=500.0
    )
    print(f"Liquidity OK: {ok}")
    print(f"Slippage: {slippage*100:.2f}%")
    print(f"Reason: {reason}")
    print(f"Levels needed: {details['levels_needed']}")
    print()
    
    # Test 2: Large trade (high slippage expected)
    print("=== Test 2: Large Trade ($5000) ===")
    ok, slippage, reason, details = analyzer.analyze_liquidity(
        order_book,
        side="buy",
        size_quote=5000.0
    )
    print(f"Liquidity OK: {ok}")
    print(f"Slippage: {slippage*100:.2f}%")
    print(f"Reason: {reason}")
    print()
    
    # Test 3: Order book imbalance
    print("=== Test 3: Order Book Imbalance ===")
    imbalance, interpretation, imb_details = analyzer.analyze_imbalance(order_book)
    print(f"Imbalance Ratio: {imbalance:.2f}")
    print(f"Interpretation: {interpretation}")
    print(f"Bid Volume: ${imb_details['bid_volume']:.2f}")
    print(f"Ask Volume: ${imb_details['ask_volume']:.2f}")
    print()
    
    # Test 4: Smart order routing
    print("=== Test 4: Smart Order Routing ===")
    router = SmartOrderRouter(analyzer)
    strategy, chunks, route_details = router.route_order(
        order_book,
        side="buy",
        size_quote=5000.0
    )
    print(f"Strategy: {strategy}")
    print(f"Reason: {route_details['reason']}")
    print(f"Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}: ${chunk['size_quote']:.2f}")
