"""
Phase 2 Intelligence: Portfolio Correlation Analysis
Analyzes correlations between assets to ensure diversification
"""
import time
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class PortfolioCorrelationAnalyzer:
    """
    Analyzes correlation between assets in portfolio.
    High correlation = portfolio is not diversified (risky).
    Low correlation = good diversification (safer).
    
    Correlation ranges:
    - 0.7 to 1.0: Very high (move together)
    - 0.4 to 0.7: High
    - 0.0 to 0.4: Moderate
    - -0.4 to 0.0: Low (independent)
    - -1.0 to -0.4: Negative (move opposite)
    """
    
    def __init__(
        self,
        high_correlation_threshold: float = 0.7,
        max_portfolio_correlation: float = 0.6,
        lookback_days: int = 30
    ):
        self.high_correlation_threshold = high_correlation_threshold
        self.max_portfolio_correlation = max_portfolio_correlation
        self.lookback_days = lookback_days
    
    def analyze_portfolio(
        self,
        active_bots: List[Dict[str, Any]],
        price_history: Dict[str, List[float]]
    ) -> Tuple[float, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Analyze portfolio for correlation and diversification.
        
        Args:
            active_bots: List of active bot configs [{symbol, ...}, ...]
            price_history: Dict of price histories {symbol: [prices...]}
        
        Returns:
            (diversification_score, high_corr_pairs, details)
            diversification_score: 0-1 (1=perfect diversification, 0=all correlated)
        """
        symbols = [bot['symbol'] for bot in active_bots]
        
        if len(symbols) < 2:
            # Single asset = perfect "diversification" (no correlation to check)
            return 1.0, [], {'message': 'Single asset portfolio'}
        
        # Calculate correlation matrix
        correlation_matrix = self._calculate_correlation_matrix(symbols, price_history)
        
        high_corr_pairs = []
        total_pairs = 0
        highly_correlated_pairs = 0
        abs_corr_sum = 0.0

        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                if symbols[i] not in correlation_matrix or symbols[j] not in correlation_matrix[symbols[i]]:
                    continue
                corr = correlation_matrix[symbols[i]][symbols[j]]
                total_pairs += 1
                abs_corr_sum += abs(corr)
                if abs(corr) >= self.high_correlation_threshold:
                    highly_correlated_pairs += 1
                    high_corr_pairs.append({
                        'symbol1': symbols[i],
                        'symbol2': symbols[j],
                        'correlation': corr,
                        'severity': 'critical' if abs(corr) >= 0.85 else 'high'
                    })

        avg_abs_correlation = (abs_corr_sum / total_pairs) if total_pairs > 0 else 0.0
        count_based_score = 1.0 - (highly_correlated_pairs / total_pairs) if total_pairs > 0 else 1.0
        diversification_score = 1.0 - avg_abs_correlation

        details = {
            'total_symbols': len(symbols),
            'total_pairs': total_pairs,
            'highly_correlated_pairs': highly_correlated_pairs,
            'avg_abs_correlation': avg_abs_correlation,
            'avg_correlation': avg_abs_correlation,
            'score_count_based': count_based_score,
            'score_avg_abs_based': diversification_score,
            'correlation_matrix': correlation_matrix
        }

        return diversification_score, high_corr_pairs, details
    
    def _calculate_correlation_matrix(
        self,
        symbols: List[str],
        price_history: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate pairwise correlation matrix.
        
        Returns:
            {symbol1: {symbol2: correlation, ...}, ...}
        """
        matrix = {}
        
        for symbol1 in symbols:
            matrix[symbol1] = {}
            
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    matrix[symbol1][symbol2] = 1.0  # Perfect correlation with self
                    continue
                
                # Get price histories
                prices1 = price_history.get(symbol1, [])
                prices2 = price_history.get(symbol2, [])
                
                if not prices1 or not prices2:
                    matrix[symbol1][symbol2] = 0.0
                    continue
                
                # Calculate correlation
                corr = self._pearson_correlation(prices1, prices2)
                matrix[symbol1][symbol2] = corr
        
        return matrix
    
    def _pearson_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """
        Calculate Pearson correlation coefficient.
        
        Formula: r = Σ((x - x̄)(y - ȳ)) / √(Σ(x - x̄)² × Σ(y - ȳ)²)
        """
        # Align lengths
        min_len = min(len(x), len(y))
        x = x[-min_len:]
        y = y[-min_len:]
        
        if min_len < 2:
            return 0.0
        
        # Calculate means
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        
        # Calculate correlation
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator_x = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        denominator_y = sum((y[i] - y_mean) ** 2 for i in range(len(y)))
        
        if denominator_x == 0 or denominator_y == 0:
            return 0.0
        
        denominator = (denominator_x * denominator_y) ** 0.5
        
        correlation = numerator / denominator
        
        return correlation
    
    def should_add_symbol(
        self,
        new_symbol: str,
        existing_symbols: List[str],
        price_history: Dict[str, List[float]]
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Determine if adding a new symbol would maintain good diversification.
        
        Returns:
            (allowed, reason, details)
        """
        if not existing_symbols:
            return True, "First symbol in portfolio", {}
        
        # Check correlation with each existing symbol
        correlations = []
        for symbol in existing_symbols:
            prices1 = price_history.get(new_symbol, [])
            prices2 = price_history.get(symbol, [])
            
            if prices1 and prices2:
                corr = self._pearson_correlation(prices1, prices2)
                correlations.append({
                    'with_symbol': symbol,
                    'correlation': corr
                })
        
        if not correlations:
            return True, "No price history available for comparison", {}
        
        # Check if any correlation is too high
        max_corr = max(abs(c['correlation']) for c in correlations)
        
        if max_corr >= self.high_correlation_threshold:
            highest = max(correlations, key=lambda c: abs(c['correlation']))
            return False, f"High correlation ({highest['correlation']:.2f}) with {highest['with_symbol']}", {'correlations': correlations}
        
        return True, f"Good diversification (max corr: {max_corr:.2f})", {'correlations': correlations}


class DiversificationRecommender:
    """
    Recommends assets to improve portfolio diversification.
    """
    
    def __init__(self, analyzer: Optional[PortfolioCorrelationAnalyzer] = None):
        self.analyzer = analyzer or PortfolioCorrelationAnalyzer()
        
        # Asset classes with typical low correlation
        self.asset_classes = {
            'large_cap_crypto': ['BTC/USD', 'ETH/USD'],
            'mid_cap_crypto': ['SOL/USD', 'ADA/USD', 'DOT/USD'],
            'defi_tokens': ['AAVE/USD', 'UNI/USD', 'LINK/USD'],
            'layer2': ['MATIC/USD', 'ARB/USD', 'OP/USD'],
            'tech_stocks': ['AAPL', 'MSFT', 'GOOGL'],
            'etfs': ['SPY', 'QQQ', 'VTI']
        }
    
    def recommend_diversification(
        self,
        current_symbols: List[str],
        price_history: Dict[str, List[float]]
    ) -> List[Dict[str, Any]]:
        """
        Recommend symbols to improve diversification.
        
        Returns:
            List of recommendations [{symbol, reason, asset_class}, ...]
        """
        recommendations = []
        
        # Determine current asset class distribution
        current_classes = set()
        for symbol in current_symbols:
            for asset_class, symbols in self.asset_classes.items():
                if symbol in symbols:
                    current_classes.add(asset_class)
        
        # Recommend symbols from underrepresented classes
        for asset_class, symbols in self.asset_classes.items():
            if asset_class in current_classes:
                continue  # Already have this class
            
            for symbol in symbols:
                if symbol in current_symbols:
                    continue  # Already trading
                
                # Check if it improves diversification
                allowed, reason, _ = self.analyzer.should_add_symbol(
                    symbol,
                    current_symbols,
                    price_history
                )
                
                if allowed:
                    recommendations.append({
                        'symbol': symbol,
                        'asset_class': asset_class,
                        'reason': f"Add {asset_class} exposure for diversification"
                    })
        
        return recommendations[:5]  # Top 5 recommendations


# =============================================================================
# Testing / Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example portfolio
    active_bots = [
        {'symbol': 'BTC/USD'},
        {'symbol': 'ETH/USD'},
        {'symbol': 'SOL/USD'}
    ]
    
    # Mock price history (30 days)
    # THIS IS FOR UNIT TESTING/DEMONSTRATION ONLY.
    # In production, this data comes from the exchange API (CCXT/Alpaca).
    price_history = {
        'BTC/USD': [100 + i * 0.5 for i in range(30)],  # Uptrend
        'ETH/USD': [50 + i * 0.3 for i in range(30)],   # Similar uptrend (high correlation)
        'SOL/USD': [20 - i * 0.1 for i in range(30)]    # Downtrend (low/negative correlation)
    }
    
    analyzer = PortfolioCorrelationAnalyzer()
    div_score, high_corr_pairs, details = analyzer.analyze_portfolio(active_bots, price_history)
    
    print("=== Portfolio Correlation Analysis ===")
    print(f"Diversification Score: {div_score:.2f} (0=bad, 1=perfect)")
    print(f"Total Symbols: {details['total_symbols']}")
    print(f"Highly Correlated Pairs: {details['highly_correlated_pairs']}/{details['total_pairs']}")
    print()
    
    if high_corr_pairs:
        print("High Correlation Pairs:")
        for pair in high_corr_pairs:
            print(f"  {pair['symbol1']} <-> {pair['symbol2']}: {pair['correlation']:.2f} ({pair['severity']})")
    
    print()
    
    # Test diversification recommendations
    recommender = DiversificationRecommender(analyzer)
    recommendations = recommender.recommend_diversification(
        [bot['symbol'] for bot in active_bots],
        price_history
    )
    
    print("=== Diversification Recommendations ===")
    for rec in recommendations:
        print(f"  {rec['symbol']} ({rec['asset_class']}): {rec['reason']}")
