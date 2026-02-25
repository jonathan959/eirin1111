"""
Integration tests for symbol routing

Tests that stock symbols are routed to Alpaca and crypto symbols to Kraken
"""

import unittest
from unittest.mock import Mock, patch, MagicMock


class TestSymbolRouting(unittest.TestCase):
    """Test that symbols are routed to correct trading providers"""
    
    @patch('worker_api.intelligence_layer')
    @patch('worker_api.alpaca_paper')
    @patch('worker_api.alpaca_live', None)
    @patch('worker_api.kc')
    def test_stock_symbol_uses_alpaca_not_kraken(self, mock_kraken, mock_alpaca_paper, mock_intelligence):
        """INTC should call Alpaca methods, not Kraken methods"""
        from worker_api import _scan_symbol
        
        # Setup Alpaca mock
        mock_alpaca_paper.get_ohlcv = Mock(return_value=[])
        mock_alpaca_paper.get_latest_quote = Mock(return_value={"price": 50.0})
        
        # Setup intelligence layer mock
        mock_intelligence.generate_recommendation = Mock(return_value={
            "symbol": "INTC",
            "score": 0.5,
            "eligible": True,
            "reasons": [],
            "risk_flags": [],
            "metrics": {},
            "regime": {}
        })
        
        # Call _scan_symbol with stock ticker
        result = _scan_symbol("INTC", "short", {})
        
        # Verify Alpaca was called (4 timeframes)
        self.assertEqual(mock_alpaca_paper.get_ohlcv.call_count, 4, 
                        "Should fetch OHLCV for 4 timeframes")
        mock_alpaca_paper.get_latest_quote.assert_called_with("INTC")
        
        # Verify Kraken methods were NOT called
        mock_kraken.fetch_ohlcv.assert_not_called()
        mock_kraken.fetch_ticker_last.assert_not_called()
        
        # Verify market_type is set
        self.assertEqual(result.get("metrics", {}).get("market_type"), "stock")
    
    @patch('worker_api.intelligence_layer')  
    @patch('worker_api.alpaca_paper', None)
    @patch('worker_api.alpaca_live', None)
    def test_stock_symbol_without_alpaca_returns_error(self, mock_intelligence):
        """When Alpaca is not configured, stock scan should return clear error"""
        from worker_api import _scan_symbol
        
        result = _scan_symbol("INTC", "short", {})
        
        # Should not be eligible
        self.assertFalse(result.get("eligible"), "Should not be eligible without Alpaca")
        
        # Should have clear error message
        reasons = result.get("reasons", [])
        self.assertTrue(any("Stock provider" in str(r) or "Alpaca" in str(r) for r in reasons),
                       f"Should mention stock provider in reasons: {reasons}")
        
        # Should have correct market type
        self.assertEqual(result["metrics"]["market_type"], "stock")
        
        # Should have appropriate risk flag
        risk_flags = result.get("risk_flags", [])
        self.assertIn("NO_STOCK_PROVIDER", risk_flags)
        
        # Intelligence layer should not have been called
        mock_intelligence.generate_recommendation.assert_not_called()
    
    @patch('worker_api.intelligence_layer')
    @patch('worker_api._kraken_ready', return_value=True)
    @patch('worker_api.kc')
    @patch('worker_api._resolve_symbol', side_effect=lambda x: x)
    @patch('worker_api._ohlcv_cached', return_value=[])
    @patch('worker_api._safe_last_price', return_value=50000.0)
    @patch('worker_api._safe_spread_pct', return_value=0.001)
    def test_crypto_symbol_uses_kraken_not_alpaca(
        self, mock_spread, mock_price, mock_ohlcv, mock_resolve, 
        mock_kraken, mock_kraken_ready, mock_intelligence
    ):
        """BTC/USD should call Kraken methods, not Alpaca methods"""
        from worker_api import _scan_symbol
        
        # Setup intelligence layer mock
        mock_intelligence.generate_recommendation = Mock(return_value={
            "symbol": "BTC/USD",
            "score": 0.7,
            "eligible": True,
            "reasons": [],
            "risk_flags": [],
            "metrics": {},
            "regime": {}
        })
        
        result = _scan_symbol("BTC/USD", "short", {})
        
        # Verify Kraken helper functions were called
        mock_ohlcv.assert_called()  # Should be called for each timeframe
        mock_price.assert_called_with("BTC/USD")
        mock_spread.assert_called_with("BTC/USD")
        
        # Verify market_type is set
        self.assertEqual(result.get("metrics", {}).get("market_type"), "crypto")
    
    @patch('worker_api.alpaca_paper')
    @patch('worker_api.alpaca_live', None)
    def test_api_market_ticker_routes_stock_to_alpaca(self, mock_alpaca):
        """API endpoint should route stock ticker requests to Alpaca"""
        from worker_api import api_market_ticker
        
        mock_alpaca.get_latest_quote = Mock(return_value={"price": 155.0, "bid": 154.9, "ask": 155.1})
        
        response = api_market_ticker("AAPL")
        
        # Should call Alpaca
        mock_alpaca.get_latest_quote.assert_called_with("AAPL")
        
        # Should return success
        self.assertTrue(response.body.get("ok") if hasattr(response, 'body') else True)
    
    @patch('worker_api._kraken_ready', return_value=True)
    @patch('worker_api._resolve_symbol', side_effect=lambda x: x)
    @patch('worker_api._markets', return_value={"BTC/USD": {}})
    @patch('worker_api._ticker_cached', return_value={"price": 50000.0})
    def test_api_market_ticker_routes_crypto_to_kraken(
        self, mock_ticker, mock_markets, mock_resolve, mock_ready
    ):
        """API endpoint should route crypto ticker requests to Kraken"""
        from worker_api import api_market_ticker
        
        response = api_market_ticker("BTC/USD")
        
        # Should call Kraken helpers
        mock_ticker.assert_called()
        mock_markets.assert_called()
    
    def test_guardrail_prevents_stock_in_ohlcv_cached(self):
        """_ohlcv_cached should raise ValueError when given a stock symbol"""
        from worker_api import _ohlcv_cached
        
        with self.assertRaises(ValueError) as context:
            _ohlcv_cached("INTC", "1h", 100, 300)
        
        error_msg = str(context.exception)
        self.assertIn("INTC", error_msg)
        self.assertIn("stock", error_msg)
        self.assertIn("_ohlcv_cached", error_msg)
    
    def test_guardrail_prevents_stock_in_safe_last_price(self):
        """_safe_last_price should raise ValueError when given a stock symbol"""
        from worker_api import _safe_last_price
        
        with self.assertRaises(ValueError) as context:
            _safe_last_price("AAPL")
        
        error_msg = str(context.exception)
        self.assertIn("AAPL", error_msg)
        self.assertIn("stock", error_msg)
        self.assertIn("_safe_last_price", error_msg)


class TestMultipleStockSymbols(unittest.TestCase):
    """Test that multiple different stock symbols are handled correctly"""
    
    @patch('worker_api.intelligence_layer')
    @patch('worker_api.alpaca_paper')
    @patch('worker_api.alpaca_live', None)
    def test_various_stock_symbols(self, mock_alpaca, mock_intelligence):
        """Test a variety of stock tickers"""
        from worker_api import _scan_symbol
        
        stock_symbols = ["INTC", "AAPL", "MSFT", "TSLA", "AMD", "NVDA", "META"]
        
        mock_alpaca.get_ohlcv = Mock(return_value=[])
        mock_alpaca.get_latest_quote = Mock(return_value={"price": 100.0})
        mock_intelligence.generate_recommendation = Mock(return_value={
            "symbol": "TEST",
            "score": 0.5,
            "eligible": True,
            "metrics": {},
            "regime": {}
        })
        
        for symbol in stock_symbols:
            with self.subTest(symbol=symbol):
                # Reset mock call counts
                mock_alpaca.get_ohlcv.reset_mock()
                mock_alpaca.get_latest_quote.reset_mock()
                
                result = _scan_symbol(symbol, "short", {})
                
                # Verify Alpaca was called for this symbol
                self.assertGreater(mock_alpaca.get_ohlcv.call_count, 0,
                                  f"{symbol} should use Alpaca for OHLCV")
                mock_alpaca.get_latest_quote.assert_called_with(symbol)
                
                # Verify market type
                self.assertEqual(result.get("metrics", {}).get("market_type"), "stock",
                               f"{symbol} should be classified as stock")


if __name__ == "__main__":
    unittest.main()
