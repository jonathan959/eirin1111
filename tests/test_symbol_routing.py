"""
Integration tests for symbol routing

Tests that stock symbols are routed to Alpaca and crypto symbols to Kraken
"""

import sys
import types
import unittest
from unittest.mock import Mock, patch, MagicMock

import pandas as pd


def _install_fake_yfinance_empty():
    """So ``import yfinance`` inside worker_api succeeds and yields no OHLCV rows."""
    fake = types.ModuleType("yfinance")

    class _Ticker:
        def __init__(self, *_a, **_k):
            pass

        def history(self, *_a, **_k):
            return pd.DataFrame()

    fake.Ticker = _Ticker
    return fake


class TestSymbolRouting(unittest.TestCase):
    """Test that symbols are routed to correct trading providers"""
    
    @patch('worker_api._scan_ohlcv_get', return_value=[])
    @patch('worker_api.intelligence_layer')
    @patch('phase2_data_fetcher.fetch_recent_candles')
    @patch('worker_api.alpaca_paper')
    @patch('worker_api.alpaca_live', None)
    @patch('worker_api.kc')
    def test_stock_symbol_uses_alpaca_not_kraken(
        self, mock_kraken, mock_alpaca_paper, mock_fetch, mock_intelligence, mock_scan_get,
    ):
        """INTC should call Alpaca methods, not Kraken methods"""
        fake_yf = _install_fake_yfinance_empty()
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            from worker_api import _scan_symbol

            # Stock path uses phase2_data_fetcher.fetch_recent_candles and client.get_ticker
            mock_fetch.return_value = [[1000000, 45, 46, 44, 45.5, 1e6]] * 100
            mock_alpaca_paper.get_ticker = Mock(return_value={"last": 50.0, "bid": 49.9, "ask": 50.1})

            # Setup intelligence layer mock
            mock_intelligence.generate_recommendation = Mock(return_value={
                "symbol": "INTC",
                "score": 0.5,
                "eligible": True,
                "reasons": [],
                "risk_flags": [],
                "metrics": {"market_type": "stocks"},
                "regime": {}
            })

            # Call _scan_symbol with stock ticker
            result = _scan_symbol("INTC", "short", {})

        # Verify fetch_recent_candles was called (4 timeframes: 1h, 4h, 1d, 1w)
        self.assertGreaterEqual(mock_fetch.call_count, 4,
                        "Should fetch candles for 4 timeframes")
        mock_alpaca_paper.get_ticker.assert_called_with("INTC")

        # Verify Kraken methods were NOT called
        mock_kraken.fetch_ohlcv.assert_not_called()
        if hasattr(mock_kraken, 'fetch_ticker_last'):
            mock_kraken.fetch_ticker_last.assert_not_called()

        # Verify market_type is set
        self.assertIn(result.get("metrics", {}).get("market_type"), ("stock", "stocks"))
    
    @patch('worker_api._scan_ohlcv_get', return_value=[])
    @patch('worker_api.intelligence_layer')
    @patch('phase2_data_fetcher.fetch_recent_candles', return_value=[])
    @patch('worker_api.alpaca_paper', None)
    @patch('worker_api.alpaca_live', None)
    def test_stock_symbol_without_alpaca_returns_error(self, mock_fetch, mock_intelligence, mock_scan_get):
        """When Alpaca is not configured and no candle fallback exists, scan should fail cleanly."""
        mock_intelligence.generate_recommendation = Mock(
            return_value={
                "symbol": "INTC", "score": 0.0, "eligible": False,
                "reasons": [], "risk_flags": [],
                "metrics": {"market_type": "stocks"},
                "regime": {},
            }
        )
        fake_yf = _install_fake_yfinance_empty()
        with patch.dict(sys.modules, {"yfinance": fake_yf}):
            from worker_api import _scan_symbol

            result = _scan_symbol("INTC", "short", {})

        result = _scan_symbol("INTC", "short", {})

        self.assertFalse(result.get("eligible"), "Should not be eligible without data")
        reasons = result.get("reasons", [])
        flags = result.get("risk_flags", [])
        diag = " ".join(str(r) for r in reasons) + " " + " ".join(str(f) for f in flags)
        self.assertTrue(
            any(x in diag for x in ("Insufficient", "Yahoo", "Data fetch", "Alpaca", "EXPLORE_V2", "GATE")),
            f"Unexpected reasons/flags: {reasons!r} / {flags!r}",
        )
        self.assertIn(result.get("metrics", {}).get("market_type"), ("stock", "stocks"))
        self.assertIn("DATA_ERROR", result.get("risk_flags", []))

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

        # api_market_ticker uses client.get_ticker(symbol), not get_latest_quote
        mock_alpaca.get_ticker = Mock(return_value={"last": 155.0, "bid": 154.9, "ask": 155.1})

        response = api_market_ticker("AAPL")

        # Should call Alpaca get_ticker
        mock_alpaca.get_ticker.assert_called_with("AAPL")

        # Should return success (FastAPI returns Response, body is JSON)
        if hasattr(response, 'body'):
            import json
            body = json.loads(response.body) if isinstance(response.body, bytes) else response.body
            self.assertTrue(body.get("ok", False), f"Response: {body}")
    
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

    @patch('worker_api._scan_ohlcv_get', return_value=[])
    @patch('worker_api.intelligence_layer')
    @patch('phase2_data_fetcher.fetch_recent_candles')
    @patch('worker_api.alpaca_paper')
    @patch('worker_api.alpaca_live', None)
    def test_various_stock_symbols(self, mock_alpaca, mock_fetch, mock_intelligence, mock_scan_get):
        """Test a variety of stock tickers"""
        from worker_api import _scan_symbol

        fake_yf = _install_fake_yfinance_empty()
        with patch.dict(sys.modules, {"yfinance": fake_yf}):

            stock_symbols = ["INTC", "AAPL", "MSFT", "TSLA", "AMD", "NVDA", "META"]

            # Stock path uses fetch_recent_candles and client.get_ticker
            mock_fetch.return_value = [[1000000, 95, 96, 94, 95.5, 1e6]] * 100
            mock_alpaca.get_ticker = Mock(return_value={"last": 100.0, "bid": 99.9, "ask": 100.1})
            mock_intelligence.generate_recommendation = Mock(return_value={
                "symbol": "TEST",
                "score": 0.5,
                "eligible": True,
                "metrics": {"market_type": "stocks"},
                "regime": {}
            })

            for symbol in stock_symbols:
                with self.subTest(symbol=symbol):
                    mock_fetch.reset_mock()
                    mock_alpaca.get_ticker.reset_mock()

                    result = _scan_symbol(symbol, "short", {})

                    # Verify fetch_recent_candles was called (stock data path)
                    self.assertGreater(mock_fetch.call_count, 0,
                                      f"{symbol} should use Alpaca/fetcher for OHLCV")
                    mock_alpaca.get_ticker.assert_called_with(symbol)

                    # Verify market type
                    self.assertIn(result.get("metrics", {}).get("market_type"), ("stock", "stocks"),
                                 f"{symbol} should be classified as stock")


if __name__ == "__main__":
    unittest.main()
