"""
Unit tests for symbol_classifier module

Tests the classification of trading symbols as stocks or cryptocurrencies.
"""

import unittest
from symbol_classifier import classify_symbol, is_stock_symbol, is_crypto_symbol, validate_symbol_type


class TestSymbolClassifier(unittest.TestCase):
    """Test symbol classification functions"""
    
    def test_stock_symbols(self):
        """Test that common stock tickers are classified as stocks"""
        stocks = ["INTC", "AAPL", "MSFT", "TSLA", "AMD", "NVDA", "META", "GOOGL", "AMZN"]
        for symbol in stocks:
            with self.subTest(symbol=symbol):
                self.assertEqual(classify_symbol(symbol), "stock", f"{symbol} should be classified as stock")
                self.assertTrue(is_stock_symbol(symbol), f"is_stock_symbol({symbol}) should be True")
                self.assertFalse(is_crypto_symbol(symbol), f"is_crypto_symbol({symbol}) should be False")
    
    def test_crypto_symbols_with_slash(self):
        """Test that crypto pairs with '/' are classified as crypto"""
        cryptos = ["BTC/USD", "ETH/USD", "SOL/USD", "XBT/USD", "DOGE/USDT", "ADA/EUR"]
        for symbol in cryptos:
            with self.subTest(symbol=symbol):
                self.assertEqual(classify_symbol(symbol), "crypto", f"{symbol} should be classified as crypto")
                self.assertTrue(is_crypto_symbol(symbol), f"is_crypto_symbol({symbol}) should be True")
                self.assertFalse(is_stock_symbol(symbol), f"is_stock_symbol({symbol}) should be False")
    
    def test_long_crypto_strings(self):
        """Test that long strings without slashes are treated as crypto (for normalization)"""
        # These would be normalized to BTC/USD, ETH/USD, etc.
        cryptos = ["BTCUSD", "ETHUSD", "XBTUSD", "SOLUSD"]
        for symbol in cryptos:
            with self.subTest(symbol=symbol):
                self.assertEqual(classify_symbol(symbol), "crypto", f"{symbol} should be classified as crypto")
                self.assertTrue(is_crypto_symbol(symbol))
                self.assertFalse(is_stock_symbol(symbol))
    
    def test_edge_cases(self):
        """Test edge cases in symbol classification"""
        # Empty string defaults to crypto
        self.assertEqual(classify_symbol(""), "crypto")
        
        # Single char - stock (very short)
        self.assertEqual(classify_symbol("F"), "stock")  # Ford
        
        # Exactly 5 chars - still stock
        self.assertEqual(classify_symbol("ORCL"), "stock")  # Oracle (4 chars)
        self.assertEqual(classify_symbol("NFLX"), "stock")  # Netflix (4 chars)
        
        # Exactly 6 chars - crypto
        self.assertEqual(classify_symbol("BTCUSD"), "crypto")  # 6 chars
    
    def test_case_insensitivity(self):
        """Test that classification works regardless of case"""
        self.assertEqual(classify_symbol("intc"), "stock")
        self.assertEqual(classify_symbol("INTC"), "stock")
        self.assertEqual(classify_symbol("Intc"), "stock")
        
        self.assertEqual(classify_symbol("btc/usd"), "crypto")
        self.assertEqual(classify_symbol("BTC/USD"), "crypto")
    
    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly"""
        self.assertEqual(classify_symbol(" INTC "), "stock")
        self.assertEqual(classify_symbol("  BTC/USD  "), "crypto")


class TestValidateSymbolType(unittest.TestCase):
    """Test the validate_symbol_type guardrail function"""
    
    def test_valid_stock_validation(self):
        """Validating a stock as stock should not raise"""
        try:
            validate_symbol_type("INTC", "stock", "test_function")
        except ValueError:
            self.fail("validate_symbol_type raised ValueError unexpectedly for valid stock")
    
    def test_valid_crypto_validation(self):
        """Validating a crypto as crypto should not raise"""
        try:
            validate_symbol_type("BTC/USD", "crypto", "test_function")
        except ValueError:
            self.fail("validate_symbol_type raised ValueError unexpectedly for valid crypto")
    
    def test_invalid_stock_as_crypto_raises(self):
        """Passing a stock to crypto function should raise ValueError"""
        with self.assertRaises(ValueError) as context:
            validate_symbol_type("INTC", "crypto", "kraken_function")
        
        error_msg = str(context.exception)
        self.assertIn("INTC", error_msg)
        self.assertIn("stock", error_msg)
        self.assertIn("crypto-only", error_msg.lower())
        self.assertIn("kraken_function", error_msg)
    
    def test_invalid_crypto_as_stock_raises(self):
        """Passing a crypto to stock function should raise ValueError"""
        with self.assertRaises(ValueError) as context:
            validate_symbol_type("BTC/USD", "stock", "alpaca_function")
        
        error_msg = str(context.exception)
        self.assertIn("BTC/USD", error_msg)
        self.assertIn("crypto", error_msg)
        self.assertIn("stock-only", error_msg.lower())
        self.assertIn("alpaca_function", error_msg)
    
    def test_error_message_provides_guidance(self):
        """Error messages should suggest the correct client to use"""
        with self.assertRaises(ValueError) as context:
            validate_symbol_type("INTC", "crypto", "some_kraken_func")
        
        error_msg = str(context.exception)
        # Should mention AlpacaClient as the correct choice
        self.assertIn("AlpacaClient", error_msg)


if __name__ == "__main__":
    unittest.main()
