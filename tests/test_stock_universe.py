import unittest
import os
import sys

# Mocking environment to avoid actual API connections unless needed
os.environ["ALPACA_API_KEY_PAPER"] = "test"
os.environ["ALPACA_API_SECRET_PAPER"] = "test"

sys.path.append(os.getcwd())

from alpaca_client import AlpacaClient

class TestStockUniverse(unittest.TestCase):
    def test_stock_universe_quality(self):
        """
        Verifies that the stock universe builder produces a high-quality list.
        """
        class MockSession:
            def request(self, method, url, *args, **kwargs):
                # Mock "get_assets" response
                if "assets" in url:
                    return MockResponse([
                        {"symbol": "AAPL", "status": "active", "tradable": True, "marginable": True},
                        {"symbol": "JUNK1", "status": "active", "tradable": False}, # Should exclude
                        {"symbol": "INACTIVE", "status": "inactive", "tradable": True}, # Should exclude
                        {"symbol": "TSLA", "status": "active", "tradable": True, "marginable": True},
                    ])
                return MockResponse({})

        class MockResponse:
            def __init__(self, json_data):
                self.json_data = json_data
                self.status_code = 200
            def json(self):
                return self.json_data
            def raise_for_status(self):
                pass

        client = AlpacaClient(mode="paper")
        client.session = MockSession() # Inject mock
        
        # Test method: get_active_assets
        assets = client.get_active_assets()
        
        # Assertions
        symbols = [a['symbol'] for a in assets]
        
        # 1. Junk/Inactive should be gone
        self.assertNotIn("JUNK1", symbols, "Found non-tradable asset")
        self.assertNotIn("INACTIVE", symbols, "Found inactive asset")
        
        # 2. Quality assets present
        self.assertIn("AAPL", symbols)
        self.assertIn("TSLA", symbols)
        
        print(f"\n✅ Universe Filter Logic Passed. Filtered {len(symbols)} valid from mock source.")

if __name__ == "__main__":
    unittest.main()
