import unittest
from unittest.mock import MagicMock, patch
import json
import sys
import os

sys.path.append(os.getcwd())

from worker_api import api_create_bot, api_update_bot, Request


class MockRequest:
    def __init__(self, json_data):
        self._json = json_data
    
    async def json(self):
        return self._json


class TestBotCRUD(unittest.IsolatedAsyncioTestCase):
    
    @patch('worker_api.get_bot')
    @patch('worker_api.create_bot')
    @patch('worker_api.bm')
    async def test_create_stock_bot_auto_detect(self, mock_bm, mock_create, mock_get_bot):
        mock_create.return_value = 99
        mock_get_bot.return_value = {"id": 99, "symbol": "INTC", "market_type": "stocks"}
        
        payload = {"symbol": "INTC", "base_quote": 100}
        req = MockRequest(payload)
        
        resp = await api_create_bot(req)
        body = json.loads(resp.body)
        
        self.assertTrue(body["ok"])
        
        args, kwargs = mock_create.call_args
        settings = args[0]
        self.assertEqual(settings["market_type"], "stocks", "INTC should be auto-detected as stock")

    @patch('worker_api.get_bot')
    @patch('worker_api.update_bot')
    @patch('worker_api.bm')
    async def test_update_stock_bot_auto_heal(self, mock_bm, mock_update, mock_get_bot):
        mock_get_bot.return_value = {"id": 16, "symbol": "AAPL", "market_type": "crypto", "enabled": 1}
        mock_update.return_value = None
        
        payload = {"symbol": "AAPL", "base_quote": 50, "market_type": ""}
        req = MockRequest(payload)
        
        await api_update_bot(16, req)
        
        args, kwargs = mock_update.call_args
        settings = args[1]
        self.assertEqual(settings["market_type"], "stocks", "Empty market_type should auto-detect AAPL as stock")

    @patch('worker_api.get_bot')
    @patch('worker_api.create_bot')
    @patch('worker_api.bm')
    async def test_create_override_crypto(self, mock_bm, mock_create, mock_get_bot):
        """Test that we FORCE stock market type for stock-looking symbols even if payload says crypto"""
        mock_create.return_value = 100
        mock_get_bot.return_value = {"id": 100, "symbol": "INTC", "market_type": "stocks"}
        
        payload = {"symbol": "INTC", "base_quote": 100, "market_type": "crypto"}
        req = MockRequest(payload)
        
        await api_create_bot(req)
        
        args, kwargs = mock_create.call_args
        settings = args[0]
        self.assertEqual(settings["market_type"], "stocks", "INTC should forcibly override 'crypto' tag")

if __name__ == "__main__":
    unittest.main()
