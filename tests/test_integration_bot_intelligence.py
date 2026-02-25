import sys
import os
sys.path.append(os.getcwd())
import unittest
import time
from unittest.mock import MagicMock, patch
import json

from bot_manager import BotRunner, RuntimeState
from intelligence_layer import IntelligenceLayer, IntelligenceDecision, IntelligenceContext, AllowedAction, DataValidityResult, MarketSafetyResult, RegimeDetectionResult, StrategyRoutingResult, PositionSizingResult, ExecutionPolicyResult, TradeManagementResult, RegimeType

class TestBotIntelligenceIntegration(unittest.TestCase):
    def setUp(self):
        # Mock KrakenClient
        self.mock_kc = MagicMock()
        self.mock_kc.load_markets.return_value = {
            "BTC/USD": {"base": "BTC", "quote": "USD"},
            "XBT/USD": {"base": "BTC", "quote": "USD"}
        }
        self.mock_kc.fetch_ticker_last.return_value = 50000.0
        self.mock_kc.fetch_balance.return_value = {"free": {"USD": 1000.0}, "total": {"USD": 1000.0}}
        self.mock_kc.fetch_ohlcv.return_value = [
            [time.time() * 1000, 50000, 50100, 49900, 50050, 100] for _ in range(300)
        ]

        # Mock DB functions to avoid writing to actual DB
        self.db_patcher = patch('bot_manager.get_bot')
        self.mock_get_bot = self.db_patcher.start()
        self.mock_get_bot.return_value = {
            "id": 1,
            "name": "TestBot",
            "symbol": "BTC/USD",
            "strategy_mode": "smart_dca",
            "poll_seconds": 0, # run fast
            "dry_run": 1,
            "base_quote": 10,
            "safety_quote": 10,
            "enabled": 1,
            "market_type": "crypto"
        }
        
        self.log_patcher = patch('bot_manager.add_log')
        self.mock_add_log = self.log_patcher.start()

        self.intel_log_patcher = patch('bot_manager.add_intelligence_decision')
        self.mock_add_intel = self.intel_log_patcher.start()

        self.regime_snap_patcher = patch('bot_manager.add_regime_snapshot')
        self.mock_add_regime = self.regime_snap_patcher.start()

        self.latest_deal_patcher = patch('bot_manager.latest_open_deal')
        self.mock_latest_deal = self.latest_deal_patcher.start()
        self.mock_latest_deal.return_value = None

        self.open_deal_patcher = patch('bot_manager.open_deal')
        self.mock_open_deal = self.open_deal_patcher.start()
        self.mock_open_deal.return_value = 101

        self.pnl_patcher = patch('bot_manager.pnl_summary')
        self.mock_pnl = self.pnl_patcher.start()
        self.mock_pnl.return_value = {"realized": 0.0}

        self.setting_patcher = patch('bot_manager.get_setting')
        self.mock_get_setting = self.setting_patcher.start()
        self.mock_get_setting.return_value = "0"

        # Initialize BotRunner
        self.runner = BotRunner(1, self.mock_kc)
        
        # Mock IntelligenceLayer
        self.runner.intelligence_layer = MagicMock(spec=IntelligenceLayer)

    def tearDown(self):
        self.db_patcher.stop()
        self.log_patcher.stop()
        self.intel_log_patcher.stop()
        self.regime_snap_patcher.stop()
        self.latest_deal_patcher.stop()
        self.latest_deal_patcher.stop()
        self.open_deal_patcher.stop()
        self.pnl_patcher.stop()
        self.setting_patcher.stop()

    def test_run_loop_calls_evaluate_and_logs(self):
        """Verify _run_loop_multi calls evaluate() and logs the decision."""
        
        # Setup mock decision
        mock_decision = IntelligenceDecision(
            data_validity=DataValidityResult(data_ok=True),
            market_safety=MarketSafetyResult(allowed_actions=AllowedAction.TRADE_ALLOWED, risk_budget=100.0),
            regime_detection=RegimeDetectionResult(regime=RegimeType.BULL, confidence=0.9, ttl_seconds=60),
            strategy_routing=StrategyRoutingResult(strategy_mode="trend_follow", entry_style="aggressive", exit_style="trailing"),
            position_sizing=PositionSizingResult(base_size=20.0, ladder_steps=3, ladder_spacing_pct=0.01, max_adds=3),
            execution_policy=ExecutionPolicyResult(order_type="market"),
            trade_management=TradeManagementResult(),
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            final_action="ENTER",
            final_reason="Test Entry",
            proposed_orders=[{"side": "buy", "amount": 20.0}],
            timestamp=int(time.time())
        )
        self.runner.intelligence_layer.evaluate.return_value = mock_decision

        # We need to break the loop after one iteration
        # _run_loop_multi uses `while not self._stop.is_set():`
        # We can simulate stop set after first check? No, the check happens at start of loop.
        # We can throw an exception from inside the loop to break it, or just set _stop after a short delay in another thread.
        # Easier: Mock `get_bot` to return None on the SECOND call to break the loop naturally (it checks `if not bot: break`).
        
        # First call returns config, second call returns None -> loop ends
        self.mock_get_bot.side_effect = [
            self.mock_get_bot.return_value, # First call at start of function
            self.mock_get_bot.return_value, # First call in loop
            None                            # Second call in loop (simulate bot deletion)
        ]

        # Call the loop
        self.runner._run_loop_multi()

        # Verify evaluate was called
        if not self.runner.intelligence_layer.evaluate.called:
             print("\n!!! DEBUG LOGS CAPTURED !!!")
             for call in self.mock_add_log.call_args_list:
                 print(f"Log: {call}")
        
        self.assertTrue(self.runner.intelligence_layer.evaluate.called)
        
        # Verify decision logging
        self.assertTrue(self.mock_add_intel.called)
        args, _ = self.mock_add_intel.call_args
        # args: (bot_id, ts, symbol, allowed, final_action, final_reason, data_ok, ...)
        self.assertEqual(args[0], 1) # bot_id
        self.assertEqual(args[3], "ENTER") # final_action
        self.assertEqual(args[4], "Test Entry") # final_reason

if __name__ == "__main__":
    unittest.main()
