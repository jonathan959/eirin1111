#!/usr/bin/env python3
"""
Test Order Sizing Validation

Tests that the executor properly validates and rejects invalid order sizes.
This prevents the "0 stock using 0 USD" bug.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


def _make_decision(proposed_orders, order_type="limit", limit_price=50000.0):
    """Helper to build a minimal IntelligenceDecision for testing."""
    from intelligence_layer import (
        IntelligenceDecision, ExecutionPolicyResult, AllowedAction,
        DataValidityResult, MarketSafetyResult, RegimeDetectionResult,
        StrategyRoutingResult, PositionSizingResult, TradeManagementResult,
        RegimeType,
    )
    return IntelligenceDecision(
        data_validity=DataValidityResult(data_ok=True),
        market_safety=MarketSafetyResult(
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            risk_budget=1000.0,
        ),
        regime_detection=RegimeDetectionResult(
            regime=RegimeType.RANGE, confidence=0.7, ttl_seconds=300,
        ),
        strategy_routing=StrategyRoutingResult(
            strategy_mode="classic_dca", entry_style="moderate",
            exit_style="fixed_tp",
        ),
        position_sizing=PositionSizingResult(
            base_size=15.0, ladder_steps=1,
            ladder_spacing_pct=1.5, max_adds=3,
        ),
        execution_policy=ExecutionPolicyResult(
            order_type=order_type,
            limit_price=limit_price,
            post_only=False,
            min_cooldown_seconds=10,
        ),
        trade_management=TradeManagementResult(),
        allowed_actions=AllowedAction.TRADE_ALLOWED,
        final_action="ENTER",
        final_reason="test",
        proposed_orders=proposed_orders,
    )


class TestOrderSizingValidation(unittest.TestCase):
    """Test order size validation in OrderExecutor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_kc = Mock()
        self.mock_kc.fetch_ticker_last.return_value = 50000.0
        
    def test_rejects_zero_base_and_zero_quote(self):
        """Executor should reject orders where both size_base and size_quote are 0"""
        from executor import OrderExecutor
        
        executor = OrderExecutor(self.mock_kc)
        decision = _make_decision([{
            "side": "buy", "type": "limit", "price": 50000.0,
            "size_base": 0.0, "size_quote": 0.0,
        }])
        
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        self.assertFalse(result["success"])
        self.assertTrue(len(result["errors"]) > 0)
        self.assertTrue(any("size" in err.lower() for err in result["errors"]))
        self.assertEqual(len(result["orders_placed"]), 0, "Should not place any orders")
    
    def test_accepts_valid_base_size(self):
        """Executor should accept orders with valid base size"""
        from executor import OrderExecutor
        
        executor = OrderExecutor(self.mock_kc)
        decision = _make_decision([{
            "side": "buy", "type": "limit", "price": 50000.0,
            "size_base": 0.001, "size_quote": 0.0,
        }])
        
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["orders_placed"]), 1, "Should place one order")
    
    def test_accepts_valid_quote_size(self):
        """Executor should accept orders with valid quote size"""
        from executor import OrderExecutor
        
        executor = OrderExecutor(self.mock_kc)
        decision = _make_decision([{
            "side": "buy", "type": "limit", "price": 50000.0,
            "size_base": 0.0, "size_quote": 50.0,
        }])
        
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        self.assertTrue(result["success"])
        self.assertEqual(len(result["orders_placed"]), 1, "Should place one order")
    
    def test_rejects_negative_sizes(self):
        """Executor should reject orders with negative sizes"""
        from executor import OrderExecutor
        
        executor = OrderExecutor(self.mock_kc)
        decision = _make_decision([{
            "side": "buy", "type": "limit", "price": 50000.0,
            "size_base": -0.001, "size_quote": 0.0,
        }])
        
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        self.assertFalse(result["success"])
        self.assertTrue(any("size" in err.lower() for err in result["errors"]))
    
    def test_limit_order_requires_valid_price(self):
        """Limit orders should require a valid price > 0"""
        from executor import OrderExecutor
        
        executor = OrderExecutor(self.mock_kc)
        decision = _make_decision(
            [{
                "side": "buy", "type": "limit", "price": None,
                "size_base": 0.001, "size_quote": 0.0,
            }],
            order_type="limit",
            limit_price=None,
        )
        
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        self.assertFalse(result["success"])
        self.assertTrue(any("price" in err.lower() for err in result["errors"]))


class TestOrderSizingInBotManager(unittest.TestCase):
    """Test that bot_manager doesn't generate invalid orders"""
    
    def test_bot_manager_uses_executor(self):
        """Bot manager should use OrderExecutor, not place orders directly"""
        import bot_manager
        import inspect
        
        source = inspect.getsource(bot_manager)
        self.assertIn("OrderExecutor", source, "Bot manager should import OrderExecutor")
        
        has_direct_order = (
            "kc.create_order" in source or
            "kc.create_limit_buy" in source or  
            "kc.create_limit_sell" in source or
            "kc.create_market_buy" in source or
            "kc.create_market_sell" in source
        )
        
        self.assertFalse(has_direct_order, 
                        "Bot manager should not call Kraken order methods directly")


if __name__ == "__main__":
    unittest.main(verbosity=2)
