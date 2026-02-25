#!/usr/bin/env python3
"""
Test Order Sizing Validation

Tests that the executor properly validates and rejects invalid order sizes.
This prevents the "0 stock using 0 USD" bug.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any


class TestOrderSizingValidation(unittest.TestCase):
    """Test order size validation in OrderExecutor"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock KrakenClient
        self.mock_kc = Mock()
        self.mock_kc.fetch_ticker_last.return_value = 50000.0
        
    def test_rejects_zero_base_and_zero_quote(self):
        """Executor should reject orders where both size_base and size_quote are 0"""
        from executor import OrderExecutor
        from intelligence_layer import IntelligenceDecision, ExecutionPolicyResult, AllowedAction
        
        executor = OrderExecutor(self.mock_kc)
        
        # Create a decision with invalid order (0 base, 0 quote)
        decision = IntelligenceDecision(
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            proposed_orders=[{
                "side": "buy",
                "type": "limit",
                "price": 50000.0,
                "size_base": 0.0,  # Invalid
                "size_quote": 0.0,  # Invalid
            }],
            execution_policy=ExecutionPolicyResult(
                order_type="limit",
                limit_price=50000.0,
                post_only=False,
                min_cooldown_seconds=10.0,
            ),
            trade_management=Mock(manage_actions=[]),
            risk_flags=[],
            reasons=[],
        )
        
        # Execute decision
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        # Should fail with error
        self.assertFalse(result["success"])
        self.assertTrue(len(result["errors"]) > 0)
        self.assertTrue(any("size" in err.lower() for err in result["errors"]))
        self.assertEqual(len(result["orders_placed"]), 0, "Should not place any orders")
    
    def test_accepts_valid_base_size(self):
        """Executor should accept orders with valid base size"""
        from executor import OrderExecutor
        from intelligence_layer import IntelligenceDecision, ExecutionPolicyResult, AllowedAction
        
        executor = OrderExecutor(self.mock_kc)
        
        # Create a decision with valid order
        decision = IntelligenceDecision(
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            proposed_orders=[{
                "side": "buy",
                "type": "limit",
                "price": 50000.0,
                "size_base": 0.001,  # Valid
                "size_quote": 0.0,   # OK since base is provided
            }],
            execution_policy=ExecutionPolicyResult(
                order_type="limit",
                limit_price=50000.0,
                post_only=False,
                min_cooldown_seconds=10.0,
            ),
            trade_management=Mock(manage_actions=[]),
            risk_flags=[],
            reasons=[],
        )
        
        # Execute decision (dry run)
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        # Should succeed
        self.assertTrue(result["success"])
        self.assertEqual(len(result["orders_placed"]), 1, "Should place one order")
    
    def test_accepts_valid_quote_size(self):
        """Executor should accept orders with valid quote size"""
        from executor import OrderExecutor
        from intelligence_layer import IntelligenceDecision, ExecutionPolicyResult, AllowedAction
        
        executor = OrderExecutor(self.mock_kc)
        
        # Create a decision with valid order
        decision = IntelligenceDecision(
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            proposed_orders=[{
                "side": "buy",
                "type": "limit",
                "price": 50000.0,
                "size_base": 0.0,    # OK since quote is provided
                "size_quote": 50.0,  # Valid
            }],
            execution_policy=ExecutionPolicyResult(
                order_type="limit",
                limit_price=50000.0,
                post_only=False,
                min_cooldown_seconds=10.0,
            ),
            trade_management=Mock(manage_actions=[]),
            risk_flags=[],
            reasons=[],
        )
        
        # Execute decision (dry run)
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        # Should succeed
        self.assertTrue(result["success"])
        self.assertEqual(len(result["orders_placed"]), 1, "Should place one order")
    
    def test_rejects_negative_sizes(self):
        """Executor should reject orders with negative sizes"""
        from executor import OrderExecutor
        from intelligence_layer import IntelligenceDecision, ExecutionPolicyResult, AllowedAction
        
        executor = OrderExecutor(self.mock_kc)
        
        # Create a decision with invalid order (negative size)
        decision = IntelligenceDecision(
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            proposed_orders=[{
                "side": "buy",
                "type": "limit",
                "price": 50000.0,
                "size_base": -0.001,  # Invalid
                "size_quote": 0.0,
            }],
            execution_policy=ExecutionPolicyResult(
                order_type="limit",
                limit_price=50000.0,
                post_only=False,
                min_cooldown_seconds=10.0,
            ),
            trade_management=Mock(manage_actions=[]),
            risk_flags=[],
            reasons=[],
        )
        
        # Execute decision
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        # Should fail
        self.assertFalse(result["success"])
        self.assertTrue(any("size" in err.lower() for err in result["errors"]))
    
    def test_limit_order_requires_valid_price(self):
        """Limit orders should require a valid price > 0"""
        from executor import OrderExecutor
        from intelligence_layer import IntelligenceDecision, ExecutionPolicyResult, AllowedAction
        
        executor = OrderExecutor(self.mock_kc)
        
        # Create a decision with limit order but no price
        decision = IntelligenceDecision(
            allowed_actions=AllowedAction.TRADE_ALLOWED,
            proposed_orders=[{
                "side": "buy",
                "type": "limit",
                "price": None,  # Invalid for limit order
                "size_base": 0.001,
                "size_quote": 0.0,
            }],
            execution_policy=ExecutionPolicyResult(
                order_type="limit",
                limit_price=None,  # Also invalid
                post_only=False,
                min_cooldown_seconds=10.0,
            ),
            trade_management=Mock(manage_actions=[]),
            risk_flags=[],
            reasons=[],
        )
        
        # Execute decision
        result = executor.execute_decision(decision, bot_id=1, symbol="BTC/USD", dry_run=True)
        
        # Should fail
        self.assertFalse(result["success"])
        self.assertTrue(any("price" in err.lower() for err in result["errors"]))


class TestOrderSizingInBotManager(unittest.TestCase):
    """Test that bot_manager doesn't generate invalid orders"""
    
    def test_bot_manager_uses_executor(self):
        """Bot manager should use OrderExecutor, not place orders directly"""
        import bot_manager
        import inspect
        
        # Check that bot_manager imports OrderExecutor
        source = inspect.getsource(bot_manager)
        self.assertIn("OrderExecutor", source, "Bot manager should import OrderExecutor")
        
        # Check that bot_manager doesn't call Kraken directly for orders
        # (It can call for data fetching, but not create_order, create_limit_buy, etc.)
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
    # Run tests
    unittest.main(verbosity=2)
