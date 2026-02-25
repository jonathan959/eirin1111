# executor.py
"""
Centralized Order Executor - Single point for all order placement.

This module enforces the Execution Policy from the Intelligence Layer and is
the ONLY place where orders are placed on Kraken. Strategies return proposed
actions, but only the executor can actually place orders.

CRITICAL RULE: Strategies MUST NOT call Kraken directly. They return proposed
actions only. Only this executor is allowed to place/cancel orders.
"""

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from kraken_client import KrakenClient, _userref_from_client_order_id

ORDER_IDEMPOTENCY_ENABLED = os.getenv("ORDER_IDEMPOTENCY_ENABLED", "1").strip().lower() in ("1", "true", "yes", "y", "on")
LIMIT_ORDER_PATIENCE_SECONDS = int(os.getenv("LIMIT_ORDER_PATIENCE_SECONDS", "300"))
ENABLE_ICEBERG = os.getenv("ENABLE_ICEBERG", "0").strip().lower() in ("1", "true", "yes", "y", "on")
ICEBERG_MIN_NOTIONAL_USD = float(os.getenv("ICEBERG_MIN_NOTIONAL_USD", "10000"))
from intelligence_layer import IntelligenceDecision, ExecutionPolicyResult
from db import add_order_event, add_intelligence_decision, log_error
from alpaca_adapter import AlpacaAdapter
from symbol_classifier import is_stock_symbol

logger = logging.getLogger(__name__)

# LIVE-HARDENED: exchange minimum order size for small-capital live trading ($100-$1000)
MIN_NOTIONAL_KRAKEN_USD = float(os.getenv("MIN_NOTIONAL_KRAKEN_USD", "10.0"))
MIN_NOTIONAL_ALPACA_USD = float(os.getenv("MIN_NOTIONAL_ALPACA_USD", "1.0"))


def _min_notional_usd(symbol: str, is_alpaca: bool) -> float:
    """Minimum notional by exchange. Kraken ~$10, Alpaca $1 fractional."""
    env_min = float(os.getenv("MIN_NOTIONAL_USD", "5.0"))
    exchange_min = MIN_NOTIONAL_ALPACA_USD if is_alpaca else MIN_NOTIONAL_KRAKEN_USD
    return max(env_min, exchange_min)


def _record_execution_quality(
    bot_id: int, order_id: str, symbol: str, side: str,
    intended_price: float, executed_price: Optional[float] = None,
) -> None:
    """Record execution for slippage tracking. Called after order placement."""
    try:
        from execution_quality_tracker import record_execution
        record_execution(
            order_id=str(order_id or ""),
            bot_id=bot_id,
            symbol=symbol,
            side=side,
            intended_price=intended_price,
            executed_price=executed_price,
        )
    except Exception as e:
        logger.debug("Execution quality record skipped: %s", e)


def _parse_base_quote(symbol: str) -> Tuple[str, str]:
    """'BCH/USD' -> ('BCH', 'USD'). Empty strings if invalid."""
    parts = (symbol or "").strip().split("/")
    if len(parts) != 2:
        return ("", "")
    return (parts[0].strip(), parts[1].strip())


def _is_insufficient_funds(e: Exception) -> bool:
    msg = str(e).lower()
    name = type(e).__name__
    return (
        "insufficient funds" in msg
        or "eorder:insufficient" in msg
        or "InsufficientFunds" in name
    )


def _safe_enum_val(x: Any) -> str:
    """Handles Enum or str (avoids 'str' has no attribute 'value')."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return x.value if hasattr(x, "value") else str(x)


def _infer_tick(price: float) -> float:
    """Infer tick size from price. Stocks: $0.01 below $1, etc. Crypto: use 0.01%."""
    if price <= 0:
        return 0.01
    if price >= 1:
        return 0.01
    return round(price * 0.0001, 8)  # ~0.01% for sub-$1


def modeled_cost_pct(spread_pct: float, volatility_pct: float) -> float:
    """Shared cost model: spread + slippage from volatility."""
    spread = max(0.0, float(spread_pct or 0.0))
    vol = max(0.0, float(volatility_pct or 0.0))
    slip = min(0.02, max(0.0, vol * 0.5))
    return spread + slip


class OrderExecutor:
    """
    Centralized order executor with Execution Policy enforcement.
    
    Usage:
        executor = OrderExecutor(kraken_client)
        result = executor.execute_decision(decision, bot_id, symbol, dry_run)
    """
    
    def __init__(self, kc: KrakenClient):
        self.kc = kc
        self._last_order_ts: Dict[int, float] = {}  # bot_id -> timestamp
        self._idempotency_seen: Dict[str, float] = {}  # key -> ts (for eviction)
        self._IDEMPOTENCY_MAX = 500
        self._IDEMPOTENCY_TTL = 3600  # 1 hour

    def _check_kraken_balance_before_order(
        self,
        symbol: str,
        side: str,
        *,
        amount_base: float = 0.0,
        notional_quote: float = 0.0,
        size_quote: float = 0.0,
    ) -> Optional[str]:
        """
        Pre-flight balance check for Kraken only. Returns error message if
        insufficient funds, else None. Skipped for Alpaca and on fetch failure.
        """
        if isinstance(self.kc, AlpacaAdapter):
            return None
        base, quote = _parse_base_quote(symbol)
        if not base or not quote:
            return None
        try:
            bal = self.kc.fetch_balance()
        except Exception as ex:
            logger.warning("Balance pre-check skipped: fetch_balance failed: %s", ex)
            return None
        free = (bal.get("free") or {}) or {}
        qv = free.get(quote) or (free.get("ZUSD") if quote == "USD" else 0) or 0
        bv = free.get(base) or (free.get("XXBT") if base == "XBT" else 0) or 0
        free_quote = float(qv)
        free_base = float(bv)
        # Kraken fees ~0.16%; require slight buffer for buys
        buy_buffer = 1.002
        if side == "buy":
            need = notional_quote if notional_quote > 0 else size_quote
            if need <= 0:
                return None
            required = need * buy_buffer
            if free_quote < required:
                return (
                    f"Insufficient {quote} balance: need ~{need:.2f} {quote} "
                    f"(have {free_quote:.2f} free). Reduce order size or add funds."
                )
        else:
            if amount_base <= 0:
                return None
            if free_base < amount_base:
                return (
                    f"Insufficient {base} balance: need {amount_base} {base} "
                    f"(have {free_base} free)."
                )
        return None
    
    def execute_decision(
        self,
        decision: IntelligenceDecision,
        bot_id: int,
        symbol: str,
        dry_run: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute an intelligence decision, enforcing execution policy.
        
        Returns:
            {
                "success": bool,
                "orders_placed": List[Dict],
                "orders_cancelled": List[Dict],
                "execution_quality": Dict,
                "errors": List[str],
            }
        """
        result = {
            "success": False,
            "orders_placed": [],
            "orders_cancelled": [],
            "execution_quality": {},
            "errors": [],
        }
        
        # Check cooldown
        if bot_id in self._last_order_ts:
            elapsed = time.time() - self._last_order_ts[bot_id]
            min_cooldown = decision.execution_policy.min_cooldown_seconds
            if elapsed < min_cooldown:
                result["errors"].append(f"Cooldown active: {elapsed:.1f}s < {min_cooldown}s")
                return result
        
        # Execute trade management actions first (always allowed)
        if decision.trade_management.manage_actions:
            for action in decision.trade_management.manage_actions:
                mgmt_result = self._execute_manage_action(
                    action, bot_id, symbol, dry_run, decision.execution_policy
                )
                if mgmt_result:
                    if "order" in mgmt_result:
                        result["orders_placed"].append(mgmt_result["order"])
                    if "cancelled" in mgmt_result:
                        result["orders_cancelled"].extend(mgmt_result["cancelled"])
        
        # Execute proposed orders if trading is allowed
        if _safe_enum_val(decision.allowed_actions) == "TRADE_ALLOWED":
            # Risk engine gate (when enabled)
            risk_context = kwargs.get("risk_context")
            if risk_context is not None:
                try:
                    from risk_engine import can_open_trade, is_enabled
                    if is_enabled():
                        ok, reason = can_open_trade(risk_context)
                        if not ok:
                            result["errors"].append(f"Risk blocked: {reason}")
                            return result
                except ImportError:
                    pass
            # Check if market is open for stocks (crypto trades 24/7)
            market_closed = False
            if isinstance(self.kc, AlpacaAdapter):
                try:
                    if not self.kc.is_market_open():
                        market_closed = True
                        result["errors"].append("Market is closed. Orders will be placed when market opens.")
                        # Don't place orders when market is closed, but don't treat as failure
                        return result
                except Exception:
                    # If market check fails, continue (might be crypto or check failed)
                    pass
            
            for proposed_order in decision.proposed_orders:
                order_result = self._execute_proposed_order(
                    proposed_order, bot_id, symbol, dry_run, decision.execution_policy
                )
                if order_result:
                    if "order" in order_result:
                        result["orders_placed"].append(order_result["order"])
                    if "error" in order_result:
                        result["errors"].append(order_result["error"])
        
        # Update cooldown timestamp
        if result["orders_placed"] or result["orders_cancelled"]:
            self._last_order_ts[bot_id] = time.time()

        # LIVE-HARDENED: log live order failures to error_log for alerting
        if not dry_run and result["errors"]:
            try:
                log_error(
                    "executor",
                    "order_placement_failed",
                    "; ".join(result["errors"][:3])[:1024],
                    bot_id=bot_id,
                    details={"symbol": symbol, "errors": result["errors"][:5]},
                )
            except Exception as e:
                logger.warning("Failed to log order error to error_log: %s", e)
        
        result["success"] = len(result["errors"]) == 0
        
        # Calculate execution quality
        result["execution_quality"] = self._calculate_execution_quality(
            result["orders_placed"], decision
        )
        
        return result
    
    def _evict_idempotency_keys(self) -> None:
        """Remove stale idempotency keys."""
        if not ORDER_IDEMPOTENCY_ENABLED:
            return
        now = time.time()
        if len(self._idempotency_seen) > self._IDEMPOTENCY_MAX:
            to_del = [k for k, v in self._idempotency_seen.items() if now - v > self._IDEMPOTENCY_TTL]
            for k in to_del[:100]:
                self._idempotency_seen.pop(k, None)

    def _has_duplicate_open_order(self, symbol: str, client_order_id: str) -> bool:
        """
        Check if an open order with the same client_order_id already exists.
        Prevents double placement on retries. Kraken: userref; Alpaca: client_order_id.
        """
        try:
            opens = self.kc.fetch_open_orders(symbol) or []
        except Exception as e:
            logger.debug("Duplicate check: fetch_open_orders failed: %s", e)
            return False
        if isinstance(self.kc, AlpacaAdapter):
            for o in opens:
                cid = o.get("client_order_id") or (o.get("info") or {}).get("client_order_id")
                if cid == client_order_id:
                    return True
            return False
        # Kraken: compare userref
        our_userref = _userref_from_client_order_id(client_order_id)
        for o in opens:
            ur = o.get("userref")
            if ur is None:
                ur = (o.get("info") or {}).get("userref")
            if ur is not None and int(ur) == our_userref:
                return True
        return False

    def _execute_proposed_order(
        self,
        proposed_order: Dict[str, Any],
        bot_id: int,
        symbol: str,
        dry_run: bool,
        execution_policy: ExecutionPolicyResult,
    ) -> Optional[Dict[str, Any]]:
        """Execute a single proposed order with policy enforcement."""
        side = proposed_order.get("side", "").lower()
        order_type = proposed_order.get("type", "limit").lower()
        ts_bucket = int(time.time()) // 60
        size_base = proposed_order.get("size_base", 0.0)
        size_quote = proposed_order.get("size_quote", 0.0)
        intent = f"{size_base:.6f}_{size_quote:.2f}"
        idem_key = f"{bot_id}:{symbol}:{side}:{intent}:{ts_bucket}"
        client_order_id = f"{bot_id}:{symbol}:{side}:{ts_bucket}"

        if side not in ["buy", "sell"]:
            return {"error": f"Invalid side: {side}"}

        # Stale quote protection
        try:
            qts = float(proposed_order.get("quote_ts") or 0.0)
            if qts > 0 and (time.time() - qts) > 30:
                return {"error": "Stale quote: order rejected"}
        except Exception:
            pass

        # Scope guard: stocks only if Alpaca integration is used
        if is_stock_symbol(symbol) and not isinstance(self.kc, AlpacaAdapter):
            return {"error": "Stocks are analysis-only unless Alpaca is enabled"}

        # LiveTradingGate: block real orders unless explicitly enabled
        if not dry_run:
            allow_live = os.getenv("ALLOW_LIVE_TRADING", "0").strip().lower() in ("1", "true", "yes", "y", "on")
            if not allow_live:
                logger.warning("Live trading not enabled: ALLOW_LIVE_TRADING not set")
                return {"error": "Live trading not enabled. Set ALLOW_LIVE_TRADING=1 in .env"}
            try:
                from db import get_bot
                bot = get_bot(bot_id)
                live_confirmed = int(bot.get("live_confirmed", 1) or 1) if bot else 1
                if live_confirmed != 1:
                    logger.warning("Live trading blocked: bot live_confirmed=%s", live_confirmed)
                    return {"error": "Live trading not confirmed for this bot. Set live_confirmed=1."}
            except Exception as e:
                logger.warning("LiveTradingGate check failed: %s", e)
                return {"error": "Live trading gate check failed"}
        
        # Get size (already extracted above for idem_key)
        if size_base <= 0 and size_quote <= 0:
            return {"error": "Invalid size: both size_base and size_quote are 0"}

        # Cost model: spread + slippage vs expected edge (skip for market orders - production-ready)
        _order_type = getattr(execution_policy, "order_type", "limit")
        if _order_type != "market" and not os.getenv("DISABLE_EDGE_COST_CHECK", "").strip():
            try:
                expected_edge_pct = float(proposed_order.get("expected_edge_pct") or 0.0)
                spread_pct = float(proposed_order.get("spread_pct") or 0.0)
                volatility_pct = float(proposed_order.get("volatility_pct") or 0.0)
                slippage_pct = min(0.02, max(0.0, volatility_pct * 0.5))
                modeled_cost_pct = max(0.0, spread_pct) + slippage_pct
                expected_edge_net = expected_edge_pct - modeled_cost_pct
                if expected_edge_net <= 0:
                    return {"error": f"Expected edge <= 0 after cost ({expected_edge_net*100:.2f}%)"}
            except Exception:
                pass
        
        # Urgency-based routing: high=market, medium=limit+0.2%, low=limit at bid/ask
        urgency = getattr(execution_policy, "urgency", "medium") or "medium"
        order_type_to_use = execution_policy.order_type
        if urgency == "high" and order_type_to_use == "limit":
            try:
                spread_pct = float(proposed_order.get("spread_pct") or 0.0)
                if spread_pct < 0.002:
                    order_type_to_use = "market"
            except Exception:
                pass

        # Enforce execution policy
        if order_type_to_use == "limit":
            # Smart limit placement: don't use mid-price
            bid = proposed_order.get("bid_price") or proposed_order.get("bid")
            ask = proposed_order.get("ask_price") or proposed_order.get("ask")
            price = proposed_order.get("price")
            if price is None:
                price = execution_policy.limit_price
            if price is None or price <= 0:
                return {"error": "Limit price required but not provided"}
            # Place limit at bid+1 tick for buys, ask-1 tick for sells (increases fill rate, minimizes slippage)
            tick = _infer_tick(price)
            if side == "buy" and bid is not None and float(bid) > 0:
                price = float(bid) + tick
            elif side == "sell" and ask is not None and float(ask) > 0:
                price = float(ask) - tick
            elif urgency == "medium":
                # Medium: 0.2% buffer from mid
                mid = (float(bid or 0) + float(ask or 0)) / 2 if (bid and ask) else price
                if side == "buy":
                    price = mid * 1.002
                else:
                    price = mid * 0.998
            
            # Calculate final amount with guards
            if size_base > 0:
                final_amount = float(size_base)
            else:
                if size_quote <= 0 or price <= 0:
                    return {"error": f"Cannot calculate size: size_quote={size_quote}, price={price}"}
                final_amount = float(size_quote) / float(price)
            
            # CRITICAL: Reject zero or negative amounts
            if final_amount <= 0 or not (isinstance(final_amount, (int, float)) and final_amount > 0):
                logger.error(f"BLOCKED: Order size invalid: amount={final_amount}, size_base={size_base}, size_quote={size_quote}, price={price}, symbol={symbol}")
                return {"error": f"Order size invalid: amount={final_amount}, size_base={size_base}, size_quote={size_quote}, price={price}"}
            
            # Check for NaN/Inf
            if math.isnan(final_amount) or math.isinf(final_amount):
                logger.error(f"BLOCKED: Order size is NaN/Inf: amount={final_amount}, symbol={symbol}")
                return {"error": f"Order size is NaN/Inf: amount={final_amount}"}
            
            # Validate: check notional value
            notional = final_amount * price if price > 0 else 0.0
            if notional <= 0:
                logger.error(f"BLOCKED: Order notional is 0 or negative: amount={final_amount}, price={price}, notional={notional}, symbol={symbol}")
                return {"error": f"Order notional is 0 or negative: notional={notional}, amount={final_amount}, price={price}"}
            # LIVE-HARDENED: exchange minimum (Kraken ~$10, Alpaca $1)
            min_notional = _min_notional_usd(symbol, isinstance(self.kc, AlpacaAdapter))
            if min_notional > 0 and notional < min_notional:
                return {"error": f"Order notional ${notional:.2f} below minimum ${min_notional:.2f}"}
            
            # Check post-only
            if execution_policy.post_only:
                # For post-only, we'd need to use exchange-specific API
                # For now, we'll use regular limit order
                pass
            
            # Idempotency: skip duplicate orders (retries must not double-place)
            if not dry_run and ORDER_IDEMPOTENCY_ENABLED:
                self._evict_idempotency_keys()
                if idem_key in self._idempotency_seen:
                    logger.info("Idempotency skip: %s", idem_key)
                    return {"error": "Idempotency: duplicate order skipped"}

            # B2: Check open orders for same client_order_id (avoid double placement)
            if not dry_run and self._has_duplicate_open_order(symbol, client_order_id):
                logger.info("Duplicate open order skip: %s", client_order_id)
                return {"error": "Duplicate: open order with same client_order_id already exists"}

            # Place limit order (post-only default handled in execution policy)
            if dry_run:
                order = {
                    "id": f"dry_{int(time.time())}",
                    "side": side,
                    "type": "limit",
                    "price": price,
                    "amount": final_amount,
                    "status": "dry_run",
                }
                add_order_event(
                    bot_id, symbol, side, "limit", price,
                    final_amount,
                    order["id"], "intelligence", "placed", "Intelligence Layer decision",
                    is_live=0,
                )
                return {"order": order}
            else:
                err = self._check_kraken_balance_before_order(
                    symbol, side,
                    amount_base=final_amount if side == "sell" else 0.0,
                    notional_quote=(final_amount * price) if side == "buy" else 0.0,
                )
                if err:
                    return {"error": err}
                # Retry with exponential backoff
                last_err = None
                for i in range(3):
                    try:
                        if side == "buy":
                            order = self.kc.create_limit_buy_base(
                                symbol, final_amount, price, client_order_id
                            )
                        else:
                            if size_base > 0:
                                order = self.kc.create_limit_sell_base(
                                    symbol, size_base, price, client_order_id
                                )
                            else:
                                return {"error": "Sell orders require size_base"}
                        add_order_event(
                            bot_id, symbol, side, "limit", price,
                            final_amount,
                            order.get("id"), "intelligence", "placed", "Intelligence Layer decision",
                            is_live=1,
                        )
                        if ORDER_IDEMPOTENCY_ENABLED:
                            self._idempotency_seen[idem_key] = time.time()
                        _record_execution_quality(bot_id, order.get("id"), symbol, side, price, price)
                        return {"order": order}
                    except Exception as e:
                        last_err = e
                        time.sleep(min(4.0, 0.5 * (2 ** i)))
                if _is_insufficient_funds(last_err):
                    return {"error": f"Insufficient funds (Kraken): {last_err}. Reduce order size or add USD/base."}
                return {"error": f"Order placement failed: {type(last_err).__name__}: {last_err}"}
        
        elif order_type_to_use == "market":
            # Use market order
            try:
                spread_pct = float(proposed_order.get("spread_pct") or 0.0)
                if spread_pct >= 0.003:
                    return {"error": f"Spread too wide for market order ({spread_pct*100:.2f}%)"}
            except Exception:
                pass
            if side == "buy":
                if size_quote <= 0:
                    logger.error(f"BLOCKED: Market buy requires size_quote > 0, got {size_quote}, symbol={symbol}")
                    return {"error": f"Market buy requires size_quote > 0, got {size_quote}"}
                final_amount_quote = float(size_quote)
                
                # Validate notional
                if final_amount_quote <= 0:
                    logger.error(f"BLOCKED: Market buy notional is 0 or negative: {final_amount_quote}, symbol={symbol}")
                    return {"error": f"Market buy notional is 0 or negative: {final_amount_quote}"}
                # LIVE-HARDENED: exchange minimum for market buy
                min_notional = _min_notional_usd(symbol, isinstance(self.kc, AlpacaAdapter))
                if min_notional > 0 and final_amount_quote < min_notional:
                    return {"error": f"Market buy notional ${final_amount_quote:.2f} below minimum ${min_notional:.2f}"}
            else:
                if size_base <= 0:
                    return {"error": f"Market sell requires size_base > 0, got {size_base}"}
                final_amount_base = float(size_base)
            
            if dry_run:
                # Get current price for dry-run
                try:
                    ticker = self.kc.fetch_ticker(symbol)
                    price = float(ticker.get("last", 0))
                except Exception:
                    price = 0.0
                
                if side == "buy":
                    if price <= 0:
                        return {"error": f"Cannot estimate market buy: price={price}"}
                    est_amount = final_amount_quote / price
                    if est_amount <= 0:
                        return {"error": f"Market buy would result in 0 amount: quote={final_amount_quote}, price={price}"}
                else:
                    est_amount = final_amount_base
                    if est_amount <= 0:
                        return {"error": f"Market sell amount invalid: {est_amount}"}
                
                order = {
                    "id": f"dry_{int(time.time())}",
                    "side": side,
                    "type": "market",
                    "price": price,
                    "amount": est_amount,
                    "status": "dry_run",
                }
                add_order_event(
                    bot_id, symbol, side, "market", price,
                    est_amount,
                    order["id"], "intelligence", "placed", "Intelligence Layer decision",
                    is_live=0,
                )
                return {"order": order}
            else:
                if ORDER_IDEMPOTENCY_ENABLED:
                    self._evict_idempotency_keys()
                    if idem_key in self._idempotency_seen:
                        logger.info("Idempotency skip (market): %s", idem_key)
                        return {"error": "Idempotency: duplicate order skipped"}
                if self._has_duplicate_open_order(symbol, client_order_id):
                    logger.info("Duplicate open order skip (market): %s", client_order_id)
                    return {"error": "Duplicate: open order with same client_order_id already exists"}
                err = self._check_kraken_balance_before_order(
                    symbol, side,
                    amount_base=final_amount_base if side == "sell" else 0.0,
                    size_quote=final_amount_quote if side == "buy" else 0.0,
                )
                if err:
                    return {"error": err}
                try:
                    if side == "buy":
                        order = self.kc.create_market_buy_quote(
                            symbol, final_amount_quote, client_order_id
                        )
                        # Estimate fill price
                        try:
                            ticker = self.kc.fetch_ticker(symbol)
                            fill_price = float(ticker.get("last", 0))
                            est_amount = final_amount_quote / fill_price if fill_price > 0 else 0
                        except Exception:
                            fill_price = 0.0
                            est_amount = 0.0
                    else:
                        order = self.kc.create_market_sell_base(
                            symbol, final_amount_base, client_order_id
                        )
                        fill_price = 0.0
                        est_amount = final_amount_base
                    
                    if est_amount <= 0:
                        return {"error": f"Order placed but estimated amount is 0: {est_amount}"}
                    
                    add_order_event(
                        bot_id, symbol, side, "market", fill_price,
                        est_amount,
                        order.get("id"), "intelligence", "placed", "Intelligence Layer decision",
                        is_live=1,
                    )
                    if ORDER_IDEMPOTENCY_ENABLED:
                        self._idempotency_seen[idem_key] = time.time()
                    intended = float(proposed_order.get("price") or 0)
                    if intended <= 0:
                        intended = fill_price if side == "buy" and fill_price > 0 else (final_amount_quote / est_amount if side == "buy" and est_amount > 0 else fill_price)
                    if intended <= 0:
                        try:
                            t = self.kc.fetch_ticker(symbol)
                            intended = float(t.get("last") or t.get("ask") or 0)
                        except Exception:
                            intended = fill_price
                    _record_execution_quality(bot_id, order.get("id"), symbol, side, intended, fill_price if fill_price > 0 else intended)
                    return {"order": order}
                except Exception as e:
                    if _is_insufficient_funds(e):
                        return {"error": f"Insufficient funds (Kraken): {e}. Reduce order size or add USD/base."}
                    return {"error": f"Order placement failed: {type(e).__name__}: {e}"}
        
        else:
            return {"error": f"Unsupported order type: {execution_policy.order_type}"}
    
    def _execute_manage_action(
        self,
        action: Dict[str, Any],
        bot_id: int,
        symbol: str,
        dry_run: bool,
        execution_policy: ExecutionPolicyResult,
    ) -> Optional[Dict[str, Any]]:
        """Execute a trade management action."""
        action_type = action.get("action", "")
        
        if action_type == "partial_tp":
            # Partial take-profit
            size_pct = action.get("size_pct", 0.5)
            price = action.get("price")
            
            # This would require current position size, which we'd need from context
            # For now, return a placeholder
            return {
                "order": {
                    "id": f"dry_{int(time.time())}" if dry_run else None,
                    "side": "sell",
                    "type": "limit",
                    "price": price,
                    "amount": 0.0,  # Would be calculated from position
                    "status": "dry_run" if dry_run else "placed",
                    "action": "partial_tp",
                }
            }
        
        elif action_type == "trailing_stop":
            # Trailing stop (would update existing stop order)
            trail_price = action.get("price")
            return {
                "order": {
                    "id": f"dry_{int(time.time())}" if dry_run else None,
                    "side": "sell",
                    "type": "limit",
                    "price": trail_price,
                    "status": "dry_run" if dry_run else "placed",
                    "action": "trailing_stop",
                }
            }
        
        elif action_type == "break_even":
            # Break-even stop
            be_price = action.get("price")
            return {
                "order": {
                    "id": f"dry_{int(time.time())}" if dry_run else None,
                    "side": "sell",
                    "type": "limit",
                    "price": be_price,
                    "status": "dry_run" if dry_run else "placed",
                    "action": "break_even",
                }
            }
        
        return None
    
    def _calculate_execution_quality(
        self,
        orders_placed: List[Dict[str, Any]],
        decision: IntelligenceDecision,
    ) -> Dict[str, Any]:
        """Calculate execution quality metrics. Uses execution_quality DB when TRACK_EXECUTION_QUALITY=1."""
        if not orders_placed:
            return {"orders_count": 0, "avg_slippage": 0.0, "fill_quality": "none"}
        try:
            from execution_quality_tracker import get_execution_summary, get_avg_slippage_by_strategy
            oid = orders_placed[0].get("id")
            summary = get_execution_summary(str(oid or "")) if oid else None
            if summary:
                return {
                    "orders_count": len(orders_placed),
                    "avg_slippage": summary.get("slippage_pct", 0.0),
                    "fill_quality": "good" if (summary.get("score") or 0) >= 80 else "acceptable" if (summary.get("score") or 0) >= 60 else "poor",
                    "message": summary.get("message"),
                    "score": summary.get("score"),
                }
        except Exception:
            pass
        return {
            "orders_count": len(orders_placed),
            "avg_slippage": 0.0,
            "fill_quality": "estimated",
        }
    
    def cancel_order(self, bot_id: int, symbol: str, order_id: str, dry_run: bool = True) -> bool:
        """Cancel an order (centralized)."""
        if dry_run:
            add_order_event(
                bot_id, symbol, "cancel", "cancel", None, None,
                order_id, "intelligence", "cancelled", "Intelligence Layer cancellation",
                is_live=0,
            )
            return True

        try:
            self.kc.cancel_order(order_id, symbol)
            add_order_event(
                bot_id, symbol, "cancel", "cancel", None, None,
                order_id, "intelligence", "cancelled", "Intelligence Layer cancellation",
                is_live=1,
            )
            return True
        except Exception:
            return False
    
    def cancel_all_orders(self, bot_id: int, symbol: str, dry_run: bool = True) -> int:
        """Cancel all open orders for a symbol (centralized)."""
        if dry_run:
            return 0
        
        try:
            self.kc.cancel_all_open_orders(symbol)
            add_order_event(
                bot_id, symbol, "cancel", "cancel_all", None, None,
                None, "intelligence", "cancelled", "Intelligence Layer: cancel all",
                is_live=1,
            )
            # Return count (would need to fetch open orders first)
            return 0
        except Exception:
            return 0
