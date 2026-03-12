# Execution path: UI → API → bot_manager → risk_check → order

## Flow summary

1. **UI** (templates/bot.html, dashboard) polls `/api/bots` or `/api/bots/{id}` and displays `snapshot`: last_event, decision_action, decision_reason, risk_state, spent_quote, etc.

2. **API** (worker_api.py) exposes `GET /api/bots`, `GET /api/bots/{id}`. Bot status/snapshot comes from `BotManager.snapshot(bot_id)` which returns `BotRunner.snapshot()`.

3. **bot_manager.py** – main loop (`_run_loop`):
   - Loads bot config (base_quote, max_spend_quote, per_symbol_exposure_pct, max_total_exposure_pct, etc.).
   - Gets account (equity, free_usd) and position value.
   - **Risk checks (order: circuit breaker → global exposure → max deals → per-symbol exposure → min free cash → daily loss → spread → open orders):**
     - **Circuit breaker** (risk_circuit_breaker.check_circuit_breakers): daily loss, drawdown, portfolio exposure %, max concurrent deals. Returns (ok, reason).
     - **Global exposure**: `total_exposure_usd / equity >= max_total_exposure_pct` → "Global exposure cap reached."
     - **Max concurrent deals**: `open_positions_count() >= max_concurrent_deals` → "Max concurrent deals reached."
     - **Per-symbol exposure**: `position_value / equity >= per_symbol_exposure_pct` → "Per-symbol exposure cap reached: position is X% (limit Y%)."
     - **Min free cash**: `free_usd / equity <= min_free_cash_pct` → "Minimum free cash reserve reached."
     - **Daily loss limit**: today’s realized + unrealized loss vs equity → "Daily loss limit: pausing THIS bot."
     - **Spread guard**: (ask-bid)/mid >= spread_guard_pct → "Spread too wide."
     - **Open orders**: len(open_orders) > max_open_orders → "Too many open orders."
   - If any `risk_reason` is set: `state.risk_state = risk_reason`, `_log_decision("PAUSE", risk_reason)`, then `continue` (no order).
   - Otherwise strategy/decision layer runs; if decision is ENTER/SCALE_IN, order is placed via executor or `create_market_buy_quote`.

4. **Where exposure caps trigger**
   - **Circuit breaker** (portfolio exposure): in `bot_manager` ~2333–2363; `check_circuit_breakers(..., portfolio_exposure_pct=exp_pct, total_exposure_usd=total_exposure, max_total_exposure_pct=_max_exp, ...)`. Trigger: `exp_pct >= _max_exp`.
   - **Global exposure** (duplicate check in bot_manager): ~2365–2380; `exp_ratio = total_exposure/equity`, trigger: `exp_ratio >= max_total_pct`.
   - **Per-symbol exposure**: ~2392–2401; `symbol_pct = position_value/equity`, trigger: `symbol_pct >= per_symbol_pct`.
   - All use `>=` (no strict `>`). Rounding: float division; no explicit rounding of the ratio before comparison.

5. **Order placement** (after risk checks pass): classic DCA path uses `self.kc.create_market_buy_quote(symbol, eff_size)`; intelligence path uses `self.executor.execute_decision(...)`.

## Fix #4 (exposure clarity)

When any risk gate triggers, `state.risk_gate_detail` is set with a dict:

- `gate`: which gate triggered (e.g. `per_symbol_exposure_cap`, `global_exposure_cap`, `circuit_breaker`, `max_concurrent_deals`, `min_free_cash_reserve`, `daily_loss_limit`, `spread_guard`, `open_orders_limit`).
- Numeric fields: `current_spent_quote`, `max_spend_quote`, `current_exposure_pct`, `per_symbol_exposure_pct`, `total_portfolio_value`, `position_value`, etc., depending on the gate.

Snapshot includes `risk_gate_detail`; the bot page shows "Pause details: Gate: … · Spent: $… · …" when present. Comparisons use `>=` (boundary inclusive).

## Files

| Area              | File                    | Relevant symbols |
|-------------------|-------------------------|------------------|
| UI snapshot       | templates/bot.html      | last_event, decision_reason, risk_state, risk_gate_detail, spent_quote |
| API               | worker_api.py           | list_bots, get_bot, snapshot |
| Risk + execution  | bot_manager.py          | risk_reason, risk_gate_detail, _log_decision, state.risk_state, total_exposure_usd, position_value, equity |
| Circuit breaker   | risk_circuit_breaker.py | check_circuit_breakers, exposure reason string |
| Config            | db.py                   | max_spend_quote, per_symbol_exposure_pct, max_total_exposure_pct |
| Tests             | tests/test_exposure_clarity.py | gate detail structure, >= boundary |
