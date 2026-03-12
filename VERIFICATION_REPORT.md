# Verification Report: Bad Trades & Unmanaged Positions (VERIFY MODE)

**Goal:** Identify why the bot enters bad trades and why positions exist without bots (e.g. NIGHT/USD). No fixes applied until verified.

---

## 1. Confirmed Issues (Bullet List)

- **Executor ignores `final_action`:** When `allowed_actions == TRADE_ALLOWED` and `final_action == "HOLD"`, the executor still places every order in `proposed_orders`. So a buy can be placed with no entry signal (e.g. neutral indicators) if the multi/intelligence loop appends a buy from `strategy.decide()` to `proposed_orders`.
- **Multi-loop path appends strategy orders regardless of `final_action`:** In `_run_loop` (and equivalent multi path), after `evaluate()` returns, `strategy.decide(ctx)` is called and its `decision.order` is appended to `intel_decision.proposed_orders` even when `intel_decision.final_action == "HOLD"`. The executor then places those orders because it only checks `allowed_actions`, not `final_action`.
- **Bad entries:** Enter can occur when position_size==0 and TRADE_ALLOWED but **no** explicit entry signal (`should_enter`/`should_add`), because the above two points combine: proposed_orders get a buy from strategy, and executor does not gate on final_action.
- **Unmanaged positions (e.g. NIGHT/USD):** Logic to list them exists (`_unmanaged_positions`, GET `/api/unmanaged_positions`). A given position can come from: (a) a deleted bot (symbol no longer in `list_bots()`), (b) manual trade, or (c) symbol normalization mismatch. Evidence for a specific case (e.g. NIGHT/USD at a given time) requires DB/logs (deals, order_events, add_log) and optional diagnostic GET `/api/diag/unmanaged_funding`.
- **Funding / cash competition:** When `funding_mode != 'reserved'`, multiple bots share the same free USD pool; deposits cannot be targeted to one bot. Per-bot reserved funding exists when `funding_mode == 'reserved'`.

---

## 2. Exact Files, Functions, and Line Numbers

### 2.1 Entry decision path (intelligence → executor)

| Location | Purpose |
|----------|--------|
| `intelligence_layer.py` ~468–535 | `evaluate()`: sets `final_action = "ENTER"` only when `strategy_routing.should_enter` (and position_size==0, TRADE_ALLOWED); otherwise HOLD. Explicit entry signals are used. |
| `intelligence_layer.py` ~1307–1317 | `_route_strategy()`: returns `StrategyRoutingResult` with `should_enter`/`should_add`/`signal_reason` from `_compute_entry_signals()`. |
| `intelligence_layer.py` ~1320+ | `_compute_entry_signals()`: crash guard (e.g. 6×1H return ≤ -4%, single 1H ≤ -3%) and per-strategy entry rules. |
| `bot_manager.py` ~2969–2974 | **Bug path:** `strategy.decide(ctx)` result is appended to `intel_decision.proposed_orders` regardless of `intel_decision.final_action`. |
| `bot_manager.py` ~3015–3018 | `executor.execute_decision(intel_decision, ...)` is called with that decision (possibly HOLD + non-empty proposed_orders). |
| `executor.py` ~241–268 | **Bug:** Condition is only `allowed_actions == "TRADE_ALLOWED"` (line 248). No check of `final_action`. Loop at 274–282 places every `proposed_order`. |

### 2.2 Order placement trace

| Location | Purpose |
|----------|--------|
| `bot_manager.py` ~1611 (ENTER), ~1645 (SAFETY_ORDER) | `self.kc.create_market_buy_quote(symbol, ...)` — real exchange order. |
| `executor.py` ~274–282 | Loop over `decision.proposed_orders`; each is executed via `_execute_proposed_order`. |

### 2.3 Unmanaged position reconciliation

| Location | Purpose |
|----------|--------|
| `worker_api.py` ~3494–3528 | `_unmanaged_positions()`: uses `_portfolio_snapshot()` holdings and `list_bots()` symbols; returns holdings with no matching bot symbol. |
| `worker_api.py` ~3532–3540 | GET `/api/unmanaged_positions`. |
| `worker_api.py` ~3542–3575 | GET `/api/diag/unmanaged_funding` (diagnostic): exchange_holdings, bot_symbols, unmanaged_positions, free_usd, funding_note. |

### 2.4 Funding

| Location | Purpose |
|----------|--------|
| `bot_manager.py` (reserved budget) | `_check_reserved_budget`, `add_bot_spend`; when `funding_mode != 'reserved'`, bots share the same free USD. |

---

## 3. Temporary Instrumentation (DEBUG_DECISIONS=1)

- **intelligence_layer.py** (before return of `evaluate()`): logs `DECISION_TRACE` with bot_id, symbol, position_size, allowed_actions, strategy, regime, final_action, final_reason, should_enter, should_add.
- **bot_manager.py** (before ENTER and SAFETY_ORDER `create_market_buy_quote`): logs `ORDER_TRACE` with bot_id, symbol, side/action, quote amount, eff_size, free_quote, ob_reason.
- **executor.py** (when proposed_orders non-empty): logs `EXECUTOR_TRACE` with bot_id, final_action, allowed_actions, proposed_orders_count and note that executor gates only on allowed_actions.

Set in environment: `DEBUG_DECISIONS=1`.

---

## 4. Sample Logs (Bad Behavior)

With `final_action=HOLD` and `proposed_orders` containing a buy (e.g. from strategy.decide in _run_loop), you would see:

```
DECISION_TRACE bot_id=1 symbol=XBT/USD position_size=0 allowed=TRADE_ALLOWED strategy=... regime=... final_action=HOLD reason=... should_enter=False should_add=False
EXECUTOR_TRACE bot_id=1 final_action=HOLD allowed=TRADE_ALLOWED proposed_orders_count=1 (executor gates only on allowed_actions)
```

Then the executor still places the order (dry_run or live), demonstrating the bug.

---

## 5. Failing Tests and Output

### Test A: Executor must not place orders when final_action=HOLD

- **File:** `test_all_features.py`, function `test_executor_no_orders_when_final_action_hold()`.
- **Assertion:** With a decision that has `final_action="HOLD"`, `allowed_actions=TRADE_ALLOWED`, and `proposed_orders=[buy]`, `result["orders_placed"]` must be empty.
- **Current result:** **FAIL** — executor places 1 order despite final_action=HOLD.

```
❌ FAIL: Executor no orders when final_action=HOLD - Executor placed 1 order(s) despite final_action=HOLD (bug: executor only checks allowed_actions)
```

### Test B: Crash guard blocks entry on dump candles

- **File:** `test_all_features.py`, function `test_crash_guard_blocks_entry()`.
- **Result:** **PASS** — crash_guard blocks entry (should_enter False) for synthetic dump candles.

### Test C: Unmanaged positions list (structure and fields)

- **File:** `test_all_features.py`, function `test_unmanaged_positions_returns_holding_without_bot()`.
- **Result:** **PASS** — endpoint returns list; items have symbol, usd_value, note.

---

## 6. NIGHT/USD and Unmanaged Position Evidence

- **Diagnostic:** GET `/api/diag/unmanaged_funding` returns:
  - `exchange_holdings`: assets/amounts from exchange
  - `bot_symbols`: symbols from `list_bots()`
  - `unmanaged_positions`: holdings with no matching bot (e.g. NIGHT/USD if no bot has that symbol)
  - `free_usd`, `funding_note`
- **To determine origin of NIGHT/USD:** Query DB for:
  - `deals` / `order_events` / logs around the time of interest (e.g. Feb 26 2026 04:24) for symbol NIGHT or asset NIGHT.
  - Any `bot_id` in those rows indicates which bot created the order; missing bot_id or deleted bot supports (a) or (b); normalization (e.g. NIGHT vs NIGHT/USD) supports (c).

---

## 7. Minimal Fixes Applied (Accepted)

1. **Executor** (`executor.py`): Only execute **buy** orders from `proposed_orders` when `final_action` is one of `ENTER`, `ADD`, or `SAFETY_ORDER`. For each proposed_order, if `side == "buy"` and `allow_buys` is False, skip (continue). Sell orders are still placed regardless of final_action.
2. **Bot manager** (`bot_manager.py`, _run_loop path): When appending `decision.order` to `intel_decision.proposed_orders`, only append buy orders when `intel_decision.final_action` is one of `ENTER`, `ADD`, or `SAFETY_ORDER`; otherwise do not append the buy (sell orders still appended).
3. **Unmanaged / NIGHT:** No code change; diagnostic and DB queries remain the way to gather evidence.
4. **Funding:** No change; behavior documented.

**Test A** (`test_executor_no_orders_when_final_action_hold`) now **PASSES** after these fixes.

---

## 8. Constraints Respected

- No refactor of unrelated files.
- Temporary logging behind `DEBUG_DECISIONS=1`.
- Minimal, reversible changes; fixes only applied after this verification report is accepted.
