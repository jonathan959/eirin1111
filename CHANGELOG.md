# Changelog

## [Unreleased] — Live trading hardening (LIVE_TRADING_IMPROVEMENT_PROMPT)

### Safety gates (confirmed / added)

- **Live trading remains OFF by default.** `ALLOW_LIVE_TRADING` and `LIVE_TRADING_ENABLED` must be set to enable real orders.
- **Executor:** Exchange-specific minimum order size: Kraken default $10, Alpaca default $1 (configurable via `MIN_NOTIONAL_KRAKEN_USD`, `MIN_NOTIONAL_ALPACA_USD`). Env `MIN_NOTIONAL_USD` still applies as a floor.
- **Executor:** Every live order placement failure is logged to `error_log` (source `executor`, type `order_placement_failed`) for alerting and debugging.
- **Kill switch and global pause** continue to be enforced in `bot_manager` and intelligence Layer 1 before any order.

### Autopilot

- **Heartbeat:** After each autopilot cycle, `autopilot_last_heartbeat_ts` is written to DB settings. A monitor can restart autopilot if the heartbeat is older than 2× the cycle interval.
- **Audit log:** Every autopilot decision (create_bot, stop_bot, skip_slots_full, skip_symbol_has_bot, cycle_complete, cycle_failed, create_bot_failed) is written to `autopilot_audit_log` with action, symbol, reason, and optional details.
- **No duplicate bots:** Autopilot already skips creating a bot when the symbol already has an active bot; audit log now records skip reasons.

### Observability

- **Health API:** `/health` and `/api/health` now include `uptime_sec` and `last_autopilot_heartbeat_ts` for dashboard monitoring.
- **SSE logstream:** `/api/bots/{id}/logstream` has a maximum duration (`LOGSTREAM_MAX_DURATION_SEC`, default 3600s). After that, the stream sends `event: bye` and exits so disconnected clients do not hold generator threads indefinitely.

### UI

- **Live trading indicator:** Dashboard (/) and bot detail (/bots/{id}) show a red “LIVE TRADING” banner when live trading is enabled and the bot (or any bot on home) is in live mode (`dry_run=0`).

### Database

- New table: `autopilot_audit_log` (ts, action, symbol, reason, details_json). Created in `init_db()` with index on `ts`.

### Files changed (with `# LIVE-HARDENED` comments where applicable)

- `RUNTIME_MAP.md` — New: runtime map for entrypoint, threads, bot loop, orders, autopilot, kill switch.
- `db.py` — `autopilot_audit_log` table, `add_autopilot_audit_log()`.
- `autopilot.py` — Heartbeat write, audit log for every decision, single log when slots full.
- `executor.py` — `log_error` on live order failure, `_min_notional_usd()` for Kraken/Alpaca minimums.
- `one_server_v2.py` — Logstream max duration, live trading banner, `_live_trading_enabled()`.
- `worker_api.py` — `/health` and `/api/health`: `uptime_sec`, `last_autopilot_heartbeat_ts`.
