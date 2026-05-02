# [Phase 3.2] hard_sl_pct defaults to 0.0 server-side (worker_api.py:5459)

**Phase:** 3.2 (risk engine — mandatory hard stop loss)
**Severity:** High — silently allows live trading without a hard stop loss; the brief explicitly forbids this.
**First observed:** 2026-05-02 — bot 1 on the live host has `hard_sl_pct=0.0`.
**Discovered by:** Phase 1.1 audit (see `audit/db_writers.md` §6) and the user's standing instruction in the brief: "bot 1 currently has hard_sl_pct=0.0 — block live mode for it."

## Symptom

A bot can be created or edited via `/api/bots/{id}` with no `hard_sl_pct`
field in the payload. The server stores `hard_sl_pct=0.0` in the DB. With
`hard_sl_pct=0.0`, the bot has **no hard stop loss** at the exchange OR in
the local watchdog — every other risk control is downstream of this value.

## Root cause

`worker_api.py:5459`:

```python
"hard_sl_pct": float(_ov("hard_sl_pct", 0.0, lambda x: float(x) if x is not None else 0.0)),
```

`0.0` is the default in the `_ov` call. When the payload omits `hard_sl_pct`
(common when migrating bots created before the column existed, or when
copying a template with the field empty), the bot inherits `0.0`. There is no
server-side validator that rejects `hard_sl_pct == 0` for live mode.

## Fix scope (Phase 3.2, not Phase 1)

Per brief Phase 3.2 the requirements are:

- The Go-Live preflight rejects any bot with `hard_sl_pct == 0` or
  `hard_sl_pct > 0.15`.
- Recommended default `0.08` (8%).
- The bot MUST place a real stop order on the exchange (where supported) AND
  maintain a local watchdog that market-outs if `price < entry * (1 - hard_sl_pct)`
  and the exchange stop is missing.
- Bot 1 specifically is blocked from live mode until its `hard_sl_pct` is set.

Implementation:

- Idempotent migration that defaults `hard_sl_pct` to `0.08` for any bot in
  paper mode where the value is currently `0.0`. Live-mode bots must be
  surfaced (not silently changed) — the user reviews and sets per bot.
- Server-side validator on `/api/bots` POST and PUT: reject payloads where
  `hard_sl_pct` falls outside `[0.03, 0.15]` (per Phase 9 final acceptance).
- The `_ov` default at `worker_api.py:5459` must NOT be `0.0`. Recommended
  `0.08`.
- Tests:
  - `test_hard_sl_pct_zero_rejected_in_live_mode`
  - `test_hard_sl_pct_above_15pct_rejected`
  - `test_hard_sl_pct_default_is_8pct_when_omitted`
  - `test_bot_1_blocked_from_live_until_sl_set` (acceptance gate)

## Why this is NOT being fixed in Phase 1

Same reason as the auto_restart regression: this is in an HTTP handler, not
the DB-locking hot path. Fixing it correctly requires the full risk engine
context from Phase 3 (hard stop placement on the exchange, local watchdog
that races the exchange stop, MIN_NOTIONAL handling for the stop order).
Adding a one-line "if hard_sl_pct == 0: reject" in Phase 1 would be a band-aid
that breaks every existing bot in the DB that has `0.0` today, including the
ones the user has not yet had time to triage.

## Cross-references

- `audit/db_writers.md` §6 (out-of-scope finds)
- `worker_api.py:5386-5469` (the PUT handler with the bad default)
- Brief Phase 3.2 for full requirements
- Brief Phase 9 final acceptance gates for `hard_sl_pct in [0.03, 0.15]`
