# Live Trading Guide

This document lists environment variables and a checklist for enabling real-money (live) trading. The system is **paper by default**; live is opt-in.

---

## Environment variables

| Variable | Default | Purpose |
|---------|---------|---------|
| `ALLOW_LIVE_TRADING` | `0` | **Master gate.** Set to `1` to allow real orders. With `0`, all live orders are blocked even if bots are set to live. |
| `LIVE_TRADING_ENABLED` | `0` | Set to `1` to load the Alpaca **live** client at startup. If `0`, only paper is used (saves connections). |
| `WORKER_API_TOKEN` | (none) | When set, live-sensitive API endpoints require this token. Recommended for any exposed server. |
| `KRAKEN_API_KEY` / `KRAKEN_API_SECRET` | (none) | Kraken API keys for crypto. Use live keys only when you intend to trade live. |
| `ALPACA_API_KEY_LIVE` / `ALPACA_API_SECRET_LIVE` | (none) | Alpaca live keys for stocks. Paper keys are separate (`ALPACA_API_KEY_PAPER` / `ALPACA_API_SECRET_PAPER`). |
| `DISCORD_WEBHOOK_URL` | (none) | Optional. When set, autopilot and risk alerts can be sent to Discord. Configure notification prefs via Safety page or `notification_prefs` setting. |

Store secrets in `.env` only. Never commit live keys to the repo.

---

## Checklist before going live

1. **Run in paper first**  
   Keep `dry_run=1` on bots and run autopilot in paper mode until you are comfortable with behavior and P&L.

2. **Set env intentionally**  
   - `ALLOW_LIVE_TRADING=1` when you are ready to allow real orders.  
   - `LIVE_TRADING_ENABLED=1` if you use Alpaca live.  
   - `WORKER_API_TOKEN` set if the server is exposed (recommended).

3. **Verify connectivity**  
   Run `python verify_live_ready.py` (server and keys). Fix any failing checks before enabling live.

4. **Use the Safety page**  
   Open **Safety** in the UI. Check the **Live trading checklist**: API token, allow live, kill switch, Kraken/Alpaca status, and “Live ready” state. Resolve any blocking issues shown there.

5. **Start small**  
   Use small position sizes or capital per bot so that a bug or bad run has limited impact.

6. **Know how to stop**  
   Use the **Kill switch** on the Safety page to stop all trading immediately if something looks wrong.

---

## How live is blocked

- **Bots** default to `dry_run=1` (paper). To run a bot live, set `dry_run=0` (e.g. in Setup or bot config).
- **Executor and BotManager** refuse real orders unless `ALLOW_LIVE_TRADING=1` in the environment.
- **API** endpoints that place or cancel live orders check `ALLOW_LIVE_TRADING` and the bot’s `dry_run`.
- **Kill switch** (Safety page or `kill_switch` setting): when enabled, the intelligence layer can block new trades.

---

## Notifications

- **Autopilot** can notify when it creates or closes a bot (Discord if `DISCORD_WEBHOOK_URL` is set).
- Enable/disable and Discord on/off via **Safety → Notifications** or `GET/POST /api/notification/prefs`.
- Webhook URL is read from `.env` only; it is not stored in the DB or config.

---

## Audit of live orders

- Every order (and cancel) is logged to the `order_events` table.
- The `is_live` column is `1` for real orders and `0` for paper/dry-run.
- You can query for live orders only, e.g. for reconciliation or compliance.

---

## Quick reference

| Goal | Action |
|------|--------|
| Stay paper | Leave `ALLOW_LIVE_TRADING=0` and bots with `dry_run=1`. |
| Enable live | Set `ALLOW_LIVE_TRADING=1` and `LIVE_TRADING_ENABLED=1` in `.env`. Set bot(s) to `dry_run=0` when ready. |
| Stop everything | Enable **Kill switch** on the Safety page. |
| Check status | Open **Safety** and review the Live trading checklist. |
