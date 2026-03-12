# Fix Deliverables: Startup, Env, DB, Explore, Autopilot, Tests

## 1. Files changed (summary)

| File | Changes |
|------|--------|
| **env_utils.py** | Canonical env loader: `load_env()` supports `ENV_FILE` override; search order: ENV_FILE → project root `.env` → cwd `.env`. Logs which files loaded and key count (no secrets). Sets default `BOT_DB_PATH` to project root if unset. `get_last_load_result()` for startup_status. |
| **worker_api.py** | **Startup:** Alpaca no longer required; `ENABLE_ALPACA=0` or missing Alpaca keys → crypto-only, BotManager(kc, None, None). Synchronous init: `_init_alpaca_and_bm_sync()` runs in main thread (no background thread). **Status:** `_STARTUP_STATUS` extended with `env_loaded_paths`, `db_path`, `kraken_ready`, `bm_ready`, `last_startup_error`, `timestamp`. **503:** Endpoints that require bm return 503 with `reason` from `_bm_not_ready_reason()`. **Debug:** `/api/debug/startup_status` added; `/api/debug/db_info` now includes `cwd`. **Explore:** `_active_symbol_set()` adds `running_bot` to active_reason when `last_running=1`. **Autopilot create_bots:** Returns `skipped` (symbol + reason) and `summary`; 503 with `reason` when bm not ready. |
| **one_server.py** | Call `env_utils.load_env()` before setting `BOT_DB_PATH` so `.env` can override. |
| **one_server_v2.py** | Same: `load_env()` before `BOT_DB_PATH` setdefault. |
| **tests/conftest.py** | Canonical env: `load_env()` with no args (uses default search); `os.chdir(_root)` so DB path is project root. |
| **tests/test_persistence_explore_autopilot.py** | `BM_READY_TIMEOUT` from env (default 15). Skip reasons use `/api/debug/startup_status` and 503 `reason`. `test_db_info` asserts `cwd` in response. |

## 2. New/updated debug endpoints and curl examples

Base URL: `http://127.0.0.1:8000` (or your server).

### `/api/debug/bm_ready`
```bash
curl -s http://127.0.0.1:8000/api/debug/bm_ready
```
Example: `{"ok":true,"bm_ready":true}`

### `/api/debug/startup_status`
```bash
curl -s http://127.0.0.1:8000/api/debug/startup_status
```
Example:
```json
{
  "ok": true,
  "startup_status": {
    "env_loaded_paths": ["C:/path/to/project/.env"],
    "db_path": "C:/path/to/project/botdb.sqlite3",
    "kraken_ready": true,
    "alpaca_ready": false,
    "bm_ready": true,
    "last_startup_error": null,
    "timestamp": 1234567890
  }
}
```

### `/api/debug/db_info`
```bash
curl -s http://127.0.0.1:8000/api/debug/db_info
```
Example:
```json
{
  "ok": true,
  "db_path": "C:/path/to/project/botdb.sqlite3",
  "bot_count": 5,
  "last_bot_id": 42,
  "cwd": "C:/path/to/project"
}
```

### `/api/startup_status` (existing, now includes new fields)
```bash
curl -s http://127.0.0.1:8000/api/startup_status
```

## 3. Smoke test checklist

1. **Start server**
   - `uvicorn one_server_v2:app --reload --port 8000` or `uvicorn one_server:app --reload --port 8000`
   - With crypto-only: set `ENABLE_ALPACA=0` or leave Alpaca keys unset; ensure Kraken keys in `.env` if you want BotManager.

2. **Verify startup_status**
   - `curl -s http://127.0.0.1:8000/api/debug/startup_status`
   - Check `startup_status.env_loaded_paths`, `startup_status.db_path`, `startup_status.kraken_ready`, `startup_status.bm_ready`, `startup_status.last_startup_error`.

3. **Verify db_info**
   - `curl -s http://127.0.0.1:8000/api/debug/db_info`
   - Check `ok`, `db_path`, `bot_count`, `cwd`.

4. **Verify Explore filtering**
   - Create an enabled bot for a symbol (e.g. ETH/USD).
   - `curl -s "http://127.0.0.1:8000/api/recommendations?horizon=short&market_type=crypto&limit=50&show_already_active=1"`
   - Items for that symbol should have `already_active: true` and `active_reason` containing `enabled_bot` (and `running_bot` if it’s running).
   - With `show_already_active=0` (default), that symbol should not appear.

5. **Verify Autopilot create_bots**
   - `curl -s -X POST http://127.0.0.1:8000/api/autopilot/create_bots -H "Content-Type: application/json" -d "{\"count\":1,\"dry_run\":1,\"horizon\":\"long\"}"`
   - If bm not ready: 503 with `reason`. If no candidates: `ok: false`, `skipped` array. If success: `ok: true`, `created`, `errors`, `skipped`, `summary`. No duplicate bots for same symbol.

6. **Run pytest**
   - `python -m pytest tests/test_persistence_explore_autopilot.py -v`
   - **Expected:** `test_db_info_returns_path_and_bot_count` passes. Other tests may skip with explicit reason if BotManager not ready (no Kraken keys). With Kraken (and optionally ENABLE_ALPACA=0) keys set, all four can pass.

## 4. Env / startup behavior summary

- **ENV_FILE:** Set to a path (e.g. `.env.local`) to load that file; otherwise project root `.env` and cwd `.env` are tried.
- **ENABLE_ALPACA:** `1` = try to init Alpaca (fail clearly if keys missing); `0` or unset when keys missing = crypto-only, BotManager with Kraken only.
- **BOT_DB_PATH:** Set by loader to project root `botdb.sqlite3` if unset; all entrypoints (one_server, one_server_v2, worker_api, worker) use the same default after loading env.
- Startup is **synchronous**: Alpaca and BotManager init run during FastAPI startup; no background thread for bm. Requests that need bm get 503 with `reason` until bm is ready (which is immediately after startup when keys are present).
