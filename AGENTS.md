# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python 3.12 / FastAPI algorithmic trading bot platform ("Eirin Bot"). It combines a web dashboard and a bot engine in a single process (`one_server.py`), backed by SQLite (`botdb.sqlite3`). No external database server is needed.

### Running the application

```bash
.venv/bin/uvicorn one_server:app --reload --port 8000 --host 0.0.0.0
```

The app starts in **minimal mode** when exchange API keys (`ALPACA_API_KEY_PAPER`, `KRAKEN_API_KEY`, etc.) are absent. The UI and all non-trading endpoints work normally without keys; only live trading and market-data features are degraded.

### Environment file

Copy `.env.example` to `.env` before first run. Keys are optional for basic UI/dev work. See `.env.example` for all options.

### Database

Run `python init_db.py` once to create `botdb.sqlite3`. The server will not create the DB automatically. If the schema changes, re-run `init_db.py` (it is safe to re-run).

### Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

- `test_symbol_routing.py` has a few subtests that depend on network access and yfinance (which fails in the sandbox due to curl_cffi); these fail in CI but pass locally.
- `test_order_sizing.py` acceptance tests depend on the executor's implementation details; rejection tests (zero/negative size) pass.
- The project also has `run_all_checks.py` which runs unittest discovery plus custom routing verification.

### Lint

No project-level linting config exists. Use `flake8` for basic checks:

```bash
.venv/bin/python -m flake8 --max-line-length=120 --select=E9,F63,F7,F82 one_server.py worker_api.py db.py
```

Compile-all check (used by deploy preflight):

```bash
.venv/bin/python -m compileall -q .
```

### Secrets / API keys

When the environment secrets `ALPACA_API_KEY_PAPER`, `ALPACA_API_SECRET_PAPER`, `KRAKEN_API_KEY`, `KRAKEN_API_SECRET` are injected, write them into `.env` before starting the server — the app reads `.env` via `env_utils.load_env()`, not shell env vars directly. Use a script like:

```python
import os
with open('.env', 'r') as f: content = f.read()
for key in ['KRAKEN_API_KEY','KRAKEN_API_SECRET','ALPACA_API_KEY_PAPER','ALPACA_API_SECRET_PAPER']:
    val = os.environ.get(key, '')
    if val:
        content = content.replace(f'{key}=\n', f'{key}={val}\n', 1)
with open('.env', 'w') as f: f.write(content)
```

Both `alpaca_client.py` and `unified_alpaca_client.py` now default to `iex` (free tier). The `ALPACA_DATA_FEED` in `.env` should remain `iex` unless you have a paid SIP subscription.

Stock bots may show WARN-level `"Account snapshot failed: request is not authorized"` — this is non-fatal and the bot continues to run. The Alpaca Trading API auth issue is logged as a warning rather than crashing the bot.

### Bot architecture notes

- All strategy modes (including `classic`) now route through `_run_loop_multi()`, which has the best error handling for exchange API calls.
- OHLCV fetches use `ohlcv_cached()` (TTL-based caching) to reduce API calls and avoid rate limit storms when running multiple crypto bots.
- Kraken DDoS/rate-limit errors are handled with exponential backoff (5 retries, up to 30s sleep, extra 2x for DDoS).
- Exchange API errors in the bot loop are caught per-iteration (not fatal) — the bot logs a WARN and retries on the next cycle.

### Gotchas

- `python3.12-venv` system package must be installed for `python3 -m venv` to work (not present by default on Ubuntu 24.04).
- TensorFlow import emits info-level messages about oneDNN; these are harmless.
- The `start.sh` script in the repo root automates venv creation, pip install, db init, and server start — but runs uvicorn in the foreground, so use it for one-off manual testing only.
- The `.cursor/rules/deploy-after-changes.mdc` rule about running `deploy.ps1` applies only when deploying to the production EC2 server. For local dev, ignore it.
- `yfinance` (Yahoo Finance fallback) fails in the Cloud Agent sandbox due to `curl_cffi` browser impersonation incompatibility. This only affects the fallback data source; primary exchange data via Kraken/Alpaca works fine.
- Alpaca WebSocket connections may hit "connection limit exceeded" on free tier — this is non-fatal; the app falls back to REST polling.
- Multiple crypto bots sharing the same Kraken API keys will share rate limits. The `ohlcv_cached` layer mitigates this.
