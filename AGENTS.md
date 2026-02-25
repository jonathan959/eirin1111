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
.venv/bin/python -m pytest tests/ -v --ignore=tests/test_bot_crud.py
```

- `tests/test_bot_crud.py` has a pre-existing import error (`api_bots_create` was renamed) — skip it.
- Some tests in `test_order_sizing.py` fail due to outdated `IntelligenceDecision` constructor signatures.
- `test_symbol_routing.py` subtests for stock symbols require network access and Alpaca API keys.
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

The `ALPACA_DATA_FEED` in `.env` should be `iex` (free tier); the older `alpaca_client.py` code path defaults to `sip` which requires a paid subscription and will return `"subscription does not permit querying recent SIP data"` errors. The newer `unified_alpaca_client.py` correctly defaults to `iex`.

Stock bots (WMT, AAPL, etc.) that use Alpaca may show `"request is not authorized"` errors if the Alpaca paper keys lack trading permissions — this does not affect Kraken crypto bots. Kraken may intermittently show DDoS protection / rate limit errors on first startup; these are transient.

### Gotchas

- `python3.12-venv` system package must be installed for `python3 -m venv` to work (not present by default on Ubuntu 24.04).
- TensorFlow import emits info-level messages about oneDNN; these are harmless.
- The `start.sh` script in the repo root automates venv creation, pip install, db init, and server start — but runs uvicorn in the foreground, so use it for one-off manual testing only.
- The `.cursor/rules/deploy-after-changes.mdc` rule about running `deploy.ps1` applies only when deploying to the production EC2 server. For local dev, ignore it.
- `yfinance` (Yahoo Finance fallback) fails in the Cloud Agent sandbox due to `curl_cffi` browser impersonation incompatibility. This only affects the fallback data source; primary exchange data via Kraken/Alpaca works fine.
- Alpaca WebSocket connections may hit "connection limit exceeded" on free tier — this is non-fatal; the app falls back to REST polling.
