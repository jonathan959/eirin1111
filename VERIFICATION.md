# Verification log

## Round 2 — Prices / Exposure / Journal

### Task 1 — /api/prices + icons

**pytest (focused):** `13 passed` for `tests/test_prices_split.py`, `tests/test_exposure_cap.py`, `tests/test_journal_close.py`, `tests/test_journal_backfill.py` (warnings only).

**curl**

```text
curl.exe -s "http://127.0.0.1:8000/api/prices?symbols=BTC/USD&market_type=crypto" -w "\n%{time_total}s %{http_code}\n"
{"ok":true,"prices":{"BTC/USD":81422.2},"changes":{"BTC/USD":0.9323156906958092},"volumes":{"BTC/USD":99122336.04076885},"partial":false}
0.020989s 200
```

```text
curl.exe -s "http://127.0.0.1:8000/api/prices?symbols=AAPL&market_type=stocks" -w "\n%{time_total}s %{http_code}\n"
{"ok":true,"prices":{"AAPL":292.85},"changes":{"AAPL":1.8963117606124027},"volumes":{"AAPL":1591559.0},"partial":false}
0.744888s 200
```

```text
curl.exe -s "http://127.0.0.1:8000/api/prices?symbols=BTC/USD,AAPL,UNI/USD,LMT&market_type=all" -w "\n%{time_total}s %{http_code}\n"
{"ok":true,"prices":{"BTC/USD":81422.2,"UNI/USD":4.055,"AAPL":292.85,"LMT":506.6},"changes":{...},"volumes":{...},"partial":false}
0.722231s 200
```

**Files touched:** `services/prices_fetch.py`, `worker_api.py` (`/api/prices`, `/api/icons/map`), `services/icon_map.py`, `static/app.js`, `templates/explore.html`.

---

### Task 2 — Exposure cap 422 + UI + autotune

**curl (bot id 66 — only bot in this DB snapshot)**

422:

```text
PUT http://127.0.0.1:8000/api/bots/66 body {"base_order_quote":25,"per_symbol_pct":0.15}
422
{"error":"exposure_cap_conflict","message":"Base order $25.00 exceeds per-symbol cap $15.98 (portfolio $106.56 x 15.00%).","suggestions":{"per_symbol_pct":0.24,"base_order_quote":15.0},"current":{"portfolio_value":106.56,"per_symbol_pct":0.15,"base_order_quote":25.0,"effective_cap_usd":15.98}}
```

200:

```text
PUT http://127.0.0.1:8000/api/bots/66 body {"base_quote":10,"per_symbol_exposure_pct":0.15}
200 {"ok":true,"bot":{...}}
```

**Runtime autotune worker log:** not re-verified with a live dry-run tick in this pass.

**Files touched:** `services/exposure_cap.py`, `worker_api.py`, `templates/bots.html`, `db.py`, `bot_manager.py`.

---

### Task 3 — Journal backfill + API

**Backfill:** `backfill_journal: inserted 0 rows` (no closed deals missing a journal row at run time).

**curl**

```text
GET http://127.0.0.1:8000/api/journal?limit=10 -> 200; response includes "journal": [...]
```

**Manual paper close + confirm new journal row:** not executed in this pass.

**Files touched:** `db.py`, `services/journal.py`, `scripts/backfill_journal.py`, `worker_api.py`.

---

### Full test suite

Last full run on this tree: **407 passed, 24 failed, 17 skipped** in ~204s (failures appear environment/integration-related; not all traced to this round).
