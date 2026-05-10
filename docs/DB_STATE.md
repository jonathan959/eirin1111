# Database state snapshot (Round 3, local `botdb.sqlite3`)

**Actual SQLite file (this run):** `C:\Users\jonat\OneDrive\Desktop\local_3comas_clone_v2\botdb.sqlite3`  
Set via `BOT_DB_PATH` (see worker startup log: `Database: … (abs=…) (… bots)` and `GET /api/health/deep` fields `db_path` / `db_path_abs`).

## 0a. Bots

Schema uses **`dry_run`**, **`enabled`**, **`last_running`** (not `mode` / `status`).

```text
SELECT id, name, symbol, dry_run, enabled, last_running FROM bots ORDER BY id;
```

Current rows (snapshot):

| id | name | symbol | dry_run | enabled | last_running |
|----|------|--------|---------|---------|--------------|
| 66 | Deal Test | BTC/USD | 1 | 0 | 0 |
| 88–102 | … | … | … | … | … |

**Note:** Bot **#1** is absent; tests created bots in higher id ranges. If anything still referenced `bot_id=1` after row deletion, `BotManager` previously spawned a Kraken `BotRunner` for a missing row. **Fix (Round 3):** missing rows register an **orphan runner** (log once, no trading, no phantom exchange client).

## 0b. Deals / journal / orders

```text
SELECT COUNT(*) FROM deals;                    → 2
SELECT COUNT(*) FROM deals WHERE state='CLOSED'; → 0
SELECT COUNT(*) FROM order_events WHERE LOWER(side)='sell' AND LOWER(status)='filled'; → 0
SELECT COUNT(*) FROM journal;                  → 7
```

**Interpretation:** There are **no closed deals**; `backfill_journal_from_closed_deals()` returning **0 rows** is correct. **Realized P&L** on the Bots page from closed journal entries in this DB is minimal/none; any small **+$ uPnL** is **unrealized** (open position vs entry), not realized from `journal.pnl_quote` for closed trades.

There is **no** generic `orders` table; live fills use **`order_events`**.

## Tables

`sqlite_master` reports **41** tables including `journal`, `deals`, `backtest_runs`, `order_events`, `bots`, etc.
