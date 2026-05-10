"""One-off DB snapshot for Round 3 docs (run from repo root)."""
import os
import sqlite3

p = os.path.abspath(os.getenv("BOT_DB_PATH", "botdb.sqlite3"))
print("DB_PATH_ABS:", p)
con = sqlite3.connect(p)
c = con.cursor()
print("\n-- bots (id, name, symbol, dry_run, enabled, last_running)")
try:
    c.execute(
        "SELECT id, name, symbol, dry_run, enabled, last_running FROM bots ORDER BY id"
    )
    for r in c.fetchall():
        print(r)
except Exception as e:
    print("ERR", e)

queries = [
    ("deals total", "SELECT COUNT(*) FROM deals"),
    ("deals CLOSED", "SELECT COUNT(*) FROM deals WHERE state = 'CLOSED'"),
    ("journal rows", "SELECT COUNT(*) FROM journal"),
    ("tables", "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"),
]
print("\n-- order_events sell+filled (if table exists)")
try:
    c.execute("PRAGMA table_info(order_events)")
    cols = [x[1] for x in c.fetchall()]
    print("order_events columns:", cols)
    if cols:
        if "side" in cols and "status" in cols:
            c.execute(
                "SELECT COUNT(*) FROM order_events WHERE LOWER(side)='sell' AND LOWER(status)='filled'"
            )
        else:
            c.execute("SELECT COUNT(*) FROM order_events")
        print("count:", c.fetchone())
except Exception as e:
    print("order_events:", e)

for label, q in queries:
    print(f"\n-- {label}")
    try:
        c.execute(q)
        rows = c.fetchall()
        if len(rows) > 30:
            print(len(rows), "rows (first 30):", rows[:30])
        else:
            print(rows)
    except Exception as e:
        print("ERR", e)
con.close()
