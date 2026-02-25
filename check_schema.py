
import sqlite3

conn = sqlite3.connect('botdb.sqlite3')
cursor = conn.cursor()
cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
rows = cursor.fetchall()
for r in rows:
    print(f"Table: {r[0]}")
    print(r[1])
    print("-" * 20)

print("\n--- Rows in 'bots' ---")
# Try `SELECT * FROM bots` with limit 1 to see columns
cursor.execute("PRAGMA table_info(bots)")
for col in cursor.fetchall():
    print(col)

conn.close()
