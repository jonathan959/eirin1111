
import sqlite3
import json

conn = sqlite3.connect('botdb.sqlite3')
cursor = conn.cursor()
cursor.execute("SELECT id, name, config_json FROM bots")
rows = cursor.fetchall()
for r in rows:
    print(f"Bot {r[0]} ({r[1]}):")
    try:
        cfg = json.loads(r[2])
        print(json.dumps(cfg, indent=2))
    except Exception as e:
        print(f"# JSON parse error: {e}\n{r[2]}")
    print("-" * 20)
conn.close()
