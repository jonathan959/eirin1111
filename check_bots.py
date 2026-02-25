import sys
import os
import sqlite3

sys.path.append(os.getcwd())
from db import list_bots

try:
    bots = list_bots()
    print(f"Found {len(bots)} bots.")
    for b in bots:
        print(f"Bot {b['id']} ({b['name']}): Enabled={b.get('enabled')}, Mode={b.get('strategy_mode')}")
except Exception as e:
    print(f"Error: {e}")
