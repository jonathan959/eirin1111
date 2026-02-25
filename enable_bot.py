import sys
import os
import sqlite3

sys.path.append(os.getcwd())
from db import update_bot, list_bots

try:
    print("Enabling Bot 8...")
    update_bot(8, {"name": "Debug Bot", "symbol": "BTC/USD", "enabled": 1, "base_quote": 20, "safety_quote": 20, "max_safety": 5, "first_dev": 0.01, "step_mult": 1.2, "tp": 0.01, "max_spend_quote": 100})
    
    bots = list_bots()
    for b in bots:
        if b['id'] == 8:
            print(f"Bot 8 status: {b.get('enabled')}")
except Exception as e:
    print(f"Error: {e}")
