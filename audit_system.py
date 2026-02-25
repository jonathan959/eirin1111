import sys
import os
import sqlite3
import json
import requests

sys.path.append(os.getcwd())
try:
    from db import list_bots, list_recommendations, _conn
except ImportError:
    print("Error importing db module")
    sys.exit(1)

def audit():
    print("=== TRUTH AUDIT ===")
    
    # 1. DB Connectivity
    try:
        con = _conn()
        print("✅ DB Connection: OK")
        con.close()
    except Exception as e:
        print(f"❌ DB Connection: FAILED ({e})")
        return

    # 2. Bots State
    bots = list_bots()
    print(f"\n--- Bots ({len(bots)}) ---")
    for b in bots:
        print(f"ID {b['id']} | {b['name']} | Enabled: {b['enabled']} | Mode: {b['strategy_mode']} | Market: {b.get('market_type')}")

    # 3. Recommendations State
    print("\n--- Recommendations ---")
    rows = list_recommendations("short", limit=1000)
    stocks = [r for r in rows if "market_type" in r.get("metrics_json", "") and '"market_type": "stocks"' in r.get("metrics_json", "")]
    crypto = [r for r in rows if "market_type" in r.get("metrics_json", "") and '"market_type": "crypto"' in r.get("metrics_json", "")]
    
    # Fallback check (my previous fix might not be in the file loaded by this script if cached?)
    # Actually, verify data in DB directly
    con = _conn()
    recos = con.execute("SELECT symbol, metrics_json FROM recommendations_snapshots ORDER BY id DESC LIMIT 50").fetchall()
    con.close()
    
    print(f"Total Cached Recos: {len(rows)}")
    print(f"Tagged Stocks: {len(stocks)}")
    print(f"Tagged Crypto: {len(crypto)}")
    
    print("\n--- Sample DB Data (Latest 5) ---")
    for r in recos[:5]:
        m = json.loads(r['metrics_json'] or "{}")
        print(f"Sym: {r['symbol']} | Market: {m.get('market_type')} | Score: {m.get('score')}")

    # 4. API Connectivity
    print("\n--- API Check ---")
    try:
        resp = requests.get("http://127.0.0.1:8000/api/health", timeout=2) # Assuming /health exists or root
        print(f"API reachable: {resp.status_code}")
    except Exception as e:
        print(f"API Unreachable: {e}")

if __name__ == "__main__":
    audit()
