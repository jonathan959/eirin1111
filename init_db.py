"""
Initialize botdb.sqlite3 with correct schema. Run once before first start.
Usage: python init_db.py
"""
import os
import sys

# Ensure we run from project root
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# Load .env before db (in case BOT_DB_PATH is set)
try:
    from env_utils import load_env
    load_env()
except Exception:
    pass

from db import init_db

if __name__ == "__main__":
    db_path = os.getenv("BOT_DB_PATH", "botdb.sqlite3")
    print(f"Initializing database: {db_path}")
    init_db()
    print("Done. Database ready.")
