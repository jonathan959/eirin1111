#!/usr/bin/env python3
"""One-off: print bot count from botdb.sqlite3 (for server diagnostic)."""
import sqlite3
import shutil
src = "/home/ubuntu/local_3comas_clone_v2/botdb.sqlite3"
copy = "/tmp/botdb_check_copy.sqlite3"
shutil.copy2(src, copy)
c = sqlite3.connect(copy)
n = c.execute("SELECT COUNT(1) FROM bots").fetchone()[0]
print("Bots:", n)
c.close()
