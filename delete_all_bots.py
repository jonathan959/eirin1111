#!/usr/bin/env python3
"""
Delete ALL bots for a fresh start.

- For LIVE bots with open positions: closes the position first (sells and frees capital)
- Then deletes every bot from the database

Run while the server is running so positions can be closed properly.
If the server is not running, only DB deletion is performed (positions would
remain on the exchange as unmanaged).
"""
from __future__ import annotations

import sys

# Use DB directly for delete; use API for close_position (needs trading client)
def _delete_via_db():
    from db import list_bots, delete_bot
    bots = list_bots() or []
    ids = [int(b.get("id") or 0) for b in bots if b.get("id")]
    for bid in ids:
        try:
            delete_bot(bid)
            print(f"  Deleted bot #{bid}")
        except Exception as e:
            print(f"  Failed to delete bot #{bid}: {e}")
    return len(ids)


def _run_via_api(base_url: str) -> bool:
    import urllib.request
    import urllib.error
    import json

    def req(method: str, path: str, body=None):
        url = f"{base_url.rstrip('/')}{path}"
        data = json.dumps(body).encode() if body else None
        req_obj = urllib.request.Request(url, data=data, method=method)
        if body:
            req_obj.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req_obj, timeout=30) as r:
            return json.loads(r.read().decode())

    try:

        # 1. List bots
        data = req("GET", "/api/bots")
        bots = data.get("bots") or []
        if not bots:
            print("No bots found.")
            return True

        print(f"Found {len(bots)} bot(s).")
        # 2. Close positions on LIVE bots with open positions
        for b in bots:
            bid = int(b.get("id") or 0)
            dry_run = bool(b.get("dry_run", 1))
            sym = (b.get("symbol") or "").strip() or "(no symbol)"
            if not dry_run:
                try:
                    status = req("GET", f"/api/bots/{bid}/status")
                    snap = status.get("snap") or status
                    base_pos = float(snap.get("base_pos") or 0)
                    if base_pos > 0:
                        print(f"  Closing position for bot #{bid} ({sym}): {base_pos} units...")
                        req("POST", f"/api/bots/{bid}/close_position")
                        print(f"    Closed.")
                except urllib.error.HTTPError as e:
                    try:
                        body = json.loads(e.read().decode())
                        if "No open position" in str(body.get("error", "")):
                            pass
                        else:
                            print(f"  Warning: close failed for bot #{bid}: {body.get('error', e)}")
                    except Exception:
                        print(f"  Warning: close failed for bot #{bid}: {e}")
                except Exception as e:
                    print(f"  Warning: could not close bot #{bid}: {e}")
            else:
                print(f"  Skipping close for dry-run bot #{bid} (no real position).")

        # 3. Delete all bots
        for b in bots:
            bid = int(b.get("id") or 0)
            if not bid:
                continue
            try:
                req("DELETE", f"/api/bots/{bid}")
                print(f"  Deleted bot #{bid}")
            except Exception as e:
                print(f"  Failed to delete bot #{bid}: {e}")

        print("Done. All bots deleted.")
        return True

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"Cannot reach server at {base_url}: {e}")
        return False
    except Exception as e:
        print(f"API error: {e}")
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="Delete all bots for a fresh start.")
    p.add_argument("--base", "-b", default="http://127.0.0.1:8000", help="Server base URL")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation for DB-only fallback")
    args = p.parse_args()

    print("Deleting all bots (fresh start)...")
    ok = _run_via_api(args.base)
    if not ok:
        print("\nFalling back to DB-only deletion (server may not be running).")
        print("Note: Any live positions will remain on the exchange as unmanaged.")
        if not args.yes:
            confirm = input("Proceed? [y/N]: ").strip().lower()
            if confirm != "y":
                print("Aborted.")
                sys.exit(1)
        n = _delete_via_db()
        print(f"Deleted {n} bot(s) from database.")


if __name__ == "__main__":
    main()
