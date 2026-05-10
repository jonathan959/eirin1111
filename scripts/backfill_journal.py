#!/usr/bin/env python3
"""Backfill canonical `journal` rows from closed deals (idempotent)."""

from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from db import backfill_journal_from_closed_deals  # noqa: E402


def main() -> int:
    n = backfill_journal_from_closed_deals()
    print("backfill_journal: inserted", n, "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
