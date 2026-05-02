# [Phase 2.5] auto_restart regression in bot-edit POST handler (worker_api.py:5444)

**Phase:** 2.5 (state machine & crash recovery hardening)
**Severity:** High — silently undoes a deploy-time migration; bots stop auto-restarting after they're edited in the UI.
**First observed:** 2026-05-02 by user — the live API at 3.151.143.63 reports `auto_restart=0` for some bots even after `scripts/migrate_auto_restart.py` ran successfully during deploy.
**Discovered by:** Phase 1.1 audit (see `audit/db_writers.md` §6).

## Symptom

After `scripts/migrate_auto_restart.py` runs, the `bots` table has every row at
`auto_restart=1`. Subsequent `GET /api/bots` confirms this. But after a user
opens the bot edit form in the UI and clicks Save (even with no apparent
changes), `auto_restart` reverts to 0 for that bot. The supervisor then refuses
to restart the bot if it crashes, because `auto_restart=0` is the gate.

## Root cause hypothesis (to confirm with a test)

`worker_api.py:5444` reads the field via the local `_ov` helper
(`worker_api.py:5411`):

```python
def _ov(key: str, default: Any, cast=None):
    v = payload.get(key)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        v = b.get(key, default)
    if cast:
        try:
            v = cast(v)
        except (TypeError, ValueError):
            v = cast(default)
    return v
```

Two failure modes:

1. **HTML form serialiser sends `auto_restart=0` for an unchecked checkbox.**
   Most JS form serialisers (`new FormData(form)`, jQuery's `.serialize()`)
   do this. The handler then casts `0` → `0` and persists it. The migration
   is silently undone.
2. **Bot row's existing value was `0` before the migration completed.** If the
   form omits the field entirely (`payload.get('auto_restart') is None`),
   `_ov` falls back to `b.get('auto_restart', 1)` — but `b` was loaded from
   the DB **before** the migration ran during the same request, so the stale
   `0` wins. This race is unlikely in practice because the migration runs at
   deploy and edit forms are user-driven, but worth ruling out.

## Reproduction (write the test first)

`tests/test_bot_edit_auto_restart.py`:

1. Insert a bot with `auto_restart=1`.
2. POST `/api/bots/{id}` with payload that **omits** `auto_restart`. Assert
   the row stays at `1`.
3. POST `/api/bots/{id}` with payload `{"auto_restart": 0}` explicitly. Assert
   the API rejects it with a clear error (or coerces to 1) — per brief
   Phase 2.5 the column gets `NOT NULL DEFAULT 1` and the form serializer
   defaults to 1.
4. POST `/api/bots/{id}` with payload `{"auto_restart": 1}` explicitly. Assert
   the row stays at `1`.

## Fix scope (Phase 2.5, not Phase 1)

- Add a NOT NULL DEFAULT 1 column constraint via an idempotent migration.
- Server-side: refuse any payload with `auto_restart=0` unless an explicit
  override flag is present (e.g. `disable_auto_restart=true` plus a confirmation
  string). Default coerces to 1.
- Frontend: the checkbox in the edit form must default to checked, and unchecked
  state must require a "are you sure?" confirmation.
- Both server-side test and end-to-end UI test in `tests/`.

## Why this is NOT being fixed in Phase 1

Phase 1's blast radius is the DB-locking hot path. `worker_api.py:5444` lives
in an HTTP handler that doesn't run inside a bot tick; touching it for an
unrelated bug while we're untangling write_txn would conflate two changes and
make the `git bisect` story worse if anything regresses. Phase 2.5 has the
right context (state-machine work) and tests this needs.

## Cross-references

- `audit/db_writers.md` §6 (out-of-scope finds)
- `worker_api.py:5386-5469` (the PUT /api/bots/{bot_id} handler)
- `scripts/migrate_auto_restart.py` (the migration this regression undoes)
