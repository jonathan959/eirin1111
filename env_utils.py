"""
Canonical .env loader. No python-dotenv dependency.
Load BEFORE importing KrakenClient, BotManager, db, or any module that reads os.environ.

Used by: worker_api.py, one_server.py, one_server_v2.py, worker.py, init_db.py, tests/conftest.py.
"""
import logging
import os

logger = logging.getLogger(__name__)

# Result of last load_env() for startup_status (paths loaded, key count). Do not log secrets.
_LAST_LOAD_RESULT: dict = {"loaded_paths": [], "keys_set": 0, "warn_no_file": False}


def load_env(paths: None | str | list[str] = None) -> dict:
    """
    Load .env from given paths or discover in order:
      1) ENV_FILE (if set)
      2) project root .env (directory of this file)
      3) os.getcwd() .env
      4) script directory .env (caller's __file__ not available here, so we use cwd as fallback)
    Later paths override earlier. Missing paths are skipped; no exception.
    Returns {"loaded_paths": [...], "keys_set": N, "warn_no_file": bool}.
    Never prints or logs secret values.
    """
    global _LAST_LOAD_RESULT
    base = os.path.dirname(os.path.abspath(__file__))
    if paths is None:
        explicit = os.getenv("ENV_FILE", "").strip()
        if explicit and os.path.isabs(explicit):
            paths = [explicit]
        elif explicit:
            # relative: try project root then cwd
            paths = [
                os.path.join(base, explicit),
                os.path.join(os.getcwd(), explicit),
            ]
        else:
            paths = [
                os.path.join(base, ".env"),
                os.path.join(os.getcwd(), ".env"),
            ]
    if isinstance(paths, str):
        paths = [paths]

    loaded_paths = []
    total_keys = 0
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                count = 0
                for raw in f.readlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
                        count += 1
                if count:
                    loaded_paths.append(os.path.abspath(p))
                    total_keys += count
                logger.info("load_env: loaded %s (%d keys set)", os.path.abspath(p), count)
        except Exception as e:
            logger.warning("load_env: failed to load %s: %s", p, e)

    warn_no_file = len(loaded_paths) == 0 and len(paths) > 0
    if warn_no_file:
        logger.warning(
            "load_env: no .env file found (tried %s). Set ENV_FILE or add .env in project root.",
            paths[:3],
        )

    # Single stable default DB path (project root) so all entrypoints use same DB
    if "BOT_DB_PATH" not in os.environ:
        os.environ["BOT_DB_PATH"] = os.path.join(base, "botdb.sqlite3")

    _LAST_LOAD_RESULT = {
        "loaded_paths": loaded_paths,
        "keys_set": total_keys,
        "warn_no_file": warn_no_file,
    }
    return _LAST_LOAD_RESULT


def get_last_load_result() -> dict:
    """Return result of last load_env() for /api/debug/startup_status."""
    return dict(_LAST_LOAD_RESULT)
