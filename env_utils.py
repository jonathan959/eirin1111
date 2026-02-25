"""
Shared .env loader. No python-dotenv dependency.
Load BEFORE importing KrakenClient, BotManager, or any module that reads os.environ.
"""
import logging
import os

logger = logging.getLogger(__name__)


def load_env(paths: None | str | list[str] = None) -> None:
    """Load .env from given paths. Later paths override. Missing paths skipped."""
    if paths is None:
        base = os.path.dirname(os.path.abspath(__file__))
        paths = [
            os.path.join(base, ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                for raw in f.readlines():
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            logger.exception("load_env: failed to load %s", p)
