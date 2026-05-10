# Canonical env load before any test imports worker_api/one_server so BOT_DB_PATH,
# KRAKEN_*, ALPACA_*, ENABLE_ALPACA are set. BotManager is inited synchronously at startup.
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)
os.chdir(_root)

from env_utils import load_env
load_env()  # uses ENV_FILE or project root .env
# Starlette TestClient uses ASGI client "testserver", not loopback — if WORKER_API_TOKEN is
# set from .env, /api/* would 401 without X-API-Key. Tests expect open access.
os.environ["WORKER_API_TOKEN"] = ""
