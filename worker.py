# worker.py
import os
import sys
import uvicorn

from env_utils import load_env

load_env()


def _bool_env(name: str, default: bool = False) -> bool:
    v = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _startup_banner() -> None:
    host = os.getenv("WORKER_HOST", "127.0.0.1")
    port = int(os.getenv("WORKER_PORT", "9001"))
    token = os.getenv("WORKER_API_TOKEN", "").strip()

    print("")
    print("=== Worker API ===")
    print(f"Bind: http://{host}:{port}")
    if token:
        print("Auth: ON (X-API-Key required for /api/*)")
    else:
        print("Auth: OFF (set WORKER_API_TOKEN in .env for real-money safety)")
    print("==================")
    print("")


if __name__ == "__main__":
    # Hard safety suggestion: for real-money use, set WORKER_API_TOKEN
    # (worker_api.py enforces it automatically when set).
    _startup_banner()

    host = os.getenv("WORKER_HOST", "127.0.0.1")
    port = int(os.getenv("WORKER_PORT", "9001"))
    log_level = os.getenv("LOG_LEVEL", "info")

    # Default: no reload for stability.
    reload_enabled = _bool_env("WORKER_RELOAD", default=False)
    print(f"WORKER_HOST={host} WORKER_PORT={port} WORKER_RELOAD={reload_enabled}")

    try:
        uvicorn.run(
            "worker_api:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=reload_enabled,
        )
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        print(f"FATAL: uvicorn crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)