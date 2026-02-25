# worker_api.py  (TOP OF FILE)

import os
import socket
import time
import threading
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

# =========================================================
# Env loader: MUST RUN BEFORE importing KrakenClient/BotManager
# =========================================================
from env_utils import load_env
load_env()

# NOW import project modules that may read env vars
from symbol_classifier import classify_symbol, is_stock_symbol, is_crypto_symbol, validate_symbol_type

from db import (
    init_db,
    now_ts,
    update_bots_by_type,
    add_log,
    create_bot,
    update_bot,
    delete_bot,
    get_bot,
    list_bots,
    list_logs,
    list_logs_since,
    list_deals,
    list_all_deals,
    list_closed_deals_for_journal,
    get_deal,
    pnl_summary,
    bot_deal_stats,
    latest_open_deal,
    bot_pnl_series,
    bot_drawdown_series,
    bot_performance_stats,
    latest_regime,
    list_strategy_decisions,
    add_order_event,
    save_recommendation_snapshot,
    list_recommendations,
    get_recommendation,
    link_recommendation_to_bot,
    get_recommendation_performance_stats,
    delete_recommendations_for_blocklist,
    set_setting,
    get_setting,
    get_strategy_leaderboard,
    get_trade_journal,
    upsert_trade_journal,
    list_trade_journals_for_deals,
    get_intelligence_decisions,
)


from kraken_client import KrakenClient
from alpaca_client import AlpacaClient
from bot_manager import BotManager, ALLOW_LIVE_TRADING
from alpaca_adapter import AlpacaAdapter

USE_UNIFIED_ALPACA = os.getenv("USE_UNIFIED_ALPACA", "1").strip().lower() in ("1", "true", "yes", "y", "on")
try:
    from unified_alpaca_client import UnifiedAlpacaClient, ALPACA_PY_AVAILABLE
    _UNIFIED_AVAILABLE = ALPACA_PY_AVAILABLE
except ImportError:
    UnifiedAlpacaClient = None
    _UNIFIED_AVAILABLE = False
from strategies import (
    detect_regime,
    select_strategy,
    DcaConfig,
    sma,
    ema,
    ema_series,
    rsi,
    adx,
    _atr,
    rolling_return,
    max_drawdown,
    current_drawdown,
    lower_lows_persistence,
    base_formation,
    clamp,
)

from intelligence_layer import IntelligenceLayer, IntelligenceContext
# Global Intelligence Layer instance
intelligence_layer = IntelligenceLayer()


# =========================================================
# App + globals
# =========================================================
app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000)

_globals_lock = threading.RLock()
_thread_started: Dict[str, bool] = {}
_thread_start_lock = threading.Lock()
_last_portfolio_ts: float = 0.0
_last_reco_short_ts: float = 0.0
_last_reco_long_ts: float = 0.0

kc: Optional[KrakenClient] = None
alpaca_paper: Optional[AlpacaClient] = None
alpaca_live: Optional[AlpacaClient] = None
bm: Optional[BotManager] = None
KRAKEN_READY: bool = False
KRAKEN_ERROR: str = ""
ALPACA_PAPER_READY: bool = False
ALPACA_LIVE_READY: bool = False
ALPACA_ERROR: str = ""
LIVE_TRADING_ENABLED: bool = os.getenv("LIVE_TRADING_ENABLED", "0").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
LIVE_ENDPOINTS_DISABLED: bool = False
LIVE_ENDPOINTS_DISABLED_REASON: str = ""

_APP_START_TIME: float = time.time()

_STARTUP_STATUS: Dict[str, Any] = {
    "flask_ready": False, "db_ready": False, "db_bots": 0,
    "alpaca_ready": False, "alpaca_buying_power": 0.0,
    "websocket_status": "not_ready", "autopilot_enabled": False, "autopilot_bots": 0,
    "candle_test": None,
}

PORT_HISTORY: List[Dict[str, Any]] = []
PORT_EVERY_SEC = int(os.getenv("PORT_EVERY_SEC", "60"))

_MARKETS_CACHE: Dict[str, Any] = {"ts": 0.0, "markets": None}
MARKETS_TTL_SEC = int(os.getenv("MARKETS_TTL_SEC", "300"))  # 5 minutes
_TICKERS_CACHE: Dict[str, Dict[str, Any]] = {}
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
DISCORD_STATUS_WEBHOOK_URL = os.getenv("DISCORD_STATUS_WEBHOOK_URL", "").strip()
DISCORD_STATUS_MSG_FILE = os.getenv(
    "DISCORD_STATUS_MSG_FILE", "/home/ubuntu/botdata/discord_status_msg_id.txt"
)
DISCORD_STATUS_LOG = os.getenv(
    "DISCORD_STATUS_LOG", "/home/ubuntu/botdata/discord_status.log"
)
AUTO_START_ENABLED = os.getenv("AUTO_START_ENABLED", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
HEALTH_WATCHDOG_SEC = int(os.getenv("HEALTH_WATCHDOG_SEC", "60"))
RECO_SYMBOLS = [
    s.strip() for s in os.getenv(
        "RECO_SYMBOLS",
        ",".join([
            # Top tier
            "XBT/USD","ETH/USD","SOL/USD","XRP/USD","ADA/USD","DOGE/USD","AVAX/USD","LINK/USD",
            "LTC/USD","BCH/USD","DOT/USD","ATOM/USD","XLM/USD","ETC/USD","UNI/USD","AAVE/USD",
            "MATIC/USD","ALGO/USD","TRX/USD","EOS/USD","ICP/USD","FTM/USD","SAND/USD","MANA/USD",
            "GRT/USD","APE/USD","FIL/USD","NEAR/USD","XTZ/USD","HBAR/USD","EGLD/USD","FLOW/USD",
            "KSM/USD","QNT/USD","CRV/USD","COMP/USD","SNX/USD","MKR/USD","ZEC/USD","DASH/USD",
            # Additional popular cryptos
            "BNB/USD","RNDR/USD","INJ/USD","TIA/USD","SUI/USD","SEI/USD","ARB/USD","OP/USD",
            "STRK/USD","IMX/USD","LDO/USD","RUNE/USD","FET/USD","AGIX/USD","OCEAN/USD","AXS/USD",
            "APT/USD","TON/USD","VET/USD","THETA/USD","CHZ/USD","GALA/USD","ENJ/USD","ONE/USD",
            "ZIL/USD","SUSHI/USD","YFI/USD","BAT/USD","CELR/USD","JASMY/USD","WOO/USD","BLUR/USD",
            "PEPE/USD","SHIB/USD","FLOKI/USD","BONK/USD","WIF/USD","BOME/USD","PEOPLE/USD","LUNC/USD",
            # Layer 1/2 and DeFi
            "ROSE/USD","KAVA/USD","MINA/USD","CELO/USD","WAVES/USD","ANT/USD","SRM/USD","OMG/USD",
            "STX/USD","GLMR/USD","MOVR/USD","KLAY/USD","CFX/USD","API3/USD","1INCH/USD","MASK/USD",
            # Gaming and metaverse
            "ILV/USD","ALICE/USD","TLM/USD","YGG/USD","GHST/USD","XYO/USD","PRIME/USD","BIGTIME/USD",
            # AI and data
            "RENDER/USD","GRT/USD","NMR/USD","LPT/USD","BAL/USD","STORJ/USD","AR/USD","KNC/USD",
        ])
    ).split(",")
    if s.strip()
]

# Append Stocks if Alpaca keys are present
_ALPACA_KEYS_PRESENT = bool(os.getenv("ALPACA_API_KEY_LIVE") or os.getenv("ALPACA_API_KEY_PAPER"))
if _ALPACA_KEYS_PRESENT:
    RECO_SYMBOLS.extend([
        # Tech / growth
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "AMD", "INTC", "QCOM", "CRM", "ADBE", "AVGO", "TXN", "PLTR", "ROKU", "SHOP", "PYPL", "SQ",
        # Financials
        "JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "BLK", "C", "AXP",
        # ETFs
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "GLD", "SLV", "TQQQ", "SQQQ", "SOXL", "ARKK",
        # Crypto proxies / miners
        "COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "HOOD",
        # Consumer / retail / travel
        "DIS", "KO", "PEP", "WMT", "TGT", "COST", "HD", "LOW", "MCD", "UBER", "LYFT", "DKNG", "AFRM", "UPST", "CVNA", "GME", "AMC", "SOFI",
        # Healthcare / pharma
        "JNJ", "PG", "PFE", "MRK", "UNH", "T", "VZ", "ABBV", "LLY",
        # Industrial / energy
        "BA", "F", "GM", "XOM", "CVX",
    ])
RECO_MAX_SYMBOLS = int(os.getenv("RECO_MAX_SYMBOLS", "300"))
RECO_SHORT_EVERY_SEC = int(os.getenv("RECO_SHORT_EVERY_SEC", "900"))  # 15m
RECO_LONG_EVERY_SEC = int(os.getenv("RECO_LONG_EVERY_SEC", "3600"))   # 60m
RECO_SHORT_MIN_DAYS = int(os.getenv("RECO_SHORT_MIN_DAYS", "90"))
RECO_LONG_MIN_DAYS = int(os.getenv("RECO_LONG_MIN_DAYS", "180"))
RECO_LONG_MIN_WEEKS = int(os.getenv("RECO_LONG_MIN_WEEKS", "52"))
RECO_MAX_SPREAD_PCT = float(os.getenv("RECO_MAX_SPREAD_PCT", "0.004"))
RECO_MAX_ATR_PCT_SHORT = float(os.getenv("RECO_MAX_ATR_PCT_SHORT", "0.06"))
RECO_ATR_PCT_MODERATE = float(os.getenv("RECO_ATR_PCT_MODERATE", "0.035"))
_RECO_STATE: Dict[str, Dict[str, Any]] = {"short": {}, "long": {}}
_RECO_OHLCV_CACHE: Dict[str, Dict[str, Any]] = {}

# Exclude fiat FX pairs from crypto universe (e.g., AUD/USD, EUR/USD)
FIAT_BASES = {
    "USD", "USDT", "USDC", "EUR", "GBP", "AUD", "CAD", "JPY", "CHF", "NZD",
    "CNY", "HKD", "SGD", "SEK", "NOK", "MXN", "BRL", "ZAR", "TRY", "INR",
    "KRW", "PLN", "CZK", "DKK",
}

# Crypto bases never recommended (not on Kraken spot or problematic)
# STABLE: L1 token with long downtrend, poor profit potential; stablecoin-like names
# Extend via env: RECO_CRYPTO_BLOCKLIST=TOKEN1,TOKEN2,TOKEN3
# BLOCK_MEME_COINS=1 adds meme coins; DEGEN_MODE=1 disables meme blocking
_default_blocklist = {"STABLE", "UST", "USTC", "LUNA2", "LUNA"}  # downtrend / dead / misleading
_env_blocklist = os.getenv("RECO_CRYPTO_BLOCKLIST", "")
_meme_block = set()
if os.getenv("BLOCK_MEME_COINS", "1").strip().lower() in ("1", "true", "yes") and os.getenv("DEGEN_MODE", "0").strip().lower() not in ("1", "true", "yes"):
    _meme_block = {"DOGE", "SHIB", "PEPE", "FLOKI", "BONK", "WIF", "MEME", "BRETT", "POPCAT", "TURBO", "WOJAK", "BOME"}
CRYPTO_BLOCKLIST: set = _default_blocklist | _meme_block | {x.strip().upper() for x in _env_blocklist.split(",") if x.strip()}

# Optional API protection (recommended for real money)
# If set, all /api/* routes require header: X-API-Key: <token>
WORKER_API_TOKEN = os.getenv("WORKER_API_TOKEN", "").strip()

# Rate limiting: max requests per window per IP for /api/* (0 = disabled)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "0"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
_RATE_LIMIT_STORE: Dict[str, List[float]] = {}
_RATE_LIMIT_LOCK = threading.Lock()

RECO_SCAN_MAX_PER_RUN = int(os.getenv("RECO_SCAN_MAX_PER_RUN", "400"))
RECO_SCAN_ERROR_LIMIT = int(os.getenv("RECO_SCAN_ERROR_LIMIT", "8"))
RECO_SCAN_SYMBOL_SLEEP_SEC = float(os.getenv("RECO_SCAN_SYMBOL_SLEEP_SEC", "0.05"))

# Scan profile: conservative (stricter) | balanced | aggressive (looser). Affects thresholds when not overridden by env.
RECO_PROFILE = (os.getenv("RECO_PROFILE", "balanced") or "balanced").strip().lower()
if RECO_PROFILE not in ("conservative", "balanced", "aggressive"):
    RECO_PROFILE = "balanced"

def _reco_buy_threshold_stocks() -> float:
    v = os.getenv("RECO_BUY_THRESHOLD_STOCKS", "").strip()
    if v and v.replace(".", "").replace("-", "").isdigit():
        return float(v)
    return {"conservative": 65.0, "balanced": 60.0, "aggressive": 52.0}.get(RECO_PROFILE, 60.0)

def _reco_buy_threshold_crypto() -> float:
    v = os.getenv("RECO_BUY_THRESHOLD_CRYPTO", "").strip()
    if v and v.replace(".", "").replace("-", "").isdigit():
        return float(v)
    return {"conservative": 45.0, "balanced": 38.0, "aggressive": 32.0}.get(RECO_PROFILE, 38.0)

def _reco_watch_threshold() -> float:
    v = os.getenv("RECO_WATCH_THRESHOLD", "").strip()
    if v and v.replace(".", "").replace("-", "").isdigit():
        return float(v)
    return {"conservative": 45.0, "balanced": 40.0, "aggressive": 35.0}.get(RECO_PROFILE, 40.0)

# Legacy names for code that still references them
RECO_BUY_THRESHOLD_CRYPTO = _reco_buy_threshold_crypto()
RECO_BUY_THRESHOLD_STOCKS = _reco_buy_threshold_stocks()
RECO_WATCH_THRESHOLD = _reco_watch_threshold()

_ALLOWED_TFS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"}


def _start_background_thread(name: str, target) -> None:
    """Start a daemon thread only if not already started (avoids duplicate on hot reload)."""
    with _thread_start_lock:
        if _thread_started.get(name):
            logger.warning("thread %s already started, skipping", name)
            return
        _thread_started[name] = True
    t = threading.Thread(target=target, daemon=True, name=name)
    t.start()
    logger.info("started background thread: %s", name)


# =========================================================
# Auth helper
# =========================================================
def _has_live_bots() -> bool:
    try:
        bots = list_bots()
        return any(not bool(b.get("dry_run", 1)) for b in (bots or []))
    except Exception:
        return False


def _alpaca_any_ready() -> bool:
    return bool(ALPACA_PAPER_READY or ALPACA_LIVE_READY)


def _alpaca_market_open() -> bool:
    """Check if US stock market is open. Returns True on error (don't block)."""
    client = alpaca_live if alpaca_live else alpaca_paper
    if not client:
        return True
    try:
        return bool(client.get_clock().get("is_open", True))
    except Exception:
        return True


def _is_live_endpoint(path: str, method: str) -> bool:
    if not path.startswith("/api/"):
        return False
    if method not in ("POST", "PUT", "DELETE"):
        return False
    return False


def _require_api_key(
    path: str, api_key: Optional[str], client_host: Optional[str] = None, force: bool = False,
    referer: Optional[str] = None, host_header: Optional[str] = None,
) -> Tuple[bool, Optional[str], int]:
    if not WORKER_API_TOKEN:
        if force:
            return False, "Live endpoints disabled: WORKER_API_TOKEN is required.", 503
        return True, None, 200
    if not path.startswith("/api/"):
        return True, None, 200
    host = (client_host or "").strip()
    if not force and host in ("127.0.0.1", "::1", "localhost"):
        return True, None, 200
    # Same-origin bypass: allow when Referer/Host match (UI page fetching from own API)
    if not force and (referer or host_header):
        try:
            from urllib.parse import urlparse
            ref_host = urlparse(referer or "").netloc.split(":")[0] if referer else ""
            req_host = (host_header or "").split(":")[0].strip()
            allowed = os.getenv("API_ALLOWED_HOSTS", "localhost,127.0.0.1,::1")
            allowed_set = {h.strip().lower() for h in allowed.split(",") if h.strip()}
            for h in (ref_host, req_host):
                if h and h.lower() in allowed_set:
                    return True, None, 200
        except Exception:
            pass
    if not api_key or api_key.strip() != WORKER_API_TOKEN:
        return False, "Unauthorized (missing/invalid X-API-Key)", 401
    return True, None, 200


def _rate_limit_check(ip: str) -> Optional[str]:
    """Return error msg if rate limited, else None."""
    if RATE_LIMIT_REQUESTS <= 0:
        return None
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW_SEC
    with _RATE_LIMIT_LOCK:
        timestamps = _RATE_LIMIT_STORE.get(ip, [])
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= RATE_LIMIT_REQUESTS:
            return "Rate limit exceeded. Try again later."
        timestamps.append(now)
        _RATE_LIMIT_STORE[ip] = timestamps[-500:]  # cap memory
    return None


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    path = request.url.path
    client_host = request.client.host if request.client else ""
    if path.startswith("/api/") and RATE_LIMIT_REQUESTS > 0:
        err = _rate_limit_check(client_host)
        if err:
            return _json({"ok": False, "error": err}, 429)
    api_key = request.headers.get("X-API-Key")
    referer = request.headers.get("Referer")
    host_header = request.headers.get("Host")
    ok, msg, status = _require_api_key(
        path, api_key, client_host=client_host, referer=referer, host_header=host_header
    )
    if not ok:
        return _json({"ok": False, "error": msg or "Unauthorized"}, status)
    return await call_next(request)


@app.post("/api/bots")
async def api_create_bot(request: Request):
    """Single handler for bot creation. Validates symbol, applies caps, returns full bot."""
    if not bm:
        return _json({"ok": False, "error": "BotManager not initialized"}, 503)
    payload = await request.json()
    if not isinstance(payload, dict):
        return _json({"ok": False, "error": "Invalid payload"}, 400)

    raw_sym = str(payload.get("symbol") or "").strip()
    detected_type = classify_symbol(raw_sym) if raw_sym else "crypto"
    market_type_val = "stocks" if detected_type == "stock" else "crypto"
    if market_type_val == "crypto" and raw_sym:
        resolved, err = _validate_crypto_symbol(raw_sym)
        if err:
            return _json({"ok": False, "error": err}, 400)
        raw_sym = resolved or raw_sym
    symbol = _resolve_symbol(raw_sym)
    name = str(payload.get("name") or f"Bot {symbol}")

    base_quote = float(payload.get("base_quote") or 20.0)
    safety_quote = float(payload.get("safety_quote") or 20.0)
    max_safety = int(payload.get("max_safety") or 5)
    max_spend_quote = float(payload.get("max_spend_quote") or (base_quote + (safety_quote * max_safety)))
    if base_quote > max_spend_quote * 0.5 or base_quote > 100:
        base_quote = min(max(5.0, max_spend_quote * 0.15), 100.0)
    if safety_quote > max_spend_quote * 0.3 or safety_quote > 75:
        safety_quote = min(max(5.0, max_spend_quote * 0.10), 75.0)

    data = {
        "name": name,
        "symbol": symbol,
        "enabled": int(payload.get("enabled", 1)),
        "dry_run": int(payload.get("dry_run", 1)) if "dry_run" in payload else 1,
        "base_quote": base_quote,
        "safety_quote": safety_quote,
        "max_safety": max_safety,
        "first_dev": float(payload.get("first_dev") or 0.015),
        "step_mult": float(payload.get("step_mult") or 1.2),
        "tp": float(payload.get("tp") or 0.015),
        "max_spend_quote": max_spend_quote,
        "strategy_mode": str(payload.get("strategy_mode") or "auto"),
        "forced_strategy": str(payload.get("forced_strategy") or ""),
        "max_open_orders": int(payload.get("max_open_orders") or 6),
        "market_type": market_type_val,
        "alpaca_mode": str(payload.get("alpaca_mode") or "paper"),
    }
    _sanitize_bot_numbers(data)
    try:
        bot_id = create_bot(data)
        bot = get_bot(int(bot_id))
        return _json({"ok": True, "bot": bot})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/bots")
def api_bots():
    try:
        bots = list_bots()
        return _json({
            "ok": True,
            "bots": bots,
            "kraken_ready": _kraken_ready(),
            "kraken_error": KRAKEN_ERROR,
        })
    except Exception as e:
        logger.exception("api_bots failed")
        return _json({"ok": False, "error": str(e), "bots": []}, 500)


@app.get("/api/bots/stream")
async def api_bots_stream():
    """SSE endpoint for real-time bot status updates (11.md Part 7)."""
    import asyncio

    async def event_gen():
        while True:
            try:
                bots = list_bots()
                data = json.dumps(bots)
                yield f"data: {data}\n\n"
            except Exception as e:
                logger.debug("bots/stream: %s", e)
            await asyncio.sleep(5)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/bots/summary")
def api_bots_summary():
    try:
        bots = list_bots()
    except Exception as e:
        logger.exception("api_bots_summary list_bots failed")
        return _json({"ok": False, "error": str(e), "total": 0, "running": 0, "paused": False}, 500)
    total = len(bots)
    running = 0
    if bm is not None:
        for b in bots:
            try:
                snap = bm.snapshot(int(b.get("id")))
                if snap.get("running"):
                    running += 1
            except Exception:
                logger.exception("api_bots_summary: snapshot failed bot_id=%s", b.get("id"))
    paused = bool(_pause_state())
    return _json({"ok": True, "total": total, "running": running, "paused": paused})




# =========================================================
# Helpers
# =========================================================

def _sanitize_bot_numbers(data: Dict[str, Any]) -> None:
    """Clamp critical bot numeric fields to sane ranges (mutates data in place)."""
    def clamp(key: str, lo: float, hi: float, default: float) -> None:
        if key not in data:
            return
        try:
            v = float(data[key])
        except (TypeError, ValueError):
            data[key] = default
            return
        data[key] = max(lo, min(hi, v))

    clamp("first_dev", 0.001, 0.5, 0.015)
    clamp("step_mult", 1.0, 10.0, 1.2)
    clamp("tp", 0.001, 0.5, 0.015)
    clamp("daily_loss_limit_pct", 0.01, 0.25, 0.06)
    clamp("max_drawdown_pct", 0.0, 0.99, 0.0)
    clamp("trailing_activation_pct", 0.001, 0.5, 0.02)
    clamp("trailing_distance_pct", 0.001, 0.2, 0.01)
    clamp("spread_guard_pct", 0.0005, 0.05, 0.003)
    clamp("min_gap_pct", 0.001, 0.1, 0.003)
    clamp("max_gap_pct", 0.01, 0.3, 0.06)
    if "max_safety" in data:
        try:
            v = int(data["max_safety"])
            data["max_safety"] = max(1, min(20, v))
        except (TypeError, ValueError):
            data["max_safety"] = 5
    if "poll_seconds" in data:
        try:
            v = int(data["poll_seconds"])
            data["poll_seconds"] = max(5, min(300, v))
        except (TypeError, ValueError):
            data["poll_seconds"] = 10


def _kraken_ready() -> bool:
    return bool(kc is not None and bm is not None and KRAKEN_READY)


def _json(payload: Dict[str, Any], status_code: int = 200) -> JSONResponse:
    return JSONResponse(payload, status_code=status_code, headers={'Cache-Control': 'no-store'})


def _serialize_order(o: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": o.get("id"),
        "symbol": o.get("symbol"),
        "side": o.get("side"),
        "type": o.get("type"),
        "price": o.get("price"),
        "amount": o.get("amount"),
        "remaining": o.get("remaining"),
        "status": o.get("status"),
        "timestamp": o.get("timestamp"),
        "client_order_id": o.get("clientOrderId") or o.get("client_id"),
    }


def _midnight_local_ts() -> int:
    lt = time.localtime()
    return int(time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, lt.tm_wday, lt.tm_yday, lt.tm_isdst)))


def _markets() -> Dict[str, Any]:
    if not _kraken_ready():
        return {}
    now = time.time()
    with _globals_lock:
        cached = _MARKETS_CACHE.get("markets")
        if cached is None or (now - float(_MARKETS_CACHE.get("ts", 0.0))) > MARKETS_TTL_SEC:
            try:
                _MARKETS_CACHE["markets"] = kc.load_markets()
                _MARKETS_CACHE["ts"] = now
            except Exception as e:
                logger.exception("_markets: load_markets failed")
                if _MARKETS_CACHE["markets"] is None:
                    _MARKETS_CACHE["markets"] = {}
                _MARKETS_CACHE["ts"] = now
        return _MARKETS_CACHE["markets"] or {}


def _strategy_display_name(mode: Optional[str]) -> str:
    """Map internal strategy_mode to Explore-friendly label."""
    if not mode or not str(mode).strip():
        return "DCA"
    m = str(mode).strip().lower()
    if m == "trend_follow":
        return "Trend Follow"
    if m == "range_mean_reversion":
        return "Range / Mean Reversion"
    if m == "high_vol_defensive":
        return "High Vol Defensive"
    if m in ("smart_dca", "smart dca"):
        return "Smart DCA"
    if m in ("classic_dca", "classic", "dca"):
        return "DCA"
    return mode.replace("_", " ").title()


def _normalize_symbol(user_symbol: str) -> str:
    """
    Accepts:
      btcusd, BTCUSD, BTC-USD, BTC/USD, xbtusd, XBTUSD, etc.
    Produces:
      XBT/USD, ETH/USD, etc.
    """
    s = (user_symbol or "").strip().upper().replace(" ", "")
    s = s.replace("-", "/")
    if "/" not in s and len(s) >= 6:
        s = f"{s[:-3]}/{s[-3:]}"
    parts = s.split("/", 1)
    base = parts[0] if len(parts) > 0 else ""
    quote = parts[1] if len(parts) > 1 else ""
    if base == "BTC":
        base = "XBT"
    return f"{base}/{quote}" if base and quote else s


def _resolve_symbol(symbol_from_db_or_user: str) -> str:
    """
    Resolve normalized symbol to a valid Kraken market key when possible.
    For stock-like symbols (short, no slash), return as-is without Kraken validation.
    """
    s = _normalize_symbol(symbol_from_db_or_user or "")
    
    # Stock symbols: short (< 6 chars) and no slash -> return as-is, skip Kraken check
    if len(s) < 6 and "/" not in s:
        return s
    
    mk = _markets()
    if not mk:
        return s

    if s in mk:
        return s

    if s.startswith("BTC/"):
        alt = s.replace("BTC/", "XBT/", 1)
        if alt in mk:
            return alt
    if s.startswith("XBT/"):
        alt = s.replace("XBT/", "BTC/", 1)
        if alt in mk:
            return alt

    return s


def _validate_crypto_symbol(symbol: str) -> tuple:
    """
    Validate crypto symbol against Kraken markets.
    Returns (resolved_symbol, None) if valid, or (None, error_message) if invalid.
    """
    s = _normalize_symbol(symbol or "")
    if not s or "/" not in s:
        return None, "Symbol must be in format BASE/USD (e.g. XBT/USD, ETH/USD)"
    mk = _markets()
    if not mk:
        return s, None  # Skip validation if Kraken not ready
    if s in mk:
        return s, None
    if s.startswith("BTC/"):
        alt = s.replace("BTC/", "XBT/", 1)
        if alt in mk:
            return alt, None
    if s.startswith("XBT/"):
        alt = s.replace("XBT/", "BTC/", 1)
        if alt in mk:
            return alt, None
    # Build suggestions from popular Kraken USD pairs
    popular = ["XBT/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOT/USD"]
    avail = [k for k in mk if str(mk.get(k, {}).get("quote", "")).upper() == "USD"][:8]
    suggestions = [p for p in popular if p in mk] or (avail[:5] if avail else ["XBT/USD", "ETH/USD"])
    return None, f"Symbol not found on Kraken: {s}. Try: {', '.join(suggestions)}"


def _sanitize_tf(tf: str) -> str:
    t = (tf or "5m").strip()
    return t if t in _ALLOWED_TFS else "5m"


def _tf_seconds(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1]) * 60
    if tf.endswith("h"):
        return int(tf[:-1]) * 3600
    if tf.endswith("d"):
        return int(tf[:-1]) * 86400
    if tf.endswith("w"):
        return int(tf[:-1]) * 604800
    if tf.endswith("M"):
        return int(tf[:-1]) * 2592000
    return 300


def _safe_last_price(symbol: str) -> Optional[float]:
    """
    Get last price for a CRYPTO symbol from Kraken.
    
    GUARDRAIL: This function is for CRYPTO symbols only.
    Use AlpacaClient.get_latest_quote() for stock symbols.
    """
    # Guardrail: Prevent stock symbols from being routed to Kraken
    validate_symbol_type(symbol, "crypto", "_safe_last_price")
    
    if not _kraken_ready():
        return None
    try:
        s = _resolve_symbol(symbol)
        mk = _markets()
        if mk and s in mk:
            return float(kc.fetch_ticker_last(s))
        return None
    except Exception:
        logger.exception("_safe_last_price: fetch failed symbol=%s", symbol)
        return None


def _ticker_cached(symbol: str, ttl_sec: int = 30) -> Optional[Dict[str, Any]]:
    now = time.time()
    with _globals_lock:
        cached = _TICKERS_CACHE.get(symbol)
        if cached and (now - float(cached.get("ts", 0.0))) < ttl_sec:
            return cached.get("data")
        cached_save = cached

    data = None
    try:
        if _kraken_ready():
            mk = _markets()
            if (mk and symbol in mk) or "/" in symbol:
                t = kc.fetch_ticker(symbol)
                last_raw = float(t.get("last") or t.get("close") or 0)
                data = {
                    "symbol": symbol,
                    "last": last_raw,
                    "bid": t.get("bid"), "ask": t.get("ask"),
                    "percentage": t.get("percentage"), "change": t.get("change"),
                    "baseVolume": t.get("baseVolume"), "quoteVolume": t.get("quoteVolume"),
                }
                try:
                    from data_validator import get_validated_crypto_price
                    validated, alert = get_validated_crypto_price(last_raw, symbol, kc)
                    if validated > 0:
                        data["last"] = validated
                    if alert:
                        logger.warning("Price divergence %s >2%%", symbol)
                except ImportError:
                    pass
    except Exception as e:
        logger.exception("_ticker_cached: Kraken fetch failed symbol=%s", symbol)
        try:
            from circuit_breaker import record_api_failure, check_and_trigger_emergency
            record_api_failure(bot_id=None, source="kraken_ticker")
            check_and_trigger_emergency()
        except ImportError:
            pass

    if not data and (alpaca_paper or alpaca_live):
        try:
            client = alpaca_live if alpaca_live else alpaca_paper
            snaps = client.get_snapshots([symbol])
            snap = snaps.get(symbol)
            if snap:
                daily = snap.get("dailyBar") or {}
                prev = snap.get("prevDailyBar") or {}
                latest = snap.get("latestTrade") or {}
                last_price = float(latest.get("p") or daily.get("c") or 0.0)
                prev_close = float(prev.get("c") or 0.0)
                change = last_price - prev_close if last_price and prev_close else 0.0
                pct = (change / prev_close) * 100.0 if prev_close else 0.0
                vol_shares = float(daily.get("v") or 0.0)
                data = {
                    "symbol": symbol, "last": last_price,
                    "bid": float(snap.get("latestQuote", {}).get("bp") or 0.0),
                    "ask": float(snap.get("latestQuote", {}).get("ap") or 0.0),
                    "percentage": pct, "change": change,
                    "baseVolume": vol_shares, "quoteVolume": vol_shares * last_price,
                }
            else:
                t = client.get_ticker(symbol)
                if t.get("last"):
                    data = {"symbol": symbol, "last": t.get("last"), "bid": t.get("bid"),
                            "ask": t.get("ask"), "percentage": 0.0, "change": 0.0,
                            "baseVolume": 0.0, "quoteVolume": 0.0}
        except Exception as e:
            logger.exception("_ticker_cached: Alpaca fetch failed symbol=%s", symbol)
            try:
                from circuit_breaker import record_api_failure, check_and_trigger_emergency
                record_api_failure(bot_id=None, source="alpaca_ticker")
                check_and_trigger_emergency()
            except ImportError:
                pass

    with _globals_lock:
        if data:
            bid = float(data.get("bid") or 0)
            ask = float(data.get("ask") or 0)
            last = float(data.get("last") or 0)
            mid = (bid + ask) / 2 if bid and ask else last
            if mid > 0 and bid and ask:
                spread_pct = abs(ask - bid) / mid * 100
                if spread_pct > 2.0:
                    try:
                        from db import log_data_quality
                        log_data_quality("ticker", "extreme_spread", "warning", {"symbol": symbol, "spread_pct": round(spread_pct, 2)})
                    except Exception:
                        pass
            _TICKERS_CACHE[symbol] = {"ts": now, "data": data}
            return data
        if cached_save:
            cached_ts = float(cached_save.get("ts", 0))
            if now - cached_ts > 300:
                try:
                    from db import log_data_quality
                    log_data_quality("ticker", "stale_price", "warning", {"symbol": symbol, "age_sec": int(now - cached_ts)})
                except Exception:
                    pass
            return cached_save.get("data")
    return None


_TICKERS_BATCH_CACHE: Dict[str, Any] = {"ts": 0.0, "data": {}}


def _tickers_batch_cached(ttl_sec: int = 15) -> Dict[str, Dict[str, Any]]:
    """Fetch all Kraken tickers once, cache briefly. Use for batch price lookups (Explore)."""
    now = time.time()
    with _globals_lock:
        if (now - _TICKERS_BATCH_CACHE.get("ts", 0.0)) < ttl_sec and _TICKERS_BATCH_CACHE.get("data"):
            return _TICKERS_BATCH_CACHE["data"]
    out: Dict[str, Dict[str, Any]] = {}
    try:
        if _kraken_ready() and kc:
            raw = kc.ex.fetch_tickers()
            for sym, t in (raw or {}).items():
                if not isinstance(t, dict):
                    continue
                out[sym] = {"symbol": sym, "last": t.get("last") or t.get("close"),
                            "percentage": t.get("percentage"), "quoteVolume": t.get("quoteVolume")}
            with _globals_lock:
                _TICKERS_BATCH_CACHE["ts"] = now
                _TICKERS_BATCH_CACHE["data"] = out
    except Exception as e:
        logger.exception("_tickers_batch_cached: fetch failed")
    return out


def _slope(values: List[float], n: int = 20) -> Optional[float]:
    if len(values) < n:
        return None
    ys = values[-n:]
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs) or 1.0
    return num / den


def _ohlcv_cached(symbol: str, timeframe: str, limit: int, ttl_sec: int) -> List[List[float]]:
    """
    Fetch OHLCV data for a CRYPTO symbol from Kraken with caching.
    
    GUARDRAIL: This function is for CRYPTO symbols only.
    Use AlpacaClient.get_ohlcv() for stock symbols.
    """
    # Guardrail: Prevent stock symbols from being routed to Kraken
    validate_symbol_type(symbol, "crypto", "_ohlcv_cached")
    
    if not _kraken_ready():
        return []
    key = f"{symbol}|{timeframe}|{limit}"
    now = time.time()
    with _globals_lock:
        cached = _RECO_OHLCV_CACHE.get(key)
        if cached and (now - float(cached.get("ts", 0.0))) < ttl_sec:
            return cached.get("data") or []

    data = []
    try:
        is_crypto = "/" in symbol
        if not is_crypto and (alpaca_live or alpaca_paper):
            client = alpaca_live if alpaca_live else alpaca_paper
            data = client.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not data and kc:
            data = kc.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        logger.exception("_ohlcv_cached: fetch failed symbol=%s tf=%s", symbol, timeframe)

    with _globals_lock:
        _RECO_OHLCV_CACHE[key] = {"ts": now, "data": data}
    return data


def _safe_spread_pct(symbol: str) -> Optional[float]:
    try:
        t = kc.fetch_ticker(symbol)
        bid = float(t.get("bid") or 0.0)
        ask = float(t.get("ask") or 0.0)
        mid = (bid + ask) / 2 if bid and ask else 0.0
        if mid <= 0:
            return None
        return float((ask - bid) / mid)
    except Exception:
        return None


def _btc_context() -> Dict[str, Any]:
    ctx = {"risk_off": False, "labels": {}, "scores": {}}
    try:
        sym = _resolve_symbol("XBT/USD")
        c4h = _ohlcv_cached(sym, "4h", 200, 300)
        c1d = _ohlcv_cached(sym, "1d", 400, 900)
        r4h = detect_regime(c4h)
        r1d = detect_regime(c1d)
        ctx["labels"] = {"4h": r4h.regime, "1d": r1d.regime}
        ctx["scores"] = {"4h": r4h.scores or {}, "1d": r1d.scores or {}}
        down = max((r1d.scores or {}).get("downtrend_score", 0.0), (r4h.scores or {}).get("downtrend_score", 0.0))
        hv = max((r1d.scores or {}).get("high_vol_score", 0.0), (r4h.scores or {}).get("high_vol_score", 0.0))
        if down >= 0.6 or hv >= 0.6 or r1d.regime in ("TREND_DOWN", "HIGH_VOL_RISK", "RISK_OFF"):
            ctx["risk_off"] = True
    except Exception:
        pass
    return ctx


def _trend_age(ema_fast: List[float], ema_slow: List[float], max_weeks: int = 52) -> int:
    if not ema_fast or not ema_slow or len(ema_fast) != len(ema_slow):
        return 0
    age = 0
    for a, b in zip(reversed(ema_fast[-max_weeks:]), reversed(ema_slow[-max_weeks:])):
        if a > b:
            age += 1
        else:
            break
    return age


def _scan_symbol(symbol: str, horizon: str, btc_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scan a symbol using the Intelligence Layer.
    Routes to appropriate provider based on symbol type (stock → Alpaca, crypto → Kraken).
    """
    market_type = classify_symbol(symbol)
    
    # Route based on symbol type
    if market_type == "stock":
        # Stock path - use Alpaca
        client = alpaca_live if alpaca_live else alpaca_paper
        if not client:
            # Return empty scan with error flag
            logger.warning(f"Stock symbol {symbol} scanned but Alpaca not configured")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": ["Stock provider (Alpaca) not configured"],
                "risk_flags": ["NO_STOCK_PROVIDER"],
                "metrics": {"market_type": "stock"},
                "regime": {}
            }
        
        # Fetch stock data: Alpaca first, Yahoo fallback when Alpaca returns no/insufficient data
        try:
            from phase2_data_fetcher import fetch_recent_candles
            candles_1h = fetch_recent_candles(symbol, "1h", 300)
            candles_4h = fetch_recent_candles(symbol, "4h", 300)
            candles_1d = fetch_recent_candles(symbol, "1d", 500)
            candles_1w = fetch_recent_candles(symbol, "1w", 300)
            
            # Get current price from ticker
            ticker_data = client.get_ticker(symbol)
            price = float(ticker_data.get("last", 0.0))
            spread_pct = 0.001  # Stocks typically have tight spreads (~0.1%)
            
            # Stock-specific metadata for intelligence scoring
            from stock_metadata import get_sector, get_liquidity_tier
            sector = get_sector(symbol)
            vol_24h = 0.0
            if candles_1d and len(candles_1d[-1]) >= 6:
                vol_24h = float(candles_1d[-1][5])
            liquidity_tier = get_liquidity_tier(price, vol_24h) if price and vol_24h else "unknown"
            stock_market_breadth = {
                "is_stock": True,
                "sector": sector,
                "liquidity_tier": liquidity_tier,
                "volume_24h": vol_24h,
            }
            # Earnings, analyst, sector ETF, market cap, IPO
            try:
                from earnings_calendar import days_until_earnings
                earnings_days = days_until_earnings(symbol)
                stock_market_breadth["earnings_days"] = earnings_days
            except Exception:
                pass
            try:
                from analyst_ratings_tracker import get_analyst_ratings, analyst_score_contribution
                ar = get_analyst_ratings(symbol)
                stock_market_breadth["analyst_consensus"] = ar.get("consensus")
                stock_market_breadth["analyst_score_delta"] = ar.get("score_delta", 0)
                contrib, _ = analyst_score_contribution(symbol)
                stock_market_breadth["analyst_score_contrib"] = contrib
            except Exception:
                pass
            try:
                from sector_etf_correlation import sector_etf_trend_ok
                sector_ok, sector_reason = sector_etf_trend_ok(symbol, candles_1d=candles_1d)
                stock_market_breadth["sector_etf_ok"] = sector_ok
                stock_market_breadth["sector_etf_reason"] = sector_reason
            except Exception:
                stock_market_breadth["sector_etf_ok"] = True
            try:
                from dividend_calendar import days_to_exdiv
                dte, amt, yld = days_to_exdiv(symbol, price)
                stock_market_breadth["days_to_exdiv"] = dte
                stock_market_breadth["dividend_yield_pct"] = yld
            except Exception:
                pass
            try:
                from stock_profile import get_stock_profile, passes_market_cap_filter, is_recent_ipo
                p = get_stock_profile(symbol)
                stock_market_breadth["market_cap_b"] = p.get("market_cap_b")
                stock_market_breadth["market_cap_tier"] = p.get("tier")
                stock_market_breadth["days_listed"] = p.get("days_listed")
                mc_ok, mc_reason = passes_market_cap_filter(symbol)
                stock_market_breadth["market_cap_ok"] = mc_ok
                stock_market_breadth["market_cap_reason"] = mc_reason
                recent_ipo, days = is_recent_ipo(symbol)
                stock_market_breadth["recent_ipo"] = recent_ipo
                stock_market_breadth["ipo_days"] = days
            except Exception:
                stock_market_breadth["market_cap_ok"] = True
                stock_market_breadth["recent_ipo"] = False
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": [f"Data fetch error: {str(e)[:100]}"],
                "risk_flags": ["DATA_ERROR"],
                "metrics": {"market_type": "stocks"},
                "regime": {}
            }
        market_breadth = stock_market_breadth
    else:
        # Crypto path - use Kraken (with guardrails now in place)
        if not _kraken_ready():
            logger.warning(f"Crypto symbol {symbol} scanned but Kraken not ready")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": ["Kraken not available"],
                "risk_flags": ["NO_CRYPTO_PROVIDER"],
                "metrics": {"market_type": "crypto"},
                "regime": {}
            }
        
        sym = _resolve_symbol(symbol)
        market_breadth = {}
        
        # Fetch crypto data from Kraken (guardrails will catch any misrouted stocks)
        try:
            candles_4h = _ohlcv_cached(sym, "4h", 300, 300)
            candles_1d = _ohlcv_cached(sym, "1d", 500, 900)
            candles_1w = _ohlcv_cached(sym, "1w", 300, 1800)
            candles_1h = _ohlcv_cached(sym, "1h", 300, 300)
            
            price = _safe_last_price(sym) or 0.0
            spread_pct = _safe_spread_pct(sym)
        except ValueError as e:
            # Guardrail caught a routing error
            logger.error(f"Routing error for {symbol}: {e}")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": [f"Routing error: {str(e)[:100]}"],
                "risk_flags": ["ROUTING_ERROR"],
                "metrics": {"market_type": "crypto"},
                "regime": {}
            }
        except Exception as e:
            logger.error(f"Error fetching crypto data for {symbol}: {e}")
            return {
                "symbol": symbol,
                "score": 0.0,
                "eligible": False,
                "data_ok": False,
                "reasons": [f"Data fetch error: {str(e)[:100]}"],
                "risk_flags": ["DATA_ERROR"],
                "metrics": {"market_type": "crypto"},
                "regime": {}
            }
    
    # Create Intelligence Context (same for both stock and crypto)
    # Note: For recommendations we don't have a specific bot config or account state,
    # so we pass defaults or "research mode" values.
    # market_breadth: for stocks includes sector, liquidity_tier, earnings_days, analyst, sector_etf, etc.
    bot_config = {}
    if market_type == "stock" and market_breadth:
        ed = market_breadth.get("earnings_days")
        if ed is not None:
            bot_config["earnings_days"] = ed
    context = IntelligenceContext(
        symbol=symbol,  # Use original symbol, not resolved
        last_price=price,
        spread_pct=spread_pct,
        candles_1h=candles_1h or [],
        candles_4h=candles_4h or [],
        candles_1d=candles_1d or [],
        candles_1w=candles_1w or [],
        btc_context=btc_ctx,
        market_breadth=market_breadth,
        bot_config=bot_config,
        now_ts=int(time.time()),
        last_price_ts=int(time.time()),  # Assuming fresh since we just fetched
        dry_run=True,
    )
    
    # Generate Recommendation via Intelligence Layer
    recommendation = intelligence_layer.generate_recommendation(context, horizon)
    
    if "metrics" in recommendation and isinstance(recommendation["metrics"], dict):
        recommendation["metrics"]["market_type"] = "stocks" if market_type == "stock" else "crypto"
        if market_type == "crypto":
            try:
                from funding_rate_tracker import get_funding_rate
                from crypto_cycle_detector import get_cycle_phase
                fr = get_funding_rate(symbol)
                if fr:
                    recommendation["metrics"]["funding_rate"] = fr.get("rate")
                    recommendation["metrics"]["funding_signal"] = fr.get("signal")
                cyc = get_cycle_phase()
                recommendation["metrics"]["cycle_phase"] = cyc.get("phase")
            except Exception:
                pass

    # Benchmark & competitive analysis (SPY for stocks, BTC for crypto)
    try:
        from benchmark_analyzer import enrich_recommendation_with_benchmark
        candles_1d = context.candles_1d if hasattr(context, "candles_1d") else []
        benchmark_candles = None
        if len(candles_1d or []) >= 30:
            if market_type == "stock" and (alpaca_live or alpaca_paper):
                client = alpaca_live or alpaca_paper
                try:
                    benchmark_candles = client.get_ohlcv("SPY", "1d", 200)
                except Exception:
                    pass
            elif market_type == "crypto" and _kraken_ready():
                try:
                    benchmark_candles = _ohlcv_cached("XBT/USD", "1d", 200, 300)
                except Exception:
                    pass
            sector = (context.market_breadth or {}).get("sector") if hasattr(context, "market_breadth") else None
            enriched = enrich_recommendation_with_benchmark(
                symbol, price, candles_1d=candles_1d, benchmark_candles=benchmark_candles, sector=sector
            )
            for k, v in enriched.items():
                if v is not None and v != "":
                    recommendation["metrics"][k] = v
            # Score boost for top-quartile peer performers
            if enriched.get("peer_quartile") == "top":
                base = float(recommendation.get("score") or 0)
                recommendation["score"] = min(98.0, base + 3.0)
                recommendation.setdefault("reasons", []).append("Top-quartile in sector")
            # Add benchmark comparison to reasons when available
            if enriched.get("benchmark_vs"):
                recommendation.setdefault("reasons", []).append(enriched["benchmark_vs"])
    except Exception as e:
        logger.debug("Benchmark enrichment failed for %s: %s", symbol, e)
    
    # Explore V2: hard gates (when enabled)
    try:
        from explore_v2 import apply_universe_gates, enhance_score, is_enabled as explore_v2_enabled
        if explore_v2_enabled():
            metrics = recommendation.get("metrics") or {}
            spread_bps = float(spread_pct or 0) * 10000.0
            vol_pct = metrics.get("volatility_pct") or metrics.get("atr_pct")
            vol_avg = metrics.get("volatility_avg_pct")
            volume_24h = metrics.get("volume_24h_quote")
            pass_gate, fail_reason = apply_universe_gates(
                symbol, volume_24h_quote=volume_24h,
                spread_bps=spread_bps if spread_bps > 0 else None,
                volatility_pct=float(vol_pct) if vol_pct is not None else None,
                volatility_avg_pct=float(vol_avg) if vol_avg is not None else None,
            )
            if not pass_gate:
                recommendation["eligible"] = False
                recommendation["score"] = 0.0
                recommendation.setdefault("risk_flags", []).append(f"EXPLORE_V2_GATE:{fail_reason}")
            else:
                base_score = float(recommendation.get("score") or 0)
                adj_score, extra_reasons = enhance_score(
                    base_score, recommendation,
                    regime=str(metrics.get("regime") or ""),
                    spread_bps=spread_bps if spread_bps > 0 else None,
                    volatility_pct=float(vol_pct) if vol_pct is not None else None,
                )
                recommendation["score"] = adj_score
                recommendation["reasons"] = (recommendation.get("reasons") or []) + extra_reasons
    except ImportError:
        pass
    
    return recommendation


def _analyze_market_data(
    symbol: str, 
    horizon: str, 
    btc_ctx: Dict[str, Any], 
    candles_1h: List[List[float]],
    candles_4h: List[List[float]],
    candles_1d: List[List[float]],
    candles_1w: List[List[float]]
) -> Dict[str, Any]:
    """
    Legacy / Helper: Kept for signature compatibility or direct analysis calls if needed.
    Now just routes to IntelligenceLayer as well.
    """
    # Simply wrap into _scan_symbol logic which builds context
    # Use the provided candles instead of fetching again
    price = 0.0
    if candles_1d: price = float(candles_1d[-1][4])
    elif candles_4h: price = float(candles_4h[-1][4])
    
    context = IntelligenceContext(
        symbol=symbol,
        last_price=price,
        candles_1h=candles_1h,
        candles_4h=candles_4h,
        candles_1d=candles_1d,
        candles_1w=candles_1w,
        btc_context=btc_ctx,
        now_ts=int(time.time()),
        last_price_ts=int(time.time()),
        dry_run=True
    )
    return intelligence_layer.generate_recommendation(context, horizon)


# Global Cache for Universe
_UNIVERSE_CACHE = {"ts": 0, "symbols": []}
UNIVERSE_TTL = 600  # 10 minutes

# Momentum-based universe filtering (stocks)
_MOMENTUM_FILTER_ENABLED = os.getenv("RECO_MOMENTUM_FILTER", "1").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
    "on",
)
_MOMENTUM_FILTER_MIN_SCORE = float(os.getenv("RECO_MOMENTUM_MIN_SCORE", "60"))
_MOMENTUM_FILTER_TOP_N_STOCKS = int(os.getenv("RECO_MOMENTUM_TOP_N_STOCKS", "200"))


def _reco_symbols(quote: str = "USD") -> List[str]:
    with _globals_lock:
        if time.time() - _UNIVERSE_CACHE["ts"] < UNIVERSE_TTL and _UNIVERSE_CACHE["symbols"]:
            return list(_UNIVERSE_CACHE["symbols"])

    symbols = set()
    mk = _markets() if _kraken_ready() else {}

    # 1. Fallback / Configured Symbols (crypto: only if on Kraken)
    for s in RECO_SYMBOLS:
        if "/" in s and mk:
            base = (s.split("/")[0] or "").upper()
            if base in CRYPTO_BLOCKLIST:
                continue
            resolved, err = _validate_crypto_symbol(s)
            if resolved:
                symbols.add(resolved)
        elif "/" not in s:
            symbols.add(s)

    # 2. Crypto via Kraken – whole market: all /USD pairs with meaningful volume
    _min_crypto_volume = float(os.getenv("RECO_CRYPTO_MIN_VOLUME", "5000"))
    _max_crypto = int(os.getenv("RECO_CRYPTO_TOP_N", "350"))
    try:
        if kc and _kraken_ready():
            tickers = kc.ex.fetch_tickers()
            sorted_tickers = sorted(
                [
                    (s, float(t.get("quoteVolume") or 0))
                    for s, t in tickers.items()
                    if "/USD" in s
                    and float(t.get("quoteVolume") or 0) >= _min_crypto_volume
                    and (s.split("/")[0] or "").upper() not in FIAT_BASES
                ],
                key=lambda x: x[1],
                reverse=True
            )
            for s, _ in sorted_tickers[:_max_crypto]:
                base = (s.split("/")[0] or "").upper()
                if base not in CRYPTO_BLOCKLIST:
                    symbols.add(s)
    except Exception as e:
        logger.error(f"Error fetching crypto universe: {e}")

    # 3. Stocks via Alpaca – whole market: movers + broad bluechip + active assets
    try:
        client = alpaca_live if alpaca_live else alpaca_paper
        if client:
            actives = client.get_top_movers()
            for item in actives.get("hot", []):
                symbols.add(item["symbol"])
            for item in actives.get("gainers", []):
                symbols.add(item["symbol"])
            for item in actives.get("losers", []):
                symbols.add(item["symbol"])
            bluechips = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX",
                "AMD", "INTC", "QCOM", "CRM", "ADBE", "AVGO", "COIN", "MSTR", "HOOD",
                "PYPL", "SQ", "SHOP", "JPM", "BAC", "V", "MA", "WFC", "GS", "MS",
                "BLK", "C", "AXP", "DIS", "BA", "F", "GM", "XOM", "CVX", "JNJ",
                "PG", "KO", "PEP", "WMT", "COST", "TGT", "HD", "LOW", "MCD", "T",
                "VZ", "PFE", "MRK", "UNH", "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF",
                "XLE", "XLV", "XLY", "XLP", "GLD", "SLV", "PLTR", "SOFI", "UBER",
                "ROKU", "GME", "AMC", "MARA", "RIOT", "CLSK", "HUT", "DKNG", "AFRM",
            ]
            for b in bluechips:
                symbols.add(b)
            try:
                assets = client.get_active_assets()
                for a in (assets or [])[:300]:
                    sym = (a.get("symbol") or "").strip()
                    if sym and "." not in sym and a.get("tradable"):
                        symbols.add(sym)
            except Exception:
                logger.exception("_reco_symbols: get_active_assets failed")
    except Exception as e:
        logger.error(f"Error fetching stock universe: {e}")

    # Filter out fiat FX pairs and blocked crypto bases
    filtered = []
    for s in symbols:
        if "/" in s:
            base = (s.split("/")[0] or "").upper()
            if base in FIAT_BASES or base in CRYPTO_BLOCKLIST:
                continue
        filtered.append(s)
    final_list = sorted(filtered)
    with _globals_lock:
        _UNIVERSE_CACHE["ts"] = time.time()
        _UNIVERSE_CACHE["symbols"] = final_list
    return final_list


def _apply_momentum_filter_to_universe(symbols: List[str]) -> List[str]:
    """
    Optional: focus stock universe on strong-momentum names.

    - Only applies to stocks (plain symbols without "/").
    - Crypto universe is kept as-is.
    - Controlled via:
        RECO_MOMENTUM_FILTER           (default: 1 = enabled)
        RECO_MOMENTUM_MIN_SCORE        (default: 60)
        RECO_MOMENTUM_TOP_N_STOCKS     (default: 150)
    """
    if not _MOMENTUM_FILTER_ENABLED:
        return symbols

    stock_symbols = [s for s in symbols if len(s) < 6 and "/" not in s]
    if not stock_symbols:
        return symbols

    try:
        from momentum_ranking import MomentumRanker
    except Exception as e:
        logger.debug("Momentum universe filter disabled (MomentumRanker import failed): %s", e)
        return symbols

    try:
        ranker = MomentumRanker()
        ranked = ranker.rank_universe(stock_symbols)
    except Exception as e:
        logger.debug("Momentum universe filter disabled (ranking error): %s", e)
        return symbols

    min_score = float(_MOMENTUM_FILTER_MIN_SCORE)
    top_n = max(1, int(_MOMENTUM_FILTER_TOP_N_STOCKS))

    picked: List[str] = []
    for r in ranked:
        try:
            score = float(r.get("score") or 0.0)
        except Exception:
            score = 0.0
        if score < min_score:
            continue
        sym = str(r.get("symbol") or "").strip()
        if not sym:
            continue
        picked.append(sym)
        if len(picked) >= top_n:
            break

    if not picked:
        logger.info(
            "Momentum universe filter produced 0 symbols (min_score=%.1f). Keeping original universe.",
            min_score,
        )
        return symbols

    crypto = [s for s in symbols if s not in stock_symbols]
    out = crypto + picked
    logger.info(
        "Momentum universe filter applied: %s -> %s total (%s stocks kept, min_score=%.1f, top_n=%s)",
        len(symbols),
        len(out),
        len(picked),
        min_score,
        top_n,
    )
    return out


# Optional: focus crypto universe on strong-momentum pairs (RECO_CRYPTO_MOMENTUM_FILTER=1, RECO_CRYPTO_MOMENTUM_TOP_N=80)
_CRYPTO_MOMENTUM_FILTER_ENABLED = os.getenv("RECO_CRYPTO_MOMENTUM_FILTER", "0").strip().lower() in ("1", "true", "yes", "y", "on")
_CRYPTO_MOMENTUM_TOP_N = int(os.getenv("RECO_CRYPTO_MOMENTUM_TOP_N", "80"))


def _apply_crypto_momentum_filter(symbols: List[str]) -> List[str]:
    """
    Optional: keep only top N crypto symbols by momentum (5d/20d/60d).
    Controlled by RECO_CRYPTO_MOMENTUM_FILTER (default 0=off), RECO_CRYPTO_MOMENTUM_TOP_N (default 80).
    """
    if not _CRYPTO_MOMENTUM_FILTER_ENABLED:
        return symbols

    crypto_symbols = [s for s in symbols if "/" in s and (s.split("/")[0] or "").upper() not in FIAT_BASES]
    if not crypto_symbols:
        return symbols

    try:
        from momentum_ranking import MomentumRanker
    except Exception as e:
        logger.debug("Crypto momentum filter disabled (MomentumRanker import failed): %s", e)
        return symbols

    try:
        ranker = MomentumRanker()
        ranked = ranker.rank_universe(crypto_symbols)
    except Exception as e:
        logger.debug("Crypto momentum filter disabled (ranking error): %s", e)
        return symbols

    top_n = max(1, _CRYPTO_MOMENTUM_TOP_N)
    picked = [str(r.get("symbol") or "").strip() for r in ranked[:top_n] if str(r.get("symbol") or "").strip()]
    if not picked:
        return symbols

    stocks = [s for s in symbols if s not in crypto_symbols]
    out = stocks + picked
    logger.info(
        "Crypto momentum filter applied: %s crypto -> %s kept (top_n=%s)",
        len(crypto_symbols),
        len(picked),
        top_n,
    )
    return out


def _scan_recommendations(horizon: str) -> None:
    global _last_reco_short_ts, _last_reco_long_ts
    if not _kraken_ready() and not (alpaca_live or alpaca_paper):
        logger.warning(f"Recommendation scan skipped: No trading clients ready (Kraken: {_kraken_ready()}, Alpaca: {bool(alpaca_live or alpaca_paper)})")
        return

    now = now_ts()
    error = ""
    btc_ctx = _btc_context()
    scanned = 0
    eligible = 0
    total_to_scan = 0
    try:
        symbols = _reco_symbols(quote="USD")
        symbols = _apply_momentum_filter_to_universe(symbols)
        symbols = _apply_crypto_momentum_filter(symbols)
        logger.info(f"Scanning {len(symbols)} symbols for {horizon} horizon")
        
        if len(symbols) == 0:
            logger.warning("No symbols to scan for recommendations")
            with _globals_lock:
                _RECO_STATE[horizon] = {
                    "last_run_ts": now,
                    "error": "No symbols available",
                    "btc_ctx": btc_ctx,
                    "scanned": 0,
                    "eligible": 0,
                }
                if horizon == "short":
                    _last_reco_short_ts = time.time()
                else:
                    _last_reco_long_ts = time.time()
            return
        
        # Separate crypto and stocks; scan whole market (max RECO_MAX_SYMBOLS)
        crypto_symbols = [s for s in symbols if ("/" in s or len(s) > 6) and (s.split("/")[0] or "").upper() not in CRYPTO_BLOCKLIST]
        stock_symbols = [s for s in symbols if len(s) < 6 and "/" not in s]
        
        # Kraken uses XBT for Bitcoin; include both for matching
        priority_crypto = [
            "XBT/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ETC/USD", "ADA/USD",
            "DOGE/USD", "LTC/USD", "BCH/USD", "DOT/USD", "AVAX/USD", "LINK/USD",
            "UNI/USD", "MATIC/USD", "ATOM/USD", "NEAR/USD", "APT/USD", "ARB/USD",
        ]
        crypto_set = set(crypto_symbols)
        crypto_selected = [pc for pc in priority_crypto if pc in crypto_set]
        crypto_selected += [s for s in crypto_symbols if s not in set(priority_crypto)]
        
        priority_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "SPY", "QQQ"]
        stock_set = set(stock_symbols)
        stock_selected = [ps for ps in priority_stocks if ps in stock_set]
        stock_selected += [s for s in stock_symbols if s not in set(priority_stocks)]
        
        n_max = min(RECO_MAX_SYMBOLS, 600)
        n_crypto = min(350, len(crypto_selected))
        n_stocks = min(n_max - n_crypto, len(stock_selected))
        symbols_to_scan = crypto_selected[:n_crypto] + stock_selected[:n_stocks]
        # Drop fiat FX pairs (base in FIAT list)
        symbols_to_scan = [
            s for s in symbols_to_scan if not ("/" in s and (s.split("/")[0] or "").upper() in FIAT_BASES)
        ]
        if RECO_SCAN_MAX_PER_RUN > 0:
            symbols_to_scan = symbols_to_scan[: min(len(symbols_to_scan), RECO_SCAN_MAX_PER_RUN)]
        
        logger.info(f"Scanning {len(symbols_to_scan)} symbols ({n_crypto} crypto, {n_stocks} stocks) for {horizon} horizon")
        
        total_to_scan = len(symbols_to_scan)
        # Set initial state so scan_status can show progress
        with _globals_lock:
            _RECO_STATE[horizon] = {
                "last_run_ts": 0,
                "error": "",
                "btc_ctx": btc_ctx,
                "scanned": 0,
                "eligible": 0,
                "total": total_to_scan,
                "scanning": True,
            }
        
        chunk_size = 25
        error_count = 0
        for chunk_start in range(0, len(symbols_to_scan), chunk_size):
            chunk = symbols_to_scan[chunk_start:chunk_start + chunk_size]
            for sym in chunk:
                try:
                    # Skip crypto symbols not on Kraken or in blocklist
                    if "/" in sym:
                        base = (sym.split("/")[0] or "").upper()
                        if base in CRYPTO_BLOCKLIST:
                            continue
                        resolved, _ = _validate_crypto_symbol(sym)
                        if not resolved:
                            continue
                    snap = _scan_symbol(sym, horizon, btc_ctx)
                    if not snap:
                        continue
                    # C3: Skip symbols with data failures or invalid data
                    if snap.get("data_ok") is False:
                        continue
                    risk_flags = snap.get("risk_flags") or []
                    if "DATA_INVALID" in risk_flags or "DATA_ERROR" in risk_flags or "ROUTING_ERROR" in risk_flags:
                        continue
                    metrics = snap.get("metrics") or {}
                    is_stock = len(sym) < 6 and "/" not in sym
                    metrics["market_type"] = "stocks" if is_stock else "crypto"
                    if metrics.get("eligible"):
                        eligible += 1
                    save_recommendation_snapshot(
                        symbol=snap["symbol"],
                        horizon=horizon,
                        score=float(snap.get("score") or 0.0),
                        regime_json=json.dumps(snap.get("regime") or {}),
                        metrics_json=json.dumps(metrics),
                        reasons_json=json.dumps(snap.get("reasons") or []),
                        risk_flags_json=json.dumps(snap.get("risk_flags") or []),
                    )
                    scanned += 1
                    if scanned % 10 == 0:
                        logger.info(f"Scan progress: {scanned}/{len(symbols_to_scan)} symbols processed")
                        with _globals_lock:
                            s = _RECO_STATE.get(horizon) or {}
                            s["scanned"] = scanned
                            s["eligible"] = eligible
                            _RECO_STATE[horizon] = s
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error scanning {sym}: {e}")
                    if error_count >= RECO_SCAN_ERROR_LIMIT:
                        error = f"Stopped after {error_count} errors (rate-limit protection)"
                        logger.error(error)
                        break
                    continue
                if RECO_SCAN_SYMBOL_SLEEP_SEC > 0:
                    time.sleep(RECO_SCAN_SYMBOL_SLEEP_SEC)
            if error_count >= RECO_SCAN_ERROR_LIMIT:
                break
            if chunk_start + chunk_size < len(symbols_to_scan):
                time.sleep(0.4)
                
        logger.info(f"Scan complete: {scanned} scanned, {eligible} eligible for {horizon}")
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
        logger.error(f"Scan failed: {error}", exc_info=True)

    with _globals_lock:
        _RECO_STATE[horizon] = {
            "last_run_ts": now,
            "error": error,
            "btc_ctx": btc_ctx,
            "scanned": scanned,
            "eligible": eligible,
            "total": total_to_scan if 'total_to_scan' in dir() else 0,
            "scanning": False,
        }
        if horizon == "short":
            _last_reco_short_ts = time.time()
        else:
            _last_reco_long_ts = time.time()


_BACKOFF_BASE = 1.0
_BACKOFF_MAX = 60.0
_CIRCUIT_BREAKER_THRESHOLD = 10

def _recommendations_loop() -> None:
    last_short = 0
    last_long = 0
    fail_count = 0
    backoff = _BACKOFF_BASE
    while True:
        try:
            now = int(time.time())
            if now - last_short >= RECO_SHORT_EVERY_SEC:
                _scan_recommendations("short")
                last_short = now
            if now - last_long >= RECO_LONG_EVERY_SEC:
                _scan_recommendations("long")
                last_long = now
            fail_count = 0
            backoff = _BACKOFF_BASE
        except Exception:
            logger.exception("_recommendations_loop: iteration failed")
            fail_count += 1
            if fail_count >= _CIRCUIT_BREAKER_THRESHOLD and DISCORD_WEBHOOK_URL:
                try:
                    import requests
                    requests.post(DISCORD_WEBHOOK_URL, json={"content": f"⚠️ Recommendations loop: {fail_count} consecutive failures. Degraded."}, timeout=3)
                except Exception:
                    pass
            backoff = min(_BACKOFF_MAX, backoff * 2)
        time.sleep(max(10, int(backoff)))


def _ml_retrain_loop() -> None:
    """Weekly ML retrain (walk-forward, deploy only if validation >60%)."""
    import os
    freq_days = int(os.getenv("ML_RETRAIN_FREQUENCY", "7"))
    interval_sec = max(86400, freq_days * 86400)
    last_run = 0
    while True:
        try:
            time.sleep(3600)  # Check hourly
            now = int(time.time())
            if now - last_run >= interval_sec:
                try:
                    from ml_ensemble import get_ml_ensemble
                    ensemble = get_ml_ensemble()
                    if len(ensemble._training_data) >= 100:
                        success = ensemble.train(force=True)
                        if success:
                            last_run = now
                            min_acc = float(os.getenv("ML_MIN_ACCURACY", "0.60"))
                            best = max(
                                getattr(ensemble._model_performance.get("xgb"), "recent_accuracy", 0) or 0,
                                getattr(ensemble._model_performance.get("rf"), "recent_accuracy", 0) or 0,
                            )
                            if best >= min_acc:
                                from db import save_ml_model_version
                                save_ml_model_version("ensemble", f"v{now}", best, deployed=True)
                                logger.info("ML retrain: deployed ensemble v%s (acc=%.2f%%)", now, best * 100)
                except Exception as e:
                    logger.debug("ML retrain loop: %s", e)
        except Exception as e:
            logger.debug("ML retrain loop error: %s", e)


def _ml_outcomes_loop() -> None:
    """Daily job: update ML predictions with actual 7d/30d outcomes."""
    last_run = 0
    while True:
        try:
            time.sleep(3600)
            now = int(time.time())
            if now - last_run >= 86400:
                try:
                    from ml_prediction_tracker import update_outcomes_job
                    update_outcomes_job()
                    last_run = now
                except Exception as e:
                    logger.debug("ML outcomes job: %s", e)
        except Exception:
            pass


def _portfolio_snapshot() -> Dict[str, Any]:
    """
    Best-effort portfolio estimation in USD from Kraken balances.
    Safe to call even when Kraken isn't ready.
    """
    if not _kraken_ready():
        with _globals_lock:
            latest = PORT_HISTORY[-1]["total_usd"] if PORT_HISTORY else 0.0
        return {
            "total_usd": float(latest),
            "free_usd": 0.0,
            "used_usd": 0.0,
            "positions_usd": 0.0,
            "holdings": [],
            "error": KRAKEN_ERROR or "Kraken not ready",
        }

    try:
        bal = kc.fetch_balance()
        total = (bal.get("total", {}) or {})
        free = (bal.get("free", {}) or {})
        used = (bal.get("used", {}) or {})
        mk = _markets()

        symbols_usd = set(
            s for s, m in mk.items()
            if m.get("spot") and m.get("active") and m.get("quote") == "USD"
        )

        def price_for_asset(asset: str) -> Optional[float]:
            if asset == "USD":
                return 1.0
            a = "XBT" if asset == "BTC" else asset
            sym = f"{a}/USD"
            if sym in symbols_usd:
                try:
                    return float(kc.fetch_ticker_last(sym))
                except Exception:
                    return None
            return None

        holdings: List[Dict[str, Any]] = []
        total_usd = 0.0
        free_usd = 0.0
        used_usd = 0.0
        positions_usd = 0.0

        for asset, amt in total.items():
            a_total = float(amt or 0.0)
            a_free = float(free.get(asset, 0.0) or 0.0)
            a_used = float(used.get(asset, 0.0) or 0.0)
            if a_total <= 0 and a_free <= 0 and a_used <= 0:
                continue

            p = price_for_asset(str(asset))
            usd_total = a_total * p if p is not None else None
            usd_free = a_free * p if p is not None else None
            usd_used = a_used * p if p is not None else None

            if usd_total is not None:
                total_usd += float(usd_total)
            if usd_free is not None:
                free_usd += float(usd_free)
            if usd_used is not None:
                used_usd += float(usd_used)

            if str(asset) != "USD" and usd_total is not None:
                positions_usd += float(usd_total)

            holdings.append(
                {
                    "asset": str(asset),
                    "amount": a_total,
                    "free": a_free,
                    "used": a_used,
                    "usd_value": usd_total,
                    "usd_free": usd_free,
                    "usd_used": usd_used,
                }
            )

        holdings.sort(key=lambda x: (x["usd_value"] or 0.0), reverse=True)
        return {
            "total_usd": float(total_usd),
            "free_usd": float(free_usd),
            "used_usd": float(used_usd),
            "positions_usd": float(positions_usd),
            "holdings": holdings,
            "error": "",
        }
    except Exception as e:
        logger.exception("_portfolio_snapshot: Kraken/balance fetch failed")
        with _globals_lock:
            latest = PORT_HISTORY[-1]["total_usd"] if PORT_HISTORY else 0.0
        return {
            "total_usd": float(latest),
            "free_usd": 0.0,
            "used_usd": 0.0,
            "positions_usd": 0.0,
            "holdings": [],
            "error": f"{type(e).__name__}: {e}",
        }


def _portfolio_loop():
    global _last_portfolio_ts
    fail_count = 0
    backoff = _BACKOFF_BASE
    while True:
        try:
            snap = _portfolio_snapshot()
            total_usd = float(snap.get("total_usd") or 0.0)
            ts = now_ts()
            with _globals_lock:
                PORT_HISTORY.append({"ts": ts, "total_usd": total_usd})
                if len(PORT_HISTORY) > 2000:
                    del PORT_HISTORY[:200]
                _last_portfolio_ts = time.time()
            # Record to portfolio_snapshots for charts (11.md)
            try:
                from db import _conn
                con = _conn()
                try:
                    con.execute(
                        "INSERT INTO portfolio_snapshots (total_value, total_pnl, active_positions, unrealized_pnl) VALUES (?, ?, ?, ?)",
                        (total_usd, 0.0, snap.get("positions_count", 0), 0.0),
                    )
                    con.commit()
                finally:
                    con.close()
            except Exception:
                pass
            fail_count = 0
            backoff = _BACKOFF_BASE
        except Exception:
            logger.exception("_portfolio_loop: iteration failed")
            fail_count += 1
            backoff = min(_BACKOFF_MAX, backoff * 2)
        time.sleep(max(5, PORT_EVERY_SEC, int(backoff)))


def _discord_notify(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        import requests
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=3)
    except Exception:
        logger.exception("_discord_notify: webhook post failed")


def _discord_status_update(message: str) -> None:
    if not DISCORD_STATUS_WEBHOOK_URL:
        return
    try:
        import requests
        try:
            os.makedirs(os.path.dirname(DISCORD_STATUS_MSG_FILE), exist_ok=True)
        except Exception:
            pass
        try:
            with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                f.write(f"[status] update start\n")
        except Exception:
            pass
        msg_id = None
        try:
            if os.path.exists(DISCORD_STATUS_MSG_FILE):
                with open(DISCORD_STATUS_MSG_FILE, "r", encoding="utf-8") as f:
                    msg_id = f.read().strip() or None
        except Exception:
            msg_id = None

        if msg_id:
            url = f"{DISCORD_STATUS_WEBHOOK_URL}/messages/{msg_id}"
            r = requests.patch(url, json={"content": message}, timeout=3)
            if r.ok:
                try:
                    with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"[status] patched {msg_id}\n")
                except Exception:
                    pass
                return
            if r.status_code != 404:
                try:
                    with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"[status] patch failed {r.status_code}\n")
                except Exception:
                    pass
                return
            try:
                if os.path.exists(DISCORD_STATUS_MSG_FILE):
                    os.remove(DISCORD_STATUS_MSG_FILE)
            except Exception:
                pass

        post_url = f"{DISCORD_STATUS_WEBHOOK_URL}?wait=true"
        r = requests.post(post_url, json={"content": message}, timeout=3)
        if r.ok:
            data = r.json()
            mid = str(data.get("id") or "")
            if mid:
                try:
                    with open(DISCORD_STATUS_MSG_FILE, "w", encoding="utf-8") as f:
                        f.write(mid)
                except Exception:
                    pass
                try:
                    with open(DISCORD_STATUS_LOG, "a", encoding="utf-8") as f:
                        f.write(f"[status] posted {mid}\n")
                except Exception:
                    pass
    except Exception as e:
        logger.exception("_discord_status_update failed: %s", e)


def _discord_status_loop() -> None:
    last_state: Dict[int, bool] = {}
    fail_count = 0
    backoff = _BACKOFF_BASE
    # initial summary
    try:
        bots = list_bots()
        running_ids = []
        if bm is not None:
            for b in bots:
                snap = bm.snapshot(int(b.get("id")))
                is_running = bool(snap.get("running"))
                last_state[int(b.get("id"))] = is_running
                if is_running:
                    running_ids.append(str(b.get("name") or b.get("id")))
        lines = []
        for b in bots:
            bid = int(b.get("id"))
            name = b.get("name") or bid
            state = "🟢 live" if last_state.get(bid) else "⚪ idle"
            lines.append(f"{name}: {state}")
            _discord_status_update("**Bot status**\n" + "\n".join(lines) if lines else "**Bot status**\n(no bots)")
    except Exception:
        logger.exception("_discord_status_loop: initial summary failed")

    while True:
        try:
            if bm is None:
                time.sleep(5)
                continue
            bots = list_bots()
            for b in bots:
                bot_id = int(b.get("id"))
                snap = bm.snapshot(bot_id)
                is_running = bool(snap.get("running"))
                if bot_id not in last_state or last_state[bot_id] != is_running:
                    last_state[bot_id] = is_running
                    # Optional: send individual start/stop notifications (default off to reduce spam)
                    if os.getenv("DISCORD_STATUS_NOTIFY_CHANGES", "0").strip().lower() in ("1", "true", "yes"):
                        name = b.get("name") or bot_id
                        msg = f"✅ {name} is running." if is_running else f"🛑 {name} stopped."
                        _discord_notify(msg)
            lines = []
            for b in bots:
                bid = int(b.get("id"))
                name = b.get("name") or bid
                state = "🟢 live" if last_state.get(bid) else "⚪ idle"
                lines.append(f"{name}: {state}")
            _discord_status_update("**Bot status**\n" + "\n".join(lines) if lines else "**Bot status**\n(no bots)")
            fail_count = 0
            backoff = _BACKOFF_BASE
        except Exception:
            logger.exception("_discord_status_loop: iteration failed")
            fail_count += 1
            backoff = min(_BACKOFF_MAX, backoff * 2)
        time.sleep(max(10, int(backoff)))


def _pause_state() -> bool:
    env = os.getenv("PAUSE_ALL_BOTS", "").strip().lower()
    if env in ("1", "true", "yes", "y", "on"):
        return True
    try:
        v = get_setting("global_pause", "0")
        if str(v).strip().lower() in ("1", "true", "yes", "y", "on"):
            until = get_setting("global_pause_until", "0")
            try:
                until_ts = int(until or 0)
            except Exception:
                until_ts = 0
            if until_ts and until_ts <= int(time.time()):
                set_setting("global_pause", "0")
                set_setting("global_pause_until", "0")
                return False
            return True
        return False
    except Exception:
        return False


def _kill_switch_state() -> bool:
    env = os.getenv("KILL_SWITCH", "").strip().lower()
    if env in ("1", "true", "yes", "y", "on"):
        return True
    try:
        v = get_setting("kill_switch", "0")
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")
    except Exception:
        return False


def _should_autostart(bot: Dict[str, Any]) -> bool:
    try:
        if int(bot.get("last_running", 0)) == 1:
            return True
        if AUTO_START_ENABLED and int(bot.get("enabled", 0)) == 1:
            return True
    except Exception:
        return False
    return False


def _autostart_loop() -> None:
    attempts = 0
    max_attempts = 36  # ~3 minutes
    while attempts < max_attempts:
        try:
            if bm is None:
                time.sleep(5)
                attempts += 1
                continue

            bots = list_bots()
            pending_live = False
            if _pause_state():
                time.sleep(5)
                attempts += 1
                continue
            for b in bots:
                if not _should_autostart(b):
                    continue
                bot_id = int(b.get("id"))
                snap = bm.snapshot(bot_id)
                if bool(snap.get("running")):
                    continue
                ok, reason = _can_start_bot_live(b)
                if not ok:
                    pending_live = True
                    logger.warning(
                        "autostart: readiness gate blocked bot_id=%s market_type=%s reason=%s",
                        bot_id,
                        b.get("market_type"),
                        reason,
                    )
                    continue
                bm.start(bot_id)
            if not pending_live:
                return
        except Exception:
            logger.exception("_autostart_loop: iteration failed")
        time.sleep(5)
        attempts += 1


def _health_watchdog_loop() -> None:
    while True:
        try:
            if bm is None:
                time.sleep(HEALTH_WATCHDOG_SEC)
                continue
            if _pause_state():
                time.sleep(HEALTH_WATCHDOG_SEC)
                continue
            bots = list_bots()
            now = int(time.time())
            for b in bots:
                if not _should_autostart(b):
                    continue
                bot_id = int(b.get("id"))
                snap = bm.snapshot(bot_id)
                if not bool(snap.get("running")):
                    # Only restart if bot is enabled and should be running
                    bot = get_bot(bot_id)
                    if bot and int(bot.get("enabled", 0)) == 1:
                        # Add a minimum cooldown to avoid rapid restart loops
                        last_restart = getattr(_health_watchdog_loop, f"_last_restart_{bot_id}", 0)
                        if (now - last_restart) > 60:  # Don't restart more than once per 60 seconds
                            ok, reason = _can_start_bot_live(b)
                            if ok:
                                bm.start(bot_id)
                                setattr(_health_watchdog_loop, f"_last_restart_{bot_id}", now)
                                add_log(bot_id, "WARN", "Watchdog restarted bot.", "SYSTEM")
                            else:
                                logger.warning(
                                    "watchdog: start blocked bot_id=%s market_type=%s reason=%s",
                                    bot_id,
                                    b.get("market_type"),
                                    reason,
                                )
                    continue
                last_tick = int(snap.get("last_tick_ts") or 0)
                # Increase threshold to 5 minutes (300s) to avoid restarting bots waiting for market open
                # Also check if bot is actually running - if running but stale tick, it might be waiting
                stale_threshold = max(300, HEALTH_WATCHDOG_SEC * 5)  # At least 5 minutes
                if last_tick and (now - last_tick) > stale_threshold:
                    # Only restart if bot claims to be running but hasn't ticked
                    # If bot is stopped, autostart will handle it above
                    if bool(snap.get("running")):
                        # Check if it's a stock bot that might be waiting for market open
                        bot = get_bot(bot_id)
                        is_stock = bot and (len(str(bot.get("symbol", ""))) < 6 and "/" not in str(bot.get("symbol", "")))
                        if is_stock:
                            # Market-hours aware: skip restart when market closed (bot waiting for open)
                            if not _alpaca_market_open():
                                continue  # Don't restart; bot is correctly idle
                            # For stocks, be even more lenient (market might be closed)
                            if (now - last_tick) > 600:  # 10 minutes for stocks
                                bm.stop(bot_id)
                                time.sleep(1)
                                ok, reason = _can_start_bot_live(b)
                                if ok:
                                    bm.start(bot_id)
                                    add_log(bot_id, "WARN", f"Watchdog restarted stalled stock bot (last tick: {now - last_tick}s ago).", "SYSTEM")
                                else:
                                    logger.warning(
                                        "watchdog: restart blocked bot_id=%s market_type=%s reason=%s",
                                        bot_id,
                                        b.get("market_type"),
                                        reason,
                                    )
                        else:
                            # For crypto, use normal threshold
                            bm.stop(bot_id)
                            time.sleep(1)
                            ok, reason = _can_start_bot_live(b)
                            if ok:
                                bm.start(bot_id)
                                add_log(bot_id, "WARN", f"Watchdog restarted stalled bot (last tick: {now - last_tick}s ago).", "SYSTEM")
                            else:
                                logger.warning(
                                    "watchdog: restart blocked bot_id=%s market_type=%s reason=%s",
                                    bot_id,
                                    b.get("market_type"),
                                    reason,
                                )
        except Exception:
            logger.exception("_health_watchdog_loop: iteration failed")
        time.sleep(HEALTH_WATCHDOG_SEC)


def _comprehensive_health_check() -> Tuple[bool, List[str]]:
    """Check all critical systems. Returns (ok, issues)."""
    issues = []
    try:
        client = alpaca_live or alpaca_paper
        if client and hasattr(client, "check_websocket_health"):
            ok, ws_issues = client.check_websocket_health()
            if not ok:
                issues.extend(ws_issues)
        elif client and hasattr(client, "get_stats"):
            stats = client.get_stats()
            ws = stats.get("websocket", {})
            if not ws.get("running", True):
                issues.append("WebSocket not running")
        autopilot_on = False
        try:
            import autopilot
            autopilot_on = autopilot.is_autopilot_enabled()
            if not autopilot_on:
                issues.append("Autopilot is disabled")
        except Exception:
            pass
        bots = list_bots() or []
        active = [b for b in bots if int(b.get("enabled", 0)) == 1]
        if len(active) == 0 and autopilot_on:
            issues.append("No active bots (autopilot enabled but no bots)")
        if client:
            try:
                client.get_account()
            except Exception as e:
                issues.append(f"API connection error: {e}")
        if bm:
            snaps = [bm.snapshot(int(b["id"])) for b in active]
            paused = [b for b, s in zip(active, snaps) if str(s.get("status", "") or "").upper() == "PAUSE"]
            if paused:
                issues.append(f"{len(paused)} bots paused")
        if issues:
            logger.error("Health check FAILED: %s", issues)
        else:
            logger.info("Health check PASSED - All systems operational")
        return len(issues) == 0, issues
    except Exception as e:
        logger.exception("comprehensive_health_check error: %s", e)
        return False, [str(e)]


def _health_comprehensive_loop() -> None:
    """Run comprehensive health check and WebSocket stats every 5 minutes."""
    while True:
        try:
            time.sleep(300)
            _comprehensive_health_check()  # (ok, issues) - logs internally
            client = alpaca_live or alpaca_paper
            if client and hasattr(client, "print_stats"):
                try:
                    client.print_stats()
                except Exception:
                    pass
        except Exception:
            logger.exception("_health_comprehensive_loop error")


# =========================================================
# Startup (fully fault-tolerant: never raise; always bind)
# =========================================================
@app.on_event("startup")
def startup():
    global kc, alpaca_paper, alpaca_live, bm, KRAKEN_READY, KRAKEN_ERROR, ALPACA_PAPER_READY, ALPACA_LIVE_READY, ALPACA_ERROR
    try:
        _startup_impl()
    except Exception as e:
        logger.exception("startup: unhandled error")
        logger.error("worker_api startup: UNHANDLED ERROR (%s). Running in minimal mode.", e)


@app.on_event("shutdown")
def shutdown():
    """Clean shutdown: stop UnifiedAlpacaClient WebSocket and cache."""
    global alpaca_paper, alpaca_live
    for client in [alpaca_paper, alpaca_live]:
        if client and hasattr(client, "shutdown"):
            try:
                client.shutdown()
                logger.info("UnifiedAlpacaClient shutdown complete")
            except Exception as e:
                logger.warning("Client shutdown: %s", e)


def _start_websocket_with_timeout(ws_manager, timeout_sec: int = 10):
    """Start WebSocket in background - never block startup."""
    try:
        if ws_manager and hasattr(ws_manager, "start"):
            ws_manager.start()
        _STARTUP_STATUS["websocket_status"] = "running"
    except Exception as e:
        logger.warning("WebSocket start failed (non-blocking): %s", e)
        _STARTUP_STATUS["websocket_status"] = "not_ready"


def _init_alpaca_background():
    """Initialize Alpaca clients in background so app can serve requests immediately."""
    global alpaca_paper, alpaca_live, bm, ALPACA_PAPER_READY, ALPACA_LIVE_READY, ALPACA_ERROR, _STARTUP_STATUS
    # Log Alpaca key status for debugging "0 candles" issues
    pk = "SET" if os.getenv("ALPACA_API_KEY_PAPER") else "blank"
    sk = "SET" if os.getenv("ALPACA_API_SECRET_PAPER") else "blank"
    lk = "SET" if os.getenv("ALPACA_API_KEY_LIVE") else "blank"
    ls = "SET" if os.getenv("ALPACA_API_SECRET_LIVE") else "blank"
    logger.info("worker_api startup: ALPACA_API_KEY_PAPER=%s ALPACA_API_SECRET_PAPER=%s ALPACA_API_KEY_LIVE=%s ALPACA_API_SECRET_LIVE=%s",
                pk, sk, lk, ls)
    try:
        with _globals_lock:
            try:
                if USE_UNIFIED_ALPACA and _UNIFIED_AVAILABLE and UnifiedAlpacaClient:
                    alpaca_paper = UnifiedAlpacaClient(mode="paper", auto_start_websocket=False)
                    if alpaca_paper and hasattr(alpaca_paper, "websocket"):
                        t = threading.Thread(target=_start_websocket_with_timeout, args=(alpaca_paper.websocket, 10), daemon=True)
                        t.start()
                    logger.info("worker_api startup: Alpaca PAPER initialized (Unified, websocket deferred).")
                else:
                    alpaca_paper = AlpacaClient(mode="paper")
                    logger.info("worker_api startup: Alpaca PAPER initialized (Legacy).")
                ALPACA_PAPER_READY = True
            except Exception as e:
                try:
                    alpaca_paper = AlpacaClient(mode="paper")
                    ALPACA_PAPER_READY = True
                    logger.info("worker_api startup: Alpaca PAPER fallback to Legacy: %s", e)
                except Exception as e2:
                    alpaca_paper = None
                    ALPACA_PAPER_READY = False
                    ALPACA_ERROR = f"Paper: {e2}"
                    logger.warning("worker_api startup: Alpaca PAPER failed: %s", e2)

        # Skip live Alpaca init when live trading disabled (saves memory + connections)
        if LIVE_TRADING_ENABLED:
            with _globals_lock:
                try:
                    if USE_UNIFIED_ALPACA and _UNIFIED_AVAILABLE and UnifiedAlpacaClient:
                        alpaca_live = UnifiedAlpacaClient(mode="live", auto_start_websocket=False)
                        if alpaca_live and hasattr(alpaca_live, "websocket"):
                            t = threading.Thread(target=_start_websocket_with_timeout, args=(alpaca_live.websocket, 10), daemon=True)
                            t.start()
                        logger.info("worker_api startup: Alpaca LIVE initialized (Unified, websocket deferred).")
                    else:
                        alpaca_live = AlpacaClient(mode="live")
                        logger.info("worker_api startup: Alpaca LIVE initialized (Legacy).")
                    ALPACA_LIVE_READY = True
                except Exception as e:
                    try:
                        alpaca_live = AlpacaClient(mode="live")
                        ALPACA_LIVE_READY = True
                        logger.info("worker_api startup: Alpaca LIVE fallback to Legacy")
                    except Exception as e2:
                        alpaca_live = None
                        ALPACA_LIVE_READY = False
                        ALPACA_ERROR = (ALPACA_ERROR or "") + f" | Live: {e2}"
                        logger.warning("worker_api startup: Alpaca LIVE failed: %s", e2)
        else:
            alpaca_live = None
            ALPACA_LIVE_READY = False
            logger.info("worker_api startup: Alpaca LIVE skipped (set LIVE_TRADING_ENABLED=1 in .env for live)")

        bp = 0.0
        if alpaca_paper:
            try:
                bp = float((alpaca_paper.get_account() or {}).get("buying_power", 0))
            except Exception:
                pass
        _STARTUP_STATUS["alpaca_ready"] = ALPACA_PAPER_READY or ALPACA_LIVE_READY
        _STARTUP_STATUS["alpaca_buying_power"] = bp

        with _globals_lock:
            bm = None
            if kc or alpaca_paper or alpaca_live:
                try:
                    bm = BotManager(kc, alpaca_paper, alpaca_live)
                    if bm and hasattr(bm, "subscribe_all_symbols"):
                        threading.Thread(target=bm.subscribe_all_symbols, daemon=True).start()
                    logger.info("worker_api startup: BotManager OK (Crypto: %s, Alpaca paper: %s, Alpaca live: %s)",
                        KRAKEN_READY, ALPACA_PAPER_READY, ALPACA_LIVE_READY)
                except Exception as e:
                    logger.exception("worker_api startup: BotManager init failed")
                    logger.warning("worker_api startup: BotManager failed (%s). Degraded mode.", e)

        # Candle pre-warm test
        if alpaca_paper and bm:
            try:
                c = alpaca_paper.get_ohlcv("AAPL", "1h", 50)
                _STARTUP_STATUS["candle_test"] = f"fetched {len(c)} candles for AAPL" if c else "failed"
            except Exception as ex:
                _STARTUP_STATUS["candle_test"] = f"error: {ex}"
    except Exception as e:
        logger.exception("_init_alpaca_background failed: %s", e)


def _validate_config() -> None:
    """Validate configuration at startup. Fail fast with clear errors if invalid."""
    errs = []
    ttl_val = os.getenv("MARKETS_TTL_SEC", "300").strip()
    if ttl_val:
        try:
            ttl = int(ttl_val)
            if not (60 <= ttl <= 3600):
                errs.append(f"MARKETS_TTL_SEC must be 60-3600, got {ttl}")
        except (ValueError, TypeError):
            errs.append("MARKETS_TTL_SEC must be an integer")
    port_val = os.getenv("PORT_EVERY_SEC", "60").strip()
    if port_val:
        try:
            port_every = int(port_val)
            if not (1 <= port_every <= 3600):
                errs.append(f"PORT_EVERY_SEC must be 1-3600, got {port_every}")
        except (ValueError, TypeError):
            errs.append("PORT_EVERY_SEC must be an integer")
    if errs:
        msg = "Config validation failed: " + "; ".join(errs)
        logger.error(msg)
        raise ValueError(msg)


def _startup_impl():
    global kc, alpaca_paper, alpaca_live, bm, KRAKEN_READY, KRAKEN_ERROR, ALPACA_PAPER_READY, ALPACA_LIVE_READY, ALPACA_ERROR
    global LIVE_ENDPOINTS_DISABLED, LIVE_ENDPOINTS_DISABLED_REASON

    _validate_config()

    def _safe_init_db():
        try:
            init_db()
            return True
        except Exception as e:
            logger.exception("startup: init_db failed")
            logger.warning("worker_api startup: init_db failed (%s). DB features degraded.", e)
            return False

    _safe_init_db()
    _STARTUP_STATUS["db_ready"] = True
    try:
        _STARTUP_STATUS["db_bots"] = len(list_bots())
    except Exception:
        pass
    logger.info("Database connected (%s bots)", _STARTUP_STATUS.get("db_bots", 0))

    # Purge blocklisted symbols from recommendations (Explore tab)
    try:
        n = delete_recommendations_for_blocklist(list(CRYPTO_BLOCKLIST))
        if n > 0:
            logger.info("Purged %d blocklisted recommendation(s) (STABLE, etc.)", n)
    except Exception as e:
        logger.warning("Purge blocklist failed: %s", e)

    try:
        t = WORKER_API_TOKEN
        if t and (len(t) < 16 or t.lower() in ("dev", "test", "secret")):
            logger.warning("WORKER_API_TOKEN looks weak or default. Use a long random secret for production.")
        elif not t:
            logger.warning("WORKER_API_TOKEN not set. API auth disabled; fine for localhost only.")
    except Exception as e:
        logger.warning("startup: token check failed: %s", e)

    with _globals_lock:
        try:
            kc = KrakenClient()
            KRAKEN_READY = True
            KRAKEN_ERROR = ""
            logger.info("worker_api startup: Kraken client initialized.")
        except Exception as e:
            kc = None
            KRAKEN_READY = False
            KRAKEN_ERROR = str(e)
            logger.warning("worker_api startup: Kraken NOT initialized: %s", e)

    # Fail fast: stock provider is Alpaca; keys required for stocks
    if not os.getenv("ALPACA_API_KEY_PAPER") or not os.getenv("ALPACA_API_SECRET_PAPER"):
        raise ValueError(
            "Alpaca paper API keys required for stock trading. "
            "Set ALPACA_API_KEY_PAPER and ALPACA_API_SECRET_PAPER in .env"
        )
    # Alpaca + BotManager init in background so site loads immediately (non-blocking)
    _STARTUP_STATUS["flask_ready"] = True
    threading.Thread(target=_init_alpaca_background, daemon=True).start()

    # Autopilot status (from DB)
    try:
        import autopilot
        _STARTUP_STATUS["autopilot_enabled"] = autopilot.is_autopilot_enabled()
        ap_bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
        _STARTUP_STATUS["autopilot_bots"] = len(ap_bots)
    except Exception:
        pass

    # API auth disabled by user request
    LIVE_ENDPOINTS_DISABLED = False
    LIVE_ENDPOINTS_DISABLED_REASON = ""

    for name, target in [
        ("portfolio", _portfolio_loop),
        ("discord_status", _discord_status_loop),
        ("autostart", _autostart_loop),
        ("health_watchdog", _health_watchdog_loop),
        ("health_comprehensive", _health_comprehensive_loop),
        ("recommendations", _recommendations_loop),
        ("ml_retrain", _ml_retrain_loop),
        ("ml_outcomes", _ml_outcomes_loop),
        ("autopilot", _autopilot_loop),
    ]:
        try:
            _start_background_thread(name, target)
        except Exception as e:
            logger.exception("startup: failed to start thread %s", name)
            logger.warning("worker_api startup: thread %s failed (%s). Continuing.", name, e)
    try:
        _comprehensive_health_check()  # returns (ok, issues)
    except Exception:
        pass
    logger.info("worker_api startup: complete.")
    logger.info(
        "Startup diagnostics: Flask=%s DB=%s (bots=%s) Alpaca=%s WebSocket=%s Autopilot=%s (bots=%s) CandleTest=%s",
        _STARTUP_STATUS.get("flask_ready"),
        _STARTUP_STATUS.get("db_ready"),
        _STARTUP_STATUS.get("db_bots"),
        ALPACA_PAPER_READY or ALPACA_LIVE_READY,
        _STARTUP_STATUS.get("websocket_status"),
        _STARTUP_STATUS.get("autopilot_enabled"),
        _STARTUP_STATUS.get("autopilot_bots"),
        _STARTUP_STATUS.get("candle_test"),
    )
    logger.info(
        "Trading readiness: Kraken=%s (err=%s) AlpacaPaper=%s AlpacaLive=%s (err=%s) AUTO_START_ENABLED=%s HEALTH_WATCHDOG_SEC=%s",
        KRAKEN_READY,
        KRAKEN_ERROR or "none",
        ALPACA_PAPER_READY,
        ALPACA_LIVE_READY,
        ALPACA_ERROR or "none",
        AUTO_START_ENABLED,
        HEALTH_WATCHDOG_SEC,
    )


@app.put("/api/bots/{bot_id}")
async def api_update_bot(bot_id: int, request: Request):
    """Update bot settings. Merges payload with existing bot for partial updates."""
    if not bm:
        return _json({"ok": False, "error": "BotManager not initialized"}, 503)
    payload = await request.json()
    if not isinstance(payload, dict):
        return _json({"ok": False, "error": "Invalid payload"}, 400)

    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    # Merge: existing bot as base, overlay payload (partial updates supported)
    raw_sym = str(payload.get("symbol") or b.get("symbol") or "").strip()
    detected_type = classify_symbol(raw_sym) if raw_sym else "crypto"
    market_type = "stocks" if detected_type == "stock" else "crypto"
    if market_type == "crypto" and raw_sym:
        resolved, err = _validate_crypto_symbol(raw_sym)
        if err:
            return _json({"ok": False, "error": err}, 400)
        raw_sym = resolved or raw_sym
    symbol = _resolve_symbol(raw_sym)

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

    settings = {
        "name": str(_ov("name", b.get("name") or f"Bot {symbol}")),
        "symbol": symbol,
        "enabled": int(_ov("enabled", 1, lambda x: int(x) if x is not None else 1)),
        "dry_run": int(_ov("dry_run", 1, lambda x: int(x) if x is not None else 1)),
        "base_quote": float(_ov("base_quote", 20.0, lambda x: float(x) if x is not None else 20.0)),
        "safety_quote": float(_ov("safety_quote", 20.0, lambda x: float(x) if x is not None else 20.0)),
        "max_safety": int(_ov("max_safety", 5, lambda x: int(x) if x is not None else 5)),
        "first_dev": float(_ov("first_dev", 0.015, lambda x: float(x) if x is not None else 0.015)),
        "step_mult": float(_ov("step_mult", 1.2, lambda x: float(x) if x is not None else 1.2)),
        "tp": float(_ov("tp", 0.015, lambda x: float(x) if x is not None else 0.015)),
        "market_type": market_type,
        "strategy_mode": str(_ov("strategy_mode", "auto")),
        "forced_strategy": str(_ov("forced_strategy", "")),
        "alpaca_mode": str(_ov("alpaca_mode", "paper")),
        "max_spend_quote": float(_ov("max_spend_quote", 0.0, lambda x: float(x) if x is not None else 0.0)),
        "poll_seconds": int(_ov("poll_seconds", 10, lambda x: int(x) if x is not None else 10)),
        "trend_filter": int(_ov("trend_filter", 0, lambda x: int(x) if x is not None else 0)),
        "trend_sma": int(min(500, max(10, int(_ov("trend_sma", 200, lambda x: int(x) if x is not None else 200))))),
        "max_open_orders": int(min(50, max(1, int(_ov("max_open_orders", 6, lambda x: int(x) if x is not None else 6))))),
        "daily_loss_limit_pct": float(_ov("daily_loss_limit_pct", 0.06, lambda x: float(x) if x is not None else 0.06)),
        "pause_hours": int(_ov("pause_hours", 6, lambda x: int(x) if x is not None else 6)),
        "auto_restart": int(_ov("auto_restart", 0, lambda x: int(x) if x is not None else 0)),
        "vol_gap_mult": float(_ov("vol_gap_mult", 1.0, lambda x: float(x) if x is not None else 1.0)),
        "tp_vol_mult": float(_ov("tp_vol_mult", 1.0, lambda x: float(x) if x is not None else 1.0)),
        "min_gap_pct": float(_ov("min_gap_pct", 0.003, lambda x: float(x) if x is not None else 0.003)),
        "max_gap_pct": float(_ov("max_gap_pct", 0.06, lambda x: float(x) if x is not None else 0.06)),
        "regime_hold_candles": int(_ov("regime_hold_candles", 2, lambda x: int(x) if x is not None else 2)),
        "regime_switch_ticks": int(_ov("regime_switch_ticks", 2, lambda x: int(x) if x is not None else 2)),
        "regime_switch_threshold": float(_ov("regime_switch_threshold", 0.6, lambda x: float(x) if x is not None else 0.6)),
        "max_total_exposure_pct": float(_ov("max_total_exposure_pct", 0.50, lambda x: float(x) if x is not None else 0.50)),
        "per_symbol_exposure_pct": float(_ov("per_symbol_exposure_pct", 0.15, lambda x: float(x) if x is not None else 0.15)),
        "min_free_cash_pct": float(_ov("min_free_cash_pct", 0.1, lambda x: float(x) if x is not None else 0.1)),
        "max_concurrent_deals": int(_ov("max_concurrent_deals", 6, lambda x: int(x) if x is not None else 6)),
        "spread_guard_pct": float(_ov("spread_guard_pct", 0.003, lambda x: float(x) if x is not None else 0.003)),
        "limit_timeout_sec": int(_ov("limit_timeout_sec", 8, lambda x: int(x) if x is not None else 8)),
        "max_drawdown_pct": float(_ov("max_drawdown_pct", 0.0, lambda x: float(x) if x is not None else 0.0)),
    }
    if settings["max_spend_quote"] <= 0:
        settings["max_spend_quote"] = settings["base_quote"] + settings["safety_quote"] * settings["max_safety"]

    _sanitize_bot_numbers(settings)
    try:
        update_bot(int(bot_id), settings)
        return _json({"ok": True, "bot": get_bot(int(bot_id))})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


def _alpaca_client_for_bot(bot: Dict[str, Any]) -> Optional[AlpacaClient]:
    mode = str(bot.get("alpaca_mode") or "paper").lower()
    return alpaca_live if mode == "live" else alpaca_paper


def _alpaca_live_block_reason() -> str:
    """User-facing reason when Alpaca live is requested but not available."""
    if alpaca_live is not None:
        return ""
    if not LIVE_TRADING_ENABLED:
        return (
            "Alpaca live trading client not initialized. "
            "Set LIVE_TRADING_ENABLED=1 in .env and add Alpaca live API keys, then restart the app. "
            "Use Paper mode until then."
        )
    return (
        "Alpaca live trading client not initialized (startup failed). "
        "Check Alpaca live API keys in .env and app logs, then restart. Use Paper mode until then."
    )


def _can_start_bot_live(bot: Dict[str, Any]) -> Tuple[bool, str]:
    if bool(bot.get("dry_run", 1)):
        return True, ""
    market_type = str(bot.get("market_type") or "").strip().lower()
    if market_type == "crypto":
        if not KRAKEN_READY or kc is None:
            return False, KRAKEN_ERROR or "Kraken not ready"
        return True, ""
    if market_type == "stocks":
        cl = _alpaca_client_for_bot(bot)
        if cl is None:
            reason = _alpaca_live_block_reason() if (str(bot.get("alpaca_mode") or "paper").lower() == "live") else "Alpaca client not initialized"
            return False, reason
        try:
            adp = AlpacaAdapter(cl)
            adp.ensure_market(str(bot.get("symbol") or ""))
        except Exception as e:
            return False, f"Alpaca market check failed: {e}"
        return True, ""
    return False, f"Unknown market_type: {market_type}"


def _get_bot_client(bot: Dict[str, Any]):
    """
    Returns (client, is_kraken).
    If stocks, returns (AlpacaAdapter, False).
    If crypto, returns (KrakenClient, True).
    """
    market_type = str(bot.get("market_type") or "").strip().lower()
    if market_type not in ("stocks", "crypto"):
        raise HTTPException(status_code=400, detail=f"Unknown market_type: {market_type}")

    if market_type == "stocks":
        cl = _alpaca_client_for_bot(bot)
        if cl is None:
            logger.warning("_get_bot_client: Alpaca client missing bot_id=%s", bot.get("id"))
            raise HTTPException(status_code=503, detail="Alpaca client not available for stocks")
        try:
            adp = AlpacaAdapter(cl)
            sym = str(bot.get("symbol") or "")
            adp.ensure_market(sym)
        except Exception as e:
            logger.warning("_get_bot_client: Alpaca ensure_market failed bot_id=%s err=%s", bot.get("id"), e)
            raise HTTPException(status_code=503, detail=f"Alpaca market check failed: {e}")
        return adp, False

    if kc is None or not KRAKEN_READY:
        logger.warning("_get_bot_client: Kraken not ready bot_id=%s", bot.get("id"))
        raise HTTPException(status_code=503, detail=KRAKEN_ERROR or "Kraken not ready")
    return kc, True


# =========================================================
# Health
# =========================================================
@app.get("/health")
def health():
    """Health check for nginx/deploy. Always 200 + JSON so upstream never 502s."""
    try:
        ts = now_ts()
    except Exception:
        ts = int(time.time())
    try:
        db_ok = True
        try:
            init_db()
        except Exception:
            db_ok = False
        status = "healthy" if db_ok else "degraded"
        try:
            kr = bool(_kraken_ready())
        except Exception:
            kr = False
        # LIVE-HARDENED: uptime and last autopilot heartbeat for monitoring
        uptime_sec = int(time.time() - _APP_START_TIME) if _APP_START_TIME else 0
        last_autopilot_heartbeat = 0
        try:
            last_autopilot_heartbeat = int(get_setting("autopilot_last_heartbeat_ts", "0") or 0)
        except Exception:
            pass
        return {
            "ok": True,
            "status": status,
            "kraken_ready": kr,
            "kraken_error": KRAKEN_ERROR or "",
            "alpaca_paper_ready": ALPACA_PAPER_READY,
            "alpaca_live_ready": ALPACA_LIVE_READY,
            "alpaca_error": ALPACA_ERROR or "",
            "bot_manager_ready": bm is not None,
            "db_ok": db_ok,
            "time": ts,
            "uptime_sec": uptime_sec,
            "last_autopilot_heartbeat_ts": last_autopilot_heartbeat if last_autopilot_heartbeat else None,
        }
    except Exception as e:
        logger.exception("health handler error")
        return {"ok": False, "status": "error", "error": str(e)[:200], "time": ts}


@app.get("/ready")
def ready():
    """Readiness: DB + Alpaca + Kraken. Fast probe for deploy/k8s."""
    checks = {"db": False, "alpaca": False, "kraken": False}
    try:
        init_db()
        checks["db"] = True
    except Exception as e:
        return _json({"ok": False, "checks": checks, "error": f"db: {e}"}, 503)
    try:
        if ALPACA_PAPER_READY and alpaca_paper:
            _ = (alpaca_paper.get_account() or {})
            checks["alpaca"] = True
        elif not os.getenv("ALPACA_API_KEY_PAPER"):
            checks["alpaca"] = True  # Not configured, skip
        else:
            checks["alpaca"] = ALPACA_PAPER_READY
    except Exception as e:
        return _json({"ok": False, "checks": checks, "error": f"alpaca: {e}"}, 503)
    try:
        checks["kraken"] = _kraken_ready() if os.getenv("KRAKEN_API_KEY") else True
    except Exception:
        checks["kraken"] = False
    ok = checks["db"] and (checks["alpaca"] or not os.getenv("ALPACA_API_KEY_PAPER"))
    return _json({"ok": ok, "checks": checks})


# =========================================================
# API: global helpers for UI
# =========================================================
@app.get("/api/startup_status")
def api_startup_status():
    """Startup diagnostics - flask, db, alpaca, websocket, autopilot, candle fetch test."""
    s = dict(_STARTUP_STATUS)
    s["alpaca_ready"] = ALPACA_PAPER_READY or ALPACA_LIVE_READY
    if alpaca_paper:
        try:
            s["alpaca_buying_power"] = float((alpaca_paper.get_account() or {}).get("buying_power", 0))
        except Exception:
            pass
    try:
        import autopilot
        s["autopilot_enabled"] = autopilot.is_autopilot_enabled()
        s["autopilot_bots"] = len([b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"])
    except Exception:
        pass
    return _json({"ok": True, "startup": s})


@app.get("/api/health")
def api_health():
    """Health check for deployment and monitoring. Expanded: bots, DB metrics, circuit breaker, data quality."""
    try:
        db_ok = True
        try:
            init_db()
        except Exception:
            db_ok = False
        kr = bool(_kraken_ready())
        expanded = {}
        try:
            from health_monitor import build_expanded_health
            expanded = build_expanded_health(
                kraken_ready=kr,
                kraken_error=KRAKEN_ERROR or "",
                alpaca_paper_ready=ALPACA_PAPER_READY,
                alpaca_live_ready=ALPACA_LIVE_READY,
                alpaca_error=ALPACA_ERROR or "",
                bot_manager_ready=bm is not None,
                db_ok=db_ok,
                list_bots_fn=list_bots,
                last_portfolio_ts=_last_portfolio_ts,
                last_reco_short_ts=_last_reco_short_ts,
                last_reco_long_ts=_last_reco_long_ts,
            )
        except Exception as e:
            logger.debug("health_monitor expand failed: %s", e)
            expanded = {"ok": db_ok, "status": "healthy" if db_ok else "degraded"}
        with _thread_start_lock:
            expanded["threads_started"] = list(_thread_started.keys())
        expanded["last_portfolio_ts"] = _last_portfolio_ts
        expanded["last_reco_short_ts"] = _last_reco_short_ts
        expanded["last_reco_long_ts"] = _last_reco_long_ts
        # LIVE-HARDENED: uptime and autopilot heartbeat for dashboard monitoring
        expanded["uptime_sec"] = int(time.time() - _APP_START_TIME) if _APP_START_TIME else 0
        try:
            expanded["last_autopilot_heartbeat_ts"] = int(get_setting("autopilot_last_heartbeat_ts", "0") or 0) or None
        except Exception:
            expanded["last_autopilot_heartbeat_ts"] = None
        return _json(expanded)
    except Exception as e:
        return _json({"ok": False, "status": "error", "error": str(e), "ts": now_ts()}, 503)


@app.get("/api/health/metrics")
def api_health_prometheus():
    """Prometheus metrics (optional, ENABLE_PROMETHEUS=1)."""
    try:
        from health_monitor import prometheus_metrics
        out = prometheus_metrics()
        if out:
            from fastapi.responses import PlainTextResponse
            return PlainTextResponse(out, media_type="text/plain; version=0.0.4")
    except Exception:
        pass
    return _json({"ok": False, "message": "Prometheus disabled"})


@app.get("/api/comprehensive_health")
def api_comprehensive_health():
    """Run comprehensive health check and return issues (if any)."""
    try:
        ok, issues = _comprehensive_health_check()
        return _json({"ok": ok, "issues": issues, "message": "All systems operational" if ok else f"{len(issues)} issue(s) found"})
    except Exception as e:
        return _json({"ok": False, "issues": [str(e)], "error": str(e)}, 500)


@app.get("/api/diag/network")
def api_diag_network():
    """Diagnose Alpaca connectivity from this server: DNS resolution + HTTPS HEAD."""
    result: Dict[str, Any] = {
        "ok": False,
        "dns": {},
        "https_head": {},
        "alpaca_data_feed": os.getenv("ALPACA_DATA_FEED", "").strip() or "(default)",
        "alpaca_mode": "paper",
    }
    host = "paper-api.alpaca.markets"
    # DNS resolution
    try:
        addrs = socket.getaddrinfo(host, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
        result["dns"] = {"ok": True, "host": host, "resolved": [a[4][0] for a in addrs]}
    except Exception as e:
        result["dns"] = {"ok": False, "host": host, "error": str(e)}
    # HTTPS HEAD
    try:
        import urllib.request
        req = urllib.request.Request("https://" + host, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result["https_head"] = {"ok": True, "status_code": resp.status}
    except Exception as e:
        result["https_head"] = {"ok": False, "error": str(e)}
    result["ok"] = result["dns"].get("ok", False) and result["https_head"].get("ok", False)
    return _json(result)


@app.get("/api/stats")
def api_stats():
    """UnifiedAlpacaClient stats: cache hits, rate limit, WebSocket subscriptions."""
    client = alpaca_live or alpaca_paper
    if not client:
        return _json({"ok": False, "message": "Alpaca not configured"})
    if not hasattr(client, "get_stats"):
        return _json({"ok": True, "message": "Using legacy AlpacaClient (no stats)", "client": "legacy"})
    try:
        stats = client.get_stats()
        return _json({"ok": True, "client": "unified", **stats})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/websocket_stats")
def api_websocket_stats():
    """WebSocket, cache, and rate limiter stats (alias for /api/stats when using UnifiedAlpacaClient)."""
    return api_stats()


@app.get("/api/safety_check")
def api_safety_check():
    """Live-trading safety check for UI and automation. Includes kill_switch and allow_live_trading for checklist."""
    blocking = []
    api_auth_enabled = bool(WORKER_API_TOKEN)
    kraken_ready = bool(_kraken_ready())
    alpaca_paper_ready = bool(ALPACA_PAPER_READY)
    alpaca_live_ready = bool(ALPACA_LIVE_READY)
    any_live = _has_live_bots()
    live_context = LIVE_TRADING_ENABLED or any_live
    kill_switch = bool(get_setting("kill_switch", "0").strip().lower() in ("1", "true", "yes", "y", "on"))
    allow_live_trading = bool(getattr(bm, "ALLOW_LIVE_TRADING", False) if bm else False)
    if not bm:
        try:
            from bot_manager import ALLOW_LIVE_TRADING as _ALLOW
            allow_live_trading = bool(_ALLOW)
        except Exception:
            pass

    if live_context and not api_auth_enabled:
        blocking.append("WORKER_API_TOKEN missing for live trading")
    if LIVE_ENDPOINTS_DISABLED:
        blocking.append(LIVE_ENDPOINTS_DISABLED_REASON or "Live endpoints disabled")
    if any_live and not (kraken_ready or alpaca_paper_ready or alpaca_live_ready):
        blocking.append("Live bots configured but no trading clients ready")

    live_ready = LIVE_TRADING_ENABLED and allow_live_trading and (kraken_ready or alpaca_live_ready) and api_auth_enabled and not blocking and not kill_switch
    return _json(
        {
            "ok": True,
            "api_auth_enabled": api_auth_enabled,
            "allow_live_trading": allow_live_trading,
            "live_trading_enabled": LIVE_TRADING_ENABLED,
            "kill_switch": kill_switch,
            "kraken_ready": kraken_ready,
            "alpaca_paper_ready": alpaca_paper_ready,
            "alpaca_live_ready": alpaca_live_ready,
            "live_ready": live_ready,
            "any_live_bots": any_live,
            "blocking_issues": blocking,
        }
    )


@app.get("/api/pnl")
def api_pnl():
    """
    Used by dashboard / UI to show today and total realized PnL.
    """
    try:
        today = pnl_summary(_midnight_local_ts())
        total = pnl_summary(0)
        return _json({"ok": True, "today": today, "total": total})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "today": {}, "total": {}}, 500)


@app.get("/api/symbols")
def api_symbols(quote: str = "USD"):
    if not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "symbols": []}, 503)
    try:
        symbols = kc.list_spot_symbols(quote=quote)
        return _json({"ok": True, "symbols": symbols})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "symbols": []}, 500)


@app.get("/api/alpaca/symbols")
def api_alpaca_symbols():
    """Get list of tradeable stock symbols from Alpaca"""
    try:
        # Use paper trading client to get symbols (same symbols for both paper and live)
        if not alpaca_paper:
            return _json({"ok": False, "error": "Alpaca not initialized", "symbols": []}, 503)
        
        # Get top tradeable stocks
        assets = alpaca_paper.search_assets(query="", asset_class="us_equity")
        
        # Filter to tradeable and sort by common popularity
        popular_symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "V", "JNJ",
            "WMT", "JPM", "MA", "PG", "UNH", "DIS", "HD", "PYPL", "BAC", "VZ",
            "ADBE", "NFLX", "CRM", "NKE", "CMCSA", "PFE", "T", "INTC", "CSCO", "ABT",
            "KO", "PEP", "MRK", "AVGO", "TMO", "COST", "ABBV", "ACN", "TXN", "NEE",
            "DHR", "LLY", "MDT", "UNP", "BMY", "PM", "QCOM", "HON", "UPS", "LOW"
        ]
        
        # Create symbols list with names
        symbols = []
        for asset in assets[:500]:  # Limit to 500 most common
            symbol = asset.get("symbol", "")
            name = asset.get("name", symbol)
            
            # Prioritize popular symbols
            if symbol in popular_symbols:
                symbols.insert(0, {"symbol": symbol, "name": name})
            else:
                symbols.append({"symbol": symbol, "name": name})
        
        return _json({"ok": True, "symbols": symbols})
    except Exception as e:
        logger.error(f"Alpaca symbols error: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "symbols": []}, 500)


@app.get("/api/prices")
def api_prices(symbols: str = "", market_type: str = "all"):
    """
    Batch price fetch endpoint. Fast, cached, supports both crypto and stocks.
    symbols: comma-separated list (e.g. "XBT/USD,ETH/USD,INTC,AAPL")
    market_type: "crypto", "stocks", or "all"
    Returns { prices: { "XBT/USD": 12345.0, "INTC": 45.23, ... }, changes: {...}, volumes: {...} }
    """
    from symbol_classifier import is_stock_symbol, is_crypto_symbol
    
    out: Dict[str, Optional[float]] = {}
    changes: Dict[str, Optional[float]] = {}
    volumes: Dict[str, Optional[float]] = {}
    
    req = [s.strip() for s in (symbols or "").split(",") if s.strip()]
    if not req:
        return _json({"ok": True, "prices": out, "changes": changes, "volumes": volumes})
    
    _prices_deadline = time.time() + 5.0
    
    # Separate crypto and stocks
    crypto_symbols = []
    stock_symbols = []
    
    for s in req:
        if is_stock_symbol(s):
            stock_symbols.append(s)
        elif is_crypto_symbol(s) or "/" in s:
            crypto_symbols.append(s)
        else:
            # Try to infer - if short and no slash, assume stock
            if len(s) < 6:
                stock_symbols.append(s)
            else:
                crypto_symbols.append(s)
    
    # Fetch crypto prices: batch ticker (1 Kraken call) when 3+ symbols, else per-symbol cache
    if crypto_symbols and (market_type == "all" or market_type == "crypto") and time.time() < _prices_deadline:
        if _kraken_ready():
            mk = _markets()
            use_batch = len(crypto_symbols) >= 3
            batch_map = _tickers_batch_cached(ttl_sec=15) if use_batch else {}
            for s in crypto_symbols:
                if time.time() >= _prices_deadline:
                    break
                norm = _normalize_symbol(s)
                resolved = _resolve_symbol(norm)
                ticker = None
                if use_batch and batch_map:
                    ticker = batch_map.get(resolved) or batch_map.get(norm) or batch_map.get(s)
                else:
                    if mk and resolved in mk:
                        ticker = _ticker_cached(resolved, ttl_sec=15) or {}
                if ticker:
                    price = float(ticker.get("last") or 0.0) if ticker.get("last") else None
                    out[norm] = price if price and price > 0 else None
                    pct = ticker.get("percentage")
                    changes[norm] = float(pct) if pct is not None else None
                    qv = ticker.get("quoteVolume")
                    volumes[norm] = float(qv) if qv is not None else None
                else:
                    out[norm] = None
    
    # Fetch stock prices (batch snapshot from Alpaca)
    if stock_symbols and (market_type == "all" or market_type == "stocks") and time.time() < _prices_deadline:
        client = alpaca_live if alpaca_live else alpaca_paper
        if client:
            try:
                for i in range(0, len(stock_symbols), 100):
                    if time.time() >= _prices_deadline:
                        break
                    batch = stock_symbols[i:i+100]
                    try:
                        snap_data = client.get_snapshots(batch)
                        # Alpaca returns {"snapshots": {"SYMBOL": {...}}}
                        snapshots = snap_data.get("snapshots", {}) if isinstance(snap_data, dict) else {}
                        for sym in batch:
                            snap = snapshots.get(sym) or snapshots.get(sym.upper()) or {}
                            if snap:
                                latest_trade = snap.get("latestTrade", {})
                                daily_bar = snap.get("dailyBar", {}) or {}
                                prev_bar = snap.get("prevDailyBar", {}) or {}
                                
                                price = None
                                if latest_trade and latest_trade.get("p"):
                                    price = float(latest_trade.get("p", 0))
                                elif daily_bar and daily_bar.get("c"):
                                    price = float(daily_bar.get("c", 0))
                                
                                out[sym] = price if price and price > 0 else None
                                
                                # Change %: prefer prev close vs current; else intraday (close vs open)
                                if price and price > 0:
                                    prev_c = prev_bar.get("c") if prev_bar else None
                                    if prev_c is not None:
                                        prev_close = float(prev_c)
                                        if prev_close > 0:
                                            changes[sym] = ((price - prev_close) / prev_close) * 100.0
                                    elif daily_bar and daily_bar.get("o") is not None:
                                        o = float(daily_bar.get("o", 0))
                                        if o > 0:
                                            changes[sym] = ((price - o) / o) * 100.0
                                
                                if daily_bar and daily_bar.get("v") is not None:
                                    volumes[sym] = float(daily_bar.get("v", 0))
                            else:
                                out[sym] = None
                    except Exception as e:
                        logger.warning(f"Batch snapshot fetch failed for batch: {e}")
                        # Fallback to individual fetches from cache
                        for sym in batch:
                            ticker = _ticker_cached(sym, ttl_sec=15) or {}
                            price = float(ticker.get("last") or 0.0) if ticker.get("last") else None
                            out[sym] = price if price and price > 0 else None
            except Exception as e:
                logger.warning(f"Stock price batch fetch error: {e}")
                # Fallback: mark all as None
                for sym in stock_symbols:
                    out[sym] = None
    
    return _json({"ok": True, "prices": out, "changes": changes, "volumes": volumes})


@app.get("/api/market/ticker")
def api_market_ticker(symbol: str):
    """Get ticker data for a symbol, routing to appropriate provider"""
    market_type = classify_symbol(symbol)
    
    if market_type == "stock":
        # Route to Alpaca
        client = alpaca_live if alpaca_live else alpaca_paper
        if not client:
            return _json({"ok": False, "error": "Alpaca not configured for stock symbols"}, 503)
        
        try:
            ticker = client.get_ticker(symbol)
            return _json({"ok": True, **ticker})
        except Exception as e:
            return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
    
    else:
        # Crypto path - existing Kraken logic
        if not _kraken_ready():
            return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        sym = _resolve_symbol(symbol)
        mk = _markets()
        if mk and sym not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {sym}"}, 400)
        data = _ticker_cached(sym, ttl_sec=30) or {}
        return _json({"ok": True, **data})


@app.get("/api/market/ohlcv")
def api_market_ohlcv(symbol: str, tf: str = "1h", limit: int = 500):
    """Get OHLCV data, routing to appropriate provider"""
    market_type = classify_symbol(symbol)
    
    # 1. STOCK ROUTING
    if market_type == "stock":
        try:
            client = alpaca_live if alpaca_live else alpaca_paper
            if client:
                safe_limit = int(max(10, min(int(limit), 1000)))
                # Alpaca get_ohlcv(symbol, timeframe, limit)
                candles = client.get_ohlcv(symbol, tf, safe_limit)
                return _json({"ok": True, "symbol": symbol, "tf": tf, "candles": candles})
            else:
                 return _json({"ok": False, "error": "Alpaca not ready", "candles": []}, 503)
        except Exception as e:
            return _json({"ok": False, "error": str(e), "candles": []}, 500)

    # 2. CRYPTO ROUTING (Existing Logic)
    if not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "candles": []}, 503)
    sym = _resolve_symbol(symbol)
    mk = _markets()
    if mk and sym not in mk:
        return _json({"ok": False, "error": f"Symbol not found on Kraken: {sym}", "candles": []}, 400)
    safe_tf = _sanitize_tf(tf)
    safe_limit = int(max(10, min(int(limit), 1000)))
    ttl = max(30, min(300, _tf_seconds(safe_tf)))
    candles = _ohlcv_cached(sym, safe_tf, safe_limit, ttl)
    return _json({"ok": True, "symbol": sym, "tf": safe_tf, "candles": candles})


def _check_trading_allowed(bot_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Returns error dict if trading blocked (kill switch, global pause, circuit breaker, data quality). None if OK."""
    if _pause_state():
        return {"ok": False, "error": "Trading paused (global pause). Resume from Safety or Pause page."}
    if _kill_switch_state():
        return {"ok": False, "error": "Kill switch is on. Turn it off on the Safety page to trade."}
    try:
        from circuit_breaker import is_emergency_stop_active, is_bot_circuit_open, get_bot_pause_until
        if is_emergency_stop_active():
            return {"ok": False, "error": "Emergency stop active. Exchange errors persist. Check /api/health."}
        if bot_id is not None and is_bot_circuit_open(int(bot_id)):
            until = get_bot_pause_until(int(bot_id))
            return {"ok": False, "error": f"Circuit breaker: bot paused until errors clear (until ts {until})"}
    except ImportError:
        pass
    try:
        from data_validator import is_data_quality_degraded
        if is_data_quality_degraded():
            return {"ok": False, "error": "Data quality degraded (5+ issues in 15 min). Trading paused."}
    except ImportError:
        pass
    return None


@app.post("/api/orders/buy")
async def api_orders_buy(request: Request):
    try:
        payload = await request.json()
    except Exception:
        # Handle case where body might be already read or invalid
        return _json({"ok": False, "error": "Invalid payload or body stream consumed"}, 400)
        
    if not isinstance(payload, dict):
        return _json({"ok": False, "error": "Invalid payload"}, 400)

    dry_run = bool(payload.get("dry_run", True))
    if not dry_run:
        block = _check_trading_allowed(bot_id=None)
        if block:
            return _json(block, 503)
    
    raw_symbol = str(payload.get("symbol") or "")
    market_type = classify_symbol(raw_symbol)
    quote_usd = float(payload.get("quote_usd") or 0.0)
    limit_price = float(payload.get("limit_price") or 0.0)
    dry_run = bool(payload.get("dry_run", True))

    if quote_usd <= 0 or limit_price <= 0:
        return _json({"ok": False, "error": "quote_usd and limit_price must be > 0"}, 400)

    # 1. STOCK ROUTING
    if market_type == "stock":
        client = alpaca_live if alpaca_live else alpaca_paper
        if not client:
            return _json({"ok": False, "error": "Alpaca not configured for stocks"}, 503)
            
        # CRITICAL: Calculate and validate amount before placing order
        if limit_price <= 0:
            return _json({"ok": False, "error": f"Invalid limit_price: {limit_price}"}, 400)
        
        base_amount = quote_usd / limit_price
        
        # CRITICAL: Reject zero or invalid amounts
        import math
        if base_amount <= 0 or math.isnan(base_amount) or math.isinf(base_amount):
            return _json({
                "ok": False, 
                "error": f"Order size invalid: amount={base_amount}, quote_usd={quote_usd}, limit_price={limit_price}. Order skipped."
            }, 400)
        
        if dry_run:
             return _json({
                "ok": True, 
                "message": "Dry run: stock limit buy simulated.",
                "order": {
                    "symbol": raw_symbol, "side": "buy", "type": "limit",
                    "price": limit_price, "amount": base_amount, "market_type": "stock"
                }
            })
            
        try:
            # Alpaca place_limit_order(symbol, qty, limit_price, side, time_in_force)
            # Check if client has this method or similar
            if hasattr(client, "place_limit_order"):
                order = client.place_limit_order(raw_symbol, base_amount, limit_price, "buy")
            else:
                 return _json({"ok": False, "error": "Alpaca client missing place_limit_order"}, 500)
                 
            return _json({"ok": True, "message": "Stock limit buy placed.", "order": order})
        except Exception as e:
            return _json({"ok": False, "error": f"Alpaca Order Failed: {e}"}, 500)

    # 2. CRYPTO ROUTING (Existing Logic)
    if not dry_run and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

    symbol = _resolve_symbol(raw_symbol)
    mk = _markets()
    if mk and symbol not in mk:
        return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    spread = _safe_spread_pct(symbol)
    if spread is not None and spread > (RECO_MAX_SPREAD_PCT * 2):
        return _json({"ok": False, "error": f"Spread too wide ({spread:.2%})"}, 400)

    # CRITICAL: Calculate and validate amount before placing order
    if limit_price <= 0:
        return _json({"ok": False, "error": f"Invalid limit_price: {limit_price}"}, 400)
    
    base_amount = quote_usd / limit_price
    
    # CRITICAL: Reject zero or invalid amounts
    import math
    if base_amount <= 0 or math.isnan(base_amount) or math.isinf(base_amount):
        return _json({
            "ok": False, 
            "error": f"Order size invalid: amount={base_amount}, quote_usd={quote_usd}, limit_price={limit_price}. Order skipped."
        }, 400)
    
    if dry_run:
        return _json(
            {
                "ok": True,
                "message": "Dry run: limit buy simulated.",
                "order": {
                    "symbol": symbol,
                    "side": "buy",
                    "type": "limit",
                    "price": limit_price,
                    "amount": base_amount,
                },
            }
        )

    try:
        order = kc.create_limit_buy_base(symbol, base_amount, limit_price)
        return _json({"ok": True, "message": "Limit buy placed.", "order": _serialize_order(order)})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.get("/api/market/overview")
def api_market_overview(quote: str = "USD", limit: int = 50, market_type: str = "crypto"):
    """
    Returns categorized market data for the Explore dashboard.
    Structure: { "ok": true, "gainers": [], "losers": [], "hot": [], "trending": [] }
    """
    # 1. Stocks Mode
    if market_type == "stocks":
        try:
            if alpaca_paper or alpaca_live:
                client = alpaca_live if alpaca_live else alpaca_paper
                data = client.get_top_movers() 
                return _json({
                    "ok": True,
                    "gainers": data.get("gainers", []),
                    "losers": data.get("losers", []),
                    "hot": data.get("hot", []),
                    "trending": data.get("hot", []),
                    "market_type": "stocks"
                })
            else:
                 return _json({"ok": False, "error": "Alpaca not ready"}, 503)
        except Exception as e:
            logger.error(f"Alpaca overview failed: {e}")
            return _json({"ok": False, "error": str(e)}, 500)

    # 2. Crypto Mode (Default)
    if not _kraken_ready():
        return _json({"ok": False, "error": "No market data available"}, 503)

    try:
        # Fetch tickers for top assets
        tickers = kc.ex.fetch_tickers()
        
        parsed = []
        q_upper = quote.upper()
        
        for sym, t in tickers.items():
            if f"/{q_upper}" not in sym:
                continue
            
            # Simple volume filter to remove garbage
            vol = float(t.get("quoteVolume") or 0)
            if vol < 50000: # Min $50k volume
                continue
                
            change = float(t.get("percentage") or 0)
            close = float(t.get("last") or 0)
            
            parsed.append({
                "symbol": sym,
                "last": close,
                "percentage": change,
                "quoteVolume": vol
            })
            
        parsed.sort(key=lambda x: x["percentage"], reverse=True)
        gainers = parsed[:6]
        losers = sorted(parsed, key=lambda x: x["percentage"])[:6]
        
        parsed.sort(key=lambda x: x["quoteVolume"], reverse=True)
        hot = parsed[:6]
        
        return _json({
            "ok": True,
            "gainers": gainers,
            "losers": losers,
            "hot": hot,
            "trending": hot
        })

    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/portfolio")
def api_portfolio():
    snap = _portfolio_snapshot()
    with _globals_lock:
        history = list(PORT_HISTORY[-500:])
    return _json({"ok": True, "portfolio": snap, "history": history})


@app.get("/api/portfolio/performance")
def api_portfolio_performance(timeframe: str = "1D"):
    """Portfolio performance for charts. Uses PORT_HISTORY."""
    try:
        snap = _portfolio_snapshot()
        total_usd = float(snap.get("total_usd") or 0)
        with _globals_lock:
            history = list(PORT_HISTORY[-500:])
        cutoff_ts = 0
        if timeframe == "1H":
            cutoff_ts = now_ts() - 3600
        elif timeframe == "4H":
            cutoff_ts = now_ts() - 14400
        elif timeframe == "1D":
            cutoff_ts = now_ts() - 86400
        elif timeframe == "1W":
            cutoff_ts = now_ts() - 604800
        elif timeframe == "1M":
            cutoff_ts = now_ts() - 2592000
        elif timeframe == "3M":
            cutoff_ts = now_ts() - 7776000
        elif timeframe == "1Y":
            cutoff_ts = now_ts() - 31536000
        filtered = [h for h in history if (h.get("ts") or 0) >= cutoff_ts] if cutoff_ts else history
        series = [
            {"timestamp": time.strftime("%Y-%m-%d %H:%M", time.localtime(h.get("ts", 0))), "value": float(h.get("total_usd") or 0), "pnl_pct": 0}
            for h in filtered
        ]
        if not series and total_usd > 0:
            series = [{"timestamp": time.strftime("%Y-%m-%d %H:%M", time.localtime()), "value": total_usd, "pnl_pct": 0}]
        pos_usd = float(snap.get("positions_usd") or 0)
        free = float(snap.get("free_usd") or 0)
        allocation = [
            {"name": "Crypto", "value": pos_usd * 0.6},
            {"name": "Stocks", "value": pos_usd * 0.4},
            {"name": "Cash", "value": free},
        ]
        return _json({"ok": True, "series": series, "allocation": allocation, "timeframe": timeframe})
    except Exception as e:
        logger.exception("portfolio/performance: %s", e)
        return _json({"ok": True, "series": [], "allocation": [], "timeframe": timeframe, "error": str(e)})


@app.get("/api/tax_optimization_suggestions")
def api_tax_optimization_suggestions(min_loss_pct: float = 5.0):
    """Tax-loss harvesting suggestions for positions with unrealized loss >= min_loss_pct."""
    try:
        from tax_optimizer import tax_harvest_suggestions, ENABLE_TAX_HARVESTING
    except ImportError:
        return _json({"ok": False, "error": "Tax optimizer not available", "suggestions": []})
    if not ENABLE_TAX_HARVESTING:
        return _json({"ok": True, "enabled": False, "suggestions": [], "message": "Tax harvesting disabled"})
    positions = []
    if bm:
        try:
            for bot in list_bots():
                od = latest_open_deal(int(bot.get("id") or 0))
                if od and od.get("state") == "OPEN":
                    sym = bot.get("symbol") or od.get("symbol")
                    entry = float(od.get("entry_avg") or 0)
                    if sym and entry > 0:
                        price = _ticker_cached(sym, ttl_sec=60)
                        cur = float(price.get("last", 0) or price.get("c", 0) or 0) if price else 0
                        if cur > 0:
                            positions.append({
                                "symbol": sym,
                                "entry_price": entry,
                                "current_price": cur,
                                "avg_entry": entry,
                                "last_price": cur,
                            })
        except Exception as e:
            logger.warning("tax_optimization positions fetch: %s", e)
    suggestions = tax_harvest_suggestions(positions, min_loss_pct=float(min_loss_pct))
    return _json({"ok": True, "enabled": True, "suggestions": suggestions})


@app.get("/api/portfolio/rebalance_suggestions")
def api_rebalance_suggestions():
    """Portfolio rebalancing suggestions based on TARGET_ALLOCATIONS vs current sector allocation."""
    try:
        from sector_rotation import get_rotation_suggestions
        from stock_metadata import get_sector
    except ImportError:
        return _json({"ok": False, "error": "Sector rotation not available", "suggestions": []})
    sector_alloc = {}
    if bm:
        try:
            for bot in list_bots():
                od = latest_open_deal(int(bot.get("id") or 0))
                if od and bot.get("symbol"):
                    sym = bot.get("symbol")
                    sector = get_sector(sym)
                    if sector:
                        sector_alloc[sector] = sector_alloc.get(sector, 0.0) + 1.0
        except Exception as e:
            logger.warning("rebalance sector alloc: %s", e)
    suggestions = get_rotation_suggestions(sector_alloc)
    return _json({"ok": True, "suggestions": suggestions, "current_allocations": sector_alloc})


# =========================================================
# API: bots list/detail (DB only)
# =========================================================
@app.get("/api/bots/{bot_id}")
def api_bot(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    return _json({"ok": True, "bot": b})


# =========================================================
# API: bot runtime control (start/stop defined later with full snap response)
# =========================================================
@app.delete("/api/bots/{bot_id}")
def api_bots_delete(bot_id: int):
    bid = int(bot_id)
    b = get_bot(bid)
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm:
        try:
            bm.stop(bid)
        except Exception as e:
            logger.warning("api_bots_delete: stop failed for bot %s: %s", bid, e)
    try:
        _discord_notify(f"🗑️ {b.get('name') or bid} deleted.")
    except Exception:
        pass
    try:
        delete_bot(bid)
        return _json({"ok": True, "message": "Bot deleted"})
    except Exception as e:
        logger.exception("api_bots_delete: delete_bot failed for bot %s", bid)
        return _json({"ok": False, "error": f"Delete failed: {e}"}, 500)


@app.get("/api/bots/{bot_id}/dealstats")
def api_bot_dealstats(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        stats = bot_deal_stats(int(bot_id))
        od = latest_open_deal(int(bot_id))
        return _json({"ok": True, "stats": stats, "open_deal": od})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "stats": {}}, 500)


@app.get("/api/bots/{bot_id}/pnl_series")
def api_bot_pnl_series(bot_id: int, limit: int = 500):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        series = bot_pnl_series(int(bot_id), limit=int(max(10, min(5000, int(limit)))))
        return _json({"ok": True, "series": series})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "series": []}, 500)


@app.get("/api/bots/{bot_id}/metrics")
def api_bot_metrics(bot_id: int, limit: int = 500):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        perf = bot_performance_stats(int(bot_id))
        dd = bot_drawdown_series(int(bot_id), limit=int(max(10, min(5000, int(limit)))))
        return _json({"ok": True, "perf": perf, "drawdown_series": dd})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "perf": {}, "drawdown_series": []}, 500)


@app.get("/api/pause")
def api_pause_state():
    return _json({"ok": True, "paused": bool(_pause_state())})


@app.post("/api/pause")
async def api_pause_set(request: Request):
    payload = await request.json()
    paused = bool(payload.get("paused"))
    pause_hours_raw = payload.get("pause_hours", os.getenv("DEFAULT_GLOBAL_PAUSE_HOURS", "6"))
    try:
        pause_hours = max(0.0, float(pause_hours_raw))
    except Exception:
        pause_hours = 6.0
    try:
        if paused:
            set_setting("global_pause", "1")
            if pause_hours > 0:
                until_ts = int(time.time()) + int(pause_hours * 3600)
                set_setting("global_pause_until", str(until_ts))
            else:
                # Explicit indefinite pause when pause_hours=0.
                set_setting("global_pause_until", "0")
        else:
            set_setting("global_pause", "0")
            set_setting("global_pause_until", "0")
    except Exception:
        pass
    return _json({
        "ok": True,
        "paused": bool(_pause_state()),
        "pause_hours": pause_hours,
        "global_pause_until": get_setting("global_pause_until", "0"),
    })


@app.get("/api/risk/kill")
def api_kill_state():
    return _json({"ok": True, "kill_switch": bool(_kill_switch_state())})


@app.post("/api/risk/kill")
async def api_kill_set(request: Request):
    payload = await request.json()
    enabled = bool(payload.get("enabled"))
    try:
        set_setting("kill_switch", "1" if enabled else "0")
    except Exception:
        pass
    return _json({"ok": True, "kill_switch": bool(_kill_switch_state())})


@app.get("/api/bots/{bot_id}/status")
def api_bot_status(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm is None:
        # Degraded mode: return minimal status so UI shows bot info instead of "Worker not initialized"
        snap = {
            "running": False,
            "last_event": "Worker not initialized. Check Kraken/Alpaca API keys and restart the service.",
            "last_price": None,
            "avg_entry": None,
            "base_pos": None,
        }
        return _json({
            "ok": True,
            "bot": b,
            "snap": snap,
            "regime": None,
            "kraken_ready": _kraken_ready(),
            "kraken_error": KRAKEN_ERROR,
            "paused": bool(_pause_state()),
            "data_health": None,
            "worker_degraded": True,
        })

    try:
        snap = bm.snapshot(int(bot_id))
    except Exception as e:
        logger.warning("api_bot_status snapshot failed bot_id=%s: %s", bot_id, e)
        # Return valid response so UI shows error instead of "Status unavailable"
        snap = {
            "running": False,
            "last_event": f"Error: {str(e)[:80]}",
            "last_price": None,
            "avg_entry": None,
            "base_pos": None,
        }

    if snap.get("running") and not snap.get("last_event"):
        snap["last_event"] = "Running."
    if not snap.get("running") and not snap.get("last_event"):
        snap["last_event"] = "Stopped."
    
    # Get last price if not in snapshot - use correct provider based on symbol type
    if snap.get("last_price") is None:
        symbol = b.get("symbol", "")
        lp = None
        try:
            market_type = classify_symbol(symbol)
            if market_type == "stock":
                # Stock: use Alpaca
                client = alpaca_live if alpaca_live else alpaca_paper
                if client:
                    try:
                        ticker = client.get_ticker(symbol)
                        lp = ticker.get("last") if ticker else None
                    except Exception:
                        pass
            else:
                # Crypto: use Kraken
                lp = _safe_last_price(symbol)
        except Exception:
            pass
        if lp is not None:
            snap["last_price"] = lp

    try:
        regime = latest_regime(int(bot_id))
    except Exception:
        regime = None

    data_health = None
    try:
        router = getattr(bm, "_md_router", None)
        if router:
            symbol = b.get("symbol", "")
            mt = b.get("market_type", "crypto")
            data_health = router.get_data_health(symbol, mt, required_tfs=["1h", "4h", "1d"], min_candles=20)
    except Exception as e:
        logger.debug("data_health fetch failed: %s", e)

    return _json({
        "ok": True,
        "bot": b,
        "snap": snap,
        "regime": regime,
        "kraken_ready": _kraken_ready(),
        "kraken_error": KRAKEN_ERROR,
        "paused": bool(_pause_state()),
        "data_health": data_health,
    })


@app.get("/api/positions/{bot_id}")
def api_positions_bot(bot_id: int):
    """Get positions for a specific bot (production-ready)."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    snap = {}
    if bm:
        try:
            snap = bm.snapshot(int(bot_id)) or {}
        except Exception:
            pass
    symbol = b.get("symbol", "")
    last_price = float(snap.get("last_price") or 0)
    if last_price <= 0 and symbol:
        try:
            if classify_symbol(symbol) == "stock" and (alpaca_live or alpaca_paper):
                client = alpaca_live or alpaca_paper
                t = client.get_ticker(symbol)
                last_price = float(t.get("last") or 0)
            else:
                tc = _ticker_cached(symbol, ttl_sec=60)
                if tc:
                    last_price = float(tc.get("last") or tc.get("c") or 0)
        except Exception:
            pass
    avg_entry = float(snap.get("avg_entry") or 0)
    base_pos = float(snap.get("base_pos") or 0)
    if avg_entry <= 0 and base_pos > 0 and last_price > 0:
        avg_entry = last_price
    position_value = base_pos * last_price if last_price > 0 else base_pos
    unrealized_pnl = 0.0
    unrealized_pnl_pct = 0.0
    if avg_entry > 0 and base_pos > 0:
        unrealized_pnl = (last_price - avg_entry) * base_pos
        unrealized_pnl_pct = ((last_price - avg_entry) / avg_entry) * 100
    pos = {
        "bot_id": int(bot_id),
        "symbol": symbol,
        "strategy": b.get("strategy_mode", "classic"),
        "avg_entry_price": avg_entry,
        "current_price": last_price,
        "position_value": position_value,
        "quantity": base_pos,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "take_profit_price": snap.get("tp_price"),
        "stop_loss_price": None,
    }
    return _json({"ok": True, "positions": [pos] if base_pos > 0 else []})


@app.get("/api/bots/{bot_id}/regime")
def api_bot_regime(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    return _json({"ok": True, "regime": latest_regime(int(bot_id))})


@app.get("/api/bots/{bot_id}/recommendation")
def api_bot_recommendation(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    try:
        client, is_kraken = _get_bot_client(b)
        if not client:
             return _json({"ok": False, "error": "Trading client not available"}, 503)
        if is_kraken and not _kraken_ready():
             return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)

        symbol = _resolve_symbol(b.get("symbol", ""))
        
        # Validation
        if is_kraken:
            mk = _markets()
            if mk and symbol not in mk:
                return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)

    try:
        # Use cached OHLCV from BotManager if available
        if bm:
            candles = bm.ohlcv_cached(symbol, "15m", limit=300)
        else:
            # Fallback if worker not running
            candles = client.fetch_ohlcv(symbol, timeframe="15m", limit=200)
            
        regime = detect_regime(candles)
        target, switched, reason = select_strategy(
            regime=regime,
            current="smart_dca",
            last_switch_ts=0,
            now_ts=now_ts(),
            forced=None,
            vol_ratio=regime.vol_ratio,
        )
        note = ""
        mode = str(b.get("strategy_mode") or "").lower()
        forced = str(b.get("forced_strategy") or "").lower()
        if forced:
            note = f"Bot is forcing '{forced}'. Recommendation ignores forced."
        elif mode and mode not in ("auto", "router"):
            note = f"Bot strategy mode is '{mode}'. Recommendation ignores manual mode."

        return _json(
            {
                "ok": True,
                "symbol": symbol,
                "regime": {"label": regime.regime, "confidence": regime.confidence, "why": regime.why},
                "recommended": target,
                "reason": reason,
                "note": note,
            }
        )
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.get("/api/bots/{bot_id}/decisions")
def api_bot_decisions(bot_id: int, limit: int = 100):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    decisions = list_strategy_decisions(int(bot_id), limit=int(max(1, min(int(limit), 500))))
    return _json({"ok": True, "decisions": decisions})


@app.get("/api/strategies/leaderboard")
def api_strategies_leaderboard(window_days: int = 90):
    """Strategy performance leaderboard: Sharpe, win rate, drawdown."""
    rows = get_strategy_leaderboard(window_days=int(min(365, max(7, window_days))))
    return _json({"ok": True, "strategies": rows, "window_days": window_days})


@app.get("/api/recommendations/scan_status")
def api_recommendations_scan_status():
    """Return current scan progress for short/long horizons. Used during Rescan."""
    with _globals_lock:
        short_state = (_RECO_STATE.get("short") or {}).copy()
        long_state = (_RECO_STATE.get("long") or {}).copy()
    return _json({
        "ok": True,
        "short": {"scanned": short_state.get("scanned", 0), "total": short_state.get("total", 0), "scanning": short_state.get("scanning", False)},
        "long": {"scanned": long_state.get("scanned", 0), "total": long_state.get("total", 0), "scanning": long_state.get("scanning", False)},
    })


@app.get("/api/recommendations")
def api_recommendations(
    horizon: str = "short",
    min_score: float = 0.0,
    include: str = "",
    exclude: str = "",
    quote: str = "",
    market_type: str = "crypto", # "crypto" or "stocks" or "all"
    include_all: int = 0,
    limit: int = 10,  # Default to 10 for fast loading
    signal: str = "buy",  # "buy" | "watch" | "all", default "buy"
    sort: str = "score",  # "score" | "profit_factor" | "drawdown" | "winrate", default "score"
    offset: int = 0,  # Pagination offset
    volatility: str = "all",  # "all" | "low" | "medium" | "high" - filter by volatility level
    regime: str = "all",  # "all" | "bull" | "breakout" | "range" | "bear" - filter by regime
    sector: str = "all",  # "all" | "Technology" | "Financial" | etc. - stocks only
):
    """
    Get market recommendations. Returns structured status for better UX.
    FAST: No network calls, uses cached DB data only. Prices filled async.
    Default: Shows top 10 buy signals, sorted by score.
    
    Volatility filter:
    - low: volatility < 0.02 (2%)
    - medium: 0.02 <= volatility < 0.05 (2-5%)
    - high: volatility >= 0.05 (5%+)
    
    Regime filter: BULL, BREAKOUT, RANGE, BEAR (from metrics.regime).
    """
    h = "long" if str(horizon).lower().startswith("l") else "short"
    if market_type == "crypto" and not quote:
        quote = "USD"
    
    # Enforce reasonable limit to prevent slow responses
    limit = min(int(limit), 100)  # Cap at 100 for pagination support
    offset = max(0, int(offset))  # Ensure non-negative
    signal_filter = str(signal).lower()  # "buy", "watch", or "all"
    sort_by = str(sort).lower()  # "score", "profit_factor", "drawdown", "winrate"
    volatility_filter = str(volatility).lower()  # "all", "low", "medium", "high"
    regime_filter = str(regime).lower().strip()  # "all", "bull", "breakout", "range", "bear"
    sector_filter = str(sector).strip()  # "all" or sector name for stocks
    
    # Check client readiness
    kraken_ready = _kraken_ready()
    alpaca_ready = _alpaca_any_ready()
    
    # Early returns for specific market types
    if market_type == "crypto" and not kraken_ready:
        return _json({
            "ok": False,
            "status": "error",
            "reason": "kraken_not_ready",
            "message": KRAKEN_ERROR or "Kraken not ready",
            "items": []
        }, 503)
    
    if market_type == "stocks" and not alpaca_ready:
        return _json({
            "ok": False,
            "status": "error",
            "reason": "alpaca_not_ready",
            "message": "Alpaca API not configured",
            "items": []
        }, 503)
    
    # For "all" market type, if neither is ready, return empty but don't block
    if market_type == "all" and not kraken_ready and not alpaca_ready:
        return _json({
            "ok": True,
            "status": "ready",
            "reason": "no_clients",
            "message": "No trading clients configured",
            "items": [],
            "count": 0
        })
    
    # Fetch cached recommendations – over-fetch so Explore has enough per asset type.
    if market_type in ("crypto", "stocks"):
        fetch_limit = 1200
    else:
        fetch_limit = min(max(limit * 4, 200), 800)
    rows = []
    try:
        # Direct call - exclude blocklisted (STABLE, etc.) at source
        rows = list_recommendations(h, limit=fetch_limit, exclude_bases=list(CRYPTO_BLOCKLIST))
        if not rows:
            rows = []  # Ensure it's a list
    except Exception as e:
        logger.error(f"list_recommendations failed: {e}")
        rows = []
        # Don't return error - just return empty list with status
    
    with _globals_lock:
        state = (_RECO_STATE.get(h) or {}).copy()
    last_scan = state.get("last_run_ts", 0)
    scan_error = state.get("error", "")
    now = int(time.time())
    scan_age = now - last_scan if last_scan > 0 else 999999
    
    # If no rows, determine status
    if len(rows) == 0:
        if last_scan == 0:
            # Never scanned - trigger async scan and return warming_up
            status = "warming_up"
            reason = "no_scan_yet"
            message = "Generating recommendations... (first scan in progress)"
            import threading
            def _trigger_scan():
                try:
                    logger.info(f"Auto-triggering initial scan for {h} horizon")
                    _scan_recommendations(h)
                except Exception as e:
                    logger.error(f"Auto-trigger scan failed: {e}")
            threading.Thread(target=_trigger_scan, daemon=True).start()
        elif scan_age > 600:  # > 10 minutes old
            if scan_error:
                status = "error"
                reason = "scan_failed"
                message = f"Last scan failed: {scan_error}"
            else:
                status = "warming_up"
                reason = "scan_stale"
                message = "Recommendations are stale. Refreshing..."
                # Trigger refresh in background
                import threading
                def _trigger_refresh():
                    try:
                        logger.info(f"Auto-refreshing stale scan for {h} horizon")
                        _scan_recommendations(h)
                    except Exception as e:
                        logger.error(f"Auto-refresh failed: {e}")
                threading.Thread(target=_trigger_refresh, daemon=True).start()
        else:
            # Recently scanned but no results - might be filtering issue
            status = "ready"
            reason = "no_matches"
            message = "No recommendations match your filters"
    else:
        status = "ready"
        reason = "ok"
        message = "Recommendations loaded" 
    
    include_set = {s.strip().upper() for s in include.split(",") if s.strip()}
    exclude_set = {s.strip().upper() for s in exclude.split(",") if s.strip()}
    
    items = []
    eligible_count = 0
    
    # Process rows with error handling to avoid blocking on malformed data.
    # When filtering by market_type, process many more rows (crypto/stocks mixed in DB).
    max_process = min(len(rows), 800) if market_type != "all" else min(len(rows), limit)
    processed = 0
    start_time = time.time()
    max_process_time = 1.5  # Max 1.5 seconds for processing (reduced for speed)
    
    for r in rows:
        if processed >= max_process:
            break
        # Timeout guard - if processing takes too long, return what we have
        elapsed = time.time() - start_time
        if elapsed > max_process_time:
            logger.warning(f"Recommendations processing timeout after {elapsed:.1f}s, returning {processed} items")
            break
        try:
            sym = str(r.get("symbol") or "")
            if not sym:
                continue
            if "/" in sym and (sym.split("/")[0] or "").upper() in FIAT_BASES:
                continue
            if "/" in sym and (sym.split("/")[0] or "").upper() in CRYPTO_BLOCKLIST:
                continue
            # Fast JSON parse with fallback
            try:
                metrics = json.loads(r.get("metrics_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                metrics = {}
            
            # 1. Market Separation (normalize legacy "stock" -> "stocks")
            item_market = (metrics.get("market_type") or "").strip().lower()
            if not item_market:
                item_market = "stocks" if (len(sym) < 6 and "/" not in sym) else "crypto"
            if item_market == "stock":
                item_market = "stocks"

            if market_type != "all":
                want = market_type.lower()
                if want == "stocks" and item_market != "stocks":
                    continue
                if want == "crypto" and item_market != "crypto":
                    continue

            # For crypto, only allow USD-quoted pairs and exclude non-USD duplicates
            if item_market == "crypto":
                if "/" not in sym:
                    continue
                base, quote_ccy = (sym.split("/") + [""])[:2]
                if (quote_ccy or "").upper() != "USD":
                    continue
                if (base or "").upper() in CRYPTO_BLOCKLIST:
                    continue
                # Ensure symbol exists on Kraken (never recommend CC or invalid pairs)
                resolved, err = _validate_crypto_symbol(sym)
                if not resolved:
                    continue

            # 2. Filters
            if include_set and sym.upper() not in include_set:
                continue
            if exclude_set and sym.upper() in exclude_set:
                continue
            if quote and not sym.upper().endswith(f"/{quote.upper()}"):
                continue

            # 2b. Sector filter (stocks only)
            if sector_filter and sector_filter.lower() != "all" and item_market == "stocks":
                item_sector = (metrics.get("sector") or "").strip()
                if item_sector.lower() != sector_filter.lower():
                    continue

            # 3. Eligibility
            eligible = bool(metrics.get("eligible"))
            research_only = bool(metrics.get("research_only"))
            
            if eligible:
                eligible_count += 1
            
            # 4. Score Filter
            score = float(r.get("score") or 0.0)
            if not include_all:
                if (not eligible or research_only) and score < min_score:
                    continue
                if score < min_score:
                    continue
            
            # 5. Signal Filter (buy/watch/all)
            # Determine signal from score: per-market buy threshold, shared watch threshold
            buy_thresh = _reco_buy_threshold_stocks() if item_market == "stocks" else _reco_buy_threshold_crypto()
            item_signal = "buy" if score >= buy_thresh else ("watch" if score >= _reco_watch_threshold() else "wait")
            if signal_filter == "buy" and item_signal != "buy":
                continue  # Skip non-buy items
            elif signal_filter == "watch" and item_signal not in ("buy", "watch"):
                continue  # Skip wait/sell items
            # "all" includes everything, no filter

            # 5b. Volatility Filter (low/medium/high/all)
            # low: < 2%, medium: 2-5%, high: >= 5%. Use atr_pct if volatility/vol missing.
            if volatility_filter != "all":
                item_volatility = float(
                    metrics.get("volatility") or metrics.get("vol") or metrics.get("atr_pct") or 0.0
                )
                if volatility_filter == "low" and item_volatility >= 0.02:
                    continue
                elif volatility_filter == "medium" and (item_volatility < 0.02 or item_volatility >= 0.05):
                    continue
                elif volatility_filter == "high" and item_volatility < 0.05:
                    continue

            # 5c. Regime Filter (bull/breakout/range/bear/all)
            if regime_filter != "all":
                item_regime = (metrics.get("regime") or "").strip().upper()
                if not item_regime:
                    continue
                want = regime_filter.replace(" ", "").lower()
                ir = item_regime.upper()
                if want == "bull" and ir not in ("BULL",):
                    continue
                if want == "breakout" and ir not in ("BREAKOUT",):
                    continue
                if want == "range" and ir not in ("RANGE",):
                    continue
                if want == "bear" and ir not in ("BEAR", "HIGH_VOL_DEFENSIVE", "RISK_OFF"):
                    continue

            # 6. Populate Ticker Data (FAST - use metrics only, no network calls)
            # Prices will be filled via /api/prices endpoint async
            price_from_metrics = metrics.get("price")
            price = float(price_from_metrics) if price_from_metrics and price_from_metrics > 0 else None
            
            # Compute sort key based on sort parameter
            sort_key = score  # Default to score
            if sort_by == "profit_factor":
                sort_key = float(metrics.get("profit_factor") or metrics.get("expected_return") or score)
            elif sort_by == "drawdown":
                # Lower drawdown is better, so negate for descending sort
                cur_dd = float(metrics.get("cur_dd") or 0.0)
                sort_key = -cur_dd  # Negate so lower drawdown sorts higher
            elif sort_by == "winrate":
                sort_key = float(metrics.get("winrate") or metrics.get("win_rate") or score)
            # else: sort_by == "score", use score as sort_key

            # Sparkline - only use cache, don't fetch on-demand (too slow)
            sparkline = []
            try:
                c_key_1d = f"{sym}|1d|500"
                c_data = _RECO_OHLCV_CACHE.get(c_key_1d, {}).get("data")
                if not c_data:
                    c_key_4h = f"{sym}|4h|300"
                    c_data = _RECO_OHLCV_CACHE.get(c_key_4h, {}).get("data")
                if c_data and len(c_data) > 0:
                    # Get last 24-30 data points for 7d trend
                    sparkline = [float(c[4]) for c in c_data[-30:]]
            except Exception:
                pass

            # Rating buckets aligned with per-market buy threshold
            buy_thresh = _reco_buy_threshold_stocks() if item_market == "stocks" else _reco_buy_threshold_crypto()
            strong_buy_thresh = buy_thresh + 20
            watch_thresh = _reco_watch_threshold()
            if score >= strong_buy_thresh:
                rating = "Strong Buy"
            elif score >= buy_thresh:
                rating = "Buy"
            elif score >= watch_thresh:
                rating = "Watch"
            elif score > 20:
                rating = "Sell"
            else:
                rating = "Strong Sell"

            # Fast JSON parse with fallbacks (lazy - only parse if needed)
            regime = {}
            reasons = []
            risk_flags = []
            try:
                regime_raw = r.get("regime_json")
                if regime_raw:
                    regime = json.loads(regime_raw)
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                reasons_raw = r.get("reasons_json")
                if reasons_raw:
                    reasons = json.loads(reasons_raw)
            except (json.JSONDecodeError, TypeError):
                pass
            try:
                risk_flags_raw = r.get("risk_flags_json")
                if risk_flags_raw:
                    risk_flags = json.loads(risk_flags_raw)
            except (json.JSONDecodeError, TypeError):
                pass

            items.append(
                {
                    "symbol": sym,
                    "score": score,
                    "sort_key": sort_key,  # For sorting
                    "signal": item_signal,  # "buy", "watch", or "wait"
                    "horizon": h,
                    "market_type": item_market,
                    "price": price,  # From metrics only, filled async via /api/prices
                    "change_pct": None,  # Filled async via /api/prices
                    "volume": None,  # Filled async via /api/prices
                    "market_cap": None, 
                    "rating": rating,
                    "confidence": float(metrics.get("confidence_score") or 0.0),
                    "sparkline": sparkline,
                    "regime_1d": (regime.get("1d") or {}).get("label"),
                    "regime_4h": (regime.get("4h") or {}).get("label"),
                    "weekly_trend": metrics.get("weekly_trend"),
                    "strategy_mode": metrics.get("strategy") or metrics.get("recommended_strategy") or metrics.get("suggested_strategy") or "smart_dca",
                    "suggested_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy") or metrics.get("suggested_strategy")),
                    "recommended_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy") or metrics.get("suggested_strategy")),
                    "volatility": metrics.get("atr_pct"),
                    "risk_flags": risk_flags,
                    "updated_ts": int(r.get("created_ts") or 0),
                    "eligible": eligible,
                    "research_only": research_only or (item_market == "stocks" and not _alpaca_any_ready()),
                    "reasons": reasons,
                    "sector": metrics.get("sector") if item_market == "stocks" else None,
                    "benchmark_vs": metrics.get("benchmark_vs") or None,
                    "peer_rank": metrics.get("peer_rank") or None,
                    "beta": metrics.get("beta"),
                    "diversify_key": (metrics.get("sector") or "unknown") if item_market == "stocks" else "crypto",
                }
            )
            processed += 1
        except Exception:
            continue
            
    # Sort items by sort_key (defaults to score)
    items.sort(key=lambda x: float(x.get("sort_key", x.get("score", 0))), reverse=True)

    # Diversify top N by sector (stocks) / crypto so list isn't all one sector
    try:
        from explore_v2 import diversify_picks, is_enabled as explore_v2_enabled
        if explore_v2_enabled() and items and "diversify_key" in (items[0] or {}):
            need = min(len(items), offset + limit)
            items = diversify_picks(items, top_k=max(need, limit), cluster_key="diversify_key")
    except ImportError:
        pass

    # Apply pagination
    total_count = len(items)
    paginated_items = items[offset:offset + limit]  # Apply offset and limit
    
    # If no items and include_all is set, try to return at least some data with lower threshold
    if len(items) == 0 and include_all:
        # Re-scan with lower score threshold (but still no network calls)
        for r in rows[:min(100, len(rows))]:  # Check limited rows for speed
            try:
                sym = str(r.get("symbol") or "")
                if not sym:
                    continue
                if "/" in sym and (sym.split("/")[0] or "").upper() in FIAT_BASES:
                    continue
                try:
                    metrics = json.loads(r.get("metrics_json") or "{}")
                except (json.JSONDecodeError, TypeError):
                    metrics = {}
                item_market = (metrics.get("market_type") or "").strip().lower() or None
                if not item_market:
                    item_market = "stocks" if (len(sym) < 6 and "/" not in sym) else "crypto"
                if item_market == "stock":
                    item_market = "stocks"
                want_mt = market_type.lower() if market_type else "all"
                if want_mt != "all" and (
                    (want_mt == "stocks" and item_market != "stocks") or (want_mt == "crypto" and item_market != "crypto")
                ):
                    continue
                score = float(r.get("score") or 0.0)
                if score < -50:
                    continue
                price_from_metrics = metrics.get("price")
                items.append({
                    "symbol": sym,
                    "score": score,
                    "horizon": h,
                    "market_type": item_market,
                    "price": float(price_from_metrics) if price_from_metrics and price_from_metrics > 0 else None,
                    "change_pct": None,  # Filled async
                    "volume": None,  # Filled async
                    "market_cap": None,
                    "rating": "Neutral",
                    "confidence": float(metrics.get("confidence_score") or 0.0),
                    "sparkline": [],
                    "regime_1d": None,
                    "regime_4h": None,
                    "weekly_trend": None,
                    "strategy_mode": metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca",
                    "suggested_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
                    "recommended_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
                    "volatility": metrics.get("atr_pct"),
                    "risk_flags": [],
                    "updated_ts": int(r.get("created_ts") or 0),
                    "eligible": bool(metrics.get("eligible")),
                    "research_only": bool(metrics.get("research_only")) or (item_market == "stocks" and not _alpaca_any_ready()),
                    "reasons": [],
                })
            except Exception:
                continue
        items.sort(key=lambda x: x["score"], reverse=True)
        items = items[:limit]
        total_count = len(items)
        paginated_items = items[offset:offset + limit]
    
    return _json({
        "ok": status == "ready",
        "status": status,
        "reason": reason,
        "message": message,
        "items": paginated_items,
        "count": len(paginated_items),
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + len(paginated_items)) < total_count,
        "scan_age_sec": scan_age if last_scan > 0 else None,
        "last_scan_ts": last_scan if last_scan > 0 else None
    })


@app.post("/api/recommendations/calibrate")
def api_recommendations_calibrate(window_days: int = 30):
    """Run adaptive scoring calibration from closed recommendation outcomes."""
    try:
        from recommendation_validator import run_calibration
        result = run_calibration(window_days=int(max(7, min(90, window_days))))
        return _json(result)
    except Exception as e:
        logger.exception("Calibration failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/ml/regime/train")
def api_ml_regime_train():
    """Train ML regime detector on historical data (Phase 2 Advanced)."""
    try:
        from ml_regime_detector import train_regime_detector_on_historical_data
        train_regime_detector_on_historical_data()
        return _json({"ok": True, "message": "ML regime detector trained"})
    except Exception as e:
        logger.exception("ML regime train failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/insider/fetch/{symbol}")
def api_insider_fetch(symbol: str, days_back: int = 90):
    """Fetch and store SEC Form 4 insider transactions for a stock (requires FINNHUB_API_KEY)."""
    sym = symbol.upper().split("/")[0]
    if sym in ("BTC", "ETH"):
        return _json({"ok": False, "error": "Insider data is for stocks only"}, 400)
    try:
        from insider_tracker import fetch_and_store_insider_transactions
        n = fetch_and_store_insider_transactions(sym, days_back=int(min(365, max(7, days_back))))
        return _json({"ok": True, "symbol": sym, "new_records": n})
    except Exception as e:
        logger.exception("Insider fetch failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/ml/performance")
def api_ml_performance(days: int = 30, symbol: Optional[str] = None):
    """ML model performance: accuracy by symbol, timeframe, precision, recall, F1."""
    try:
        from db import get_ml_model_accuracy, get_ml_predictions
        stats = get_ml_model_accuracy(days_back=int(min(365, max(7, days))))
        preds = get_ml_predictions(symbol=symbol, limit=500, days_back=days)
        by_symbol = {}
        for p in preds:
            sym = p.get("symbol", "?")
            if sym not in by_symbol:
                by_symbol[sym] = {"correct": 0, "total": 0}
            if p.get("actual_outcome_7d") is not None:
                pred_up = str(p.get("predicted_direction", "")).upper() == "UP"
                actual_up = float(p.get("actual_outcome_7d", 0)) > 0
                by_symbol[sym]["total"] += 1
                if pred_up == actual_up:
                    by_symbol[sym]["correct"] += 1
        by_symbol = {k: {"accuracy": v["correct"] / v["total"] if v["total"] > 0 else 0.5, "total": v["total"]} for k, v in by_symbol.items()}
        try:
            from ml_ensemble import get_ml_ensemble
            ensemble = get_ml_ensemble()
            status = ensemble.get_status()
            feature_imp = ensemble.get_feature_importance()
        except Exception:
            status = {}
            feature_imp = {}
        return _json({
            "ok": True,
            "days": days,
            "accuracy": round(stats["accuracy"], 4),
            "precision": round(stats["precision"], 4),
            "recall": round(stats["recall"], 4),
            "f1": round(stats["f1"], 4),
            "total_predictions": stats["total"],
            "by_symbol": by_symbol,
            "ensemble_status": status,
            "feature_importance": feature_imp,
        })
    except Exception as e:
        logger.exception("ML performance failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/ml/predictions")
def api_ml_predictions(symbol: Optional[str] = None, limit: int = 50, days_back: int = 30):
    """List recent ML predictions."""
    try:
        from db import get_ml_predictions
        rows = get_ml_predictions(symbol=symbol, limit=int(min(200, limit)), days_back=days_back)
        return _json({"ok": True, "predictions": rows, "count": len(rows)})
    except Exception as e:
        logger.exception("ML predictions list failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/ml/retrain")
def api_ml_retrain():
    """Trigger ML model retraining (walk-forward, deploy only if validation >60%)."""
    try:
        from ml_ensemble import get_ml_ensemble
        from db import save_ml_model_version
        import os
        ensemble = get_ml_ensemble()
        if len(ensemble._training_data) < 100:
            return _json({"ok": False, "error": "Insufficient training data (need 100+ samples)"}, 400)
        success = ensemble.train(force=True)
        if not success:
            return _json({"ok": False, "error": "Training failed"}, 500)
        min_acc = float(os.getenv("ML_MIN_ACCURACY", "0.60"))
        best_acc = max(
            ensemble._model_performance.get("xgb", type("O", (), {"recent_accuracy": 0})()).recent_accuracy,
            ensemble._model_performance.get("rf", type("O", (), {"recent_accuracy": 0})()).recent_accuracy,
        )
        deployed = best_acc >= min_acc
        version = f"v{int(time.time())}"
        save_ml_model_version("ensemble", version, best_acc, deployed=deployed)
        return _json({"ok": True, "validation_accuracy": best_acc, "deployed": deployed, "version": version})
    except Exception as e:
        logger.exception("ML retrain failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/portfolio/capital")
def api_portfolio_capital():
    """Portfolio-level capital management: total, reserve, allocation, heat map, CAGR, leverage."""
    try:
        from db import list_bots, list_all_deals, get_bot_recent_streak, bot_performance_stats, bot_deal_stats, all_deal_stats
        from portfolio_manager import (
            compute_cash_reserve,
            get_portfolio_heat_map_data,
            get_portfolio_cagr,
            check_leverage,
        )
        from capital_allocator import get_allocation_mult, AUTO_SCALE_ENABLED
        bots = list_bots()
        deals = list_all_deals(state="OPEN", limit=500)
        portfolio_total = 0.0
        try:
            bm_ref = globals().get("bm")
            if bm_ref and hasattr(bm_ref, "get_portfolio_total"):
                portfolio_total = float(bm_ref.get_portfolio_total())
        except Exception:
            pass
        if portfolio_total <= 0:
            try:
                all_stats = all_deal_stats()
                realized = float(all_stats.get("realized_total", 0))
                portfolio_total = max(1000.0, 1000.0 + realized)
            except Exception:
                portfolio_total = 1000.0
        reserve = compute_cash_reserve(portfolio_total)
        heat_map = get_portfolio_heat_map_data(bots, deals, portfolio_total)
        cagr = get_portfolio_cagr(365)
        leverage_info = check_leverage(portfolio_total, 0)
        per_bot = []
        for b in bots:
            bid = b.get("id")
            if not bid:
                continue
            streak = get_bot_recent_streak(bid, 5)
            perf = bot_performance_stats(bid)
            stats = bot_deal_stats(bid)
            mult = get_allocation_mult(streak, float(perf.get("win_rate", 0.5)), float(stats.get("realized_total", 0)))
            per_bot.append({
                "bot_id": bid,
                "symbol": b.get("symbol"),
                "streak": streak,
                "allocation_mult": mult,
                "realized_total": stats.get("realized_total"),
            })
        return _json({
            "ok": True,
            "portfolio_total": round(portfolio_total, 2),
            "cash_reserve": round(reserve, 2),
            "reserve_pct": round(reserve / portfolio_total * 100, 1) if portfolio_total > 0 else 20,
            "heat_map": heat_map,
            "cagr_1y": round(cagr * 100, 2) if cagr is not None else None,
            "leverage": leverage_info,
            "per_bot_allocation": per_bot,
            "auto_scale_enabled": AUTO_SCALE_ENABLED,
        })
    except Exception as e:
        logger.exception("Portfolio capital API failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/execution/quality")
def api_execution_quality(days: int = 30):
    """Return execution quality stats: avg slippage by symbol and by strategy."""
    try:
        from execution_quality_tracker import get_avg_slippage_by_symbol, get_avg_slippage_by_strategy
        by_symbol = get_avg_slippage_by_symbol(days=int(days))
        by_strategy = get_avg_slippage_by_strategy(days=int(days))
        return _json({
            "ok": True,
            "days": int(days),
            "by_symbol": by_symbol,
            "by_strategy": by_strategy,
        })
    except Exception as e:
        logger.exception("Execution quality API failed: %s", e)
        return _json({"ok": False, "error": str(e)}, 500)


@app.get("/api/recommendations/performance")
def api_recommendations_performance(days: int = 30):
    """Return recommendation accuracy metrics: win rate, avg profit, performance by score range, regime, and alpha by strategy."""
    stats = get_recommendation_performance_stats(days=int(max(7, min(365, days))))
    alpha_by_strategy = {}
    try:
        from benchmark_analyzer import get_alpha_by_strategy
        alpha_by_strategy = get_alpha_by_strategy(days=int(days))
    except Exception:
        pass
    return _json({
        "ok": True,
        "days": int(days),
        "total_closed": stats["total_closed"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "win_rate_pct": round(stats["win_rate"], 1),
        "avg_profit_per_recommendation": stats["avg_profit_per_recommendation"],
        "by_score_range": stats["by_score_range"],
        "by_regime": stats["by_regime"],
        "alpha_by_strategy": alpha_by_strategy,
    })


@app.get("/api/recommendations/{symbol}")
def api_recommendation_symbol(symbol: str, horizon: str = "short"):
    """Get recommendation for a specific symbol. Returns single item."""
    from urllib.parse import unquote
    sym = unquote(symbol)
    if "/" in sym and (sym.split("/")[0] or "").upper() in FIAT_BASES:
        return _json({"ok": False, "error": "Fiat FX pairs are excluded"}, 404)
    if "/" in sym and (sym.split("/")[0] or "").upper() in CRYPTO_BLOCKLIST:
        return _json({"ok": False, "error": "Symbol is blocklisted"}, 404)
    h = "long" if str(horizon).lower().startswith("l") else "short"
    row = get_recommendation(_resolve_symbol(sym), h)
    if not row:
        return _json({"ok": False, "error": "No recommendation found"}, 404)
    
    metrics = json.loads(row.get("metrics_json") or "{}")
    regime = json.loads(row.get("regime_json") or "{}")
    reasons = json.loads(row.get("reasons_json") or "[]")
    risk_flags = json.loads(row.get("risk_flags_json") or "[]")
    
    item_market = (metrics.get("market_type") or "").strip().lower()
    if not item_market:
        item_market = "stocks" if (len(sym) < 6 and "/" not in sym) else "crypto"
    if item_market == "stock":
        item_market = "stocks"
    
    ticker = {}
    price_from_metrics = metrics.get("price")
    if price_from_metrics and price_from_metrics > 0:
        ticker = {"last": price_from_metrics}
    else:
        if item_market == "stocks":
            ticker = _ticker_cached(sym, ttl_sec=120) or {}
        else:
            ticker = _ticker_cached(sym, ttl_sec=120) or {}
    
    score = float(row.get("score") or 0.0)
    if score >= 80: rating = "Strong Buy"
    elif score >= 60: rating = "Buy"
    elif score >= 40: rating = "Watch"
    elif score > 20: rating = "Sell"
    else: rating = "Strong Sell"
    
    return _json({
        "ok": True,
        "item": {
            "symbol": sym,
            "score": score,
            "horizon": h,
            "market_type": item_market,
            "price": float(metrics.get("price") or ticker.get("last") or 0.0) if (metrics.get("price") or ticker.get("last")) else None,
            "change_pct": float(ticker.get("percentage") or 0.0),
            "volume": float(ticker.get("quoteVolume") or 0.0),
            "rating": rating,
            "confidence": float(metrics.get("confidence_score") or 0.0),
            "regime_1d": (regime.get("1d") or {}).get("label"),
            "regime_4h": (regime.get("4h") or {}).get("label"),
            "weekly_trend": metrics.get("weekly_trend"),
            "strategy_mode": metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca",
            "suggested_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
            "recommended_strategy": _strategy_display_name(metrics.get("strategy") or metrics.get("recommended_strategy")),
            "volatility": metrics.get("atr_pct"),
            "risk_flags": risk_flags,
            "updated_ts": int(row.get("created_ts") or 0),
            "eligible": bool(metrics.get("eligible")),
            "research_only": bool(metrics.get("research_only")) or (item_market == "stocks" and not _alpaca_any_ready()),
            "reasons": reasons,
            "benchmark_vs": metrics.get("benchmark_vs") or None,
            "peer_rank": metrics.get("peer_rank") or None,
            "beta": metrics.get("beta"),
        }
    })


@app.get("/api/diagnostics")
def api_diagnostics():
    """Get system diagnostics: client status, scan state, recommendation counts."""
    now = int(time.time())
    uptime_sec = int(time.time() - _APP_START_TIME)
    
    # Kraken status
    kraken_ready = _kraken_ready()
    kraken_error = KRAKEN_ERROR or None
    
    # Alpaca status
    alpaca_ready = _alpaca_any_ready()
    alpaca_error = None
    if not alpaca_ready:
        if not os.getenv("ALPACA_API_KEY_PAPER") and not os.getenv("ALPACA_API_KEY_LIVE"):
            alpaca_error = "No Alpaca API keys configured"
        else:
            alpaca_error = "Alpaca client initialization failed"
    
    with _globals_lock:
        short_state = (_RECO_STATE.get("short") or {}).copy()
        long_state = (_RECO_STATE.get("long") or {}).copy()

    # Count recommendations in DB
    short_count = len(list_recommendations("short", limit=1000))
    long_count = len(list_recommendations("long", limit=1000))
    
    return _json({
        "ok": True,
        "uptime_sec": uptime_sec,
        "kraken": {
            "ready": kraken_ready,
            "error": kraken_error
        },
        "alpaca": {
            "ready": alpaca_ready,
            "error": alpaca_error
        },
        "recommendations": {
            "short": {
                "count": short_count,
                "last_scan_ts": short_state.get("last_run_ts"),
                "last_scan_age_sec": now - short_state.get("last_run_ts") if short_state.get("last_run_ts") else None,
                "last_error": short_state.get("error"),
                "scanned": short_state.get("scanned", 0),
                "eligible": short_state.get("eligible", 0)
            },
            "long": {
                "count": long_count,
                "last_scan_ts": long_state.get("last_run_ts"),
                "last_scan_age_sec": now - long_state.get("last_run_ts") if long_state.get("last_run_ts") else None,
                "last_error": long_state.get("error"),
                "scanned": long_state.get("scanned", 0),
                "eligible": long_state.get("eligible", 0)
            }
        },
        "timestamp": now
    })


@app.get("/api/logs")
def api_logs(lines: int = 200, service: str = "tradingserver"):
    """Return last N lines of service logs (journalctl). For diagnostics."""
    import subprocess
    try:
        out = subprocess.run(
            ["journalctl", "-u", service, f"-n{min(lines, 2000)}", "--no-pager", "-o", "short-iso"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout:
            return JSONResponse(
                content={"ok": True, "service": service, "lines": out.stdout.strip().split("\n")[-min(lines, 2000):]},
                headers={"Cache-Control": "no-store"},
            )
        return _json({"ok": False, "error": out.stderr or "journalctl failed"}, 500)
    except FileNotFoundError:
        return _json({"ok": False, "error": "journalctl not available"}, 501)
    except subprocess.TimeoutExpired:
        return _json({"ok": False, "error": "journalctl timed out"}, 504)
    except Exception as e:
        logger.exception("api_logs failed")
        return _json({"ok": False, "error": str(e)[:200]}, 500)


@app.post("/api/recommendations/{symbol}/create_bot")
async def api_recommendation_create_bot(symbol: str, request: Request):
    payload = await request.json()
    horizon = str(payload.get("horizon") or "short")
    if "/" in symbol and (symbol.split("/")[0] or "").upper() in CRYPTO_BLOCKLIST:
        return _json({"ok": False, "error": "Symbol is blocklisted"}, 400)
    row = get_recommendation(_resolve_symbol(symbol), "long" if horizon.startswith("l") else "short")
    if not row:
        return _json({"ok": False, "error": "No recommendation found"}, 404)
    metrics = json.loads(row.get("metrics_json") or "{}")
    strategy = str(metrics.get("strategy") or metrics.get("recommended_strategy") or "smart_dca")
    # Derive market_type from recommendation so stock bots use Alpaca, crypto use Kraken
    item_market = (metrics.get("market_type") or "").strip().lower()
    if item_market == "stock":
        item_market = "stocks"
    if not item_market:
        item_market = "stocks" if (len(str(symbol)) < 6 and "/" not in str(symbol)) else "crypto"
    name = str(payload.get("name") or f"Reco {symbol} {horizon.upper()}")
    enabled = int(bool(payload.get("enabled", False)))
    dry_run = int(bool(payload.get("dry_run", True)))
    auto_restart = int(bool(payload.get("auto_restart", False)))
    start_now = bool(payload.get("start_now", False))

    sym_resolved = str(row.get("symbol") or symbol)
    bot_id = create_bot(
        {
            "name": name,
            "symbol": sym_resolved,
            "enabled": enabled,
            "dry_run": dry_run,
            "strategy_mode": strategy,
            "forced_strategy": "",
            "auto_restart": auto_restart,
            "market_type": item_market,
            "alpaca_mode": str(payload.get("alpaca_mode") or "paper"),
            "base_quote": float(payload.get("base_quote") or 25.0),
            "safety_quote": float(payload.get("safety_quote") or 25.0),
            "max_safety": int(payload.get("max_safety") or 3),
            "first_dev": float(payload.get("first_dev") or 0.015),
            "step_mult": float(payload.get("step_mult") or 1.2),
            "tp": float(payload.get("tp") or 0.012),
            "trend_filter": int(payload.get("trend_filter") or 0),
            "trend_sma": int(payload.get("trend_sma") or 200),
            "max_spend_quote": float(payload.get("max_spend_quote") or 250.0),
            "poll_seconds": int(payload.get("poll_seconds") or 10),
            "max_open_orders": int(payload.get("max_open_orders") or 6),
            "max_total_exposure_pct": float(payload.get("max_total_exposure_pct") or 0.50),
            "per_symbol_exposure_pct": float(payload.get("per_symbol_exposure_pct") or (0.05 if horizon.startswith("l") else 0.1)),
            "min_free_cash_pct": float(payload.get("min_free_cash_pct") or 0.2),
            "max_concurrent_deals": int(payload.get("max_concurrent_deals") or 4),
            "spread_guard_pct": float(payload.get("spread_guard_pct") or 0.004),
            "limit_timeout_sec": int(payload.get("limit_timeout_sec") or 8),
            "daily_loss_limit_pct": float(payload.get("daily_loss_limit_pct") or 0.05),
            "pause_hours": int(payload.get("pause_hours") or 6),
        }
    )

    # Link recommendation to bot for performance tracking
    try:
        regime_obj = json.loads(row.get("regime_json") or "{}")
        regime_name = str(regime_obj.get("regime") or regime_obj.get("name") or metrics.get("regime") or "")
        link_recommendation_to_bot(
            bot_id=int(bot_id),
            symbol=sym_resolved,
            recommendation_date=int(row.get("created_ts") or 0),
            score_at_recommendation=float(row.get("score") or 0),
            regime_at_recommendation=regime_name,
            metrics_json=row.get("metrics_json") or "{}",
            reasons_json=row.get("reasons_json") or "[]",
            snapshot_id=int(row["id"]) if row.get("id") else None,
        )
    except Exception as e:
        logger.warning("link_recommendation_to_bot failed for bot_id=%s: %s", bot_id, e)

    if start_now and bm is not None:
        try:
            bot = get_bot(int(bot_id))
            ok, reason = _can_start_bot_live(bot or {})
            if ok:
                bm.start(int(bot_id))
            else:
                logger.warning(
                    "api_recommendation_create_bot: start_now blocked bot_id=%s reason=%s",
                    bot_id,
                    reason,
                )
        except Exception:
            logger.exception("api_recommendation_create_bot: start_now failed bot_id=%s", bot_id)

    return _json({"ok": True, "bot_id": int(bot_id)})


@app.post("/api/recommendations/scan")
def api_recommendations_scan(horizon: str = "short"):
    """Trigger a manual scan of recommendations. Returns immediately, scan runs in background."""
    import threading
    h = "long" if str(horizon).lower().startswith("l") else "short"

    def _scan_async():
        try:
            n = delete_recommendations_for_blocklist(list(CRYPTO_BLOCKLIST))
            if n > 0:
                logger.info("Purged %d blocklisted recommendation(s) before scan", n)
            logger.info(f"Starting recommendation scan for {h} horizon")
            _scan_recommendations(h)
            with _globals_lock:
                state = (_RECO_STATE.get(h) or {}).copy()
            logger.info(f"Scan completed: {state.get('scanned', 0)} scanned, {state.get('eligible', 0)} eligible")
        except Exception as e:
            logger.error(f"Background scan failed: {e}", exc_info=True)
    
    threading.Thread(target=_scan_async, daemon=True).start()
    return _json({"ok": True, "message": f"Scan triggered for {h} horizon"})


def _try_init_bot_manager() -> bool:
    """Lazy-init BotManager if bm is None and we have at least one client. Returns True if bm is now ready."""
    global bm
    if bm is not None:
        return True
    if not (kc or alpaca_paper or alpaca_live):
        return False
    try:
        with _globals_lock:
            if bm is not None:
                return True
            bm = BotManager(kc, alpaca_paper, alpaca_live)
            logger.info("BotManager lazy-init OK (Crypto: %s, Alpaca paper: %s, Alpaca live: %s)",
                       KRAKEN_READY, ALPACA_PAPER_READY, ALPACA_LIVE_READY)
            return True
    except Exception as e:
        logger.warning("BotManager lazy-init failed: %s", e)
        return False


@app.post("/api/bots/{bot_id}/start")
def api_bot_start(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    try:
        from circuit_breaker import is_emergency_stop_active, is_bot_circuit_open, get_bot_pause_until
        if is_emergency_stop_active():
            return _json({"ok": False, "error": "Emergency stop active. Exchange errors persist. Check /api/health."}, 503)
        if is_bot_circuit_open(int(bot_id)):
            until = get_bot_pause_until(int(bot_id))
            return _json({"ok": False, "error": f"Circuit breaker: bot paused until errors clear (until ts {until})"}, 503)
    except ImportError:
        pass
    try:
        from data_validator import is_data_quality_degraded
        if is_data_quality_degraded():
            return _json({"ok": False, "error": "Data quality degraded (5+ issues in 15 min). Trading paused."}, 503)
    except ImportError:
        pass
    if bm is None:
        if _try_init_bot_manager():
            pass  # bm now set, continue
        else:
            return _json({"ok": False, "error": "Worker not initialized. Check Kraken/Alpaca API keys in .env and restart."}, 503)

    ok, reason = _can_start_bot_live(b)
    if not ok:
        logger.warning("api_bot_start blocked bot_id=%s reason=%s", bot_id, reason)
        return _json({"ok": False, "error": reason}, 503)

    msg = bm.start(int(bot_id))
    snap = bm.snapshot(int(bot_id))
    return _json({"ok": True, "message": msg, "snap": snap})


@app.post("/api/bots/{bot_id}/stop")
def api_bot_stop(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if bm is None:
        if not _try_init_bot_manager():
            return _json({"ok": False, "error": "Worker not initialized. Check Kraken/Alpaca API keys in .env and restart."}, 503)

    msg = bm.stop(int(bot_id))
    snap = bm.snapshot(int(bot_id))
    return _json({"ok": True, "message": msg, "snap": snap})


# NOTE: _get_bot_client is defined once at line ~1410 with full AlpacaAdapter support


@app.get("/api/bots/{bot_id}/orders")
def api_bot_orders(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
        
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available", "orders": []}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "orders": []}, 503)
        
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Manual orders are disabled in dry run mode.", "orders": []}, 400)

    symbol = _resolve_symbol(b.get("symbol", ""))
    
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}", "orders": []}, 400)

    try:
        orders = client.fetch_open_orders(symbol)
        return _json({"ok": True, "orders": [_serialize_order(o) for o in (orders or [])]})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "orders": []}, 500)


@app.post("/api/bots/{bot_id}/orders")
async def api_bot_order_create(bot_id: int, request: Request):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
        
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Manual orders are disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading is disabled. Set ALLOW_LIVE_TRADING=1 in .env to place real orders."}, 403)
    block = _check_trading_allowed(bot_id=int(bot_id))
    if block:
        return _json(block, 503)

    payload = await request.json()
    action = str(payload.get("action") or "").strip().lower()
    size_quote = payload.get("size_quote")
    size_base = payload.get("size_base")
    price = payload.get("price")

    symbol = _resolve_symbol(b.get("symbol", ""))
    
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    try:
        order = None
        side = ""
        ord_type = ""
        if action == "market_buy":
            side = "buy"
            ord_type = "market"
            q = float(size_quote or 0)
            if q <= 0:
                return _json({"ok": False, "error": "size_quote must be > 0 for market buy"}, 400)
            order = client.create_market_buy_quote(symbol, q)
        elif action == "market_sell":
            side = "sell"
            ord_type = "market"
            amt = float(size_base or 0)
            if amt <= 0:
                return _json({"ok": False, "error": "size_base must be > 0 for market sell"}, 400)
            order = client.create_market_sell_base(symbol, amt)
        elif action == "limit_buy":
            side = "buy"
            ord_type = "limit"
            amt = float(size_base or 0)
            px = float(price or 0)
            if amt <= 0 or px <= 0:
                return _json({"ok": False, "error": "size_base and price must be > 0 for limit buy"}, 400)
            order = client.create_limit_buy_base(symbol, amt, px)
        elif action == "limit_sell":
            side = "sell"
            ord_type = "limit"
            amt = float(size_base or 0)
            px = float(price or 0)
            if amt <= 0 or px <= 0:
                return _json({"ok": False, "error": "size_base and price must be > 0 for limit sell"}, 400)
            order = client.create_limit_sell_base(symbol, amt, px)
        else:
            return _json({"ok": False, "error": "Invalid action"}, 400)

        price_val = None
        amount_val = None
        try:
            if price is not None and float(price) > 0:
                price_val = float(price)
        except Exception:
            price_val = None
        try:
            if size_base is not None and float(size_base) > 0:
                amount_val = float(size_base)
        except Exception:
            amount_val = None

        add_order_event(
            bot_id=int(bot_id),
            symbol=symbol,
            side=side,
            ord_type=ord_type,
            price=price_val,
            amount=amount_val,
            order_id=str(order.get("id")) if isinstance(order, dict) else None,
            tag="manual",
            status="submitted",
            reason="manual",
            is_live=0 if bool(b.get("dry_run", 1)) else 1,
        )
        add_log(int(bot_id), "INFO", f"Manual order submitted ({action}).", "ORDER")
        return _json({"ok": True, "order": order})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.post("/api/bots/{bot_id}/close_position")
async def api_bot_close_position(bot_id: int):
    """One-click close: sell full position and close the deal."""
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    if not bool(b.get("dry_run", 1)):
        block = _check_trading_allowed(bot_id=int(bot_id))
        if block:
            return _json(block, 503)
    client, is_kraken = _get_bot_client(b)
    if not client:
        return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Close position is disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading disabled. Set ALLOW_LIVE_TRADING=1."}, 403)

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found: {symbol}"}, 400)

    try:
        snap = bm.snapshot(int(bot_id)) if bm else {}
        base_pos = float(snap.get("base_pos") or 0)
        if base_pos <= 0:
            return _json({"ok": False, "error": "No open position to close"}, 400)

        order = client.create_market_sell_base(symbol, base_pos)
        add_log(int(bot_id), "INFO", f"Quick close: sold {base_pos} {symbol}", "ORDER")

        od = latest_open_deal(int(bot_id))
        if od:
            deal_id = int(od["id"])
            deal_opened = int(od.get("opened_at") or 0)
            from db import close_deal
            entry_avg = float(od.get("entry_avg") or snap.get("avg_entry") or 0)
            exit_avg = float(snap.get("last_price") or entry_avg)
            close_deal(
                deal_id,
                entry_avg=entry_avg,
                exit_avg=exit_avg,
                base_amount=base_pos,
                realized_pnl_quote=float((exit_avg - entry_avg) * base_pos) if entry_avg > 0 else 0.0,
                hold_sec=int(time.time()) - deal_opened,
                exit_strategy="manual_close",
            )
            add_log(int(bot_id), "INFO", f"Deal {deal_id} closed (manual).", "SYSTEM")

        return _json({"ok": True, "message": f"Sold {base_pos} {symbol}", "order": order})
    except Exception as e:
        logger.exception("close_position failed bot_id=%s: %s", bot_id, e)
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.delete("/api/bots/{bot_id}/orders/{order_id}")
def api_bot_order_cancel(bot_id: int, order_id: str):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
        
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Manual orders are disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading is disabled. Set ALLOW_LIVE_TRADING=1 in .env to cancel real orders."}, 403)

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    try:
        res = client.cancel_order(str(order_id), symbol)
        add_order_event(
            bot_id=int(bot_id),
            symbol=symbol,
            side="",
            ord_type="cancel",
            price=None,
            amount=None,
            order_id=str(order_id),
            tag="manual",
            status="cancelled",
            reason="manual",
            is_live=0 if bool(b.get("dry_run", 1)) else 1,
        )
        add_log(int(bot_id), "INFO", f"Manual order cancelled ({order_id}).", "ORDER")
        return _json({"ok": True, "result": res})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


@app.post("/api/bots/{bot_id}/orders/cancel_all")
def api_bot_order_cancel_all(bot_id: int):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available"}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready"}, 503)
        
    if bool(b.get("dry_run", 1)):
        return _json({"ok": False, "error": "Manual orders are disabled in dry run mode."}, 400)
    if not ALLOW_LIVE_TRADING:
        return _json({"ok": False, "error": "Live trading is disabled. Set ALLOW_LIVE_TRADING=1 in .env to cancel real orders."}, 403)

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}"}, 400)

    try:
        client.cancel_all_open_orders(symbol)
        add_order_event(
            bot_id=int(bot_id),
            symbol=symbol,
            side="",
            ord_type="cancel_all",
            price=None,
            amount=None,
            order_id=None,
            tag="manual",
            status="cancelled",
            reason="manual",
            is_live=0 if bool(b.get("dry_run", 1)) else 1,
        )
        add_log(int(bot_id), "INFO", "Manual cancel-all submitted.", "ORDER")
        return _json({"ok": True})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)


def _bot_live_degraded(b: Dict[str, Any], bot_id: int, logs_limit: int, deals_limit: int, last_event: str, price_error: Optional[str] = None) -> Dict[str, Any]:
    """Return a successful live payload with degraded snap so the UI loads instead of showing 'Live refresh failed'."""
    sym = b.get("symbol", "")
    snap = {
        "running": False,
        "last_event": last_event,
        "last_price": None,
        "avg_entry": None,
        "base_pos": None,
    }
    return {
        "ok": True,
        "bot": b,
        "snap": snap,
        "logs": list_logs(int(bot_id), limit=int(max(1, min(int(logs_limit), 2000)))),
        "deals": list_deals(int(bot_id), limit=int(max(1, min(int(deals_limit), 1000)))),
        "market_type": classify_symbol(sym),
        "kraken_ready": _kraken_ready(),
        "kraken_error": KRAKEN_ERROR,
        "alpaca_paper_ready": ALPACA_PAPER_READY,
        "alpaca_live_ready": ALPACA_LIVE_READY,
        "alpaca_error": ALPACA_ERROR or "",
        "price_error": price_error or last_event,
        "data_health": None,
        "worker_degraded": True,
    }


@app.get("/api/bots/{bot_id}/live")
def api_bot_live(bot_id: int, logs_limit: int = 150, deals_limit: int = 30):
    """
    Single call used by the UI to avoid "Loading..." races.
    """
    try:
        b = get_bot(int(bot_id))
        if not b:
            return _json({"ok": False, "error": "Bot not found"}, 404)
        if bm is None:
            return _json(_bot_live_degraded(
                b, bot_id, logs_limit, deals_limit,
                "Worker not initialized. Check Kraken/Alpaca API keys and restart the service.",
                "Worker not initialized",
            ))

        try:
            snap = bm.snapshot(int(bot_id))
        except ValueError as ve:
            msg = str(ve)
            if "Alpaca live" in msg or "not initialized" in msg:
                return _json(_bot_live_degraded(
                    b, bot_id, logs_limit, deals_limit,
                    msg,
                    msg,
                ))
            raise
        except Exception:
            raise
        if snap.get("running") and not snap.get("last_event"):
            snap["last_event"] = "Running."
        if not snap.get("running") and not snap.get("last_event"):
            snap["last_event"] = "Stopped."
        
        # Try to fetch current price if missing or zero (non-blocking, with timeout)
        sym = b.get("symbol", "")
        market_type = classify_symbol(sym)
        current_price = snap.get("last_price")
        price_error = None
        
        if current_price is None or current_price <= 0:
            # Use cached price if available to avoid blocking
            # Only fetch fresh price if absolutely necessary and do it quickly
            import threading
            import time as time_module
            price_fetched = threading.Event()
            fetched_price = [None]
            fetch_error = [None]
            
            def _fetch_price():
                try:
                    if market_type == "stock":
                        # Stock: Try getting from Alpaca
                        client, _ = _get_bot_client(b)
                        if client:
                            # Use get_ticker with timeout protection
                            t = client.get_ticker(sym)
                            lp = float(t.get("last", 0) or t.get("price", 0) or 0)
                            if lp > 0:
                                fetched_price[0] = lp
                        else:
                            fetch_error[0] = "Alpaca client not available"
                    else:
                        # Crypto: Use Kraken safe price
                        lp = _safe_last_price(sym)
                        if lp is not None and lp > 0:
                            fetched_price[0] = lp
                        elif not _kraken_ready():
                            fetch_error[0] = "Kraken not ready"
                except Exception as e:
                    fetch_error[0] = str(e)
                finally:
                    price_fetched.set()
            
            # Start fetch in background thread with 2 second timeout
            fetch_thread = threading.Thread(target=_fetch_price, daemon=True)
            fetch_thread.start()
            fetch_thread.join(timeout=2.0)  # Max 2 seconds wait
            
            # If we got a price, use it; otherwise keep existing (might be 0 or None)
            if fetched_price[0] is not None and fetched_price[0] > 0:
                snap["last_price"] = fetched_price[0]
            elif fetch_error[0]:
                price_error = fetch_error[0]

        logs = list_logs(int(bot_id), limit=int(max(1, min(int(logs_limit), 2000))))
        deals = list_deals(int(bot_id), limit=int(max(1, min(int(deals_limit), 1000))))

        data_health = None
        try:
            router = getattr(bm, "_md_router", None)
            if router:
                data_health = router.get_data_health(sym, b.get("market_type", "crypto"), required_tfs=["1h", "4h", "1d"], min_candles=20)
        except Exception:
            pass
        
        return _json(
            {
                "ok": True,
                "bot": b,
                "snap": snap,
                "logs": logs,
                "deals": deals,
                "market_type": market_type,
                "kraken_ready": _kraken_ready(),
                "kraken_error": KRAKEN_ERROR,
                "alpaca_paper_ready": ALPACA_PAPER_READY,
                "alpaca_live_ready": ALPACA_LIVE_READY,
                "alpaca_error": ALPACA_ERROR or "",
                "price_error": price_error,
                "data_health": data_health,
            }
        )
    except Exception as e:
        logger.error(f"api_bot_live error for bot {bot_id}: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"Live refresh error: {type(e).__name__}: {e}"}, 500)


@app.get("/api/bots/{bot_id}/logs")
def api_bot_logs(bot_id: int, limit: int = 200):
    b = get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")

    safe_limit = int(max(1, min(int(limit), 2000)))
    logs = list_logs(int(bot_id), limit=safe_limit)

    # logs usually come newest-first. We will compress repeated spam lines.
    compressed = []
    last_msg = None
    prev_ts = 0
    prev_level = "INFO"
    repeat_count = 0

    for row in logs:
        msg = (row.get("message") or "").strip()
        if msg == last_msg:
            repeat_count += 1
            continue

        # flush previous repeated message
        if last_msg is not None:
            if repeat_count > 0:
                compressed.append(
                    {"ts": prev_ts, "level": prev_level, "message": f"{last_msg} (x{repeat_count+1})"}
                )
            else:
                compressed.append({"ts": prev_ts, "level": prev_level, "message": last_msg})

        # start tracking new message
        last_msg = msg
        prev_ts = int(row.get("ts") or 0)
        prev_level = row.get("level") or "INFO"
        repeat_count = 0

    # flush last
    if last_msg is not None:
        if repeat_count > 0:
            compressed.append({"ts": prev_ts, "level": prev_level, "message": f"{last_msg} (x{repeat_count+1})"})
        else:
            compressed.append({"ts": prev_ts, "level": prev_level, "message": last_msg})

    # return newest-first like before
    return {"ok": True, "logs": compressed[:200]}


@app.get("/api/bots/{bot_id}/deals")
def api_bot_deals(bot_id: int, limit: int = 50):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    return _json({"ok": True, "deals": list_deals(int(bot_id), limit=int(max(1, min(int(limit), 1000))))})


# =========================================================
# API: Trade Journal
# =========================================================
@app.get("/api/journal")
def api_journal_list(days: int = 90, strategy: Optional[str] = None):
    """List closed deals with journal entries for trade journal page."""
    since = now_ts() - (int(days) * 86400) if days else None
    deals = list_closed_deals_for_journal(since_ts=since, limit=500)
    if strategy and str(strategy).strip():
        strat = str(strategy).strip().lower()
        deals = [d for d in deals if (str(d.get("entry_strategy") or "").lower() == strat or str(d.get("exit_strategy") or "").lower() == strat)]
    deal_ids = [int(d["id"]) for d in deals]
    journals = list_trade_journals_for_deals(deal_ids)
    bot_map = {int(b["id"]): b for b in list_bots()}
    out = []
    for d in deals:
        j = journals.get(int(d["id"]), {})
        bot = bot_map.get(int(d.get("bot_id") or 0), {})
        out.append({
            **d,
            "journal": j,
            "bot_name": bot.get("name"),
            "bot_symbol": bot.get("symbol"),
        })
    return _json({"ok": True, "deals": out})


@app.get("/api/journal/{deal_id}")
def api_journal_get(deal_id: int):
    deal = get_deal(int(deal_id), full=True)
    if not deal:
        return _json({"ok": False, "error": "Deal not found"}, 404)
    journal = get_trade_journal(int(deal_id))
    bot = get_bot(int(deal.get("bot_id") or 0)) or {}
    return _json({
        "ok": True,
        "deal": deal,
        "journal": journal,
        "bot_name": bot.get("name"),
        "bot_symbol": bot.get("symbol"),
    })


@app.put("/api/journal/{deal_id}")
async def api_journal_put(deal_id: int, request: Request):
    deal = get_deal(int(deal_id))
    if not deal:
        return _json({"ok": False, "error": "Deal not found"}, 404)
    try:
        body = await request.json()
    except Exception:
        body = {}
    upsert_trade_journal(
        int(deal_id),
        entry_reason=body.get("entry_reason"),
        exit_reason=body.get("exit_reason"),
        lessons_learned=body.get("lessons_learned"),
        screenshot_data=body.get("screenshot_data"),
    )
    return _json({"ok": True, "journal": get_trade_journal(int(deal_id))})


# =========================================================
# API: Performance Analytics (Sharpe, Sortino, win rate, etc.)
# =========================================================
@app.get("/api/analytics/performance")
def api_analytics_performance(days: int = 90):
    """Sharpe, Sortino, max drawdown, win rate by strategy/symbol."""
    import math
    since = now_ts() - (int(days) * 86400)
    closed = list_closed_deals_for_journal(since_ts=since, limit=2000)
    closed = [d for d in closed if d.get("closed_at") and int(d.get("closed_at") or 0) >= since]
    pnls = [float(d.get("realized_pnl_quote") or 0) for d in closed]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    total_pnl = sum(pnls)
    win_rate = wins / len(pnls) if pnls else 0.0
    n = len(pnls)
    mean_ret = total_pnl / n if n else 0.0
    variance = sum((p - mean_ret) ** 2 for p in pnls) / n if n else 0.0
    std = math.sqrt(variance) if variance > 0 else 0.0
    downside = [p for p in pnls if p < 0]
    downside_var = sum(p ** 2 for p in downside) / len(downside) if downside else 0.0
    downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.0
    sharpe = (mean_ret / std * math.sqrt(252)) if std > 0 else 0.0
    sortino = (mean_ret / downside_std * math.sqrt(252)) if downside_std > 0 else 0.0
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        dd = peak - cum if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    by_strategy = {}
    by_symbol = {}
    for d in closed:
        s = d.get("entry_strategy") or "unknown"
        sym = d.get("symbol") or "?"
        if s not in by_strategy:
            by_strategy[s] = {"count": 0, "wins": 0, "pnl": 0.0}
        by_strategy[s]["count"] += 1
        by_strategy[s]["wins"] += 1 if float(d.get("realized_pnl_quote") or 0) > 0 else 0
        by_strategy[s]["pnl"] += float(d.get("realized_pnl_quote") or 0)
        if sym not in by_symbol:
            by_symbol[sym] = {"count": 0, "wins": 0, "pnl": 0.0}
        by_symbol[sym]["count"] += 1
        by_symbol[sym]["wins"] += 1 if float(d.get("realized_pnl_quote") or 0) > 0 else 0
        by_symbol[sym]["pnl"] += float(d.get("realized_pnl_quote") or 0)
    from datetime import datetime
    daily_pnl: Dict[str, float] = {}
    for d in closed:
        ts = int(d.get("closed_at") or 0)
        if ts:
            dt = datetime.utcfromtimestamp(ts)
            key = dt.strftime("%Y-%m-%d")
            daily_pnl[key] = daily_pnl.get(key, 0.0) + float(d.get("realized_pnl_quote") or 0)
    return _json({
        "ok": True,
        "days": days,
        "trades": n,
        "total_pnl": total_pnl,
        "win_rate": round(win_rate, 2),
        "wins": wins,
        "losses": losses,
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "by_strategy": by_strategy,
        "by_symbol": by_symbol,
        "daily_pnl": daily_pnl,
    })


# =========================================================
# API: chart candles + markers
# =========================================================
@app.get("/api/bots/{bot_id}/ohlc")
def api_bot_ohlc(bot_id: int, timeframe: str = "5m", limit: int = 200):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)
    
    client, is_kraken = _get_bot_client(b)
    if not client:
         return _json({"ok": False, "error": "Trading client not available", "candles": []}, 503)
    if is_kraken and not _kraken_ready():
        return _json({"ok": False, "error": KRAKEN_ERROR or "Kraken not ready", "candles": []}, 503)

    tf = _sanitize_tf(timeframe)
    lim = int(max(10, min(2000, int(limit))))

    symbol = _resolve_symbol(b.get("symbol", ""))
    if is_kraken:
        mk = _markets()
        if mk and symbol not in mk:
            return _json({"ok": False, "error": f"Symbol not found on Kraken: {symbol}", "candles": []}, 400)

    try:
        # Optimizations: Use BotManager cache for standard timeframes (Kraken only for now)
        # Only if bot is actually running in BM, otherwise BM cache might be stale/empty?
        # Actually BM cache is global for market data, so it's fine if BM is running.
        used_cache = False
        ohlcv = []
        
        if is_kraken and bm and tf in ("5m", "15m", "1h", "4h"):
            # Try cache first
            cached = bm.ohlcv_cached(symbol, tf, limit=lim)
            if cached:
                ohlcv = cached
                used_cache = True
        
        if not used_cache:
            # Direct fetch (Stocks, or Kraken cache miss, or custom TF)
            if is_kraken:
                ohlcv = client.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
            else:
                # Alpaca: use fetch_ohlcv (AlpacaAdapter) or get_ohlcv (AlpacaClient)
                try:
                    if hasattr(client, "fetch_ohlcv"):
                        ohlcv = client.fetch_ohlcv(symbol, timeframe=tf, limit=lim)
                    else:
                        ohlcv = client.get_ohlcv(symbol, tf, lim)
                except Exception as ex:
                    logger.warning("OHLC fetch failed for %s %s: %s", symbol, tf, ex)
                    ohlcv = []
            
        candles = []
        if not ohlcv:
             return _json({"ok": True, "candles": []})

        for row in ohlcv:
            # Handle potential different formats (just in case)
            # Expecting [ts, o, h, l, c, v] where ts might be in seconds or milliseconds
            if len(row) < 5: 
                continue
            
            try:
                ts_raw = float(row[0])
                # Convert to seconds: if timestamp > 1e10, it's in milliseconds
                if ts_raw > 1e10:
                    ts_sec = int(ts_raw // 1000)
                else:
                    ts_sec = int(ts_raw)
                
                candles.append(
                    {
                        "time": ts_sec,
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                    }
                )
            except (ValueError, TypeError, IndexError) as e:
                # Skip invalid rows
                continue
        
        return _json({"ok": True, "candles": candles, "symbol": symbol, "source": "cache" if used_cache else "api"})
    except Exception as e:
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "candles": []}, 500)


_BUY_RE = re.compile(r"\bbuy\b", re.IGNORECASE)
_SELL_RE = re.compile(r"\bsell\b", re.IGNORECASE)


@app.get("/api/bots/{bot_id}/markers")
def api_bot_markers(bot_id: int, timeframe: str = "5m", limit: int = 250):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    tf = _sanitize_tf(timeframe)
    bucket = _tf_seconds(tf)

    logs = list_logs(int(bot_id), limit=int(max(50, min(2000, int(limit)))))
    markers = []

    for x in logs:
        msg = str(x.get("message") or "")
        ts = int(x.get("ts") or 0)
        if ts <= 0:
            continue

        is_buy = bool(_BUY_RE.search(msg))
        is_sell = bool(_SELL_RE.search(msg))
        if not (is_buy or is_sell):
            continue

        t = int((ts // bucket) * bucket)
        markers.append(
            {
                "time": t,
                "position": "belowBar" if is_buy else "aboveBar",
                "color": "#16a34a" if is_buy else "#ef4444",
                "shape": "arrowUp" if is_buy else "arrowDown",
                "text": "BUY" if is_buy else "SELL",
            }
        )

    markers.sort(key=lambda m: int(m.get("time") or 0))
    return _json({"markers": markers})


# =========================================================
# API: stream (optional)
# =========================================================
@app.get("/api/bots/{bot_id}/stream")
def api_bot_stream(bot_id: int):
    """
    Optional SSE feed. UI doesn't need this (polling is safer),
    but it's correct and ready for later.
    """
    b = get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")
    if bm is None:
        raise HTTPException(status_code=503, detail="Worker not initialized")

    async def gen():
        while True:
            snap = bm.snapshot(int(bot_id))
            payload = {"ts": now_ts(), "snap": snap}
            yield f"data: {json.dumps(payload)}\n\n"
            await _async_sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})



@app.get("/api/bots/{bot_id}/logstream")
def api_bot_logstream(bot_id: int):
    """Server-Sent Events stream of new logs (no page refresh needed)."""
    b = get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")

    def gen():
        last_id = 0
        last_ping = 0.0
        while True:
            now = time.time()
            if now - last_ping >= 5.0:
                last_ping = now
                yield "event: ping\ndata: {}\n\n"

            try:
                rows = list_logs_since(int(bot_id), int(last_id), limit=200)
            except Exception:
                rows = []

            for r in rows:
                try:
                    last_id = int(r.get("id") or last_id)
                except Exception:
                    pass
                payload = {"log": {"id": r.get("id"), "ts": r.get("ts"), "level": r.get("level"), "message": r.get("message")}}
                yield f"data: {json.dumps(payload)}\n\n"

            time.sleep(1.0)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

async def _async_sleep(sec: float):
    import asyncio
    await asyncio.sleep(sec)


# =========================================================
# API: Intelligence Dashboard
# =========================================================
@app.get("/api/intelligence/decisions")
def api_intelligence_decisions(bot_id: Optional[int] = None, limit: int = 50):
    """Get recent intelligence decisions for dashboard"""
    try:
        if bot_id:
            decisions = get_intelligence_decisions(bot_id, limit=limit)
        else:
            # Get decisions for all bots
            decisions = []
            bots = list_bots()
            for bot in bots[:10]:  # Limit to first 10 bots
                bot_decisions = get_intelligence_decisions(bot.get("id"), limit=10)
                decisions.extend(bot_decisions)
            # Sort by timestamp descending
            decisions.sort(key=lambda d: d.get("ts", 0), reverse=True)
            decisions = decisions[:limit]
        
        return _json({"ok": True, "decisions": decisions})
    except Exception as e:
        logger.error(f"Intelligence decisions error: {type(e).__name__}: {e}")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}", "decisions": []}, 500)


@app.get("/intelligence")
def ui_intelligence_dashboard(request: Request):
    """Intelligence Dashboard UI"""
    _templates = Jinja2Templates(directory="templates")
    return _templates.TemplateResponse("intelligence_dashboard.html", {"request": request})


def _scan_stock_symbol(symbol: str, horizon: str, btc_ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not alpaca_paper and not alpaca_live:
         return {"symbol": symbol, "score": 0.0, "eligible": False, "reasons": ["Alpaca not ready"]}

    client = alpaca_live or alpaca_paper
    def fetch(tf):
        return client.get_ohlcv(symbol, timeframe=tf, limit=500)

    try:
        candles_1h = fetch("1h")
        candles_4h = fetch("4h")
        candles_1d = fetch("1d")
        candles_1w = fetch("1w")
    except Exception as e:
        return {"symbol": symbol, "score": 0.0, "eligible": False, "reasons": [f"Data fetch error: {e}"]}

    res = _analyze_market_data(symbol, horizon, btc_ctx, candles_1h, candles_4h, candles_1d, candles_1w)
    # Benchmark enrichment (SPY for stocks)
    try:
        from benchmark_analyzer import enrich_recommendation_with_benchmark
        from stock_metadata import get_sector
        if candles_1d and len(candles_1d) >= 30:
            benchmark_candles = None
            try:
                benchmark_candles = client.get_ohlcv("SPY", "1d", 200)
            except Exception:
                pass
            price = float(candles_1d[-1][4]) if candles_1d else 0.0
            enriched = enrich_recommendation_with_benchmark(
                symbol, price, candles_1d=candles_1d, benchmark_candles=benchmark_candles, sector=get_sector(symbol)
            )
            metrics = res.get("metrics") or {}
            for k, v in enriched.items():
                if v is not None and v != "":
                    metrics[k] = v
            res["metrics"] = metrics
            if enriched.get("peer_quartile") == "top":
                base = float(res.get("score") or 0)
                res["score"] = min(98.0, base + 3.0)
                res.setdefault("reasons", []).append("Top-quartile in sector")
            if enriched.get("benchmark_vs"):
                res.setdefault("reasons", []).append(enriched["benchmark_vs"])
    except Exception as e:
        logger.debug("Benchmark enrichment failed for stock %s: %s", symbol, e)
    return res


@app.post("/api/recommendations/scan_stocks")
def api_recommendations_scan_stocks(horizon: str = "short", limit: int = 150):
    if not alpaca_paper and not alpaca_live:
         return _json({"ok": False, "error": "Alpaca not configured"}, 503)
         
    client = alpaca_live if alpaca_live else alpaca_paper
    
    # 1. Build Stock Universe
    universe = []
    
    # Preset: "Mega-Cap + Popular"
    # Ideally we'd fetch active assets, but searching all active is heavy.
    # We'll use a larger static list + some dynamic checks if possible.
    # For now, let's use a robust list of liquid stocks/ETFs.
    
    # Tech / Growth
    universe += ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "NFLX", "INTC", "QCOM", "CRM", "ADBE", "AVGO", "TXN"]
    # Financials
    universe += ["JPM", "BAC", "V", "MA", "WFC", "GS", "MS", "BLK", "C", "AXP"]
    # ETFs
    universe += ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "GLD", "SLV", "TQQQ", "SQQQ", "SOXL", "ARKK"]
    # Crypto proxies
    universe += ["COIN", "MSTR", "MARA", "RIOT", "CLSK", "HUT", "BITF", "HOOD"]
    # Retail / Meme / High Vol
    universe += ["GME", "AMC", "ROKU", "PLTR", "SOFI", "UBER", "LYFT", "DKNG", "AFRM", "UPST", "CVNA"]
    # Defensive / Value
    universe += ["JNJ", "PG", "KO", "PEP", "WMT", "COST", "TGT", "HD", "LOW", "MCD", "DIS", "T", "VZ", "PFE", "MRK", "UNH"]
    
    # Deduplicate
    universe = list(set(universe))

    # --- EXPANDED UNIVERSE START ---
    # Fill up to 'limit' with random active assets to discover new stocks
    if len(universe) < limit:
        try:
            import random
            all_assets = client.get_active_assets()
            # Filter somewhat for quality (e.g. marginable usually implies better liquidity/status)
            candidates = [
                a["symbol"] for a in all_assets 
                if a.get("symbol") not in universe 
                and a.get("marginable") # simplistic quality filter
                and "." not in a.get("symbol") # avoid weird warrants/classes often
            ]
            
            needed = limit - len(universe)
            if needed > 0 and candidates:
                # Random sample
                if len(candidates) > needed:
                    fill = random.sample(candidates, needed)
                else:
                    fill = candidates
                universe.extend(fill)
        except Exception as e:
            logger.warning("Error expanding universe: %s", e)
    # --- EXPANDED UNIVERSE END ---

    # Dynamic: Add Top Gainers (if feasible) to ensure we find "Strong Buys" even if market is mixed
    try:
        movers = client.get_top_movers()
        gainers = [x["symbol"] for x in movers.get("gainers", [])]
        universe.extend(gainers)
        # Re-deduplicate
        universe = list(set(universe))
    except Exception:
        pass
    
    results = []
    processed = 0
    errors = 0
    
    # Get BTC context for correlation? Stocks have their own beta.
    # We'll use SPY regime as context.
    spy_ctx = {"risk_off": False}
    try:
        spy_c = client.get_ohlcv("SPY", "1d", 200)
        spy_r = detect_regime(spy_c)
        if spy_r.regime in ("TREND_DOWN", "CRASH") or (spy_r.scores or {}).get("downtrend_score", 0) > 0.65:
            spy_ctx["risk_off"] = True
    except Exception as e:
        logger.debug("SPY regime check failed: %s", e)

    # Batch processing to avoid rate limits
    chunk_size = 50
    for i in range(0, min(len(universe), int(limit)), chunk_size):
        chunk = universe[i:i+chunk_size]
        
        # We can pre-fetch snapshots to check volume/price filter
        try:
            snaps = client.get_snapshots(chunk)
        except Exception as e:
            logger.debug("Batch snapshots failed: %s", e)
            snaps = {}
             
        for sym in chunk:
            try:
                # Pre-filter Check
                snap = snaps.get(sym)
                if snap:
                    price = 0.0
                    vol = 0.0
                    if snap.get("dailyBar"):
                        price = float(snap["dailyBar"].get("c", 0))
                        vol = float(snap["dailyBar"].get("v", 0)) * price
                    elif snap.get("latestTrade"):
                         price = float(snap["latestTrade"].get("p", 0))
                    
                    # Skip penny stocks or illiquid
                    if price < 5.0 or vol < 500000: # Min $5 price, $500k volume
                        continue

                res = _scan_stock_symbol(sym, horizon, spy_ctx)
                
                # Check eligibility
                # We save ALL valid scans so user sees "Weak" stocks too, 
                # but we can flag them.
                if res.get("score") is not None:
                     metrics = res.get("metrics") or {}
                     metrics["market_type"] = "stocks" # TAG AS STOCKS
                     
                     save_recommendation_snapshot(
                        symbol=sym,
                        horizon=horizon,
                        score=float(res.get("score") or 0.0),
                        regime_json=json.dumps(res.get("regime") or {}),
                        metrics_json=json.dumps(metrics),
                        reasons_json=json.dumps(res.get("reasons") or []),
                        risk_flags_json=json.dumps(res.get("risk_flags") or []),
                    )
                     processed += 1
                     results.append(res)
            except Exception:
                errors += 1
                continue
                
    return _json({
        "ok": True, 
        "message": f"Scanned {processed} stocks",
        "processed": processed,
        "errors": errors
    })


# =========================================================
# Autopilot API
# =========================================================
_autopilot_last_run: float = 0.0
_autopilot_next_run: float = 0.0


def _autopilot_loop() -> None:
    """Background loop: run autopilot cycle every scan_interval when enabled."""
    import autopilot
    import autopilot as ap_mod
    global _autopilot_last_run, _autopilot_next_run
    interval = int(getattr(ap_mod, "AUTOPILOT_SCAN_INTERVAL_SEC", 14400))
    try:
        cfg = ap_mod.get_autopilot_config()
        hours = cfg.get("scan_interval_hours")
        if hours is not None and int(hours) > 0:
            interval = int(hours) * 3600
    except Exception:
        pass
    while True:
        try:
            time.sleep(min(60, interval // 10))
            if not ap_mod.is_autopilot_enabled():
                continue
            now = time.time()
            if now < _autopilot_next_run:
                continue
            _autopilot_next_run = now + interval
            _autopilot_last_run = now

            def _create_bot_fn(payload):
                return create_bot(payload)

            def _delete_bot_fn(bot_id):
                delete_bot(int(bot_id))

            def _start_bot_fn(bot_id):
                if bm:
                    try:
                        bot = get_bot(int(bot_id))
                        if bot and _can_start_bot_live(bot)[0]:
                            bm.start(int(bot_id))
                    except Exception as e:
                        logger.warning("autopilot start_bot %s: %s", bot_id, e)

            def _stop_bot_fn(bot_id):
                if bm:
                    try:
                        bm.stop(int(bot_id))
                    except Exception as e:
                        logger.warning("autopilot stop_bot %s: %s", bot_id, e)

            def _get_portfolio_fn():
                try:
                    snap = _portfolio_snapshot()
                    t = float(snap.get("total_usd") or 0)
                    if t > 0:
                        return t
                    if bm and hasattr(bm, "get_portfolio_total"):
                        return float(bm.get_portfolio_total())
                except Exception:
                    pass
                return 0.0

            notify_fn = None
            try:
                from notification_manager import notify
                notify_fn = notify
            except Exception:
                pass

            res = ap_mod.run_autopilot_cycle(
                create_bot_fn=_create_bot_fn,
                delete_bot_fn=_delete_bot_fn,
                start_bot_fn=_start_bot_fn,
                stop_bot_fn=_stop_bot_fn,
                get_portfolio_total_fn=_get_portfolio_fn,
                notify_fn=notify_fn,
            )
            if res.get("created") or res.get("closed"):
                logger.info("autopilot cycle: created=%s closed=%s", res.get("created", 0), res.get("closed", 0))
        except Exception as e:
            logger.exception("autopilot loop error: %s", e)


@app.get("/api/autopilot/activity")
def api_autopilot_activity(request: Request):
    """Return latest autopilot audit log entries for the dashboard."""
    try:
        raw = request.query_params.get("limit") if request else None
        limit = min(100, int(raw) if raw not in (None, "") else 50)
    except (TypeError, ValueError):
        limit = 50
    try:
        from db import list_autopilot_audit_log
        rows = list_autopilot_audit_log(limit=limit)
    except Exception as e:
        logger.warning("autopilot activity log failed: %s", e)
        rows = []
    for r in rows:
        if r.get("details_json"):
            try:
                r["details"] = json.loads(r["details_json"])
            except Exception:
                r["details"] = None
    return _json({"ok": True, "items": rows})


@app.get("/api/autopilot/config")
def api_autopilot_config_get():
    """Get autopilot_config from db (Master Upgrade Part 4)."""
    from db import get_autopilot_config_row
    row = get_autopilot_config_row()
    if not row:
        return _json({"ok": True, "config": {}})
    cfg = {k: v for k, v in row.items() if k != "id"}
    return _json({"ok": True, "config": cfg})


@app.post("/api/autopilot/config")
async def api_autopilot_config_save(request: Request):
    """Save autopilot_config (Master Upgrade Part 4)."""
    from db import save_autopilot_config
    body = await request.json()
    save_autopilot_config(body or {})
    return _json({"ok": True})


@app.post("/api/autopilot/config/update")
async def api_autopilot_config_update(request: Request):
    """Merge body into settings-stored autopilot_config (used by dashboard edit)."""
    import autopilot
    body = await request.json() or {}
    cfg = dict(autopilot.get_autopilot_config())
    for k, v in body.items():
        if v is not None and k != "id":
            cfg[k] = v
    set_setting("autopilot_config", json.dumps(cfg))
    return _json({"ok": True, "config": cfg})


@app.get("/api/autopilot/status")
def api_autopilot_status():
    import autopilot
    enabled = autopilot.is_autopilot_enabled()
    cfg = autopilot.get_autopilot_config()
    now = time.time()
    next_sec = max(0, int(_autopilot_next_run - now)) if _autopilot_next_run > now else 0
    portfolio_value = 0.0
    active_positions = 0
    total_pnl = 0.0
    try:
        snap = _portfolio_snapshot()
        portfolio_value = float(snap.get("total_usd") or 0)
        bots = list_bots()
        active_positions = sum(1 for b in bots if int(b.get("enabled", 0)) == 1)
        from db import all_deal_stats
        stats = all_deal_stats()
        total_pnl = float(stats.get("realized_total", 0) or 0)
    except Exception:
        pass
    last_heartbeat = int(get_setting("autopilot_last_heartbeat_ts", "0") or 0)
    last_run_ts = int(_autopilot_last_run)
    if last_run_ts <= 0 and last_heartbeat > 0:
        last_run_ts = last_heartbeat
    return _json({
        "ok": True,
        "enabled": enabled,
        "config": cfg,
        "last_run_ts": last_run_ts,
        "next_scan_in_sec": next_sec,
        "last_autopilot_heartbeat_ts": last_heartbeat if last_heartbeat else None,
        "portfolio_value": portfolio_value,
        "active_positions": active_positions,
        "max_positions": int(cfg.get("max_positions") or 6),
        "total_pnl": total_pnl,
    })


@app.get("/api/autopilot/positions")
def api_autopilot_positions():
    """Active bots with live P&L for dashboard (123.md Fix 3)."""
    bots = [b for b in list_bots() if int(b.get("enabled", 0)) == 1]
    out = []
    for b in bots:
        bot_id = int(b.get("id", 0))
        snap = {}
        if bm:
            try:
                snap = bm.snapshot(bot_id) or {}
            except Exception:
                pass
        last_price = float(snap.get("last_price") or 0)
        if last_price <= 0:
            try:
                sym = b.get("symbol", "")
                if sym:
                    if classify_symbol(sym) == "stock" and (alpaca_live or alpaca_paper):
                        client = alpaca_live or alpaca_paper
                        t = client.get_ticker(sym)
                        last_price = float(t.get("last") or 0)
                    else:
                        tc = _ticker_cached(sym, ttl_sec=60)
                        if tc:
                            last_price = float(tc.get("last") or tc.get("c") or 0)
            except Exception:
                pass
        avg_entry = float(snap.get("avg_entry") or 0)
        base_pos = float(snap.get("base_pos") or 0)
        if avg_entry <= 0 and base_pos > 0 and last_price > 0:
            avg_entry = last_price
        position_value = base_pos * last_price if last_price > 0 else base_pos
        unrealized_pnl = 0.0
        unrealized_pnl_pct = 0.0
        if avg_entry > 0 and base_pos > 0:
            unrealized_pnl = (last_price - avg_entry) * base_pos
            unrealized_pnl_pct = ((last_price - avg_entry) / avg_entry) * 100
        out.append({
            "id": bot_id,
            "symbol": b.get("symbol", ""),
            "strategy": b.get("strategy_mode", "classic"),
            "enabled": int(b.get("enabled", 0)),
            "avg_entry_price": avg_entry,
            "current_price": last_price,
            "position_value": position_value,
            "quantity": base_pos,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "take_profit_price": snap.get("tp_price"),
            "stop_loss_price": None,
            "trading_mode": b.get("trading_mode", "swing_trade"),
        })
    return _json({"ok": True, "positions": out})


@app.post("/api/autopilot/toggle")
def api_autopilot_toggle():
    val = get_setting("autopilot_enabled", "0")
    new_val = "0" if str(val).strip().lower() in ("1", "true", "yes", "y", "on") else "1"
    set_setting("autopilot_enabled", new_val)
    return _json({"ok": True, "enabled": new_val == "1"})


@app.post("/api/autopilot/start")
def api_autopilot_start():
    """Start ALL autopilot bots: enable setting + enable all bot_type='autopilot' bots + start in BotManager."""
    set_setting("autopilot_enabled", "1")
    n = update_bots_by_type("autopilot", 1)
    autopilot_bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
    started = 0
    for bot in autopilot_bots:
        bot_id = int(bot.get("id", 0))
        if bot_id and bm:
            try:
                if _can_start_bot_live(bot)[0]:
                    bm.start(bot_id)
                    started += 1
            except Exception as e:
                logger.warning("autopilot start bot %s: %s", bot_id, e)
    logger.info("Autopilot STARTED - %d bots enabled, %d started in BotManager", n, started)
    return _json({"ok": True, "enabled": True, "bots_updated": n, "bots_started": started})


@app.post("/api/autopilot/stop")
def api_autopilot_stop():
    """Stop ALL autopilot bots: disable setting + disable all bot_type='autopilot' bots + stop in BotManager."""
    set_setting("autopilot_enabled", "0")
    n = update_bots_by_type("autopilot", 0)
    autopilot_bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
    stopped = 0
    for bot in autopilot_bots:
        bot_id = int(bot.get("id", 0))
        if bot_id and bm:
            try:
                bm.stop(bot_id)
                stopped += 1
            except Exception as e:
                logger.warning("autopilot stop bot %s: %s", bot_id, e)
    logger.info("Autopilot STOPPED - %d bots disabled, %d stopped in BotManager", n, stopped)
    return _json({"ok": True, "enabled": False, "bots_updated": n, "bots_stopped": stopped})


@app.post("/api/autopilot/run")
def api_autopilot_run():
    global _autopilot_last_run, _autopilot_next_run
    import autopilot
    import autopilot as ap_mod

    def _create_bot_fn(payload):
        return create_bot(payload)

    def _delete_bot_fn(bot_id):
        delete_bot(int(bot_id))

    def _start_bot_fn(bot_id):
        if bm:
            try:
                bot = get_bot(int(bot_id))
                if bot and _can_start_bot_live(bot)[0]:
                    bm.start(int(bot_id))
            except Exception as e:
                logger.warning("autopilot start_bot %s: %s", bot_id, e)

    def _stop_bot_fn(bot_id):
        if bm:
            try:
                bm.stop(int(bot_id))
            except Exception as e:
                logger.warning("autopilot stop_bot %s: %s", bot_id, e)

    def _get_portfolio_fn():
        try:
            snap = _portfolio_snapshot()
            t = float(snap.get("total_usd") or 0)
            if t > 0:
                return t
            if bm and hasattr(bm, "get_portfolio_total"):
                return float(bm.get_portfolio_total())
        except Exception:
            pass
        return 0.0

    notify_fn = None
    try:
        from notification_manager import notify
        notify_fn = notify
    except Exception:
        pass

    res = ap_mod.run_autopilot_cycle(
        create_bot_fn=_create_bot_fn,
        delete_bot_fn=_delete_bot_fn,
        start_bot_fn=_start_bot_fn,
        stop_bot_fn=_stop_bot_fn,
        get_portfolio_total_fn=_get_portfolio_fn,
        notify_fn=notify_fn,
        force_run=True,
    )
    _autopilot_last_run = time.time()
    interval = getattr(ap_mod, "AUTOPILOT_SCAN_INTERVAL_SEC", 14400)
    try:
        cfg = ap_mod.get_autopilot_config()
        if cfg.get("scan_interval_hours"):
            interval = int(cfg["scan_interval_hours"]) * 3600
    except Exception:
        pass
    _autopilot_next_run = time.time() + interval
    return _json({"ok": True, "created": res.get("created", 0), "closed": res.get("closed", 0), "errors": res.get("errors", [])})


@app.get("/api/autopilot/top")
def api_autopilot_top():
    import autopilot
    cfg = autopilot.get_autopilot_config()
    min_score = float(cfg.get("min_score") or cfg.get("min_score_threshold") or getattr(autopilot, "AUTOPILOT_MIN_SCORE", 75))
    min_score = max(50, min(95, min_score))
    items = autopilot.get_top_recommendations(
        min_score=min_score,
        max_count=15,
        asset_filter=cfg.get("asset_types"),
        sectors_avoid=cfg.get("sectors_avoid"),
    )
    out = []
    for r in items:
        m = r.get("metrics") or {}
        reason = m.get("explanation") or m.get("recommended_strategy") or m.get("reason") or ""
        out.append({
            "symbol": r.get("symbol"),
            "score": r.get("score"),
            "horizon": "long",
            "reason": (str(reason)[:200] if reason else None),
        })
    return _json({"ok": True, "items": out})


@app.get("/api/opportunities/now")
def api_now_opportunities(max_count: int = 3, asset_filter: Optional[str] = None):
    """
    Return best opportunities right now for radar/autopilot.

    Uses autopilot config, recommendations, and active bots to pick top candidates.
    """
    import autopilot
    try:
        max_count = max(1, min(int(max_count), 10))
    except Exception:
        max_count = 3

    cfg = autopilot.get_autopilot_config()
    opp_defaults = cfg.get("opportunity_defaults") or {}
    if not asset_filter:
        asset_filter = str(opp_defaults.get("asset_filter") or cfg.get("asset_types") or "both")
    min_score = float(
        opp_defaults.get("min_score")
        or cfg.get("min_score")
        or cfg.get("min_score_threshold")
        or getattr(autopilot, "AUTOPILOT_MIN_SCORE", 75)
    )
    min_score = max(50, min(95, min_score))

    items = autopilot.get_now_opportunities(
        asset_filter=asset_filter,
        max_count=max_count,
        min_score=min_score,
    )
    return _json(
        {
            "ok": True,
            "items": items,
            "config": {
                "asset_filter": asset_filter,
                "min_score": min_score,
            },
        }
    )


@app.get("/api/autopilot/watchlist")
def api_autopilot_watchlist():
    import autopilot
    cfg = autopilot.get_autopilot_config()
    items = autopilot.get_watchlist(asset_filter=cfg.get("asset_types"))
    out = []
    for r in items:
        m = r.get("metrics") or {}
        reason = m.get("explanation") or m.get("recommended_strategy") or ""
        out.append({
            "symbol": r.get("symbol"),
            "score": r.get("score"),
            "horizon": "long",
            "reason": (str(reason)[:200] if reason else None),
        })
    return _json({"ok": True, "items": out})


@app.post("/api/autopilot/setup")
async def api_autopilot_setup(request: Request):
    from portfolio_initializer import compute_optimal_allocation, save_autopilot_config
    body = await request.json()
    total_capital = float(body.get("total_capital") or 10000)
    risk_tolerance = str(body.get("risk_tolerance") or "moderate")
    asset_types = str(body.get("asset_types") or "both")
    max_positions = int(body.get("max_positions") or 10)
    max_bots_per_sector = body.get("max_bots_per_sector")
    if max_bots_per_sector is not None:
        max_bots_per_sector = max(1, min(10, int(max_bots_per_sector)))
    sectors_avoid = body.get("sectors_avoid") or []
    if isinstance(sectors_avoid, str):
        sectors_avoid = [s.strip() for s in sectors_avoid.split(",") if s.strip()]
    dry_run = int(body.get("dry_run", 1))
    min_score = body.get("min_score")
    if min_score is not None:
        min_score = max(50, min(95, float(min_score)))
    scan_interval_hours = body.get("scan_interval_hours")
    if scan_interval_hours is not None:
        scan_interval_hours = max(1, min(168, int(scan_interval_hours)))
    auto_delete_closed = body.get("auto_delete_closed", False)

    alloc = compute_optimal_allocation(
        total_capital=total_capital,
        risk_tolerance=risk_tolerance,
        max_positions=max_positions,
        asset_types=asset_types,
        sectors_avoid=sectors_avoid,
    )
    alloc["dry_run"] = dry_run
    alloc["alpaca_mode"] = "paper" if dry_run else str(body.get("alpaca_mode") or "live")
    if min_score is not None:
        alloc["min_score"] = min_score
    if scan_interval_hours is not None:
        alloc["scan_interval_hours"] = scan_interval_hours
    alloc["auto_delete_closed"] = bool(auto_delete_closed)
    if max_bots_per_sector is not None:
        alloc["max_bots_per_sector"] = max_bots_per_sector
    # Preserve opportunity_defaults when saving portfolio config
    try:
        import autopilot
        existing = autopilot.get_autopilot_config()
        alloc["opportunity_defaults"] = existing.get("opportunity_defaults") or {}
    except Exception:
        alloc["opportunity_defaults"] = {}
    save_autopilot_config(alloc)
    return _json({"ok": True, "config": alloc})


@app.post("/api/autopilot/opportunity-defaults")
async def api_autopilot_opportunity_defaults(request: Request):
    """Save Now opportunities configuration (min score, defaults for Open bot)."""
    from portfolio_initializer import save_autopilot_config
    import autopilot
    body = await request.json() or {}
    min_score = body.get("min_score")
    if min_score is not None:
        min_score = max(50, min(95, float(min_score)))
    defaults = {
        "min_score": min_score,
        "asset_filter": str(body.get("asset_filter") or "both"),
        "mode": str(body.get("mode") or "dry"),
        "capital_per_bot": float(body.get("capital_per_bot") or 250),
        "base_quote": float(body.get("base_quote") or 25),
        "safety_quote": float(body.get("safety_quote") or 25),
        "max_safety": int(body.get("max_safety") or 3),
        "tp_pct": float(body.get("tp_pct") or 1.2),
        "first_dev_pct": float(body.get("first_dev_pct") or 1.5),
        "step_mult": float(body.get("step_mult") or 1.2),
    }
    cfg = dict(autopilot.get_autopilot_config())
    cfg["opportunity_defaults"] = defaults
    save_autopilot_config(cfg)
    return _json({"ok": True, "opportunity_defaults": defaults})


@app.get("/api/notification/prefs")
def api_notification_prefs():
    """Get notification preferences (autopilot, Discord, etc.). Webhook URL is in .env only."""
    try:
        from notification_manager import _get_notification_prefs
        prefs = _get_notification_prefs()
        has_webhook = bool(os.getenv("DISCORD_WEBHOOK_URL", "").strip())
        return _json({"ok": True, "prefs": prefs, "discord_configured": has_webhook})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/notification/prefs")
async def api_notification_prefs_save(request: Request):
    """Update notification preferences (enabled, discord on/off). Set DISCORD_WEBHOOK_URL in .env for Discord."""
    try:
        body = await request.json() or {}
        from notification_manager import _get_notification_prefs
        prefs = dict(_get_notification_prefs())
        if "enabled" in body:
            prefs["enabled"] = bool(body["enabled"])
        if "discord" in body:
            prefs["discord"] = bool(body["discord"])
        set_setting("notification_prefs", json.dumps(prefs))
        return _json({"ok": True, "prefs": prefs})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@app.post("/api/autopilot/capital/add")
async def api_autopilot_capital_add(request: Request):
    """Add capital to autopilot bots: all (split equally), all_enabled (any enabled bot), single bot, or percentage to each.
    Only updates bot config (base_quote, max_spend_quote). No order is placed. The bot uses
    the extra budget on its next tick when the strategy sees a good entry point."""
    body = await request.json() or {}
    amount_usd = float(body.get("amount_usd") or 0)
    mode = str(body.get("mode") or "all").strip().lower()
    bot_id = body.get("bot_id")
    pct_per_bot = body.get("pct_per_bot")
    if amount_usd <= 0:
        return _json({"ok": False, "error": "amount_usd must be positive"}, 400)
    bots = [b for b in list_bots() if str(b.get("bot_type") or "").lower() == "autopilot"]
    if mode == "all_enabled":
        bots = [b for b in list_bots() if int(b.get("enabled", 0)) == 1]
    if mode == "single":
        if not bot_id:
            return _json({"ok": False, "error": "bot_id required for single"}, 400)
        bots = [b for b in list_bots() if int(b.get("id")) == int(bot_id)]
        if not bots:
            return _json({"ok": False, "error": "Bot not found"}, 404)
    if not bots:
        return _json({"ok": False, "error": "No bots to update. Create bots (Setup Autopilot or Bots page) or use 'All enabled bots' if you have enabled bots."}, 400)
    updated = 0
    for b in bots:
        bid = int(b.get("id"))
        cur = get_bot(bid)
        if not cur:
            continue
        base = float(cur.get("base_quote") or 0)
        spend = float(cur.get("max_spend_quote") or 0)
        if mode == "all":
            add = amount_usd / len(bots)
        elif mode == "single":
            add = amount_usd
        else:
            add = (float(pct_per_bot or 0) / 100.0) * base
        if add <= 0:
            continue
        data = dict(cur)
        data["base_quote"] = base + add
        data["max_spend_quote"] = spend + add
        try:
            update_bot(bid, data)
            updated += 1
        except Exception as e:
            logger.warning("autopilot capital add bot %s: %s", bid, e)
    msg = f"Added capital to {updated} bot(s)."
    return _json({"ok": True, "updated": updated, "message": msg})
# NOTE: Duplicate startup event was removed - the main startup() function at line ~1312 
# handles all initialization correctly (Kraken, Alpaca, BotManager with proper signatures)
