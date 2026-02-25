"""
one_server.py

Single-server entrypoint: combines UI pages + worker API into ONE FastAPI app.

How to run (Windows PowerShell) from your project folder:
  py -m pip install -r requirements.txt
  py -m uvicorn one_server:app --reload --port 8000

Then open:
  http://127.0.0.1:8000/
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time as time_mod
from datetime import datetime, time, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import db
import worker_api

logger = logging.getLogger(__name__)
app: FastAPI = worker_api.app
templates = Jinja2Templates(directory="templates")


def _base_ctx(request: Request) -> Dict[str, Any]:
    """Base context for all pages; includes api_token for fetch when WORKER_API_TOKEN is set."""
    token = getattr(worker_api, "WORKER_API_TOKEN", "") or ""
    lang = (request.cookies.get("lang") or "").strip() or os.getenv("LANGUAGE", "en").strip() or "en"
    return {
        "request": request,
        "api_token": token,
        "enable_pwa": os.getenv("ENABLE_PWA", "1").strip().lower() in ("1", "true", "yes"),
        "default_theme": os.getenv("DEFAULT_THEME", "auto").strip() or "auto",
        "language": lang,
        "enable_voice_alerts": os.getenv("ENABLE_VOICE_ALERTS", "0").strip().lower() in ("1", "true", "yes"),
    }

UI_PASSWORD = os.getenv("UI_PASSWORD", "").strip()
_ui_secret_env = os.getenv("UI_SESSION_SECRET", "").strip()
_worker_token = os.getenv("WORKER_API_TOKEN", "").strip()
if _ui_secret_env:
    UI_SESSION_SECRET = _ui_secret_env
elif _worker_token:
    UI_SESSION_SECRET = _worker_token
    logger.warning(
        "UI_SESSION_SECRET derived from WORKER_API_TOKEN. "
        "Set UI_SESSION_SECRET explicitly for production."
    )
else:
    UI_SESSION_SECRET = secrets.token_hex(32)
    logger.warning(
        "UI_SESSION_SECRET not set; generated ephemeral secret. "
        "Sessions will not persist across restarts. Set UI_SESSION_SECRET for production."
    )

UI_SESSION_TTL_SEC = int(os.getenv("UI_SESSION_TTL_SEC", "86400"))
UI_REMEMBER_TTL_SEC = int(os.getenv("UI_REMEMBER_TTL_SEC", "1209600"))


def _sign_session(raw: str) -> str:
    secret = UI_SESSION_SECRET.encode("utf-8")
    return hmac.new(secret, raw.encode("utf-8"), hashlib.sha256).hexdigest()


def _encode_session(user: str, exp_ts: int) -> str:
    raw = f"{user}|{exp_ts}"
    sig = _sign_session(raw)
    token = f"{raw}|{sig}"
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("utf-8")


def _decode_session(token: str) -> Optional[Tuple[str, int]]:
    if not token or not UI_SESSION_SECRET:
        return None
    try:
        raw = base64.urlsafe_b64decode(token.encode("utf-8")).decode("utf-8")
        user, exp_str, sig = raw.split("|", 2)
        if not user or not exp_str or not sig:
            return None
        exp_ts = int(exp_str)
        if exp_ts < int(time_mod.time()):
            return None
        expected = _sign_session(f"{user}|{exp_ts}")
        if not hmac.compare_digest(sig, expected):
            return None
        return user, exp_ts
    except Exception:
        return None


def _auth_enabled() -> bool:
    return bool(UI_PASSWORD and UI_SESSION_SECRET)


def _is_https(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    return (request.headers.get("x-forwarded-proto") or "").strip().lower() == "https"


def _is_public_path(path: str) -> bool:
    return (
        path.startswith("/static/")
        or path in ("/login", "/logout", "/favicon.ico", "/health")
    )


@app.middleware("http")
async def ui_auth_guard(request: Request, call_next):
    if not _auth_enabled():
        request.state.user = None
        return await call_next(request)

    path = request.url.path or ""
    if _is_public_path(path):
        request.state.user = None
        return await call_next(request)

    token = request.cookies.get("eirin_session", "")
    session = _decode_session(token)
    if session:
        request.state.user = session[0]
        return await call_next(request)

    # Not authenticated
    request.state.user = None
    if path.startswith("/api/"):
        return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
    return RedirectResponse(url=f"/login?next={path}", status_code=307)


def _local_midnight_ts() -> int:
    """Local midnight timestamp (seconds)."""
    now = datetime.now()
    midnight = datetime.combine(now.date(), time.min)
    return int(midnight.timestamp())


def _build_pnl() -> Dict[str, float]:
    """PnL shown on dashboard template: today + total (realized only)."""
    today = db.pnl_summary(_local_midnight_ts()).get("realized", 0.0) or 0.0
    total = db.pnl_summary(0).get("realized", 0.0) or 0.0
    try:
        return {"today": float(today), "total": float(total)}
    except Exception:
        return {"today": 0.0, "total": 0.0}


def _json_payload(resp: Any) -> Dict[str, Any]:
    """
    Accept dict or JSONResponse and return dict payload.
    """
    if isinstance(resp, dict):
        return resp
    try:
        body = getattr(resp, "body", None)
        if body:
            import json
            return json.loads(body.decode("utf-8"))
    except Exception:
        pass
    return {}


def _portfolio_snapshot() -> Dict[str, Any]:
    """Prefer worker_api portfolio snapshot (needs Kraken); fallback to empty."""
    try:
        payload = _json_payload(worker_api.api_portfolio())  # type: ignore[misc]
        if isinstance(payload, dict) and payload.get("ok") and "portfolio" in payload:
            return payload["portfolio"]
    except Exception:
        pass
    return {
        "holdings": [],
        "total_usd": 0.0,
        "free_usd": 0.0,
        "used_usd": 0.0,
        "positions_usd": 0.0,
        "ts": int(datetime.now(tz=timezone.utc).timestamp()),
    }


def _bots_snapshot() -> Any:
    """Bots list from DB."""
    try:
        return db.list_bots()
    except Exception:
        return []


def _bot_runtime(bot_id: int) -> Dict[str, Any]:
    try:
        payload = _json_payload(worker_api.api_bot_status(int(bot_id)))  # type: ignore[misc]
        if isinstance(payload, dict) and payload.get("ok"):
            return payload.get("snap") or {}
    except Exception:
        pass
    return {}


# ----------------------------
# UI routes
# ----------------------------

@app.get("/bot", include_in_schema=False)
def ui_bot_redirect() -> RedirectResponse:
    # People often try /bot (singular). Send them to /bots.
    return RedirectResponse(url="/bots", status_code=307)


@app.get("/login", include_in_schema=False)
def ui_login(request: Request, next: str = "/"):
    if _auth_enabled():
        token = request.cookies.get("eirin_session", "")
        if _decode_session(token):
            return RedirectResponse(url=next or "/", status_code=307)
    return templates.TemplateResponse(
        "login.html",
        {**_base_ctx(request), "next": next or "/", "auth_enabled": _auth_enabled()},
    )


@app.post("/login", include_in_schema=False)
async def ui_login_post(request: Request):
    form = await request.form()
    password = str(form.get("password") or "")
    next_url = str(form.get("next") or "/")
    remember = str(form.get("remember") or "").lower() in ("1", "true", "yes", "on")

    if not _auth_enabled():
        return RedirectResponse(url=next_url or "/", status_code=307)

    if password != UI_PASSWORD:
        return templates.TemplateResponse(
            "login.html",
            {
                **_base_ctx(request),
                "next": next_url,
                "error": "Invalid password.",
                "auth_enabled": _auth_enabled(),
            },
            status_code=401,
        )

    ttl = UI_REMEMBER_TTL_SEC if remember else UI_SESSION_TTL_SEC
    exp_ts = int(time_mod.time()) + int(ttl)
    token = _encode_session("admin", exp_ts)
    resp = RedirectResponse(url=next_url or "/", status_code=307)
    resp.set_cookie(
        "eirin_session",
        token,
        httponly=True,
        samesite="lax",
        secure=_is_https(request),
        max_age=int(ttl),
    )
    return resp


@app.get("/logout", include_in_schema=False)
def ui_logout():
    resp = RedirectResponse(url="/login", status_code=307)
    resp.delete_cookie("eirin_session")
    return resp


_DASH_CACHE: Dict[str, Any] = {"ts": 0, "data": None}
_DASH_TTL_SEC = 3


def _dashboard_data() -> Dict[str, Any]:
    bots = _bots_snapshot()
    bot_rows = []
    unrealized_total = 0.0
    runtime_map: Dict[int, Dict[str, Any]] = {}
    for b in bots or []:
        try:
            stats = db.bot_deal_stats(int(b.get("id")))
        except Exception:
            stats = {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
        snap = _bot_runtime(int(b.get("id")))
        runtime_map[int(b.get("id"))] = snap
        unrealized = 0.0
        try:
            avg = snap.get("avg_entry")
            last = snap.get("last_price")
            pos = snap.get("base_pos")
            if avg is not None and last is not None and pos is not None:
                unrealized = float(last - avg) * float(pos)
        except Exception:
            pass
        unrealized_total += float(unrealized or 0.0)
        bot_rows.append({**b, "stats": stats, "runtime": snap, "unrealized_pnl": unrealized})

    try:
        deals_stats = db.all_deal_stats()
    except Exception:
        deals_stats = {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
    deals_stats["unrealized_total"] = float(unrealized_total)

    bot_map = {int(b.get("id")): b for b in bots or []}

    def enrich_deal(d: Dict[str, Any]) -> Dict[str, Any]:
        bot_id = int(d.get("bot_id") or 0)
        bot = bot_map.get(bot_id, {})
        tp = float(bot.get("tp") or 0.0)
        entry = d.get("entry_avg")
        tp_target = (float(entry) * (1.0 + tp)) if entry is not None else None
        snap = runtime_map.get(bot_id, {})
        unreal = None
        try:
            avg = snap.get("avg_entry")
            last = snap.get("last_price")
            pos = snap.get("base_pos")
            if avg is not None and last is not None and pos is not None:
                unreal = float(last - avg) * float(pos)
        except Exception:
            pass
        return {
            **d,
            "bot_name": bot.get("name"),
            "bot_symbol": bot.get("symbol"),
            "tp_target": tp_target,
            "unrealized_pnl": unreal,
            "last_price": snap.get("last_price"),
        }

    try:
        active_deals = [enrich_deal(d) for d in db.list_all_deals("OPEN", limit=200)]
    except Exception:
        active_deals = []

    try:
        closed_deals = [enrich_deal(d) for d in db.list_all_deals("CLOSED", limit=200)]
    except Exception:
        closed_deals = []

    return {
        "portfolio": _portfolio_snapshot(),
        "pnl": _build_pnl(),
        "bots": bots,
        "bot_rows": bot_rows,
        "deals_stats": deals_stats,
        "active_deals": active_deals,
        "closed_deals": closed_deals,
        "kraken_ready": bool(getattr(worker_api, "KRAKEN_READY", False)),
        "kraken_error": str(getattr(worker_api, "KRAKEN_ERROR", "") or ""),
    }


def _dashboard_context(request: Request) -> Dict[str, Any]:
    now = int(time_mod.time())
    cached = _DASH_CACHE.get("data")
    if cached and (now - int(_DASH_CACHE.get("ts") or 0)) <= _DASH_TTL_SEC:
        return {**_base_ctx(request), **cached}
    data = _dashboard_data()
    _DASH_CACHE["ts"] = now
    _DASH_CACHE["data"] = data
    return {**_base_ctx(request), **data}


@app.get("/", include_in_schema=False)
def ui_root(request: Request):
    """Redirect / to /explore so the site loads instantly (dashboard is slow)."""
    return RedirectResponse(url="/explore", status_code=302)


@app.get("/dashboard", include_in_schema=False)
def ui_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", _dashboard_context(request))


@app.get("/strategies", include_in_schema=False)
async def page_strategies(request: Request):
    return templates.TemplateResponse("strategies_leaderboard.html", _base_ctx(request))


@app.get("/explore", include_in_schema=False)
def ui_explore(request: Request):
    return templates.TemplateResponse("explore.html", _base_ctx(request))


@app.get("/safety", include_in_schema=False)
def ui_safety(request: Request):
    return templates.TemplateResponse("safety.html", _base_ctx(request))


@app.get("/setup-autopilot", include_in_schema=False)
def ui_setup_autopilot(request: Request):
    return templates.TemplateResponse("setup_autopilot.html", _base_ctx(request))


@app.get("/autopilot", include_in_schema=False)
def ui_autopilot(request: Request):
    return templates.TemplateResponse("autopilot_dashboard.html", _base_ctx(request))


@app.get("/journal", include_in_schema=False)
def ui_journal(request: Request):
    return templates.TemplateResponse("journal.html", _base_ctx(request))


@app.get("/analytics", include_in_schema=False)
def ui_analytics(request: Request):
    return templates.TemplateResponse("analytics.html", _base_ctx(request))


@app.get("/scenario-simulator", include_in_schema=False)
def ui_scenario_simulator(request: Request):
    return templates.TemplateResponse("scenario-simulator.html", _base_ctx(request))


@app.get("/bots", include_in_schema=False)
def ui_bots(request: Request):
    ctx = {
        **_base_ctx(request),
        "bots": _bots_snapshot(),
        "kraken_ready": bool(getattr(worker_api, "KRAKEN_READY", False)),
        "kraken_error": str(getattr(worker_api, "KRAKEN_ERROR", "") or ""),
    }
    return templates.TemplateResponse("bots.html", ctx)


@app.get("/bots/{bot_id}", include_in_schema=False)
def ui_bot(request: Request, bot_id: int):
    bot = db.get_bot(int(bot_id))
    if not bot:
        return RedirectResponse(url="/bots", status_code=302)

    # Lightweight snapshot for initial render; the page JS keeps it live.
    snap: Dict[str, Any] = {}
    try:
        snap_payload = _json_payload(worker_api.api_bot_status(int(bot_id)))  # type: ignore[misc]
        if isinstance(snap_payload, dict) and snap_payload.get("ok"):
            snap = snap_payload.get("snap", {}) or {}
    except Exception:
        snap = {}

    ctx = {
        **_base_ctx(request),
        "bot": bot,
        "snap": snap,
        "kraken_ready": bool(getattr(worker_api, "KRAKEN_READY", False)),
        "kraken_error": str(getattr(worker_api, "KRAKEN_ERROR", "") or ""),
    }
    return templates.TemplateResponse("bot.html", ctx)


@app.get("/dca", include_in_schema=False)
def ui_dca_dashboard(request: Request):
    bots = _bots_snapshot()
    bot_rows = []
    unrealized_total = 0.0
    runtime_map: Dict[int, Dict[str, Any]] = {}
    for b in bots or []:
        try:
            stats = db.bot_deal_stats(int(b.get("id")))
        except Exception:
            stats = {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
        snap = _bot_runtime(int(b.get("id")))
        runtime_map[int(b.get("id"))] = snap
        unrealized = 0.0
        try:
            avg = snap.get("avg_entry")
            last = snap.get("last_price")
            pos = snap.get("base_pos")
            if avg is not None and last is not None and pos is not None:
                unrealized = float(last - avg) * float(pos)
        except Exception:
            pass
        unrealized_total += float(unrealized or 0.0)
        bot_rows.append({**b, "stats": stats, "runtime": snap, "unrealized_pnl": unrealized})

    try:
        deals_stats = db.all_deal_stats()
    except Exception:
        deals_stats = {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
    deals_stats["unrealized_total"] = float(unrealized_total)

    bot_map = {int(b.get("id")): b for b in bots or []}

    def enrich_deal(d: Dict[str, Any]) -> Dict[str, Any]:
        bot_id = int(d.get("bot_id") or 0)
        bot = bot_map.get(bot_id, {})
        tp = float(bot.get("tp") or 0.0)
        entry = d.get("entry_avg")
        tp_target = (float(entry) * (1.0 + tp)) if entry is not None else None
        snap = runtime_map.get(bot_id, {})
        unreal = None
        try:
            avg = snap.get("avg_entry")
            last = snap.get("last_price")
            pos = snap.get("base_pos")
            if avg is not None and last is not None and pos is not None:
                unreal = float(last - avg) * float(pos)
        except Exception:
            pass
        return {
            **d,
            "bot_name": bot.get("name"),
            "bot_symbol": bot.get("symbol"),
            "tp_target": tp_target,
            "unrealized_pnl": unreal,
            "last_price": snap.get("last_price"),
        }

    try:
        active_deals = [enrich_deal(d) for d in db.list_all_deals("OPEN", limit=200)]
    except Exception:
        active_deals = []

    try:
        closed_deals = [enrich_deal(d) for d in db.list_all_deals("CLOSED", limit=200)]
    except Exception:
        closed_deals = []

    ctx = {
        **_base_ctx(request),
        "bots": bots,
        "bot_rows": bot_rows,
        "deals_stats": deals_stats,
        "active_deals": active_deals,
        "closed_deals": closed_deals,
        "kraken_ready": bool(getattr(worker_api, "KRAKEN_READY", False)),
        "kraken_error": str(getattr(worker_api, "KRAKEN_ERROR", "") or ""),
    }
    return templates.TemplateResponse("dca.html", ctx)


@app.get("/deals/{deal_id}", include_in_schema=False)
def ui_deal_detail(request: Request, deal_id: int):
    deal = db.get_deal(int(deal_id))
    if not deal:
        return RedirectResponse(url="/dca", status_code=302)

    bot = db.get_bot(int(deal.get("bot_id") or 0)) or {}
    snap = _bot_runtime(int(deal.get("bot_id") or 0))
    tp = float(bot.get("tp") or 0.0)
    entry = deal.get("entry_avg")
    tp_target = (float(entry) * (1.0 + tp)) if entry is not None else None
    unreal = None
    try:
        avg = snap.get("avg_entry")
        last = snap.get("last_price")
        pos = snap.get("base_pos")
        if avg is not None and last is not None and pos is not None:
            unreal = float(last - avg) * float(pos)
    except Exception:
        pass

    start_ts = int(deal.get("opened_at") or 0) - 60
    end_ts = int(deal.get("closed_at") or int(datetime.now(tz=timezone.utc).timestamp())) + 60
    timeline = []
    try:
        timeline = db.list_logs_window(int(deal.get("bot_id") or 0), start_ts, end_ts, limit=300)
    except Exception:
        timeline = []

    ctx = {
        **_base_ctx(request),
        "deal": deal,
        "bot": bot,
        "tp_target": tp_target,
        "unrealized_pnl": unreal,
        "last_price": snap.get("last_price"),
        "timeline": timeline,
        "kraken_ready": bool(getattr(worker_api, "KRAKEN_READY", False)),
        "kraken_error": str(getattr(worker_api, "KRAKEN_ERROR", "") or ""),
    }
    return templates.TemplateResponse("deal_detail.html", ctx)


# ----------------------------
# Static files
# ----------------------------
# If you already have /static folder (app.css, bot_chart.js, etc) this serves it.
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    # Already mounted
    pass


# ----------------------------
# Startup: optional auto-start enabled bots
# ----------------------------

@app.on_event("startup")
async def _auto_start_enabled_bots() -> None:
    """
    If you want bots to start automatically after a restart:
      - Set AUTO_START_BOTS=1 in your environment (optional).
    """
    import os

    if os.getenv("AUTO_START_BOTS", "").strip() != "1":
        return

    # Wait until worker_api has finished its startup.
    bm = getattr(worker_api, "bm", None)
    if bm is None:
        return

    for b in db.list_bots():
        try:
            if int(b.get("enabled", 0)) == 1:
                bm.start(int(b["id"]))
        except Exception:
            continue
