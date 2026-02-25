# app.py  (REPLACE ENTIRE FILE)  -- UI ONLY (no trading engine)
import os
import time
from typing import Any, Dict, Optional

import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from db import (
    init_db,
    list_bots,
    get_bot,
    list_logs,
    list_deals,
    pnl_summary,
    bot_deal_stats,
    all_deal_stats,
    list_all_deals,
    get_deal,
    list_logs_window,
    get_setting,
)


# =========================================================
# Minimal .env loader (no python-dotenv dependency)
# =========================================================
def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f.readlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("_load_env_file: %s", e)


_load_env_file(os.path.join(os.path.dirname(__file__), ".env"))
_load_env_file(".env")


# =========================================================
# Config
# =========================================================
# Worker base URL. Default matches your worker.py (127.0.0.1:9001).
WORKER_URL = os.getenv("WORKER_URL", "http://127.0.0.1:9001").rstrip("/")

# Optional worker API token; if you set WORKER_API_TOKEN in .env, app will forward it.
WORKER_API_TOKEN = os.getenv("WORKER_API_TOKEN", "").strip()

# Requests timeouts (1-60 seconds)
WORKER_TIMEOUT_SEC = max(1.0, min(60.0, float(os.getenv("WORKER_TIMEOUT_SEC", "8"))))


# =========================================================
# App init
# =========================================================
app = FastAPI()


def _check_worker_health() -> Dict[str, Any]:
    """Check worker connectivity. Returns health dict or error."""
    if not WORKER_URL.startswith(("http://", "https://")):
        return {"ok": False, "error": "WORKER_URL must start with http:// or https://"}
    try:
        r = requests.get(f"{WORKER_URL}/health", headers=_worker_headers(), timeout=5)
        return r.json() if r.ok else {"ok": False, "status": r.status_code}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/health")
def health() -> JSONResponse:
    """Health check for nginx/load balancer. 200 if worker reachable, 503 otherwise."""
    h = _check_worker_health()
    status = 200 if h.get("ok") else 503
    return _json(h, status_code=status)


templates = Jinja2Templates(directory="templates")
def _json(data: Any, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=data, status_code=status_code, headers={'Cache-Control': 'no-store'})


@app.post("/api/bots")
async def api_create_bot(request: Request):
    payload = await request.json()
    return _worker_json("POST", "/api/bots", body=payload)


@app.put("/api/bots/{bot_id}")
async def api_update_bot(bot_id: int, request: Request):
    payload = await request.json()
    return _worker_json("PUT", f"/api/bots/{int(bot_id)}", body=payload)


@app.get("/api/recommendations")
def api_recommendations_proxy(request: Request):
    """Forward all query params to worker (Explore uses market_type, signal, sort, volatility, regime, etc.)."""
    params = dict(request.query_params)
    return _worker_json("GET", "/api/recommendations", params=params)


@app.get("/api/recommendations/scan_status")
def api_recommendations_scan_status_proxy():
    return _worker_json("GET", "/api/recommendations/scan_status")


@app.get("/api/recommendations/performance")
def api_recommendations_performance_proxy(days: int = 30):
    return _worker_json("GET", "/api/recommendations/performance", params={"days": days})


@app.post("/api/recommendations/calibrate")
def api_recommendations_calibrate_proxy(window_days: int = 30):
    return _worker_json("POST", "/api/recommendations/calibrate", params={"window_days": window_days})


@app.get("/api/recommendations/{symbol}")
def api_recommendation_symbol_proxy(symbol: str, horizon: str = "short"):
    return _worker_json("GET", f"/api/recommendations/{symbol}", params={"horizon": horizon})


@app.post("/api/recommendations/{symbol}/create_bot")
async def api_recommendation_create_bot_proxy(symbol: str, request: Request):
    payload = await request.json()
    return _worker_json("POST", f"/api/recommendations/{symbol}/create_bot", body=payload)


@app.post("/api/recommendations/scan")
def api_recommendations_scan_proxy(horizon: str = "short"):
    return _worker_json("POST", "/api/recommendations/scan", params={"horizon": horizon})


@app.get("/api/opportunities/now")
def api_now_opportunities_proxy(request: Request):
    """Proxy for \"now opportunities\" radar endpoint."""
    params = dict(request.query_params)
    return _worker_json("GET", "/api/opportunities/now", params=params)


@app.get("/api/market/ticker")
def api_market_ticker_proxy(symbol: str):
    return _worker_json("GET", "/api/market/ticker", params={"symbol": symbol})


@app.get("/api/market/ohlcv")
def api_market_ohlcv_proxy(symbol: str, tf: str = "1h", limit: int = 500):
    return _worker_json("GET", "/api/market/ohlcv", params={"symbol": symbol, "tf": tf, "limit": limit})


@app.get("/api/market/overview")
def api_market_overview_proxy(quote: str = "USD", limit: int = 120):
    return _worker_json("GET", "/api/market/overview", params={"quote": quote, "limit": limit})


@app.get("/api/tax_optimization_suggestions")
def api_tax_suggestions_proxy(min_loss_pct: float = 5.0):
    return _worker_json("GET", "/api/tax_optimization_suggestions", params={"min_loss_pct": min_loss_pct})


@app.get("/api/portfolio/rebalance_suggestions")
def api_rebalance_proxy():
    return _worker_json("GET", "/api/portfolio/rebalance_suggestions")


@app.post("/api/orders/buy")
async def api_orders_buy_proxy(request: Request):
    payload = await request.json()
    return _worker_json("POST", "/api/orders/buy", body=payload)


@app.get("/api/bots/{bot_id}/logs")
def api_bot_logs(bot_id: int, limit: int = 200):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    safe_limit = int(max(1, min(int(limit), 2000)))
    return _json({"ok": True, "logs": list_logs(int(bot_id), limit=safe_limit)})


@app.get("/api/bots/{bot_id}/deals")
def api_bot_deals(bot_id: int, limit: int = 50):
    b = get_bot(int(bot_id))
    if not b:
        return _json({"ok": False, "error": "Bot not found"}, 404)

    safe_limit = int(max(1, min(int(limit), 1000)))
    return _json({"ok": True, "deals": list_deals(int(bot_id), limit=safe_limit)})


if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================================================
# Worker proxy helpers
# =========================================================
def _worker_headers() -> Dict[str, str]:
    h: Dict[str, str] = {}
    if WORKER_API_TOKEN:
        h["X-API-Key"] = WORKER_API_TOKEN
    return h


def _worker_json(method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Any = None) -> JSONResponse:
    """
    Make a request to worker_api and return the JSON as a JSONResponse.
    Converts worker errors into consistent responses so UI never hard-crashes.
    """
    url = f"{WORKER_URL}{path}"
    try:
        r = requests.request(
            method=method,
            url=url,
            params=params or None,
            json=body,
            headers=_worker_headers(),
            timeout=WORKER_TIMEOUT_SEC,
        )
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            data = r.json()
            return JSONResponse(data, status_code=r.status_code)
        # fallback: wrap text
        return JSONResponse(
            {"ok": False, "error": "worker_non_json", "status": r.status_code, "detail": r.text},
            status_code=502,
        )
    except requests.exceptions.RequestException as e:
        return JSONResponse(
            {"ok": False, "error": "worker_unreachable", "detail": str(e), "worker_url": WORKER_URL},
            status_code=502,
        )



@app.get("/api/bots/{bot_id}/stream")
def api_bot_stream(bot_id: int):
    """Proxy SSE stream from worker (/api/bots/{id}/stream)."""
    url = f"{WORKER_URL}/api/bots/{int(bot_id)}/stream"
    try:
        r = requests.get(url, headers=_worker_headers(), stream=True, timeout=WORKER_TIMEOUT_SEC)
    except requests.exceptions.RequestException as e:
        return _json({"ok": False, "error": "worker_unreachable", "detail": str(e)}, 502)

    if r.status_code != 200:
        try:
            return _json(r.json(), r.status_code)
        except Exception:
            return _json({"ok": False, "error": "stream_failed", "status": r.status_code, "detail": r.text}, 502)

    def gen():
        try:
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        finally:
            try:
                r.close()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/bots/{bot_id}/logstream")
def api_bot_logstream(bot_id: int):
    """Proxy SSE log stream from worker (/api/bots/{id}/logstream)."""
    url = f"{WORKER_URL}/api/bots/{int(bot_id)}/logstream"
    try:
        r = requests.get(url, headers=_worker_headers(), stream=True, timeout=WORKER_TIMEOUT_SEC)
    except requests.exceptions.RequestException as e:
        return _json({"ok": False, "error": "worker_unreachable", "detail": str(e)}, 502)

    if r.status_code != 200:
        try:
            return _json(r.json(), r.status_code)
        except Exception:
            return _json({"ok": False, "error": "logstream_failed", "status": r.status_code, "detail": r.text}, 502)

    def gen():
        try:
            for chunk in r.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        finally:
            try:
                r.close()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

def _worker_raw(method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Response:
    """
    For streaming endpoints if needed later. For now, not used by templates.
    """
    url = f"{WORKER_URL}{path}"
    try:
        r = requests.request(
            method=method,
            url=url,
            params=params or None,
            headers=_worker_headers(),
            timeout=WORKER_TIMEOUT_SEC,
        )
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))
    except requests.exceptions.RequestException as e:
        return Response(content=str(e), status_code=502, media_type="text/plain")


def _safe_worker_health() -> Dict[str, Any]:
    """
    Used by templates so dashboard never breaks.
    """
    resp = _worker_json("GET", "/health")
    try:
        payload = resp.body
        # resp.body is bytes; decode safely
        import json
        data = json.loads(payload.decode("utf-8"))
        return data if isinstance(data, dict) else {"ok": False, "error": "bad_health_payload"}
    except Exception:
        return {"ok": False, "error": "health_parse_failed"}


# =========================================================
# UI Routes (templates only)
# =========================================================
def _build_dashboard_context(request: Request) -> Dict[str, Any]:
    # Pull everything from worker in a stable way
    health = _safe_worker_health()
    kraken_ready = bool(health.get("kraken_ready"))
    kraken_error = str(health.get("kraken_error") or "")
    alpaca_ready = bool(health.get("alpaca_paper_ready") or health.get("alpaca_live_ready"))
    alpaca_error = str(health.get("alpaca_error") or "")

    # portfolio + bots are worker-owned now
    port_resp = _worker_json("GET", "/api/portfolio")
    bots_resp = _worker_json("GET", "/api/bots")
    pnl_today_resp = _worker_json("GET", "/api/pnl")

    portfolio = {"total_usd": 0.0, "holdings": [], "error": kraken_error or "Worker not ready"}
    pnl = {"today": 0.0, "total": 0.0}
    bots = []

    try:
        import json
        p = json.loads(port_resp.body.decode("utf-8"))
        if isinstance(p, dict) and p.get("ok"):
            portfolio = (p.get("portfolio") or p.get("snapshot") or portfolio)
    except Exception:
        pass

    try:
        import json
        b = json.loads(bots_resp.body.decode("utf-8"))
        if isinstance(b, dict) and b.get("ok"):
            bots = b.get("bots") or []
    except Exception:
        pass

    try:
        import json
        pj = json.loads(pnl_today_resp.body.decode("utf-8"))
        if isinstance(pj, dict) and pj.get("ok"):
            today = pj.get("today") or {}
            total = pj.get("total") or {}
            pnl["today"] = float(today.get("realized", 0.0) or 0.0) if isinstance(today, dict) else 0.0
            pnl["total"] = float(total.get("realized", 0.0) or 0.0) if isinstance(total, dict) else 0.0
    except Exception:
        pass

    bot_rows = []
    unrealized_total = 0.0
    runtime_map: Dict[int, Dict[str, Any]] = {}
    for b in bots or []:
        try:
            stats = bot_deal_stats(int(b.get("id")))
        except Exception:
            stats = {"open_count": 0, "closed_count": 0, "realized_total": 0.0}
        snap = {}
        unrealized = 0.0
        try:
            import json
            sresp = _worker_json("GET", f"/api/bots/{int(b.get('id'))}/status")
            sdata = json.loads(sresp.body.decode("utf-8"))
            if isinstance(sdata, dict) and sdata.get("ok"):
                snap = sdata.get("snap") or {}
                runtime_map[int(b.get("id"))] = snap
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
        deals_stats = all_deal_stats()
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
        active_deals = [enrich_deal(d) for d in list_all_deals("OPEN", limit=200)]
    except Exception:
        active_deals = []

    try:
        closed_deals = [enrich_deal(d) for d in list_all_deals("CLOSED", limit=200)]
    except Exception:
        closed_deals = []

    perf_by_strat_regime = {}
    try:
        for d in closed_deals:
            strat = d.get("entry_strategy") or "unknown"
            reg = d.get("entry_regime") or "unknown"
            key = (strat, reg)
            if key not in perf_by_strat_regime:
                perf_by_strat_regime[key] = {"count": 0, "pnl": 0.0}
            perf_by_strat_regime[key]["count"] += 1
            perf_by_strat_regime[key]["pnl"] += float(d.get("realized_pnl_quote") or 0.0)
    except Exception:
        perf_by_strat_regime = {}

    pause_state = False
    pause_until = 0
    try:
        pause_state = str(get_setting("global_pause", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
        pause_until = int(get_setting("global_pause_until", "0") or 0)
    except Exception:
        pause_state = False

    return {
            "request": request,
            "portfolio": portfolio,
            "pnl": pnl,
            "bots": bots,
        "bot_rows": bot_rows,
        "deals_stats": deals_stats,
        "active_deals": active_deals,
        "closed_deals": closed_deals,
            "kraken_ready": kraken_ready,
            "kraken_error": kraken_error,
            "alpaca_ready": alpaca_ready,
            "alpaca_error": alpaca_error,
            "pause_state": pause_state,
            "pause_until": pause_until,
            "perf_by_strat_regime": perf_by_strat_regime,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", _build_dashboard_context(request))


@app.get("/dca", response_class=HTMLResponse)
def dca_dashboard(request: Request):
    return templates.TemplateResponse("dca.html", _build_dashboard_context(request))


@app.get("/explore", response_class=HTMLResponse)
def explore_page(request: Request):
    ctx = _build_dashboard_context(request)
    return templates.TemplateResponse("explore.html", ctx)


@app.get("/bots", response_class=HTMLResponse)
def bots_page(request: Request):
    health = _safe_worker_health()
    kraken_ready = bool(health.get("kraken_ready"))
    kraken_error = str(health.get("kraken_error") or "")

    bots_resp = _worker_json("GET", "/api/bots")
    bots = []
    try:
        import json
        b = json.loads(bots_resp.body.decode("utf-8"))
        if isinstance(b, dict) and b.get("ok"):
            bots = b.get("bots") or []
    except Exception:
        pass

    return templates.TemplateResponse(
        "bots.html",
        {
            "request": request,
            "bots": bots,
            "kraken_ready": kraken_ready,
            "kraken_error": kraken_error,
        },
    )


@app.get("/bots/{bot_id}", response_class=HTMLResponse)
def bot_page(bot_id: int, request: Request):
    health = _safe_worker_health()
    kraken_ready = bool(health.get("kraken_ready"))
    kraken_error = str(health.get("kraken_error") or "")

    bot_resp = _worker_json("GET", f"/api/bots/{int(bot_id)}")
    bot = None
    try:
        import json
        b = json.loads(bot_resp.body.decode("utf-8"))
        if isinstance(b, dict) and b.get("ok"):
            bot = b.get("bot")
    except Exception:
        pass

    if not bot:
        return templates.TemplateResponse(
            "layout.html",
            {"request": request, "content": "Bot not found"},
            status_code=404,
        )

    return templates.TemplateResponse(
        "bot.html",
        {
            "request": request,
            "bot": bot,
            "kraken_ready": kraken_ready,
            "kraken_error": kraken_error,
        },
    )


@app.get("/deals/{deal_id}", response_class=HTMLResponse)
def deal_detail(deal_id: int, request: Request):
    deal = get_deal(int(deal_id))
    if not deal:
        return templates.TemplateResponse(
            "layout.html",
            {"request": request},
            status_code=404,
        )

    bot = get_bot(int(deal.get("bot_id") or 0)) or {}
    tp = float(bot.get("tp") or 0.0)
    entry = deal.get("entry_avg")
    tp_target = (float(entry) * (1.0 + tp)) if entry is not None else None

    snap = {}
    unreal = None
    try:
        import json
        sresp = _worker_json("GET", f"/api/bots/{int(deal.get('bot_id') or 0)}/status")
        sdata = json.loads(sresp.body.decode("utf-8"))
        if isinstance(sdata, dict) and sdata.get("ok"):
            snap = sdata.get("snap") or {}
            avg = snap.get("avg_entry")
            last = snap.get("last_price")
            pos = snap.get("base_pos")
            if avg is not None and last is not None and pos is not None:
                unreal = float(last - avg) * float(pos)
    except Exception:
        pass

    start_ts = int(deal.get("opened_at") or 0) - 60
    end_ts = int(deal.get("closed_at") or int(time.time())) + 60
    timeline = []
    try:
        timeline = list_logs_window(int(deal.get("bot_id") or 0), start_ts, end_ts, limit=300)
    except Exception:
        timeline = []

    health = _safe_worker_health()
    kraken_ready = bool(health.get("kraken_ready"))
    kraken_error = str(health.get("kraken_error") or "")

    return templates.TemplateResponse(
        "deal_detail.html",
        {
            "request": request,
            "deal": deal,
            "bot": bot,
            "tp_target": tp_target,
            "unrealized_pnl": unreal,
            "last_price": snap.get("last_price"),
            "timeline": timeline,
            "kraken_ready": kraken_ready,
            "kraken_error": kraken_error,
        },
    )


# =========================================================
# API Proxy (keep same paths your templates call)
# =========================================================
@app.get("/api/health")
def api_health():
    return _worker_json("GET", "/health")


@app.get("/api/symbols")
def api_symbols(quote: str = "USD"):
    return _worker_json("GET", "/api/symbols", params={"quote": quote})


@app.get("/api/prices")
def api_prices(symbols: str = ""):
    return _worker_json("GET", "/api/prices", params={"symbols": symbols})


@app.get("/api/portfolio")
def api_portfolio():
    return _worker_json("GET", "/api/portfolio")


@app.get("/api/bots")
def api_bots():
    return _worker_json("GET", "/api/bots")


@app.delete("/api/bots/{bot_id}")
def api_delete_bot(bot_id: int):
    return _worker_json("DELETE", f"/api/bots/{int(bot_id)}")


@app.get("/api/bots/{bot_id}")
def api_bot(bot_id: int):
    return _worker_json("GET", f"/api/bots/{int(bot_id)}")


@app.get("/api/bots/{bot_id}/status")
def api_bot_status(bot_id: int):
    return _worker_json("GET", f"/api/bots/{int(bot_id)}/status")


@app.post("/api/bots/{bot_id}/start")
def api_bot_start(bot_id: int):
    return _worker_json("POST", f"/api/bots/{int(bot_id)}/start")


@app.post("/api/bots/{bot_id}/stop")
def api_bot_stop(bot_id: int):
    return _worker_json("POST", f"/api/bots/{int(bot_id)}/stop")


@app.get("/api/bots/{bot_id}/live")
def api_bot_live(bot_id: int, logs_limit: int = 150, deals_limit: int = 30):
    return _worker_json(
        "GET",
        f"/api/bots/{int(bot_id)}/live",
        params={"logs_limit": logs_limit, "deals_limit": deals_limit},
    )


@app.get("/api/bots/{bot_id}/ohlc")
def api_bot_ohlc(bot_id: int, timeframe: str = "5m", limit: int = 200):
    return _worker_json(
        "GET",
        f"/api/bots/{int(bot_id)}/ohlc",
        params={"timeframe": timeframe, "limit": limit},
    )


@app.get("/api/bots/{bot_id}/markers")
def api_bot_markers(bot_id: int, timeframe: str = "5m", limit: int = 250):
    return _worker_json(
        "GET",
        f"/api/bots/{int(bot_id)}/markers",
        params={"timeframe": timeframe, "limit": limit},
    )


# =========================================================
# Autopilot Dashboard proxies (fix: dashboard was 404 without these)
# =========================================================
@app.get("/api/autopilot/config")
def api_autopilot_config_get():
    return _worker_json("GET", "/api/autopilot/config")


@app.post("/api/autopilot/config")
async def api_autopilot_config_save(request: Request):
    body = await request.json()
    return _worker_json("POST", "/api/autopilot/config", body=body)


@app.post("/api/autopilot/config/update")
async def api_autopilot_config_update(request: Request):
    """Dashboard edit: merge body into autopilot config (e.g. budget per bot, total capital)."""
    body = await request.json()
    return _worker_json("POST", "/api/autopilot/config/update", body=body)


@app.get("/api/autopilot/status")
def api_autopilot_status():
    return _worker_json("GET", "/api/autopilot/status")


@app.get("/api/autopilot/positions")
def api_autopilot_positions():
    return _worker_json("GET", "/api/autopilot/positions")


@app.post("/api/autopilot/toggle")
def api_autopilot_toggle():
    return _worker_json("POST", "/api/autopilot/toggle")


@app.post("/api/autopilot/start")
def api_autopilot_start():
    return _worker_json("POST", "/api/autopilot/start")


@app.post("/api/autopilot/stop")
def api_autopilot_stop():
    return _worker_json("POST", "/api/autopilot/stop")


@app.post("/api/autopilot/run")
def api_autopilot_run():
    return _worker_json("POST", "/api/autopilot/run")


@app.get("/api/autopilot/top")
def api_autopilot_top():
    return _worker_json("GET", "/api/autopilot/top")


@app.get("/api/autopilot/watchlist")
def api_autopilot_watchlist():
    return _worker_json("GET", "/api/autopilot/watchlist")


@app.post("/api/autopilot/setup")
async def api_autopilot_setup(request: Request):
    body = await request.json()
    return _worker_json("POST", "/api/autopilot/setup", body=body)


@app.post("/api/autopilot/capital/add")
async def api_autopilot_capital_add(request: Request):
    body = await request.json()
    return _worker_json("POST", "/api/autopilot/capital/add", body=body)


@app.get("/api/portfolio/performance")
def api_portfolio_performance(timeframe: str = "1D"):
    return _worker_json("GET", "/api/portfolio/performance", params={"timeframe": timeframe})
