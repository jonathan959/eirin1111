"""
one_server_v2.py

ONE server = API + UI together.

What you do:
  1) Put this file next to your other .py files (do NOT delete anything).
  2) Run:  uvicorn one_server_v2:app --reload --port 8000
  3) Open: http://127.0.0.1:8000/

What it does:
- Imports your existing worker_api.py (so it reuses all your bot logic + API routes).
- Adds simple built-in web pages (no templates needed).
- Makes logs truly live (no refresh) using SSE: /api/bots/{id}/logstream
- Keeps "Enabled" and "Running" in sync:
    Enable -> saves enabled=1 AND starts the bot
    Disable -> saves enabled=0 AND stops the bot
    If enabled=1 but running=0 after opening page -> it tries start once.

Important:
- This does NOT replace your project. It's just a "one-command runner".
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------
# Stable DB path
# ---------------------------
_THIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("BOT_DB_PATH", str(_THIS_DIR / "botdb.sqlite3"))

# Import the existing bot engine + API (this defines `app` + all /api/* routes)
import worker_api  # noqa: E402

# Re-export the SAME FastAPI instance => ONE server.
app = worker_api.app

# Templates + static for Autopilot Dashboard (same as one_server so live server has /autopilot)
_templates_dir = _THIS_DIR / "templates"
_static_dir = _THIS_DIR / "static"
if _templates_dir.is_dir():
    templates = Jinja2Templates(directory=str(_templates_dir))
else:
    templates = None
if _static_dir.is_dir():
    try:
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
    except Exception:
        pass


def _base_ctx(request: Request) -> Dict[str, Any]:
    """Minimal context for template pages (Autopilot Dashboard, Setup Autopilot)."""
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


@app.on_event("startup")
def _startup_banner():
    import sys
    print("AI Bot (one_server_v2) ready - API + UI on port 8000", flush=True)
    sys.stdout.flush()


# ---------------------------
# DB helpers (read + small updates)
# ---------------------------
def _db_path() -> str:
    return os.getenv("BOT_DB_PATH", str(_THIS_DIR / "botdb.sqlite3"))


def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path())
    con.row_factory = sqlite3.Row
    return con


def _list_bots() -> list[dict]:
    con = _conn()
    try:
        rows = con.execute("SELECT * FROM bots ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def _get_bot(bot_id: int) -> Optional[dict]:
    con = _conn()
    try:
        row = con.execute("SELECT * FROM bots WHERE id=?", (int(bot_id),)).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def _html_page(title: str, body: str) -> HTMLResponse:
    css = """
    body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;margin:0;background:#0b1020;color:#e5e7eb;}
    a{color:#93c5fd;text-decoration:none} a:hover{text-decoration:underline}
    .wrap{max-width:1100px;margin:0 auto;padding:18px;}
    .top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:14px;}
    .h1{font-size:22px;font-weight:800;}
    .muted{color:#94a3b8;}
    .grid{display:grid;gap:12px;}
    @media(min-width:900px){.grid-2{grid-template-columns:1.2fr .8fr;}.grid-3{grid-template-columns:repeat(3,1fr);}}
    .card{background:rgba(255,255,255,.04);border:1px solid rgba(148,163,184,.15);border-radius:14px;padding:14px;}
    .btn{background:rgba(255,255,255,.08);border:1px solid rgba(148,163,184,.2);color:#e5e7eb;padding:8px 12px;border-radius:10px;cursor:pointer;}
    .btn:hover{background:rgba(255,255,255,.12);}
    .btn-primary{background:rgba(139,92,246,.25);border-color:rgba(139,92,246,.55);}
    .row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
    .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}
    table{width:100%;border-collapse:collapse;}
    th,td{padding:10px 8px;border-bottom:1px solid rgba(148,163,184,.12);text-align:left;vertical-align:top;}
    .tag{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid rgba(148,163,184,.25);color:#cbd5e1;}
    .tag-ok{border-color:rgba(34,197,94,.35);color:#86efac;}
    .tag-bad{border-color:rgba(239,68,68,.35);color:#fca5a5;}
    .log{height:360px;overflow:auto;background:rgba(0,0,0,.25);border:1px solid rgba(148,163,184,.18);border-radius:12px;padding:10px;}
    .logline{white-space:pre-wrap;margin:0 0 6px 0;}
    .err{color:#fca5a5;}
    """
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<style>{css}</style>
<script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
</head><body>{body}</body></html>"""
    return HTMLResponse(html)


# ---------------------------
# UI routes
# ---------------------------
def _live_trading_enabled() -> bool:
    """LIVE-HARDENED: True if live trading is allowed (env gate)."""
    v = os.getenv("ALLOW_LIVE_TRADING", "0") or os.getenv("LIVE_TRADING_ENABLED", "0")
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


@app.get("/autopilot", include_in_schema=False)
def ui_autopilot(request: Request):
    """Autopilot Dashboard (full UI with sidebar). Works on live server after deploy."""
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found. Deploy with templates folder.")
    return templates.TemplateResponse("autopilot_dashboard.html", _base_ctx(request))


@app.get("/setup-autopilot", include_in_schema=False)
def ui_setup_autopilot(request: Request):
    """Setup Autopilot page."""
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found. Deploy with templates folder.")
    return templates.TemplateResponse("setup_autopilot.html", _base_ctx(request))


@app.get("/safety", include_in_schema=False)
def ui_safety(request: Request):
    """Safety page: kill switch, pause, and live-trading checklist."""
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found. Deploy with templates folder.")
    return templates.TemplateResponse("safety.html", _base_ctx(request))


# --- Template-based pages (rich UI with sidebar) ---
# These use Jinja2 templates from /templates/ for the full dashboard experience.

@app.get("/dashboard", include_in_schema=False)
def ui_dashboard_v2(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found.")
    try:
        from one_server import _dashboard_context
        return templates.TemplateResponse("dashboard.html", _dashboard_context(request))
    except Exception:
        return templates.TemplateResponse("dashboard.html", _base_ctx(request))


@app.get("/dca", include_in_schema=False)
def ui_dca_v2(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found.")
    try:
        from one_server import ui_dca_dashboard
        return ui_dca_dashboard(request)
    except Exception:
        return templates.TemplateResponse("dca.html", _base_ctx(request))


@app.get("/explore", include_in_schema=False)
def ui_explore_v2(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found.")
    return templates.TemplateResponse("explore.html", _base_ctx(request))


@app.get("/analytics", include_in_schema=False)
def ui_analytics_v2(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found.")
    return templates.TemplateResponse("analytics.html", _base_ctx(request))


@app.get("/journal", include_in_schema=False)
def ui_journal_v2(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found.")
    return templates.TemplateResponse("trade_journal.html", _base_ctx(request))


@app.get("/strategies", include_in_schema=False)
@app.get("/strategies-leaderboard", include_in_schema=False)
def ui_strategies_v2(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found.")
    return templates.TemplateResponse("strategies_leaderboard.html", _base_ctx(request))


@app.get("/scenario", include_in_schema=False)
def ui_scenario_v2(request: Request):
    if templates is None:
        raise HTTPException(status_code=503, detail="Templates not found.")
    return templates.TemplateResponse("scenario_simulator.html", _base_ctx(request))


@app.get("/", response_class=HTMLResponse)
def ui_home():
    bots = _list_bots()
    kraken_ready = bool(getattr(worker_api, "KRAKEN_READY", False))
    kraken_error = getattr(worker_api, "KRAKEN_ERROR", "") or ""
    any_live = _live_trading_enabled() and any(int(b.get("dry_run") or 1) == 0 for b in bots)
    live_banner = '<div class="card" style="background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.4); margin-bottom:12px;"><span style="color:#fca5a5; font-weight:700;">LIVE TRADING</span> — Real orders enabled. Some bots may be placing live orders.</div>' if any_live else ''
    body = f"""
    <div class="wrap">
      {live_banner}
      <div class="top">
        <div>
          <div class="h1">AI Bot Dashboard</div>
          <div class="muted">One server mode • API + UI together</div>
        </div>
        <div class="row">
          <a class="btn" href="/autopilot">Autopilot</a>
          <a class="btn" href="/bots">Bots</a>
          <a class="btn" href="/safety">Safety</a>
          <a class="btn" href="/health">Health</a>
        </div>
      </div>

      <div class="grid grid-3">
        <div class="card">
          <div class="muted">Kraken</div>
          <div class="row" style="margin-top:8px;">
            <span class="tag {'tag-ok' if kraken_ready else 'tag-bad'}">{'READY' if kraken_ready else 'NOT READY'}</span>
            <span class="mono muted" style="font-size:12px;">{kraken_error}</span>
          </div>
        </div>
        <div class="card">
          <div class="muted">Bots</div>
          <div class="h1" style="margin-top:6px;">{len(bots)}</div>
        </div>
        <div class="card">
          <div class="muted">Time</div>
          <div class="mono" style="margin-top:8px;">{int(time.time())}</div>
        </div>
      </div>

      <div class="card" style="margin-top:12px;">
        <div class="row" style="justify-content:space-between;">
          <div class="h1" style="font-size:18px;">Recent bots</div>
          <a class="btn" href="/bots">View all</a>
        </div>
        <div style="margin-top:10px;">
          {"" if bots else '<div class="muted">No bots yet. Create one via the API: POST /api/bots</div>'}
          {"".join(
            f'<div class="row" style="justify-content:space-between; padding:10px 0; border-bottom:1px solid rgba(148,163,184,.12);">'
            f'<div><div><a href="/bots/{b["id"]}"><b>{b.get("name","Bot")}</b></a> '
            f'<span class="muted mono">#{b["id"]}</span></div>'
            f'<div class="muted mono">{(b.get("symbol") or "").upper()} • enabled={int(b.get("enabled") or 0)} • dry_run={int(b.get("dry_run") or 0)}</div></div>'
            f'<a class="btn" href="/bots/{b["id"]}">Open</a>'
            f'</div>'
            for b in bots[:5]
          )}
        </div>
      </div>
    </div>
    """
    return _html_page("AI Bot Dashboard", body)


@app.get("/bots", response_class=HTMLResponse)
def ui_bots():
    bots = _list_bots()
    rows = ""
    for b in bots:
        rows += f"""
        <tr>
          <td><a href="/bots/{b["id"]}"><b>{b.get("name","Bot")}</b></a><div class="muted mono">#{b["id"]}</div></td>
          <td class="mono">{(b.get("symbol") or "").upper()}</td>
          <td class="mono">{int(b.get("enabled") or 0)}</td>
          <td class="mono">{int(b.get("dry_run") or 0)}</td>
        </tr>
        """
    body = f"""
    <div class="wrap">
      <div class="top">
        <div>
          <div class="h1">Bots</div>
          <div class="muted">Click a bot to see live status, live logs, and candles</div>
        </div>
        <div class="row">
          <a class="btn" href="/">Dashboard</a>
          <a class="btn" href="/health">Health</a>
        </div>
      </div>

      <div class="card">
        <table>
          <thead><tr><th>Name</th><th>Symbol</th><th>Enabled</th><th>Dry Run</th></tr></thead>
          <tbody>
            {rows if rows else '<tr><td colspan="4" class="muted">No bots in database.</td></tr>'}
          </tbody>
        </table>
      </div>

      <div class="card" style="margin-top:12px;">
        <div class="muted">Create a bot</div>
        <div class="mono" style="margin-top:6px;">POST /api/bots</div>
        <div class="mono muted" style="margin-top:6px; font-size:12px;">
          Example JSON: {{ "name":"MyBot","symbol":"BTC/USD","enabled":1,"dry_run":1,"strategy":"scalp" }}
        </div>
      </div>
    </div>
    """
    return _html_page("Bots", body)


@app.get("/bots/{bot_id}", response_class=HTMLResponse)
def ui_bot(bot_id: int):
    b = _get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")

    kraken_ready = bool(getattr(worker_api, "KRAKEN_READY", False))
    kraken_error = getattr(worker_api, "KRAKEN_ERROR", "") or ""

    enabled = int(b.get("enabled") or 0)

    js = """
    <script>
      const BOT_ID = __BOT_ID__;
      window.__ENABLED__ = __ENABLED__;

      function $(id){ return document.getElementById(id); }

      function setStatus(snap){
        const running = !!snap?.running;
        $("statusText").textContent = running ? "RUNNING" : "STOPPED";
        $("tagRun").textContent = running ? "RUNNING" : "STOPPED";
        $("tagRun").className = "tag " + (running ? "tag-ok" : "tag-bad");
        const ev = snap?.last_event || "—";
        const px = (snap?.last_price === null || snap?.last_price === undefined) ? "—" : String(snap.last_price);
        const en = (window.__ENABLED__ === 1) ? "enabled=1" : "enabled=0";
        $("statusSub").textContent = `event=${ev} • last_price=${px} • ${en}`;
      }

      async function post(url){
        const r = await fetch(url, { method:"POST", cache:"no-store" });
        const j = await r.json().catch(()=> ({}));
        if(!r.ok) throw new Error(j.detail || j.error || (r.status+" "+r.statusText));
        return j;
      }

      async function putJSON(url, payload){
        const r = await fetch(url, {
          method:"PUT",
          headers: { "Content-Type":"application/json" },
          body: JSON.stringify(payload || {}),
          cache:"no-store"
        });
        const j = await r.json().catch(()=> ({}));
        if(!r.ok) throw new Error(j.detail || j.error || (r.status+" "+r.statusText));
        return j;
      }

      async function setEnabled(enabled){
        const j = await putJSON(`/api/bots/${BOT_ID}`, { enabled: enabled ? 1 : 0 });
        window.__ENABLED__ = enabled ? 1 : 0;

        // This is the "SYNC" part:
        if(enabled){
          try{ await post(`/api/bots/${BOT_ID}/start`); }catch(e){ $("statusSub").textContent = "Enabled, but start failed: " + e.message; }
        }else{
          try{ await post(`/api/bots/${BOT_ID}/stop`); }catch(_){}
        }
        return j;
      }

      $("btnStart").addEventListener("click", async () => {
        $("statusSub").textContent = "Starting...";
        try{ await post(`/api/bots/${BOT_ID}/start`); }catch(e){ $("statusSub").textContent = "Start failed: " + e.message; }
      });

      $("btnStop").addEventListener("click", async () => {
        $("statusSub").textContent = "Stopping...";
        try{ await post(`/api/bots/${BOT_ID}/stop`); }catch(e){ $("statusSub").textContent = "Stop failed: " + e.message; }
      });

      $("btnEnable").addEventListener("click", async () => { $("statusSub").textContent="Enabling..."; setEnabled(true).catch(e => $("statusSub").textContent = e.message); });
      $("btnDisable").addEventListener("click", async () => { $("statusSub").textContent="Disabling..."; setEnabled(false).catch(e => $("statusSub").textContent = e.message); });

      // -------- Status stream (SSE) --------
      (function statusStream(){
        let autoStarted = false;

        try{
          const es = new EventSource(`/api/bots/${BOT_ID}/stream`);
          es.onmessage = (ev) => {
            try{
              const payload = JSON.parse(ev.data || "{}");
              if(payload?.snap){
                setStatus(payload.snap);

                // If enabled=1 but running=0, start once automatically.
                if(!autoStarted && window.__ENABLED__===1 && payload.snap.running===false){
                  autoStarted = true;
                  post(`/api/bots/${BOT_ID}/start`).catch(()=>{});
                }
              }
            }catch(_){}
          };
        }catch(e){
          $("statusSub").textContent = "Live status not supported in this browser.";
        }
      })();

      // -------- Live log stream (SSE, no refresh) --------
      (function logStream(){
        const box = $("logBox");
        function addLine(o){
          const d = new Date((o.ts||0) * 1000);
          const t = d.toLocaleTimeString();
          const line = `[${t}] ${String(o.level||"INFO").padEnd(5)} ${o.message||""}`;
          const p = document.createElement("div");
          p.className = "logline";
          p.textContent = line;
          box.appendChild(p);
          while(box.children.length > 400) box.removeChild(box.firstChild);
          box.scrollTop = box.scrollHeight;
        }
        try{
          const es = new EventSource(`/api/bots/${BOT_ID}/logstream`);
          es.onmessage = (ev) => {
            try{
              const payload = JSON.parse(ev.data || "{}");
              if(payload?.log) addLine(payload.log);
            }catch(_){}
          };
        }catch(e){
          const p = document.createElement("div");
          p.className="logline err";
          p.textContent="Live logs not supported.";
          box.appendChild(p);
        }
      })();

      // -------- Candles (LightweightCharts) --------
      (function candleChart(){
        const el = $("tvChart");
        const err = $("chartErr");
        const tf = $("tf");
        const btn = $("btnChart");

        function showErr(msg){
          err.style.display = msg ? "block" : "none";
          err.textContent = msg || "";
        }

        const chart = LightweightCharts.createChart(el, {
          width: el.clientWidth || 900,
          height: 380,
          layout: { background: { type:"solid", color:"transparent" }, textColor:"#cbd5e1" },
          grid: { vertLines: { color:"rgba(148,163,184,.12)" }, horzLines: { color:"rgba(148,163,184,.12)" } },
          timeScale: { timeVisible:true, secondsVisible:false, borderColor:"rgba(148,163,184,.2)" },
          rightPriceScale: { borderColor:"rgba(148,163,184,.2)" },
        });
        const series = chart.addCandlestickSeries();

        function resize(){
          try{ chart.applyOptions({ width: el.clientWidth || 900 }); }catch(_){}
        }
        window.addEventListener("resize", resize);

        async function load(){
          showErr("");
          const timeframe = tf.value || "5m";
          const r = await fetch(`/api/bots/${BOT_ID}/ohlc?timeframe=${encodeURIComponent(timeframe)}&limit=500`, { cache:"no-store" });
          const j = await r.json().catch(()=> ({}));
          if(!r.ok) throw new Error(j.error || j.detail || (r.status+" "+r.statusText));
          const candles = Array.isArray(j.candles) ? j.candles : [];
          series.setData(candles);
          try{
            const mr = await fetch(`/api/bots/${BOT_ID}/markers?timeframe=${encodeURIComponent(timeframe)}&limit=250`, { cache:"no-store" });
            const mj = await mr.json().catch(()=> ({}));
            if(mr.ok && Array.isArray(mj.markers)) series.setMarkers(mj.markers);
          }catch(_){}
          chart.timeScale().fitContent();
        }

        btn.addEventListener("click", () => load().catch(e => showErr(e.message || String(e))));
        tf.addEventListener("change", () => load().catch(e => showErr(e.message || String(e))));

        load().catch(e => showErr(e.message || String(e)));
        setInterval(() => load().catch(()=>{}), 15000);
      })();
    </script>
    """
    js = js.replace("__BOT_ID__", str(int(b["id"]))).replace("__ENABLED__", str(enabled))

    is_live = _live_trading_enabled() and int(b.get("dry_run") or 1) == 0
    live_banner = '<div class="card" style="background:rgba(239,68,68,.12); border-color:rgba(239,68,68,.4); margin-bottom:12px;"><span style="color:#fca5a5; font-weight:700;">LIVE TRADING</span> — This bot places real orders.</div>' if is_live else ''
    body = f"""
    <div class="wrap">
      {live_banner}
      <div class="top">
        <div>
          <div class="h1">{b.get("name","Bot")} <span class="muted mono">#{b["id"]}</span></div>
          <div class="muted mono">{(b.get("symbol") or "").upper()} • dry_run={int(b.get("dry_run") or 0)} • enabled={enabled}</div>
          <div class="row" style="margin-top:8px;">
            <span class="tag {'tag-ok' if kraken_ready else 'tag-bad'}">Kraken {'READY' if kraken_ready else 'NOT READY'}</span>
            <span class="mono muted" style="font-size:12px;">{kraken_error}</span>
          </div>
        </div>
        <div class="row">
          <button class="btn btn-primary" id="btnStart">Start</button>
          <button class="btn" id="btnStop">Stop</button>
          <button class="btn" id="btnEnable">Enable</button>
          <button class="btn" id="btnDisable">Disable</button>
          <a class="btn" href="/bots">Back</a>
        </div>
      </div>

      <div class="grid grid-2">
        <div class="card">
          <div class="row" style="justify-content:space-between;">
            <div>
              <div class="muted">Candles</div>
              <div class="muted mono" style="font-size:12px;">Refreshes every 15 seconds</div>
            </div>
            <div class="row">
              <select id="tf" class="btn">
                <option value="1m">1m</option>
                <option value="5m" selected>5m</option>
                <option value="15m">15m</option>
                <option value="1h">1h</option>
                <option value="4h">4h</option>
                <option value="1d">1d</option>
              </select>
              <button class="btn" id="btnChart">Refresh</button>
            </div>
          </div>
          <div id="chartErr" class="err mono" style="margin-top:10px; display:none;"></div>
          <div id="tvChart" style="height:380px; margin-top:10px;"></div>
        </div>

        <div class="card">
          <div class="row" style="justify-content:space-between;">
            <div>
              <div class="muted">Live status</div>
              <div class="h1" id="statusText" style="font-size:18px; margin-top:6px;">...</div>
              <div class="muted mono" id="statusSub" style="font-size:12px; margin-top:6px;"></div>
            </div>
            <div class="row">
              <span class="tag" id="tagRun">...</span>
            </div>
          </div>

          <div class="muted" style="margin-top:12px;">Live logs</div>
          <div class="log mono" id="logBox" style="margin-top:10px;"></div>
          <div class="muted mono" style="font-size:12px; margin-top:8px;">(No refresh needed)</div>
        </div>
      </div>
    </div>
    {js}
    """
    return _html_page(f"Bot #{b['id']}", body)


# ---------------------------
# Live log SSE endpoint
# ---------------------------
# LIVE-HARDENED: max SSE stream duration so disconnected clients don't hold generator forever
LOGSTREAM_MAX_DURATION_SEC = int(os.getenv("LOGSTREAM_MAX_DURATION_SEC", "3600"))


@app.get("/api/bots/{bot_id}/logstream")
def api_log_stream(bot_id: int):
    b = _get_bot(int(bot_id))
    if not b:
        raise HTTPException(status_code=404, detail="Bot not found")

    def iter_sse() -> Iterable[str]:
        last_id = 0
        heartbeat = 0.0
        started = time.time()
        while True:
            now = time.time()
            if now - started >= LOGSTREAM_MAX_DURATION_SEC:
                yield "event: bye\ndata: {\"reason\":\"max_duration\"}\n\n"
                return
            if now - heartbeat >= 5:
                heartbeat = now
                yield "event: ping\ndata: {}\n\n"

            try:
                con = _conn()
                rows = con.execute(
                    "SELECT id, ts, level, message FROM bot_logs WHERE bot_id=? AND id>? ORDER BY id ASC LIMIT 200",
                    (int(bot_id), int(last_id)),
                ).fetchall()
                con.close()
            except Exception:
                rows = []

            for r in rows:
                last_id = int(r["id"])
                payload = {"log": {"id": last_id, "ts": int(r["ts"]), "level": r["level"], "message": r["message"]}}
                yield f"data: {json.dumps(payload)}\n\n"

            time.sleep(1.0)

    return StreamingResponse(iter_sse(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})
