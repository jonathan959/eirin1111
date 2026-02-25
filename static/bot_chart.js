(function () {
  const botId = Number(window.__BOT_ID__);
  const el = document.getElementById("tvChart");
  const errEl = document.getElementById("chartErr");
  const tfEl = document.getElementById("tf");
  const btn = document.getElementById("chartRefresh");

  function showErr(msg) {
    if (!errEl) return;
    errEl.classList.toggle("hidden", !msg);
    errEl.textContent = msg || "";
  }

  async function fetchJSON(url) {
    const r = await fetch(url, {
      cache: "no-store",
      headers: { "Accept": "application/json" },
    });
    let j = {};
    try {
      j = await r.json();
    } catch (_) {}
    if (!r.ok) {
      const detail = (j && (j.detail || j.error))
        ? (j.detail || j.error)
        : (await r.text().catch(() => ""));
      throw new Error(`${r.status} ${r.statusText}${detail ? `: ${detail}` : ""}`);
    }
    return j || {};
  }

  function hasLWCharts() {
    return typeof window.LightweightCharts !== "undefined" &&
      !!window.LightweightCharts.createChart;
  }

  if (!el || !hasLWCharts() || !Number.isFinite(botId)) {
    showErr(!hasLWCharts() ? "Chart library failed to load." : "Chart container missing.");
    return;
  }

  function themeColors() {
    const styles = getComputedStyle(document.documentElement);
    return {
      text: styles.getPropertyValue("--text").trim() || "#e2e8f0",
      border: styles.getPropertyValue("--border").trim() || "rgba(148, 163, 184, 0.2)",
      success: styles.getPropertyValue("--success").trim() || "#22c55e",
      danger: styles.getPropertyValue("--danger").trim() || "#ef4444",
    };
  }

  function ensureSize() {
    if (el.clientHeight < 50) {
      el.style.height = "420px";
    }
    if (el.clientWidth < 50) {
      el.style.width = "100%";
    }
  }

  function applyTheme() {
    const colors = themeColors();
    chart.applyOptions({
      layout: {
        background: { type: "solid", color: "transparent" },
        textColor: colors.text,
      },
      grid: {
        vertLines: { color: colors.border },
        horzLines: { color: colors.border },
      },
      timeScale: { borderColor: colors.border },
      rightPriceScale: { borderColor: colors.border },
    });
    if (candlesSeries) {
      candlesSeries.applyOptions({
        upColor: colors.success,
        downColor: colors.danger,
        borderUpColor: colors.success,
        borderDownColor: colors.danger,
        wickUpColor: colors.success,
        wickDownColor: colors.danger,
      });
    }
  }

  ensureSize();
  const chart = window.LightweightCharts.createChart(el, {
    width: el.clientWidth || 900,
    height: el.clientHeight || 360,
    layout: {
      background: { type: "solid", color: "transparent" },
      textColor: themeColors().text,
    },
    grid: {
      vertLines: { color: themeColors().border },
      horzLines: { color: themeColors().border },
    },
    timeScale: {
      timeVisible: true,
      secondsVisible: false,
      borderColor: themeColors().border,
    },
    rightPriceScale: { borderColor: themeColors().border },
    crosshair: { mode: 1 },
  });

  let candlesSeries;
  const colors = themeColors();
  const candleOpts = {
    upColor: colors.success,
    downColor: colors.danger,
    borderUpColor: colors.success,
    borderDownColor: colors.danger,
    wickUpColor: colors.success,
    wickDownColor: colors.danger,
  };
  if (typeof chart.addCandlestickSeries === "function") {
    candlesSeries = chart.addCandlestickSeries(candleOpts);
  } else if (typeof chart.addSeries === "function" && window.LightweightCharts.CandlestickSeries) {
    // Newer API (v5+) uses addSeries with series constructor
    candlesSeries = chart.addSeries(window.LightweightCharts.CandlestickSeries, candleOpts);
  } else {
    showErr("Chart API mismatch: candlestick series not supported.");
    return;
  }

  function resize() {
    try {
      chart.applyOptions({
        width: el.clientWidth || 900,
        height: el.clientHeight || 360,
      });
    } catch (_) {}
  }
  window.addEventListener("resize", resize);
  window.addEventListener("themechange", applyTheme);
  applyTheme();

  let lastTf = null;

  async function loadChart(opts) {
    const options = opts || { forceFit: false };
    showErr("");
    const tf = tfEl ? (tfEl.value || "5m") : "5m";
    const limit = 500;

    const [c, m] = await Promise.all([
      fetchJSON(`/api/bots/${botId}/ohlc?timeframe=${encodeURIComponent(tf)}&limit=${limit}`),
      fetchJSON(`/api/bots/${botId}/markers?timeframe=${encodeURIComponent(tf)}&limit=${limit}`),
    ]);

    const candles = Array.isArray(c.candles) ? c.candles : [];
    const markers = Array.isArray(m.markers) ? m.markers : [];

    if (!candles.length) {
      showErr("No candle data available.");
      return;
    }

    candlesSeries.setData(candles);
    if (typeof candlesSeries.setMarkers === "function") {
      candlesSeries.setMarkers(markers);
    }

    if (options.forceFit || lastTf !== tf) {
      chart.timeScale().fitContent();
    }
    lastTf = tf;
  }

  // Allow template to trigger reload after Start
  window.__chartReload = () => loadChart({ forceFit: false }).catch(() => {});

  function init() {
    loadChart({ forceFit: true }).catch((e) => showErr(e.message || String(e)));

    if (btn) {
      btn.addEventListener("click", () => {
        loadChart({ forceFit: true }).catch((e) => showErr(e.message || String(e)));
      });
    }
    if (tfEl) {
      tfEl.addEventListener("change", () => {
        loadChart({ forceFit: true }).catch((e) => showErr(e.message || String(e)));
      });
    }

    setInterval(() => loadChart({ forceFit: false }).catch(() => {}), 15000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
