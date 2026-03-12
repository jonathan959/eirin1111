(function () {
  const botId = Number(window.__BOT_ID__);
  const el = document.getElementById("tvChart");
  const rsiEl = document.getElementById("tvChartRSI");
  const macdEl = document.getElementById("tvChartMACD");
  const errEl = document.getElementById("chartErr");
  const tfEl = document.getElementById("tf");
  const btn = document.getElementById("chartRefresh");
  const controlsEl = document.getElementById("chartControls");

  // Chart state
  let chart = null;
  let rsiChart = null;
  let macdChart = null;
  let candlesSeries = null;
  let volumeSeries = null;
  let ema20Series = null;
  let ema50Series = null;
  let ema200Series = null;
  let bbUpperSeries = null;
  let bbMiddleSeries = null;
  let bbLowerSeries = null;
  let rsiSeries = null;
  let macdSeries = null;
  let macdSignalSeries = null;
  let macdHistogramSeries = null;

  // Toggle state
  const toggleState = {
    ema: false,
    bb: false,
    rsi: false,
    macd: false,
    vol: false,
  };

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
      muted: styles.getPropertyValue("--muted").trim() || "#94a3b8",
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
    if (chart) {
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
    }
    if (rsiChart) {
      rsiChart.applyOptions({
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
    }
    if (macdChart) {
      macdChart.applyOptions({
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
    }
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
    if (volumeSeries) {
      volumeSeries.applyOptions({
        color: "rgba(34, 197, 94, 0.3)",
      });
    }
  }

  // Initialize main chart
  ensureSize();
  chart = window.LightweightCharts.createChart(el, {
    width: el.clientWidth || 900,
    height: el.clientHeight || 420,
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

  // Initialize RSI chart
  if (rsiEl) {
    if (rsiEl.clientHeight < 50) {
      rsiEl.style.height = "180px";
    }
    if (rsiEl.clientWidth < 50) {
      rsiEl.style.width = "100%";
    }
    rsiChart = window.LightweightCharts.createChart(rsiEl, {
      width: rsiEl.clientWidth || 900,
      height: rsiEl.clientHeight || 180,
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
  }

  // Initialize MACD chart
  if (macdEl) {
    if (macdEl.clientHeight < 50) {
      macdEl.style.height = "180px";
    }
    if (macdEl.clientWidth < 50) {
      macdEl.style.width = "100%";
    }
    macdChart = window.LightweightCharts.createChart(macdEl, {
      width: macdEl.clientWidth || 900,
      height: macdEl.clientHeight || 180,
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
  }

  // Create series
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
    candlesSeries = chart.addSeries(window.LightweightCharts.CandlestickSeries, candleOpts);
  } else {
    showErr("Chart API mismatch: candlestick series not supported.");
    return;
  }

  // Volume series
  volumeSeries = chart.addHistogramSeries({
    color: "rgba(34, 197, 94, 0.3)",
  });

  // EMA series
  ema20Series = chart.addLineSeries({
    color: "#3b82f6",
    lineWidth: 2,
    title: "EMA 20",
  });
  ema50Series = chart.addLineSeries({
    color: "#f97316",
    lineWidth: 2,
    title: "EMA 50",
  });
  ema200Series = chart.addLineSeries({
    color: "#ef4444",
    lineWidth: 2,
    title: "EMA 200",
  });

  // Bollinger Bands series
  bbUpperSeries = chart.addLineSeries({
    color: "rgba(148, 163, 184, 0.5)",
    lineWidth: 1,
    lineStyle: 2,
    title: "BB Upper",
  });
  bbMiddleSeries = chart.addLineSeries({
    color: "rgba(148, 163, 184, 0.3)",
    lineWidth: 1,
    lineStyle: 2,
    title: "BB Middle",
  });
  bbLowerSeries = chart.addLineSeries({
    color: "rgba(148, 163, 184, 0.5)",
    lineWidth: 1,
    lineStyle: 2,
    title: "BB Lower",
  });

  // RSI series (on RSI chart)
  if (rsiChart) {
    rsiSeries = rsiChart.addLineSeries({
      color: "#8b5cf6",
      lineWidth: 2,
      title: "RSI",
    });
  }

  // MACD series (on MACD chart)
  if (macdChart) {
    macdSeries = macdChart.addLineSeries({
      color: "#3b82f6",
      lineWidth: 2,
      title: "MACD",
    });
    macdSignalSeries = macdChart.addLineSeries({
      color: "#f97316",
      lineWidth: 2,
      title: "Signal",
    });
    macdHistogramSeries = macdChart.addHistogramSeries({
      color: "#10b981",
      title: "Histogram",
    });
  }

  // Hide indicators initially
  ema20Series.applyOptions({ visible: false });
  ema50Series.applyOptions({ visible: false });
  ema200Series.applyOptions({ visible: false });
  bbUpperSeries.applyOptions({ visible: false });
  bbMiddleSeries.applyOptions({ visible: false });
  bbLowerSeries.applyOptions({ visible: false });
  volumeSeries.applyOptions({ visible: false });
  if (rsiChart) rsiChart.timeScale().applyOptions({ visible: false });
  if (macdChart) macdChart.timeScale().applyOptions({ visible: false });

  function resize() {
    try {
      chart.applyOptions({
        width: el.clientWidth || 900,
        height: el.clientHeight || 420,
      });
    } catch (_) {}
    try {
      if (rsiChart) {
        rsiChart.applyOptions({
          width: rsiEl.clientWidth || 900,
          height: rsiEl.clientHeight || 180,
        });
      }
    } catch (_) {}
    try {
      if (macdChart) {
        macdChart.applyOptions({
          width: macdEl.clientWidth || 900,
          height: macdEl.clientHeight || 180,
        });
      }
    } catch (_) {}
  }

  window.addEventListener("resize", resize);
  window.addEventListener("themechange", applyTheme);
  applyTheme();

  let lastTf = null;

  // Indicator calculations
  function calculateEMA(data, period) {
    const result = [];
    const k = 2 / (period + 1);
    let ema = null;

    for (let i = 0; i < data.length; i++) {
      const close = data[i].close;
      if (ema === null) {
        ema = close;
      } else {
        ema = close * k + ema * (1 - k);
      }
      result.push({ time: data[i].time, value: ema });
    }
    return result;
  }

  function calculateBollingerBands(data, period = 20, stdDev = 2) {
    const result = { upper: [], middle: [], lower: [] };

    for (let i = period - 1; i < data.length; i++) {
      const slice = data.slice(i - period + 1, i + 1);
      const mean = slice.reduce((sum, c) => sum + c.close, 0) / period;
      const variance = slice.reduce((sum, c) => sum + Math.pow(c.close - mean, 2), 0) / period;
      const std = Math.sqrt(variance);

      result.upper.push({ time: data[i].time, value: mean + stdDev * std });
      result.middle.push({ time: data[i].time, value: mean });
      result.lower.push({ time: data[i].time, value: mean - stdDev * std });
    }
    return result;
  }

  function calculateRSI(data, period = 14) {
    const result = [];
    const changes = [];

    for (let i = 1; i < data.length; i++) {
      changes.push(data[i].close - data[i - 1].close);
    }

    for (let i = 0; i < changes.length; i++) {
      if (i < period - 1) {
        result.push({ time: data[i + 1].time, value: 50 });
      } else {
        const gains = changes.slice(i - period + 1, i + 1).filter(c => c > 0).reduce((a, b) => a + b, 0) / period;
        const losses = Math.abs(changes.slice(i - period + 1, i + 1).filter(c => c < 0).reduce((a, b) => a + b, 0)) / period;
        const rs = losses === 0 ? 100 : gains === 0 ? 0 : gains / losses;
        const rsi = 100 - (100 / (1 + rs));
        result.push({ time: data[i + 1].time, value: rsi });
      }
    }
    return result;
  }

  function calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    const fastEMA = calculateEMA(data, fastPeriod);
    const slowEMA = calculateEMA(data, slowPeriod);

    const macd = [];
    for (let i = slowPeriod - 1; i < data.length; i++) {
      const fast = fastEMA[i] ? fastEMA[i].value : 0;
      const slow = slowEMA[i] ? slowEMA[i].value : 0;
      macd.push({ time: data[i].time, value: fast - slow });
    }

    const signal = calculateEMA(
      macd.map((m, i) => ({ ...m, close: m.value })),
      signalPeriod
    );

    const histogram = [];
    for (let i = 0; i < macd.length; i++) {
      const sig = signal[i] ? signal[i].value : 0;
      histogram.push({ time: macd[i].time, value: macd[i].value - sig });
    }

    return { macd, signal, histogram };
  }

  async function loadChart(opts) {
    const options = opts || { forceFit: false };
    showErr("");
    const tf = tfEl ? (tfEl.value || "1d") : "1d";
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

    // Set candles
    candlesSeries.setData(candles);

    // Set volume data
    const volumeData = candles.map(c => ({
      time: c.time,
      value: c.volume || 0,
      color: c.close >= c.open ? "rgba(34, 197, 94, 0.3)" : "rgba(239, 68, 68, 0.3)",
    }));
    volumeSeries.setData(volumeData);

    // Calculate and set indicators
    if (toggleState.ema) {
      const ema20 = calculateEMA(candles, 20);
      const ema50 = calculateEMA(candles, 50);
      const ema200 = calculateEMA(candles, 200);
      ema20Series.setData(ema20);
      ema50Series.setData(ema50);
      ema200Series.setData(ema200);
    }

    if (toggleState.bb) {
      const bb = calculateBollingerBands(candles, 20, 2);
      bbUpperSeries.setData(bb.upper);
      bbMiddleSeries.setData(bb.middle);
      bbLowerSeries.setData(bb.lower);
    }

    if (toggleState.rsi && rsiChart && rsiSeries) {
      const rsi = calculateRSI(candles, 14);
      rsiSeries.setData(rsi);
      rsiChart.timeScale().syncToChart(chart.timeScale());
    }

    if (toggleState.macd && macdChart && macdSeries) {
      const macdData = calculateMACD(candles, 12, 26, 9);
      macdSeries.setData(macdData.macd);
      macdSignalSeries.setData(macdData.signal);
      macdHistogramSeries.setData(macdData.histogram);
      macdChart.timeScale().syncToChart(chart.timeScale());
    }

    // Set markers (buy/sell)
    if (typeof candlesSeries.setMarkers === "function") {
      candlesSeries.setMarkers(markers);
    }

    if (options.forceFit || lastTf !== tf) {
      chart.timeScale().fitContent();
      if (rsiChart) rsiChart.timeScale().fitContent();
      if (macdChart) macdChart.timeScale().fitContent();
    }
    lastTf = tf;
  }

  // Chart controls
  function setupControls() {
    if (!controlsEl) return;

    const buttons = controlsEl.querySelectorAll("button[data-toggle]");
    buttons.forEach(btn => {
      btn.addEventListener("click", async function () {
        const toggle = this.getAttribute("data-toggle");
        if (toggle === "vol") {
          toggleState.vol = !toggleState.vol;
          volumeSeries.applyOptions({ visible: toggleState.vol });
          this.classList.toggle("active", toggleState.vol);
        } else if (toggle === "ema") {
          toggleState.ema = !toggleState.ema;
          ema20Series.applyOptions({ visible: toggleState.ema });
          ema50Series.applyOptions({ visible: toggleState.ema });
          ema200Series.applyOptions({ visible: toggleState.ema });
          this.classList.toggle("active", toggleState.ema);
          if (toggleState.ema) await loadChart({ forceFit: false });
        } else if (toggle === "bb") {
          toggleState.bb = !toggleState.bb;
          bbUpperSeries.applyOptions({ visible: toggleState.bb });
          bbMiddleSeries.applyOptions({ visible: toggleState.bb });
          bbLowerSeries.applyOptions({ visible: toggleState.bb });
          this.classList.toggle("active", toggleState.bb);
          if (toggleState.bb) await loadChart({ forceFit: false });
        } else if (toggle === "rsi") {
          toggleState.rsi = !toggleState.rsi;
          if (rsiEl) rsiEl.classList.toggle("hidden", !toggleState.rsi);
          this.classList.toggle("active", toggleState.rsi);
          if (toggleState.rsi) await loadChart({ forceFit: false });
        } else if (toggle === "macd") {
          toggleState.macd = !toggleState.macd;
          if (macdEl) macdEl.classList.toggle("hidden", !toggleState.macd);
          this.classList.toggle("active", toggleState.macd);
          if (toggleState.macd) await loadChart({ forceFit: false });
        }
      });
    });

    // Fullscreen button
    const fsBtn = document.getElementById("chartFullscreen");
    if (fsBtn) {
      fsBtn.addEventListener("click", function () {
        const container = el.parentElement;
        if (document.fullscreenElement) {
          document.exitFullscreen();
        } else {
          container.requestFullscreen().catch(err => {
            console.error("Fullscreen failed:", err);
          });
        }
      });
    }
  }

  // Allow template to trigger reload
  window.__chartReload = () => loadChart({ forceFit: false }).catch(() => {});

  function init() {
    setupControls();
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
