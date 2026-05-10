/**
 * Eirin Bot v2.0.0 - Global UI Utilities
 * Toast notifications, modals, dark mode, global search, keyboard shortcuts
 */

// App version
window.__APP_VERSION = "v2.0.0";

/**
 * Toast notification system
 */
function showToast(title, message, type, duration) {
  type = type || 'info';
  duration = duration || 4000;
  const icons = {
    success: '✅',
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️'
  };
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = 'toast toast-' + type;
  toast.innerHTML =
    '<span class="toast-icon">' + (icons[type] || icons.info) + '</span>' +
    '<div class="toast-body">' +
    '<div class="toast-title">' + (title || '') + '</div>' +
    (message ? '<div class="toast-message">' + message + '</div>' : '') +
    '</div>' +
    '<button class="toast-close" onclick="this.parentElement.remove()">&times;</button>';

  container.appendChild(toast);

  setTimeout(function() {
    toast.style.animation = 'toast-out 0.3s ease forwards';
    setTimeout(function() { toast.remove(); }, 300);
  }, duration);
}

/**
 * Confirmation modal system
 */
window.confirmAction = function(title, message, onConfirm) {
  const modal = document.getElementById("confirmModal");
  const titleEl = document.getElementById("confirmTitle");
  const msgEl = document.getElementById("confirmMessage");
  const okBtn = document.getElementById("confirmOk");
  const cancelBtn = document.getElementById("confirmCancel");

  if (!modal) return;

  titleEl.textContent = title;
  msgEl.textContent = message;

  const handleConfirm = function() {
    modal.classList.add("hidden");
    okBtn.removeEventListener("click", handleConfirm);
    cancelBtn.removeEventListener("click", handleCancel);
    if (onConfirm) onConfirm();
  };

  const handleCancel = function() {
    modal.classList.add("hidden");
    okBtn.removeEventListener("click", handleConfirm);
    cancelBtn.removeEventListener("click", handleCancel);
  };

  okBtn.addEventListener("click", handleConfirm);
  cancelBtn.addEventListener("click", handleCancel);
  modal.classList.remove("hidden");

  // Allow Escape to cancel
  const handleEscape = function(e) {
    if (e.key === "Escape") {
      handleCancel();
      document.removeEventListener("keydown", handleEscape);
    }
  };
  document.addEventListener("keydown", handleEscape);
};

/**
 * Dark/light mode toggle
 */
function initThemeToggle() {
  const themeBtn = document.getElementById("themeToggle");
  if (!themeBtn) return;

  function updateLabel() {
    const theme = document.documentElement.getAttribute("data-theme") || "light";
    themeBtn.textContent = theme === "dark" ? "🌙 Light" : "☀️ Dark";
  }

  themeBtn.addEventListener("click", function() {
    const html = document.documentElement;
    const current = html.getAttribute("data-theme") || "light";
    const next = current === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
    updateLabel();
    window.dispatchEvent(new Event("themechange"));
  });

  updateLabel();
}

/**
 * Enhanced command palette with navigation and actions
 */
function initGlobalSearch() {
  const globalSearchModal = document.getElementById("globalSearchModal");
  const globalSearchBtn = document.getElementById("globalSearchBtn");
  const globalSearchInput = document.getElementById("globalSearchInput");
  const globalSearchResults = document.getElementById("globalSearchResults");

  if (!globalSearchBtn) return;

  // Command palette entries
  const commands = [
    { category: "Navigation", icon: "📊", name: "Go to Dashboard", action: () => window.location.href = "/dashboard", keywords: ["dashboard", "home"] },
    { category: "Navigation", icon: "🤖", name: "Go to Bots", action: () => window.location.href = "/bots", keywords: ["bots", "my bots"] },
    { category: "Navigation", icon: "🔍", name: "Go to Explore", action: () => window.location.href = "/explore", keywords: ["explore", "recommendations", "search"] },
    { category: "Navigation", icon: "📈", name: "Go to Analytics", action: () => window.location.href = "/analytics", keywords: ["analytics", "stats"] },
    { category: "Navigation", icon: "⚙️", name: "Go to Settings", action: () => window.location.href = "/settings", keywords: ["settings", "config"] },
    { category: "Navigation", icon: "📔", name: "Go to Journal", action: () => window.location.href = "/journal", keywords: ["journal", "trades"] },
    { category: "Actions", icon: "➕", name: "Create new bot", action: () => window.location.href = "/bots?action=create", keywords: ["create", "new", "bot"] },
    { category: "Actions", icon: "🧪", name: "Run backtest", action: () => window.location.href = "/backtest", keywords: ["backtest", "test"] },
    { category: "Theme", icon: "🌙", name: "Toggle dark mode", action: () => {
      const html = document.documentElement;
      const current = html.getAttribute("data-theme") || "light";
      const next = current === "dark" ? "light" : "dark";
      html.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
    }, keywords: ["theme", "dark", "light", "mode"] }
  ];

  // Fuzzy match function
  function fuzzyMatch(query, text) {
    let qIdx = 0;
    let score = 0;
    for (let i = 0; i < text.length && qIdx < query.length; i++) {
      if (text[i].toLowerCase() === query[qIdx].toLowerCase()) {
        qIdx++;
        score += 10;
      } else {
        score -= 1;
      }
    }
    return qIdx === query.length ? Math.max(score, 0) : -1;
  }

  function openSearch() {
    globalSearchModal.classList.remove("hidden");
    setTimeout(() => globalSearchInput?.focus(), 0);
    updateResults("");
  }

  function updateResults(query) {
    if (!query || query.length === 0) {
      globalSearchResults.innerHTML = commands.map(cmd => `
        <div class="search-result" onclick="(${cmd.action.toString()})(); document.getElementById('globalSearchModal').classList.add('hidden')">
          <div class="search-result-icon">${cmd.icon}</div>
          <div class="search-result-text">
            <div class="search-result-title">${cmd.name}</div>
            <div class="search-result-sub">${cmd.category}</div>
          </div>
        </div>
      `).join("");
      return;
    }

    const q = query.toLowerCase();
    const results = commands.map(cmd => {
      let score = 0;
      score = Math.max(fuzzyMatch(q, cmd.name), score);
      cmd.keywords.forEach(kw => {
        score = Math.max(fuzzyMatch(q, kw), score);
      });
      return { ...cmd, score };
    }).filter(cmd => cmd.score > 0).sort((a, b) => b.score - a.score);

    if (results.length === 0) {
      globalSearchResults.innerHTML = '<div class="search-empty">No commands found</div>';
      return;
    }

    globalSearchResults.innerHTML = results.map(cmd => `
      <div class="search-result" onclick="(${cmd.action.toString()})(); document.getElementById('globalSearchModal').classList.add('hidden')">
        <div class="search-result-icon">${cmd.icon}</div>
        <div class="search-result-text">
          <div class="search-result-title">${cmd.name}</div>
          <div class="search-result-sub">${cmd.category}</div>
        </div>
      </div>
    `).join("");
  }

  globalSearchBtn.addEventListener("click", openSearch);

  if (globalSearchModal) {
    globalSearchModal.addEventListener("click", function(e) {
      if (e.target === globalSearchModal) {
        globalSearchModal.classList.add("hidden");
      }
    });
  }

  if (globalSearchInput) {
    globalSearchInput.addEventListener("keydown", function(e) {
      if (e.key === "Escape") {
        globalSearchModal.classList.add("hidden");
      } else if (e.key === "Enter") {
        const first = globalSearchResults.querySelector(".search-result");
        if (first) first.click();
      }
    });

    globalSearchInput.addEventListener("input", debounce(function() {
      const q = globalSearchInput.value.trim();
      updateResults(q);
    }, 100));
  }

  // Keyboard shortcut Cmd+K or Ctrl+K
  document.addEventListener("keydown", function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === "k") {
      e.preventDefault();
      openSearch();
    }
  });
}

function debounce(fn, ms) {
  let timer;
  return function(...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

async function searchBots(query) {
  const globalSearchResults = document.getElementById("globalSearchResults");
  try {
    const r = await fetch("/api/bots?search=" + encodeURIComponent(query));
    if (!r.ok) return;
    const data = await r.json();
    const bots = data.bots || [];

    if (!bots.length) {
      globalSearchResults.innerHTML = '<div class="search-empty">No results found</div>';
      return;
    }

    globalSearchResults.innerHTML = bots.map(b => `
      <a href="/bots/${b.id}" class="search-result" onclick="document.getElementById('globalSearchModal').classList.add('hidden')">
        <div class="search-result-icon">🤖</div>
        <div class="search-result-text">
          <div class="search-result-title">${b.name}</div>
          <div class="search-result-sub">${b.symbol} • ${b.dry_run ? "DRY" : "LIVE"}</div>
        </div>
      </a>
    `).join("");
  } catch (e) {
    console.warn("search error", e);
  }
}

/**
 * Sidebar collapse toggle
 */
function initSidebarToggle() {
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebar = document.getElementById("sidebar");

  if (!sidebarToggle || !sidebar) return;

  sidebarToggle.addEventListener("click", function() {
    sidebar.classList.toggle("collapsed");
    localStorage.setItem("sidebarCollapsed", sidebar.classList.contains("collapsed"));
  });

  if (localStorage.getItem("sidebarCollapsed") === "true") {
    sidebar.classList.add("collapsed");
  }
}

/**
 * Notification bell update
 */
function initNotificationBell() {
  const notificationBell = document.getElementById("notificationBell");
  const notificationBadge = document.getElementById("notificationBadge");

  if (!notificationBell || !notificationBadge) return;

  function updateBadge() {
    fetch("/api/notifications/unread_count", { headers: { "X-API-Key": window.__API_TOKEN || "" } })
      .then(r => r.json())
      .then(d => {
        const count = d.unread_count || d.count || 0;
        if (count > 0) {
          notificationBadge.textContent = count > 99 ? "99+" : count;
          notificationBadge.style.display = "block";
        } else {
          notificationBadge.style.display = "none";
        }
      })
      .catch(() => {});
  }

  notificationBell.addEventListener("click", function() {
    window.location.href = "/activity";
  });

  updateBadge();
  setInterval(updateBadge, 60000); // 60s — was 5s (CPU reduction)
}

/**
 * Number animation utility
 */
function easeOutQuad(t) {
  return t * (2 - t);
}

window.formatNumber = function(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '0';
  if (Math.abs(value) >= 1e6) {
    return (value / 1e6).toFixed(1) + 'M';
  }
  if (Math.abs(value) >= 1e3) {
    return (value / 1e3).toFixed(1) + 'K';
  }
  return value.toFixed(2);
};

window.animateValue = function(el, start, end, duration = 500) {
  const range = end - start;
  const startTime = performance.now();
  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const value = start + range * easeOutQuad(progress);
    if (el && el.textContent !== undefined) {
      el.textContent = formatNumber(value);
    }
    if (progress < 1) requestAnimationFrame(update);
  }
  requestAnimationFrame(update);
};

/**
 * Real-time P&L ticker
 *
 * Caches the most recent portfolio + active-bots values in localStorage so that
 * subsequent page loads don't flash "Portfolio: $—" / "Active: 0 bots" while the
 * first /api/portfolio + /api/health responses are in flight. Falls back to the
 * "$—" / "0" placeholders only when nothing has ever been cached.
 */
const TICKER_CACHE_LS_KEY = "pnl_ticker_cache_v1";

function _readTickerCache() {
  try {
    const raw = localStorage.getItem(TICKER_CACHE_LS_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw);
    if (!obj || typeof obj !== "object") return null;
    // Discard cache older than 7 days (stale data is worse than the placeholder).
    if (obj.ts && (Date.now() - obj.ts) > 7 * 86400 * 1000) return null;
    return obj;
  } catch (e) { return null; }
}

function _writeTickerCache(patch) {
  try {
    const cur = _readTickerCache() || {};
    const next = { ...cur, ...patch, ts: Date.now() };
    localStorage.setItem(TICKER_CACHE_LS_KEY, JSON.stringify(next));
  } catch (e) {}
}

function initPnLTicker() {
  const ticker = document.getElementById("pnlTicker");
  if (!ticker) return;
  let lastValidActiveBots = null;

  // Hydrate from localStorage immediately so the header doesn't flash "$—" / "0 bots"
  // before the first API call returns. Stale-but-known is better than blank.
  try {
    const cached = _readTickerCache();
    const valueEl = document.getElementById("tickerValue");
    const pnlEl = document.getElementById("tickerPnl");
    const activeBotsEl = document.getElementById("tickerBots");
    if (cached) {
      if (valueEl && typeof cached.totalValue === "number" && cached.totalValue > 0) {
        valueEl.textContent = "$" + cached.totalValue.toFixed(2);
        valueEl.title = "Showing last cached value — refreshing…";
      }
      if (pnlEl && typeof cached.todayPnL === "number") {
        pnlEl.textContent = (cached.todayPnL >= 0 ? "+$" : "-$") + Math.abs(cached.todayPnL).toFixed(2);
        pnlEl.className = cached.todayPnL >= 0 ? "pos" : "neg";
      }
      if (activeBotsEl && typeof cached.activeBots === "number") {
        activeBotsEl.textContent = String(cached.activeBots);
        lastValidActiveBots = cached.activeBots;
      }
    }
  } catch (e) {}

  function updateTicker() {
    Promise.all([
      fetch("/api/portfolio").then(r => r.ok ? r.json() : null).catch(() => null),
      fetch("/api/health").then(r => r.ok ? r.json() : null).catch(() => null),
    ]).then(([portData, healthData]) => {
      const valueEl = document.getElementById("tickerValue");
      const pnlEl = document.getElementById("tickerPnl");
      const activeBotsEl = document.getElementById("tickerBots");

      // Build a single cache patch from this update so the localStorage write
      // happens even if individual fields (e.g. totalValue==0) would have skipped
      // a per-field write previously.
      const patch = {};
      let havePort = false;
      let haveHealth = false;

      if (portData && portData.portfolio) {
        havePort = true;
        const portfolio = portData.portfolio;
        const totalValue = Number(portfolio.total_usd || portfolio.total_value) || 0;
        patch.totalValue = totalValue;

        if (valueEl && totalValue > 0) {
          valueEl.textContent = '$' + totalValue.toFixed(2);
          valueEl.title = "";
        }

        if (pnlEl) {
          const history = portData.history || [];
          let todayPnL = 0;
          if (history.length > 0 && totalValue > 0) {
            const now = Date.now() / 1000;
            const target24h = now - 86400;
            let oldest = history.reduce((best, h) =>
              Math.abs(h.ts - target24h) < Math.abs(best.ts - target24h) ? h : best
            , history[0]);
            todayPnL = totalValue - (oldest.total_usd || totalValue);
          }
          pnlEl.textContent = (todayPnL >= 0 ? '+$' : '-$') + Math.abs(todayPnL).toFixed(2);
          pnlEl.className = todayPnL >= 0 ? 'pos' : 'neg';
          patch.todayPnL = todayPnL;
        }
      }

      if (activeBotsEl && healthData && healthData.bots && typeof healthData.bots.running === "number") {
        haveHealth = true;
        lastValidActiveBots = healthData.bots.running;
        activeBotsEl.textContent = String(lastValidActiveBots);
        patch.activeBots = lastValidActiveBots;
      } else if (activeBotsEl && lastValidActiveBots !== null) {
        activeBotsEl.textContent = String(lastValidActiveBots);
      }

      // Single write per update, only when at least one source returned successfully —
      // avoids overwriting good cached values with empty data from a failed fetch.
      if (havePort || haveHealth) {
        _writeTickerCache(patch);
      }
    }).catch(() => {});
  }

  updateTicker();
  setInterval(updateTicker, 30000); // 30s — was 10s (CPU reduction)
}

/** CoinGecko: at most one request per second globally. */
window.__cgIconQueue = window.__cgIconQueue || [];
window.__cgIconPumpScheduled = window.__cgIconPumpScheduled || false;

function pumpCgIconQueue() {
  var fn = window.__cgIconQueue.shift();
  if (!fn) {
    window.__cgIconPumpScheduled = false;
    return;
  }
  Promise.resolve(fn())
    .catch(function() {})
    .then(function() {
      setTimeout(pumpCgIconQueue, 1000);
    });
}

window.scheduleCoingeckoFetch = function(task) {
  return new Promise(function(resolve, reject) {
    window.__cgIconQueue.push(function() {
      return Promise.resolve(task()).then(resolve, reject);
    });
    if (!window.__cgIconPumpScheduled) {
      window.__cgIconPumpScheduled = true;
      setTimeout(pumpCgIconQueue, 0);
    }
  });
};

/**
 * Initialize all UI enhancements on page load
 */
document.addEventListener("DOMContentLoaded", function() {
  initThemeToggle();
  initGlobalSearch();
  initSidebarToggle();
  initNotificationBell();
  initPnLTicker();
});
