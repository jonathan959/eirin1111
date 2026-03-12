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
 * Global search modal
 */
function initGlobalSearch() {
  const globalSearchModal = document.getElementById("globalSearchModal");
  const globalSearchBtn = document.getElementById("globalSearchBtn");
  const globalSearchInput = document.getElementById("globalSearchInput");
  const globalSearchResults = document.getElementById("globalSearchResults");

  if (!globalSearchBtn) return;

  function openSearch() {
    globalSearchModal.classList.remove("hidden");
    setTimeout(() => globalSearchInput?.focus(), 0);
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
      }
    });

    globalSearchInput.addEventListener("input", debounce(function() {
      const q = globalSearchInput.value.trim();
      if (!q || q.length < 2) {
        globalSearchResults.innerHTML = "";
        return;
      }
      searchBots(q);
    }, 200));
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
    fetch("/api/activity/unread", { headers: { "X-API-Key": window.__API_TOKEN || "" } })
      .then(r => r.json())
      .then(d => {
        const count = d.count || 0;
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
  setInterval(updateBadge, 5000);
}

/**
 * Initialize all UI enhancements on page load
 */
document.addEventListener("DOMContentLoaded", function() {
  initThemeToggle();
  initGlobalSearch();
  initSidebarToggle();
  initNotificationBell();
});
