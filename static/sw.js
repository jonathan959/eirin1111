/* Eirin Bot Service Worker - Enhanced PWA with offline support */
const CACHE_VERSION = 'v1';
const CACHE_SHELL = 'shell-' + CACHE_VERSION;
const CACHE_API = 'api-' + CACHE_VERSION;
const CACHE_OFFLINE = 'offline-' + CACHE_VERSION;

const SHELL_ASSETS = [
  '/static/app.css',
  '/static/app.js',
  '/static/manifest.json',
  '/static/favicon.svg',
  '/static/voice_alerts.js'
];

const OFFLINE_PAGE = '/offline.html';

// Install: cache shell + create offline fallback
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_SHELL).then((cache) => {
      return cache.addAll(SHELL_ASSETS).catch(() => {});
    }).then(() => self.skipWaiting())
  );
});

// Activate: clean old caches
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys
        .filter((k) => !k.startsWith('shell-' + CACHE_VERSION) &&
                       !k.startsWith('api-' + CACHE_VERSION) &&
                       !k.startsWith('offline-' + CACHE_VERSION))
        .map((k) => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

// Fetch: smart routing
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // Skip chrome extensions and unsupported protocols
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;

  // API endpoints: network-first with fallback to cache
  if (url.pathname.startsWith('/api/') || url.pathname === '/health') {
    return e.respondWith(networkFirstStrategy(e.request));
  }

  // HTML documents: network-first for updates
  if (e.request.mode === 'navigate') {
    return e.respondWith(networkFirstStrategy(e.request, OFFLINE_PAGE));
  }

  // Static assets: cache-first
  e.respondWith(cacheFirstStrategy(e.request));
});

// Network-first: try network, fallback to cache
function networkFirstStrategy(request, fallback = null) {
  return fetch(request)
    .then((response) => {
      if (!response || !response.ok) throw new Error('Network error');
      // Cache successful responses
      if (request.method === 'GET') {
        const cache_key = request.url.includes('/api/') ? CACHE_API : CACHE_SHELL;
        caches.open(cache_key).then((cache) => cache.put(request, response.clone()));
      }
      return response;
    })
    .catch(() => {
      return caches.match(request).then((cached) => {
        if (cached) return cached;
        if (fallback && request.mode === 'navigate') {
          return caches.match(fallback);
        }
        return new Response('Offline - no cache', { status: 503 });
      });
    });
}

// Cache-first: use cache, fallback to network
function cacheFirstStrategy(request) {
  return caches.match(request).then((cached) => {
    if (cached) return cached;
    return fetch(request)
      .then((response) => {
        if (response && response.ok && request.method === 'GET') {
          caches.open(CACHE_SHELL).then((cache) => cache.put(request, response.clone()));
        }
        return response;
      })
      .catch(() => {
        // Return placeholder for failed requests
        return new Response(
          '<!DOCTYPE html><html><body style="font-family:sans-serif;padding:2rem">Offline</body></html>',
          { headers: { 'Content-Type': 'text/html' } }
        );
      });
  });
}

// Push notifications
self.addEventListener('push', (e) => {
  if (!e.data) return;
  try {
    const d = e.data.json();
    const opts = {
      body: d.body || d.message || 'New notification',
      icon: '/static/icon-192.png',
      badge: '/static/icon-192.png',
      tag: d.tag || 'eirin-bot',
      data: d.data || {},
      requireInteraction: !!d.requireInteraction,
      vibrate: [200, 100, 200]
    };
    e.waitUntil(self.registration.showNotification(d.title || 'Eirin Bot', opts));
  } catch (_) {}
});

// Notification click handler
self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  const url = e.notification.data?.url || '/';
  e.waitUntil(
    self.clients.matchAll({ type: 'window' }).then((clients) => {
      const found = clients.find((c) => c.url === url);
      if (found) return found.focus();
      if (clients.length) return clients[0].navigate(url);
      return self.clients.openWindow(url);
    })
  );
});
