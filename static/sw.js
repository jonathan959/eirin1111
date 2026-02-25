/* Eirin Bot Service Worker - PWA */
const CACHE_NAME = 'tradingserver-v1';
const STATIC_ASSETS = [
  '/',
  '/static/app.css',
  '/static/manifest.json',
  '/static/voice_alerts.js',
  '/explore',
  '/dashboard',
  '/bots',
  '/journal',
  '/analytics',
  '/scenario-simulator'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS).catch(() => {});
    }).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.url.includes('/api/') || e.request.url.includes('/health')) {
    return;
  }
  e.respondWith(
    fetch(e.request).catch(() =>
      caches.match(e.request).then((r) => r || caches.match('/'))
    )
  );
});

self.addEventListener('push', (e) => {
  if (!e.data) return;
  try {
    const d = e.data.json();
    const opts = {
      body: d.body || d.message || 'New notification',
      icon: '/static/icon-192.png',
      badge: '/static/icon-192.png',
      tag: d.tag || 'tradingserver',
      data: d.data || {},
      requireInteraction: !!d.requireInteraction
    };
    e.waitUntil(self.registration.showNotification(d.title || 'Eirin Bot', opts));
  } catch (_) {}
});

self.addEventListener('notificationclick', (e) => {
  e.notification.close();
  const url = e.notification.data?.url || '/';
  e.waitUntil(self.clients.matchAll({ type: 'window' }).then((clients) => {
    if (clients.length) clients[0].navigate(url);
    else self.clients.openWindow(url);
  }));
});
