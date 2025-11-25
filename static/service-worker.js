self.addEventListener('install', event => {
  // Wird beim ersten Installieren der PWA ausgeführt
  console.log('Service Worker installiert');
});

self.addEventListener('activate', event => {
  // Wird aktiviert, wenn ein neuer Service Worker übernommen wird
  console.log('Service Worker aktiviert');
});

self.addEventListener('fetch', event => {
  // Im Moment leiten wir alle Requests einfach weiter
  event.respondWith(fetch(event.request));
});
