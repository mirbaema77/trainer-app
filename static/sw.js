// Very basic service worker for PWA install support
const CACHE_NAME = "noqe-cache-v1";
const URLS_TO_CACHE = [
  "/",
  "/start",
  "/training",
  "/focus",
  "/duration",
  "/physical",
  "/players-count"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(URLS_TO_CACHE);
    })
  );
});

self.addEventListener("fetch", (event) => {
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});
