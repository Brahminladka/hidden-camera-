const CACHE_NAME = 'camguard-v7';
const ASSETS = [
    './',
    './index.html',
    './index-enhanced.html',
    './style.css',
    './app.js',
    './fusion-engine.js',
    './fusion-panel.js',
    './manifest.json',
    './assets/icons/icon.svg',
    './assets/icons/icon-512.png',
    './assets/vendor/tf.min.js',
    './assets/vendor/coco-ssd.min.js',
    './assets/custom_model_embed.js',
    './data/brains/runtime_brain.json'
];

self.addEventListener('install', e =>
{
    e.waitUntil(
        caches.open(CACHE_NAME).then(cache => cache.addAll(ASSETS))
    );
    self.skipWaiting();
});

async function warmModelCache()
{
    const cache = await caches.open(CACHE_NAME);
    const urls = [
        './models/tfjs/model.json',
        './models/tfjs/group1-shard1of3.bin',
        './models/tfjs/group1-shard2of3.bin',
        './models/tfjs/group1-shard3of3.bin'
    ];
    for (const url of urls) {
        try {
            await cache.add(url);
        } catch (_) {
            // ignore missing model files
        }
    }
}

self.addEventListener('activate', e =>
{
    e.waitUntil(
        caches.keys().then(keys => Promise.all(
            keys.map(k => k !== CACHE_NAME ? caches.delete(k) : null)
        )).then(() => warmModelCache())
    );
    self.clients.claim();
});

// Cache-First with validation (Don't cache tunnel error pages)
self.addEventListener('fetch', e =>
{
    if (e.request.method !== 'GET') return;

    e.respondWith(
        caches.match(e.request).then(cached =>
        {
            // Return cached immediately for speed/independence
            const fetchPromise = fetch(e.request)
                .then(res =>
                {
                    // CRITICAL: Only cache if it's a real successful response
                    // prevents localtunnel 401/lobby pages from overwriting our app code
                    if (res && res.status === 200 && res.type === 'basic') {
                        const cacheCopy = res.clone();
                        caches.open(CACHE_NAME).then(cache => cache.put(e.request, cacheCopy));
                    }
                    return res;
                })
                .catch(() => cached);

            return cached || fetchPromise;
        })
    );
});
