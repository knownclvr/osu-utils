// worker.js — runs the ported DanOverlay engine in a module Web Worker, off the overlay's
// main thread. The page (shared/tm-engine.js) plays the part of DanOverlay's
// tosu_source + AnalysisCoordinator and sends one job at a time:
//
//   → { type: 'analyze', id, path, md5, bytes, mod, rate, engine, difficulty, srOfficial, modLabel }
//   ← { type: 'ready' } | { type: 'fatal', error }
//   ← { type: 'result', id, ok: true, payload, ms } | { type: 'result', id, ok: false, error }
import { setupEngine } from './setup.js';

// The engine modules read config/*.json when they are imported (as in Python), so they are
// imported only after setupEngine() has put the resources into the virtual file system.
const ready = (async () => {
  await setupEngine({ baseUrl: new URL('./', import.meta.url) });
  return import('./job.js');
})();
ready.then(
  () => self.postMessage({ type: 'ready' }),
  (err) => self.postMessage({ type: 'fatal', error: String((err && err.message) || err) }),
);

let liveMap = null;
const liveReady = ready.then(() => import('./live-segment.js'));

self.onmessage = async (event) => {
  const job = event.data || {};
  if (job.type !== 'analyze' && job.type !== 'segment') return;
  let runAnalysisJob;
  try {
    ({ runAnalysisJob } = await ready);
  } catch {
    return; // already reported as fatal
  }
  const t0 = performance.now();
  try {
    let payload;
    const live = await liveReady;
    if (job.type === 'segment') {
      if (!liveMap || liveMap.token !== job.token) throw new Error('Stale segment source');
      const rate = Number(liveMap.job.rate) || 1;
      const bucket = Math.floor(Math.max(0, job.timeMs) / (live.SEGMENT_STEP_MS * rate));
      payload = liveMap.cache.get(bucket);
      if (!payload) {
        payload = live.analyzeSegment(liveMap.job, liveMap.source, job.timeMs, liveMap.whole);
        liveMap.cache.set(bucket, payload);
        if (liveMap.cache.size > 90) liveMap.cache.delete(liveMap.cache.keys().next().value);
      }
    } else {
      payload = job.cachedPayload || runAnalysisJob(job);
      liveMap = { token: job.token, job: { ...job, bytes: undefined, cachedPayload: undefined },
        source: live.readSegmentSource(new Uint8Array(job.bytes)), whole: payload, cache: new Map() };
    }
    self.postMessage({ type: 'result', id: job.id, ok: true, payload, ms: performance.now() - t0 });
  } catch (err) {
    console.error('[total_maniaset engine]', err);
    self.postMessage({ type: 'result', id: job.id, ok: false, error: String((err && err.message) || err) });
  }
};
