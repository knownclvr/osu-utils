// job.js — one analysis job of the overlay's engine worker (shared by worker.js and the tests).
//
// Does what DanOverlay's AnalysisCoordinator._worker_loop + OverlayBridge do for a map:
//   raw = analyze_map(path, mod=mod, rate=rate, engine=engine, difficulty=difficulty)
//   raw["osu_sr"] = map_info.sr_official
//   payload = AnalysisResult.from_pipeline(raw, mod_label=map_info.mod_label).to_dict()
// and returns the payload as the overlay received it (after json.dumps → JSON.parse).
import { putFile, removeFile, norm } from './vfs.js';
import { srContexts } from './primary_sr_bridge.js';
import { analyze_map } from './pipeline.js';
import { AnalysisResult } from './contracts.js';
import { calibrateReformResult } from './reform-calibration.js';

/** json.dumps → JSON.parse of a payload: Map → object, typed arrays / Sets → arrays, undefined → null. */
export function jsonValue(v) {
  if (v === null || typeof v !== 'object') return v === undefined ? null : v;
  if (Array.isArray(v) || ArrayBuffer.isView(v) || v instanceof Set) return Array.from(v, jsonValue);
  const out = {};
  const entries = v instanceof Map ? v.entries() : Object.entries(v);
  for (const [k, x] of entries) out[String(k)] = jsonValue(x);
  return out;
}

/**
 * Pseudo st_mtime_ns of a beatmap file: derived from its MD5, so it stays the same while the
 * content is unchanged — like a real file's mtime. The engine's per-map caches and the monotonic
 * rate→SR registry (primary_sr_bridge) key on it, exactly as in DanOverlay.
 */
export function mtimeFromMd5(md5) {
  return /^[0-9a-f]{12}/i.test(md5 || '') ? parseInt(md5.slice(0, 12), 16) + 1 : undefined;
}

/**
 * job = { path, md5, bytes, mod, rate, engine, difficulty, srOfficial, modLabel }.
 * The .osu is present in the virtual file system only while it is analysed (the page caches results).
 * Throws what analyze_map / from_pipeline throw.
 */
export function runAnalysisJob(job) {
  putFile(job.path, job.bytes instanceof Uint8Array ? job.bytes : new Uint8Array(job.bytes), mtimeFromMd5(job.md5));
  if (job.srContext) srContexts.set(norm(job.path), job.srContext);
  try {
    const analysis = analyze_map(job.path, job.mod, false, job.rate, job.engine, job.difficulty);
    const raw = job.calibration === false ? analysis : calibrateReformResult(analysis);
    raw.osu_sr = job.srOfficial;
    return jsonValue(AnalysisResult.from_pipeline(raw, job.modLabel).to_dict());
  } finally {
    removeFile(job.path);
    srContexts.delete(norm(job.path));
  }
}
