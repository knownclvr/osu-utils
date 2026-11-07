// Rolling excerpts use the same engine as the full map. Times here are map milliseconds.
import { runAnalysisJob } from './job.js';
import { _sr_cache, _mono_registry } from './primary_sr_bridge.js';
import { _cache as msdCache } from './minacalc_bridge.js';
import { _impl_cache } from './pipeline.js';

export const SEGMENT_STEP_MS = 500;

export function readSegmentSource(bytes) {
  const text = new TextDecoder().decode(bytes);
  let section = '';
  const header = [], timing = [], objects = [];
  for (const line of text.split(/\r?\n/)) {
    const match = line.trim().match(/^\[([^\]]+)\]$/);
    if (match) section = match[1];
    if (section === 'HitObjects') {
      const fields = line.split(',');
      const time = Number(fields[2]);
      if (fields.length >= 5 && Number.isFinite(time)) objects.push({ fields, time });
    } else if (section === 'TimingPoints') {
      const fields = line.split(',');
      if (fields.length >= 2 && Number.isFinite(Number(fields[0]))) timing.push(fields);
    } else if (section !== 'Events') header.push(line);
  }
  if (!text.includes('[HitObjects]')) return null;
  return { header, timing, objects };
}

export function sliceSegment(source, from, to) {
  const lines = [];
  // Start each excerpt at 1s so analyzers do not process the original song's lead-in.
  const shift = 1000 - from;
  for (const object of source.objects) {
    const fields = [...object.fields];
    const hold = (Number(fields[3]) & 128) !== 0;
    const endFields = hold ? String(fields[5] || '').split(':') : null;
    const end = hold ? Number(endFields[0]) : object.time;
    if (object.time >= to || end < from || (!hold && object.time < from)) continue;
    fields[2] = String(Math.round(Math.max(from, object.time) + shift));
    if (hold) {
      endFields[0] = String(Math.round(Math.min(to, end) + shift));
      fields[5] = endFields.join(':');
    }
    lines.push(fields.join(','));
  }
  const timing = source.timing.filter((f) => Number(f[0]) <= to).map((f) => {
    const fields = [...f];
    fields[0] = String(Number(fields[0]) + shift);
    return fields.join(',');
  });
  return { text: [...source.header, '[TimingPoints]', ...timing, '[HitObjects]', ...lines].join('\n'), count: lines.length };
}

export function analyzeSegment(job, source, timeMs, whole) {
  if (!source) return { ...whole, analysis_scope: 'unavailable' };
  const rate = Number(job.rate) > 0 ? Number(job.rate) : 1;
  const bucket = Math.floor(Math.max(0, timeMs) / (SEGMENT_STEP_MS * rate));
  const center = bucket * SEGMENT_STEP_MS * rate;
  const from = Math.max(0, center - 5000 * rate), to = center + 5000 * rate;
  const excerpt = sliceSegment(source, from, to);
  const info = { from_ms: from, to_ms: to, center_ms: center, window_s: 10, rest: excerpt.count === 0 };
  if (!excerpt.count) return { ...whole, analysis_scope: 'segment', segment: info };
  const path = `/live/${job.md5 || 'map'}/${job.engine}/${bucket}.osu`;
  let result;
  try {
    result = runAnalysisJob({ ...job, path, bytes: new TextEncoder().encode(excerpt.text),
      srContext: { noteCountReference: source.objects.length } });
  } finally {
    // The worker keeps a bounded excerpt cache; avoid retaining every second again
    // in the original engines' session-long per-file caches (including rate anchors).
    for (const cache of [_sr_cache, _mono_registry, msdCache, _impl_cache]) {
      for (const key of cache.keys()) if (String(key).includes(path)) cache.delete(key);
    }
  }
  if (result.error || !(Number(result.dp) > 0 || Number(result.dp_7k) > 0)) {
    return { ...whole, analysis_scope: 'unavailable', segment: info };
  }
  // Keep map metadata from the whole song; only the difficulty and skillsets change.
  for (const key of ['osu_sr', 'od', 'bpm', 'bpm_min', 'bpm_max', 'bpm_common']) result[key] = whole[key];
  return { ...result, analysis_scope: 'segment', segment: info };
}
