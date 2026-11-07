// primary_sr_bridge.js — primary (Sunny) SR bridge: runs sr_core.algorithm with rate scaling + caching.
//
// JavaScript port of engine/dan_engine/primary_sr_bridge.py (DanOverlay 3.1.0 by 8doul).
import * as py from './py.js';
import * as np from './np.js';
import * as vfs from './vfs.js';
import * as sr_core_algorithm from './sr_core/algorithm.js';

// Where the Python module looked for sr_core (sys.path entry); kept for reference only.
export const _SRC_DIR = vfs.path.dirname(vfs.RES_ROOT);
export const _SR_CORE_DIR = vfs.path.join(_SRC_DIR, '03_engine_reference', 'sr_core');

// Lazily "imported" sr_core algorithm module (see _import_sr_core).
export let _sr_core_alg = null;

// Result cache: (abs path, mtime_ns, mod, rate) -> result dict (keyed by the tuple's text form).
export const _sr_cache = new Map();

// Per-map (abs path, mtime_ns, mod) -> {rate: sr} registry used by _enforce_monotonic_sr.
export const _mono_registry = new Map();

// Optional per-job SR context, scoped by runAnalysisJob and removed in its finally.
export const srContexts = new Map();

/** Map key standing for a Python tuple key (values compare like the tuple's items). */
function tupleKey(t) {
  return t.map((v) => `${typeof v}:${typeof v === 'number' && v === 0 ? 0 : v}`).join('\u0000');
}

/** os.stat(file_path).st_mtime_ns, 0 on OSError. */
function mtimeNs(file_path) {
  try {
    return vfs.statMtimeNs(file_path);
  } catch (e) {
    if (!(e instanceof py.OSError_)) throw e;
    return 0;
  }
}

export function _sr_cache_key(file_path, mod, rate = 1.0) {
  const mtime = mtimeNs(file_path);
  return [vfs.path.abspath(file_path), mtime, mod, py.round(rate, 3)];
}

export function _mono_key(file_path, mod) {
  const mtime = mtimeNs(file_path);
  return [vfs.path.abspath(file_path), mtime, mod];
}

/**
 * Isotonic CLAMP over the per-map rate->SR registry: SR(rate) must be >= the max SR of any lower
 * rate seen (floor) and <= the min SR of any higher rate seen (ceiling). The raw value is kept
 * when it is consistent with everything seen so far — only violators are clamped.
 */
export function _enforce_monotonic_sr(file_path, mod, rate, sr) {
  if (rate === null || rate === undefined) return sr;
  const key = tupleKey(_mono_key(file_path, mod));
  let reg = _mono_registry.get(key);
  if (reg === undefined) {
    reg = new Map();
    _mono_registry.set(key, reg);
  }

  const r = py.round(py.float(rate), 3);
  reg.set(r, sr);

  // Floor: max SR of any lower rate seen.
  let floor_sr = 0.0;
  for (const [x, v] of reg) if (x < r && v > floor_sr) floor_sr = v;
  // Ceiling: min SR of any higher rate seen.
  let ceil_sr = Infinity;
  for (const [x, v] of reg) if (x > r && v < ceil_sr) ceil_sr = v;

  let clamped = sr;
  if (clamped < floor_sr) clamped = floor_sr;
  if (clamped > ceil_sr) clamped = ceil_sr;

  reg.set(r, clamped);

  // Keep the registry bounded.
  if (reg.size > 64) {
    for (const stale of py.sorted([...reg.keys()]).slice(0, -64)) reg.delete(stale);
  }

  return py.float(clamped);
}

export function _import_sr_core() {
  if (_sr_core_alg !== null) return _sr_core_alg;
  _sr_core_alg = sr_core_algorithm;
  return _sr_core_alg;
}

export const _EMPTY_RESULT = {
  sr: 0.0,
  jbar_max: 0.0,
  pbar_max: 0.0,
  xbar_max: 0.0,
  abar_mean: 0.0,
  jack_ratio: 0.0,
  d93: 0.0,
  d83: 0.0,
  d_weighted_mean: 0.0,
  total_notes_eff: 0.0,
  jbar_share: 0.0,
  pbar_share: 0.0,
  xbar_share: 0.0,
  rbar_max: 0.0,
  success: false,
  error: null,
};

/** str() of a Python int held in a double (exact digits, no exponent). */
function intStr(n) {
  return Number.isSafeInteger(n) ? String(n) : BigInt(n).toString();
}

/**
 * Create a temp .osu with hit times scaled for a custom clock rate (hit times divided by the
 * rate, rounded to ints), so the engine can run it with mod="NM".
 */
export function _scale_osu_timings(file_path, rate) {
  const lines = [];
  let section = null;
  for (const raw of py.textLines(vfs.readText(file_path, 'replace'))) {
    const line = py.rstrip(raw, '\r\n');
    const stripped = py.strip(line);
    if (stripped.startsWith('[')) {
      section = stripped;
      lines.push(line);
      continue;
    }
    if (section === '[HitObjects]' && stripped && !stripped.startsWith('//')) {
      const parts = stripped.split(',');
      if (parts.length >= 3) {
        try {
          const t = py.div(py.float(parts[2]), rate);
          parts[2] = intStr(py.roundInt(t));
        } catch (e) {
          if (!(e instanceof py.ValueError)) throw e;
        }
      }
      lines.push(parts.join(','));
      continue;
    }
    lines.push(line);
  }

  const tmp = vfs.tempPath('danoverlay_rate_', '.osu');
  vfs.putFile(tmp, lines.join('\n'));
  return tmp;
}

/** np.max(a) (maximum.reduce: NaN propagates). */
function npMaxReduce(a) {
  let m = a[0];
  if (m !== m) return m;
  for (let i = 1; i < a.length; i++) {
    const v = a[i];
    if (v !== v) return v;
    if (v > m) m = v;
  }
  return m;
}

/** np.round(np.linspace(0, n - 1, 300)).astype(int) */
function linspaceIndices(n) {
  const num = 300;
  const stop = n - 1;
  const step = stop / (num - 1);
  const out = new Array(num);
  for (let i = 0; i < num; i++) {
    const y = i === num - 1 ? stop : i * step + 0.0;
    // np.round → rint (half to even)
    const f = Math.floor(y);
    const d = y - f;
    out[i] = d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f;
  }
  return out;
}

/** f"{type(exc).__name__}: {exc}" for errors raised by the ported code. */
function excText(e) {
  let name = e?.name || e?.constructor?.name || 'Exception';
  const msg = e?.message ?? String(e);
  if (e instanceof py.OSError_) name = msg.startsWith('[Errno 2] ') ? 'FileNotFoundError' : 'OSError';
  if (name.endsWith('_')) name = name.slice(0, -1); // py.TypeError_ → TypeError
  return `${name}: ${msg}`;
}

/**
 * Run the primary SR algorithm on an .osu file.
 * mod: "NM" | "DT" | "HT" | "NC" (NC is treated as DT, same 1.5x rate); rate: custom speed rate
 * from Lazer (null = the mod's own rate). Returns a dict with sr, component maxes, success flag
 * and error string.
 */
export function analyze_primary_sr(file_path, mod = 'NM', rate = null) {
  const _core_mod = mod === 'NC' ? 'DT' : mod;

  const _MOD_RATE = { HT: 0.75, DT: 1.5, NC: 1.5, NM: 1.0 };
  const standard_rate = typeof _core_mod === 'string' ? py.get(_MOD_RATE, _core_mod, 1.0) : 1.0;

  // Native rates (0.75 / 1.0 / 1.5) run the engine directly with the matching mod; any other
  // rate is computed on a temp .osu with scaled hit times.
  let is_standard_rate;
  let native_mod;
  let native_rate;
  let _r = null;
  if (rate !== null && rate !== undefined) {
    _r = py.round(py.float(rate), 3);
    if (_r === 1.0) {
      is_standard_rate = true;
      native_mod = 'NM';
      native_rate = 1.0;
    } else if (_r === 0.75) {
      is_standard_rate = true;
      native_mod = 'HT';
      native_rate = 0.75;
    } else if (_r === 1.5) {
      is_standard_rate = true;
      native_mod = 'DT';
      native_rate = 1.5;
    } else {
      is_standard_rate = false;
      native_mod = _core_mod;
      native_rate = standard_rate;
    }
  } else {
    is_standard_rate = true;
    native_mod = _core_mod;
    native_rate = standard_rate;
  }

  const effective_rate = native_rate;

  const _rate_for_key = !is_standard_rate ? py.float(rate) : effective_rate;
  const key = _sr_cache_key(file_path, native_mod, _rate_for_key);
  const keyText = tupleKey(key);
  // a NaN rate never matches a cached key in Python (NaN != NaN)
  if (key[3] === key[3] && _sr_cache.has(keyText)) return { ..._sr_cache.get(keyText) };

  try {
    const alg = _import_sr_core();

    let target_path = file_path;
    let target_mod = native_mod;
    let temp_file = null;

    if (!is_standard_rate) {
      temp_file = _scale_osu_timings(file_path, _r);
      target_path = temp_file;
      target_mod = 'NM';
    }

    let SR;
    let all_corners;
    let D_graph;
    let components;
    try {
      [SR, all_corners, D_graph, components] = alg.calculate(target_path, target_mod, srContexts.get(vfs.norm(file_path)));
    } finally {
      if (temp_file && vfs.path.exists(temp_file)) {
        try {
          vfs.removeFile(temp_file);
        } catch (e) {
          if (!(e instanceof py.OSError_)) throw e;
        }
      }
    }

    const Pbar = py.get(components, 'Pressing Intensity', Float64Array.of(0.0));
    const Abar = py.get(components, 'Unevenness', Float64Array.of(0.0));
    const Jbar = py.get(components, 'Same-Column Pressure', Float64Array.of(0.0));
    const Xbar = py.get(components, 'Cross-Column Pressure', Float64Array.of(0.0));

    const jbar_max = Jbar.length > 0 ? npMaxReduce(Jbar) : 0.0;
    const pbar_max = Pbar.length > 0 ? npMaxReduce(Pbar) : 0.0;
    const xbar_max = Xbar.length > 0 ? npMaxReduce(Xbar) : 0.0;
    const abar_mean = Abar.length > 0 ? np.mean(Abar) : 0.0;

    const comp_total = jbar_max + pbar_max + xbar_max + 1.0;
    const jbar_share = py.div(jbar_max, comp_total);
    const pbar_share = py.div(pbar_max, comp_total);
    const xbar_share = py.div(xbar_max, comp_total);

    const j_mean = jbar_max;
    const nj_mean = pbar_max + xbar_max;
    const jack_ratio = py.div(j_mean, py.max2(j_mean + nj_mean, 1e-9));

    // Strain graph, downsampled to at most 300 points.
    let strain_graph;
    if (D_graph !== null && D_graph !== undefined && D_graph.length > 1) {
      const n = D_graph.length;
      let strain_sampled;
      let times_sampled;
      if (n > 300) {
        const indices = linspaceIndices(n);
        strain_sampled = indices.map((i) => py.round(D_graph[i], 4));
        times_sampled = indices.map((i) => py.round(all_corners[i], 2));
      } else {
        strain_sampled = Array.from(D_graph, (v) => py.round(v, 4));
        times_sampled = Array.from(all_corners, (t) => py.round(t, 2));
      }
      strain_graph = { values: strain_sampled, times: times_sampled };
    } else {
      strain_graph = null;
    }

    const result = {
      sr: py.round(SR, 4),
      jack_ratio: py.round(jack_ratio, 4),
      jbar_max: py.round(jbar_max, 4),
      pbar_max: py.round(pbar_max, 4),
      xbar_max: py.round(xbar_max, 4),
      abar_mean: py.round(abar_mean, 4),
      rbar_max: 0.0,
      d93: 0.0,
      d83: 0.0,
      d_weighted_mean: 0.0,
      total_notes_eff: D_graph !== null && D_graph !== undefined ? D_graph.length : 0,
      jbar_share: py.round(jbar_share, 4),
      pbar_share: py.round(pbar_share, 4),
      xbar_share: py.round(xbar_share, 4),
      strain_graph,
      success: true,
      error: null,
    };
    if (key[3] === key[3]) _sr_cache.set(keyText, result);
    return result;
  } catch (exc) {
    const result = { ..._EMPTY_RESULT };
    result.error = excText(exc);
    return result;
  }
}
