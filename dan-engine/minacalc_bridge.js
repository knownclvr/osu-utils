// minacalc_bridge.js — MinaCalc MSD bridge: .osu rows -> MinaCalc (msd.js, in-process), with caching.
//
// JavaScript port of engine/dan_engine/minacalc_bridge.py (DanOverlay 3.1.0 by 8doul).
//
// Parses a .osu file, converts hit objects to Etterna row format and hands them to
// MinaCalc. Returns a dict of 8 MSD skillset scores. Results are cached by
// (path, mtime, rate) so the same map is only computed once per session.
//
// Public API
//   is_available() -> bool
//   calc(osu_path, rate=1.0, goal=0.96, mode="ssr") -> dict | None
//   dominant_skillset(scores) -> str
//   derive_family(scores) -> str
//
// goal and mode are accepted for API compatibility but are not forwarded (msd.exe
// computes pure MSD regardless of goal/mode).
//
// The Python module pipes json.dumps(rows) to msd.exe through subprocess.run; here that
// single call is msd.runMsd(rows), which returns { returncode, data } with data equal to
// what json.loads(stdout) gives (null on failure). Everything around it is unchanged.
import * as py from './py.js';
import * as vfs from './vfs.js';
import { runMsd } from './msd.js';

const { path, sys } = vfs;

// ── Binary location ────────────────────────────────────────────────

// Frozen build (always the case here): modules live directly in sys._MEIPASS.
export const _THIS_DIR = sys._MEIPASS;
export const _REPO_ROOT = path.normpath(path.join(_THIS_DIR, '..', '..'));

/** Return the path to msd.exe, or None if not found. */
export function _find_cli() {
  let candidates;
  if (py.truthy(sys.frozen ?? false)) {
    // Frozen build: msd.exe is bundled in the resource dir (sys._MEIPASS).
    const base = sys._MEIPASS;
    candidates = [path.join(base, 'msd.exe'), path.join(base, 'msd')];
  } else {
    // Dev layout: <repo>/tools/bin/msd(.exe)
    candidates = [
      path.join(_REPO_ROOT, 'tools', 'bin', 'msd.exe'),
      path.join(_REPO_ROOT, 'tools', 'bin', 'msd'),
    ];
  }

  for (const p of candidates) {
    const norm = path.normpath(p);
    if (path.isfile(norm)) return norm;
  }
  return null;
}

export let _CLI_PATH = _find_cli();

/**
 * _CLI_PATH as Python sees it after `import minacalc_bridge`. Python resolves it once at
 * import; here the /res/msd.exe placeholder is created by setupEngine(), which may run after
 * this module was imported (static imports), so a missing CLI is looked up again on use.
 */
function _cli_path() {
  if (_CLI_PATH === null) _CLI_PATH = _find_cli();
  return _CLI_PATH;
}

// ── Cache ──────────────────────────────────────────────────────────

/** dict[tuple, dict]: keyed by _cache_key_str(_cache_key(...)). */
export const _cache = new Map();

/** Hashable form of a (abspath, mtime, rate) key with Python's tuple equality (1 == 1.0, -0.0 == 0.0). */
function _cache_key_str(key) {
  const part = (v) => {
    if (typeof v === 'number') return `n${v === 0 ? 0 : v}`;
    if (typeof v === 'boolean') return `n${v ? 1 : 0}`;
    if (typeof v === 'string') return `s${v}`;
    return `o${String(v)}`;
  };
  return key.map(part).join('\u0000');
}

export function _cache_key(osu_path, rate) {
  if (typeof osu_path !== 'string') {
    throw new py.TypeError_(`stat: path should be string, bytes, os.PathLike or integer, not ${osu_path === null || osu_path === undefined ? 'NoneType' : typeof osu_path}`);
  }
  let mtime;
  try {
    mtime = vfs.statMtimeNs(osu_path);
  } catch (e) {
    if (!(e instanceof py.OSError_)) throw e;
    mtime = 0;
  }
  return [path.abspath(osu_path), mtime, rate];
}

// ── .osu → Etterna rows ────────────────────────────────────────────

/**
 * Parse a .osu file and return [rows, note_count, duration_s].
 *
 * Returns None if the file is unreadable or not a 4K/7K mania map.
 * Hit object timestamps are scaled by *rate* (e.g. rate=1.5 for DT).
 */
export function _parse_osu_rows(osu_path, rate = 1.0) {
  let keycount = 0;
  let in_difficulty_section = false;
  let in_hitobjects_section = false;
  const raw_hits = [];

  let text;
  try {
    text = vfs.readText(osu_path, 'replace');
  } catch (e) {
    if (e instanceof py.OSError_) return null;
    throw e;
  }
  for (const raw of py.textLines(text)) {
    const line = py.strip(raw);

    // Section headers
    if (line.startsWith('[') && line.endsWith(']')) {
      in_difficulty_section = line === '[Difficulty]';
      in_hitobjects_section = line === '[HitObjects]';
      continue;
    }

    if (in_difficulty_section && line.startsWith('CircleSize:')) {
      try {
        keycount = py.int(py.float(py.strip(py.split(line, ':', 1)[1])));
      } catch (e) {
        if (!(e instanceof py.ValueError || e instanceof py.IndexError)) throw e;
      }
      continue;
    }

    if (in_hitobjects_section && line) {
      const parts = line.split(',');
      if (parts.length < 4) continue;
      try {
        raw_hits.push([py.int(parts[0]), py.int(parts[2])]);
      } catch (e) {
        if (!(e instanceof py.ValueError || e instanceof py.IndexError)) throw e;
        continue;
      }
    }
  }

  if (!(keycount === 4 || keycount === 7) || !raw_hits.length) return null;

  // raw_ms / rate: Python raises TypeError for a non-number rate (bool counts as int)
  const r = typeof rate === 'boolean' ? (rate ? 1 : 0) : rate;
  if (typeof r !== 'number') {
    throw new py.TypeError_(`unsupported operand type(s) for /: 'int' and '${r === null || r === undefined ? 'NoneType' : typeof r}'`);
  }

  // Column index from x (osu!mania lanes), row bitmask keyed by time in seconds.
  const column_width = 512.0 / keycount;
  // dict {time_s: bits}: entries in insertion order; the first key object is kept (-0.0 like a
  // dict), and every NaN is its own key (Python hashes NaN by identity)
  const rows = new Map();
  const entries = [];
  for (const [x, raw_ms] of raw_hits) {
    // py.int() of a > 309-digit literal is Infinity here; Python fails on the int -> float step
    if (!Number.isFinite(raw_ms)) throw new py.OverflowError('int too large to convert to float');
    const time_s = py.round(py.div(raw_ms, r) / 1000.0, 4);
    if (!Number.isFinite(x)) throw new py.OverflowError('int too large to convert to float');
    const column = py.min2(py.int(py.floordiv(x, column_width)), keycount - 1);
    if (column < 0) throw new py.ValueError('negative shift count');
    let entry = Number.isNaN(time_s) ? undefined : rows.get(time_s);
    if (entry === undefined) {
      entry = { t: time_s, bits: 0 };
      entries.push(entry);
      if (!Number.isNaN(time_s)) rows.set(time_s, entry);
    }
    entry.bits |= 1 << column;
  }

  const ordered = py.sortedBy(entries, (e) => e.t);
  const etterna_rows = ordered.map((e) => ({ notes: e.bits, time: e.t }));
  const times = ordered.map((e) => e.t);
  const duration_s = times.length >= 2 ? times[times.length - 1] - times[0] : 0.0;

  return [etterna_rows, raw_hits.length, duration_s];
}

// ── Public API ─────────────────────────────────────────────────────

export const _SKILLSET_KEYS = [
  'overall', 'stream', 'jumpstream', 'handstream',
  'stamina', 'jackspeed', 'chordjack', 'technical',
];

/** Return True if msd.exe is present and executable. */
export function is_available() {
  return _cli_path() !== null;
}

/**
 * Run MinaCalc MSD on *osu_path* and return the skillset score dict.
 *
 * Returns None on any error (binary not found, parse failure, non-4K map, MinaCalc failure,
 * or unexpected output format). The dict contains overall, stream, jumpstream, handstream,
 * stamina, jackspeed, chordjack, technical (MSD floats), duration_s and note_count.
 *
 * Custom rates (anything other than HT 0.75x / NM 1.0x / DT 1.5x) are linearly interpolated
 * between the two nearest native rates (linear extrapolation outside [0.75, 1.5]).
 */
export function calc(osu_path, rate = 1.0, goal = 0.96, mode = 'ssr') {
  if (_cli_path() === null) return null;

  const _r = py.round(py.float(rate), 4);
  const _NATIVE = [0.75, 1.0, 1.5];
  if (_NATIVE.includes(_r)) return _calc_native(osu_path, _r);

  // Bracketing native anchors (linear extrapolation outside [0.75, 1.5]).
  let lo;
  let hi;
  let t;
  if (_r < _NATIVE[0]) {
    lo = _NATIVE[0];
    hi = _NATIVE[1];
    t = (_r - lo) / py.max2(hi - lo, 1e-9);
  } else if (_r > _NATIVE[_NATIVE.length - 1]) {
    lo = _NATIVE[_NATIVE.length - 2];
    hi = _NATIVE[_NATIVE.length - 1];
    t = 1.0 + (_r - hi) / py.max2(hi - lo, 1e-9);
  } else {
    lo = null;
    hi = null;
    for (let i = 0; i < _NATIVE.length - 1; i++) {
      if (_NATIVE[i] <= _r && _r <= _NATIVE[i + 1]) {
        lo = _NATIVE[i];
        hi = _NATIVE[i + 1];
        break;
      }
    }
    // only reachable for a NaN rate: Python evaluates `_r - None`
    if (lo === null) throw new py.TypeError_("unsupported operand type(s) for -: 'float' and 'NoneType'");
    t = (_r - lo) / py.max2(hi - lo, 1e-9);
  }

  const res_lo = _calc_native(osu_path, lo);
  const res_hi = _calc_native(osu_path, hi);
  if (res_lo === null || res_hi === null) return null;

  const out = {};
  for (const [k, v_lo] of Object.entries(res_lo)) {
    if (k === 'duration_s' || k === 'note_count') {
      out[k] = v_lo;
      continue;
    }
    const got = py.get(res_hi, k, v_lo);
    const v_hi = py.float(py.truthy(got) ? got : v_lo);
    const f_lo = py.float(v_lo);
    const interpolated = f_lo + t * (v_hi - f_lo);
    out[k] = py.round(py.max2(0.0, interpolated), 4);
  }

  const key = _cache_key(osu_path, rate);
  _cache.set(_cache_key_str(key), out);
  return out;
}

/** Compute MSD at a native rate only (0.75 / 1.0 / 1.5). */
export function _calc_native(osu_path, rate) {
  const key = _cache_key_str(_cache_key(osu_path, rate));
  if (_cache.has(key)) return _cache.get(key);

  const parsed = _parse_osu_rows(osu_path, rate);
  if (parsed === null) return null;

  const [etterna_rows, note_count, duration_s] = parsed;

  // subprocess.run([_CLI_PATH], input=json.dumps(etterna_rows), ...) → MinaCalc in-process.
  // A crash of the calculator is a failed subprocess (non-zero return code) → None.
  let result;
  try {
    result = runMsd(etterna_rows);
  } catch {
    return null;
  }

  if (result.returncode !== 0 || result.data === null || result.data === undefined) return null;

  let data = result.data;
  if (typeof data !== 'object' || Array.isArray(data) || Object.prototype.hasOwnProperty.call(data, 'error')) return null;

  // Normalise skillset keys to lower case.
  const lowered = {};
  for (const [k, v] of Object.entries(data)) lowered[k.toLowerCase()] = v;
  data = lowered;

  if (!_SKILLSET_KEYS.some((k) => Object.prototype.hasOwnProperty.call(data, k))) return null;

  data.duration_s = duration_s;
  data.note_count = note_count;

  _cache.set(key, data);
  return data;
}

/** Return the name of the highest non-overall skillset. */
export function dominant_skillset(scores) {
  const skills = ['stream', 'jumpstream', 'handstream', 'stamina',
    'jackspeed', 'chordjack', 'technical'];
  return py.max(skills, (k) => py.get(scores, k, 0.0));
}

/** Map MinaCalc dominant skillset to a family label used by the overlay. */
export function derive_family(scores) {
  const dom = dominant_skillset(scores);
  const overall = py.get(scores, 'overall', 0.0);
  if (overall === 0 || overall === false) return 'hybrid';

  const mapping = {
    technical: 'tech',
    chordjack: 'jack',
    jackspeed: 'jack',
    handstream: 'stamina',
    stamina: 'stamina',
    stream: 'stream',
    jumpstream: 'stream',
  };
  const base_family = py.get(mapping, dom, 'hybrid');

  // Too close between the top two skillsets -> hybrid.
  const skills = ['stream', 'jumpstream', 'handstream', 'stamina',
    'jackspeed', 'chordjack', 'technical'];
  const vals = py.sortedBy(skills.map((s) => py.get(scores, s, 0.0)), (v) => v, true);
  if (vals.length >= 2 && vals[0] > 0) {
    const gap = (vals[0] - vals[1]) / vals[0];
    if (gap < 0.08) return 'hybrid';
  }

  return base_family;
}
