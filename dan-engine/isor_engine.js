// isor_engine.js — ISOR alternate dan-estimation engine (SR / choke / MSD triangulation + ridge corrector).
//
// JavaScript port of engine/dan_engine/isor_engine.py (DanOverlay 3.1.0 by 8doul).
//
// ISOR: Isotonic Strain Organic Residual. Alternate dan-estimation engine for
// DanOverlay (osu!mania 4K rice). Triangulates three continuous human-aligned
// difficulty vectors (dp_sr, dp_choke, dp_msd), refines them with an
// isotonic-trained ridge corrector (dual-band, benchmark-calibrated), and rescues
// the official dan-ladder top scale with a continuous gate. Rate-aware and
// monotonic in rate (isotonic SR clamp + native-anchor DP floor).
import * as py from './py.js';
import * as np from './np.js';
import * as vfs from './vfs.js';
import { parsear_osu_v2 } from './parser.js';
import { validate_domain } from './validator.js';
import { extract_features } from './feature_extractor.js';
import { analyze_primary_sr } from './primary_sr_bridge.js';
import { classify_family } from './classifier.js';
import { classify_from_parsed } from './rhythm_profile.js';
import { calc as calculate_msd } from './minacalc_bridge.js';
import { BIO_CONFIG, build_bio_rows, biomech_numeric } from './strain.js';

// ── local helpers (Python semantics) ─────────────────────────────────────
/** isinstance(x, dict): plain objects (JSON / dict literals) and Maps. */
function _isDict(x) {
  if (x instanceof Map) return true;
  if (x === null || typeof x !== 'object') return false;
  const proto = Object.getPrototypeOf(x);
  return proto === Object.prototype || proto === null;
}

function _typeName(x) {
  if (x === null || x === undefined) return 'NoneType';
  if (typeof x === 'string') return 'str';
  if (Array.isArray(x)) return 'list';
  if (_isDict(x)) return 'dict';
  return typeof x;
}

/** A value used in arithmetic: Python raises TypeError for None / str / containers. */
function _num(x) {
  if (typeof x === 'number') return x;
  if (typeof x === 'boolean') return x ? 1 : 0;
  throw new py.TypeError_(`unsupported operand type(s) for arithmetic: '${_typeName(x)}'`);
}

/** float(x or d) */
const _f = (x, d = 0.0) => py.float(py.truthy(x) ? x : d);

/** str(x) for the values that reach strings here (None, bools, str, numbers). */
function _str(x) {
  if (x === null || x === undefined) return 'None';
  if (x === true) return 'True';
  if (x === false) return 'False';
  if (typeof x === 'number') return Number.isInteger(x) ? py.reprInt(x) : py.reprFloat(x);
  return String(x);
}

/** np.array(list, dtype=float) for a flat list of numbers (None → nan like numpy). */
function _npFloatArray(list) {
  const out = new Float64Array(list.length);
  for (let i = 0; i < list.length; i++) {
    const x = list[i];
    out[i] = x === null || x === undefined ? NaN : typeof x === 'number' ? x : py.float(x);
  }
  return out;
}

/** Element-wise a <op> b for 1-D float arrays with numpy broadcasting of length-1 operands. */
function _npBinary(a, b, f) {
  const na = a.length;
  const nb = b.length;
  if (na !== nb && na !== 1 && nb !== 1) {
    throw new py.ValueError(`operands could not be broadcast together with shapes (${na},) (${nb},)`);
  }
  const n = na === 1 ? nb : na;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = f(a[na === 1 ? 0 : i], b[nb === 1 ? 0 : i]);
  return out;
}

/** a @ b for two 1-D float64 arrays (numpy matmul → DOUBLE_dot: 0. + cblas_ddot). */
function _matmul1d(a, b) {
  if (a.length !== b.length) {
    throw new py.ValueError(`matmul: Input operand 1 has a mismatch in its core dimension 0 (size ${b.length} is different from ${a.length})`);
  }
  return 0.0 + np.dot(a, b);
}

/** ThreadPoolExecutor stand-in: the task runs immediately; .result() returns or re-raises. */
class _ImmediateFuture {
  constructor(fn, args) {
    try {
      this._value = fn(...args);
      this._ok = true;
    } catch (e) {
      this._error = e;
      this._ok = false;
    }
  }
  result() {
    if (!this._ok) throw this._error;
    return this._value;
  }
}
export const _ISOR_POOL = {
  submit: (fn, ...args) => new _ImmediateFuture(fn, args),
};

// Virtual location of this module (only used for the ridge model's fallback paths).
export const _THIS_DIR = '/engine/dan_engine';
export const _PROJECT_ROOT = vfs.path.dirname(vfs.path.dirname(_THIS_DIR));

// Dan ladder: an integer DP tier n maps to DAN_ORDER[n - 1].
export const DAN_ORDER = [
  '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
  'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa',
];

export const DAN_DISPLAY = {
  '1st': '1st Dan', '2nd': '2nd Dan', '3rd': '3rd Dan',
  '4th': '4th Dan', '5th': '5th Dan', '6th': '6th Dan',
  '7th': '7th Dan', '8th': '8th Dan', '9th': '9th Dan',
  '10th': '10th Dan',
  Alpha: 'Alpha', Beta: 'Beta', Gamma: 'Gamma',
  Delta: 'Delta', Epsilon: 'Epsilon',
  Zeta: 'Zeta', Eta: 'Eta', Theta: 'Theta',
  Iota: 'Iota', Kappa: 'Kappa',
};

export const SUBLEVEL_QUINTILES = [
  [0.0, 0.2, 'Low'],
  [0.2, 0.4, 'Mid-Low'],
  [0.4, 0.6, 'Mid'],
  [0.6, 0.8, 'Mid-High'],
  [0.8, 1.0, 'High'],
];

// Per-skillset rulers: mean value of each metric at every dan (dan -> value).
export const SKILLSET_SR_MEANS = {
  jack: {
    '1st': 2.34, '2nd': 2.35, '3rd': 3.17, '4th': 3.48, '5th': 3.98,
    '6th': 4.75, '7th': 4.97, '8th': 5.84, '9th': 5.85, '10th': 6.3,
    Alpha: 6.45, Beta: 6.78, Gamma: 6.95, Delta: 7.85, Epsilon: 8.76,
    Zeta: 9.35, Eta: 10.38, Theta: 10.96, Iota: 12.27, Kappa: 13.13,
  },
  speed: {
    '1st': 2.94, '2nd': 3.5, '3rd': 3.78, '4th': 4.16, '5th': 4.86,
    '6th': 5.1, '7th': 5.25, '8th': 5.45, '9th': 5.7, '10th': 6.0,
    Alpha: 6.55, Beta: 6.8, Gamma: 7.45, Delta: 7.76, Epsilon: 9.16,
    Zeta: 9.55, Eta: 10.05, Theta: 10.67, Iota: 11.16, Kappa: 12.05,
  },
  stamina: {
    '1st': 3.38, '2nd': 3.48, '3rd': 3.79, '4th': 4.69, '5th': 5.23,
    '6th': 5.65, '7th': 5.75, '8th': 6.15, '9th': 6.26, '10th': 6.45,
    Alpha: 6.58, Beta: 6.88, Gamma: 7.35, Delta: 7.85, Epsilon: 8.92,
    Zeta: 9.6, Eta: 9.96, Theta: 10.81, Iota: 11.66, Kappa: 12.41,
  },
  tech: {
    '1st': 2.84, '2nd': 3.08, '3rd': 3.09, '4th': 3.9, '5th': 4.18,
    '6th': 4.5, '7th': 5.43, '8th': 5.69, '9th': 6.31, '10th': 6.55,
    Alpha: 6.62, Beta: 6.85, Gamma: 7.3, Delta: 8.17, Epsilon: 9.22,
    Zeta: 9.6, Eta: 10.25, Theta: 10.64, Iota: 11.69, Kappa: 12.1,
  },
  general: {
    '1st': 2.94, '2nd': 3.23, '3rd': 3.51, '4th': 4.16, '5th': 4.71,
    '6th': 5.12, '7th': 5.36, '8th': 5.8, '9th': 6.0, '10th': 6.25,
    Alpha: 6.5, Beta: 6.8, Gamma: 7.35, Delta: 7.85, Epsilon: 9.0,
    Zeta: 9.4, Eta: 10.13, Theta: 10.74, Iota: 11.68, Kappa: 12.25,
  },
};

// Raw choke rulers (the module-level SKILLSET_CHOKE_MEANS is their sanitized copy, below).
const _SKILLSET_CHOKE_MEANS_RAW = {
  speed: {
    '1st': 11.5, '2nd': 13.0, '3rd': 14.5, '4th': 16.0, '5th': 18.0,
    '6th': 18.5, '7th': 19.2, '8th': 19.8, '9th': 20.4, '10th': 21.0,
    Alpha: 25.2, Beta: 26.5, Gamma: 28.0, Delta: 30.0, Epsilon: 31.8,
    Zeta: 35.0, Eta: 36.5, Theta: 41.5, Iota: 45.0, Kappa: 49.0,
  },
  stamina: {
    '1st': 12.9, '2nd': 15.4, '3rd': 15.5, '4th': 17.8, '5th': 23.0,
    '6th': 20.6, '7th': 22.3, '8th': 23.5, '9th': 24.5, '10th': 24.0,
    Alpha: 24.4, Beta: 25.5, Gamma: 28.5, Delta: 31.4, Epsilon: 34.0,
    Zeta: 38.6, Eta: 42.7, Theta: 40.3, Iota: 45.9, Kappa: 50.0,
  },
  jack: {
    '1st': 10.5, '2nd': 12.0, '3rd': 10.5, '4th': 13.1, '5th': 14.3,
    '6th': 14.4, '7th': 17.9, '8th': 19.5, '9th': 20.5, '10th': 24.8,
    Alpha: 24.8, Beta: 26.5, Gamma: 27.2, Delta: 31.0, Epsilon: 34.0,
    Zeta: 35.2, Eta: 39.1, Theta: 45.0, Iota: 49.3, Kappa: 53.7,
  },
  tech: {
    '1st': 10.5, '2nd': 12.0, '3rd': 10.5, '4th': 13.1, '5th': 14.3,
    '6th': 14.4, '7th': 17.9, '8th': 19.5, '9th': 20.5, '10th': 21.8,
    Alpha: 22.5, Beta: 23.8, Gamma: 26.0, Delta: 26.8, Epsilon: 29.8,
    Zeta: 36.0, Eta: 39.2, Theta: 39.3, Iota: 42.4, Kappa: 46.6,
  },
  general: {
    '1st': 11.5, '2nd': 13.5, '3rd': 14.5, '4th': 16.0, '5th': 18.5,
    '6th': 19.5, '7th': 20.5, '8th': 21.5, '9th': 22.5, '10th': 22.8,
    Alpha: 24.2, Beta: 25.0, Gamma: 27.0, Delta: 30.0, Epsilon: 32.5,
    Zeta: 35.5, Eta: 38.5, Theta: 42.0, Iota: 46.0, Kappa: 50.5,
  },
};

/** Pool-adjacent-violators: closest non-decreasing fit of *values*. */
export function _isotonic_monotone(values) {
  const blocks = values.map((v) => [v]);
  let changed = true;
  while (changed) {
    changed = false;
    let i = 0;
    while (i < blocks.length - 1) {
      const mean_i = py.sum(blocks[i]) / blocks[i].length;
      const mean_j = py.sum(blocks[i + 1]) / blocks[i + 1].length;
      if (mean_i > mean_j) {
        blocks[i] = blocks[i].concat(blocks[i + 1]);
        blocks.splice(i + 1, 1);
        changed = true;
      } else {
        i += 1;
      }
    }
  }
  const flat = [];
  for (const b of blocks) {
    const m = py.sum(b) / b.length;
    for (let k = 0; k < b.length; k++) flat.push(m);
  }
  const out = [];
  for (let v of flat) {
    if (out.length && v <= out[out.length - 1]) v = out[out.length - 1] + 0.005;
    out.push(v);
  }
  return out;
}

/** Return a strictly monotone copy of a {dan: value} ruler. */
export function _sanitize_ruler(means_dict) {
  const values = DAN_ORDER.map((d) => py.float(py.getitem(means_dict, d)));
  const mono = _isotonic_monotone(values);
  const out = {};
  DAN_ORDER.forEach((d, i) => {
    out[d] = py.round(mono[i], 3);
  });
  return out;
}

export const SKILLSET_CHOKE_MEANS = Object.fromEntries(
  Object.entries(_SKILLSET_CHOKE_MEANS_RAW).map(([sk, ruler]) => [sk, _sanitize_ruler(ruler)]),
);

/** Report (do not modify) non-monotonic rulers used by the triangulation (printing dropped). */
export function _report_ruler_health(name, means_dict) {
  const vals = DAN_ORDER.map((d) => py.getitem(means_dict, d));
  const bad = [];
  for (let i = 1; i < vals.length; i++) if (vals[i] <= vals[i - 1]) bad.push(i + 1);
  return bad; // Python prints a warning when non-empty
}

for (const [_sk_name, _sk_ruler] of Object.entries(SKILLSET_SR_MEANS)) {
  _report_ruler_health(`SKILLSET_SR_MEANS[${_sk_name}]`, _sk_ruler);
}

export const MSD_MEANS = {
  '1st': 13.5, '2nd': 15.5, '3rd': 16.8, '4th': 18.5, '5th': 20.5,
  '6th': 22.0, '7th': 23.5, '8th': 24.8, '9th': 25.8, '10th': 26.8,
  Alpha: 27.5, Beta: 28.5, Gamma: 30.5, Delta: 33.0, Epsilon: 36.0,
  Zeta: 37.8, Eta: 40.5, Theta: 43.5, Iota: 46.5, Kappa: 50.0,
};

_report_ruler_health('MSD_MEANS', MSD_MEANS);

export function _sigmoid(x, k = 1.0, x0 = 0.0) {
  let z = k * (_num(x) - x0);
  z = z < 50.0 ? z : 50.0;   // min(50.0, z)  (NaN → 50.0, like Python)
  z = z > -50.0 ? z : -50.0; // max(-50.0, z)
  return 1.0 / (1.0 + py.math.exp(-z));
}

// Triangulation weights (SR / choke / MSD blend in compute_triangulated_rank).
export const _W_BASE_SR = 0.5;
export const _W_BASE_CHOKE = 0.3;
export const _W_BASE_MSD = 0.2;
export const _W_SR_FLAT_GAIN = 0.25;
export const _W_SR_CAP = 0.8;
export const _W_SR_FLAT_K = 1.0;
export const _W_SR_FLAT_X0 = 13.0;
export const _W_SR_SIGNAL_K = 2.0;
export const _W_SR_SIGNAL_X0 = 3.0;
export const _W_MSD_AGREE_GAIN = 0.15;
export const _W_MSD_CAP = 0.35;
export const _W_MSD_AGREE_K = 2.0;
export const _W_MSD_AGREE_X0 = 1.2;
export const _W_CHOKE_FLOOR = 0.05;

// BPM modulation of the SR vector.
export const _BPM_GAIN = 0.06;
export const _BPM_PURITY_K = 20.0;
export const _BPM_PURITY_X0 = 0.85;
export const _BPM_K = 0.08;
export const _BPM_X0 = 200.0;

// Chord (hand ratio) signal inside the 9.8-12.8 DP band.
export const _CHORD_GAIN = 0.35;
export const _CHORD_K = 25.0;
export const _CHORD_X0 = 0.12;

export const _BURST_GAIN = 0.08;
export const _JACK2_GAIN = 0.05;
export const _DAMP_GAIN = 0.2;
export const _STREAM_STAM_THR = 0.35;
export const _HYBRID_STAM_THR = 0.35;
export const _STAM_FAM_THR = 0.25;
export const _JD_JACK_THR = 0.15;

// Marathon / fatigue terms.
export const _MARA_CAP = 0.08;
export const _MARA_DUR_K = 0.02;
export const _MARA_DUR_X0 = 200.0;
export const _MARA_CV_REF = 0.4;
export const _FATIGUE_TOTAL_CAP = 0.12;

export function extract_choke_and_local_strains(notes, drain_s, window_ms = 10000, stride_ms = 250) {
  if (!py.truthy(notes) || _num(drain_s) <= 0) {
    return {
      choke_10s_nps: 0.0,
      mean_nps: 0.0,
      burst_ratio: 1.0,
      min_col_delta_ms: 9999.0,
      jack_burst_density: 0.0,
    };
  }

  const times = notes.map((n) => n[0]);
  const t_min = times[0];
  const t_max = times[times.length - 1];
  const total_notes = notes.length;
  const mean_nps = total_notes / py.max2(drain_s, 1.0);
  const n_times = times.length;

  // Densest window_ms window, sliding by stride_ms (two-pointer scan).
  let max_notes_window = 0;
  let t = t_min;
  let left_idx = 0;
  let right_idx = 0;
  while (t <= t_max) {
    const lo = t;
    const hi = t + window_ms;
    while (left_idx < n_times && times[left_idx] < lo) left_idx += 1;
    while (right_idx < n_times && times[right_idx] <= hi) right_idx += 1;
    const c = right_idx - left_idx;
    if (c > max_notes_window) max_notes_window = c;
    t += stride_ms;
  }

  const choke_10s_nps = py.div(max_notes_window, window_ms / 1000.0);
  const burst_ratio = choke_10s_nps / py.max2(mean_nps, 1.0);

  // Same-column deltas (jack detection).
  const by_col = new py.DefaultDict(() => []);
  for (const note of notes) {
    if (note.length !== 2) throw new py.ValueError(`too many values to unpack (expected 2, got ${note.length})`);
    const [t_val, c_val] = note;
    by_col.getOrCreate(c_val).push(t_val);
  }

  const col_deltas = [];
  let jack_burst_count = 0;
  for (const t_list of by_col.values()) {
    const t_sorted = py.sorted(t_list);
    for (let i = 1; i < t_sorted.length; i++) {
      const dt = t_sorted[i] - t_sorted[i - 1];
      col_deltas.push(dt);
      if (dt <= 120.0) jack_burst_count += 1;
    }
  }

  const min_col_delta = col_deltas.length ? py.min(col_deltas) : 9999.0;
  const jack_burst_density = jack_burst_count / py.max2(total_notes, 1.0);

  return {
    choke_10s_nps: py.round(choke_10s_nps, 2),
    mean_nps: py.round(mean_nps, 2),
    burst_ratio: py.round(burst_ratio, 3),
    min_col_delta_ms: py.round(min_col_delta, 1),
    jack_burst_density: py.round(jack_burst_density, 3),
  };
}

export function _build_ruler_boundaries(mean_dict) {
  const means = DAN_ORDER.map((d) => py.getitem(mean_dict, d));
  const n = means.length;
  const boundaries = [];
  for (let i = 0; i < n; i++) {
    let lo;
    let hi;
    if (i > 0) lo = (means[i - 1] + means[i]) / 2.0;
    else lo = means[0] - (means[1] - means[0]) / 2.0;

    if (i < n - 1) hi = (means[i] + means[i + 1]) / 2.0;
    else hi = means[n - 1] + (means[n - 1] - means[n - 2]) / 2.0;

    boundaries.push([lo, hi, i + 1]);
  }
  return boundaries;
}

// Python keys this cache by id(mean_dict); object identity is the JS equivalent.
export const _RULER_BOUNDARIES_CACHE = new WeakMap();

/** Converts any continuous scalar metric into a 1.0-20.99 DP coordinate. */
export function interpolate_ruler(val, mean_dict) {
  let boundaries = mean_dict !== null && typeof mean_dict === 'object' ? _RULER_BOUNDARIES_CACHE.get(mean_dict) : undefined;
  if (boundaries === undefined) {
    boundaries = _build_ruler_boundaries(mean_dict);
    _RULER_BOUNDARIES_CACHE.set(mean_dict, boundaries);
  }
  val = _num(val);

  if (val < boundaries[0][0]) {
    const t = py.max2(0.0, (val - (boundaries[0][0] - 1.0)) / 1.0);
    return 1.0 + t * 0.2;
  }

  if (val >= boundaries[boundaries.length - 1][1]) return 20.99;

  for (const [lo, hi, dp_base] of boundaries) {
    if (lo <= val && val < hi) {
      const t = (val - lo) / py.max2(hi - lo, 1e-06);
      return dp_base + t;
    }
  }

  return 20.0;
}

export const _BIO_ENABLED = false;
export const _BIO_W_BASE = 0.15;

// Ridge corrector (vsrg_ridge_model.json).
export const _RIDGE_ENABLED = true;
export const _RIDGE_CAP = 1.0;

export let _RIDGE_MODEL = null;
export let _RIDGE_HIGH_MEAN = null;
export let _RIDGE_HIGH_SCALE = null;
export let _RIDGE_HIGH_BETA = null;
export let _RIDGE_LOW_MEAN = null;
export let _RIDGE_LOW_SCALE = null;
export let _RIDGE_LOW_BETA = null;
export let _RIDGE_SINGLE_MEAN = null;
export let _RIDGE_SINGLE_SCALE = null;
export let _RIDGE_SINGLE_BETA = null;

// true once a candidate model file existed in the vfs when the loader ran
let _ridgeFileSeen = false;

/** Module-level ridge model load of the Python module (candidate paths, json.load, np arrays). */
function _load_ridge_model() {
  _RIDGE_MODEL = null;
  if (_RIDGE_ENABLED) {
    const _data_dir = vfs.path.join(vfs.path.dirname(_THIS_DIR), '03_data');
    const _candidate_paths = [
      vfs.path.join(_data_dir, 'vsrg_ridge_model.json'),
      vfs.path.join(_THIS_DIR, 'vsrg_ridge_model.json'),
      vfs.path.join(vfs.path.dirname(vfs.path.dirname(_THIS_DIR)), 'config', 'vsrg_ridge_model.json'),
    ];
    if (vfs.sys.frozen) {
      _candidate_paths.unshift(vfs.path.join(vfs.sys._MEIPASS, 'config', 'vsrg_ridge_model.json'));
    }
    for (const _rp of _candidate_paths) {
      if (vfs.path.exists(_rp)) {
        _ridgeFileSeen = true;
        try {
          _RIDGE_MODEL = JSON.parse(vfs.readText(_rp));
          break;
        } catch (e) {
          _RIDGE_MODEL = null;
        }
      }
    }
  }

  _RIDGE_HIGH_MEAN = null;
  _RIDGE_HIGH_SCALE = null;
  _RIDGE_HIGH_BETA = null;
  _RIDGE_LOW_MEAN = null;
  _RIDGE_LOW_SCALE = null;
  _RIDGE_LOW_BETA = null;
  _RIDGE_SINGLE_MEAN = null;
  _RIDGE_SINGLE_SCALE = null;
  _RIDGE_SINGLE_BETA = null;

  if (py.truthy(_RIDGE_MODEL)) {
    if (py.truthy(py.get(_RIDGE_MODEL, 'dual'))) {
      const _h = py.getitem(_RIDGE_MODEL, 'high');
      const _l = py.getitem(_RIDGE_MODEL, 'low');
      _RIDGE_HIGH_MEAN = _npFloatArray(py.getitem(_h, 'mean'));
      _RIDGE_HIGH_SCALE = _npFloatArray(py.getitem(_h, 'scale'));
      _RIDGE_HIGH_BETA = _npFloatArray(py.getitem(_h, 'beta'));
      _RIDGE_LOW_MEAN = _npFloatArray(py.getitem(_l, 'mean'));
      _RIDGE_LOW_SCALE = _npFloatArray(py.getitem(_l, 'scale'));
      _RIDGE_LOW_BETA = _npFloatArray(py.getitem(_l, 'beta'));
    } else {
      _RIDGE_SINGLE_MEAN = _npFloatArray(py.get(_RIDGE_MODEL, 'mean', []));
      _RIDGE_SINGLE_SCALE = _npFloatArray(py.get(_RIDGE_MODEL, 'scale', []));
      _RIDGE_SINGLE_BETA = _npFloatArray(py.get(_RIDGE_MODEL, 'beta', []));
    }
  }
}

/**
 * Python loads the model when the module is imported. If this module was imported before
 * setupEngine() filled the vfs (no candidate file existed yet), load it on first use instead.
 */
function _ensure_ridge_model() {
  if (_RIDGE_ENABLED && !_ridgeFileSeen) _load_ridge_model();
}

_load_ridge_model();

export const _RIDGE_STREAM_NAMES = ['speed', 'hand', 'jack', 'chordjack', 'tech', 'stamina', 'course'];
export const _RIDGE_STAT_KEYS = [
  'chord_rate', 'three_rate', 'overlap_rate', 'rotation_rate',
  'fast_jack_rate', 'anchor_rate', 'hand_bias', 'avg_nps', 'active_duration_s',
];

export const _RIDGE_SUSTAIN_KEYS = [
  'nps1k_q50', 'nps1k_q75', 'nps1k_q90',
  'nps4k_q50', 'nps4k_q75', 'nps4k_q90',
  'nps1k_flatness', 'log_nps1k_mean',
  'dt_row_q25', 'dt_row_q50', 'dt_row_q75',
  'dt_hand_q25', 'dt_hand_median',
  'dt_same_q25', 'dt_same_median',
];

export const _RIDGE_SUSTAIN_V2_KEYS = [
  'g_jack_npsq75', 'g_jack_dtrow',
  'g_speed_npsq75', 'g_speed_dtrow',
  'g_stam_npsq75', 'g_stam_dtrow',
  'g_tech_npsq75', 'g_tech_dtrow',
];
export const _RIDGE_SUSTAIN_CENTER_NPS_Q75 = 19.5;
export const _RIDGE_SUSTAIN_CENTER_DT_ROW = 90.0;

/** Build the feature vector exactly as in tune_offline._ridge_feats. */
export function _ridge_feature_vector(bio_res, msd_dict, sunny_strains = null, purity = 0.0, subrank = null) {
  _ensure_ridge_model();
  const feats = py.truthy(bio_res) ? bio_res : {};
  const summaries = py.get(feats, 'summaries', {});
  const streams = py.get(feats, 'streams', {});
  const streams_q97 = py.get(feats, 'streams_q97', {});
  const stats = py.get(feats, 'stats', {});
  const keys = _RIDGE_STREAM_NAMES;

  const v = [];
  for (const k of keys) {
    if (py.contains(summaries, k) && _isDict(py.getitem(summaries, k))) {
      v.push(_f(py.get(py.getitem(summaries, k), 'aggregate', 0.0)));
    } else {
      v.push(_f(py.get(streams, k, 0.0)));
    }
  }

  // summaries[k]["aggregate"] is used raw (no float()/or) when summaries[k] is a dict
  const agg = (name) => (_isDict(py.get(summaries, name))
    ? py.get(py.get(summaries, name, {}), 'aggregate', 0.0)
    : _f(py.get(streams, name, 0.0)));
  const jack_agg = agg('jack');
  const chordjack_agg = agg('chordjack');
  const speed_agg = agg('speed');
  const stamina_agg = agg('stamina');

  const raw_agg = py.get(feats, 'raw_agg', 0.0);
  const log_raw = py.math.log1p(py.max2(0.0, _num(py.truthy(raw_agg) ? raw_agg : 0.0)));
  v.push(log_raw);
  for (const k of _RIDGE_STAT_KEYS) v.push(_f(py.get(stats, k, 0.0)));
  for (const k of keys) {
    if (py.contains(summaries, k) && _isDict(py.getitem(summaries, k))) {
      v.push(_f(py.get(py.getitem(summaries, k), 'q97', 0.0)));
    } else {
      v.push(_f(py.get(streams_q97, k, 0.0)));
    }
  }
  const msd = py.truthy(msd_dict) ? msd_dict : {};
  const roles = {
    msd_jack: py.max2(_f(py.get(msd, 'jackspeed', 0.0)), _f(py.get(msd, 'chordjack', 0.0))),
    msd_speed: py.max2(_f(py.get(msd, 'stream', 0.0)), _f(py.get(msd, 'jumpstream', 0.0))),
    msd_stamina: 0.7 * _f(py.get(msd, 'stamina', 0.0)) + 0.3 * _f(py.get(msd, 'handstream', 0.0)),
    msd_tech: _f(py.get(msd, 'technical', 0.0)),
  };
  const roles_sum = py.sum(Object.values(roles));
  const total = py.truthy(roles_sum) ? roles_sum : 1.0;
  for (const k of ['msd_jack', 'msd_speed', 'msd_stamina', 'msd_tech']) v.push(py.div(roles[k], total));
  const sun = py.truthy(sunny_strains) ? sunny_strains : {};
  for (const k of ['jbar_max', 'pbar_max', 'xbar_max', 'abar_mean']) v.push(_f(py.get(sun, k, 0.0)));
  v.push(_f(py.get(msd, 'overall', 0.0)));
  for (const k of ['stream', 'jumpstream', 'handstream', 'stamina', 'jackspeed', 'chordjack', 'technical']) {
    v.push(_f(py.get(msd, k, 0.0)));
  }
  const stat = (k) => _f(py.get(stats, k, 0.0));
  v.push(
    stat('chord_rate') * stat('fast_jack_rate'),
    stat('overlap_rate') * stat('rotation_rate'),
    py.div(_num(jack_agg), _num(chordjack_agg) + 1e-06),
    _num(speed_agg) * _f(purity),
    stat('anchor_imbalance'),
    stat('peak_to_sustain_gap'),
    stat('break_density'),
    stat('fast_jack_rate') * py.div(100.0, stat('same_hand_q10') + 10.0),
    stat('rotation_rate') * _f(py.get(subrank, 'ti', 0.0)),
    _f(py.get(subrank, 'burst_ratio', 1.0), 1.0) * stat('peak_to_sustain_gap'),
    stat('avg_nps') * _f(py.get(subrank, 'dur', 0.0)) / 60.0,
    _f(py.get(subrank, 'quad', 0.0)) * _f(py.get(subrank, 'hand', 0.0)),
    stat('rotation_rate') * (1.0 - stat('overlap_rate')),
    py.div(_f(py.get(subrank, 'choke', 0.0)), stat('avg_nps') + 1.0),
    Math.abs(_f(py.get(subrank, 'dp_sr', 0.0)) - _f(py.get(subrank, 'dp_msd', 0.0))),
  );

  const sub = py.truthy(subrank) ? subrank : {};
  const dp_sr = _f(py.get(sub, 'dp_sr', 0.0));
  const dp_choke = _f(py.get(sub, 'dp_choke', 0.0));
  const dp_msd = _f(py.get(sub, 'dp_msd', 0.0));
  const mod_sr = _f(py.get(sub, 'mod_sr', 0.0));
  const raw_sr = _f(py.get(sub, 'raw_sr', 0.0));
  const jbar = _f(py.get(sun, 'jbar_max', 0.0));
  const pbar = _f(py.get(sun, 'pbar_max', 0.0));
  const msd_overall = _f(py.get(msd, 'overall', 0.0));

  v.push(
    py.math.sqrt(py.max2(0.0, log_raw)),
    py.pow(log_raw, 2),
    py.div(raw_sr, dp_sr + 1e-06),
    py.div(dp_choke, dp_msd + 1e-06),
    mod_sr - raw_sr,
    (dp_sr + dp_choke + dp_msd) / 3.0,
    py.max2(0.0, dp_msd - dp_sr),
    py.max2(0.0, dp_sr - dp_msd),
    py.max2(0.0, dp_choke - dp_sr),
    Math.abs(jbar - pbar),
    py.div(jbar, pbar + 1e-06),
    py.div(msd_overall, raw_sr * 3.5 + 1e-06),
  );

  const bpm_val = _f(py.get(sub, 'bpm', 180.0), 180.0);
  const bpm_excess = py.max2(0.0, bpm_val - 240.0) / 100.0;
  v.push(
    bpm_excess,
    bpm_excess * _num(stamina_agg),
    bpm_excess * _num(speed_agg),
    py.div(_num(stamina_agg), dp_choke + 1e-06),
    py.div(_num(jack_agg), dp_choke + 1e-06),
    py.div(_num(speed_agg), dp_sr + 1e-06),
  );

  const purity_val = _f(purity);
  const dur_val = _f(py.get(sub, 'dur', 120.0), 120.0);
  const avg_nps = _f(py.get(stats, 'avg_nps', 0.0));
  v.push(
    (dp_choke - dp_sr) * purity_val,
    (dp_msd - dp_sr) * purity_val,
    py.math.log1p(py.max2(0.0, dur_val)) / py.math.log1p(180.0),
    py.div(bpm_val, avg_nps * 12.0 + 1e-06),
  );
  v.push(dp_sr, dp_choke, dp_msd, mod_sr, raw_sr);
  const q10 = _f(py.get(stats, 'same_hand_q10', 0.0));
  v.push(py.max2(0.0, py.min2(1.0, (q10 - 60.0) / 40.0)));
  const fs = py.get(py.truthy(_RIDGE_MODEL) ? _RIDGE_MODEL : {}, 'feature_set');
  if (fs === 'sustain_v1' || fs === 'sustain_v2') {
    for (const k of _RIDGE_SUSTAIN_KEYS) v.push(_f(py.get(stats, k, 0.0)));
  }
  if (fs === 'sustain_v2') {
    const skRaw = py.get(py.truthy(subrank) ? subrank : {}, 'sk_key', '');
    const sk = _str(py.truthy(skRaw) ? skRaw : '');
    const nps_q75 = _f(py.get(stats, 'nps1k_q75', 0.0));
    const dt_row = _f(py.get(stats, 'dt_row_q50', 0.0));
    for (const tag of ['jack', 'speed', 'stamina', 'tech']) {
      const oh = sk === tag ? 1.0 : 0.0;
      v.push(
        oh * (nps_q75 - _RIDGE_SUSTAIN_CENTER_NPS_Q75) / 10.0,
        oh * (dt_row - _RIDGE_SUSTAIN_CENTER_DT_ROW) / 100.0,
      );
    }
  }
  return v;
}

/** Return [dp_corrected, correction] applying the calibrated ridge. */
export function apply_ridge_correction(dp_base, bio_res, msd_dict, sunny_strains = null, purity = 0.0, subrank = null) {
  _ensure_ridge_model();
  if (!_RIDGE_ENABLED || !py.truthy(_RIDGE_MODEL) || !py.truthy(bio_res)) return [dp_base, 0.0];
  try {
    const v = _npFloatArray(_ridge_feature_vector(bio_res, msd_dict, sunny_strains, purity, subrank));
    if (py.truthy(py.get(_RIDGE_MODEL, 'dual')) && _RIDGE_HIGH_BETA !== null) {
      const xh = _npBinary(_npBinary(v, _RIDGE_HIGH_MEAN, (a, b) => a - b), _RIDGE_HIGH_SCALE, (a, b) => a / b);
      const corr_h = py.float(np.clip(_matmul1d(xh, _RIDGE_HIGH_BETA), -_RIDGE_CAP, _RIDGE_CAP));

      const xl = _npBinary(_npBinary(v, _RIDGE_LOW_MEAN, (a, b) => a - b), _RIDGE_LOW_SCALE, (a, b) => a / b);
      const corr_l = py.float(np.clip(_matmul1d(xl, _RIDGE_LOW_BETA), -_RIDGE_CAP, _RIDGE_CAP));

      const gate = _sigmoid(dp_base, 4.0, 10.0);
      const correction = gate * corr_h + (1.0 - gate) * corr_l;
      return [py.round(_num(dp_base) + correction, 2), py.round(correction, 3)];
    }
    if (_RIDGE_SINGLE_BETA !== null) {
      if (v.length !== _RIDGE_SINGLE_BETA.length) return [dp_base, 0.0];
      const x = _npBinary(_npBinary(v, _RIDGE_SINGLE_MEAN, (a, b) => a - b), _RIDGE_SINGLE_SCALE, (a, b) => a / b);
      const correction = py.float(np.clip(_matmul1d(x, _RIDGE_SINGLE_BETA), -_RIDGE_CAP, _RIDGE_CAP));
      return [py.round(_num(dp_base) + correction, 2), py.round(correction, 3)];
    }
    return [dp_base, 0.0];
  } catch (e) {
    return [dp_base, 0.0];
  }
}

/** Triangulates SR, Choke Point 10s NPS, and MinaCalc MSD into human DP. */
export function compute_triangulated_rank(raw_sr, sunny_strains, choke_info, msd_dict, family, conf, features_ref = null, bio_numeric = null) {
  const choke_nps = py.get(choke_info, 'choke_10s_nps', 0.0);
  const burst_ratio = py.get(choke_info, 'burst_ratio', 1.0);
  const jbar_max = _f(py.get(sunny_strains, 'jbar_max', 0.0));

  // SR modulation: burst / jack / fatigue / marathon / BPM deltas.
  const burst_activation = _sigmoid(burst_ratio, 5.0, 1.35);
  const jbar_intensity = _sigmoid(jbar_max, 0.08, 60.0);
  let delta_burst = _BURST_GAIN * burst_activation * jbar_intensity;

  let delta_jack;
  if ((family === 'jack' || family === 'tech' || family === 'hybrid') && jbar_max > 55.0) {
    const jack_factor = _sigmoid(jbar_max, 0.1, 70.0);
    delta_jack = _JACK2_GAIN * jack_factor * py.min2(1.0, _num(py.get(choke_info, 'jack_burst_density', 0.0)) * 2.0);
  } else {
    delta_jack = 0.0;
  }

  let fatigue_spike;
  if (_num(burst_ratio) < 1.15 && _num(raw_sr) > 9.5 && (family === 'tech' || family === 'jack')) {
    fatigue_spike = _sigmoid(raw_sr, 2.0, 9.7) * (1.15 - burst_ratio) * 0.4;
    fatigue_spike = py.min2(0.06, fatigue_spike);
  } else {
    fatigue_spike = 0.0;
  }

  const has_features = py.truthy(features_ref);
  const duration_s = has_features ? _f(py.get(features_ref, 'duration_s', 0.0)) : 0.0;
  const density_cv = has_features ? _f(py.get(features_ref, 'density_cv', 0.0)) : 0.0;
  const mara_dur_sig = _sigmoid(duration_s, _MARA_DUR_K, _MARA_DUR_X0);
  const cv_pen = 1.0 - py.min2(1.0, density_cv / _MARA_CV_REF);
  const delta_marathon = _MARA_CAP * mara_dur_sig * cv_pen;
  const delta_fatigue = py.min2(_FATIGUE_TOTAL_CAP, fatigue_spike + delta_marathon);

  const stream_purity = has_features ? _f(py.get(features_ref, 'stream_purity', 0.0)) : 0.0;
  const bpm_val = has_features ? _f(py.get(features_ref, 'bpm', 0.0)) : 0.0;
  let delta_bpm = (
    _BPM_GAIN
    * _sigmoid(stream_purity, _BPM_PURITY_K, _BPM_PURITY_X0)
    * _sigmoid(bpm_val, _BPM_K, _BPM_X0)
  );

  const taper = py.max2(0.25, 1.0 - py.max2(0.0, _num(raw_sr) - 9.0) * 0.12);
  delta_burst *= taper;
  delta_jack *= taper;
  delta_bpm *= taper;

  const mod_sr = raw_sr * (1.0 + delta_burst + delta_jack + delta_bpm - delta_fatigue);

  // Skillset routing -> which rulers to read.
  const jump_ratio = has_features ? _f(py.get(features_ref, 'jump_ratio', 0.0)) : 0.0;
  const jack_density = has_features ? _f(py.get(features_ref, 'jack_density', 0.0)) : 0.0;
  const fam_lower = _str(family).toLowerCase();

  let sk_key;
  if (fam_lower.includes('jack') || jbar_max > 55.0) {
    sk_key = 'jack';
  } else if (fam_lower.includes('tech') || fam_lower.includes('ln')) {
    sk_key = 'tech';
  } else if (jack_density >= _JD_JACK_THR) {
    sk_key = 'jack';
  } else if (fam_lower === 'stream' || fam_lower === 'speed') {
    sk_key = jump_ratio >= _STREAM_STAM_THR ? 'stamina' : 'speed';
  } else if (fam_lower === 'chordstream' || fam_lower === 'stamina') {
    sk_key = jump_ratio >= _STAM_FAM_THR ? 'stamina' : 'speed';
  } else {
    sk_key = jump_ratio >= _HYBRID_STAM_THR ? 'stamina' : 'general';
  }

  const sr_ruler = py.get(SKILLSET_SR_MEANS, sk_key, SKILLSET_SR_MEANS.general);
  const choke_ruler = py.get(SKILLSET_CHOKE_MEANS, sk_key, SKILLSET_CHOKE_MEANS.general);

  const dp_sr = interpolate_ruler(mod_sr, sr_ruler);
  const dp_choke = interpolate_ruler(choke_nps, choke_ruler);

  let dp_msd;
  let has_msd;
  if (py.truthy(msd_dict) && _isDict(msd_dict) && py.contains(msd_dict, 'overall')) {
    const dominant_msd = _f(py.get(msd_dict, 'overall', 0.0));
    dp_msd = interpolate_ruler(dominant_msd, MSD_MEANS);
    has_msd = true;
  } else {
    dp_msd = dp_sr;
    has_msd = false;
  }

  let dp_bio = null;
  if (_BIO_ENABLED && bio_numeric !== null && bio_numeric !== undefined && _num(bio_numeric) > 0.0) {
    dp_bio = interpolate_ruler(bio_numeric, sr_ruler);
  }
  let w_sr;
  let w_choke;
  let w_msd;
  let w_bio;
  if (has_msd) {
    const choke_flat = _sigmoid(_W_SR_FLAT_X0 - _num(choke_nps), _W_SR_FLAT_K, 0.0);
    const sr_signal = _sigmoid(_num(raw_sr) - _W_SR_SIGNAL_X0, _W_SR_SIGNAL_K, 0.0);
    w_sr = _W_BASE_SR + _W_SR_FLAT_GAIN * choke_flat * sr_signal;
    w_sr = py.min2(_W_SR_CAP, w_sr);

    const disagree = _sigmoid(Math.abs(dp_sr - dp_choke) - _W_MSD_AGREE_X0, _W_MSD_AGREE_K, 0.0);
    w_msd = _W_BASE_MSD + _W_MSD_AGREE_GAIN * disagree;
    w_msd = py.min2(_W_MSD_CAP, w_msd);

    w_bio = dp_bio !== null ? _BIO_W_BASE : 0.0;
    w_choke = py.max2(_W_CHOKE_FLOOR, 1.0 - w_sr - w_msd - w_bio);
    const _w_total = w_sr + w_choke + w_msd + w_bio;
    w_sr = py.div(w_sr, _w_total);
    w_choke = py.div(w_choke, _w_total);
    w_msd = py.div(w_msd, _w_total);
    w_bio = py.truthy(w_bio) ? py.div(w_bio, _w_total) : 0.0;
  } else {
    w_sr = 0.6;
    w_choke = 0.4;
    w_msd = 0.0;
    w_bio = 0.0;
  }

  let raw_combined_dp = w_sr * dp_sr + w_choke * dp_choke + w_msd * dp_msd;
  if (dp_bio !== null && w_bio > 0.0) raw_combined_dp += w_bio * dp_bio;

  // Low-density damping.
  if (raw_combined_dp <= 4.5 && choke_nps < 18.0) {
    const low_density_factor = _sigmoid(16.0 - choke_nps, 0.5, 2.0);
    const dp_damped = raw_combined_dp - _DAMP_GAIN * low_density_factor * py.max2(0.0, raw_combined_dp - 1.5);
    raw_combined_dp = py.max2(1.0, dp_damped);
  }

  // 9.8-12.8 DP band: alpha / beta / chord signals.
  const hand_ratio = has_features ? _f(py.get(features_ref, 'hand_ratio', 0.0)) : 0.0;
  if (raw_combined_dp >= 9.8 && raw_combined_dp <= 12.8) {
    const alpha_signal = _sigmoid(choke_nps, 3.0, 25.2) * _sigmoid(hand_ratio, 20.0, 0.15);
    raw_combined_dp += 0.2 * alpha_signal;

    const beta_signal = _sigmoid(choke_nps, 3.0, 27.2);
    raw_combined_dp += 0.25 * beta_signal;

    const chord_signal = _sigmoid(hand_ratio, _CHORD_K, _CHORD_X0) - 0.5;
    raw_combined_dp += _CHORD_GAIN * chord_signal;
  }

  const final_dp = py.round(raw_combined_dp, 2);
  const tier_idx = Math.min(DAN_ORDER.length - 1, Math.max(0, py.int(final_dp) - 1));
  const dan_label = py.get(DAN_DISPLAY, DAN_ORDER[tier_idx], DAN_ORDER[tier_idx]);

  const frac = py.round(final_dp - py.math.floor(final_dp), 2);
  let sublevel = 'High';
  for (const [q_lo, q_hi, label] of SUBLEVEL_QUINTILES) {
    if (q_lo <= frac && frac <= q_hi) {
      sublevel = label;
      break;
    }
  }

  return {
    dp: final_dp,
    dan_label,
    sublevel,
    mod_sr: py.round(mod_sr, 4),
    dp_sr: py.round(dp_sr, 2),
    dp_choke: py.round(dp_choke, 2),
    dp_msd: py.round(dp_msd, 2),
    dp_bio: dp_bio !== null ? py.round(dp_bio, 2) : null,
    sk_key,
    weights: { w_sr: py.round(w_sr, 3), w_choke: py.round(w_choke, 3), w_msd: py.round(w_msd, 3), w_bio: py.round(w_bio, 3) },
  };
}

/**
 * Analyze one map with the prototype engine.
 *
 * mod: "NM" | "DT" | "HT" | "NC" (NC == DT rate).
 * rate: custom lazer clock rate (0.5-2.0) or null for the mod's native rate.
 * The SR bridge enforces non-decreasing SR in rate (isotonic clamp, per-map
 * registry), so a higher rate never shows a lower difficulty.
 */
export function analyze_beatmap_prototype(osu_path, mod = 'NM', rate = null, parsed = null, domain = null) {
  _ensure_ridge_model();
  if (!vfs.path.exists(osu_path)) return { error: 'file_not_found' };

  if (parsed === null || parsed === undefined) parsed = parsear_osu_v2(osu_path, null, true, true);
  if (py.truthy(py.get(parsed, 'rejected'))) return { error: 'rejected_domain' };

  if (domain === null || domain === undefined) domain = validate_domain(parsed);
  const drain_s = _f(py.get(domain, 'drain_time_s', 0.0));
  const notes = py.get(parsed, 'notes', []);

  const _rate_msd = py.truthy(rate) ? py.float(rate) : py.get({ HT: 0.75, DT: 1.5, NC: 1.5, NM: 1.0 }, mod, 1.0);

  // Independent analyses (a thread pool in Python) run here one after another.
  const fut_sunny = _ISOR_POOL.submit(analyze_primary_sr, osu_path, mod, rate);
  const fut_msd = _ISOR_POOL.submit(calculate_msd, osu_path, _rate_msd);
  const fut_feat = _ISOR_POOL.submit(extract_features, parsed);

  const _run_bio = () => {
    try {
      const bio_rows = build_bio_rows(py.get(parsed, 'rows', []), py.getitem(BIO_CONFIG, 'row_tolerance_ms'));
      const bio_res = biomech_numeric(bio_rows, BIO_CONFIG);
      return [bio_res, _f(py.get(bio_res, 'bio_numeric', 0.0))];
    } catch (e) {
      return [null, 0.0];
    }
  };

  const fut_bio = _ISOR_POOL.submit(_run_bio);
  const choke_info = extract_choke_and_local_strains(notes, drain_s);

  const sunny_res = fut_sunny.result();
  const msd_raw = fut_msd.result();
  const msd_res = py.truthy(msd_raw) ? msd_raw : {};
  const features = fut_feat.result();
  const [bio_res, bio_numeric] = fut_bio.result();

  if (!py.truthy(py.get(sunny_res, 'success'))) {
    return { error: `sunny_error: ${_str(py.get(sunny_res, 'error'))}` };
  }

  const raw_sr = py.float(py.get(sunny_res, 'sr', 0.0));
  const sunny_strains = {
    jbar_max: py.float(py.get(sunny_res, 'jbar_max', 0.0)),
    pbar_max: py.float(py.get(sunny_res, 'pbar_max', 0.0)),
    xbar_max: py.float(py.get(sunny_res, 'xbar_max', 0.0)),
    abar_mean: py.float(py.get(sunny_res, 'abar_mean', 0.0)),
  };

  const sr_gate = raw_sr;
  let classification;
  if (sr_gate < 7.0) {
    classification = classify_from_parsed(parsed);
    if (_num(py.get(classification, 'confidence', 0.0)) < 0.1) {
      classification = classify_family(sunny_res, features, domain);
    }
  } else {
    classification = classify_family(sunny_res, features, domain);
  }

  const family = py.get(classification, 'family', 'hybrid');
  const conf = py.get(classification, 'confidence', 0.7);

  const triang = compute_triangulated_rank(raw_sr, sunny_strains, choke_info, msd_res, family, conf, features, bio_numeric);

  // Per-skillset linear base (ridge model "lin_base"), blended in by a sigmoid gate on DP ("lin_blend").
  let base = py.getitem(triang, 'dp');
  if (py.truthy(_RIDGE_MODEL) && py.truthy(py.get(_RIDGE_MODEL, 'lin_base'))) {
    const _lb = py.getitem(_RIDGE_MODEL, 'lin_base');
    const _c = py.get(_lb, py.getitem(triang, 'sk_key'));
    const _coef = py.truthy(_c) ? _c : py.get(_lb, 'general');
    const _base_lin = (py.at(_coef, 0) + py.at(_coef, 1) * triang.dp_sr + py.at(_coef, 2) * triang.dp_choke
      + py.at(_coef, 3) * triang.dp_msd);
    const _blend = py.get(_RIDGE_MODEL, 'lin_blend', { k: 8.0, x0: 11.0 });
    const _w = _sigmoid(triang.dp, py.float(py.get(_blend, 'k', 8.0)), py.float(py.get(_blend, 'x0', 11.0)));
    base = _w * _base_lin + (1.0 - _w) * triang.dp;
  }

  // Continuous top-scale gate.
  const _GATE_LO = 18.8;
  const _GATE_HI = 19.2;
  const _TOP_TARGET = 20.2;
  if (base > _GATE_LO) {
    let _gt = py.min2(1.0, (base - _GATE_LO) / (_GATE_HI - _GATE_LO));
    _gt = 0.5 * (1.0 - py.math.cos(py.math.pi * _gt));
    const _lift = py.min2(0.6, py.max2(0.0, _TOP_TARGET - triang.dp_sr));
    base = _gt * (triang.dp_sr + _lift) + (1.0 - _gt) * base;
  }

  let dp_ridge = base;
  let ridge_corr = 0.0;
  const _ln_ratio = _f(py.get(domain, 'ln_ratio', 0.0));
  const _dual_low = py.truthy(_RIDGE_MODEL) && py.truthy(py.get(_RIDGE_MODEL, 'dual'));
  if (_ln_ratio <= 0.18 && (base >= 10.5 || (_dual_low && base >= 1.2))) {
    const _purity = _f(py.get(features, 'stream_purity', 0.0));
    const _subrank = {
      sk_key: py.get(triang, 'sk_key', ''),
      dp_sr: triang.dp_sr,
      dp_choke: triang.dp_choke,
      dp_msd: triang.dp_msd,
      mod_sr: triang.mod_sr,
      raw_sr,
      ti: _f(py.get(features, 'timing_irregularity', 0.0)),
      burst_ratio: _f(py.get(choke_info, 'burst_ratio', 1.0), 1.0),
      dur: py.float(drain_s),
      quad: _f(py.get(features, 'quad_ratio', 0.0)),
      hand: _f(py.get(features, 'hand_ratio', 0.0)),
      choke: _f(py.get(choke_info, 'choke_10s_nps', 0.0)),
      msd_overall: py.truthy(msd_res) ? _f(py.get(msd_res, 'overall', 0.0)) : 0.0,
      bpm: py.float(py.get(parsed, 'bpm', 0.0)),
    };
    [dp_ridge, ridge_corr] = apply_ridge_correction(base, bio_res, msd_res, sunny_strains, _purity, _subrank);
  }

  dp_ridge = py.min2(dp_ridge, 20.5);

  // Custom rates (not 0.75/1.0/1.5): floor DP at the nearest lower native anchor rate
  // (cap at 0.75x below it), keeping DP monotonic in rate.
  if (rate !== null && rate !== undefined) {
    const _r_val = py.round(py.float(rate), 4);
    if (!(_r_val === 0.75 || _r_val === 1.0 || _r_val === 1.5)) {
      if (_r_val > 0.75) {
        let _anchor;
        if (_r_val >= 1.5) _anchor = 1.5;
        else if (_r_val >= 1.0) _anchor = 1.0;
        else _anchor = 0.75;
        const _res_a = analyze_beatmap_prototype(osu_path, 'NM', _anchor, parsed, domain);
        if (!py.contains(_res_a, 'error') && py.get(_res_a, 'dp') !== null) {
          dp_ridge = py.max2(dp_ridge, py.getitem(_res_a, 'dp'));
        }
      } else {
        const _res_a = analyze_beatmap_prototype(osu_path, 'NM', 0.75, parsed, domain);
        if (!py.contains(_res_a, 'error') && py.get(_res_a, 'dp') !== null) {
          dp_ridge = py.min2(dp_ridge, py.getitem(_res_a, 'dp'));
        }
      }
    }
  }

  triang.dp = py.round(dp_ridge, 2);
  triang.ridge_correction = ridge_corr;

  const _tier_idx = Math.min(DAN_ORDER.length - 1, Math.max(0, py.int(triang.dp) - 1));
  triang.dan_short = DAN_ORDER[_tier_idx];
  triang.dan_label = py.get(DAN_DISPLAY, DAN_ORDER[_tier_idx], DAN_ORDER[_tier_idx]);

  const _frac = py.round(triang.dp - py.math.floor(triang.dp), 2);
  let _sublevel = 'High';
  for (const [q_lo, q_hi, label] of SUBLEVEL_QUINTILES) {
    if (q_lo <= _frac && _frac <= q_hi) {
      _sublevel = label;
      break;
    }
  }
  triang.sublevel = _sublevel;

  return {
    title: py.get(parsed, 'title', 'Unknown'),
    version: py.get(parsed, 'version', 'Unknown'),
    bpm: py.float(py.get(parsed, 'bpm', 0.0)),
    drain_s,
    note_count: py.int(py.get(domain, 'note_count', 0)),
    family,
    family_confidence: conf,
    raw_sr,
    modulated_sr: triang.mod_sr,
    dp: triang.dp,
    dp_sr: triang.dp_sr,
    dp_choke: triang.dp_choke,
    dp_msd: triang.dp_msd,
    dp_bio: py.get(triang, 'dp_bio'),
    bio_numeric: py.truthy(bio_numeric) ? py.round(bio_numeric, 4) : null,
    sk_key: py.get(triang, 'sk_key', ''),
    weights: py.get(triang, 'weights', {}),
    dan_label: triang.dan_label,
    dan_short: triang.dan_short,
    sublevel: triang.sublevel,
    ridge_correction: py.get(triang, 'ridge_correction', 0.0),
    choke_info,
    sunny_strains,
    strain_graph: py.get(sunny_res, 'strain_graph'),
    msd: msd_res,
    features,
  };
}

export const analyze_beatmap_isor = analyze_beatmap_prototype;
