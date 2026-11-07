// algorithm.js — numpy implementation of the "Sunny" (Star-Rating-Rebirth) osu!mania star rating + strain graph.
//
// JavaScript port of engine/dan_engine/sr_core/algorithm.py (DanOverlay 3.1.0 by 8doul).
//
// Bit-exactness notes (what numpy / CPython do on the reference machine):
//   * np.exp / np.cos / np.power and CPython float ** float call the Windows CRT → np.crtExp /
//     np.crtCos / np.crtPow (Math.exp / Math.cos / ** differ in the last bit far too often).
//   * ndarray ** python-scalar has numpy fast paths for 2 (x*x), 0.5 (sqrt) and -1 (1/x), used
//     below where the Python code has them; np.float64 ** scalar and array ** array always call
//     pow(), even for those exponents (e.g. compute_Pbar's scalar delta ** (-1)).
//   * np.sum(1-D) is pairwise; np.sum(axis=0) of a (K, n) stack accumulates the rows in order
//     starting from 0.0; np.sum(axis=1) is pairwise per row (verified for the shapes used here).
import * as py from '../py.js';
import * as np from '../np.js';
import * as osu_parser from './osu_file_parser.js';

const { crtExp, crtCos, crtPow } = np;
const ss = np.searchsorted1;
const npMax = np.scalarMax; // np.maximum on scalars (NaN-propagating)
const npMin = np.scalarMin;
/** Python max(a, b) / min(a, b) on two numbers: the first argument wins ties and NaN comparisons. */
const pmax = (a, b) => (b > a ? b : a);
const pmin = (a, b) => (b < a ? b : a);

// Strain-graph post-processing (display only; does not affect SR).
export const BREAK_ZERO_THRESHOLD_MS = 400;
export const GRAPH_RESAMPLE_INTERVAL_MS = 100;
export const SMOOTH_SIGMA_MS = 800;

// ── Python / numpy value helpers ───────────────────────────────────────

/** np.asarray(a, dtype=float) for a 1-D sequence. */
function asF64(a) {
  return a instanceof Float64Array ? a : Float64Array.from(a, Number);
}

/** list[i] with Python's IndexError (non-negative i). */
function item(list, i) {
  if (i < 0 || i >= list.length) throw new py.IndexError('list index out of range');
  return list[i];
}

/** ndarray[i] for an integer index with numpy's negative wrap-around and IndexError. */
function aidx(arr, i) {
  const n = arr.length;
  const k = i < 0 ? i + n : i;
  if (k < 0 || k >= n) throw new py.IndexError(`index ${i} is out of bounds for axis 0 with size ${n}`);
  return arr[k];
}

/** np.clip(i, lo, hi) for integers. */
const iclip = (i, lo, hi) => Math.min(Math.max(i, lo), hi);

/** DOUBLE_IS_ODD_INTEGER */
function isOddInt(w) {
  const a = Math.abs(w);
  return a < 9007199254740992 && a % 2 === 1;
}

/** CPython float ** float (float_pow): its own special cases, then the CRT's pow(). */
function fpow(v, w) {
  if (w === 0) return 1.0;
  if (v !== v) return v;
  if (w !== w) return v === 1 ? 1.0 : w;
  if (w === Infinity || w === -Infinity) {
    const av = Math.abs(v);
    if (av === 1) return 1.0;
    return (w > 0) === (av > 1) ? Infinity : 0.0;
  }
  if (v === Infinity || v === -Infinity) {
    const odd = isOddInt(w);
    if (w > 0) return odd ? v : Math.abs(v);
    return odd ? (v < 0 ? -0.0 : 0.0) : 0.0;
  }
  if (v === 0) {
    if (w < 0) throw new py.ZeroDivisionError('zero to a negative power');
    return isOddInt(w) ? v : 0.0;
  }
  let neg = false;
  if (v < 0) {
    // Python returns a complex number here; the engine never relies on it
    if (w !== Math.floor(w)) throw new py.ValueError('negative number cannot be raised to a fractional power (complex result)');
    v = -v;
    neg = isOddInt(w);
  }
  if (v === 1) return neg ? -1.0 : 1.0;
  const r = crtPow(v, w);
  if (r === Infinity) throw new py.OverflowError("(34, 'Result too large')");
  return neg ? -r : r;
}

// ── Numeric helpers ────────────────────────────────────────────────

export function gaussian_filter1d(data, sigma, mode = 'constant', cval = 0.0) {
  const kernel_radius = py.int(4 * sigma + 0.5);
  const size = Math.max(0, 2 * kernel_radius + 1);
  const kernel = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    const v = (i - kernel_radius) / sigma; // np.arange(-r, r + 1) / sigma
    kernel[i] = crtExp(-0.5 * (v * v)); // (x / sigma) ** 2 → square fast path
  }
  const s = np.sum(kernel);
  for (let i = 0; i < size; i++) kernel[i] /= s;
  const padded = mode === 'constant'
    ? np.pad(asF64(data), kernel_radius, 'constant', cval)
    : np.pad(asF64(data), kernel_radius, mode);
  return np.convolveValid(padded, kernel);
}

/** Vectorised exact cumulative integral of piecewise-constant f on sorted x. */
export function cumulative_sum(x, f) {
  const n = x.length;
  const F = new Float64Array(n);
  let acc = 0;
  for (let i = 1; i < n; i++) {
    const v = f[i - 1] * (x[i] - x[i - 1]); // f[:-1] * np.diff(x)
    acc = i === 1 ? v : acc + v; // np.cumsum
    F[i] = acc;
  }
  return F;
}

/** Vectorised sliding-window integral of piecewise-constant f. */
export function smooth_on_corners(x, f, window, scale = 1.0, mode = 'sum') {
  x = asF64(x);
  f = asF64(f);
  const F = cumulative_sum(x, f);
  const n = x.length;
  const lo = aidx(x, 0);
  const hi = aidx(x, -1);
  const last = n - 2;

  const query = (q) => {
    let idx = ss(x, q, 'left') - 1;
    idx = iclip(idx, 0, last);
    if (idx < 0) idx += n; // numpy negative index (only when len(x) == 1)
    return F[idx] + f[idx] * (q - x[idx]);
  };

  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let a = x[i] - window;
    a = a < lo ? lo : a > hi ? hi : a;
    let b = x[i] + window;
    b = b < lo ? lo : b > hi ? hi : b;
    const val = query(b) - query(a);
    if (mode === 'avg') {
      const span = b - a;
      out[i] = span > 0 ? val / span : 0.0;
    } else {
      out[i] = scale * val;
    }
  }
  return out;
}

export function interp_values(new_x, old_x, old_vals) {
  return np.interp(asF64(new_x), asF64(old_x), asF64(old_vals));
}

export function step_interp(new_x, old_x, old_vals) {
  const out = new Float64Array(new_x.length);
  const last = old_vals.length - 1;
  for (let i = 0; i < new_x.length; i++) {
    const idx = iclip(ss(old_x, new_x[i], 'right') - 1, 0, last);
    out[i] = aidx(old_vals, idx);
  }
  return out;
}

export function rescale_high(sr) {
  if (sr <= 9) return sr;
  return 9 + (sr - 9) / 1.2;
}

// ── Parsing / preprocessing ────────────────────────────────────────

export function preprocess_file(file_path, mod) {
  const p_obj = new osu_parser.parser(file_path);
  p_obj.process();
  const p = p_obj.get_parsed_data();

  const note_seq = [];
  for (let i = 0; i < p[1].length; i++) {
    const k = p[1][i];
    let h = item(p[2], i);
    if (mod === 'DT') h = py.int(py.math.floor((h * 2) / 3));
    else if (mod === 'HT') h = py.int(py.math.floor((h * 4) / 3));
    note_seq.push([k, h]);
  }

  let x = 0.3 * fpow((64.5 - py.math.ceil(p[5] * 3)) / 500, 0.5);
  x = py.min2(x, 0.6 * (x - 0.09) + 0.09);
  note_seq.sort((a, b) => a[1] - b[1] || a[0] - b[0]); // key=(h, k); stable like list.sort

  const note_dict = new Map();
  for (const tup of note_seq) {
    let lst = note_dict.get(tup[0]);
    if (lst === undefined) note_dict.set(tup[0], (lst = []));
    lst.push(tup);
  }

  const K = p[0];
  const T = py.max(note_seq.map((n) => n[1])) + 1;

  const note_seq_by_column = [];
  for (let k = 0; k < K; k++) note_seq_by_column.push(note_dict.get(k) ?? []);

  return [x, K, T, note_seq, note_seq_by_column];
}

// ── Corners / key usage ────────────────────────────────────────────

function sortedCorners(set, T) {
  const out = [];
  for (const s of set) if (s >= 0 && s <= T) out.push(s);
  out.sort((a, b) => a - b);
  return out;
}

export function get_corners(T, note_seq) {
  const base = new Set();
  for (const [, h] of note_seq) {
    base.add(h);
    base.add(h + 501);
    base.add(h - 499);
    base.add(h + 1);
  }
  base.add(0);
  base.add(T);
  const corners_base = sortedCorners(base, T);

  const cornersA = new Set();
  for (const [, h] of note_seq) {
    cornersA.add(h);
    cornersA.add(h + 1000);
    cornersA.add(h - 1000);
  }
  cornersA.add(0);
  cornersA.add(T);
  const corners_A = sortedCorners(cornersA, T);

  const all = new Set(corners_base);
  for (const s of corners_A) all.add(s);
  const all_corners = [...all].sort((a, b) => a - b);
  return [Float64Array.from(all_corners), Float64Array.from(corners_base), Float64Array.from(corners_A)];
}

export function get_key_usage(K, T, note_seq, base_corners) {
  const n = base_corners.length;
  const key_usage = new Map();
  for (let k = 0; k < K; k++) key_usage.set(k, new Array(n).fill(false));
  const bc = asF64(base_corners);
  for (let k = 0; k < K; k++) {
    const arr_k = key_usage.get(k);
    for (const [col, h] of note_seq) {
      if (col !== k) continue;
      const start = npMax(h - 150.0, 0.0);
      const end = npMin(h + 150.0, T - 1.0);
      const li = ss(bc, start, 'left');
      const ri = ss(bc, end, 'left');
      for (let i = li; i < ri; i++) arr_k[i] = true;
    }
  }
  return key_usage;
}

export function get_key_usage_400(K, T, note_seq, base_corners) {
  const n = base_corners.length;
  const key_usage_400 = new Map();
  for (let k = 0; k < K; k++) key_usage_400.set(k, new Float64Array(n));
  const inv_400_sq = 3.75 / 400 ** 2;
  const bc = asF64(base_corners);

  for (let k = 0; k < K; k++) {
    const col_notes = [];
    for (const [col, h] of note_seq) if (col === k) col_notes.push(pmax(h, 0.0));
    if (col_notes.length === 0) continue;
    const arr_k = key_usage_400.get(k);
    const m = col_notes.length;
    const lis = new Int32Array(m);
    const ris = new Int32Array(m);
    const mids = new Int32Array(m);
    for (let i = 0; i < m; i++) {
      lis[i] = ss(bc, col_notes[i] - 400.0, 'left');
      ris[i] = ss(bc, col_notes[i] + 400.0, 'left');
      mids[i] = ss(bc, col_notes[i], 'left');
    }
    // np.add.at(arr_k, mids, 3.75)
    for (let i = 0; i < m; i++) {
      if (mids[i] >= n) throw new py.IndexError(`index ${mids[i]} is out of bounds for axis 0 with size ${n}`);
      arr_k[mids[i]] += 3.75;
    }
    for (let i = 0; i < m; i++) {
      const start = col_notes[i];
      const li = lis[i];
      const mid = mids[i];
      const ri = ris[i];
      if (mid > li) {
        for (let j = li; j < mid; j++) {
          const diff = bc[j] - start;
          arr_k[j] += 3.75 - inv_400_sq * (diff * diff);
        }
      }
      if (ri > mid + 1) {
        for (let j = mid + 1; j < ri; j++) {
          const diff = bc[j] - start;
          arr_k[j] += 3.75 - inv_400_sq * (diff * diff);
        }
      }
    }
  }
  return key_usage_400;
}

export function compute_anchor(K, key_usage_400, base_corners) {
  if (!(K >= 1)) throw new py.ValueError('need at least one array to stack');
  const cols = [];
  for (let k = 0; k < K; k++) cols.push(py.getitem(key_usage_400, k));
  const n = cols[0].length;
  const out = new Float64Array(n);
  const row = new Float64Array(K);
  const wv = new Float64Array(K - 1);
  const mv = new Float64Array(K - 1);
  for (let i = 0; i < n; i++) {
    // counts row, np.sort ascending then reversed
    for (let k = 0; k < K; k++) {
      const v = cols[k][i];
      let p = k;
      while (p > 0 && row[p - 1] > v) {
        row[p] = row[p - 1];
        p--;
      }
      row[p] = v;
    }
    let n_nz = 0;
    for (let k = 0; k < K; k++) if (row[k] > 0) n_nz++;
    for (let t = 0; t < K - 1; t++) {
      const c0 = row[K - 1 - t]; // counts[:, :-1] of the descending row
      const c1 = row[K - 2 - t]; // counts[:, 1:]
      const ratio = c0 > 0 ? c1 / (c0 > 0 ? c0 : 1.0) : 0.0;
      const q = 0.5 - ratio;
      const weight = 1 - 4 * (q * q); // (0.5 - ratio) ** 2 → square fast path
      const pair_valid = c0 > 0 && c1 > 0;
      wv[t] = pair_valid ? c0 * weight : 0.0;
      mv[t] = pair_valid ? c0 : 0.0;
    }
    const walk = 0.0 + np.pairwiseSum(wv, 0, K - 1, 1);
    const max_walk = 0.0 + np.pairwiseSum(mv, 0, K - 1, 1);
    const raw_anchor = n_nz > 1 ? walk / npMax(max_walk, 1e-9) : 0.0;
    out[i] = 1 + npMin(raw_anchor - 0.18, 5 * crtPow(raw_anchor - 0.22, 3));
  }
  return out;
}

// ── Difficulty components ──────────────────────────────────────────

export function compute_Jbar(K, T, x, note_seq_by_column, base_corners) {
  const n = base_corners.length;
  const J_ks = new Map();
  const delta_ks = new Map();
  for (let k = 0; k < K; k++) {
    J_ks.set(k, new Float64Array(n));
    delta_ks.set(k, np.full(n, 1e9));
  }
  const bc = asF64(base_corners);
  const xq = 0.11 * fpow(x, 0.25);

  for (let k = 0; k < K; k++) {
    const notes = item(note_seq_by_column, k);
    if (notes.length < 2) continue;
    const Jk = J_ks.get(k);
    const Dk = delta_ks.get(k);
    for (let i = 0; i < notes.length - 1; i++) {
      const start = notes[i][1];
      const end = notes[i + 1][1];
      const delta = 0.001 * (end - start);
      const jack = 1 - 7e-5 * crtPow(0.15 + Math.abs(delta - 0.08), -4); // jack_nerfer
      const val = ((1.0 / delta) * (1.0 / (delta + xq))) * jack; // deltas ** -1 → reciprocal fast path
      const li = ss(bc, start, 'left');
      const ri = ss(bc, end, 'left');
      if (ri > li) {
        Jk.fill(val, li, ri);
        Dk.fill(delta, li, ri);
      }
    }
  }

  // num = sum_k max(Jbar_k, 0) ** 5 / delta_k, den = sum_k 1 / delta_k (axis-0 sums, in k order)
  const num = new Float64Array(n);
  const den = new Float64Array(n);
  for (let k = 0; k < K; k++) {
    const Jbar_k = smooth_on_corners(base_corners, J_ks.get(k), 500, 0.001, 'sum');
    const Dk = delta_ks.get(k);
    for (let j = 0; j < n; j++) {
      const w = 1.0 / Dk[j];
      num[j] += crtPow(npMax(Jbar_k[j], 0), 5) * w;
      den[j] += w;
    }
  }
  if (!(K >= 1)) throw new py.ValueError('need at least one array to stack');
  const Jbar = new Float64Array(n);
  for (let j = 0; j < n; j++) Jbar[j] = crtPow(num[j] / npMax(den[j], 1e-9), 1 / 5);

  return [delta_ks, Jbar];
}

const CROSS_MATRIX = [
  [-1],
  [0.075, 0.075],
  [0.125, 0.05, 0.125],
  [0.125, 0.125, 0.125, 0.125],
  [0.175, 0.25, 0.05, 0.25, 0.175],
  [0.175, 0.25, 0.175, 0.175, 0.25, 0.175],
  [0.225, 0.35, 0.25, 0.05, 0.25, 0.35, 0.225],
  [0.225, 0.35, 0.25, 0.225, 0.225, 0.25, 0.35, 0.225],
  [0.275, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.275],
  [0.275, 0.45, 0.35, 0.25, 0.275, 0.275, 0.25, 0.35, 0.45, 0.275],
  [0.325, 0.55, 0.45, 0.35, 0.25, 0.05, 0.25, 0.35, 0.45, 0.55, 0.325],
];

export function compute_Xbar(K, T, x, note_seq_by_column, active_columns, base_corners) {
  const cross_coeff = py.at(CROSS_MATRIX, K);
  const n = base_corners.length;
  const X_ks = [];
  const fast_cross = [];
  for (let k = 0; k < K + 1; k++) {
    X_ks.push(new Float64Array(n));
    fast_cross.push(new Float64Array(n));
  }
  const bc = asF64(base_corners);
  const fc_floor = py.max2(0.06, 0.75 * x);

  for (let k = 0; k < K + 1; k++) {
    let notes_in_pair;
    if (k === 0) notes_in_pair = item(note_seq_by_column, 0);
    else if (k === K) notes_in_pair = item(note_seq_by_column, K - 1);
    else {
      notes_in_pair = item(note_seq_by_column, k - 1).concat(item(note_seq_by_column, k));
      notes_in_pair.sort((a, b) => a[1] - b[1]); // sorted(key=t[1]), stable
    }
    if (notes_in_pair.length < 2) continue;

    const Xk = X_ks[k];
    const FCk = fast_cross[k];
    const coeffK = item(cross_coeff, k);
    for (let i = 0; i < notes_in_pair.length - 1; i++) {
      const start = notes_in_pair[i][1];
      const end = notes_in_pair[i + 1][1];
      const li = ss(bc, start, 'left');
      const ri = ss(bc, end, 'left');
      if (ri <= li) continue;
      const delta = 0.001 * (end - start);
      let val = 0.16 * crtPow(npMax(x, delta), -2);
      const fc_val = npMax(0.0, 0.4 * crtPow(npMax(delta, fc_floor), -2) - 80.0);

      const acL = item(active_columns, li);
      const left_inactive = !acL.includes(k - 1) && !item(active_columns, ri).includes(k - 1);
      const right_inactive = !acL.includes(k) && !item(active_columns, ri).includes(k);
      if (left_inactive || right_inactive) val *= 1 - coeffK;

      Xk.fill(val, li, ri);
      FCk.fill(fc_val, li, ri);
    }
  }

  // X_base = np.sum(X_ks_arr * coeff_arr, axis=0): rows accumulated in k order from 0.0
  const X_base = new Float64Array(n);
  for (let k = 0; k < K + 1; k++) {
    const c = item(cross_coeff, k);
    const Xk = X_ks[k];
    for (let j = 0; j < n; j++) X_base[j] += Xk[j] * c;
  }
  for (let k = 0; k < K; k++) {
    const a = fast_cross[k];
    const b = fast_cross[k + 1];
    const ck = cross_coeff[k];
    const ck1 = cross_coeff[k + 1];
    for (let j = 0; j < n; j++) X_base[j] += Math.sqrt(((a[j] * ck) * b[j]) * ck1);
  }

  return smooth_on_corners(base_corners, X_base, 500, 0.001, 'sum');
}

export function compute_Pbar(K, T, x, note_seq, anchor, base_corners) {
  const stream_booster = (delta) => {
    const bpm = npMin(npMax(7.5 / delta, 0), 420);
    const primary = 0.1 / (1 + crtExp(-0.06 * (bpm - 175)));
    const secondary = bpm >= 200 && bpm <= 350 ? 0.3 * (1 - crtExp(-0.02 * (bpm - 200))) : 0.0;
    return 1 + primary + secondary;
  };

  const n = base_corners.length;
  const P_step = new Float64Array(n);
  const bc = asF64(base_corners);
  if (note_seq.length > 1) {
    const base_inc = fpow(0.08 * fpow(x, -1) * (1 - 24 * fpow(x, -1) * fpow(x / 6, 2)), 0.25);
    const spike_val = 1000 * fpow(0.02 * (py.div(4, x) - 24), 0.25);
    const two_x_3 = (2 * x) / 3;
    const x_2 = x / 2;
    const c008 = 0.08 * fpow(x, -1);
    const c24 = 24 * fpow(x, -1);

    for (let i = 0; i < note_seq.length - 1; i++) {
      const h_l = note_seq[i][1];
      const h_r = note_seq[i + 1][1];
      const dt = h_r - h_l;
      const li = ss(bc, h_l, 'left');
      if (dt < 1e-9) {
        const ri_spike = ss(bc, h_l, 'right');
        for (let j = li; j < ri_spike; j++) P_step[j] += spike_val;
        continue;
      }

      const ri = ss(bc, h_r, 'left');
      if (ri <= li) continue;

      const delta = 0.001 * dt;
      const b_val = stream_booster(delta);
      const bmax = pmax(b_val, 1);
      let inc;
      if (delta < two_x_3) {
        // np.float64 ** k → pow() (no fast path for numpy scalars)
        inc = crtPow(delta, -1) * crtPow(c008 * (1 - c24 * crtPow(delta - x_2, 2)), 0.25) * bmax;
      } else {
        inc = crtPow(delta, -1) * base_inc * bmax;
      }

      const cap = npMax(inc, inc * 2 - 10);
      for (let j = li; j < ri; j++) P_step[j] += npMin(inc * anchor[j], cap);
    }
  }

  return smooth_on_corners(base_corners, P_step, 500, 0.001, 'sum');
}

export function compute_Abar(K, T, x, note_seq_by_column, active_columns, delta_ks, A_corners, base_corners) {
  const n = base_corners.length;
  const dks = new Map();
  for (let k = 0; k < K - 1; k++) dks.set(k, new Float64Array(n));
  for (let i = 0; i < n; i++) {
    const cols = item(active_columns, i);
    for (let j = 0; j < cols.length - 1; j++) {
      const k0 = cols[j];
      const k1 = cols[j + 1];
      const d0 = py.getitem(delta_ks, k0)[i];
      const d1 = py.getitem(delta_ks, k1)[i];
      py.getitem(dks, k0)[i] = Math.abs(d0 - d1) + 0.4 * pmax(0, pmax(d0, d1) - 0.11);
    }
  }

  const nA = A_corners.length;
  const A_step = np.ones(nA);
  for (let i = 0; i < nA; i++) {
    let idx = iclip(ss(base_corners, A_corners[i], 'left'), 0, n - 1);
    if (idx < 0) idx += n;
    const cols = item(active_columns, idx);
    for (let j = 0; j < cols.length - 1; j++) {
      const k0 = cols[j];
      const k1 = cols[j + 1];
      const d_val = py.getitem(dks, k0)[idx];
      const dk0 = py.getitem(delta_ks, k0)[idx];
      const dk1 = py.getitem(delta_ks, k1)[idx];
      if (d_val < 0.02) A_step[i] *= pmin(0.75 + 0.5 * pmax(dk0, dk1), 1);
      else if (d_val < 0.07) A_step[i] *= pmin(0.65 + 5 * d_val + 0.5 * pmax(dk0, dk1), 1);
    }
  }

  return smooth_on_corners(A_corners, A_step, 250, 1.0, 'avg');
}

export function compute_C_and_Ks(K, T, note_seq, key_usage, base_corners) {
  const note_hit_times = Float64Array.from(note_seq, (t) => t[1]).sort();
  const n = base_corners.length;
  const C_step = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const lo = ss(note_hit_times, base_corners[i] - 500, 'left');
    const hi = ss(note_hit_times, base_corners[i] + 500, 'left');
    C_step[i] = hi - lo;
  }

  if (!(K >= 1)) throw new py.ValueError('need at least one array to stack');
  const rows = [];
  for (let k = 0; k < K; k++) rows.push(py.getitem(key_usage, k));
  const m = rows[0].length;
  const Ks_step = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    let c = 0;
    for (let k = 0; k < K; k++) if (rows[k][i]) c++;
    Ks_step[i] = Math.max(c, 1);
  }
  return [C_step, Ks_step];
}

// ── Strain graph ───────────────────────────────────────────────────

export function _apply_proximity_envelope(all_corners, D_all, note_seq) {
  if (note_seq.length === 0) return Float64Array.from(D_all);

  const note_times = Float64Array.from(note_seq, (t) => t[1]).sort();
  const PROXIMITY_FADE_MS = 500.0;
  const nt = note_times.length;
  const n = all_corners.length;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const c = all_corners[i];
    const idx = ss(note_times, c, 'left');
    const d_after = Math.abs(note_times[iclip(idx, 0, nt - 1)] - c);
    const d_before = Math.abs(note_times[iclip(idx - 1, 0, nt - 1)] - c);
    const d = npMin(d_after, d_before);
    const envelope = 0.5 * (1.0 + crtCos(Math.PI * np.clip(d / PROXIMITY_FADE_MS, 0.0, 1.0)));
    out[i] = D_all[i] * envelope;
  }
  return out;
}

export function smooth_D_for_graph(all_corners, D_all, note_seq) {
  const note_times = Float64Array.from(note_seq, (t) => t[1]).sort();

  const t_start = aidx(all_corners, 0);
  const t_end = aidx(all_corners, -1);
  const uniform_t = np.arange(t_start, t_end + GRAPH_RESAMPLE_INTERVAL_MS, GRAPH_RESAMPLE_INTERVAL_MS);
  const nu = uniform_t.length;

  const break_mask = new Uint8Array(nu);
  const nt = note_times.length;
  if (nt > 0) {
    for (let i = 0; i < nu; i++) {
      const u = uniform_t[i];
      const idx = ss(note_times, u, 'left');
      const dist = npMin(Math.abs(u - note_times[iclip(idx - 1, 0, nt - 1)]), Math.abs(u - note_times[iclip(idx, 0, nt - 1)]));
      break_mask[i] = dist > BREAK_ZERO_THRESHOLD_MS ? 1 : 0;
    }
  }

  const uniform_D = np.interp(uniform_t, asF64(all_corners), asF64(D_all));
  for (let i = 0; i < nu; i++) if (break_mask[i]) uniform_D[i] = 0.0;

  const sigma_samples = SMOOTH_SIGMA_MS / GRAPH_RESAMPLE_INTERVAL_MS;
  const uniform_result = gaussian_filter1d(uniform_D, sigma_samples, 'constant', 0.0);
  for (let i = 0; i < nu; i++) if (break_mask[i]) uniform_result[i] = 0.0;

  return np.interp(asF64(all_corners), uniform_t, uniform_result);
}

// ── Main entry point ───────────────────────────────────────────────

export function calculate(file_path, mod, { noteCountReference = null } = {}) {
  const [x, K, T, note_seq, note_seq_by_column] = preprocess_file(file_path, mod);
  const [all_corners, base_corners, A_corners] = get_corners(T, note_seq);

  const key_usage = get_key_usage(K, T, note_seq, base_corners);
  const nb = base_corners.length;
  const active_columns = new Array(nb);
  for (let i = 0; i < nb; i++) {
    const cols = [];
    for (let k = 0; k < K; k++) if (key_usage.get(k)[i]) cols.push(k);
    active_columns[i] = cols;
  }
  const key_usage_400 = get_key_usage_400(K, T, note_seq, base_corners);
  const anchor = compute_anchor(K, key_usage_400, base_corners);

  const [delta_ks, Jbar0] = compute_Jbar(K, T, x, note_seq_by_column, base_corners);
  const Jbar = interp_values(all_corners, base_corners, Jbar0);

  const Xbar = interp_values(all_corners, base_corners, compute_Xbar(K, T, x, note_seq_by_column, active_columns, base_corners));

  const Pbar = interp_values(all_corners, base_corners, compute_Pbar(K, T, x, note_seq, anchor, base_corners));

  const Abar = interp_values(all_corners, A_corners,
    compute_Abar(K, T, x, note_seq_by_column, active_columns, delta_ks, A_corners, base_corners));

  const [C_step, Ks_step] = compute_C_and_Ks(K, T, note_seq, key_usage, base_corners);
  const C_arr = step_interp(all_corners, base_corners, C_step);
  const Ks_arr = step_interp(all_corners, base_corners, Ks_step);

  const n = all_corners.length;
  const D_all = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const A = Abar[i];
    const J = Jbar[i];
    const aks = crtPow(A, 3 / Ks_arr[i]); // Abar ** (3 / Ks_arr): array ** array → pow()
    const S = crtPow(
      0.4 * crtPow(aks * npMin(J, 8 + 0.85 * J), 1.5)
      + 0.6 * crtPow(crtPow(A, 2 / 3) * (0.8 * Pbar[i]), 1.5),
      2 / 3,
    );
    const X = Xbar[i];
    const Tv = (aks * X) / (X + S + 1);
    D_all[i] = 2.7 * Math.sqrt(S) * crtPow(Tv, 1.5) + S * 0.27; // S_all ** 0.5 → sqrt fast path
  }

  const gaps = new Float64Array(n);
  gaps[0] = (aidx(all_corners, 1) - all_corners[0]) / 2.0;
  gaps[n - 1] = (all_corners[n - 1] - aidx(all_corners, -2)) / 2.0;
  for (let i = 1; i < n - 1; i++) gaps[i] = (all_corners[i + 1] - all_corners[i - 1]) / 2.0;

  const sorted_indices = np.argsort(D_all);
  const D_sorted = new Float64Array(n);
  const w_sorted = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const k = sorted_indices[i];
    D_sorted[i] = D_all[k];
    w_sorted[i] = C_arr[k] * gaps[k]; // effective_weights[sorted_indices]
  }

  const cum_weights = np.cumsum(w_sorted);
  const total = cum_weights[n - 1];
  const norm_cum_weights = new Float64Array(n);
  for (let i = 0; i < n; i++) norm_cum_weights[i] = cum_weights[i] / total;

  const target_percentiles = [0.945, 0.935, 0.925, 0.915, 0.845, 0.835, 0.825, 0.815];
  const indices = target_percentiles.map((t) => ss(norm_cum_weights, t, 'left'));

  const percentile_93 = np.mean(Float64Array.from(indices.slice(0, 4), (i) => aidx(D_sorted, i)));
  const percentile_83 = np.mean(Float64Array.from(indices.slice(4, 8), (i) => aidx(D_sorted, i)));
  const d5w = new Float64Array(n);
  for (let i = 0; i < n; i++) d5w[i] = crtPow(D_sorted[i], 5) * w_sorted[i];
  const weighted_mean = crtPow(np.sum(d5w) / np.sum(w_sorted), 1 / 5);

  let SR = 0.88 * percentile_93 * 0.25 + 0.94 * percentile_83 * 0.2 + weighted_mean * 0.55;
  // Rolling excerpts share their parent map's length normalization. Their local
  // pattern difficulty must not acquire a second short-map penalty just from cropping.
  const total_notes = Number(noteCountReference) > 0 ? Number(noteCountReference) : note_seq.length;
  SR *= total_notes / (total_notes + 60);
  SR = rescale_high(SR) * 0.975;

  const D_pre = _apply_proximity_envelope(all_corners, D_all, note_seq);
  const D_graph = smooth_D_for_graph(all_corners, D_pre, note_seq);

  return [SR, all_corners, D_graph, {
    'Pressing Intensity': Pbar,
    Unevenness: Abar,
    'Same-Column Pressure': Jbar,
    'Cross-Column Pressure': Xbar,
  }];
}

// ── Extras ─────────────────────────────────────────────────────────

export function factor_averages(times, factors) {
  times = asF64(times);
  const names = factors instanceof Map ? [...factors.keys()] : Object.keys(factors);
  if (names.length === 0) throw new py.ValueError('need at least one array to stack');
  const rows = names.map((nm) => asF64(factors instanceof Map ? factors.get(nm) : factors[nm]));
  const integrals = np.trapezoidRows(rows, times);
  const duration = aidx(times, -1) - aidx(times, 0);
  const out = {};
  names.forEach((nm, i) => {
    out[nm] = integrals[i] / duration;
  });
  return out;
}

export function parse_hitobjects(file_path, mod = 'NM') {
  const p_obj = new osu_parser.parser(file_path);
  p_obj.process();
  const p = p_obj.get_parsed_data();

  const hitobjects = [];
  for (let i = 0; i < p[1].length; i++) {
    const x = p[1][i];
    let time = item(p[2], i);
    if (mod === 'DT') time *= 2 / 3;
    else if (mod === 'HT') time *= 4 / 3;
    hitobjects.push({ x, time });
  }
  return hitobjects;
}
