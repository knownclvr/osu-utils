// strain.js — biomechanical strain model (decaying strain streams → bio numeric).
//
// JavaScript port of engine/dan_engine/strain.py (DanOverlay 3.1.0 by 8doul).
//
// Row model + strain streams with dual exponential decay (burst/sustain), Shannon
// entropy over column distribution and mask transitions, weighted quantile
// aggregation and section-based aggregation. The output is a "bio numeric" on the
// numeric difficulty scale (~0-20).
//
// Porting notes
//   * row['nps'] is a dict keyed by float window sizes → a Map keyed by number.
//   * compute_bio_curve iterates a Python set of transition codes while summing the
//     transition entropy; the summation order is the set's iteration order, so the
//     set is emulated slot for slot (_PyIntSet, CPython 3.14 layout).
import * as py from './py.js';
import * as np from './np.js';

export const BIO_CONFIG = {
  row_tolerance_ms: 2.0,
  entropy_window_ms: 750.0,
  nps_windows_ms: [250.0, 500.0, 1000.0, 4000.0],
  section_ms: 400.0,
  section_decay: 0.9,
  section_ema_alpha: 0.15,
  correction_clamp: 1.25,
  raw_map: { p02: 4.0, p98: 7.5, y0: -2.0, y1: 20.0 },

  bio_sr_map: { p02: 2.0, p98: 9.0, y0: 1.0, y1: 14.0 },

  streams: {
    speed: { burst_tau: 200.0, sustain_tau: 2500.0, burst_mix: 0.75 },
    hand: { burst_tau: 240.0, sustain_tau: 3000.0, burst_mix: 0.78 },
    jack: { burst_tau: 280.0, sustain_tau: 2500.0, burst_mix: 0.85 },
    chordjack: { burst_tau: 260.0, sustain_tau: 3000.0, burst_mix: 0.8 },
    tech: { burst_tau: 450.0, sustain_tau: 3500.0, burst_mix: 0.7 },
    stamina: { burst_tau: 1200.0, sustain_tau: 10000.0, burst_mix: 0.58 },
    course: { burst_tau: 30000.0, sustain_tau: 120000.0, burst_mix: 0.35 },
  },
  stream_weights: {
    speed: 0.22, hand: 0.18, jack: 0.16, chordjack: 0.16,
    tech: 0.12, stamina: 0.11, course: 0.05,
  },

  r: { base: 180.0, offset: 40.0, power: 1.08, cap: 8.0 },
  r_same_col: { base: 185.0, offset: 35.0, power: 1.18, cap: 8.0 },
  r_fast_hand: { base: 205.0, offset: 45.0, power: 1.05, cap: 8.0 },

  speed_in: { w_row: 0.55, w_hand_max: 0.3, w_hand_mean: 0.15 },
  jack_in: { w_chord: 0.2, w_anchor: 0.15, anchor_ms: 220.0 },
  hand_in: { w_base: 0.7, w_rotation: 0.3, r_hand: { base: 180.0, offset: 38.0, power: 1.1, cap: 8.0 } },
  chord_in: { w_speed: 0.18, w_same_hand: 0.22, body_r: { base: 150.0, offset: 80.0, power: 0.85, cap: 8.0 } },
  chordjack_in: { w_jack: 0.55, w_overlap: 0.3, w_hand: 0.15 },
  tech_in: { w_chaos: 0.32, w_entropy: 0.24, w_transition: 0.24, w_change: 0.2 },
  stamina_in: { w_nps1k: 0.4, w_nps4k: 0.35, w_hand: 0.25, hand_tau: 8000.0, log_base: 24.0 },
  course_in: { dur_gate: [90.0, 300.0], break_damp: [0.006, 0.018], w_break: 0.25 },

  agg: { q97: 0.3, q90: 0.22, tail: 0.18, q75: 0.15, power: 0.1, q50: 0.05 },
  power_mean_p: 2.4,
  tail_ratio: 0.04,

  corrections: {
    low_cj: { gain: 0.0, chord: [0.48, 0.68], overlap: [0.75, 1.25], nps: [19.0, 23.0], imbalance: [0.06, 0.12] },
    high_stream: { gain: 0.0, rotation: [0.68, 0.86], q10: [100.0, 130.0], chord: [0.25, 0.42], overlap: [0.65, 0.95] },
    course_break: { gain: 0.0, dur: [240.0, 480.0], break: [0.006, 0.018], peak_gap: [0.35, 0.75], nps: [12.0, 18.0] },
    course_sustain: { gain: 0.0, dur: [240.0, 600.0], break: [0.004, 0.012], peak_gap: [0.15, 0.45], nps: [15.0, 21.0] },
    anchor_lift: { gain: 0.0, anchor: [0.18, 0.38], fast_jack: [0.25, 0.55], chord: [0.65, 0.85] },
    hand_bias: { gain: 0.0, bias: [0.25, 0.55], nps: [12.0, 20.0] },
    handstream_damp: { gain: 0.0, rotation: [0.6, 0.8], overlap_inv: [0.6, 0.9], chord: [0.3, 0.5] },
    cj_over_damp: { gain: 0.0, chord: [0.55, 0.75], fast_jack_inv: [0.5, 0.7], overlap: [0.4, 0.7] },
  },
};

// ── local helpers (Python semantics) ──────────────────────────────────
/** Python max(a, b): b replaces a only when b > a (first wins ties / NaN comparisons). */
const pymax = (a, b) => (b > a ? b : a);
/** Python min(a, b): b replaces a only when b < a. */
const pymin = (a, b) => (b < a ? b : a);
/** Python `x or d`. */
const orElse = (x, d) => (py.truthy(x) ? x : d);
/** list(d.keys()) for a dict given as a plain object or a Map. */
const dictKeys = (d) => (d instanceof Map ? [...d.keys()] : Object.keys(d));
/** d[k] for a plain object or a Map (KeyError when missing). */
const item = (d, k) => py.getitem(d, k);
const EMPTY_DICT = new Map();

/**
 * A Python set of small non-negative ints (hash(i) == i), laid out exactly like
 * CPython 3.14's setobject.c (8-slot small table, 9 linear probes then perturbed
 * probing, the last dummy on the probe path reused by add, resize at fill*5 >= mask*3
 * to the smallest power of two > used*4) so that iteration order — slot order — is
 * Python's. Verified against CPython 3.14 on random add/discard sequences
 * (tools/_work/js-c/set_emul_check.py + set_emul_check.mjs). Keys must be < 2**48.
 */
export class _PyIntSet {
  constructor() {
    this.keys = new Float64Array(8).fill(-1); // -1 = never used, -2 = dummy
    this.mask = 7;
    this.fill = 0; // active + dummy slots
    this.used = 0; // active slots
  }

  get size() {
    return this.used;
  }

  add(key) {
    const mask = this.mask;
    const keys = this.keys;
    let i = key % (mask + 1);
    let perturb = key;
    let freeslot = -1;
    for (;;) {
      let probes = i + 9 <= mask ? 9 : 0;
      let j = i;
      for (;;) {
        const k = keys[j];
        if (k === -1) {
          if (freeslot >= 0) {
            keys[freeslot] = key;
            this.used++;
            return;
          }
          keys[j] = key;
          this.fill++;
          this.used++;
          if (this.fill * 5 < mask * 3) return;
          this._resize(this.used > 50000 ? this.used * 2 : this.used * 4);
          return;
        }
        if (k === -2) freeslot = j;
        else if (k === key) return;
        if (probes === 0) break;
        probes--;
        j++;
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) % (mask + 1);
    }
  }

  discard(key) {
    const mask = this.mask;
    const keys = this.keys;
    let i = key % (mask + 1);
    let perturb = key;
    for (;;) {
      let probes = i + 9 <= mask ? 9 : 0;
      let j = i;
      for (;;) {
        const k = keys[j];
        if (k === -1) return false;
        if (k === key) {
          keys[j] = -2;
          this.used--;
          return true;
        }
        if (probes === 0) break;
        probes--;
        j++;
      }
      perturb = Math.floor(perturb / 32);
      i = (i * 5 + 1 + perturb) % (mask + 1);
    }
  }

  has(key) {
    for (const k of this.keys) if (k === key) return true;
    return false;
  }

  _resize(minused) {
    let newsize = 8;
    while (newsize <= minused) newsize *= 2;
    const old = this.keys;
    if (newsize === 8 && old.length === 8 && this.fill === this.used) return; // small table, no dummies
    const keys = new Float64Array(newsize).fill(-1);
    const mask = newsize - 1;
    for (const key of old) {
      if (key < 0) continue;
      // set_insert_clean
      let i = key % newsize;
      let perturb = key;
      for (;;) {
        if (keys[i] === -1) {
          keys[i] = key;
          break;
        }
        let placed = false;
        if (i + 9 <= mask) {
          for (let j = 1; j <= 9; j++) {
            if (keys[i + j] === -1) {
              keys[i + j] = key;
              placed = true;
              break;
            }
          }
        }
        if (placed) break;
        perturb = Math.floor(perturb / 32);
        i = (i * 5 + 1 + perturb) % newsize;
      }
    }
    this.keys = keys;
    this.mask = mask;
    this.fill = this.used;
  }

  * [Symbol.iterator]() {
    const keys = this.keys;
    for (let s = 0; s < keys.length; s++) if (keys[s] >= 0) yield keys[s];
  }
}

// ── module functions ──────────────────────────────────────────────────
export function _clamp(x, lo, hi) {
  return pymax(lo, pymin(hi, x));
}

/** Rising gate: 0 below a, 1 above b (continuous). */
export function _gate(x, a, b) {
  if (b <= a) return x >= b ? 1.0 : 0.0;
  return _clamp((x - a) / (b - a), 0.0, 1.0);
}

/** Falling gate: 1 below a, 0 above b (continuous). */
export function _inverse_gate(x, a, b) {
  return 1.0 - _gate(x, a, b);
}

export function _strain_rate(dt, base, offset, power, cap) {
  const effective = pymax(16.0, dt + offset);
  const value = py.pow(py.div(base, effective), power);
  return Number.isFinite(value) ? pymin(cap, value) : 0.0;
}

export function _decay(state, input_val, dt, tau) {
  const delta = dt > 0 ? dt : 0.0;
  return state * py.math.exp(py.div(-delta, tau)) + input_val;
}

export function _bitcount(mask) {
  let m = mask & 15;
  let c = 0;
  while (m) {
    c += m & 1;
    m >>= 1;
  }
  return c;
}

export function _entropy_from_counts(counts, total, normalizer) {
  if (total <= 0) return 0.0;
  let ent = 0.0;
  for (const c of counts) {
    if (c <= 0) continue;
    const p = c / total;
    ent -= p * py.math.log2(p);
  }
  return _clamp(py.div(ent, normalizer), 0.0, 1.0);
}

/** Rows from parser rows [{'t': ms, 'cols': (c, ...)}] merged at tolerance. */
export function build_bio_rows(parsed_rows, tolerance_ms = 2.0) {
  const merged = [];
  for (const r of parsed_rows) {
    const t = py.float(item(r, 't'));
    const cols = py.sorted([...item(r, 'cols')].map((c) => py.int(c)));
    if (merged.length && Math.abs(t - merged[merged.length - 1].t) <= tolerance_ms) {
      const prev = merged[merged.length - 1];
      prev.cols = py.sorted([...new Set([...prev.cols, ...cols])]);
    } else {
      merged.push({ t, cols });
    }
  }
  const rows = [];
  for (const r of merged) {
    let mask = 0;
    for (const c of r.cols) {
      if (c >= 0 && c <= 3) mask |= 1 << c;
    }
    if (mask === 0) continue;
    const left = mask & 3;
    const right = mask & 12;
    rows.push({
      t: r.t,
      mask,
      row_size: _bitcount(mask),
      left_count: _bitcount(left),
      right_count: _bitcount(right),
      hand_mask: [left, right],
    });
  }
  return rows;
}

export function compute_activity_stats(rows) {
  const n = py.len(rows);
  if (n < 2) {
    return { inactive_ms: 0.0, break_count: 0, active_duration_s: 1.0, break_density: 0.0, avg_nps: n };
  }
  let inactive_ms = 0.0;
  let break_count = 0;
  for (let i = 1; i < n; i++) {
    const gap = rows[i].t - rows[i - 1].t;
    if (gap > 1000) {
      inactive_ms += gap - 1000;
      break_count += 1;
    }
  }
  const duration_ms = pymax(1.0, rows[n - 1].t - rows[0].t - inactive_ms);
  const active_s = duration_ms / 1000.0;
  return {
    inactive_ms,
    break_count,
    active_duration_s: active_s,
    break_density: break_count / pymax(active_s / 60.0, 1.0),
    avg_nps: n / pymax(active_s, 1.0),
  };
}

/** row.setdefault('nps', {}) — the dict is a Map keyed by the (float) window size. */
function npsSetdefault(row) {
  if (!Object.prototype.hasOwnProperty.call(row, 'nps')) row.nps = new Map();
  let d = row.nps;
  if (d === null || d === undefined) throw new py.TypeError_("'NoneType' object does not support item assignment");
  if (!(d instanceof Map)) {
    // a dict decoded as a plain object (only possible when it has no numeric keys)
    d = new Map(Object.entries(d));
    row.nps = d;
  }
  return d;
}

export function compute_nps_rows(rows, windows) {
  const starts = new Array(windows.length).fill(0);
  let end = 0;
  const times = rows.map((r) => r.t);
  const n = times.length;
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    while (end < n && times[end] <= row.t + 1e-9) end += 1;
    for (let w = 0; w < windows.length; w++) {
      const window_ms = windows[w];
      const min_time = row.t - window_ms;
      while (starts[w] < n && times[starts[w]] <= min_time) starts[w] += 1;
      const value = py.div(end - starts[w], window_ms / 1000.0);
      npsSetdefault(row).set(window_ms, value);
    }
  }
}

export function compute_bio_curve(rows, cfg) {
  // cfg lookups happen where Python performs them (a partial cfg fails at the same point)
  const cc = (a, b) => item(item(cfg, a), b); // cfg[a][b]
  const streams_cfg = item(cfg, 'streams');
  const stream_names = dictKeys(item(cfg, 'stream_weights'));
  const states = {};
  for (const name of stream_names) states[name] = { burst: 0.0, sustain: 0.0 };
  const series = {};
  for (const name of stream_names) series[name] = [];

  const r_cfg = item(cfg, 'r');
  const r_same = item(cfg, 'r_same_col');
  const r_fast = item(cfg, 'r_fast_hand');
  const r_hand = cc('hand_in', 'r_hand');
  const body_r = cc('chord_in', 'body_r');
  // _strain_rate(dt, r['base'], r['offset'], r['power'], r['cap'])
  const rate_of = (dt, r) => _strain_rate(dt, item(r, 'base'), item(r, 'offset'), item(r, 'power'), item(r, 'cap'));

  const last_col_time = [NaN, NaN, NaN, NaN];
  const last_hand_time = [NaN, NaN];
  const prev_hand_mask = [0, 0];
  const hand_stamina = [0.0, 0.0];

  const col_counts = [0, 0, 0, 0];
  const dt_same_vals = [];
  const dt_hand_vals = [];
  const local_raw = [];

  const mask_counts = new Array(16).fill(0);
  const transition_counts = new Array(256).fill(0);
  const active_transitions = new _PyIntSet();
  const queue = [];
  let queue_head = 0;
  let mask_total = 0;
  let transition_total = 0;

  let prev_row_time = rows.length ? rows[0].t - 1000 : 0.0;
  let prev_dt_row = 1000.0;
  let prev_mask = 0;

  const dt_row_vals = [];

  let left_load = 0.0;
  let right_load = 0.0;
  let chord_rows = 0;
  let three_rows = 0;
  let overlap_events_total = 0;
  let rotation_total = 0;
  let eligible_hand = 0;
  let anchor_strength_total = 0.0;
  let fast_jack_strength_total = 0.0;
  let dt_hand_count = 0; // eslint-disable-line no-unused-vars

  const activity = compute_activity_stats(rows);
  const windows = [...item(cfg, 'nps_windows_ms')];
  compute_nps_rows(rows, windows);

  for (const row of rows) {
    const t = row.t;
    const dt_row = pymax(1.0, t - prev_row_time);
    dt_row_vals.push(dt_row);
    const mask = row.mask;
    const left = row.hand_mask[0];
    const right = row.hand_mask[1];
    const hand_masks = [left, right];

    const dt_hand = [NaN, NaN];
    const rotation = [0, 0];
    let overlap_row = 0;
    for (let h = 0; h < 2; h++) {
      if (hand_masks[h] === 0) continue;
      if (Number.isFinite(last_hand_time[h])) {
        dt_hand[h] = pymax(1.0, t - last_hand_time[h]);
        dt_hand_vals.push(dt_hand[h]);
        dt_hand_count += 1;
        eligible_hand += 1;
        if (prev_hand_mask[h] !== 0 && (hand_masks[h] & prev_hand_mask[h]) === 0) {
          rotation[h] = 1.0;
          rotation_total += 1;
        }
        if ((hand_masks[h] & prev_hand_mask[h]) !== 0) overlap_row += 1;
      }
    }
    const same_hand_overlap = overlap_row / 2.0;
    overlap_events_total += same_hand_overlap;

    const dt_same = [NaN, NaN, NaN, NaN];
    let jack_max = 0.0;
    let anchor_row = 0.0;
    for (let c = 0; c < 4; c++) {
      if ((mask & (1 << c)) === 0) continue;
      col_counts[c] += 1;
      if (Number.isFinite(last_col_time[c])) {
        dt_same[c] = pymax(1.0, t - last_col_time[c]);
        dt_same_vals.push(dt_same[c]);
        anchor_row = pymax(anchor_row, _inverse_gate(dt_same[c], cc('jack_in', 'anchor_ms'), cc('jack_in', 'anchor_ms') + 40.0));
        fast_jack_strength_total += _inverse_gate(dt_same[c], 120.0, 150.0);
        jack_max = pymax(jack_max, rate_of(dt_same[c], r_same));
      }
    }
    anchor_strength_total += anchor_row;

    left_load += row.left_count;
    right_load += row.right_count;
    if (row.row_size >= 2) chord_rows += 1;
    if (row.row_size >= 3) three_rows += 1;

    mask_counts[mask] += 1;
    mask_total += 1;
    let transition_code = -1;
    if (prev_mask) {
      transition_code = (prev_mask << 4) | mask;
      transition_counts[transition_code] += 1;
      transition_total += 1;
      active_transitions.add(transition_code);
    }
    queue.push([t, mask, transition_code]);
    while (queue_head < queue.length && queue[queue_head][0] < t - item(cfg, 'entropy_window_ms')) {
      const [, old_mask, old_code] = queue[queue_head];
      mask_counts[old_mask] -= 1;
      mask_total -= 1;
      if (old_code >= 0) {
        transition_counts[old_code] -= 1;
        if (transition_counts[old_code] === 0) active_transitions.discard(old_code);
        transition_total -= 1;
      }
      queue_head += 1;
    }

    // Shannon entropy of the row masks / mask transitions in the sliding window.
    let entropy_mask;
    if (mask_total > 0) {
      let ent_m = 0.0;
      for (let m = 0; m < 16; m++) {
        const c = mask_counts[m];
        if (c > 0) {
          const p = c / mask_total;
          ent_m -= p * py.math.log2(p);
        }
      }
      entropy_mask = _clamp(ent_m / 4.0, 0.0, 1.0);
    } else {
      entropy_mask = 0.0;
    }

    let entropy_transition;
    if (transition_total > 0 && active_transitions.size > 0) {
      let ent_t = 0.0;
      for (const code of active_transitions) { // Python set iteration order
        const c = transition_counts[code];
        if (c > 0) {
          const p = c / transition_total;
          ent_t -= p * py.math.log2(p);
        }
      }
      entropy_transition = _clamp(ent_t / 8.0, 0.0, 1.0);
    } else {
      entropy_transition = 0.0;
    }

    const row_chord = (row.row_size - 1) / 3.0;
    const same_hand_chord = (pymax(0, row.left_count - 1) + pymax(0, row.right_count - 1)) / 2.0;

    const hand_rates = [];
    for (let h = 0; h < 2; h++) {
      if (hand_masks[h] === 0) continue;
      const hdt = Number.isFinite(dt_hand[h]) ? dt_hand[h] : 1000.0;
      const rate = rate_of(hdt, r_hand);
      hand_rates.push(rate);
      hand_stamina[h] = _decay(hand_stamina[h], rate, hdt, cc('stamina_in', 'hand_tau'));
    }
    for (let h = 0; h < 2; h++) {
      if (hand_masks[h] === 0) {
        hand_stamina[h] = _decay(hand_stamina[h], 0.0, dt_row, cc('stamina_in', 'hand_tau'));
      }
    }

    const hand_max = hand_rates.length ? py.max(hand_rates) : 0.0;
    const hand_mean = hand_rates.length ? py.sum(hand_rates) / hand_rates.length : 0.0;

    const speed_in = (
      cc('speed_in', 'w_row') * rate_of(dt_row, r_cfg)
      + cc('speed_in', 'w_hand_max') * hand_max
      + cc('speed_in', 'w_hand_mean') * hand_mean
    );

    const jack_in = jack_max * (1.0 + cc('jack_in', 'w_chord') * row_chord + cc('jack_in', 'w_anchor') * anchor_row);
    let hand_in = 0.0;
    for (let h = 0; h < 2; h++) {
      if (hand_masks[h] === 0) continue;
      const hdt = Number.isFinite(dt_hand[h]) ? dt_hand[h] : 1000.0;
      const base_rate = rate_of(hdt, r_hand);
      const fast_rate = rate_of(hdt, r_fast);
      hand_in = pymax(hand_in, cc('hand_in', 'w_base') * base_rate + cc('hand_in', 'w_rotation') * rotation[h] * fast_rate);
    }

    const body = pymax(0, row.row_size - 2) * rate_of(dt_row, body_r);
    const chord_in = row_chord * (1.0 + cc('chord_in', 'w_speed') * speed_in) + cc('chord_in', 'w_same_hand') * same_hand_chord + body;
    const chordjack_in = row_chord * (
      cc('chordjack_in', 'w_jack') * jack_in
      + cc('chordjack_in', 'w_overlap') * same_hand_overlap
      + cc('chordjack_in', 'w_hand') * hand_in
    );

    const rhythm_chaos = prev_mask
      ? pymin(2.0, Math.abs(py.math.log2((dt_row + 24.0) / (prev_dt_row + 24.0)))) / 2.0
      : 0.0;
    const tech_in = (
      cc('tech_in', 'w_chaos') * rhythm_chaos
      + cc('tech_in', 'w_entropy') * entropy_mask
      + cc('tech_in', 'w_transition') * entropy_transition
      + cc('tech_in', 'w_change') * (mask !== prev_mask ? 1.0 : 0.0)
    );

    const nps = item(row, 'nps');
    const nps1k = orElse(py.get(nps, 1000.0, 0.0), 0.0);
    const nps4k = orElse(py.get(nps, 4000.0, 0.0), 0.0);
    const log_base = cc('stamina_in', 'log_base');
    const stamina_in = (
      py.div(cc('stamina_in', 'w_nps1k') * py.math.log1p(nps1k), py.math.log(log_base))
      + py.div(cc('stamina_in', 'w_nps4k') * py.math.log1p(nps4k), py.math.log(log_base))
      + cc('stamina_in', 'w_hand') * pymax(hand_stamina[0], hand_stamina[1])
    );

    const [d1, d2] = cc('course_in', 'dur_gate');
    const [b1, b2] = cc('course_in', 'break_damp');
    const course_in = stamina_in * _gate(activity.active_duration_s, d1, d2)
      * (1.0 - cc('course_in', 'w_break') * _gate(activity.break_density, b1, b2));

    const inputs = {
      speed: speed_in, hand: hand_in, jack: jack_in, chordjack: chordjack_in,
      tech: tech_in, stamina: stamina_in, course: course_in,
    };

    for (const name of stream_names) {
      const sc = item(streams_cfg, name);
      const st = states[name];
      st.burst = _decay(st.burst, item(inputs, name), dt_row, item(sc, 'burst_tau'));
      st.sustain = _decay(st.sustain, item(inputs, name), dt_row, item(sc, 'sustain_tau'));
      const mix = item(sc, 'burst_mix');
      series[name].push(mix * st.burst + (1.0 - mix) * st.sustain);
    }

    const raw = py.sum(stream_names.map((name) => cc('stream_weights', name) * series[name][series[name].length - 1]));
    local_raw.push(raw);

    for (let c = 0; c < 4; c++) {
      if ((mask & (1 << c)) !== 0) last_col_time[c] = t;
    }
    for (let h = 0; h < 2; h++) {
      if (hand_masks[h] !== 0) {
        last_hand_time[h] = t;
        prev_hand_mask[h] = hand_masks[h];
      }
    }

    prev_row_time = t;
    prev_dt_row = dt_row;
    prev_mask = mask;
  }

  let peak_to_sustain_gap;
  if (local_raw.length) {
    const sorted_raw = py.sorted(local_raw);
    const q97_local = _quantile(sorted_raw, 0.97);
    const q75_local = _quantile(sorted_raw, 0.75);
    peak_to_sustain_gap = _clamp((q97_local - q75_local) / pymax(q97_local, 1e-6), 0.0, 1.0);
  } else {
    peak_to_sustain_gap = 0.0;
  }

  const nps1k_sorted = py.sorted(rows.map((row) => orElse(py.get(py.get(row, 'nps', EMPTY_DICT), 1000.0, 0.0), 0.0)));
  const nps4k_sorted = py.sorted(rows.map((row) => orElse(py.get(py.get(row, 'nps', EMPTY_DICT), 4000.0, 0.0), 0.0)));
  const q50_1k = _quantile(nps1k_sorted, 0.5);
  const q75_1k = _quantile(nps1k_sorted, 0.75);
  const q90_1k = _quantile(nps1k_sorted, 0.9);
  const q97_1k = _quantile(nps1k_sorted, 0.97);
  const log_nps_vals = nps1k_sorted.map((v) => py.div(py.math.log1p(pymax(0.0, v)), py.math.log(cc('stamina_in', 'log_base'))));
  const dt_row_sorted = py.sorted(dt_row_vals);
  const dt_hand_sorted = py.sorted(dt_hand_vals);
  const dt_same_sorted = py.sorted(dt_same_vals);
  const sustain_stats = {
    nps1k_q50: q50_1k,
    nps1k_q75: q75_1k,
    nps1k_q90: q90_1k,
    nps4k_q50: _quantile(nps4k_sorted, 0.5),
    nps4k_q75: _quantile(nps4k_sorted, 0.75),
    nps4k_q90: _quantile(nps4k_sorted, 0.9),
    nps1k_flatness: _clamp(q50_1k / pymax(q97_1k, 1e-6), 0.0, 1.0),
    log_nps1k_mean: py.sum(log_nps_vals) / pymax(log_nps_vals.length, 1),
    dt_row_q25: _quantile(dt_row_sorted, 0.25),
    dt_row_q50: _quantile(dt_row_sorted, 0.5),
    dt_row_q75: _quantile(dt_row_sorted, 0.75),
    dt_hand_q25: _quantile(dt_hand_sorted, 0.25),
    dt_hand_median: _quantile(dt_hand_sorted, 0.5),
    dt_same_q25: _quantile(dt_same_sorted, 0.25),
    dt_same_median: _quantile(dt_same_sorted, 0.5),
  };

  const n = rows.length;
  const stats = {
    ...activity,
    chord_rate: chord_rows / pymax(n, 1),
    three_rate: three_rows / pymax(n, 1),
    overlap_rate: overlap_events_total / pymax(n, 1),
    rotation_rate: rotation_total / pymax(eligible_hand, 1),
    same_hand_q10: dt_hand_vals.length ? _quantile(py.sorted(dt_hand_vals), 0.1) : 0.0,
    fast_jack_rate: fast_jack_strength_total / pymax(dt_same_vals.length, 1),
    anchor_rate: anchor_strength_total / pymax(n, 1),
    anchor_imbalance: n ? (py.max(col_counts) - py.min(col_counts)) / pymax(n, 1) : 0.0,
    hand_bias: Math.abs(left_load - right_load) / py.max2(left_load, right_load, 1e-6),
    peak_to_sustain_gap,
    ...sustain_stats,
  };
  return { series, local_raw, stats, n_rows: n };
}

export function _quantile(sorted_vals, q) {
  const n = sorted_vals.length;
  if (n === 0) return 0.0;
  const t = q * (n - 1);
  const lo = py.int(t);
  const hi = pymin(n - 1, lo + 1);
  const w = t - lo;
  return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w;
}

export function _summarize(values, cfg) {
  if (!py.truthy(values)) return { aggregate: 0.0 };
  const sorted_vals = np.sort(Float64Array.from(values, Number)); // np.sort(np.asarray(values, dtype=float))
  const n = sorted_vals.length;
  const q50 = _quantile(sorted_vals, 0.5);
  const q75 = _quantile(sorted_vals, 0.75);
  const q90 = _quantile(sorted_vals, 0.9);
  const q97 = _quantile(sorted_vals, 0.97);
  const tail_count = pymax(1, py.math.ceil(n * item(cfg, 'tail_ratio')));
  const tail_mean = np.mean(sorted_vals.subarray(n - tail_count));
  const p = item(cfg, 'power_mean_p');
  // np.mean(sorted_vals ** p) ** (1.0 / p): ndarray ** scalar (numpy fast paths, else the CRT
  // pow), then np.float64 ** float = the CRT pow() without Python exceptions
  const power_mean = np.crtPow(np.mean(np.power(sorted_vals, p)), py.div(1.0, p));
  const agg_cfg = item(cfg, 'agg');
  const agg = (agg_cfg.q97 * q97 + agg_cfg.q90 * q90
    + agg_cfg.tail * tail_mean + agg_cfg.q75 * q75
    + agg_cfg.power * power_mean + agg_cfg.q50 * q50);
  return { q50, q75, q90, q97, tail_mean, power_mean, aggregate: agg };
}

export function _section_aggregate(rows, local_raw, cfg) {
  if (!py.truthy(rows) || !py.truthy(local_raw)) return 0.0;
  const first_t = rows[0].t;
  const section_max = new Map();
  let smoothed = local_raw[0];
  const section_ms = item(cfg, 'section_ms');
  const alpha = item(cfg, 'section_ema_alpha');
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const section = pymax(0, py.int(py.div(row.t - first_t, section_ms)));
    const raw = py.at(local_raw, i);
    smoothed += alpha * (raw - smoothed);
    section_max.set(section, pymax(section_max.has(section) ? section_max.get(section) : 0.0, smoothed));
  }
  const values = py.sortedBy([...section_max.values()].filter((v) => v > 0), (v) => v, true);
  if (!values.length) return 0.0;
  let weight = 1.0;
  let total = 0.0;
  let weight_total = 0.0;
  const decay = item(cfg, 'section_decay');
  for (const v of values) {
    total += v * weight;
    weight_total += weight;
    weight *= decay;
  }
  return py.div(total, weight_total);
}

export function compute_corrections(stats, cfg) {
  const c = item(cfg, 'corrections');
  let total = 0.0;
  const details = {};
  const has = (spec, k) => py.contains(spec, k);
  for (const name of dictKeys(c)) {
    const spec = item(c, name);
    const gain = py.get(spec, 'gain', 0.0);
    if (gain === 0) {
      details[name] = 0.0;
      continue;
    }
    let val = 1.0;
    if (has(spec, 'chord')) val *= _gate(item(stats, 'chord_rate'), ...item(spec, 'chord'));
    if (has(spec, 'overlap')) val *= _gate(item(stats, 'overlap_rate'), ...item(spec, 'overlap'));
    if (has(spec, 'overlap_inv')) val *= _inverse_gate(item(stats, 'overlap_rate'), ...item(spec, 'overlap_inv'));
    if (has(spec, 'nps')) val *= 1.0 - _gate(item(stats, 'avg_nps'), ...item(spec, 'nps'));
    if (has(spec, 'imbalance')) val *= 1.0 - _gate(item(stats, 'anchor_imbalance'), ...item(spec, 'imbalance'));
    if (has(spec, 'rotation')) val *= _gate(item(stats, 'rotation_rate'), ...item(spec, 'rotation'));
    if (has(spec, 'q10')) val *= _inverse_gate(item(stats, 'same_hand_q10'), ...item(spec, 'q10'));
    if (has(spec, 'dur')) val *= _gate(item(stats, 'active_duration_s'), ...item(spec, 'dur'));
    if (has(spec, 'break')) val *= _gate(item(stats, 'break_density'), ...item(spec, 'break'));
    if (has(spec, 'peak_gap')) val *= _gate(item(stats, 'peak_to_sustain_gap'), ...item(spec, 'peak_gap'));
    if (has(spec, 'anchor')) val *= _gate(item(stats, 'anchor_rate'), ...item(spec, 'anchor'));
    if (has(spec, 'fast_jack')) val *= _gate(item(stats, 'fast_jack_rate'), ...item(spec, 'fast_jack'));
    if (has(spec, 'fast_jack_inv')) val *= _inverse_gate(item(stats, 'fast_jack_rate'), ...item(spec, 'fast_jack_inv'));
    if (has(spec, 'bias')) val *= _gate(item(stats, 'hand_bias'), ...item(spec, 'bias'));
    details[name] = gain * val;
    total += details[name];
  }
  const clampV = item(cfg, 'correction_clamp');
  return { details, total: _clamp(total, -clampV, clampV) };
}

/** Full pipeline: rows -> curve -> aggregated numeric (~0-20 scale). */
export function biomech_numeric(rows, cfg = null) {
  cfg = cfg !== null && cfg !== undefined ? cfg : BIO_CONFIG;
  if (!py.truthy(rows)) return { bio_numeric: 1.0, raw_agg: 0.0, details: {} };
  const curve = compute_bio_curve(rows, cfg);
  let weighted_agg = 0.0;
  const summaries = {};
  const stream_weights = item(cfg, 'stream_weights');
  for (const name of dictKeys(stream_weights)) {
    const s = _summarize(item(curve.series, name), cfg);
    summaries[name] = s;
    weighted_agg += item(stream_weights, name) * s.aggregate;
  }
  const section_agg = _section_aggregate(rows, curve.local_raw, cfg);
  const raw_agg = 0.8 * weighted_agg + 0.2 * section_agg;
  const log_raw = py.math.log1p(pymax(0.0, raw_agg));
  const rm = item(cfg, 'raw_map');
  const pre_numeric = _clamp(rm.y0 + (log_raw - rm.p02) / pymax(1e-9, rm.p98 - rm.p02) * (rm.y1 - rm.y0), -2.5, 21.0);
  const corrections = compute_corrections(curve.stats, cfg);
  const numeric = _clamp(pre_numeric + corrections.total, -2.0, 20.0);
  return {
    bio_numeric: py.round(numeric, 4),
    pre_numeric: py.round(pre_numeric, 4),
    raw_agg: py.round(raw_agg, 4),
    correction_total: py.round(corrections.total, 4),
    corrections: corrections.details,
    summaries,
    stats: curve.stats,
  };
}
