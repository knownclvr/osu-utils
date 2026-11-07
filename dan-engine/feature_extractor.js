// feature_extractor.js — structural chart features (row mix, jacks, NPS, LN holds).
//
// JavaScript port of engine/dan_engine/feature_extractor.py (DanOverlay 3.1.0 by 8doul).
import * as py from './py.js';

/** Python `x or dflt`. */
const or = (v, dflt) => (py.truthy(v) ? v : dflt);

/** Python two-target unpacking `a, b = x` (ValueError when x does not hold exactly 2 items). */
function unpack2(x) {
  if (x.length !== 2) {
    throw new py.ValueError(x.length > 2
      ? `too many values to unpack (expected 2, got ${x.length})`
      : `not enough values to unpack (expected 2, got ${x.length})`);
  }
  return x;
}

/**
 * Compute NPS in overlapping windows across the chart.
 *
 * Uses a two-pointer sliding window for pure O(n + m) linear total.
 * Returns list of (center_ms, nps) for all windows with at least 1 note.
 */
export function _nps_windows(notes, window_ms = 500, stride_ms = 250) {
  if (!py.truthy(notes)) return [];
  const times = notes.map((note) => unpack2(note)[0]);
  const t_min = times[0];
  const t_max = times[times.length - 1];
  const half = py.floordiv(window_ms, 2);
  const duration_s = window_ms / 1000.0;
  const results = [];
  const n = times.length;
  let left_idx = 0;
  let right_idx = 0;
  let t = t_min;
  while (t <= t_max) {
    const lo = t - half;
    const hi = t + half;
    while (left_idx < n && times[left_idx] < lo) left_idx += 1;
    while (right_idx < n && times[right_idx] <= hi) right_idx += 1;
    const count = right_idx - left_idx;
    const nps = duration_s > 0 ? count / duration_s : 0.0;
    results.push([t, nps]);
    t += stride_ms;
  }
  return results;
}

/** Coefficient of variation (std/mean, 0 if mean ≈ 0). */
export function _cv(values) {
  if (values.length < 2) return 0.0;
  const mu = py.sum(values) / values.length;
  if (mu < 1e-6) return 0.0;
  // (v - mu) ** 2 : CPython float ** int calls the CRT pow(|v - mu|, 2.0), which is not always
  // the correctly rounded square (differs from x*x for ~0.2% of x) → py.pow
  const variance = py.sum(values.map((v) => py.pow(v - mu, 2))) / values.length;
  return py.math.sqrt(variance) / mu;
}

/**
 * Extracts structural features from a parsed dict (output of parsear_osu_v2()).
 *
 * Returns a dict with: stream_purity, jump_ratio, hand_ratio, quad_ratio, jack_ratio,
 * jack_density, vibro_density, anchor_ratio, minijack_ratio, density_cv, transition_var,
 * nps_p50/p90/p95, nps_sustained_top30, nps_active_ratio, nps_active_cv, bpm, duration_s,
 * stamina_index, burst_ratio, pattern_irregularity, timing_irregularity, chord_complexity,
 * nps_curve, plus the LN features of _extract_ln_features().
 */
export function extract_features(parsed) {
  const notes = py.get(parsed, 'notes', []);
  const rows = py.get(parsed, 'rows', []);
  const bpm = py.float(or(py.get(parsed, 'bpm', 120.0), 120.0));
  const drain_s = py.float(or(py.get(parsed, 'drain_time_s', 0.0), 0.0));
  const note_count = py.len(notes);
  const row_count = py.len(rows);

  // Notes ordered by time.
  const notes_sorted = py.sortedBy(notes, (n) => n[0]);

  // ── Row structure ──
  // Fraction of rows holding 1 / 2 / 3 / 4+ simultaneous notes.
  let stream_purity;
  let jump_ratio;
  let hand_ratio;
  let quad_ratio;
  if (row_count > 0) {
    const sizes = rows.map((r) => py.len(py.getitem(r, 'cols')));
    let n1 = 0;
    let n2 = 0;
    let n3 = 0;
    let n4 = 0;
    for (const s of sizes) {
      if (s === 1) n1 += 1;
      if (s === 2) n2 += 1;
      if (s === 3) n3 += 1;
      if (s >= 4) n4 += 1;
    }
    stream_purity = n1 / row_count;
    jump_ratio = n2 / row_count;
    hand_ratio = n3 / row_count;
    quad_ratio = n4 / row_count;
  } else {
    stream_purity = jump_ratio = hand_ratio = quad_ratio = 0.0;
  }

  // ── Jacks ──
  // Gaps between consecutive notes of the same column:
  // vibro ≤ 80ms, strict jack ≤ 120ms, jack ≤ 180ms, minijack 100-200ms.
  const by_col = new py.DefaultDict(() => []);
  for (const note of notes_sorted) {
    const [t, c] = unpack2(note);
    by_col.getOrCreate(py.int(c)).push(py.int(t));
  }

  const jack_threshold_ms = 180;
  const jack_strict_ms = 120;
  const vibro_threshold_ms = 80;

  let jack_hits = 0;
  let jack_strict_hits = 0;
  let vibro_hits = 0;
  let minijack_hits = 0;
  let total_pairs = 0;

  for (const times of by_col.values()) {
    const col_sorted = py.sorted(times);
    for (let i = 1; i < col_sorted.length; i++) {
      const gap = col_sorted[i] - col_sorted[i - 1];
      total_pairs += 1;
      if (gap <= vibro_threshold_ms) {
        vibro_hits += 1;
        jack_strict_hits += 1;
        jack_hits += 1;
      } else if (gap <= jack_strict_ms) {
        jack_strict_hits += 1;
        jack_hits += 1;
      } else if (gap <= jack_threshold_ms) {
        jack_hits += 1;
      }
      if (gap >= 100 && gap <= 200) minijack_hits += 1;
    }
  }

  let jack_ratio;
  let jack_density;
  let vibro_density;
  if (note_count > 0 && total_pairs > 0) {
    jack_ratio = jack_hits / total_pairs;
    jack_density = jack_strict_hits / total_pairs;
    vibro_density = vibro_hits / total_pairs;
  } else {
    jack_ratio = jack_density = vibro_density = 0.0;
  }

  const minijack_ratio = minijack_hits / py.max2(total_pairs, 1);

  // ── Anchors ──
  // How far the busiest column exceeds an even split across the columns.
  let anchor_ratio;
  if (note_count > 0) {
    const col_counts = new Map();
    for (const [c, ts] of by_col) col_counts.set(c, ts.length);
    const max_col_count = col_counts.size ? py.max(col_counts.values()) : 0;
    const num_cols = col_counts.size || 4;
    const expected_per_col = note_count / num_cols;

    anchor_ratio = py.max2(0.0, (max_col_count - expected_per_col) / py.max2(note_count, 1));
  } else {
    anchor_ratio = 0.0;
  }

  // ── Transitions ──
  // Mean Jaccard distance between the column sets of consecutive rows.
  let transition_var;
  if (rows.length >= 2) {
    const transition_diffs = [];
    for (let i = 1; i < rows.length; i++) {
      const prev_cols = new Set(py.getitem(rows[i - 1], 'cols'));
      const curr_cols = new Set(py.getitem(rows[i], 'cols'));
      let inter = 0;
      for (const c of curr_cols) if (prev_cols.has(c)) inter += 1;
      const union = prev_cols.size + curr_cols.size - inter;

      if (union > 0) transition_diffs.push(1.0 - inter / union);
    }
    transition_var = transition_diffs.length ? py.sum(transition_diffs) / transition_diffs.length : 0.0;
  } else {
    transition_var = 0.0;
  }

  // ── Density (NPS) ──
  // 500ms windows sampled every 250ms.
  const nps_data = _nps_windows(notes_sorted, 500, 250);
  const nps_values = nps_data.length ? nps_data.map((w) => unpack2(w)[1]) : [0.0];

  // NPS curve [[ms, nps], ...], downsampled to at most 200 points.
  const _NPS_CHART_MAX = 200;
  let nps_curve;
  if (nps_data.length && nps_data.length > _NPS_CHART_MAX) {
    const _step = nps_data.length / _NPS_CHART_MAX;
    nps_curve = [];
    for (let i = 0; i < _NPS_CHART_MAX; i++) {
      const w = nps_data[py.int(i * _step)];
      nps_curve.push([py.int(w[0]), py.round(w[1], 2)]);
    }
  } else {
    nps_curve = nps_data.length ? nps_data.map((w) => {
      const [ms, v] = unpack2(w);
      return [py.int(ms), py.round(v, 2)];
    }) : [];
  }

  const density_cv = _cv(nps_values);

  const nps_sorted = py.sorted(nps_values);
  let nps_p90;
  let nps_p50;
  let nps_p95;
  if (nps_sorted.length) {
    const p90_idx = py.int(0.9 * nps_sorted.length);
    nps_p90 = nps_sorted[py.min2(p90_idx, nps_sorted.length - 1)];
    const p50_idx = py.int(0.5 * nps_sorted.length);
    nps_p50 = nps_sorted[py.min2(p50_idx, nps_sorted.length - 1)];
    const p95_idx = py.int(0.95 * nps_sorted.length);
    nps_p95 = nps_sorted[py.min2(p95_idx, nps_sorted.length - 1)];
  } else {
    nps_p90 = nps_p50 = nps_p95 = 0.0;
  }

  const top30_cutoff = py.int(0.7 * nps_sorted.length);
  const top30_vals = nps_sorted.length ? py.slice(nps_sorted, top30_cutoff) : [];
  const nps_sustained_top30 = top30_vals.length ? py.sum(top30_vals) / top30_vals.length : 0.0;

  const active_threshold = 2.0;
  const active_vals = nps_values.filter((v) => v > active_threshold);
  const nps_active_ratio = active_vals.length / py.max2(nps_values.length, 1);
  const nps_active_cv = active_vals.length >= 2 ? _cv(active_vals) : 0.0;

  // ── Stamina ──
  // sustained_fraction: mean NPS of the densest 30% of windows divided
  // by the p90 NPS (how well the chart holds its peak density).
  // stamina_index: sustained_fraction scaled by log(1 + drain_s / 90),
  // so longer charts weigh more (0 when there is no drain time or no density).
  let stamina_index;
  if (drain_s > 0 && nps_p90 > 0) {
    const sustained_fraction = nps_sustained_top30 / nps_p90;
    stamina_index = sustained_fraction * py.math.log(1 + drain_s / 90.0);
  } else {
    stamina_index = 0.0;
  }

  // Burst ratio: p95 NPS relative to the median (p50) NPS.
  const burst_ratio = nps_p95 / py.max2(nps_p50, 0.01);

  // ── Pattern irregularity ──
  // CV of the per-window column entropy (2s windows, 1s stride, columns 0-3).
  const _WINDOW_ENTROPY_MS = 2000;
  const _ENTROPY_STRIDE_MS = 1000;
  let pattern_irregularity;
  if (notes_sorted.length && drain_s > 0) {
    const t_min = notes_sorted[0][0];
    const t_max = notes_sorted[notes_sorted.length - 1][0];
    const entropies = [];
    let t = t_min;
    let ni = 0;
    const n_notes = notes_sorted.length;
    while (t <= t_max) {
      const lo = t;
      const hi = t + _WINDOW_ENTROPY_MS;
      // skip notes before the window start
      while (ni < n_notes && notes_sorted[ni][0] < lo) ni += 1;
      // column histogram of the notes inside [lo, hi]
      const col_counts_w = [0, 0, 0, 0];
      let total_w = 0;
      for (let j = ni; j < n_notes; j++) {
        const [nt, nc] = unpack2(notes_sorted[j]);
        if (nt > hi) break;
        const c_int = py.int(nc);
        if (c_int >= 0 && c_int <= 3) {
          col_counts_w[c_int] += 1;
          total_w += 1;
        }
      }
      if (total_w >= 4) {
        // Shannon entropy (bits) of the column distribution
        let h = 0.0;
        for (const cnt of col_counts_w) {
          if (cnt > 0) {
            const p = cnt / total_w;
            h -= p * py.math.log2(p);
          }
        }
        entropies.push(h);
      }
      t += _ENTROPY_STRIDE_MS;
    }
    pattern_irregularity = entropies.length >= 2 ? _cv(entropies) : 0.0;
  } else {
    pattern_irregularity = 0.0;
  }

  // ── Timing irregularity ──
  // Coefficient of variation of the gaps between consecutive rows:
  // near 0 for a steady rhythm, higher with mixed snaps and rhythm changes.
  let timing_irregularity;
  if (row_count >= 3) {
    const row_times = rows.map((r) => py.getitem(r, 't'));
    const inter_row_gaps = [];
    for (let i = 1; i < row_times.length; i++) inter_row_gaps.push(row_times[i] - row_times[i - 1]);
    timing_irregularity = inter_row_gaps.length ? _cv(inter_row_gaps) : 0.0;
  } else {
    timing_irregularity = 0.0;
  }

  // ── Chord complexity ──
  // Share of 3+ note rows, amplified by the note density around chords.
  const frac_chords_3plus = hand_ratio + quad_ratio;
  let chord_complexity;
  if (frac_chords_3plus > 0 && row_count > 0 && nps_values.length) {
    // timestamps of the rows with 3+ notes
    const chord_times = new Set();
    for (const r of rows) {
      if (py.len(py.getitem(r, 'cols')) >= 3) chord_times.add(py.getitem(r, 't'));
    }
    if (chord_times.size) {
      // NPS of the windows within ±250ms of a chord vs. the overall mean
      const nps_during_chords = [];
      const overall_nps_mean = py.sum(nps_values) / nps_values.length;
      // any(lo <= ct <= hi for ct in chord_times): the smallest chord time >= lo is <= hi
      const chord_sorted = py.sorted(chord_times);
      for (const w of nps_data) {
        const [center_t, nps_val] = unpack2(w);
        const lo = center_t - 250;
        const hi = center_t + 250;
        const k = py.bisect_left(chord_sorted, lo);
        if (k < chord_sorted.length && chord_sorted[k] <= hi) nps_during_chords.push(nps_val);
      }
      if (nps_during_chords.length && overall_nps_mean > 0) {
        const density_amp = (py.sum(nps_during_chords) / nps_during_chords.length) / overall_nps_mean;
        chord_complexity = frac_chords_3plus * (1 + density_amp);
      } else {
        chord_complexity = frac_chords_3plus;
      }
    } else {
      chord_complexity = 0.0;
    }
  } else {
    chord_complexity = 0.0;
  }

  return {
    stream_purity: py.round(stream_purity, 4),
    jump_ratio: py.round(jump_ratio, 4),
    hand_ratio: py.round(hand_ratio, 4),
    quad_ratio: py.round(quad_ratio, 4),
    jack_ratio: py.round(jack_ratio, 4),
    jack_density: py.round(jack_density, 4),
    vibro_density: py.round(vibro_density, 4),
    anchor_ratio: py.round(anchor_ratio, 4),
    minijack_ratio: py.round(minijack_ratio, 4),
    density_cv: py.round(density_cv, 4),
    transition_var: py.round(transition_var, 4),
    nps_p50: py.round(nps_p50, 3),
    nps_p90: py.round(nps_p90, 3),
    nps_p95: py.round(nps_p95, 3),
    nps_sustained_top30: py.round(nps_sustained_top30, 3),
    nps_active_ratio: py.round(nps_active_ratio, 4),
    nps_active_cv: py.round(nps_active_cv, 4),
    bpm: py.round(bpm, 2),
    duration_s: py.round(drain_s, 2),

    stamina_index: py.round(stamina_index, 4),
    burst_ratio: py.round(burst_ratio, 4),
    pattern_irregularity: py.round(pattern_irregularity, 4),
    timing_irregularity: py.round(timing_irregularity, 4),
    chord_complexity: py.round(chord_complexity, 4),
    nps_curve,
    // LN-specific features
    ..._extract_ln_features(parsed),
  };
}

// LN features are merged into the extract_features() result so the
// pipeline can route LN maps without re-parsing the file.
/**
 * Extract LN-specific structural features for LN family classification.
 *
 * Keys: ln_ratio, hold_occupancy, ln_duration_mean_ms, ln_duration_cv,
 * simultaneous_hold, release_density, hold_chord_ratio.
 */
export function _extract_ln_features(parsed) {
  const note_events = py.get(parsed, 'note_events', []);
  const ln_ratio = py.float(or(py.get(parsed, 'ln_ratio', 0.0), 0.0));
  const drain_s = py.float(or(py.get(parsed, 'drain_time_s', 0.0), 0.0));

  const ln_starts = note_events.filter((e) => py.get(e, 'event_type') === 'ln_start');

  if (!ln_starts.length || drain_s < 1.0) {
    return {
      ln_ratio: py.round(ln_ratio, 4),
      hold_occupancy: 0.0,
      ln_duration_mean_ms: 0.0,
      ln_duration_cv: 0.0,
      simultaneous_hold: 0.0,
      release_density: 0.0,
      hold_chord_ratio: 0.0,
    };
  }

  // Hold durations
  const durations = [];
  for (const e of ln_starts) if (py.get(e, 'duration_ms', 0) > 0) durations.push(py.getitem(e, 'duration_ms'));
  const ln_duration_mean_ms = durations.length ? py.sum(durations) / durations.length : 0.0;
  const ln_duration_cv = durations.length >= 2 ? _cv(durations) : 0.0;

  // Hold occupancy: fraction of the chart span covered by at least one
  // active hold (sweep over the start/end events).
  const first_ms = py.truthy(py.get(parsed, 'notes')) ? py.getitem(parsed, 'notes')[0][0] : 0;
  const last_ms = py.truthy(py.get(parsed, 'notes')) ? py.at(py.getitem(parsed, 'notes'), -1)[0] : 0;
  const chart_span_ms = py.max2(last_ms - first_ms, 1);

  let occupied_ms = 0;
  let events_sorted = [];
  for (const e of ln_starts) {
    const s = py.get(e, 'ln_start_ms', 0);
    const end = py.get(e, 'ln_end_ms', 0);
    if (end > s) {
      events_sorted.push([s, 1]);
      events_sorted.push([end, -1]);
    }
  }
  events_sorted = py.sortedBy(events_sorted, (x) => [x[0], x[1]]);

  let active = 0;
  let seg_start = 0;
  for (const [t, delta] of events_sorted) {
    if (active > 0 && t > seg_start) occupied_ms += t - seg_start;
    active += delta;
    seg_start = t;
  }

  const hold_occupancy = py.min2(1.0, occupied_ms / chart_span_ms);

  // Simultaneous holds: LN starts that begin while another hold is still active.
  const starts_by_time = py.sortedBy(ln_starts, (e) => py.get(e, 'ln_start_ms', 0));
  let simultaneous_count = 0;
  let active_ends = [];
  for (const e of starts_by_time) {
    const s = py.get(e, 'ln_start_ms', 0);
    // drop the holds that ended at or before this start
    active_ends = active_ends.filter((end) => end > s);
    if (active_ends.length) simultaneous_count += 1;
    active_ends.push(py.get(e, 'ln_end_ms', 0));
    active_ends = py.sorted(active_ends);
  }
  const simultaneous_hold = simultaneous_count / starts_by_time.length;

  // Release density: LN ends per second of drain time.
  const ln_ends = note_events.filter((e) => py.get(e, 'event_type') === 'ln_end');
  const release_density = drain_s > 0 ? ln_ends.length / drain_s : 0.0;

  // Hold chords: LN starts sharing their timestamp with another LN start.
  const start_times = ln_starts.map((e) => py.get(e, 'ln_start_ms', 0));
  const time_counts = new py.Counter(start_times);
  let chorded = 0;
  for (const t of start_times) if (time_counts.count(t) >= 2) chorded += 1;
  const hold_chord_ratio = start_times.length ? chorded / start_times.length : 0.0;

  return {
    ln_ratio: py.round(ln_ratio, 4),
    hold_occupancy: py.round(hold_occupancy, 4),
    ln_duration_mean_ms: py.round(ln_duration_mean_ms, 1),
    ln_duration_cv: py.round(ln_duration_cv, 4),
    simultaneous_hold: py.round(simultaneous_hold, 4),
    release_density: py.round(release_density, 3),
    hold_chord_ratio: py.round(hold_chord_ratio, 4),
  };
}
