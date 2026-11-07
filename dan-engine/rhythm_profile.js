// rhythm_profile.js — pattern-based chart family classifier (rhythm texture detectors).
//
// JavaScript port of engine/dan_engine/rhythm_profile.py (DanOverlay 3.1.0 by 8doul).
//
// This classifier is based on the chart classification engine from ManiaMapAnalyser
// (by Leo_Black). It detects rhythmic textures (streams, jacks, chords, coordination
// patterns) directly from note data and maps them to the DanOverlay family taxonomy,
// without depending on the Sunny SR components.
//
// Port note: _extract_textures() scans `remaining = remaining[1:]` in Python (a fresh
// list per step). Here the detectors take an extra optional start offset `o` instead
// (`detector(frames, o)` sees frames[o:]), which is equivalent and avoids O(n²) copies.
import * as py from './py.js';

export const _PATTERN_CONFIG = {
  stability_threshold: 5.0,
  bpm_cluster_tolerance: 5.0,
  important_cluster_ratio: 0.5,
  jack_min_bpm: 90.0,
  shield_max_beat_ratio: 0.25,
  inverse_gap_tolerance_ms: 5.0,
  inverse_min_filled_lanes: 3,
  release_scan_rows: 4,
  release_min_tail_rows: 4,
  release_roll_points: 2,
  release_full_match_rows: 5,
  jack_context_window: 6,
  jack_fallback_max_ms_per_beat: 185.0,
  sv_speed_eps: 0.05,
  sv_extreme_bpm_min: 20.0,
  sv_extreme_bpm_max: 450.0,
  sv_extreme_bpm_ratio: 4.0,
  sv_amount_threshold: 2000.0,
};

export const _PRIMARY_RHYTHM_WEIGHTS = {
  linear_stream: 1 / 3,
  harmonic_flow: 0.65,
  anchor_burst: 0.9,
  coordination: 0.75,
  density: 0.9,
  wildcard: 1.0,
};

export const _TEXTURE_TO_FAMILY = {
  linear_stream: 'stream',
  harmonic_flow: 'speed',
  anchor_burst: 'jack',
  coordination: 'tech',
  density: 'stamina',
  wildcard: 'tech',
};

export const _SUB_TEXTURE_FAMILY = {
  rolls: 'stream',
  trills: 'tech',
  mini_trills: 'tech',

  chord_stream: 'speed',
  split_trill: 'tech',
  jump_trill: 'tech',
  jumpstream: 'speed',
  handstream: 'stamina',
  double_jump: 'speed',
  triple_jump: 'speed',
  quad_stream: 'stamina',
  light_chords: 'speed',
  dense_chords: 'stamina',
  chord_roll: 'tech',
  brackets: 'tech',

  long_jacks: 'jack',
  chord_jacks: 'jack',
  mini_jacks: 'jack',
  gluts: 'jack',

  column_lock: 'tech',
  shield: 'tech',
  release: 'tech',

  inverse: 'tech',
  js_density: 'speed',
  hs_density: 'stamina',
  ds_density: 'stamina',
  dcs_density: 'stamina',
  lcs_density: 'speed',

  jack_wild: 'jack',
  speed_wild: 'speed',
};

// ── small Python helpers ───────────────────────────────────────────
/** Python `x or dflt`. */
const or = (v, dflt) => (py.truthy(v) ? v : dflt);
/** dict membership (`k in d`) for the plain-object tables above. */
const has = (d, k) => Object.prototype.hasOwnProperty.call(d, k);
/** str(list_a) == str(list_b) for lists of ints. */
function sameList(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
/** str() of None / bool / str values used in f-strings. */
function pyStr(v) {
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  return String(v);
}

// ── Beat frames ────────────────────────────────────────────────────

/** Number of keys assigned to left hand for a given key-mode. */
export function _hands_per_side(key_count) {
  if (key_count === 3) return 2;
  if (key_count === 4) return 2;
  if (key_count === 5) return 3;
  if (key_count === 6) return 3;
  if (key_count === 7) return 4;
  if (key_count === 8) return 4;
  if (key_count === 9) return 5;
  if (key_count === 10) return 5;
  return py.max2(1, py.floordiv(key_count, 2));
}

export function _is_same_hand(col_a, col_b, split) {
  if (Math.abs(col_a - col_b) !== 1) return false;
  return (col_a < split) === (col_b < split);
}

/** Determine direction of movement and whether it is a roll. */
export function _detect_flow(prev_cols, cur_cols) {
  if (!prev_cols.length || !cur_cols.length) return ['none', false];
  const pl = prev_cols[0];
  const pr = prev_cols[prev_cols.length - 1];
  const cl = cur_cols[0];
  const cr = cur_cols[cur_cols.length - 1];
  const dl = cl - pl;
  const dr = cr - pr;

  let direction;
  if (dl > 0) direction = dr > 0 ? 'right' : 'inward';
  else if (dl < 0) direction = dr < 0 ? 'left' : 'outward';
  else if (dr < 0) direction = 'inward';
  else if (dr > 0) direction = 'outward';
  else direction = 'none';

  const is_roll = pl > cr || pr < cl;
  return [direction, is_roll];
}

/**
 * Convert raw notes into beat-frames with timing and column metadata.
 *
 * Each note dict has: time (ms) and data (per-column kind: 'tap', 'hold_head',
 * 'hold_body', 'hold_tail' or None).
 */
export function _build_beat_frames(notes, key_count, bpm_timeline) {
  if (!py.truthy(notes)) return [];

  const first_time = py.getitem(notes[0], 'time');
  const left_keys = _hands_per_side(key_count);

  const _ms_per_beat_at = (time_ms) => {
    if (!py.truthy(bpm_timeline)) return 500.0;
    let current = py.get(bpm_timeline[0], 'ms_per_beat', 500.0);
    for (const item of bpm_timeline) {
      if (py.getitem(item, 'time') > time_ms) break;
      current = py.get(item, 'ms_per_beat', 500.0);
    }
    return current;
  };

  const first_data = py.get(notes[0], 'data', []);
  let previous_cols = [];
  for (let k = 0; k < key_count; k++) {
    if (k < first_data.length && (first_data[k] === 'tap' || first_data[k] === 'hold_head')) previous_cols.push(k);
  }
  if (!previous_cols.length) return [];

  let previous_time = first_time;
  const frames = [];
  let idx = 0;

  for (let i = 1; i < notes.length; i++) {
    const item = notes[i];
    const t = py.getitem(item, 'time');
    const row_data = py.get(item, 'data', []);
    idx += 1;

    const tap_cols = [];
    const head_cols = [];
    const body_cols = [];
    const tail_cols = [];
    for (let k = 0; k < key_count; k++) {
      if (k >= row_data.length) continue;
      const kind = row_data[k];
      if (kind === 'tap') tap_cols.push(k);
      else if (kind === 'hold_head') head_cols.push(k);
      else if (kind === 'hold_body') body_cols.push(k);
      else if (kind === 'hold_tail') tail_cols.push(k);
    }

    const active_cols = tap_cols.concat(head_cols);
    if (!active_cols.length && !head_cols.length && !body_cols.length && !tail_cols.length) continue;

    let direction = 'none';
    let is_roll = false;
    let jacks = 0;

    if (active_cols.length) {
      [direction, is_roll] = _detect_flow(previous_cols, active_cols);
      const prev_set = new Set(previous_cols);
      for (const c of active_cols) if (prev_set.has(c)) jacks += 1;
    }

    frames.push({
      index: idx,
      offset_ms: t - first_time,
      ms_per_beat: (t - previous_time) * 4.0,
      beat_length: _ms_per_beat_at(t),
      active_count: active_cols.length,
      jack_count: jacks,
      direction,
      is_roll,
      key_count,
      left_keys,
      head_cols,
      body_cols,
      tail_cols,
      tap_cols,
      raw_cols: active_cols,
    });

    if (active_cols.length) previous_cols = active_cols;
    previous_time = t;
  }

  return frames;
}

// ── Primary rhythm detectors (return the matched frame count, 0 = no match) ──
// Every detector looks at frames[o:] (Python: the detector's `frames` argument).

/** Linear stream: 5 consecutive single notes, no jacks, different start/end col. */
export function _detect_stream(frames, o = 0) {
  if (frames.length - o < 5) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  const e = frames[o + 4];
  if (a.active_count === 1 && a.jack_count === 0 &&
      b.active_count === 1 && b.jack_count === 0 &&
      c.active_count === 1 && c.jack_count === 0 &&
      d.active_count === 1 && d.jack_count === 0 &&
      e.active_count === 1 && e.jack_count === 0 &&
      a.raw_cols[0] !== e.raw_cols[0]) {
    return 5;
  }
  return 0;
}

/** Jack/anchor: first frame has same-column repeats at moderate speed. */
export function _detect_anchor_burst(frames, o = 0) {
  if (frames.length - o <= 0) return 0;
  const f0 = frames[o];
  return f0.jack_count > 1 && f0.ms_per_beat < 2000 ? 1 : 0;
}

/** Chordstream: multi-note without jacks, followed by more chord content. */
export function _detect_harmonic_flow(frames, o = 0) {
  if (frames.length - o < 4) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  if (a.active_count > 1 && a.jack_count === 0 &&
      b.jack_count === 0 && c.jack_count === 0 && d.jack_count === 0 &&
      (b.active_count > 1 || c.active_count > 1 || d.active_count > 1)) {
    return 4;
  }
  return 0;
}

/** LN-coordination: frame has hold-related activity. */
export function _detect_coordination(frames, o = 0) {
  if (frames.length - o <= 0) return 0;
  const a = frames[o];
  return a.head_cols.length || a.body_cols.length || a.tail_cols.length ? 1 : 0;
}

/** Dense-LN: starts with hold head. */
export function _detect_density(frames, o = 0) {
  if (frames.length - o <= 0) return 0;
  return frames[o].head_cols.length ? 1 : 0;
}

/** Wildcard-LN: starts with hold head (same as density at core level). */
export function _detect_wildcard(frames, o = 0) {
  return _detect_density(frames, o);
}

// ── Sub-texture detectors (4K only, see _SPECIFIC_4K) ──

export function _sub_long_jacks(frames, o = 0) {
  if (frames.length - o < 5) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  const e = frames[o + 4];
  if (a.jack_count > 0 && b.jack_count > 0 && c.jack_count > 0 && d.jack_count > 0 && e.jack_count > 0) {
    for (const col of a.raw_cols) {
      if (b.raw_cols.includes(col) && c.raw_cols.includes(col) && d.raw_cols.includes(col) && e.raw_cols.includes(col)) return 5;
    }
  }
  return 0;
}

export function _sub_chord_jacks(frames, o = 0) {
  if (frames.length - o < 2) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  if (a.active_count > 2 && b.active_count > 1 && b.jack_count >= 1 &&
      (b.active_count < a.active_count || b.jack_count < b.active_count)) {
    return 2;
  }
  return 0;
}

export function _sub_mini_jacks(frames, o = 0) {
  if (frames.length - o < 2) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  return a.jack_count > 0 && b.jack_count === 0 ? 2 : 0;
}

export function _sub_handstream(frames, o = 0) {
  if (frames.length - o < 4) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  return a.active_count === 3 && a.jack_count === 0 && b.jack_count === 0 && c.jack_count === 0 && d.jack_count === 0 ? 4 : 0;
}

export function _sub_jumpstream(frames, o = 0) {
  if (frames.length - o < 4) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  if (a.active_count === 2 && a.jack_count === 0 && b.active_count === 1 && b.jack_count === 0 && c.jack_count === 0 && d.jack_count === 0) {
    if (c.active_count < 3 && d.active_count < 3) return 4;
  }
  return 0;
}

export function _sub_jump_trill(frames, o = 0) {
  if (frames.length - o < 4) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  return a.active_count === 2 && b.active_count === 2 && c.active_count === 2 && d.active_count === 2 &&
    py.truthy(b.is_roll) && py.truthy(c.is_roll) && py.truthy(d.is_roll) ? 4 : 0;
}

export function _sub_split_trill(frames, o = 0) {
  if (frames.length - o < 3) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  return a.active_count === 2 && b.active_count === 2 && c.active_count === 2 && b.jack_count === 0 && c.jack_count === 0 &&
    !py.truthy(b.is_roll) && !py.truthy(c.is_roll) ? 3 : 0;
}

export function _sub_gluts(frames, o = 0) {
  if (frames.length - o < 3) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  if (b.jack_count === 1 && c.jack_count === 1) {
    for (const col of a.raw_cols) {
      if (b.raw_cols.includes(col) && c.raw_cols.includes(col)) return 0;
    }
    return 3;
  }
  return 0;
}

export function _sub_quad_stream(frames, o = 0) {
  if (frames.length - o < 4) return 0;
  const a = frames[o];
  const c = frames[o + 2];
  const d = frames[o + 3];
  return a.active_count === 4 && c.jack_count === 0 && d.jack_count === 0 ? 4 : 0;
}

export function _sub_roll(frames, o = 0) {
  if (frames.length - o < 3) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  if (a.active_count === 1 && b.active_count === 1 && c.active_count === 1) {
    const left = a.direction === 'left' && b.direction === 'left' && c.direction === 'left';
    const right = a.direction === 'right' && b.direction === 'right' && c.direction === 'right';
    if (left || right) return 3;
  }
  return 0;
}

export function _sub_trill(frames, o = 0) {
  if (frames.length - o < 4) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  if (b.jack_count === 0 && c.jack_count === 0 && d.jack_count === 0 &&
      sameList(a.raw_cols, c.raw_cols) && sameList(b.raw_cols, d.raw_cols)) {
    return 4;
  }
  return 0;
}

export function _sub_mini_trill(frames, o = 0) {
  if (frames.length - o < 4) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const c = frames[o + 2];
  const d = frames[o + 3];
  if (b.jack_count === 0 && c.jack_count === 0 &&
      sameList(a.raw_cols, c.raw_cols) && !sameList(b.raw_cols, d.raw_cols)) {
    return 4;
  }
  return 0;
}

export function _sub_column_lock(frames, o = 0) {
  if (frames.length - o < 3) return 0;
  const f0 = frames[o];
  const split = f0.left_keys;
  const ln_col = f0.head_cols.length ? f0.head_cols[0] : null;
  if (ln_col === null) return 0;
  const adj_cols = [ln_col - 1, ln_col + 1].filter((c) => c >= 0 && c < f0.key_count && _is_same_hand(ln_col, c, split));
  if (!adj_cols.length) return 0;
  for (const adj of adj_cols) {
    const hits = [];
    for (const row of frames.slice(o, o + 8)) {
      if (row.body_cols.includes(ln_col) && row.tap_cols.includes(adj)) hits.push(row.offset_ms);
    }
    if (hits.length < 3) continue;
    const bpms = [];
    for (let i = 0; i < hits.length - 1; i++) {
      bpms.push(hits[i + 1] > hits[i] ? 15000.0 / (hits[i + 1] - hits[i]) : 90.0);
    }
    if (bpms.length && py.max(bpms) >= _PATTERN_CONFIG.jack_min_bpm) return 3;
  }
  return 0;
}

export function _sub_shield(frames, o = 0) {
  if (frames.length - o < 2) return 0;
  const a = frames[o];
  const b = frames[o + 1];
  const dt = b.offset_ms - a.offset_ms;
  const beat_limit = b.beat_length * _PATTERN_CONFIG.shield_max_beat_ratio;
  if (dt < 0 || dt > beat_limit) return 0;
  for (const col of a.tap_cols) {
    if (b.head_cols.includes(col)) return 2;
  }
  for (const col of a.tail_cols) {
    if (b.tap_cols.includes(col)) return 2;
  }
  return 0;
}

export const _PRIMARY_DETECTORS = [
  ['linear_stream', _detect_stream],
  ['harmonic_flow', _detect_harmonic_flow],
  ['anchor_burst', _detect_anchor_burst],
  ['coordination', _detect_coordination],
  ['density', _detect_density],
  ['wildcard', _detect_wildcard],
];

export const _SPECIFIC_4K = {
  linear_stream: [['rolls', _sub_roll], ['trills', _sub_trill], ['mini_trills', _sub_mini_trill]],
  harmonic_flow: [['handstream', _sub_handstream], ['split_trill', _sub_split_trill], ['jump_trill', _sub_jump_trill], ['jumpstream', _sub_jumpstream]],
  anchor_burst: [['long_jacks', _sub_long_jacks], ['quad_stream', _sub_quad_stream], ['gluts', _sub_gluts], ['chord_jacks', _sub_chord_jacks], ['mini_jacks', _sub_mini_jacks]],
  coordination: [['column_lock', _sub_column_lock], ['shield', _sub_shield]],

  density: [],
  wildcard: [],
};

// ── Texture extraction, bundling and profiling ──

/** Scan beat-frames and extract all texture matches (port of findPatterns/find). */
export function _extract_textures(frames, key_count, total_ms) {
  const remaining = [...frames];
  const results = [];
  const specific_map = key_count === 4 ? _SPECIFIC_4K : {};

  // while remaining: ...; remaining = remaining[1:]   (remaining == frames[o:])
  for (let o = 0; o < remaining.length; o++) {
    const n_remaining = remaining.length - o;
    for (const [rhythm_name, detector] of _PRIMARY_DETECTORS) {
      const n = detector(remaining, o);
      if (n === 0) continue;

      const sub_list = py.get(specific_map, rhythm_name, []);
      let best_match = [n, null];
      for (const [sub_name, sub_fn] of sub_list) {
        const m = sub_fn(remaining, o);
        if (m > best_match[0]) best_match = [m, sub_name];
      }

      let [matched_len, matched_sub] = best_match;
      if (matched_len <= 0) matched_len = n;

      // segment = remaining[:matched_len]
      const seg_len = Math.min(matched_len, n_remaining);
      const mpbs = [];
      for (let k = 0; k < seg_len; k++) mpbs.push(remaining[o + k].ms_per_beat);
      const mean_mpb = py.div(py.sum(mpbs), seg_len);
      const mixed = !mpbs.every((m) => Math.abs(m - mean_mpb) < _PATTERN_CONFIG.stability_threshold);

      const start_ms = remaining[o].offset_ms;
      const end_ms = matched_len < n_remaining ? remaining[o + matched_len].offset_ms : total_ms;

      results.push({
        rhythm: rhythm_name,
        sub_texture: matched_sub,
        is_volatile: mixed,
        start_ms,
        end_ms,
        ms_per_beat: rhythm_name === 'density' && matched_sub === 'inverse' ? 0.0 : mean_mpb,
      });
    }
  }

  return results;
}

/** Group texture observations into BPM-timbre bundles (port of clustering). */
export function _bundle_textures(textures) {
  const stable_groups = [];
  const volatile_groups = new Map();

  const _add_stable = (mpb) => {
    for (const g of stable_groups) {
      if (Math.abs(g.anchor_mpb - mpb) < _PATTERN_CONFIG.bpm_cluster_tolerance) {
        g.sum_mpb += mpb;
        g.count += 1;
        return g;
      }
    }
    const g = { anchor_mpb: mpb, sum_mpb: mpb, count: 1, bpm: 0 };
    stable_groups.push(g);
    return g;
  };

  const _add_volatile = (rhythm, mpb) => {
    if (!volatile_groups.has(rhythm)) {
      volatile_groups.set(rhythm, { sum_mpb: mpb, count: 1, bpm: 0 });
    } else {
      const v = volatile_groups.get(rhythm);
      v.sum_mpb += mpb;
      v.count += 1;
    }
    return volatile_groups.get(rhythm);
  };

  const pairs = [];
  for (const tx of textures) {
    let c;
    if (py.truthy(tx.is_volatile)) c = _add_volatile(tx.rhythm, tx.ms_per_beat);
    else c = _add_stable(tx.ms_per_beat);
    pairs.push([tx, c]);
  }

  for (const g of stable_groups) {
    const avg = g.sum_mpb / py.max2(g.count, 1);
    g.bpm = avg > 0 ? py.roundInt(60000.0 / avg) : 0;
  }
  for (const v of volatile_groups.values()) {
    const avg = v.sum_mpb / py.max2(v.count, 1);
    v.bpm = avg > 0 ? py.roundInt(60000.0 / avg) : 0;
  }

  return pairs;
}

/** Summarize texture bundles into importance-ranked profiles (port of specificClusters). */
export function _profile_bundles(pairs) {
  const groups = new py.DefaultDict(() => ({ items: [], bpm: 0, rhythm: '', is_volatile: false }));

  for (const [tx, cluster] of pairs) {
    const key = `${tx.rhythm}@@${pyStr(tx.is_volatile)}@@${pyStr(py.get(cluster, 'bpm', 0))}`;
    const g = groups.getOrCreate(key);
    g.items.push(tx);
    g.rhythm = tx.rhythm;
    g.is_volatile = tx.is_volatile;
    g.bpm = py.get(cluster, 'bpm', 0);
  }

  let profiles = [];
  for (const g of groups.values()) {
    const starts_ends = py.sorted(g.items.map((it) => [it.start_ms, it.end_ms]));
    let total_duration = 0;
    if (starts_ends.length) {
      let [cur_start, cur_end] = starts_ends[0];
      for (let i = 1; i < starts_ends.length; i++) {
        const [s, e] = starts_ends[i];
        if (cur_end < e) {
          total_duration += cur_end - cur_start;
          cur_start = s;
          cur_end = e;
        } else {
          cur_end = py.max2(cur_end, e);
        }
      }
      total_duration += cur_end - cur_start;
    }

    const n_total = g.items.length;
    const sub_counter = new py.DefaultDict(() => 0);
    for (const it of g.items) {
      if (py.truthy(it.sub_texture)) sub_counter.set(it.sub_texture, sub_counter.getOrCreate(it.sub_texture) + 1);
    }
    const sub_textures = py.sortedBy([...sub_counter.entries()], (x) => x[1], true)
      .map(([name, cnt]) => [name, cnt / n_total]);

    const dominant_sub = sub_textures.length ? sub_textures[0][0] : null;
    const weight = py.get(_PRIMARY_RHYTHM_WEIGHTS, g.rhythm, 1.0);

    profiles.push({
      rhythm: g.rhythm,
      sub_textures,
      dominant_sub,
      weight,
      bpm: g.bpm,
      is_volatile: g.is_volatile,
      coverage_ms: total_duration,
      importance: total_duration * weight * py.max2(1, g.bpm),
    });
  }

  profiles = py.sortedBy(profiles, (p) => p.importance, true);
  return profiles;
}

// ── Public API ─────────────────────────────────────────────────────

/** Full pipeline: notes → beat frames → textures → bundles → profiles. */
export function profile_rhythms(notes, key_count, bpm_timeline = null, total_ms = 0) {
  if (!py.truthy(bpm_timeline)) bpm_timeline = [];
  const frames = _build_beat_frames(notes, key_count, bpm_timeline);
  if (!frames.length) return [];

  // total_ms or (notes[-1]['time'] - notes[0]['time']) if notes else 60000
  let true_total;
  if (!py.truthy(notes)) true_total = 60000;
  else if (py.truthy(total_ms)) true_total = total_ms;
  else true_total = py.getitem(notes[notes.length - 1], 'time') - py.getitem(notes[0], 'time');
  const textures = _extract_textures(frames, key_count, true_total);
  const paired = _bundle_textures(textures);
  const profiles = _profile_bundles(paired);

  // Drop profiles dominated by a same-rhythm profile with more than twice
  // the coverage and a higher BPM.
  const filtered = [];
  for (const p of profiles) {
    let dominated = false;
    for (const other of profiles) {
      if (other.rhythm === p.rhythm && other.coverage_ms * 0.5 > p.coverage_ms && other.bpm > p.bpm) {
        dominated = true;
        break;
      }
    }
    if (!dominated) filtered.push(p);
  }

  // Keep at most 3 profiles per rhythm.
  let capped = [];
  const rhythm_counts = new py.DefaultDict(() => 0);
  for (const p of filtered) {
    if (rhythm_counts.getOrCreate(p.rhythm) < 3) {
      capped.push(p);
      rhythm_counts.set(p.rhythm, rhythm_counts.get(p.rhythm) + 1);
    }
  }
  capped = py.sortedBy(capped, (p) => p.importance, true);
  return capped;
}

/**
 * Map rhythm profiles to DanOverlay family taxonomy.
 *
 * Returns the same dict shape as classifier.classify_family():
 *     {family, confidence, scores, subtype, subtype_scores, reason}
 */
export function classify_texture_profile(profiles, key_count, bpm = 0.0, drain_s = 0.0) {
  if (!py.truthy(profiles)) {
    return {
      family: 'hybrid',
      confidence: 0.0,
      scores: { jack: 0, speed: 0, stamina: 0, tech: 0, stream: 0 },
      subtype: 'generic',
      subtype_scores: {},
      reason: 'no patterns detected',
    };
  }

  const top_importance = profiles[0].importance;
  const important = profiles.filter(
    (p) => p.importance / py.max2(top_importance, 1) > _PATTERN_CONFIG.important_cluster_ratio,
  );

  const primary = important.length ? important[0] : profiles[0];
  const secondary = important.length > 1 ? important[1] : null;

  let primary_family = py.get(_TEXTURE_TO_FAMILY, primary.rhythm, 'hybrid');
  if (py.truthy(primary.dominant_sub) && has(_SUB_TEXTURE_FAMILY, primary.dominant_sub)) {
    primary_family = _SUB_TEXTURE_FAMILY[primary.dominant_sub];
  }

  let is_hybrid = false;
  if (py.truthy(secondary) && py.truthy(secondary.dominant_sub)) {
    const sec_family = py.get(_SUB_TEXTURE_FAMILY, secondary.dominant_sub,
      py.get(_TEXTURE_TO_FAMILY, secondary.rhythm, 'hybrid'));
    if (sec_family !== primary_family) is_hybrid = true;
  }

  let is_tech = py.truthy(primary.is_volatile) || (is_hybrid && primary.bpm < 150);
  if (['trills', 'mini_trills', 'split_trill', 'jump_trill', 'chord_roll', 'brackets', 'column_lock', 'shield', 'inverse']
    .includes(primary.dominant_sub)) {
    is_tech = true;
  }

  let family;
  if (is_hybrid && is_tech) family = 'tech';
  else if (is_hybrid) family = 'hybrid';
  else if (is_tech) family = 'tech';
  else family = primary_family;

  // Fast (> 200 BPM), long (> 100 s drain) speed/stream charts are rated as stamina.
  if ((family === 'speed' || family === 'stream') && bpm > 200.0 && drain_s > 100.0) {
    family = 'stamina';
    if (primary_family === 'speed' || primary_family === 'stream') primary_family = 'stamina';
  }

  const total_importance = py.sum(profiles.slice(0, 5).map((p) => p.importance)) + 1;
  const confidence = py.min2(1.0, py.round(primary.importance / total_importance, 2));

  let scores = { jack: 0, speed: 0, stamina: 0, tech: 0, stream: 0 };
  for (const p of important.slice(0, 5)) {
    let fam = py.get(_TEXTURE_TO_FAMILY, p.rhythm, 'hybrid');
    if (py.truthy(p.dominant_sub) && has(_SUB_TEXTURE_FAMILY, p.dominant_sub)) fam = _SUB_TEXTURE_FAMILY[p.dominant_sub];
    scores[fam] = py.getitem(scores, fam) + (p.importance / py.max2(total_importance, 1)) * 100;
  }

  const max_score = or(py.max(Object.values(scores)), 1);
  const rescaled = {};
  for (const [k, v] of Object.entries(scores)) rescaled[k] = py.round((v / max_score) * 100, 1);
  scores = rescaled;

  const subtype_scores = {};
  for (const p of important.slice(0, 5)) {
    for (const [sub_name, ratio] of py.get(p, 'sub_textures', [])) {
      if (!has(subtype_scores, sub_name)) subtype_scores[sub_name] = 0;
      subtype_scores[sub_name] += ratio * 100;
    }
  }

  const subtype = or(primary.dominant_sub, 'generic');
  const reason = `primary rhythm: ${primary.rhythm} (${pyStr(py.get(primary, 'dominant_sub', 'none'))}), bpm: ${pyStr(primary.bpm)}`;

  return {
    family,
    confidence,
    scores,
    subtype,
    subtype_scores,
    reason,
  };
}

// parser.py note types → beat-frame kinds.
export const _NOTE_KIND_MAP = {
  normal: 'tap',
  hold_head: 'hold_head',
  hold_body: 'hold_body',
  hold_tail: 'hold_tail',
};

/** Python list item assignment `lst[i] = v` (negative indices, IndexError). */
function setItem(lst, i, v) {
  const k = i < 0 ? lst.length + i : i;
  if (k < 0 || k >= lst.length) throw new py.IndexError('list assignment index out of range');
  lst[k] = v;
}

/**
 * Thin wrapper: parse parser.py output → rhythm profiles → family dict.
 *
 * Accepts the dict returned by parsear_osu_v2() and returns the same
 * {family, confidence, scores, ...} shape as classifier.classify_family().
 */
export function classify_from_parsed(parsed) {
  const raw_notes = py.get(parsed, 'notes', []);
  const key_count = py.get(parsed, 'key_count', 4);

  const bpm_timeline = [];
  for (const tp of py.get(parsed, 'timing_points', [])) {
    bpm_timeline.push({
      time: py.float(or(py.get(tp, 'time', 0), 0)),
      ms_per_beat: py.float(or(py.get(tp, 'ms_per_beat', 500), 500)),
    });
  }

  if (!py.truthy(raw_notes)) {
    return {
      family: 'hybrid',
      confidence: 0.0,
      scores: { jack: 0, speed: 0, stamina: 0, tech: 0, stream: 0 },
      subtype: 'generic',
      subtype_scores: {},
      reason: 'no notes',
    };
  }

  // Notes may be (time, column, type) tuples or dicts.
  const first_time = py.float(Array.isArray(raw_notes[0]) ? py.at(raw_notes[0], 0) : or(py.get(raw_notes[0], 'time', 0), 0));
  let last_time = first_time;

  const time_map = new Map();
  for (const n of raw_notes) {
    let t;
    let col;
    let kind;
    if (Array.isArray(n)) {
      t = py.float(or(py.at(n, 0), 0));
      col = n.length >= 2 ? py.int(or(n[1], 0)) : 0;
      kind = n.length >= 3 ? or(n[2], 'tap') : 'tap';
    } else {
      t = py.float(or(py.get(n, 'time', 0), 0));
      col = py.int(or(py.get(n, 'column', 0), 0));
      kind = or(py.get(n, 'type', 'normal'), 'normal');
    }
    // str(kind): only the exact strings of _NOTE_KIND_MAP map to something other than 'tap'
    const mapped_kind = typeof kind === 'string' ? py.get(_NOTE_KIND_MAP, kind, 'tap') : 'tap';

    if (!time_map.has(t)) time_map.set(t, new Array(Math.max(0, key_count)).fill(null));
    if (col < key_count) setItem(time_map.get(t), col, mapped_kind);
    last_time = py.max2(last_time, t);
  }

  const beat_notes = py.sorted(time_map.keys()).map((t) => ({ time: t, data: time_map.get(t) }));

  const total_ms = or(last_time - first_time, 60000);

  const profiles = profile_rhythms(beat_notes, key_count, bpm_timeline, total_ms);
  const _bpm = py.float(or(py.get(parsed, 'bpm', 0.0), 0.0));
  const _drain = py.float(or(py.get(parsed, 'drain_time_s', 0.0), 0.0));
  return classify_texture_profile(profiles, key_count, _bpm, _drain);
}
