// rank_engine.js — SR → DP rulers, dan labels / sublevels and final rank computation.
//
// JavaScript port of engine/dan_engine/rank_engine.py (DanOverlay 3.1.0 by 8doul).
//
// Porting note: Python loads config/sr_means.json at import time. Here the virtual file
// system is only filled by setupEngine(), so the means are loaded lazily on first use
// (_ensure_sr_means) and cached — same values, same results.
import * as py from './py.js';
import * as vfs from './vfs.js';
import { resource_path } from './resource_path.js';

export const _DAN_ORDER = [
  '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
  'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa',
];

export const _DAN_DISPLAY = {
  '1st': '1st Dan', '2nd': '2nd Dan', '3rd': '3rd Dan',
  '4th': '4th Dan', '5th': '5th Dan', '6th': '6th Dan',
  '7th': '7th Dan', '8th': '8th Dan', '9th': '9th Dan',
  '10th': '10th Dan',
  Alpha: 'Alpha', Beta: 'Beta', Gamma: 'Gamma',
  Delta: 'Delta', Epsilon: 'Epsilon',
  Zeta: 'Zeta', Eta: 'Eta', Theta: 'Theta',
  Iota: 'Iota', Kappa: 'Kappa',
};

/** Python max(a, b) / min(a, b): the first argument wins ties and NaN comparisons. */
const pymax = (a, b) => (b > a ? b : a);
const pymin = (a, b) => (b < a ? b : a);
/** Python `x or d`. */
const orElse = (x, d) => (py.truthy(x) ? x : d);
/** float(d.get(k, dflt) or dflt) */
const getFloat = (d, k, dflt) => py.float(orElse(py.get(d, k, dflt), dflt));
/** dict.get for hashable keys that may be None (a JS object would look up the key "null"). */
const dget = (d, k, dflt = null) => (k === null || k === undefined ? dflt : py.get(d, k, dflt));
/**
 * A numeric operand as Python sees it: bool is an int (True == 1); None/str/... meeting a
 * float in arithmetic or ordering raise TypeError.
 */
function asNumber(x, what) {
  if (typeof x === 'number') return x;
  if (typeof x === 'boolean') return x ? 1 : 0;
  throw new py.TypeError_(`unsupported operand type(s) for ${what}: '${py.typeName(x)}' and 'float'`);
}

/**
 * Load SR means from config/sr_means.json.
 *
 * Returns [general_means, skillset_means] where general_means maps the 20 dans to SR
 * and skillset_means maps 4 families × 20 dans.
 */
export function _load_sr_means() {
  const path = vfs.path.join(resource_path('config'), 'sr_means.json');
  const data = JSON.parse(vfs.readText(path));
  return [py.getitem(data, 'general'), py.getitem(data, 'skillsets')];
}

// Module-level state of the Python module (filled on first use, see header note).
export let _GENERAL_SR_MEANS = null;
export let _SKILLSET_SR_MEANS = null;

export function _ensure_sr_means() {
  if (_GENERAL_SR_MEANS === null) {
    [_GENERAL_SR_MEANS, _SKILLSET_SR_MEANS] = _load_sr_means();
  }
}

export const _SUBLEVEL_LABELS = [
  [0.0, 0.2, 'Low'],
  [0.2, 0.4, 'Mid-Low'],
  [0.4, 0.6, 'Mid'],
  [0.6, 0.8, 'Mid-High'],
  [0.8, 1.0, 'High'],
];

export const _LN_CONFIDENCE_MULT = {
  high: 1.0,
  degraded: 0.75,
  gray: 0.45,
  out_of_domain: 0.15,
};

export const _ruler_cache = new Map();

/**
 * Get the SR->DP ruler for a given skillset.
 *
 * Returns a list of [sr_mean, dp_int] sorted by dp. Uses per-skillset means for
 * Alpha-Epsilon when available, falls back to General means for all other dans.
 */
export function _get_ruler(skillset = null) {
  const cache_key = orElse(skillset, '__general__');
  if (_ruler_cache.has(cache_key)) return _ruler_cache.get(cache_key);

  _ensure_sr_means();
  const overrides = py.truthy(skillset) ? dget(_SKILLSET_SR_MEANS, skillset, {}) : {};

  const table = [];
  _DAN_ORDER.forEach((dan_name, i) => {
    const dp = i + 1;
    const sr = py.get(overrides, dan_name, py.getitem(_GENERAL_SR_MEANS, dan_name));
    table.push([sr, dp]);
  });

  for (let i = 1; i < table.length; i++) {
    if (table[i][0] <= table[i - 1][0]) {
      table[i] = [table[i - 1][0] + 0.01, table[i][1]];
    }
  }

  _ruler_cache.set(cache_key, table);
  return table;
}

/**
 * Compute midpoint boundaries between adjacent dan means.
 *
 * Each dan occupies the SR range from the midpoint to its lower neighbor to the
 * midpoint to its upper neighbor. Returns a list of [lower_sr, upper_sr, dp_int].
 */
export function _precompute_boundaries(ruler) {
  const means = ruler.map(([sr]) => sr);
  const n = means.length;
  const boundaries = [];

  for (let i = 0; i < n; i++) {
    let lower;
    let upper;
    if (i > 0) {
      lower = (means[i - 1] + means[i]) / 2.0;
    } else {
      const gap = n > 1 ? means[1] - means[0] : 1.0;
      lower = means[0] - gap / 2.0;
    }

    if (i < n - 1) {
      upper = (means[i] + means[i + 1]) / 2.0;
    } else {
      const gap = n > 1 ? py.at(means, -1) - py.at(means, -2) : 1.0;
      upper = py.at(means, -1) + gap / 2.0;
    }

    boundaries.push([lower, upper, ruler[i][1]]);
  }

  return boundaries;
}

/**
 * Convert SR to DP using per-skillset ruler + linear interpolation.
 *
 * Within each dan's boundary zone the fractional part maps linearly from 0.0 (lower
 * boundary) to 1.0 (upper boundary). skillset: "jack", "speed", "stamina", "tech" or
 * null for the general ruler. Returns DP on the 1-20 scale, clamped to [0.5, 20.99].
 */
export function sr_to_dp(sr, skillset = null) {
  const ruler = _get_ruler(skillset);
  const boundaries = _precompute_boundaries(ruler);
  sr = asNumber(sr, '<');

  if (sr < boundaries[0][0]) {
    return pymax(0.5, py.float(boundaries[0][2]));
  }

  const last = py.at(boundaries, -1);
  if (sr >= last[1]) {
    return pymin(20.99, py.float(last[2]) + 0.99);
  }

  for (const [lower, upper, dp_int] of boundaries) {
    if (lower <= sr && sr < upper) {
      const t = (sr - lower) / pymax(upper - lower, 1e-6);
      return py.float(dp_int) + t;
    }
  }
  return py.float(last[2]);
}

/** Dan label from DP value. Returns [display_label, short_label]. */
export function dp_to_label(dp) {
  const idx = pymax(0, pymin(_DAN_ORDER.length - 1, py.int(dp) - 1));
  const short = _DAN_ORDER[idx];
  return [py.get(_DAN_DISPLAY, short, short), short];
}

/** Sublevel string (Low, Mid-Low, Mid, Mid-High, High) from DP fraction. */
export function dp_to_sublevel(dp) {
  dp = asNumber(dp, '>=');
  const frac = dp >= 1 ? py.round(dp - py.math.floor(dp), 2) : 0.0;

  for (const [lo, hi, label] of _SUBLEVEL_LABELS) {
    if (lo <= frac && frac <= hi) return label;
  }
  return 'High';
}

/**
 * Compensate peak-heavy jack charts that raw Sunny SR compresses.
 *
 * A small ranking-only bonus when the chart is already classified as jack and its
 * Jbar peak exceeds the measured Epsilon ceiling by a meaningful margin.
 */
export function _jack_peak_bonus(sr_result, family) {
  if (family !== 'jack') return 0.0;

  const total_notes = getFloat(sr_result, 'total_notes_eff', 0);
  if (total_notes > 15000) return 0.0;

  const jbar_max = getFloat(sr_result, 'jbar_max', 0.0);
  const sr_val = getFloat(sr_result, 'sr', 0.0);

  let peak_start;
  let peak_scale;
  let peak_cap;
  if (sr_val >= 11.0) {
    peak_start = 64.5;
    peak_scale = 0.031;
    peak_cap = sr_val < 12.0 ? 0.35 : 0.20;
  } else if (sr_val >= 9.0) {
    peak_start = 58.0;
    peak_scale = 0.02;
    peak_cap = 0.25;
  } else if (sr_val >= 7.0) {
    peak_start = 50.0;
    peak_scale = 0.02;
    peak_cap = 0.30;
  } else {
    peak_start = 48.0;
    peak_scale = 0.015;
    peak_cap = 0.20;
  }

  return pymin(peak_cap, pymax(0.0, jbar_max - peak_start) * peak_scale);
}

/**
 * Derive chart family from MinaCalc skillset MSD values.
 *
 * Compresses the 7 raw skillsets into the same 4 role scores used by the MinaCalc
 * estimator and returns the dominant family, or null when the signal is too
 * ambiguous to override the Sunny-based classifier.
 */
export function _msd_to_family(msd_dict, sunny_family = null) {
  if (!py.truthy(msd_dict)) return null;

  let groups = {
    jack: pymax(
      getFloat(msd_dict, 'jackspeed', 0.0),
      getFloat(msd_dict, 'chordjack', 0.0),
    ),
    speed: pymax(
      getFloat(msd_dict, 'stream', 0.0),
      getFloat(msd_dict, 'jumpstream', 0.0),
    ),
    stamina: (
      0.7 * getFloat(msd_dict, 'stamina', 0.0)
      + 0.3 * getFloat(msd_dict, 'handstream', 0.0)
    ),
    tech: getFloat(msd_dict, 'technical', 0.0),
  };
  const kept = {};
  for (const [fam, val] of Object.entries(groups)) if (val > 0.0) kept[fam] = val;
  groups = kept;

  if (!py.truthy(groups)) return null;

  const total = py.sum(Object.values(groups));
  if (total < 1.0) return null;

  const ranked = py.sortedBy(Object.entries(groups), (kv) => kv[1], true);
  const [top_fam, top_val] = ranked[0];
  const second_fam = ranked.length > 1 ? ranked[1][0] : null; // eslint-disable-line no-unused-vars
  const second_val = ranked.length > 1 ? ranked[1][1] : 0.0;

  const dominance = top_val / total;
  const margin_ratio = (top_val - second_val) / top_val;
  const sunny_role = sunny_family === 'stream' ? 'speed' : sunny_family;
  const sunny_val = py.float(orElse(dget(groups, sunny_role, 0.0), 0.0));

  const sunnyNone = sunny_family === null || sunny_family === undefined;

  if (dominance < 0.26) return null;

  if ((sunnyNone || sunny_family === '' || sunny_family === 'hybrid') && (dominance >= 0.30 || margin_ratio >= 0.04)) {
    return top_fam;
  }

  if ((sunny_family === 'jack' || sunny_family === 'stamina')
      && (top_fam === 'jack' || top_fam === 'stamina') && margin_ratio >= 0.02) {
    return top_fam;
  }

  if (sunny_family === 'stream' && top_fam === 'speed' && margin_ratio >= 0.01) {
    return top_fam;
  }

  if (top_fam === 'tech' && !(sunnyNone || sunny_family === '' || sunny_family === 'hybrid' || sunny_family === 'tech')
      && sunny_val > 0.0 && top_val >= sunny_val * 1.2) {
    return top_fam;
  }

  if (sunny_family === 'tech' && top_fam === 'jack' && margin_ratio >= 0.02) {
    return top_fam;
  }

  if (dominance >= 0.34 && margin_ratio >= 0.08) {
    return top_fam;
  }

  return null;
}

/**
 * Recompute DP and labels using a given family (for MSD-based overrides).
 *
 * sr: raw SR from the primary engine (before any bonus); sr_result_for_bonus: sr_result
 * dict (needs jbar_max for the jack peak bonus); family: overriding family name.
 * Returns { dp, dan_label, dan_short, sublevel, beyond, family, ranking_sr }.
 */
export function recompute_dp_for_family(sr, sr_result_for_bonus, family) {
  const _F2S = {
    jack: 'jack', speed: 'speed',
    stamina: 'stamina', tech: 'tech',
    stream: 'speed', hybrid: null,
  };
  const skillset = dget(_F2S, family, null);
  const jack_bonus = _jack_peak_bonus(orElse(sr_result_for_bonus, {}), family);
  sr = asNumber(sr, '+');
  const ranking_sr = sr + jack_bonus;
  const dp = py.round(sr_to_dp(ranking_sr, skillset), 2);
  const [label, short] = dp_to_label(dp);
  const sublevel = dp_to_sublevel(dp);
  return {
    dp,
    dan_label: label,
    dan_short: short,
    sublevel,
    beyond: dp > 20.5,
    family,
    ranking_sr: py.round(ranking_sr, 4),
  };
}

/**
 * Compute final ranking: SR -> DP -> label.
 *
 * Uses per-skillset SR rulers when the family classifier identifies a dominant
 * skillset, otherwise falls back to the general ruler. msd is reserved (unused).
 * Returns { dp, dan_label, dan_short, sublevel, beyond, confidence, sr, family,
 * corrections, debug }.
 */
export function compute_rank(sr_result, features, classification, domain_info, msd = null) { // eslint-disable-line no-unused-vars
  const sr = getFloat(sr_result, 'sr', 0.0);
  const family = py.get(classification, 'family', 'hybrid');
  const family_confidence = py.float(py.get(classification, 'confidence', 0.0));

  const _FAMILY_TO_SKILLSET = {
    jack: 'jack',
    speed: 'speed',
    stamina: 'stamina',
    tech: 'tech',
    stream: 'speed',
    hybrid: null,
  };

  let skillset;
  if (family_confidence >= 0.5) {
    skillset = dget(_FAMILY_TO_SKILLSET, family, null);
  } else {
    skillset = null;
  }

  const _SJ_JBAR_FLOOR = 70.0;
  const _SJ_SHARE_FLOOR = 0.55;
  const _sj_jbar = getFloat(sr_result, 'jbar_max', 0.0);
  const _sj_share = getFloat(sr_result, 'jbar_share', 0.0);
  const _sj_pbar = getFloat(sr_result, 'pbar_max', 0.0);
  const _pbar_taper = _sj_pbar > 55.0 ? pymax(0.0, pymin(1.0, (65.0 - _sj_pbar) / 10.0)) : 1.0;
  const _sr_taper = sr > 10.5 ? pymax(0.0, pymin(1.0, (11.5 - sr) / 1.0)) : 1.0;
  const _sj_taper = _pbar_taper * _sr_taper;
  const _is_speedjack = _sj_jbar >= _SJ_JBAR_FLOOR && _sj_share >= _SJ_SHARE_FLOOR && _sj_taper > 0.0;

  const corrections = [];
  let jack_bonus = 0.0;
  let stamina_bonus = 0.0;
  let ranking_sr;
  let dp;

  if (_is_speedjack) {
    jack_bonus = _jack_peak_bonus(sr_result, 'jack');
    const _sj_extra = pymin(0.65, pymax(0.0, _sj_jbar - 72.0) * 0.06) * _sj_taper;
    const _sj_total = jack_bonus + _sj_extra;
    ranking_sr = sr + _sj_total;
    dp = sr_to_dp(ranking_sr, 'jack');
    skillset = 'jack';
    corrections.push(`speedjack_rescue:+${py.reprFloat(py.round(_sj_total, 3))}`);
  } else {
    jack_bonus = _jack_peak_bonus(sr_result, family);

    const di = orElse(domain_info, {});
    const bpm = py.float(orElse(py.get(di, 'bpm', py.get(orElse(features, {}), 'bpm', 0.0)), 0.0));
    const drain_s = getFloat(di, 'drain_time_s', 0.0);

    if (family === 'stamina' && bpm >= 240.0 && drain_s >= 120.0) {
      const _sp = getFloat(features, 'stream_purity', 0.0);
      const _jr = getFloat(features, 'jump_ratio', 0.0);
      const _hr = getFloat(features, 'hand_ratio', 0.0);
      const _qr = getFloat(features, 'quad_ratio', 0.0);
      const _chord = _jr + _hr + _qr;
      const _dcv = getFloat(features, 'density_cv', 0.0);

      if (_sp >= 0.4 && _chord <= 0.4 && _dcv <= 0.3) {
        stamina_bonus = pymin(0.2, (bpm - 240.0) * 0.005);
        const _dp_no = sr_to_dp(sr + jack_bonus, skillset);
        for (let k = 0; k < 3; k++) {
          const _dp_yes = sr_to_dp(sr + jack_bonus + stamina_bonus, skillset);
          if (py.int(_dp_yes) - py.int(_dp_no) >= 2) {
            stamina_bonus = py.round(stamina_bonus * 0.5, 4);
          } else {
            break;
          }
        }
      }
    }

    ranking_sr = sr + jack_bonus + stamina_bonus;
    if (jack_bonus > 0.0) corrections.push(`jack_peak_bonus:+${py.format(jack_bonus, '.3f')}`);
    if (stamina_bonus > 0.0) corrections.push(`stamina_rescue:+${py.format(stamina_bonus, '.3f')}`);

    dp = sr_to_dp(ranking_sr, skillset);
  }

  const di = orElse(domain_info, {});
  const note_count = py.int(py.get(di, 'note_count', 0));
  const note_confidence = pymin(1.0, note_count / 200.0);
  const ln_confidence = py.get(di, 'ln_confidence', 'high');
  const ln_mult = dget(_LN_CONFIDENCE_MULT, ln_confidence, 1.0);

  let overall_confidence = family_confidence * ln_mult * note_confidence;
  overall_confidence = py.round(pymin(1.0, overall_confidence), 3);

  dp = py.round(dp, 2);
  const [dan_label, dan_short] = dp_to_label(dp);
  const sublevel = dp_to_sublevel(dp);

  const debug = {
    sr: py.round(sr, 4),
    ranking_sr: py.round(ranking_sr, 4),
    jack_peak_bonus: py.round(jack_bonus, 4),
    stamina_rescue_bonus: py.round(stamina_bonus, 4),
    skillset_ruler: orElse(skillset, 'general'),
    family,
    family_confidence: py.round(family_confidence, 3),
    ln_confidence,
    ln_multiplier: ln_mult,
    note_confidence: py.round(note_confidence, 3),
    formula_version: 3,
  };

  return {
    dp,
    dan_label,
    dan_short,
    sublevel,
    beyond: dp > 20.5,
    confidence: overall_confidence,
    sr: py.round(sr, 4),
    family,
    corrections,
    debug,
  };
}

/**
 * Reload SR means from JSON and clear ruler cache.
 *
 * Call this after re-running calibration to pick up new sr_means.json values.
 */
export function reload_config() {
  [_GENERAL_SR_MEANS, _SKILLSET_SR_MEANS] = _load_sr_means();
  _ruler_cache.clear();
}
