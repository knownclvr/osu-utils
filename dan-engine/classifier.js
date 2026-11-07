// classifier.js — chart family classification (hybrid thresholds, tech subtypes).
//
// JavaScript port of engine/dan_engine/classifier.py (DanOverlay 3.1.0 by 8doul).
import * as py from './py.js';
import * as vfs from './vfs.js';
import { resource_path } from './resource_path.js';

export const _CONFIG_DIR = resource_path('config');

export const _HYBRID_STREAM_THRESHOLD = 0.85;
export const _HYBRID_JACK_THRESHOLD = 0.10;
export const _HYBRID_CV_THRESHOLD = 0.30;
export const _HYBRID_XBAR_THRESHOLD = 0.30;
export const _HYBRID_COEFF = 0.025;
export const _HYBRID_MIN = 0.97;
export const _HYBRID_MAX = 1.10;

/** Python max(a, b) / min(a, b): the first argument wins ties and NaN comparisons. */
const pymax = (a, b) => (b > a ? b : a);
const pymin = (a, b) => (b < a ? b : a);
/** Python `x or d`. */
const orElse = (x, d) => (py.truthy(x) ? x : d);
/** float(d.get(k, dflt) or dflt) */
const getFloat = (d, k, dflt) => py.float(orElse(py.get(d, k, dflt), dflt));

/** Reads config/family_profiles.json (re-read on every call, like the Python module). */
export function _load_family_profiles() {
  const path = vfs.path.join(_CONFIG_DIR, 'family_profiles.json');
  return JSON.parse(vfs.readText(path));
}

export function _clamp(value, low, high) {
  return pymax(low, pymin(high, value));
}

/**
 * Small multiplicative bonus for mixed-tech patterns.
 *
 * Kept local so family classification no longer depends on the old legacy helper module.
 */
export function _tech_hybrid_factor(stream_purity, jack_density, density_cv, xbar_max_norm) {
  const stream_excess = pymax(0.0, stream_purity - _HYBRID_STREAM_THRESHOLD);
  const jack_excess = pymax(0.0, jack_density - _HYBRID_JACK_THRESHOLD);
  const cv_signal = pymax(0.0, density_cv - _HYBRID_CV_THRESHOLD);
  const xbar_signal = pymax(0.0, xbar_max_norm - _HYBRID_XBAR_THRESHOLD);

  const hybrid_signal = (
    stream_excess * 1.5
    + jack_excess * 2.0
    + cv_signal * 1.2
    + xbar_signal * 1.0
  );

  const factor = 1.0 + _HYBRID_COEFF * hybrid_signal;
  return pymax(_HYBRID_MIN, pymin(_HYBRID_MAX, factor));
}

/**
 * Recover a lightweight tech subtype split inspired by the legacy project.
 *
 * The old project differentiated chaos/control/hybrid tech. We do not have the full
 * legacy strain skillsets here, so this uses current structural proxies.
 */
export function _classify_tech_subtype(features, sunny_result, profiles_cfg) {
  const subtype_cfg = py.get(orElse(profiles_cfg, {}), 'tech_subtypes', {});

  const transition_var = getFloat(features, 'transition_var', 0.0);
  const density_cv = getFloat(features, 'density_cv', 0.0);
  const jump_ratio = getFloat(features, 'jump_ratio', 0.0);
  const hand_ratio = getFloat(features, 'hand_ratio', 0.0);
  const anchor_ratio = getFloat(features, 'anchor_ratio', 0.0);
  const chord_complexity = getFloat(features, 'chord_complexity', 0.0);
  const nps_active_cv = getFloat(features, 'nps_active_cv', 0.0);
  const stream_purity = getFloat(features, 'stream_purity', 0.0);

  const jbar_max = getFloat(sunny_result, 'jbar_max', 0.0);
  const pbar_max = getFloat(sunny_result, 'pbar_max', 0.0);
  const xbar_max = getFloat(sunny_result, 'xbar_max', 0.0);
  const bar_total = jbar_max + pbar_max + 1.0;
  const tech_dom = (xbar_max + bar_total) > 0 ? xbar_max / (xbar_max + bar_total) : 0.0;

  const chaos_floor = py.float(py.get(subtype_cfg, 'chaos_transition_floor', 0.38));
  const control_hand_floor = py.float(py.get(subtype_cfg, 'control_hand_floor', 0.10));
  const control_chord_floor = py.float(py.get(subtype_cfg, 'control_chord_floor', 0.18));

  const chaos_score = (
    transition_var * 42.0
    + density_cv * 26.0
    + nps_active_cv * 20.0
    + tech_dom * 18.0
    - anchor_ratio * 10.0
  );
  const control_score = (
    hand_ratio * 45.0
    + jump_ratio * 10.0
    + chord_complexity * 12.0
    + tech_dom * 15.0
    + pymax(0.0, 1.0 - density_cv) * 8.0
    + anchor_ratio * 6.0
  );
  const hybrid_score = (
    tech_dom * 22.0
    + jump_ratio * 14.0
    + density_cv * 10.0
    + transition_var * 10.0
    + pymax(0.0, 0.9 - Math.abs(stream_purity - 0.45)) * 5.0
  );

  const ranked = py.sortedBy(
    Object.entries({
      chaos_tech: chaos_score,
      control_tech: control_score,
      hybrid_tech: hybrid_score,
    }),
    (kv) => kv[1],
    true,
  );

  let subtype = ranked[0][0];
  const top_score = ranked[0][1];
  const second_score = ranked[1][1];
  const total = pymax(py.sum(ranked.map(([, score]) => score)), 1e-9);
  let confidence = _clamp((top_score - second_score) / pymax(total * 0.18, 1.0), 0.15, 1.0);

  if (subtype === 'chaos_tech' && transition_var < chaos_floor) {
    subtype = 'hybrid_tech';
    confidence = pymin(confidence, 0.45);
  } else if (subtype === 'control_tech' && hand_ratio < control_hand_floor && chord_complexity < control_chord_floor) {
    subtype = 'hybrid_tech';
    confidence = pymin(confidence, 0.45);
  }

  const scores = {};
  for (const [name, score] of ranked) scores[name] = py.round(score, 2);
  return {
    subtype,
    confidence: py.round(confidence, 3),
    scores,
  };
}

/**
 * Classifies the chart family based on Sunny internals + features.
 *
 * sunny_result: output from sunny_analyze() — sr, jack_ratio, jbar_max, pbar_max, etc.
 * features: extracted features — chord density, jack_density, stream_purity, density_cv, ...
 * domain_info: output from validate_domain() — bpm, drain_time_s, etc. (optional)
 *
 * Returns { family, confidence, scores, subtype, subtype_confidence, subtype_scores, reason }.
 */
export function classify_family(sunny_result, features, domain_info = null) {
  const profiles_cfg = _load_family_profiles();
  const get_s = (k, d = 0.0) => py.float(orElse(py.get(sunny_result, k, d), d));
  const get_f = (k, d = 0.0) => py.float(orElse(py.get(features, k, d), d));
  const get_d = (k, d = 0.0) => py.float(orElse(py.get(orElse(domain_info, {}), k, d), d));

  const bpm = get_d('bpm', get_f('bpm', 0.0));
  const drain_s = get_d('drain_time_s', get_f('duration_s', 0.0));

  const jack_ratio = get_s('jack_ratio'); // eslint-disable-line no-unused-vars
  const jbar_max = get_s('jbar_max');
  const pbar_max = get_s('pbar_max');
  const xbar_max = get_s('xbar_max');
  const abar_mean = get_s('abar_mean'); // eslint-disable-line no-unused-vars
  const d93 = get_s('d93');
  const d83 = get_s('d83');
  const d_weighted_mean = get_s('d_weighted_mean'); // eslint-disable-line no-unused-vars
  const sr = get_s('sr'); // eslint-disable-line no-unused-vars

  const bar_total = jbar_max + pbar_max + 1.0;
  const jack_dom = py.div(jbar_max, bar_total);
  const stream_dom = py.div(pbar_max, bar_total);
  const tech_dom = (xbar_max + bar_total) > 0 ? xbar_max / (xbar_max + bar_total) : 0.0;
  const peak_ratio = d83 > 0.1 ? d93 / pymax(d83, 0.01) : 1.0;

  const stream_purity = get_f('stream_purity');
  const jack_ratio_broad = get_f('jack_ratio', 0.0);
  const jack_density = get_f('jack_density', jack_ratio_broad);
  const vibro_density = get_f('vibro_density', get_f('vibro_ratio', 0.0));
  const density_cv = pymin(2.0, get_f('density_cv'));
  const transition_var = get_f('transition_var');
  const jump_ratio = get_f('jump_ratio');
  const hand_ratio = get_f('hand_ratio');
  const quad_ratio = get_f('quad_ratio');

  const nps_p90 = get_f('nps_p90'); // eslint-disable-line no-unused-vars
  const nps_sustained = get_f('nps_sustained_top30');
  const nps_active_ratio = get_f('nps_active_ratio');
  const nps_active_cv = get_f('nps_active_cv');

  const chord_fraction = jump_ratio + hand_ratio + quad_ratio;
  const anchor_ratio = get_f('anchor_ratio', 0.0);
  const pattern_irregularity = get_f('pattern_irregularity', 0.0);
  const timing_irregularity = get_f('timing_irregularity', 0.0);

  const rep_margin = pymax(0.0, 0.83 - transition_var);
  const is_repetitive = pymin(1.0, rep_margin * 12.0);

  let jack_baseline;
  if (bpm > 0) {
    const quarter_ms = 60000.0 / bpm;
    jack_baseline = pymin(0.9, pymax(0.0, 1.0 - quarter_ms / 250.0));
  } else {
    jack_baseline = 0.3;
  }
  const jack_excess = pymax(0.0, jack_ratio_broad - jack_baseline);

  const bpm_signal = pymin(1.5, pymax(0.0, bpm - 140.0) / 80.0);

  const jack_score = (
    is_repetitive * 50.0
    + jack_density * 45.0
    + chord_fraction * 20.0
    + jack_excess * 15.0
    + vibro_density * 20.0
    + anchor_ratio * 15.0
    + jack_dom * 10.0
  );

  const stream_score = (
    stream_purity * 45.0
    + pymax(0.0, 1.0 - chord_fraction * 3.0) * 15.0
    - chord_fraction * 12.0
    + stream_dom * 20.0
    + pymax(0.0, 1.0 - density_cv) * 15.0
    + pymax(0.0, 0.5 - jack_density) * 10.0
  );

  const tech_score = (
    density_cv * 55.0
    + pattern_irregularity * 25.0
    + nps_active_cv * 20.0
    + tech_dom * 20.0
    + transition_var * 12.0
    + chord_fraction * 8.0
  );

  const speed_raw = (
    bpm_signal * 42.0
    + stream_purity * 22.0
    + pymax(0.0, 1.0 - density_cv) * 10.0
    + pymax(0.0, peak_ratio - 1.02) * 12.0
  );
  const speed_chord_gate = pymax(0.25, 1.0 - chord_fraction * 1.8);
  const speed_regularity = pymax(0.5, 1.0 - pymax(0.0, density_cv - 0.3) * 2.5);
  const speed_score = speed_raw * speed_chord_gate * speed_regularity;

  const drain_gate = pymax(0.0, drain_s - 60.0) / 120.0;
  const short_penalty = pymax(0.0, 1.0 - drain_s / 90.0);
  const chord_req = pymin(1.0, pymax(0.0, chord_fraction - 0.15) / 0.2);
  const stamina_raw = (
    drain_gate * 28.0
    + chord_fraction * 25.0
    - stream_purity * 20.0
    + nps_active_ratio * 12.0
    + pymax(0.0, 1.0 - density_cv) * 8.0
    + nps_sustained * 0.3
    + pymin(1.0, drain_s / 150.0) * 12.0
  );
  let stamina_score = stamina_raw * chord_req * pymax(0.4, 1.0 - short_penalty * 0.5);

  if (is_repetitive > 0.15) {
    stamina_score *= pymax(0.3, 1.0 - is_repetitive * 0.7);
  }

  const scores = {
    stream: stream_score,
    jack: jack_score,
    tech: tech_score,
    speed: speed_score,
    stamina: stamina_score,
  };

  const ranked = py.sortedBy(Object.entries(scores), (kv) => kv[1], true);
  const top_family = ranked[0][0];
  const top_score = ranked[0][1];
  const second_score = ranked.length > 1 ? ranked[1][1] : 0.0;

  const gap = top_score - second_score;
  const total = py.sum(ranked.map(([, s]) => s));
  let dominance; // eslint-disable-line no-unused-vars
  if (total > 0) dominance = top_score / total;
  else dominance = 0.0;

  let confidence = pymin(1.0, gap / pymax(total * 0.1, 1.0));
  let family;
  if (confidence < 0.15) {
    family = 'hybrid';
    confidence = pymax(0.1, confidence);
  } else {
    family = top_family;
  }

  if ((family === 'hybrid' || family === 'stream') && density_cv > 0.35) {
    const tech_s = py.get(scores, 'tech', 0);
    const stream_s = py.get(scores, 'stream', 0);
    if (tech_s > 0 && stream_s > 0 && tech_s / stream_s > 0.75) {
      family = 'tech';
      confidence = pymax(confidence, 0.3);
    }
  }

  if (family === 'stream' && bpm >= 155 && stream_purity > 0.7 && chord_fraction < 0.3) {
    const speed_s = py.get(scores, 'speed', 0);
    const stream_s = py.get(scores, 'stream', 0);
    if (speed_s > 0 && stream_s > 0 && speed_s / stream_s > 0.35) {
      family = 'speed';
      confidence = pymax(confidence, 0.25);
    }
  }

  const total_notes_eff = py.float(orElse(py.get(sunny_result, 'total_notes_eff', 0), 0));
  if (family === 'jack' && total_notes_eff > 15000) {
    family = py.get(scores, 'stamina', 0) > py.get(scores, 'tech', 0) ? 'stamina' : 'hybrid';
    confidence = pymin(confidence, 0.5);
  }

  if ((family === 'stream' || family === 'tech') && drain_s > 120 && chord_fraction > 0.25) {
    const stamina_s = py.get(scores, 'stamina', 0);
    const stream_s = py.get(scores, 'stream', 0);
    const tech_s = py.get(scores, 'tech', 0);
    if (stamina_s > 0 && stamina_s > pymax(stream_s, tech_s) * 0.85) {
      family = 'stamina';
      confidence = pymax(confidence, 0.35);
    }
  }

  if (!(family === 'tech' || family === 'jack') && timing_irregularity > 0.4
      && jack_density > 0.08 && chord_fraction > 0.15 && chord_fraction < 0.55
      && stream_purity > 0.4 && stream_purity < 0.85) {
    const tech_s = py.get(scores, 'tech', 0);
    const winner_s = py.get(scores, family, 0);
    if (tech_s > 0 && winner_s > 0 && tech_s > winner_s * 0.75) {
      family = 'tech';
      confidence = pymax(confidence, 0.35);
    }
  }

  let subtype_info = null;
  if (family === 'tech') {
    subtype_info = _classify_tech_subtype(features, sunny_result, profiles_cfg);
  } else if (family === 'hybrid' && tech_score >= top_score * 0.85) {
    subtype_info = _classify_tech_subtype(features, sunny_result, profiles_cfg);
  }

  const subtype_label = subtype_info !== null ? subtype_info.subtype : 'generic';
  const reason = (
    `${family} (score=${py.format(top_score, '.1f')}, gap=${py.format(gap, '.1f')}, `
    + `jack_ratio=${py.format(jack_ratio_broad, '.2f')}, jack_density=${py.format(jack_density, '.2f')}, `
    + `stream_purity=${py.format(stream_purity, '.2f')}, `
    + `density_cv=${py.format(density_cv, '.2f')}, bpm=${py.format(bpm, '.0f')}, `
    + `drain=${py.format(drain_s, '.0f')}s, subtype=${subtype_label})`
  );

  const rounded_scores = {};
  for (const [k, v] of Object.entries(scores)) rounded_scores[k] = py.round(v, 2);
  return {
    family,
    confidence: py.round(confidence, 3),
    scores: rounded_scores,
    subtype: subtype_label,
    subtype_confidence: subtype_info !== null ? subtype_info.confidence : 0.0,
    subtype_scores: subtype_info !== null ? subtype_info.scores : {},
    reason,
  };
}
