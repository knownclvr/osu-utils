// minacalc_estimator.js — role-based dan estimation from MinaCalc skillsets.
//
// JavaScript port of engine/dan_engine/minacalc_estimator.py (DanOverlay 3.1.0 by 8doul).
//
// Architecture:
//     .osu → MinaCalc MSD@1.0x → 4 role scores → per-role dan lookup
//          → structural feature modulation → composite dan → result dict
//
// Roles:
//     jack    = max(jackspeed, chordjack)
//     tech    = technical
//     speed   = max(stream, jumpstream)
//     stamina = 0.7 * stamina + 0.3 * handstream  (blended — must match calibration script)
//
// Structural integration (3 tiers):
//     A — confidence-only: features modulate confidence, never shift dan
//     B — microshift: features allow ±0.5 dan adjustment for well-supported signals
//     C — explanation: features only appear in the explanation text
//
// Requires config/role_scales.json (loaded into the vfs by setup.js).
// Python imports parser / feature_extractor lazily inside _extract_structural; here they are
// static imports (everything stays synchronous).
import * as py from './py.js';
import * as vfs from './vfs.js';
import { calc as minacalc_calc } from './minacalc_bridge.js';
import { parsear_osu_v2 } from './parser.js';
import { extract_features } from './feature_extractor.js';

export { minacalc_calc };

const { path, sys } = vfs;

// Frozen build (always the case here): resources live under sys._MEIPASS.
export const _SCALES_PATH = path.join(sys._MEIPASS, 'config', 'role_scales.json');

export const _ROLES = ['jack', 'speed', 'tech', 'stamina'];

export const _DAN_ORDER = [
  '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
  'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa',
];

export const _DAN_LABELS = {
  '1st': '1st Dan', '2nd': '2nd Dan', '3rd': '3rd Dan',
  '4th': '4th Dan', '5th': '5th Dan', '6th': '6th Dan',
  '7th': '7th Dan', '8th': '8th Dan', '9th': '9th Dan',
  '10th': '10th Dan', Alpha: 'Alpha', Beta: 'Beta',
  Gamma: 'Gamma', Delta: 'Delta', Epsilon: 'Epsilon',
  Zeta: 'Zeta', Eta: 'Eta', Theta: 'Theta',
  Iota: 'Iota', Kappa: 'Kappa',
};

export let _scales = null;

// ── JS-only guards: Python raises TypeError where JS would coerce None ──
function _num(x, op) {
  if (typeof x === 'number') return x;
  if (typeof x === 'boolean') return x ? 1 : 0;
  throw new py.TypeError_(`unsupported operand type(s) for ${op}: '${x === null || x === undefined ? 'NoneType' : typeof x}'`);
}

/** round(x, n) — TypeError for None like Python (py.round would pass it through). */
function _round(x, n) {
  if (typeof x === 'boolean') x = x ? 1 : 0;
  if (typeof x !== 'number') {
    throw new py.TypeError_(`type ${x === null || x === undefined ? 'NoneType' : typeof x} doesn't define __round__ method`);
  }
  return py.round(x, n);
}

export function _load_scales() {
  if (_scales !== null) return _scales;
  if (!path.isfile(_SCALES_PATH)) return null;
  try {
    let text;
    try {
      text = vfs.readText(_SCALES_PATH);
    } catch (e) {
      if (e instanceof py.OSError_) return null;
      throw e;
    }
    _scales = JSON.parse(text);
    return _scales;
  } catch (e) {
    if (e instanceof SyntaxError) return null; // json.JSONDecodeError
    throw e;
  }
}

export const _DAN_SHORT = {
  '1st': '1', '2nd': '2', '3rd': '3', '4th': '4', '5th': '5',
  '6th': '6', '7th': '7', '8th': '8', '9th': '9', '10th': '10',
  Alpha: 'α', Beta: 'β', Gamma: 'γ', Delta: 'δ', Epsilon: 'ε',
  Zeta: 'ζ', Eta: 'η', Theta: 'θ', Iota: 'ι', Kappa: 'κ',
};

/** Compute the 4 role scores from raw MinaCalc skillsets. */
export function _role_scores(scores) {
  const stam = py.get(scores, 'stamina', 0.0);
  const hs = py.get(scores, 'handstream', 0.0);
  const jack = py.max2(py.get(scores, 'jackspeed', 0.0), py.get(scores, 'chordjack', 0.0));
  const tech = py.get(scores, 'technical', 0.0);
  const speed = py.max2(py.get(scores, 'stream', 0.0), py.get(scores, 'jumpstream', 0.0));
  const stamina = 0.7 * _num(stam, '*') + 0.3 * _num(hs, '*');
  return { jack, tech, speed, stamina };
}

/** Parse the .osu and extract structural features. Returns {} on failure. */
export function _extract_structural(osu_path) {
  try {
    const parsed = parsear_osu_v2(osu_path, null, true, true, 4, 0, false);
    if (py.truthy(py.get(parsed, 'rejected', null))) return {};
    return extract_features(parsed);
  } catch {
    return {};
  }
}

/** Position within band: 0.0 at lower, 0.5 at center, 1.0 at upper. */
export function _band_position(value, lower, center, upper) {
  let pos;
  if (value <= center) {
    const half = py.max2(center - lower, 0.001);
    pos = (0.5 * (value - lower)) / half;
  } else {
    const half = py.max2(upper - center, 0.001);
    pos = 0.5 + (0.5 * (value - center)) / half;
  }
  return py.max2(0.0, py.min2(1.0, pos));
}

/** Confidence based on proximity to center within the band. 1.0 at center, 0.0 at edges. */
export function _center_confidence(value, lower, center, upper) {
  const half_width = py.max2((upper - lower) / 2, 0.001);
  const dist = Math.abs(value - center);
  return py.max2(0.0, py.min2(1.0, 1.0 - dist / half_width));
}

/** Quality multiplier based on anchor reliability. */
export function _anchor_quality(sample_count, risk_flag, dispersion) {
  if (sample_count < 2 || risk_flag === 'weak') return 0.65;
  if (risk_flag === 'dispersed' || dispersion > 3.0) return 0.8;
  return 1.0;
}

/**
 * Return [dan, position, confidence] using interval-based lookup.
 *
 * Uses [lower, upper] bands with center as primary anchor. Confidence is modulated by
 * proximity to center and anchor quality. Overlap zones reduce confidence to signal ambiguity.
 */
export function _lookup_role_dan(value, role_thresholds) {
  const bands = [];
  for (const d of _DAN_ORDER) {
    if (py.contains(role_thresholds, d)) {
      const t = py.getitem(role_thresholds, d);
      bands.push([
        d,
        py.getitem(t, 'lower'), py.getitem(t, 'center'), py.getitem(t, 'upper'),
        py.get(t, 'sample_count', 1),
        py.get(t, 'risk_flag', 'weak'),
        py.get(t, 'dispersion', 0.0),
      ]);
    }
  }

  if (!bands.length) return ['1st', 0.5, 0.0];

  // `b[1] <= value <= b[3]` raises TypeError for a None score
  _num(value, '<=');
  const containing = [];
  bands.forEach((b, i) => {
    if (b[1] <= value && value <= b[3]) containing.push([i, b]);
  });

  if (containing.length === 1) {
    const [, [dan, lower, center, upper, n, flag, disp]] = containing[0];
    const position = _band_position(value, lower, center, upper);
    const base_conf = _center_confidence(value, lower, center, upper);
    const quality = _anchor_quality(n, flag, disp);
    return [dan, py.round(position, 3), py.round(base_conf * quality, 3)];
  }

  if (containing.length > 1) {
    const best = py.min(containing, (x) => Math.abs(value - x[1][2]));
    const [, [dan, lower, center, upper, n, flag, disp]] = best;
    const position = _band_position(value, lower, center, upper);
    const base_conf = _center_confidence(value, lower, center, upper);
    const quality = _anchor_quality(n, flag, disp);
    const overlap_penalty = 0.7;
    return [dan, py.round(position, 3), py.round(base_conf * quality * overlap_penalty, 3)];
  }

  const best_idx = py.min(py.range(bands.length), (i) => Math.abs(value - bands[i][2]));
  const [dan, lower, center, upper, n, flag, disp] = bands[best_idx];
  const position = _band_position(value, lower, center, upper);
  const half_width = py.max2((upper - lower) / 2, 0.001);
  const dist = Math.abs(value - center);
  const base_conf = py.max2(0.0, 1.0 - dist / half_width) * 0.5;
  const quality = _anchor_quality(n, flag, disp);
  return [dan, py.round(position, 3), py.round(base_conf * quality, 3)];
}

export function _sublevel(position) {
  if (position < 0.2) return 'Low';
  else if (position < 0.4) return 'Mid-Low';
  else if (position < 0.6) return 'Mid';
  else if (position < 0.8) return 'Mid-High';
  return 'High';
}

export function _dan_index(dan) {
  const i = _DAN_ORDER.indexOf(dan);
  return i < 0 ? 0 : i;
}

export const _STRUCTURAL_ROLE_BIAS = {
  jack: { feature: 'jack_density', threshold: 0.10, bonus: 0.15 },
  speed: { feature: 'stream_purity', threshold: 0.65, bonus: 0.15 },
  stamina: { feature: 'stamina_index', threshold: 0.50, bonus: 0.10 },
  tech: { feature: 'pattern_irregularity', threshold: 0.10, bonus: 0.10 },
};

/** Return the dominant role using normalized MSD + confidence + structural bias. */
export function _primary_role_enhanced(rs, role_estimates, scales_data, features) {
  const role_scales = py.get(scales_data, 'scales', {});
  const combined = {};
  for (const role of Object.keys(role_estimates)) {
    const scale = py.get(role_scales, role, {});
    const max_anchor = py.max(Object.values(scale).map((e) => py.get(e, 'role_score', 1.0)), undefined, 1.0);
    const norm = py.getitem(rs, role) / py.max2(max_anchor, 0.001);

    const conf = py.getitem(py.getitem(role_estimates, role), 'confidence');

    let score = norm * (0.1 + 0.9 * conf);

    const bias_info = py.get(_STRUCTURAL_ROLE_BIAS, role, null);
    if (py.truthy(bias_info) && py.truthy(features)) {
      const feat_val = py.get(features, bias_info.feature, 0.0);
      if (feat_val > bias_info.threshold) score += bias_info.bonus;
    }
    combined[role] = score;
  }

  if (py.truthy(rs)) {
    const max_non_jack = py.max(Object.keys(rs).filter((r) => r !== 'jack').map((r) => py.get(rs, r, 0.0)), undefined, 0.0);
    const jack_score = py.get(rs, 'jack', 0.0);
    if (max_non_jack > jack_score + 6.0) {
      const scale = py.get(role_scales, 'jack', {});
      const max_anchor = py.max(Object.values(scale).map((e) => py.get(e, 'role_score', 1.0)), undefined, 1.0);
      const norm_jack = jack_score / py.max2(max_anchor, 0.001);
      const conf_jack = py.get(py.get(role_estimates, 'jack', {}), 'confidence', 0.0);
      combined.jack = norm_jack * (0.1 + 0.9 * conf_jack);
    }
  }

  if (py.truthy(rs) && py.len(rs) > 1) {
    const combined_leader = py.max(Object.keys(combined), (r) => combined[r]);
    let override = null;

    const raw_vals = py.sortedBy(Object.values(rs), (v) => v, true);
    const raw_leader = py.max(Object.keys(rs), (r) => rs[r]);
    if ((raw_leader === 'tech' || raw_leader === 'jack')
        && raw_vals[0] - raw_vals[1] >= 2.7
        && combined_leader !== raw_leader) {
      override = raw_leader;
    }

    if (override === null && (combined_leader === 'tech' || combined_leader === 'jack')) {
      const winner_conf = py.get(py.get(role_estimates, combined_leader, {}), 'confidence', 1.0);
      if (winner_conf < 0.6) {
        const combined_winner_raw = py.get(rs, combined_leader, 0.0);
        for (const candidate of ['tech', 'jack']) {
          if (candidate !== combined_leader && py.contains(rs, candidate)
              && rs[candidate] - combined_winner_raw >= 4.0) {
            override = candidate;
            break;
          }
        }
      }
    }
    if (py.truthy(override) && py.contains(combined, override)) {
      combined[override] = combined[combined_leader] + 0.01;
    }
  }

  return py.max(Object.keys(combined), (r) => combined[r]);
}

/**
 * Compute composite dan with specialist/hybrid awareness.
 *
 * - Specialist (spread >= 3): primary role dominates; composite = primary dan
 * - Balanced (spread <= 2): confidence-weighted average across roles
 * - Generalist (spread <= 1): all roles agree
 */
export function _composite_dan(role_estimates) {
  let estimates = [];
  for (const [role, est] of Object.entries(role_estimates)) {
    const idx = _dan_index(py.getitem(est, 'dan'));
    estimates.push([idx, py.getitem(est, 'position'), py.getitem(est, 'confidence'), role]);
  }

  if (!estimates.length) return ['1st', 'jack', false];

  estimates = py.sortedBy(estimates, (x) => [x[0], x[1]]);

  const [bottom_idx, , , bottom_role] = estimates[0];
  const top_idx = estimates[estimates.length - 1][0];
  const spread = top_idx - bottom_idx;
  const is_generalist = spread <= 1;

  let composite_idx;
  if (spread <= 2) {
    const total_w = py.sum(estimates.map((e) => py.max2(e[2], 0.05)));
    const weighted_idx = py.sum(estimates.map((e) => (e[0] + e[1]) * py.max2(e[2], 0.05))) / total_w;
    composite_idx = py.roundInt(weighted_idx);
  } else {
    const best = py.max(estimates, (e) => e[2]);
    composite_idx = best[0];

    if (spread >= 5) composite_idx = py.max2(0, composite_idx - 1);
  }

  const composite_dan = _DAN_ORDER[py.min2(py.max2(0, composite_idx), _DAN_ORDER.length - 1)];
  return [composite_dan, bottom_role, is_generalist];
}

/**
 * Return a multiplier [0.5, 1.0] that penalizes confidence when structural
 * features contradict the assigned primary role.
 */
export function _structural_confidence_mod(primary, features) {
  if (!py.truthy(features)) return 1.0;

  let mod = 1.0;

  if (primary === 'jack') {
    const jd = py.get(features, 'jack_density', 0.0);
    if (jd < 0.03) mod *= 0.6;
    else if (jd < 0.06) mod *= 0.8;
  } else if (primary === 'speed') {
    const sp = py.get(features, 'stream_purity', 0.0);
    if (sp < 0.25) mod *= 0.7;
    else if (sp < 0.4) mod *= 0.85;
  } else if (primary === 'stamina') {
    const si = py.get(features, 'stamina_index', 0.0);
    const dur = py.get(features, 'duration_s', 0.0);
    if (dur < 50) mod *= 0.7;
    else if (si < 0.15) mod *= 0.75;
  } else if (primary === 'tech') {
    const pi = py.get(features, 'pattern_irregularity', 0.0);
    if (pi < 0.04) mod *= 0.8;
  }

  return py.max2(0.5, mod);
}

export const _MICROSHIFT_CAP = 0.5;

/**
 * Return a small dan shift [-0.5, +0.5] based on structural features.
 *
 * Only fires for well-supported signals where structural data adds real
 * discriminative power beyond what MinaCalc MSD already captures.
 */
export function _structural_microshift(primary, primary_dan, features, scales_data) {
  if (!py.truthy(features)) return 0.0;

  let shift = 0.0;

  if (primary === 'jack') {
    const jd = py.get(features, 'jack_density', 0.0);
    const cc = py.get(features, 'chord_complexity', 0.0);

    if (jd > 0.25) shift += 0.25;
    else if (jd > 0.15 && cc < 0.1) shift += 0.15;

    if (jd < 0.03) shift -= 0.25;
  } else if (primary === 'speed') {
    const sp = py.get(features, 'stream_purity', 0.0);
    const dcv = py.get(features, 'density_cv', 0.0);

    if (sp > 0.8) shift += 0.15;

    if (dcv > 0.5) shift -= 0.2;
  } else if (primary === 'stamina') {
    const si = py.get(features, 'stamina_index', 0.0);
    const dur = py.get(features, 'duration_s', 0.0);

    if (si > 0.7) shift += 0.2;

    if (dur < 60) shift -= 0.3;
    else if (dur < 90) shift -= 0.1;
  } else if (primary === 'tech') {
    const pi = py.get(features, 'pattern_irregularity', 0.0);
    const tv = py.get(features, 'transition_var', 0.0);

    if (pi > 0.15) shift += 0.15;

    if (tv > 0.65) shift += 0.1;
  }

  return py.max2(-_MICROSHIFT_CAP, py.min2(_MICROSHIFT_CAP, shift));
}

export const _JACK_DOM_THRESHOLD = 0.55;
export const _JACK_DOM_MAX_SHIFT = 0.25;
export const _JACK_DOM_SCALE = 0.45;

/**
 * Additional shift for jack-dominant charts that MinaCalc under-rates.
 *
 * Adapted from Block E2 strain.py: uses jack_ratio as dominance proxy
 * and density_cv as variance proxy.
 */
export function _jack_dominance_shift(features) {
  if (!py.truthy(features)) return 0.0;
  const jr = py.get(features, 'jack_ratio', 0.0);
  if (jr < _JACK_DOM_THRESHOLD) return 0.0;
  const dcv = py.min2(1.0, py.get(features, 'density_cv', 0.0));
  return py.min2(_JACK_DOM_MAX_SHIFT, dcv * _JACK_DOM_SCALE * jr);
}

/** Apply a fractional dan shift to [dan, position], clamping to valid range. */
export function _apply_shift(dan, position, shift) {
  const idx = _dan_index(dan);
  const effective = idx + position + shift;
  let new_idx = py.int(effective);
  let new_pos = effective - new_idx;
  new_idx = py.max2(0, py.min2(_DAN_ORDER.length - 1, new_idx));
  new_pos = py.max2(0.0, py.min2(0.99, new_pos));
  return [_DAN_ORDER[new_idx], py.round(new_pos, 3)];
}

/** Build a human-readable explanation of the dan assignment. */
export function _build_explanation(primary, primary_dan, features, microshift, role_estimates, agreement_conf) {
  const parts = [];

  const role_label = { jack: 'Jack', speed: 'Speed', tech: 'Tech', stamina: 'Stamina' };
  parts.push(`${primary_dan} by ${py.get(role_label, primary, primary)} dominance`);

  if (py.truthy(features)) {
    const tags = [];
    const sp = py.get(features, 'stream_purity', 0.0);
    const jd = py.get(features, 'jack_density', 0.0);
    const pi = py.get(features, 'pattern_irregularity', 0.0);
    const si = py.get(features, 'stamina_index', 0.0);
    const cc = py.get(features, 'chord_complexity', 0.0);
    const dcv = py.get(features, 'density_cv', 0.0);
    const dur = py.get(features, 'duration_s', 0.0);

    if (sp > 0.75) tags.push('high stream purity');
    if (jd > 0.15) tags.push('dense jacks');
    else if (jd > 0.08) tags.push('moderate jacks');
    if (pi > 0.15) tags.push('irregular patterns');
    if (si > 0.6) tags.push('sustained density');
    if (cc > 0.3) tags.push('complex chords');
    if (dcv > 0.5) tags.push('variable density');
    if (dur > 180) tags.push('long map');

    if (tags.length) parts.push(tags.join('; '));
  }

  if (Math.abs(microshift) > 0.05) {
    const direction = microshift > 0 ? 'up' : 'down';
    parts.push(`structural shift ${direction} (${py.format(microshift, '+.2f')})`);
  }

  if (agreement_conf < 0.5) {
    if (py.truthy(role_estimates)) {
      const primary_idx = _dan_index(primary_dan);
      const worst = py.max(
        Object.entries(role_estimates),
        (kv) => Math.abs(_dan_index(py.getitem(kv[1], 'dan')) - primary_idx),
      );
      parts.push(`low agreement (${worst[0]} at ${py.getitem(worst[1], 'dan')})`);
    }
  }

  return parts.join(' · ');
}

/** Map the primary role to a family label for the overlay. */
export function _derive_family(primary) {
  const mapping = { jack: 'jack', tech: 'tech', speed: 'stream', stamina: 'stamina' };
  return py.get(mapping, primary, 'hybrid');
}

/**
 * Estimate the Reform dan for *osu_path* using role-based MSD lookup with structural
 * feature modulation.
 *
 * Returns None if prerequisites are missing (no CLI, no scales, non-4K).
 *
 * When *features* is supplied by the caller (keyword-only in Python), the redundant .osu
 * parse inside _extract_structural is skipped.
 */
export function estimate(osu_path, rate = 1.0, features = null) {
  const scales_data = _load_scales();
  if (scales_data === null) return null;

  const scores = minacalc_calc(osu_path, rate, 0.93, 'msd');
  if (scores === null) return null;

  const duration_s = py.float(py.get(scores, 'duration_s', 0));
  const warnings = [];

  if (duration_s < 30) warnings.push(`very_short_map_${py.reprInt(py.int(duration_s))}s`);

  if (features === null || features === undefined) features = _extract_structural(osu_path);
  if (!py.truthy(features)) warnings.push('structural_features_unavailable');

  const rs = _role_scores(scores);

  const thresholds = py.get(scales_data, 'thresholds', {});
  const role_estimates = {};

  for (const role of _ROLES) {
    const role_thresh = py.get(thresholds, role, {});
    if (!py.truthy(role_thresh)) continue;
    const [dan, position, confidence] = _lookup_role_dan(rs[role], role_thresh);
    role_estimates[role] = {
      dan,
      dan_label: py.get(_DAN_LABELS, dan, dan),
      position,
      sublevel: _sublevel(position),
      confidence,
      msd: _round(rs[role], 2),
    };
  }

  if (!py.truthy(role_estimates)) return null;

  const primary = _primary_role_enhanced(rs, role_estimates, scales_data, features);
  const primary_est = py.getitem(role_estimates, primary);
  const primary_dan = primary_est.dan;
  const primary_position = primary_est.position;

  let microshift = _structural_microshift(primary, primary_dan, features, scales_data);

  if (primary === 'jack') {
    microshift += _jack_dominance_shift(features);
    microshift = py.max2(-_MICROSHIFT_CAP, py.min2(_MICROSHIFT_CAP, microshift));
  }

  const [adjusted_dan, adjusted_position] = _apply_shift(primary_dan, primary_position, microshift);

  const [composite_dan, bottleneck_role, is_generalist] = _composite_dan(role_estimates);

  const primary_conf_raw = role_estimates[primary].confidence;
  const composite_idx_raw = _dan_index(composite_dan);
  const adjusted_idx_raw = _dan_index(adjusted_dan);
  const _use_composite = (
    primary_conf_raw < 0.5
    && composite_idx_raw === 7
    && adjusted_idx_raw <= 5
  );

  const final_dan = _use_composite ? composite_dan : adjusted_dan;
  const final_position = _use_composite ? 0.5 : adjusted_position;

  const primary_idx = _dan_index(final_dan);
  let agreeing = 0;
  for (const e of Object.values(role_estimates)) {
    if (Math.abs(_dan_index(e.dan) - primary_idx) <= 1) agreeing += 1;
  }

  const agreement_confidence = agreeing / py.max2(py.len(role_estimates), 1);
  const struct_conf_mod = _structural_confidence_mod(primary, features);
  const final_confidence = py.round(agreement_confidence * struct_conf_mod, 3);

  const role_breakdown_text = _ROLES
    .filter((role) => py.contains(role_estimates, role))
    .map((role) => `${role[0].toUpperCase()}:${py.get(_DAN_SHORT, role_estimates[role].dan, role_estimates[role].dan)}`)
    .join(' · ');

  const explanation = _build_explanation(
    primary, final_dan, features, microshift,
    role_estimates, agreement_confidence,
  );

  const family = _derive_family(primary);

  const dan_idx = _dan_index(final_dan);
  const dp_approx = py.round(1.0 + dan_idx + final_position, 2);

  const skillsets = {};
  for (const k of ['stream', 'jumpstream', 'handstream', 'stamina', 'jackspeed', 'chordjack', 'technical']) {
    if (py.contains(scores, k)) skillsets[k] = _round(py.get(scores, k, 0.0), 2);
  }
  const overall_msd = _round(py.get(scores, 'overall', 0.0), 2);
  const role_scores = {};
  for (const [r, v] of Object.entries(rs)) role_scores[r] = _round(v, 2);

  return {
    dp: dp_approx,
    dan_label: py.get(_DAN_LABELS, final_dan, final_dan),
    dan_short: final_dan,
    sublevel: _sublevel(final_position),
    confidence: final_confidence,
    family,
    corrections: [],
    primary_dan: final_dan,
    primary_role: primary,
    role_estimates,
    composite_dan,
    bottleneck_role,
    is_generalist,
    role_breakdown_text,
    structural_shift: py.round(microshift, 3),
    structural_conf_mod: py.round(struct_conf_mod, 3),
    explanation,
    features: py.truthy(features) ? features : null,
    skillsets,
    overall_msd,
    role_scores,
    duration_s,
    note_count: py.get(scores, 'note_count', 0),
    warnings,
    engine: 'minacalc_roles_v2',
  };
}
