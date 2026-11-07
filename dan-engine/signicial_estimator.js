// signicial_estimator.js — Dan Signicial stage estimator (SR ruler + MSD fallback).
//
// JavaScript port of engine/dan_engine/signicial_estimator.py (DanOverlay 3.1.0 by 8doul).
//
// Architecture (mirrors rank_engine.py Reform + celestial_estimator.py):
//     SR (primary engine) + Signicial SR means per stage
//     → boundary interpolation → dp_signicial (1.0–16.0) → stage
//
// Stage order (weakest → strongest):
//     I, II, III, IV, V, VI, VII, VIII, IX, X,   ← Roman numeral stages (Prelude–Finale)
//     XI (Alpha), XII (Beta), XIII (Gamma), XIV (Delta),
//     LastStage (Epsilon), ExtraStageI (Zeta)
// dp_signicial: 1.0 (Stage I) → 16.0 (Zeta)
import * as py from './py.js';
import { path, sys, readText } from './vfs.js';

// The overlay always runs "frozen": resources live under sys._MEIPASS (vfs.RES_ROOT).
export const _PROFILES_PATH = path.join(sys._MEIPASS, 'config', 'signicial_profiles.json');

export const STAGE_KEYS = [
  'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
  'XI', 'XII', 'XIII', 'XIV', 'LastStage', 'ExtraStageI', 'ExtraStageII', 'ExtraStageIII',
];

export const _STAGE_DISPLAY = {
  I: 'Stage I',
  II: 'Stage II',
  III: 'Stage III',
  IV: 'Stage IV',
  V: 'Stage V',
  VI: 'Stage VI',
  VII: 'Stage VII',
  VIII: 'Stage VIII',
  IX: 'Stage IX',
  X: 'Stage X',
  XI: 'Alpha',
  XII: 'Beta',
  XIII: 'Gamma',
  XIV: 'Delta',
  LastStage: 'Epsilon',
  ExtraStageI: 'Zeta',
  ExtraStageII: 'Eta',
  ExtraStageIII: 'Theta',
};

// Compact labels (Greek letters from XI upward)
export const _STAGE_SHORT = {
  I: 'I',
  II: 'II',
  III: 'III',
  IV: 'IV',
  V: 'V',
  VI: 'VI',
  VII: 'VII',
  VIII: 'VIII',
  IX: 'IX',
  X: 'X',
  XI: 'α',
  XII: 'β',
  XIII: 'γ',
  XIV: 'δ',
  LastStage: 'ε',
  ExtraStageI: 'ζ',
  ExtraStageII: 'η',
  ExtraStageIII: 'θ',
};

// Stage names for the Roman numeral stages
export const STAGE_SUBTITLE = {
  I: 'Prelude',
  II: 'Abnormality',
  III: 'Termination',
  IV: 'Resuscitation',
  V: 'Disturbance',
  VI: 'Revitalization',
  VII: 'Motivation',
  VIII: 'Misfortune',
  IX: 'Catastrophe',
  X: 'Finale',
};

export const SKILLSET_KEYS = [
  'stream', 'jumpstream', 'handstream', 'stamina',
  'jackspeed', 'chordjack', 'technical',
];

export const _STAGE_DP = Object.fromEntries(STAGE_KEYS.map((key, i) => [key, i + 1]));

export let _profiles = null;
export const _ruler_caches = new Map(); // chart_type (str | null) → ruler | null
export const _boundaries_caches = new Map(); // chart_type (str | null) → boundaries | null

// classifier family → Signicial chart type (None = general ruler)
export const _FAMILY_TO_TYPE = {
  jack: 'jack', speed: 'speed', stamina: 'stamina', tech: 'tech',
  stream: null, hybrid: null,
};

// Weight of the type-specific SR vs the general SR in typed rulers
// (the remainder shrinks toward the general mean).
export const _SHRINKAGE = 0.6;

// ── Profile loading ──────────────────────────────────────────────────

export function _load_profiles() {
  if (_profiles !== null) return _profiles;
  if (!path.isfile(_PROFILES_PATH)) return null;
  try {
    _profiles = JSON.parse(readText(_PROFILES_PATH));
    return _profiles;
  } catch (e) {
    // except (json.JSONDecodeError, OSError)
    if (e instanceof SyntaxError || e instanceof py.JSONDecodeError || e instanceof py.OSError_) return null;
    throw e;
  }
}

// ── SR ruler ─────────────────────────────────────────────────────────

/**
 * Build SR ruler from profiles.
 *
 * When chart_type is provided (e.g. "jack"), uses sr_by_type for that type, falling back to
 * sr_mean for stages where the type value is missing. When chart_type is None, uses the
 * general sr_mean. Returns sorted list of [sr_mean, dp_int] where dp_int is 1-16, or null.
 */
export function _build_ruler(chart_type = null) {
  if (_ruler_caches.has(chart_type)) return _ruler_caches.get(chart_type);

  const profiles = _load_profiles();
  if (!py.truthy(profiles)) return null;

  // General ruler: fallback / shrinkage target for typed rulers
  const general = chart_type !== null ? _build_ruler(null) : null;
  const general_map = new Map();
  for (const [sr, dp] of (py.truthy(general) ? general : [])) general_map.set(dp, sr);

  let ruler = [];
  for (const key of STAGE_KEYS) {
    const slot = py.get(profiles, key);
    if (!py.truthy(slot)) continue;
    const dp_int = py.getitem(_STAGE_DP, key);

    let sr;
    if (chart_type === null) {
      sr = py.get(slot, 'sr_mean');
    } else {
      const by_type = py.get(slot, 'sr_by_type');
      const type_sr = py.get(py.truthy(by_type) ? by_type : {}, chart_type);
      const gen_sr = py.get(general_map, dp_int);
      if (type_sr !== null && gen_sr !== null) {
        sr = _SHRINKAGE * py.float(type_sr) + (1.0 - _SHRINKAGE) * gen_sr;
      } else if (type_sr !== null) {
        sr = type_sr;
      } else {
        sr = gen_sr;
      }
    }

    if (sr === null || py.float(sr) <= 0) continue;
    ruler.push([py.float(sr), dp_int]);
  }

  if (ruler.length < 2) {
    _ruler_caches.set(chart_type, null);
    _boundaries_caches.set(chart_type, null);
    return null;
  }

  // Sort by stage, then enforce strict monotonicity
  ruler = py.sortedBy(ruler, (x) => x[1]);
  for (let i = 1; i < ruler.length; i++) {
    if (ruler[i][0] <= ruler[i - 1][0]) ruler[i] = [ruler[i - 1][0] + 0.001, ruler[i][1]];
  }

  // Boundaries: midpoints between adjacent means, extrapolated at both ends
  const means = ruler.map(([sr]) => sr);
  const n = means.length;
  const boundaries = [];
  for (let i = 0; i < n; i++) {
    const lower = i > 0 ? (means[i - 1] + means[i]) / 2.0 : means[0] - (n > 1 ? (means[1] - means[0]) / 2.0 : 1.0);
    const upper = i < n - 1 ? (means[i] + means[i + 1]) / 2.0 : means[n - 1] + (n > 1 ? (means[n - 1] - means[n - 2]) / 2.0 : 1.0);
    boundaries.push([lower, upper, ruler[i][1]]);
  }

  _ruler_caches.set(chart_type, ruler);
  _boundaries_caches.set(chart_type, boundaries);
  return ruler;
}

// ── SR → dp ──────────────────────────────────────────────────────────

/**
 * Convert SR → dp_signicial (1.0–18.0+) via boundary interpolation.
 * Uses chart_type-specific ruler when available, general otherwise.
 */
export function _sr_to_dp_signicial(sr, chart_type = null) {
  let resolved_type = chart_type;
  let ruler = _build_ruler(resolved_type);
  if (ruler === null && resolved_type !== null) {
    resolved_type = null;
    ruler = _build_ruler(null);
  }
  if (ruler === null) return null;

  const boundaries = py.get(_boundaries_caches, resolved_type);
  if (!py.truthy(boundaries)) return null;

  // Below the first stage
  if (sr < boundaries[0][0]) return py.max2(0.5, py.float(boundaries[0][2]));

  // Above the last stage
  const last = boundaries[boundaries.length - 1];
  if (sr >= last[1]) return py.min2(18.99, py.float(last[2]) + 0.99);

  // Interpolate within the matching stage
  for (const [low_sr, high_sr, dp_int] of boundaries) {
    if (low_sr <= sr && sr < high_sr) {
      const width = py.max2(high_sr - low_sr, 1e-6);
      const t = (sr - low_sr) / width;
      return py.float(dp_int) + t;
    }
  }

  return null;
}

/** Map a dp_signicial float (1.0-18.0+) to the stage key. */
export function _dp_to_stage_key(dp) {
  let idx = py.min2(py.math.floor(dp) - 1, STAGE_KEYS.length - 1);
  idx = py.max2(0, idx);
  return STAGE_KEYS[idx];
}

// ── MSD fallback ─────────────────────────────────────────────────────

/**
 * Flat MSD distance fallback when SR is unavailable: overall MSD from the skillsets,
 * mapped through the stage means stored in the profiles.
 */
export function _estimate_from_msd(skillsets) {
  const profiles = _load_profiles();
  if (!py.truthy(profiles)) return null;

  const vals = SKILLSET_KEYS.map((k) => {
    const v = py.get(skillsets, k, 0.0);
    return py.float(py.truthy(v) ? v : 0.0);
  });
  if (!py.truthy(vals) || py.max(vals) < 1.0) return null;
  const overall = py.max(vals);

  // MSD ruler: [overall_msd, dp_int] per stage
  const msd_ruler = [];
  for (const key of STAGE_KEYS) {
    const slot = py.get(profiles, key, {});
    const msd_val = py.get(slot, 'overall_msd');
    if (py.truthy(msd_val)) msd_ruler.push([py.float(msd_val), py.getitem(_STAGE_DP, key)]);
  }

  if (msd_ruler.length < 2) return null;

  // Clamp / extrapolate at the ends, interpolate in between
  if (overall <= msd_ruler[0][0]) return py.float(msd_ruler[0][1]);
  if (overall >= msd_ruler[msd_ruler.length - 1][0]) {
    const [sr_last, dp_last] = msd_ruler[msd_ruler.length - 1];
    const [sr_prev] = msd_ruler[msd_ruler.length - 2];
    const step = sr_last - sr_prev;
    if (step > 0) return py.float(dp_last) + (overall - sr_last) / step;
    return py.float(dp_last);
  }

  for (let i = 0; i < msd_ruler.length - 1; i++) {
    const [m_lo, dp_lo] = msd_ruler[i];
    const [m_hi, dp_hi] = msd_ruler[i + 1];
    if (m_lo <= overall && overall <= m_hi) {
      const frac = py.div(overall - m_lo, m_hi - m_lo);
      return dp_lo + frac * (dp_hi - dp_lo);
    }
  }

  return null;
}

// ── Confidence ───────────────────────────────────────────────────────

/**
 * Confidence 0-1 based on how centred dp is within its stage slot
 * (1.0 = exactly at slot centre, 0.0 = at the boundary between two stages).
 */
export function _confidence_from_dp_frac(dp) {
  const frac = dp - py.int(dp);
  // Distance from the slot centre (0 … 0.5)
  const dist = Math.abs(frac - 0.5);
  // Raised cosine: 1.0 at the centre, 0.0 at the boundaries
  const conf = 0.5 * (1.0 + py.math.cos(py.math.pi * dist / 0.5));
  return py.max2(0.0, py.min2(1.0, conf));
}

// ── Result ───────────────────────────────────────────────────────────

/** @dataclass SignicialResult */
export class SignicialResult {
  constructor(stage_key, label, short, subtitle, confidence, dp_signicial, beyond = false) {
    this.stage_key = stage_key;
    this.label = label;
    this.short = short;
    this.subtitle = subtitle;
    this.confidence = confidence;
    this.dp_signicial = dp_signicial;
    this.beyond = beyond;
  }

  to_dict() {
    return {
      stage_key: this.stage_key,
      label: this.label,
      short: this.short,
      subtitle: this.subtitle,
      confidence: py.round(this.confidence, 4),
      dp_signicial: py.round(this.dp_signicial, 3),
      beyond: this.beyond,
    };
  }
}

// ── Public API ───────────────────────────────────────────────────────

/**
 * Re-derive Signicial result fields from a ``dp_signicial`` value (used by the pipeline's
 * custom-rate interpolation to refresh the stage/label fields).
 */
export function fields_from_dp(dp) {
  const beyond = dp > 18.99;
  dp = py.max2(1.0, dp);
  const stage_key = _dp_to_stage_key(dp);
  return {
    stage_key,
    label: py.getitem(_STAGE_DISPLAY, stage_key),
    short: py.getitem(_STAGE_SHORT, stage_key),
    subtitle: py.get(STAGE_SUBTITLE, stage_key, ''),
    dp_signicial: py.round(dp, 3),
    beyond,
  };
}

/**
 * Estimate Signicial stage from MSD skillsets and optional SR (SR ruler first, MSD distance
 * fallback). family_hint selects a type-specific ruler when one exists. Returns a
 * SignicialResult or null.
 */
export function estimate(skillsets, sr = null, family_hint = null) {
  let dp = null;
  const chart_type = py.truthy(family_hint) ? py.get(_FAMILY_TO_TYPE, family_hint) : null;

  // Primary: SR ruler
  if (sr !== null && sr !== undefined && sr > 0.0) dp = _sr_to_dp_signicial(sr, chart_type);

  // Fallback: MSD
  if (dp === null && py.truthy(skillsets)) dp = _estimate_from_msd(skillsets);

  if (dp === null) return null;

  const beyond = dp > 18.99;
  dp = py.max2(1.0, dp);

  const stage_key = _dp_to_stage_key(dp);
  const label = py.getitem(_STAGE_DISPLAY, stage_key);
  const short = py.getitem(_STAGE_SHORT, stage_key);
  const subtitle = py.get(STAGE_SUBTITLE, stage_key, '');
  const confidence = _confidence_from_dp_frac(dp);

  return new SignicialResult(stage_key, label, short, subtitle, confidence, py.round(dp, 3), beyond);
}
