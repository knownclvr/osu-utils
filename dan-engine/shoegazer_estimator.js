// shoegazer_estimator.js — Dan Shoegazer stage estimator (SR ruler + MSD fallback).
//
// JavaScript port of engine/dan_engine/shoegazer_estimator.py (DanOverlay 3.1.0 by 8doul).
//
// Architecture (mirrors signicial_estimator.py):
//     SR (primary engine) + Shoegazer SR means per stage
//     → boundary interpolation → dp_shoegazer (1.0–12.0) → stage
//
// Stage order (weakest → strongest):
//     1st, 2nd, 3rd, 4th, 5th, 6th, 7th, 8th, 9th, 10th, Luminal, Tachyon
// dp_shoegazer: 1.0 (1st Dan) → 12.0 (Tachyon)
import * as py from './py.js';
import { path, sys, readText } from './vfs.js';

// The overlay always runs "frozen": resources live under sys._MEIPASS (vfs.RES_ROOT).
export const _PROFILES_PATH = path.join(sys._MEIPASS, 'config', 'shoegazer_profiles.json');

export const STAGE_KEYS = [
  '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', 'Luminal', 'Tachyon',
];

export const _STAGE_DISPLAY = {
  '1st': '1st Dan',
  '2nd': '2nd Dan',
  '3rd': '3rd Dan',
  '4th': '4th Dan',
  '5th': '5th Dan',
  '6th': '6th Dan',
  '7th': '7th Dan',
  '8th': '8th Dan',
  '9th': '9th Dan',
  '10th': '10th Dan',
  Luminal: 'Luminal',
  Tachyon: 'Tachyon',
};

// Compact labels
export const _STAGE_SHORT = {
  '1st': '1',
  '2nd': '2',
  '3rd': '3',
  '4th': '4',
  '5th': '5',
  '6th': '6',
  '7th': '7',
  '8th': '8',
  '9th': '9',
  '10th': '10',
  Luminal: '☆',
  Tachyon: 'ε',
};

export const SKILLSET_KEYS = [
  'stream', 'jumpstream', 'handstream', 'stamina',
  'jackspeed', 'chordjack', 'technical',
];

export const _STAGE_DP = Object.fromEntries(STAGE_KEYS.map((key, i) => [key, i + 1]));

export let _profiles = null;
export let _ruler_cache = null;
export let _boundaries_cache = null;

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
 * Build the 12-slot SR ruler from profiles.
 *
 * Returns sorted list of [sr_mean, dp_int] where dp_int is 1-12, or null if profiles lack
 * sr_mean data. Enforces strict monotonicity to handle calibration artifacts.
 */
export function _build_sr_ruler() {
  if (_ruler_cache !== null) return _ruler_cache;

  const profiles = _load_profiles();
  if (!py.truthy(profiles)) return null;

  let ruler = [];
  for (const key of STAGE_KEYS) {
    const slot = py.get(profiles, key);
    if (!py.truthy(slot)) continue;
    const sr = py.get(slot, 'sr_mean');
    if (sr === null || sr <= 0) continue;
    const dp_int = py.getitem(_STAGE_DP, key);
    ruler.push([py.float(sr), dp_int]);
  }

  if (ruler.length < 2) return null;

  // Sort by stage, then enforce strict monotonicity
  ruler = py.sortedBy(ruler, (x) => x[1]);
  for (let i = 1; i < ruler.length; i++) {
    if (ruler[i][0] <= ruler[i - 1][0]) ruler[i] = [ruler[i - 1][0] + 0.001, ruler[i][1]];
  }

  _ruler_cache = ruler;

  // Boundaries: midpoints between adjacent means, extrapolated at both ends
  const means = ruler.map(([sr]) => sr);
  const n = means.length;
  const boundaries = [];
  for (let i = 0; i < n; i++) {
    const lower = i > 0 ? (means[i - 1] + means[i]) / 2.0 : means[0] - (n > 1 ? (means[1] - means[0]) / 2.0 : 1.0);
    const upper = i < n - 1 ? (means[i] + means[i + 1]) / 2.0 : means[n - 1] + (n > 1 ? (means[n - 1] - means[n - 2]) / 2.0 : 1.0);
    boundaries.push([lower, upper, ruler[i][1]]);
  }
  _boundaries_cache = boundaries;

  return _ruler_cache;
}

// ── SR → dp ──────────────────────────────────────────────────────────

/**
 * Convert SR → dp_shoegazer (1.0–12.0+) via boundary interpolation. Each stage occupies
 * dp_int ∈ [1, 12]; the fractional part t ∈ [0, 1) is the position within that stage.
 */
export function _sr_to_dp(sr) {
  const ruler = _build_sr_ruler();
  if (!py.truthy(ruler)) return null;

  const boundaries = _boundaries_cache;
  if (!py.truthy(boundaries)) return null;

  // Below the first stage
  if (sr < boundaries[0][0]) return py.max2(0.5, py.float(boundaries[0][2]));

  // Above the last stage
  const last = boundaries[boundaries.length - 1];
  if (sr >= last[1]) return py.min2(12.99, py.float(last[2]) + 0.99);

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

/**
 * Map a dp float (1.0-12.0+) to the stage key: dp ∈ [1.0, 2.0) → 1st, [2.0, 3.0) → 2nd, …
 */
export function _dp_to_stage_key(dp) {
  let idx = py.min2(py.int(dp) - 1, STAGE_KEYS.length - 1);
  idx = py.max2(0, idx);
  return STAGE_KEYS[idx];
}

// ── MSD fallback ─────────────────────────────────────────────────────

/** Flat MSD distance fallback when SR is unavailable. */
export function _estimate_from_msd(skillsets) {
  const profiles = _load_profiles();
  if (!py.truthy(profiles)) return null;

  const vals = SKILLSET_KEYS.map((k) => {
    const v = py.get(skillsets, k, 0.0);
    return py.float(py.truthy(v) ? v : 0.0);
  });
  if (!py.truthy(vals) || py.max(vals) < 1.0) return null;
  const overall = py.max(vals);

  const msd_ruler = [];
  for (const key of STAGE_KEYS) {
    const slot = py.get(profiles, key, {});
    const msd_val = py.get(slot, 'overall_msd');
    if (py.truthy(msd_val) && msd_val > 0) msd_ruler.push([py.float(msd_val), py.getitem(_STAGE_DP, key)]);
  }

  if (msd_ruler.length < 2) return null;

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

export function _confidence_from_dp_frac(dp) {
  const frac = dp - py.int(dp);
  const dist = Math.abs(frac - 0.5);
  const conf = 0.5 * (1.0 + py.math.cos(py.math.pi * dist / 0.5));
  return py.max2(0.0, py.min2(1.0, conf));
}

// ── Result ───────────────────────────────────────────────────────────

/** @dataclass ShoegazerResult */
export class ShoegazerResult {
  constructor(stage_key, label, short, confidence, dp_shoegazer, beyond = false) {
    this.stage_key = stage_key;
    this.label = label;
    this.short = short;
    this.confidence = confidence;
    this.dp_shoegazer = dp_shoegazer;
    this.beyond = beyond;
  }

  to_dict() {
    return {
      stage_key: this.stage_key,
      label: this.label,
      short: this.short,
      confidence: py.round(this.confidence, 4),
      dp_shoegazer: py.round(this.dp_shoegazer, 3),
      beyond: this.beyond,
    };
  }
}

// ── Public API ───────────────────────────────────────────────────────

/**
 * Re-derive Shoegazer result fields from a ``dp_shoegazer`` value (used by the pipeline's
 * custom-rate interpolation to refresh the stage/label fields).
 */
export function fields_from_dp(dp) {
  const beyond = dp > 12.99;
  dp = py.max2(1.0, dp);
  const stage_key = _dp_to_stage_key(dp);
  return {
    stage_key,
    label: py.getitem(_STAGE_DISPLAY, stage_key),
    short: py.getitem(_STAGE_SHORT, stage_key),
    dp_shoegazer: py.round(dp, 3),
    beyond,
  };
}

/**
 * Estimate Shoegazer stage from MSD skillsets and optional SR (SR ruler first, MSD distance
 * fallback). Returns a ShoegazerResult or null.
 */
export function estimate(skillsets, sr = null) {
  let dp = null;

  // Primary: SR ruler
  if (sr !== null && sr !== undefined && sr > 0.0) dp = _sr_to_dp(sr);

  // Fallback: MSD
  if (dp === null && py.truthy(skillsets)) dp = _estimate_from_msd(skillsets);

  if (dp === null) return null;

  const beyond = dp > 12.99;
  dp = py.max2(1.0, dp);

  const stage_key = _dp_to_stage_key(dp);
  const label = py.getitem(_STAGE_DISPLAY, stage_key);
  const short = py.getitem(_STAGE_SHORT, stage_key);
  const confidence = _confidence_from_dp_frac(dp);

  return new ShoegazerResult(stage_key, label, short, confidence, py.round(dp, 3), beyond);
}
