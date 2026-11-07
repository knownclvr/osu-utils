// celestial_estimator.js — Dan Celestial tier+category estimator (SR ruler + MSD fallback).
//
// JavaScript port of engine/dan_engine/celestial_estimator.py (DanOverlay 3.1.0 by 8doul).
//
// Architecture (mirrors rank_engine.py Reform):
//     SR (primary engine) + celestial SR means per slot
//     → boundary interpolation → dp_celestial (1.0–35.0) → (tier, category)
//     MSD skillsets are retained as a fallback when SR is unavailable.
//
// Tier order (weakest → strongest):
//     Beginner, Intermediate, Expert, Mastery, Ascension, Transcendence, Singularity
// Categories per tier: I < II < III < IV < V
// dp_celestial: 1.0 (Beginner I) → 35.0 (Singularity V)
import * as py from './py.js';
import { path, sys, readText } from './vfs.js';

// The overlay always runs "frozen": resources live under sys._MEIPASS (vfs.RES_ROOT).
// (The non-frozen branch of the Python module, based on _ROOT/__file__, does not apply.)
export const _PROFILES_PATH = path.join(sys._MEIPASS, 'config', 'celestial_profiles.json');

export const TIERS = ['Beginner', 'Intermediate', 'Expert', 'Mastery', 'Ascension', 'Transcendence', 'Singularity'];
export const CATEGORIES = ['I', 'II', 'III', 'IV', 'V'];

export const _TIER_SHORT = {
  Beginner: 'B',
  Intermediate: 'Int',
  Expert: 'E',
  Mastery: 'M',
  Ascension: 'A',
  Transcendence: 'T',
  Singularity: 'S',
};

export const _CAT_TO_TIER = { I: 1, II: 2, III: 3, IV: 4, V: 5 };

// 35 slots in order (Beginner I … Singularity V); dp_int = index + 1
export const _SLOTS = [];
for (const t of TIERS) for (const c of CATEGORIES) _SLOTS.push([t, c]);

/**
 * Key of a (tier, category) tuple in _SLOT_INDEX. Python keys that dict by tuple value;
 * JS Maps compare arrays by identity, so the tuple is keyed as 'tier|category'.
 */
export const _slot_key = (tier, cat) => `${tier}|${cat}`;
export const _SLOT_INDEX = new Map(_SLOTS.map(([t, c], i) => [_slot_key(t, c), i + 1]));

export const SKILLSET_KEYS = ['stream', 'jumpstream', 'handstream', 'stamina', 'jackspeed', 'chordjack', 'technical'];

export let _profiles = null;
export const _ruler_caches = new Map(); // chart_type (str | null) → ruler | null
export const _boundaries_caches = new Map(); // chart_type (str | null) → boundaries | null

// classifier family → celestial chart type (None = general ruler)
export const _FAMILY_TO_TYPE = {
  jack: 'jack', speed: 'speed', stamina: 'stamina', tech: 'tech',
  stream: null, hybrid: null,
};

// Weight of the type-specific SR vs the general SR in typed rulers
// (1.0 = type-specific value only, no shrinkage toward the general mean).
export const _SHRINKAGE = 1.0;

/** float(v or 0.0) */
const _num = (v) => py.float(py.truthy(v) ? v : 0.0);

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
 * Build 35-slot SR ruler from profiles.
 *
 * When chart_type is provided, uses sr_by_type for that type (with shrinkage toward the
 * general mean), falling back to sr_mean where the type is absent. When chart_type is
 * None, uses the general sr_mean. Returns list of [sr_mean, dp_int] sorted by dp (1–35), or null.
 */
export function _build_ruler(chart_type = null) {
  if (_ruler_caches.has(chart_type)) return _ruler_caches.get(chart_type);

  const profiles = _load_profiles();
  if (!py.truthy(profiles)) return null;

  // General ruler: fallback / shrinkage target for typed rulers
  const general = chart_type !== null ? _build_ruler(null) : null;
  const general_map = new Map();
  for (const [sr, dp] of (py.truthy(general) ? general : [])) general_map.set(dp, sr);

  const table = [];
  for (const [tier, cat] of _SLOTS) {
    const slot = py.get(py.get(profiles, tier, {}), cat);
    const dp_int = py.getitem(_SLOT_INDEX, _slot_key(tier, cat));
    let sr;

    if (chart_type === null) {
      if (!py.truthy(slot) || !py.truthy(py.get(slot, 'sr_mean'))) {
        _ruler_caches.set(chart_type, null);
        _boundaries_caches.set(chart_type, null);
        return null;
      }
      sr = py.float(py.getitem(slot, 'sr_mean'));
    } else {
      let type_sr = null;
      if (py.truthy(slot)) {
        const by_type = py.get(slot, 'sr_by_type');
        type_sr = py.get(py.truthy(by_type) ? by_type : {}, chart_type);
      }
      const gen_sr = py.get(general_map, dp_int);
      if (type_sr !== null && gen_sr !== null) {
        sr = _SHRINKAGE * py.float(type_sr) + (1.0 - _SHRINKAGE) * gen_sr;
      } else if (type_sr !== null) {
        sr = py.float(type_sr);
      } else if (gen_sr !== null) {
        sr = gen_sr;
      } else {
        _ruler_caches.set(chart_type, null);
        _boundaries_caches.set(chart_type, null);
        return null;
      }
    }

    table.push([sr, dp_int]);
  }

  // Enforce strict monotonicity
  for (let i = 1; i < table.length; i++) {
    if (table[i][0] <= table[i - 1][0]) table[i] = [table[i - 1][0] + 0.001, table[i][1]];
  }

  // Boundaries: midpoints between adjacent means, extrapolated at both ends
  const means = table.map(([sr]) => sr);
  const n = means.length;
  const boundaries = [];
  for (let i = 0; i < n; i++) {
    const lower = i > 0 ? (means[i - 1] + means[i]) / 2.0 : means[0] - (n > 1 ? (means[1] - means[0]) / 2.0 : 1.0);
    const upper = i < n - 1 ? (means[i] + means[i + 1]) / 2.0 : means[n - 1] + (n > 1 ? (means[n - 1] - means[n - 2]) / 2.0 : 1.0);
    boundaries.push([lower, upper, table[i][1]]);
  }

  _ruler_caches.set(chart_type, table);
  _boundaries_caches.set(chart_type, boundaries);
  return table;
}

/**
 * Convert primary SR → dp_celestial (1.0–35.0) via boundary interpolation.
 * Uses chart_type-specific ruler when available, general otherwise.
 */
export function _sr_to_dp_celestial(sr, chart_type = null) {
  let resolved_type = chart_type;
  let ruler = _build_ruler(resolved_type);
  if (ruler === null && resolved_type !== null) {
    resolved_type = null;
    ruler = _build_ruler(null);
  }
  if (ruler === null) return null;

  const boundaries = py.get(_boundaries_caches, resolved_type);
  if (!py.truthy(boundaries)) return null;

  // Below the first slot
  if (sr < boundaries[0][0]) return py.max2(0.5, py.float(boundaries[0][2]));

  // Above the last slot
  const last = boundaries[boundaries.length - 1];
  if (sr >= last[1]) return py.min2(35.99, py.float(last[2]) + 0.99);

  // Interpolate within the matching slot
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
 * Convert a dp_celestial value → [tier, category].
 * int(dp) ensures the slot matches the integer part: dp ∈ [1.0, 2.0) → Beginner I, …
 */
export function _dp_to_slot(dp) {
  const idx = py.max2(0, py.min2(34, py.int(dp) - 1));
  return _SLOTS[idx];
}

/**
 * Confidence from fractional position within slot boundary (raised cosine):
 * frac=0.0 / 1.0 → edge (~0.5 conf), frac=0.5 → centre (1.0 conf).
 */
export function _confidence_from_frac(frac) {
  const conf = 0.5 * (1.0 + py.math.cos(py.math.pi * (frac - 0.5) * 2.0));
  return py.round(py.max2(0.1, py.min2(1.0, conf)), 3);
}

// ── MSD fallback ─────────────────────────────────────────────────────

/** Weighted RMS distance: map MSD vs profile slot (MSD-based fallback). */
export function _weighted_distance(msd, profile) {
  const prof_skillsets = py.get(profile, 'skillsets', {});
  const prof_overall = _num(py.get(profile, 'overall_msd', 0.0));
  if (prof_overall <= 0.0) return Infinity;

  const map_overall = py.max(SKILLSET_KEYS.map((k) => _num(py.get(msd, k, 0.0))), undefined, 0.0);
  const overall_dist = py.pow(map_overall - prof_overall, 2) * 4.0;

  let skill_dist = 0.0;
  let total_weight = 0.0;
  for (const k of SKILLSET_KEYS) {
    const pv = _num(py.get(prof_skillsets, k, 0.0));
    const mv = _num(py.get(msd, k, 0.0));
    const w = pv / py.max2(prof_overall, 0.001);
    skill_dist += w * py.pow(mv - pv, 2);
    total_weight += w;
  }
  if (total_weight > 0) skill_dist /= total_weight;

  return py.math.sqrt(py.max2(0.0, overall_dist + skill_dist));
}

export function _confidence_from_distance(dist, bandwidth) {
  if (bandwidth <= 0.001) bandwidth = 1.0;
  const conf = py.math.exp(-py.pow(dist / py.max2(bandwidth, 0.001), 2));
  return py.round(py.max2(0.0, py.min2(1.0, conf)), 3);
}

/** Original MSD distance-matching fallback — used when SR is unavailable. */
export function _estimate_msd_fallback(msd) {
  const profiles = _load_profiles();
  if (!py.truthy(profiles)) return null;

  const map_overall = py.max(SKILLSET_KEYS.map((k) => _num(py.get(msd, k, 0.0))), undefined, 0.0);
  if (map_overall < 0.5) return null;

  let best_slot = null;
  let best_dist = Infinity;

  for (const tier of TIERS) {
    const tier_data = py.get(profiles, tier, {});
    for (const cat of CATEGORIES) {
      const slot = py.get(tier_data, cat);
      if (slot === null) continue;
      const dist = _weighted_distance(msd, slot);
      if (dist < best_dist) {
        best_dist = dist;
        best_slot = [tier, cat, slot];
      }
    }
  }

  if (best_slot === null) return null;

  const [tier, cat, slot] = best_slot;
  const bandwidth = py.float(py.get(slot, 'upper', 0.0)) - py.float(py.get(slot, 'lower', 0.0));
  const conf = _confidence_from_distance(best_dist, bandwidth);

  // Fractional dp inside the matched slot
  const prof_overall = _num(py.get(slot, 'overall_msd', 0.0));
  let dp = py.float(py.get(_SLOT_INDEX, _slot_key(tier, cat), 1));
  if (prof_overall > 0.0) {
    const bw = py.float(py.get(slot, 'upper', prof_overall)) - py.float(py.get(slot, 'lower', prof_overall));
    const diff = (py.max(SKILLSET_KEYS.map((k) => _num(py.get(msd, k, 0.0))), undefined, 0.0) - prof_overall) / py.max2(bw, 0.001);
    dp += py.max2(-0.5, py.min2(0.5, diff * 0.5));
  }

  const short = `${py.getitem(_TIER_SHORT, tier)}-${cat}`;
  const label = `${tier} ${cat}`;
  return new CelestialResult(tier, cat, short, label, conf, py.round(dp, 3));
}

// ── Result ───────────────────────────────────────────────────────────

/** @dataclass CelestialResult */
export class CelestialResult {
  constructor(tier, category, short_label, label, confidence, dp_celestial, beyond = false) {
    this.tier = tier;
    this.category = category;
    this.short_label = short_label;
    this.label = label;
    this.confidence = confidence;
    this.dp_celestial = dp_celestial;
    this.beyond = beyond;
  }

  to_dict() {
    return {
      tier: this.tier,
      category: this.category,
      short: this.short_label,
      label: this.label,
      confidence: py.round(this.confidence, 3),
      dp_celestial: py.round(this.dp_celestial, 3),
      beyond: this.beyond,
    };
  }
}

// ── Custom-rate field refresh ────────────────────────────────────────

/**
 * Re-derive Celestial result fields from a ``dp_celestial`` value (used by the pipeline's
 * custom-rate interpolation to refresh the tier/category/label fields).
 */
export function fields_from_dp(dp) {
  const beyond = dp > 35.99;
  const dp_clamped = py.max2(0.5, py.min2(35.99, dp));
  const slot = _dp_to_slot(dp_clamped);
  if (slot === null || slot === undefined) return {};
  const [tier, cat] = slot;
  return {
    tier,
    category: cat,
    short: `${py.getitem(_TIER_SHORT, tier)}-${cat}`,
    label: `${tier} ${cat}`,
    dp_celestial: py.round(dp_clamped, 3),
    beyond,
  };
}

/**
 * Estimate Dan Celestial tier + category.
 *
 * Primary path (when SR available): SR → boundary interpolation on the 35-slot ruler
 * → dp_celestial → (tier, category). Fallback (SR unavailable or profiles lack sr_mean):
 * MinaCalc MSD skillsets → weighted-distance matching → (tier, category).
 * Returns a CelestialResult or null.
 */
export function estimate(skillsets, sr = null, family_hint = null) {
  const src = py.truthy(skillsets) ? skillsets : {};
  const msd = {};
  for (const k of SKILLSET_KEYS) msd[k] = _num(py.get(src, k, 0.0));
  let chart_type = py.truthy(family_hint) ? py.get(_FAMILY_TO_TYPE, family_hint) : null;

  // ── Primary: SR ruler ──
  if (sr !== null && sr !== undefined && sr > 0.0) {
    let ruler = _build_ruler(chart_type);
    if (ruler === null && chart_type !== null) {
      chart_type = null;
      ruler = _build_ruler(null);
    }
    if (ruler !== null) {
      const dp_raw = _sr_to_dp_celestial(sr, chart_type);
      if (dp_raw !== null) {
        const beyond = dp_raw > 35.99;
        const dp_clamped = py.max2(0.5, py.min2(35.99, dp_raw));
        const [tier, cat] = _dp_to_slot(dp_clamped);

        // Confidence from the position inside the slot's SR boundaries
        const resolved_type = py.get(_ruler_caches, chart_type) !== null ? chart_type : null;
        const boundaries = py.get(_boundaries_caches, resolved_type, []);
        const slot_dp = py.getitem(_SLOT_INDEX, _slot_key(tier, cat));
        let conf;
        if (slot_dp - 1 < py.len(boundaries)) {
          const [low_sr, high_sr] = boundaries[slot_dp - 1];
          const slot_width = py.max2(high_sr - low_sr, 0.001);
          const frac = (sr - low_sr) / slot_width;
          conf = _confidence_from_frac(py.max2(0.0, py.min2(1.0, frac)));
        } else {
          conf = 0.5;
        }

        const short = `${py.getitem(_TIER_SHORT, tier)}-${cat}`;
        const label = `${tier} ${cat}`;
        return new CelestialResult(tier, cat, short, label, conf, py.round(dp_clamped, 3), beyond);
      }
    }
  }

  // ── Fallback: MSD distance matching ──
  return _estimate_msd_fallback(msd);
}

export const _RULER_PATH = path.join(sys._MEIPASS, 'config', 'celestial_ruler.json');

export let _isor_celestial_boundaries = null;

export function _load_isor_celestial_ruler() {
  if (_isor_celestial_boundaries !== null) return _isor_celestial_boundaries;

  if (!path.isfile(_RULER_PATH)) return null;

  let means_clean;
  try {
    const data = JSON.parse(readText(_RULER_PATH));
    const means = py.get(data, 'slot_means', []);
    if (!py.truthy(means) || py.len(means) !== 35) return null;
    means_clean = Array.from(means, (m) => py.float(m));
  } catch {
    // except Exception
    return null;
  }

  const n = means_clean.length;
  const boundaries = [];
  for (let i = 0; i < n; i++) {
    let lo;
    if (i > 0) lo = (means_clean[i - 1] + means_clean[i]) / 2.0;
    else lo = means_clean[0] - (n > 1 ? (means_clean[1] - means_clean[0]) / 2.0 : 1.0);

    let hi;
    if (i < n - 1) hi = (means_clean[i] + means_clean[i + 1]) / 2.0;
    else hi = means_clean[n - 1] + (n > 1 ? (means_clean[n - 1] - means_clean[n - 2]) / 2.0 : 1.0);

    boundaries.push([lo, hi, i + 1]);
  }

  _isor_celestial_boundaries = boundaries;
  return _isor_celestial_boundaries;
}

/** Estimate Dan Celestial tier + category directly from ISOR continuous DP using celestial_ruler.json. */
export function estimate_from_isor_dp(dp) {
  const boundaries = _load_isor_celestial_ruler();
  if (!py.truthy(boundaries)) return null;

  const dp_val = py.float(dp);
  let t = 0.5;
  let dp_celestial;
  let beyond;
  const last = boundaries[boundaries.length - 1];
  if (dp_val < boundaries[0][0]) {
    dp_celestial = 1.0;
    beyond = false;
    t = 0.0;
  } else if (dp_val >= last[1]) {
    dp_celestial = py.min2(35.99, 35.0 + (dp_val - last[1]) / 2.0);
    beyond = true;
    t = 1.0;
  } else {
    dp_celestial = 1.0;
    beyond = false;
    for (const [lo, hi, slot_num] of boundaries) {
      if (lo <= dp_val && dp_val < hi) {
        const width = py.max2(hi - lo, 1e-6);
        t = (dp_val - lo) / width;
        dp_celestial = py.float(slot_num) + t;
        beyond = false;
        break;
      }
    }
  }

  const dp_clamped = py.max2(0.5, py.min2(35.99, dp_celestial));
  const slot = _dp_to_slot(dp_clamped);
  if (slot === null || slot === undefined) return null;

  const [tier, cat] = slot;
  const conf = _confidence_from_frac(py.max2(0.0, py.min2(1.0, t)));
  const short = `${py.getitem(_TIER_SHORT, tier)}-${cat}`;
  const label = `${tier} ${cat}`;
  return new CelestialResult(tier, cat, short, label, conf, py.round(dp_clamped, 3), beyond);
}
