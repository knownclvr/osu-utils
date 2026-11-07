// validator.js — domain validation for parsed charts (4K/7K rice, LN routing).
//
// JavaScript port of engine/dan_engine/validator.py (DanOverlay 3.1.0 by 8doul).
import * as py from './py.js';

// LN ratio thresholds for per-map mode routing (see _ln_route).
export const _LN_ROUTE_HYBRID_FLOOR = 0.30;
export const _LN_ROUTE_LN_FLOOR = 0.45;

/** Validates a parsed chart and produces a structured domain result. */
export function validate_domain(parsed) {
  const warnings = [...py.get(parsed, 'warnings', [])];

  const mode = py.get(parsed, 'mode', -1);
  const keycount = py.get(parsed, 'keycount_native', 0);
  const note_count = py.get(parsed, 'note_count', 0);
  const ln_ratio = py.get(parsed, 'ln_ratio', 0.0);
  const drain_time_s = py.get(parsed, 'drain_time_s', 0.0);
  const bpm = py.get(parsed, 'bpm', 0.0);
  const od = py.get(parsed, 'od', 0.0);
  const hp = py.get(parsed, 'hp', 0.0);
  const rejected = py.get(parsed, 'rejected', false);

  const is_4k = keycount === 4;
  const is_7k = keycount === 7;
  let rejection_reason = null;

  // Parser-level rejection
  if (py.truthy(rejected)) {
    rejection_reason = `parser rejected: mode=${mode}, keycount=${keycount}`;
  }
  // Mode check
  if (mode !== 3) {
    rejection_reason = rejection_reason || `not mania mode (mode=${mode})`;
  }
  // Keycount check
  if (!(is_4k || is_7k)) {
    rejection_reason = rejection_reason || `not 4K or 7K (keycount=${keycount})`;
  }
  // Minimum note count
  if (note_count < 20) {
    rejection_reason = rejection_reason || `too few notes (${note_count})`;
  }
  // Minimum drain time
  if (drain_time_s < 5.0) {
    rejection_reason = rejection_reason || `drain too short (${py.format(drain_time_s, '.1f')}s)`;
  }

  // LN domain band
  const ln_confidence = _ln_confidence_band(ln_ratio);
  if (ln_confidence === 'out_of_domain') {
    warnings.push(`LN ratio ${py.format(ln_ratio, '.1%')} — out of rice domain`);
  }

  // Mode routing (rice / hybrid / ln)
  const ln_route = _ln_route(ln_ratio);
  const valid = rejection_reason === null;

  return {
    valid,
    is_4k,
    is_7k,
    ln_ratio,
    ln_confidence,
    ln_route,
    note_count,
    drain_time_s,
    bpm,
    od,
    hp,
    warnings,
    rejection_reason,
  };
}

/** Classifies the rice domain confidence based on the LN ratio. */
export function _ln_confidence_band(ln_ratio) {
  if (ln_ratio <= 0.10) return 'high';
  if (ln_ratio <= 0.30) return 'degraded';
  if (ln_ratio <= 0.45) return 'gray';
  return 'out_of_domain';
}

/** Returns a 0-1 multiplier associated with the LN band. */
export function confidence_multiplier(ln_confidence) {
  return py.get({ high: 1.0, degraded: 0.75, gray: 0.45, out_of_domain: 0.15 }, ln_confidence, 0.0);
}

/** Decide if this map should auto-route to LN Course mode. */
export function _ln_route(ln_ratio) {
  if (ln_ratio <= _LN_ROUTE_HYBRID_FLOOR) return 'rice';
  if (ln_ratio <= _LN_ROUTE_LN_FLOOR) return 'hybrid';
  return 'ln';
}
