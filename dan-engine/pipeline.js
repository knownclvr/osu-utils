// pipeline.js — analysis orchestration: parse → estimators → rank → overlay payload dict.
//
// JavaScript port of engine/dan_engine/pipeline.py (DanOverlay 3.1.0 by 8doul).
//
// analyze_map() is the entry point (the overlay's Web Worker calls it the way
// analysis_coordinator does in Python); the dict it returns is consumed by
// contracts.AnalysisResult.from_pipeline(), so every key and nesting is kept as is.
// .sm/.ssc charts need the `etterna` package, which the Python engine doesn't ship
// either: both builds answer them with "parse_error: No module named 'etterna'".
//
// Python's function-level imports are static imports here; everything is synchronous.
// The Python module also imports rank_engine._msd_to_family / recompute_dp_for_family
// inside _merge_primary_and_mina without ever calling them — not imported here.
import * as py from './py.js';
import * as vfs from './vfs.js';
import { resource_path } from './resource_path.js';
import { parsear_osu_v2 } from './parser.js';
import { validate_domain } from './validator.js';
import { extract_features } from './feature_extractor.js';
import { analyze_primary_sr } from './primary_sr_bridge.js';
import { classify_family } from './classifier.js';
import { classify_from_parsed } from './rhythm_profile.js';
import { compute_rank, sr_to_dp, dp_to_label, dp_to_sublevel } from './rank_engine.js';
import { analyze_beatmap_isor } from './isor_engine.js';
import { estimate as _minacalc_estimate } from './minacalc_estimator.js';
import {
  estimate as _celestial_estimate,
  fields_from_dp as _celestial_fields_from_dp,
  estimate_from_isor_dp as _celestial_from_isor_dp,
} from './celestial_estimator.js';
import { estimate as _signicial_estimate, fields_from_dp as _signicial_fields_from_dp } from './signicial_estimator.js';
import { estimate as _shoegazer_estimate, fields_from_dp as _shoegazer_fields_from_dp } from './shoegazer_estimator.js';
import { estimate as _ln_course_estimate, fields_from_dp as _ln_course_fields_from_dp } from './ln_course_estimator.js';

// Module-level names of the Python module (its `from X import estimate as _x_estimate` lines).
export {
  _minacalc_estimate,
  _celestial_estimate,
  _celestial_fields_from_dp,
  _signicial_estimate,
  _signicial_fields_from_dp,
  _shoegazer_estimate,
  _shoegazer_fields_from_dp,
  _ln_course_estimate,
  _ln_course_fields_from_dp,
};

// _ROOT = os.path.dirname(os.path.abspath(__file__)) — only used by Python for its sys.path
// setup (same virtual location as isor_engine._THIS_DIR).
export const _ROOT = '/engine/dan_engine';

// ── Python semantics used throughout this module ─────────────────────
/** Python `v is None` (undefined coming from JS callers counts as None). */
const isNone = (v) => v === null || v === undefined;
/** Python `v or d` (d is evaluated eagerly — only used with side-effect-free defaults). */
const orv = (v, d) => (py.truthy(v) ? v : d);
/** isinstance(v, dict): a plain object or a Map. */
function isDict(v) {
  if (v === null || typeof v !== 'object') return false;
  if (v instanceof Map) return true;
  const proto = Object.getPrototypeOf(v);
  return proto === Object.prototype || proto === null;
}
/** dict(d): shallow copy of a dict (Map stays a Map). */
function dictCopy(d) {
  if (d instanceof Map) return new Map(d);
  if (isDict(d)) return { ...d };
  if (Array.isArray(d)) {
    const pairs = d.map((kv) => [kv[0], kv[1]]);
    if (pairs.every(([k]) => typeof k === 'string')) return Object.fromEntries(pairs);
    return new Map(pairs);
  }
  throw new py.TypeError_(`'${typeName(d)}' object is not iterable`);
}
/** dict.update(src) */
function dictUpdate(target, src) {
  const entries = src instanceof Map ? src.entries() : Object.entries(dictCopy(src));
  for (const [k, v] of entries) {
    if (target instanceof Map) target.set(k, v);
    else target[k] = v;
  }
}
/** dict.setdefault(k, dflt) */
function setdefault(d, k, dflt) {
  if (d instanceof Map) {
    if (!d.has(k)) d.set(k, dflt);
    return d.get(k);
  }
  if (!Object.prototype.hasOwnProperty.call(d, k)) d[k] = dflt;
  return d[k];
}
/** list(x) */
function pyList(x) {
  if (x === null || x === undefined) throw new py.TypeError_("'NoneType' object is not iterable");
  if (x instanceof Map) return [...x.keys()];
  if (typeof x === 'object' && !(Symbol.iterator in x)) return Object.keys(x);
  return Array.from(x);
}
/** list + list */
function listAdd(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b)) {
    throw new py.TypeError_(`can only concatenate list (not "${typeName(b)}") to list`);
  }
  return [...a, ...b];
}
function typeName(v) {
  if (v === null || v === undefined) return 'NoneType';
  if (typeof v === 'number') return Number.isInteger(v) ? 'int' : 'float';
  if (typeof v === 'string') return 'str';
  if (typeof v === 'boolean') return 'bool';
  if (Array.isArray(v)) return 'list';
  return isDict(v) ? 'dict' : 'object';
}
/** str(v) for the values this module stringifies (str / None / bool; numbers never occur). */
function pyStr(v) {
  if (v === null || v === undefined) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  if (typeof v === 'string') return v;
  if (typeof v === 'number') return Number.isInteger(v) ? py.reprInt(v) : py.reprFloat(v);
  return String(v);
}
/** str(exc) — KeyError prints the repr of its key. */
function excStr(e) {
  if (e instanceof py.KeyError) return `'${e.message}'`;
  if (e instanceof Error) return e.message;
  return String(e);
}
/**
 * float * rate. The caller's rate is used as is (rate if rate is not None else …), so Python
 * raises TypeError when it is not a number (e.g. passed as a string) — JS would coerce it.
 */
function mulRate(x, rate) {
  if (typeof rate !== 'number' && typeof rate !== 'boolean') {
    throw new py.TypeError_(typeof rate === 'string' || Array.isArray(rate)
      ? "can't multiply sequence by non-int of type 'float'"
      : `unsupported operand type(s) for *: 'float' and '${typeName(rate)}'`);
  }
  return x * rate;
}
/** `a < b` raising TypeError on None like Python. */
function lt(a, b) {
  if (isNone(a) || isNone(b)) {
    throw new py.TypeError_(`'<' not supported between instances of '${typeName(a)}' and '${typeName(b)}'`);
  }
  return a < b;
}
/** Python's ModuleNotFoundError (raised for the missing `etterna` package). */
class ModuleNotFoundError extends py.PyError {}

// ── thread pool ──────────────────────────────────────────────────────
/**
 * ThreadPoolExecutor(max_workers=2, thread_name_prefix="Pipeline-Worker") stand-in:
 * submit(fn, ...args) runs the task immediately (tasks therefore run in submit order);
 * .result() returns its value or re-raises its exception.
 */
export const _PIPELINE_POOL = {
  submit(fn, ...args) {
    let value = null;
    let error = null;
    let failed = false;
    try {
      value = fn(...args);
    } catch (e) {
      failed = true;
      error = e;
    }
    return {
      result() {
        if (failed) throw error;
        return value;
      },
    };
  },
};

export function _error_payload(error, warnings = null) {
  return {
    error,
    dp: null,
    dan_label: null,
    dan_short: null,
    sublevel: null,
    confidence: 0.0,
    family: null,
    primary_dan: null,
    primary_role: null,
    role_estimates: {},
    composite_dan: null,
    bottleneck_role: null,
    is_generalist: false,
    skillsets: {},
    role_scores: {},
    overall_msd: 0.0,
    duration_s: 0,
    note_count: 0,
    warnings: pyList(orv(warnings, [])),
    engine: null,
    corrections: [],
    ln_route: 'rice',
    ln_course: null,
  };
}

/**
 * Attach Sunny SR component strains to the debug dict for the ui-7 skin.
 *
 * contracts.to_dict() exposes jbar_max/pbar_max/xbar_max/abar_mean from
 * debug["sr_result"]; rank_engine's own debug dict does not carry them, so they
 * are injected here from the raw SR output.
 */
export function _inject_sunny_components(debug, sr_result) {
  const _debug = dictCopy(orv(debug, {}));
  const _raw = orv(sr_result, {});
  _debug.sr_result = {
    jbar_max: py.float(orv(py.get(_raw, 'jbar_max', 0.0), 0.0)),
    pbar_max: py.float(orv(py.get(_raw, 'pbar_max', 0.0), 0.0)),
    xbar_max: py.float(orv(py.get(_raw, 'xbar_max', 0.0), 0.0)),
    abar_mean: py.float(orv(py.get(_raw, 'abar_mean', 0.0), 0.0)),
  };
  return _debug;
}

/** Inject MSD ratios into the feature dict for rank_engine consumption. */
export function _augment_features_with_msd(features, mina_result) {
  const enriched = dictCopy(orv(features, {}));
  if (!isDict(mina_result)) return enriched;

  const overall_msd = py.float(orv(py.get(mina_result, 'overall_msd', 0.0), 0.0));
  const skillsets = dictCopy(orv(py.get(mina_result, 'skillsets'), {}));
  if (overall_msd <= 0.0 || !py.truthy(skillsets)) return enriched;

  const ratio_map = {
    handstream: 'msd_handstream_ratio',
    jackspeed: 'msd_jackspeed_ratio',
    chordjack: 'msd_chordjack_ratio',
    technical: 'msd_technical_ratio',
    stamina: 'msd_stamina_ratio',
    stream: 'msd_stream_ratio',
  };
  for (const [skill, feat_name] of Object.entries(ratio_map)) {
    const value = py.float(orv(py.get(skillsets, skill, 0.0), 0.0));
    if (value > 0.0) enriched[feat_name] = value / overall_msd;
  }

  enriched.msd_overall = overall_msd;
  if (py.truthy(py.get(mina_result, 'primary_role'))) {
    enriched.msd_primary_role = py.getitem(mina_result, 'primary_role');
  }
  return enriched;
}

/**
 * Primary SR path: SR engine -> per-skillset interpolation.
 *
 * MinaCalc data is injected into features for potential future use but does not
 * affect the primary DP calculation. When parsed, domain and features are supplied
 * by the caller, the redundant .osu parse is skipped.
 */
export function _compute_primary_rank_result(osu_path, mod = 'NM', strict_domain = false, mina_result = null,
  parsed = null, domain = null, features = null, rate = null) {
  const error = null;

  if (isNone(parsed)) {
    try {
      parsed = parsear_osu_v2(osu_path, null, true, true);
    } catch (exc) {
      return { error: `parse_error: ${excStr(exc)}`, dp: null, dan_label: null };
    }

    if (py.truthy(py.get(parsed, 'rejected'))) {
      return {
        error: 'domain_rejected',
        reason: 'non_4k_or_invalid',
        dp: null,
        dan_label: null,
        warnings: py.get(parsed, 'warnings', []),
      };
    }

    domain = validate_domain(parsed);
    if (py.truthy(strict_domain) && !py.truthy(py.get(domain, 'valid', true))) {
      return {
        error: 'domain_out_of_range',
        dp: null,
        dan_label: null,
        domain,
      };
    }

    features = extract_features(parsed);
  }

  features = _augment_features_with_msd(features, mina_result);

  const sr_result = analyze_primary_sr(osu_path, mod, rate);
  if (!py.truthy(py.get(sr_result, 'success', false))) {
    return {
      error: `primary_sr_error: ${pyStr(py.get(sr_result, 'error'))}`,
      dp: null,
      dan_label: null,
      dan_short: null,
      sublevel: null,
      confidence: 0.0,
      family: null,
      sr: 0.0,
      corrections: [],
      domain,
      features,
    };
  }

  // Family classification: below SR 7.0 the rhythm-profile classifier runs on the
  // parsed chart and falls back to the Sunny-based classify_family when its
  // confidence is < 0.1; from SR 7.0 up classify_family is always used.
  // compute_rank() then produces the DP / labels (MinaCalc skillsets are passed as msd=).
  const _sr_gate = py.float(orv(py.get(sr_result, 'sr', 0.0), 0.0));
  let classification;
  if (_sr_gate < 7.0) {
    classification = classify_from_parsed(parsed);
    if (lt(py.get(classification, 'confidence', 0.0), 0.1)) {
      classification = classify_family(sr_result, features, domain);
    }
  } else {
    classification = classify_family(sr_result, features, domain);
  }
  const rank = compute_rank(sr_result, features, classification, domain,
    orv(py.get(orv(mina_result, {}), 'skillsets'), {}));

  return {
    dp: py.getitem(rank, 'dp'),
    dan_label: py.getitem(rank, 'dan_label'),
    dan_short: py.getitem(rank, 'dan_short'),
    sublevel: py.getitem(rank, 'sublevel'),
    confidence: py.getitem(rank, 'confidence'),
    sr: py.getitem(rank, 'sr'),
    family: py.getitem(rank, 'family'),
    corrections: py.getitem(rank, 'corrections'),
    domain,
    features,
    primary_sr: sr_result,
    classification,
    debug: _inject_sunny_components(py.get(rank, 'debug', {}), sr_result),
    error,
    warnings: listAdd(py.get(parsed, 'warnings', []), py.get(domain, 'warnings', [])),
    duration_s: py.float(orv(py.get(domain, 'drain_time_s', 0.0), 0.0)),
    note_count: py.int(orv(py.get(domain, 'note_count', 0), 0)),
    peak_nps: py.float(orv(py.get(features, 'nps_p90', 0.0), 0.0)),
    nps_curve: py.get(features, 'nps_curve', []),
  };
}

/**
 * Merge primary SR ranking with MinaCalc data for UI display.
 *
 * The primary DP comes from the SR ruler path; MinaCalc skillsets, role estimates
 * and warnings are merged in, and a marathon duration penalty may lower the DP.
 */
export function _merge_primary_and_mina(primary_result, mina_result) {
  const merged = dictCopy(primary_result);
  merged.engine = 'sr_ruler_v3';

  if (!isDict(mina_result) || isNone(py.get(mina_result, 'dp'))) return merged;

  // warnings.extend(x for x in mina warnings if x not in warnings) — the generator sees
  // the items appended so far, so duplicates inside mina's list are dropped as well.
  const warnings = pyList(py.get(primary_result, 'warnings', []));
  for (const x of py.get(mina_result, 'warnings', [])) {
    if (!py.contains(warnings, x)) warnings.push(x);
  }

  const debug = dictCopy(orv(py.get(primary_result, 'debug'), {}));
  const mina_dp = py.float(orv(py.get(mina_result, 'dp', 0.0), 0.0));
  debug.mina_reference = {
    dp: py.round(mina_dp, 3),
    confidence: py.round(py.float(orv(py.get(mina_result, 'confidence', 0.0), 0.0)), 3),
    family: pyStr(orv(py.get(mina_result, 'family', ''), '')),
  };

  dictUpdate(merged, {
    warnings,
    debug,
    overall_msd: py.float(orv(py.get(mina_result, 'overall_msd', 0.0), 0.0)),
    primary_role: py.get(mina_result, 'primary_role'),
    primary_dan: py.get(mina_result, 'primary_dan'),
    role_estimates: orv(py.get(mina_result, 'role_estimates'), {}),
    composite_dan: py.get(mina_result, 'composite_dan'),
    bottleneck_role: py.get(mina_result, 'bottleneck_role'),
    is_generalist: py.truthy(py.get(mina_result, 'is_generalist', false)),
    role_breakdown_text: py.get(mina_result, 'role_breakdown_text', ''),
    skillsets: orv(py.get(mina_result, 'skillsets'), {}),
    role_scores: orv(py.get(mina_result, 'role_scores'), {}),
  });

  // Marathon duration penalty: for drain times over 300 s whose MSD skill groups are
  // balanced (no group above 45 % of the total), SR is reduced by
  // min(0.65, 0.08 per minute beyond 5 min), tapered linearly to 0 between SR 6.5 and
  // 7.0. DP / labels are only replaced when the re-derived DP is lower than the current one.
  const sunny_family = py.get(primary_result, 'family', 'hybrid');
  const msd_skillsets = orv(py.get(mina_result, 'skillsets'), {});
  const _dur_s = py.float(orv(py.get(primary_result, 'duration_s', 0.0), 0.0));
  if (_dur_s > 300.0 && py.truthy(msd_skillsets)) {
    const sk = (k) => py.float(orv(py.get(msd_skillsets, k, 0), 0));
    const _mj = py.max2(sk('jackspeed'), sk('chordjack'));
    const _ms = py.max2(sk('stream'), sk('jumpstream'));
    const _mt = sk('technical');
    const _mst = 0.7 * sk('stamina') + 0.3 * sk('handstream');
    const _tot = _mj + _ms + _mt + _mst;
    if (_tot > 1.0 && py.max2(_mj, _ms, _mt, _mst) / _tot < 0.45) {
      const _sr = py.float(py.get(merged, 'sr', 0.0));
      const _excess_min = (_dur_s - 300.0) / 60.0;
      const _raw_corr = py.min2(0.65, _excess_min * 0.08);
      let _taper;
      if (_sr >= 7.0) _taper = 0.0;
      else if (_sr >= 6.5) _taper = 1.0 - (_sr - 6.5) / 0.5;
      else _taper = 1.0;
      const _mara_corr = _raw_corr * _taper;
      if (_mara_corr > 0.005) {
        const _fam = orv(py.get(merged, 'family'), sunny_family);
        const _sk = py.get({ jack: 'jack', speed: 'speed', stamina: 'stamina', tech: 'tech' }, _fam);
        const _new_dp = py.round(sr_to_dp(_sr - _mara_corr, _sk), 2);
        if (_new_dp < py.float(py.get(merged, 'dp', 99.0))) {
          merged.dp = _new_dp;
          [merged.dan_label, merged.dan_short] = dp_to_label(_new_dp);
          merged.sublevel = dp_to_sublevel(_new_dp);
          const _corrs = pyList(orv(py.get(merged, 'corrections'), []));
          _corrs.push(`marathon_duration_penalty:-${py.format(_mara_corr, '.3f')}`);
          merged.corrections = _corrs;
        }
      }
    }
  }

  return merged;
}

// ── LN Course helpers ────────────────────────────────────────────────

/**
 * Lightweight heuristic: classify LN subfamily from structural features.
 * The family label is cosmetic only (tells the user what kind of LN map it is).
 */
export function _classify_ln_family(features) {
  const f = (k) => py.float(orv(py.get(features, k, 0), 0));
  const ln_ratio = f('ln_ratio');
  const sim_hold = f('simultaneous_hold');
  const release_dens = f('release_density');
  const ln_cv = f('ln_duration_cv');
  const hold_chord = f('hold_chord_ratio'); // computed but not used by the rules below
  void hold_chord;

  // Dense releases on an LN-heavy chart.
  if (release_dens > 3.5 && ln_ratio > 0.5) return 'speed_density';
  // Almost all LNs, many simultaneous holds.
  if (ln_ratio > 0.85 && sim_hold > 0.3) return 'inverse';
  // Mixed chart with irregular LN lengths.
  if (ln_ratio < 0.55 && ln_cv > 0.5) return 'jack_technical';
  // Default.
  return 'allround';
}

/**
 * Floor hi_res against lo_res so results never decrease with rate (labels are
 * re-derived from a floored DP). Not called by the pipeline itself.
 */
export function _monotonic_floor(lo_res, hi_res, dp_to_label, dp_to_sublevel, fields_fns, dp_key) {
  const hi = dictCopy(hi_res);
  let dp_floored = false;
  for (const _f of ['dp', 'sr']) {
    const _v_lo = py.float(orv(py.get(lo_res, _f, 0.0), 0.0));
    const _v_hi = py.float(orv(py.get(hi, _f, 0.0), 0.0));
    if (_v_hi < _v_lo) {
      hi[_f] = _v_lo;
      if (_f === 'dp') dp_floored = true;
    }
  }
  if (dp_floored) {
    const _dp = py.float(orv(py.get(hi, 'dp', 0.0), 0.0));
    if (_dp > 0) {
      const [_label, _short] = dp_to_label(_dp);
      hi.dan_label = _label;
      hi.dan_short = _short;
      hi.sublevel = dp_to_sublevel(_dp);
    }
  }
  for (const _mk of ['celestial', 'signicial', 'shoegazer', 'ln_course']) {
    const _m_lo = py.get(lo_res, _mk);
    const _m_hi = py.get(hi, _mk);
    if (isDict(_m_lo) && isDict(_m_hi)) {
      const _dpk = py.getitem(dp_key, _mk);
      const _a = py.float(orv(py.get(_m_lo, _dpk, 0.0), 0.0));
      const _b = py.float(orv(py.get(_m_hi, _dpk, 0.0), 0.0));
      if (_b < _a) {
        hi[_mk] = dictCopy(_m_hi);
        hi[_mk][_dpk] = py.round(_a, 3);
        dictUpdate(hi[_mk], py.getitem(fields_fns, _mk)(_a));
      }
    }
  }
  return hi;
}

/** Parses .osu, .sm, or .ssc into the unified DanOverlay chart dict. */
export function _load_parsed_chart(file_path, difficulty = '') {
  const ext = vfs.path.splitext(file_path)[1].toLowerCase();
  if (ext === '.sm' || ext === '.ssc') {
    // `from etterna import parse_simfile` → ImportError → `from etterna.sm_parser import
    // parse_simfile` → ModuleNotFoundError: the etterna package is not part of the engine.
    void difficulty;
    throw new ModuleNotFoundError("No module named 'etterna'");
  }
  return parsear_osu_v2(file_path, null, true, true);
}

/**
 * Ensures a valid .osu path for the SR / MinaCalc engines.
 * If file_path is already .osu, returns [file_path, false]; otherwise writes a
 * temporary .osu (removed by the caller) and returns [temp_path, true].
 */
export function _ensure_osu_path_for_c_engines(file_path, parsed = null, difficulty = '') {
  const ext = vfs.path.splitext(file_path)[1].toLowerCase();
  if (ext === '.osu') return [file_path, false];

  if (isNone(parsed)) parsed = _load_parsed_chart(file_path, difficulty);

  // tempfile.mkstemp(suffix=".osu") creates the (empty) file right away
  const tmp_path = vfs.tempPath('tmp', '.osu');
  vfs.putFile(tmp_path, '');
  const bpm = orv(py.get(parsed, 'bpm', 120.0), 120.0);
  const beat_len = bpm > 0 ? 60000.0 / bpm : 500.0;
  const tkey = (t, c) => JSON.stringify([t, c]); // tuple (time_ms, col) as a Map key
  const ln_map = new Map();
  for (const ev of py.get(parsed, 'note_events', [])) {
    if (py.get(ev, 'event_type') === 'ln_start' && py.truthy(py.get(ev, 'ln_end_ms'))) {
      ln_map.set(tkey(py.getitem(ev, 'time_ms'), py.getitem(ev, 'col')), py.getitem(ev, 'ln_end_ms'));
    }
  }

  const lines = [
    'osu file format v14\n\n[General]\nMode: 3\n\n[Difficulty]\nCircleSize: 4\nOverallDifficulty: 8\n\n[TimingPoints]\n',
    `0,${py.format(beat_len, '.4f')},4,2,0,0,1,0\n\n[HitObjects]\n`,
  ];
  // note times / columns / LN ends are Python ints (parser._safe_int)
  for (const [t_ms, col] of py.get(parsed, 'notes', [])) {
    const x = py.int(col * 128 + 64);
    const k = tkey(t_ms, col);
    if (ln_map.has(k)) {
      const end_ms = ln_map.get(k);
      lines.push(`${py.reprInt(x)},192,${py.reprInt(t_ms)},128,0,${py.reprInt(end_ms)}:0:0:0:0:\n`);
    } else {
      lines.push(`${py.reprInt(x)},192,${py.reprInt(t_ms)},1,0,0:0:0:0:\n`);
    }
  }

  // os.fdopen(fd, "w", encoding="utf-8"): text mode on Windows writes '\n' as '\r\n'
  vfs.putFile(tmp_path, lines.join('').replace(/\n/g, '\r\n'));
  return [tmp_path, true];
}

/** Remove the temporary .osu written by _ensure_osu_path_for_c_engines (the `finally:` blocks). */
function _remove_temp(engine_path, is_temp) {
  if (is_temp && vfs.path.exists(engine_path)) {
    try {
      vfs.removeFile(engine_path);
    } catch (e) {
      if (!(e instanceof py.OSError_)) throw e;
    }
  }
}

/** Execution path for the ISOR engine (Isotonic Strain Organic Residual). */
export function _analyze_map_isor(osu_path, mod = 'NM', strict_domain = false, rate = null, difficulty = '') {
  let parsed;
  try {
    parsed = _load_parsed_chart(osu_path, difficulty);
  } catch (exc) {
    return _error_payload(`parse_error: ${excStr(exc)}`);
  }

  if (py.truthy(py.get(parsed, 'rejected'))) {
    return _error_payload('domain_rejected', py.get(parsed, 'warnings', []));
  }

  const domain = validate_domain(parsed);
  if (py.truthy(strict_domain) && !py.truthy(py.get(domain, 'valid', true))) {
    return _error_payload('domain_out_of_range');
  }

  // 7K charts and charts routed to LN Course use the legacy implementation.
  const _ln_route = pyStr(orv(py.get(domain, 'ln_route', 'rice'), 'rice'));
  if (py.truthy(py.get(domain, 'is_7k')) || _ln_route === 'ln') {
    const res = _analyze_map_impl_inner(osu_path, mod, strict_domain, rate);
    if (isDict(res)) res.engine = 'isor';
    return res;
  }

  // .sm/.ssc charts are converted to a temporary .osu for the binary engines.
  const [engine_path, is_temp] = _ensure_osu_path_for_c_engines(osu_path, parsed);
  let raw_isor;
  try {
    raw_isor = analyze_beatmap_isor(engine_path, mod, rate, parsed, domain);
    if (!py.truthy(raw_isor) || py.truthy(py.get(raw_isor, 'error'))) {
      return _error_payload(py.truthy(raw_isor) ? py.get(raw_isor, 'error') : 'isor_analysis_failed');
    }
  } catch (exc) {
    return _error_payload(`isor_error: ${excStr(exc)}`);
  } finally {
    _remove_temp(engine_path, is_temp);
  }

  const _feat = py.get(raw_isor, 'features');
  const features = py.truthy(_feat) ? _feat : extract_features(parsed);
  const msd_res = orv(py.get(raw_isor, 'msd'), {});
  const sunny_strains = orv(py.get(raw_isor, 'sunny_strains'), {});
  const raw_sr = py.float(orv(py.get(raw_isor, 'raw_sr', 0.0), 0.0));
  const family = pyStr(orv(py.get(raw_isor, 'family', 'hybrid'), 'hybrid'));

  // MSD overall / per-skillset values reported by the ISOR engine.
  const _overall_msd = isDict(msd_res) ? py.float(orv(py.get(msd_res, 'overall', 0.0), 0.0)) : 0.0;
  const _skillsets = {};
  if (isDict(msd_res)) {
    for (const k of ['stream', 'jumpstream', 'handstream', 'stamina', 'jackspeed', 'chordjack', 'technical']) {
      if (py.contains(msd_res, k)) _skillsets[k] = py.float(py.getitem(msd_res, k));
    }
  }

  // BPM values scaled by the effective rate.
  const _MOD_RATE = { HT: 0.75, DT: 1.5, NC: 1.5, NM: 1.0 };
  const _effective_bpm_rate = !isNone(rate) ? rate : py.get(_MOD_RATE, mod, 1.0);
  const _raw_bpm = py.float(orv(py.get(features, 'bpm', 0.0), 0.0));
  const _raw_min = py.float(orv(py.get(parsed, 'bpm_min', _raw_bpm), _raw_bpm));
  const _raw_max = py.float(orv(py.get(parsed, 'bpm_max', _raw_bpm), _raw_bpm));
  const _raw_common = py.float(orv(py.get(parsed, 'bpm_common', _raw_bpm), _raw_bpm));

  // Celestial is derived from the ISOR DP; Signicial / Shoegazer are not set here.
  const cel_obj = _celestial_from_isor_dp(py.get(raw_isor, 'dp', 0.0));
  const celestial_res = !isNone(cel_obj) ? cel_obj.to_dict() : null;
  const signicial_res = null;
  const shoegazer_res = null;

  // debug["sr_result"] carries the Sunny component strains for the ui-7 skin.
  const debug = {
    sr_result: {
      jbar_max: py.float(orv(py.get(sunny_strains, 'jbar_max', 0.0), 0.0)),
      pbar_max: py.float(orv(py.get(sunny_strains, 'pbar_max', 0.0), 0.0)),
      xbar_max: py.float(orv(py.get(sunny_strains, 'xbar_max', 0.0), 0.0)),
      abar_mean: py.float(orv(py.get(sunny_strains, 'abar_mean', 0.0), 0.0)),
    },
    isor: {
      weights: py.get(raw_isor, 'weights', {}),
      dp_sr: py.get(raw_isor, 'dp_sr'),
      dp_choke: py.get(raw_isor, 'dp_choke'),
      dp_msd: py.get(raw_isor, 'dp_msd'),
      dp_bio: py.get(raw_isor, 'dp_bio'),
      bio_numeric: py.get(raw_isor, 'bio_numeric'),
      sk_key: py.get(raw_isor, 'sk_key'),
      ridge_correction: py.get(raw_isor, 'ridge_correction', 0.0),
      modulated_sr: py.get(raw_isor, 'modulated_sr'),
    },
  };

  const _drain_s = py.float(orv(py.get(domain, 'drain_time_s', 0.0), 0.0));
  const _note_count = py.int(orv(py.get(domain, 'note_count', 0), 0));
  const _avg_nps = _drain_s > 0 ? _note_count / _drain_s : 0.0;

  return {
    engine: 'isor',
    dp: py.getitem(raw_isor, 'dp'),
    dan_label: py.getitem(raw_isor, 'dan_label'),
    dan_short: py.getitem(raw_isor, 'dan_short'),
    sublevel: py.getitem(raw_isor, 'sublevel'),
    confidence: py.float(orv(py.get(raw_isor, 'family_confidence', 1.0), 1.0)),
    sr: raw_sr,
    family,
    mod: orv(mod, 'NM'),
    corrections: py.truthy(py.get(raw_isor, 'ridge_correction'))
      ? [`isor_ridge:${py.format(py.get(raw_isor, 'ridge_correction', 0.0), '+.3f')}`]
      : [],
    nps: py.round(_avg_nps, 1),
    peak_nps: py.float(orv(py.get(features, 'nps_p90', 0.0), 0.0)),
    nps_curve: py.get(features, 'nps_curve', []),
    duration_s: _drain_s,
    note_count: _note_count,
    warnings: listAdd(py.get(parsed, 'warnings', []), py.get(domain, 'warnings', [])),
    error: null,
    debug,
    overall_msd: _overall_msd,
    primary_role: family,
    role_estimates: {},
    skillsets: _skillsets,
    composite_dan: '',
    bottleneck_role: '',
    is_generalist: false,
    role_breakdown_text: '',
    celestial: celestial_res,
    signicial: signicial_res,
    shoegazer: shoegazer_res,
    ln_course: null,
    ln_route: _ln_route,
    strain_graph: py.get(raw_isor, 'strain_graph'),
    bpm: py.round(mulRate(_raw_bpm, _effective_bpm_rate), 1),
    bpm_min: py.roundInt(mulRate(_raw_min, _effective_bpm_rate)),
    bpm_max: py.roundInt(mulRate(_raw_max, _effective_bpm_rate)),
    bpm_common: py.roundInt(mulRate(_raw_common, _effective_bpm_rate)),
    od: py.round(py.float(orv(py.get(parsed, 'od', 0.0), 0.0)), 1),
  };
}

/**
 * Estimate the Dan rank using the core pipeline runtime logic.
 *
 * @param {string} osu_path  path of the .osu file (in the vfs)
 * @param {string} mod  active mods (DT, HT, ...)
 * @param {boolean} strict_domain  if true, returns an error when the beatmap is outside standard constraints
 * @param {number|null} rate  custom rate (e.g. from lazer); null → the mod's default rate
 * @param {string|null} engine  "isor" (default) or "legacy"
 * @param {string} difficulty  target difficulty name (StepMania .sm/.ssc only)
 * @returns {object} structured analysis payload for the overlay
 */
export function analyze_map(osu_path, mod = 'NM', strict_domain = false, rate = null, engine = null, difficulty = '') {
  const _engine = py.strip(pyStr(orv(engine, 'isor')).toLowerCase());

  // Legacy engine at a non-native custom rate: interpolate between native anchors.
  if (_engine === 'legacy' && !isNone(rate)) {
    const _r = py.round(py.float(rate), 4);
    const _NATIVE = [0.75, 1.0, 1.5];
    // Native rates (HT / NM / DT) are analysed directly below.
    if (!_NATIVE.includes(_r)) {
      let _lo = null;
      let _hi = null;
      let _t;
      if (_r < _NATIVE[0]) {
        // Below HT: extrapolate from the HT..NM segment (_t < 0).
        [_lo, _hi] = [_NATIVE[0], _NATIVE[1]];
        _t = (_r - _lo) / py.max2(_hi - _lo, 1e-9);
      } else if (_r > _NATIVE[_NATIVE.length - 1]) {
        [_lo, _hi] = [_NATIVE[_NATIVE.length - 2], _NATIVE[_NATIVE.length - 1]];
        _t = 1.0 + (_r - _hi) / py.max2(_hi - _lo, 1e-9);
      } else {
        for (let i = 0; i < _NATIVE.length - 1; i++) {
          if (_NATIVE[i] <= _r && _r <= _NATIVE[i + 1]) {
            [_lo, _hi] = [_NATIVE[i], _NATIVE[i + 1]];
            break;
          }
        }
        if (_lo === null) {
          // rate NaN: no segment matched and Python computes `_r - None`
          throw new py.TypeError_("unsupported operand type(s) for -: 'float' and 'NoneType'");
        }
        _t = (_r - _lo) / py.max2(_hi - _lo, 1e-9);
      }

      const _res_lo = analyze_map(osu_path, mod, strict_domain, _lo, 'legacy', difficulty);
      const _res_hi = analyze_map(osu_path, mod, strict_domain, _hi, 'legacy', difficulty);
      if (isDict(_res_lo) && isDict(_res_hi)
          && !isNone(py.get(_res_lo, 'dp')) && !isNone(py.get(_res_hi, 'dp'))) {
        const _interp_result = dictCopy(_res_lo);
        const _NUMERIC = [
          'dp', 'sr', 'overall_msd', 'confidence',
          'bpm', 'bpm_min', 'bpm_max', 'bpm_common', 'od',
        ];
        for (const _f of _NUMERIC) {
          const _v_lo = py.float(orv(py.get(_res_lo, _f, 0.0), 0.0));
          const _v_hi = py.float(orv(py.get(_res_hi, _f, 0.0), 0.0));
          _interp_result[_f] = py.round(py.max2(0.0, _v_lo + _t * (_v_hi - _v_lo)), 2);
        }

        const _dlo = py.get(_res_lo, 'dp', 0.0);
        const _dhi = py.get(_res_hi, 'dp', 0.0);
        const _dp_val = py.round(py.max2(0.5, _dlo + _t * (_dhi - _dlo)), 2);
        _interp_result.dp = _dp_val;
        const [_label, _short] = dp_to_label(_dp_val);
        _interp_result.dan_label = _label;
        _interp_result.dan_short = _short;
        _interp_result.sublevel = dp_to_sublevel(_dp_val);

        // Per-estimator DPs, re-derived through each estimator's fields_from_dp().
        const _fields_fns = {
          celestial: _celestial_fields_from_dp,
          signicial: _signicial_fields_from_dp,
          shoegazer: _shoegazer_fields_from_dp,
          ln_course: _ln_course_fields_from_dp,
        };
        const _dp_key = {
          celestial: 'dp_celestial',
          signicial: 'dp_signicial',
          shoegazer: 'dp_shoegazer',
          ln_course: 'dp_ln',
        };
        for (const _mk of ['celestial', 'signicial', 'shoegazer', 'ln_course']) {
          const _m_lo = py.get(_res_lo, _mk);
          const _m_hi = py.get(_res_hi, _mk);
          if (isDict(_m_lo) && isDict(_m_hi)) {
            _interp_result[_mk] = dictCopy(_m_lo);
            for (const _mf of ['dp_celestial', 'dp_signicial', 'dp_shoegazer', 'dp_ln', 'confidence']) {
              if (py.contains(_m_lo, _mf) && py.contains(_m_hi, _mf)) {
                const _a = py.float(orv(py.get(_m_lo, _mf, 0.0), 0.0));
                const _b = py.float(orv(py.get(_m_hi, _mf, 0.0), 0.0));
                _interp_result[_mk][_mf] = py.round(py.max2(0.0, _a + _t * (_b - _a)), 2);
              }
            }
            const _dpf = py.get(_interp_result[_mk], py.getitem(_dp_key, _mk));
            if (!isNone(_dpf)) dictUpdate(_interp_result[_mk], py.getitem(_fields_fns, _mk)(py.float(_dpf)));
          }
        }

        _interp_result.custom_rate_interpolated = true;
        return _interp_result;
      }
    }
  }

  return _analyze_map_impl(osu_path, mod, strict_domain, rate, _engine, difficulty);
}

// Result cache for _analyze_map_impl (deep copies; oldest entry evicted first).
// Python keys it by the tuple (abspath, difficulty, mtime_ns, mod, strict, rate, engine);
// here the tuple is serialised to a string (Map keeps dict insertion order).
export const _impl_cache = new Map();
export const _IMPL_CACHE_MAX = 96;

export function _analyze_map_impl(osu_path, mod = 'NM', strict_domain = false, rate = null, engine = 'isor', difficulty = '') {
  if (typeof osu_path !== 'string') {
    // os.stat() only swallows OSError; a non-path argument raises TypeError
    throw new py.TypeError_(`stat: path should be string, bytes, os.PathLike or integer, not ${typeName(osu_path)}`);
  }
  let _mtime;
  try {
    _mtime = vfs.statMtimeNs(osu_path);
  } catch (e) {
    if (!(e instanceof py.OSError_)) throw e;
    _mtime = 0;
  }
  const _key = JSON.stringify([
    vfs.path.abspath(osu_path),
    pyStr(orv(difficulty, '')),
    String(_mtime),
    isNone(mod) ? null : mod,
    py.truthy(strict_domain),
    py.round(!isNone(rate) ? py.float(rate) : 1.0, 4),
    pyStr(engine).toLowerCase(),
  ]);
  const _hit = _impl_cache.get(_key);
  if (!isNone(_hit)) return py.deepcopy(_hit);

  let _result;
  if (pyStr(engine).toLowerCase() === 'isor') {
    _result = _analyze_map_isor(osu_path, mod, strict_domain, rate, difficulty);
  } else {
    _result = _analyze_map_impl_inner(osu_path, mod, strict_domain, rate, difficulty);
    if (isDict(_result)) _result.engine = 'legacy';
  }

  if (isDict(_result)) {
    _impl_cache.set(_key, py.deepcopy(_result));
    if (_impl_cache.size > _IMPL_CACHE_MAX) _impl_cache.delete(_impl_cache.keys().next().value);
  }

  return _result;
}

/** Core pipeline implementation (no custom-rate interpolation). */
export function _analyze_map_impl_inner(osu_path, mod = 'NM', strict_domain = false, rate = null, difficulty = '') {
  // Parse (.osu / .sm / .ssc).
  let parsed;
  try {
    parsed = _load_parsed_chart(osu_path, difficulty);
  } catch (exc) {
    return _error_payload(`parse_error: ${excStr(exc)}`);
  }

  if (py.truthy(py.get(parsed, 'rejected'))) {
    return _error_payload('domain_rejected', py.get(parsed, 'warnings', []));
  }

  const domain = validate_domain(parsed);
  if (py.truthy(strict_domain) && !py.truthy(py.get(domain, 'valid', true))) {
    return _error_payload('domain_out_of_range');
  }

  if (py.truthy(py.get(domain, 'is_7k'))) {
    // 7K: SR from the primary SR engine, placed on the 7K tier scale built from the
    // per-tier SR medians in config/sr_means_7k.json. Tier boundaries are the midpoints
    // between neighbouring medians; the last tier extends by one median gap and SR above
    // it is reported as "Beyond Stellium".
    let algo_res;
    let sr;
    try {
      algo_res = analyze_primary_sr(osu_path, mod, rate);
      if (!py.truthy(py.get(algo_res, 'success', false))) {
        return _error_payload(`7k_algorithm_error: ${pyStr(py.get(algo_res, 'error'))}`);
      }
      sr = py.float(orv(py.get(algo_res, 'sr', 0.0), 0.0));
    } catch (exc) {
      return _error_payload(`7k_algorithm_error: ${excStr(exc)}`);
    }

    let means_7k;
    try {
      means_7k = py.getitem(JSON.parse(vfs.readText(resource_path('config', 'sr_means_7k.json'))), 'general');
    } catch (exc) {
      return _error_payload(`7k_config_error: ${excStr(exc)}`);
    }

    const tier_order = ['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
      'Gamma', 'Azimuth', 'Zenith', 'Stellium'];

    const medians = tier_order.map((t) => py.getitem(py.getitem(means_7k, t), 'median'));

    // (lo, hi, tier index) for each tier.
    const boundaries = [];
    for (let i = 0; i < medians.length - 1; i++) {
      const lo = (medians[i] + medians[i + 1]) / 2;
      let hi;
      if (i + 2 < medians.length) hi = (medians[i + 1] + medians[i + 2]) / 2;
      else hi = medians[i + 1] + (medians[i + 1] - medians[i]);
      boundaries.push([lo, hi, i]);
    }

    let best_idx = null;
    let dp_7k = 0.0;
    let beyond = false;

    if (sr < py.at(boundaries, 0)[0]) {
      best_idx = 0;
      dp_7k = 0.0;
    } else if (sr >= py.at(boundaries, -1)[1]) {
      best_idx = tier_order.length - 1;
      const lo_sr = py.at(boundaries, -1)[0];
      const hi_sr = py.max2(py.at(boundaries, -1)[1], sr + 0.01);
      const t = hi_sr > lo_sr ? (sr - lo_sr) / (hi_sr - lo_sr) : 0.5;
      dp_7k = best_idx + t;
      beyond = true;
    } else {
      for (const [lo, hi, idx] of boundaries) {
        if (lo <= sr && sr < hi) {
          const t = hi > lo ? (sr - lo) / (hi - lo) : 0.5;
          dp_7k = idx + t;
          best_idx = idx;
          break;
        }
      }
    }

    if (best_idx === null) {
      best_idx = 11;
      dp_7k = 11.0;
    }

    let best_tier = tier_order[best_idx];
    dp_7k = py.round(dp_7k, 2);

    // Sub-level from the fractional part of the DP.
    const dp_frac = py.round(dp_7k - py.int(dp_7k), 2);
    let sub;
    if (dp_frac <= 0.2) sub = 'Low';
    else if (dp_frac <= 0.4) sub = 'Mid-Low';
    else if (dp_frac <= 0.6) sub = 'Mid';
    else if (dp_frac <= 0.8) sub = 'Mid-High';
    else sub = 'High';

    // Numbered tiers get a " Dan" suffix.
    if (['0th', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'].includes(best_tier)) {
      best_tier += ' Dan';
    }

    if (beyond) {
      best_tier = 'Beyond Stellium';
      sub = 'Beyond';
    }

    // BPM values scaled by the effective rate.
    const _mod_rate_map = { HT: 0.75, DT: 1.5, NC: 1.5, NM: 1.0 };
    const _eff_rate = !isNone(rate) ? rate : py.get(_mod_rate_map, mod, 1.0);
    const bpm_val = mulRate(py.float(orv(orv(py.get(parsed, 'bpm_common', 0.0), py.get(parsed, 'bpm', 0.0)), 0.0)), _eff_rate);
    const bpm_common_raw = mulRate(py.float(orv(py.get(parsed, 'bpm_common', bpm_val), bpm_val)), _eff_rate);

    return {
      mode: '7k',
      tier_7k: best_tier,
      sublevel_7k: sub,
      dp_7k,
      sr: py.round(sr, 2),
      bpm: py.round(bpm_val, 1),
      bpm_min: py.roundInt(mulRate(py.float(orv(py.get(parsed, 'bpm_min', bpm_common_raw), bpm_common_raw)), _eff_rate)),
      bpm_max: py.roundInt(mulRate(py.float(orv(py.get(parsed, 'bpm_max', bpm_common_raw), bpm_common_raw)), _eff_rate)),
      bpm_common: py.roundInt(bpm_common_raw),
      od: py.round(py.float(orv(py.get(parsed, 'od', 0.0), 0.0)), 1),
      note_count: py.int(py.get(domain, 'note_count', 0)),
      duration_s: py.float(py.get(domain, 'drain_time_s', 0.0)),
      warnings: [],
      debug: _inject_sunny_components({}, algo_res),
      error: null,
    };
  }

  const features = extract_features(parsed);
  // .sm/.ssc charts are converted to a temporary .osu (removed in `finally` below).
  const [engine_path, is_temp] = _ensure_osu_path_for_c_engines(osu_path, parsed);

  // Estimator wrappers: each one returns null on any failure.
  const _safe_mina = () => {
    try {
      const _MOD_RATE = { HT: 0.75, DT: 1.5, NC: 1.5 };
      const effective_rate = !isNone(rate) ? rate : py.get(_MOD_RATE, mod, 1.0);
      return _minacalc_estimate(engine_path, effective_rate, features);
    } catch {
      return null;
    }
  };

  // Derive Celestial estimate using primary SR + MSD skillsets.
  const _safe_celestial = (mina_result, primary_sr = null, family_hint = null) => {
    try {
      if (!isDict(mina_result)) return null;
      const skillsets = py.get(mina_result, 'skillsets');
      if (!py.truthy(skillsets)) return null;
      const result = _celestial_estimate(skillsets, primary_sr, family_hint);
      return !isNone(result) ? result.to_dict() : null;
    } catch {
      return null;
    }
  };

  // Derive Signicial estimate using primary SR + MSD skillsets.
  const _safe_signicial = (mina_result, primary_sr = null, family_hint = null) => {
    try {
      if (!isDict(mina_result)) return null;
      const skillsets = py.get(mina_result, 'skillsets');
      if (!py.truthy(skillsets)) return null;
      const result = _signicial_estimate(skillsets, primary_sr, family_hint);
      return !isNone(result) ? result.to_dict() : null;
    } catch {
      return null;
    }
  };

  // Derive Shoegazer estimate using primary SR + MSD skillsets.
  const _safe_shoegazer = (mina_result, primary_sr = null, family_hint = null) => {
    void family_hint;
    try {
      if (!isDict(mina_result)) return null;
      const skillsets = py.get(mina_result, 'skillsets');
      if (!py.truthy(skillsets)) return null;
      const result = _shoegazer_estimate(skillsets, primary_sr);
      return !isNone(result) ? result.to_dict() : null;
    } catch {
      return null;
    }
  };

  // Derive LN Course estimate using primary SR + MSD skillsets.
  const _safe_ln_course = (mina_result, primary_sr = null, ln_family = null, feats = null) => {
    try {
      if (!isDict(mina_result)) return null;
      const skillsets = py.get(mina_result, 'skillsets');
      if (!py.truthy(skillsets)) return null;
      const family = orv(ln_family, 'allround');
      const result = _ln_course_estimate(skillsets, primary_sr, family, feats);
      return !isNone(result) ? result.to_dict() : null;
    } catch {
      return null;
    }
  };

  let primary_core;
  let mina;
  try {
    const mina_future = _PIPELINE_POOL.submit(_safe_mina);
    const primary_future = _PIPELINE_POOL.submit(
      _compute_primary_rank_result,
      engine_path, mod, strict_domain, null,
      parsed, domain, features, rate,
    );
    primary_core = primary_future.result();
    mina = mina_future.result();
  } finally {
    _remove_temp(engine_path, is_temp);
  }

  const primary_ok = isDict(primary_core) && !isNone(py.get(primary_core, 'dp'));

  // BPM values scaled by the effective rate.
  const _MOD_RATE = { HT: 0.75, DT: 1.5, NC: 1.5, NM: 1.0 };
  const _effective_bpm_rate = !isNone(rate) ? rate : py.get(_MOD_RATE, mod, 1.0);
  const _raw_bpm = py.float(orv(py.get(features, 'bpm', 0.0), 0.0));
  const _raw_min = py.float(orv(py.get(parsed, 'bpm_min', _raw_bpm), _raw_bpm));
  const _raw_max = py.float(orv(py.get(parsed, 'bpm_max', _raw_bpm), _raw_bpm));
  const _raw_common = py.float(orv(py.get(parsed, 'bpm_common', _raw_bpm), _raw_bpm));

  if (primary_ok) {
    const merged = _merge_primary_and_mina(primary_core, mina);
    const _primary_sr = orv(py.float(orv(py.get(primary_core, 'sr', 0.0), 0.0)), null);
    const _family = pyStr(orv(py.get(primary_core, 'family', ''), ''));
    merged.celestial = _safe_celestial(mina, _primary_sr, _family);
    merged.signicial = _safe_signicial(mina, _primary_sr, _family);
    merged.shoegazer = _safe_shoegazer(mina, _primary_sr, _family);

    // LN Course only for maps routed to "ln".
    const _ln_route = pyStr(orv(py.get(domain, 'ln_route', 'rice'), 'rice'));
    merged.ln_route = _ln_route;
    if (_ln_route === 'ln') {
      const _ln_family = _classify_ln_family(features);
      merged.ln_course = _safe_ln_course(mina, _primary_sr, _ln_family, features);
    } else {
      merged.ln_course = null;
    }

    // BPM / OD
    merged.bpm = py.round(mulRate(_raw_bpm, _effective_bpm_rate), 1);
    merged.bpm_min = py.roundInt(mulRate(_raw_min, _effective_bpm_rate));
    merged.bpm_max = py.roundInt(mulRate(_raw_max, _effective_bpm_rate));
    merged.bpm_common = py.roundInt(mulRate(_raw_common, _effective_bpm_rate));
    merged.od = py.round(py.float(orv(py.get(parsed, 'od', 0.0), 0.0)), 1);

    // Strain graph from the primary SR result.
    const _sr_raw = orv(py.get(primary_core, 'primary_sr'), {});
    merged.strain_graph = py.get(_sr_raw, 'strain_graph');

    return merged;
  }

  // Fallback: MinaCalc-only result when the primary SR path failed.
  if (isDict(mina)) mina.error = null;
  const mina_ok = isDict(mina) && !isNone(py.get(mina, 'dp'));
  if (mina_ok) {
    setdefault(mina, 'warnings', []).push('primary_sr_path_unavailable');
    mina.engine = 'minacalc_fallback';
    mina.celestial = _safe_celestial(mina);
    mina.signicial = _safe_signicial(mina);
    mina.shoegazer = _safe_shoegazer(mina);
    mina.ln_route = pyStr(orv(py.get(domain, 'ln_route', 'rice'), 'rice'));
    mina.ln_course = null;
    mina.bpm = py.round(mulRate(_raw_bpm, _effective_bpm_rate), 1);
    mina.bpm_min = py.roundInt(mulRate(_raw_min, _effective_bpm_rate));
    mina.bpm_max = py.roundInt(mulRate(_raw_max, _effective_bpm_rate));
    mina.bpm_common = py.roundInt(mulRate(_raw_common, _effective_bpm_rate));
    mina.od = py.round(py.float(orv(py.get(parsed, 'od', 0.0), 0.0)), 1);
    if (mina.ln_route === 'ln') {
      const _ln_family = _classify_ln_family(features);
      mina.ln_course = _safe_ln_course(mina, null, _ln_family, features);
    }
    return mina;
  }

  return _error_payload(
    isDict(primary_core) ? py.get(primary_core, 'error') : 'analysis_unavailable',
    ['minacalc_cli_or_config_missing', 'primary_sr_path_failed'],
  );
}
