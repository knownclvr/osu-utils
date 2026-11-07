// contracts.js — shared data contracts (MapInfo, AnalysisResult) and overlay state constants.
//
// JavaScript port of engine/dan_engine/contracts.py (DanOverlay 3.1.0 by 8doul).
//
// The Web Worker calls AnalysisResult.from_pipeline(raw, mod_label).to_dict() on the output
// of pipeline.analyze_map() — the payload DanOverlay's UI bridge sends to overlay.js.
import * as py from './py.js';

// ── Overlay states ───────────────────────────────────────────────────

export const STATE_IDLE = 'idle';
export const STATE_WAITING_TOSU = 'waiting_tosu';
export const STATE_ANALYZING = 'analyzing';
export const STATE_READY = 'ready';
export const STATE_WARNING = 'warning';
export const STATE_ERROR = 'error';

// Below this confidence the overlay shows a dan range instead of a single dan
export const RANGE_CONFIDENCE_THRESHOLD = 0.5;

// Dan order used for range labels (dp 1.0 → "1st" … dp 20.x → "Kappa")
export const DAN_ORDER = [
  '1st', '2nd', '3rd', '4th', '5th',
  '6th', '7th', '8th', '9th', '10th',
  'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon',
  'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa',
];

/** DP float → short label string. */
export function _dp_to_short(dp) {
  const idx = py.max2(0, py.min2(DAN_ORDER.length - 1, py.int(dp) - 1));
  return DAN_ORDER[idx];
}

/** When confidence is low, return a two-label range string or null. */
export function _compute_dan_range(dp, confidence) {
  if (confidence >= RANGE_CONFIDENCE_THRESHOLD) return null;

  const spread = (1.0 - confidence) * 1.5;
  const lo = py.max2(0.5, dp - spread);
  const hi = py.min2(20.5, dp + spread);
  const label_lo = _dp_to_short(lo);
  const label_hi = _dp_to_short(hi);
  if (label_lo === label_hi) return null;
  return `${label_lo}–${label_hi}`; // en dash
}

// ── Python builtins used by from_pipeline / to_dict ──────────────────

/** isinstance(x, dict): a Map (dict / defaultdict / Counter) or a plain object. */
function _isDict(x) {
  if (x instanceof Map) return true;
  if (x === null || typeof x !== 'object' || Array.isArray(x) || ArrayBuffer.isView(x) || x instanceof Set) return false;
  const proto = Object.getPrototypeOf(x);
  return proto === Object.prototype || proto === null;
}

function _typeName(x) {
  if (x === null || x === undefined) return 'NoneType';
  if (typeof x === 'boolean') return 'bool';
  if (typeof x === 'number') return Number.isInteger(x) ? 'int' : 'float';
  if (typeof x === 'string') return 'str';
  return typeof x;
}

/** list(x) */
function _list(x) {
  if (Array.isArray(x)) return x.slice();
  if (typeof x === 'string') return Array.from(x);
  if (ArrayBuffer.isView(x)) return Array.from(x);
  if (x instanceof Map) return [...x.keys()];
  if (x instanceof Set) return [...x];
  if (_isDict(x)) return Object.keys(x);
  throw new py.TypeError_(`'${_typeName(x)}' object is not iterable`);
}

/** dict(x) — a shallow copy (Map stays a Map, plain object stays a plain object). */
function _dict(x) {
  if (x instanceof Map) return new Map(x);
  if (_isDict(x)) return { ...x };
  if (Array.isArray(x)) {
    // dict(iterable of key/value pairs)
    const pairs = x.map((kv) => {
      if (!Array.isArray(kv) || kv.length !== 2) throw new py.ValueError('dictionary update sequence element has wrong length');
      return kv;
    });
    if (pairs.every(([k]) => typeof k === 'string')) return Object.fromEntries(pairs);
    return new Map(pairs);
  }
  throw new py.TypeError_(`'${_typeName(x)}' object is not iterable`);
}

/**
 * str(x) for the values the pipeline stores in these fields (strings in practice).
 * A JS number cannot tell a Python int from a float: integral values print as ints.
 */
function _str(x) {
  if (typeof x === 'string') return x;
  if (x === null || x === undefined) return 'None';
  if (typeof x === 'boolean') return x ? 'True' : 'False';
  if (typeof x === 'number') return Number.isInteger(x) ? py.reprInt(x) : py.reprFloat(x);
  if (x instanceof Error) return x.message;
  return String(x);
}

/** x or dflt */
const _or = (x, dflt) => (py.truthy(x) ? x : dflt);

// ── Map info ─────────────────────────────────────────────────────────

/** @dataclass MapInfo — metadata about the currently selected beatmap from tosu. */
export class MapInfo {
  constructor(
    md5,
    path,
    artist = '',
    title = '',
    version = '',
    mapper = '',
    sr_official = 0.0,
    keycount = 4,
    mod_speed = 1.0,
    mod_label = '',
    bg_path = '',
    banner_path = '',
    audio_path = '',
    pack = '',
    game = 'osu',
    // Live playback state
    music_time_ms = 0,
    music_playing = false,
    total_time_ms = 0,
    game_state = 0,
  ) {
    this.md5 = md5;
    this.path = path;
    this.artist = artist;
    this.title = title;
    this.version = version;
    this.mapper = mapper;
    this.sr_official = sr_official;
    this.keycount = keycount;
    this.mod_speed = mod_speed;
    this.mod_label = mod_label;
    this.bg_path = bg_path;
    this.banner_path = banner_path;
    this.audio_path = audio_path;
    this.pack = pack;
    this.game = game;
    this.music_time_ms = music_time_ms;
    this.music_playing = music_playing;
    this.total_time_ms = total_time_ms;
    this.game_state = game_state;
  }

  to_dict() {
    return {
      md5: this.md5,
      path: this.path,
      artist: this.artist,
      title: this.title,
      version: this.version,
      mapper: this.mapper,
      sr_official: this.sr_official,
      keycount: this.keycount,
      mod_speed: this.mod_speed,
      mod_label: this.mod_label,
      game: this.game,
      bg_path: this.bg_path,
      banner_path: this.banner_path,
      audio_path: this.audio_path,
      pack: this.pack,
    };
  }
}

// ── Analysis result ──────────────────────────────────────────────────

/** Field order of the AnalysisResult dataclass (= constructor parameter order). */
const _ANALYSIS_RESULT_FIELDS = [
  'state', 'dp', 'dan_label', 'dan_short', 'sublevel', 'confidence', 'dan_range', 'sr', 'family', 'mod',
  'corrections', 'nps', 'peak_nps', 'nps_curve', 'duration_s', 'note_count', 'warnings', 'error', 'debug',
  'overall_msd', 'primary_role', 'role_estimates', 'skillsets', 'composite_dan', 'bottleneck_role',
  'is_generalist', 'role_breakdown_text', 'celestial', 'signicial', 'shoegazer', 'ln_course', 'ln_route',
  'strain_graph', 'bpm', 'bpm_min', 'bpm_max', 'bpm_common', 'osu_sr', 'od', 'mode_7k', 'tier_7k',
  'sublevel_7k', 'dp_7k', 'engine',
];

/** AnalysisResult(**kw): keyword construction (omitted fields take their defaults). */
function _analysisResultKw(kw) {
  return new AnalysisResult(..._ANALYSIS_RESULT_FIELDS.map((f) => (Object.prototype.hasOwnProperty.call(kw, f) ? kw[f] : undefined)));
}

/**
 * @dataclass AnalysisResult — structured output from one analysis pass.
 *
 * The UI bridge serialises this to JSON and sends it to overlay.js as a single
 * ``type: "analysis"`` payload. No string parsing needed.
 */
export class AnalysisResult {
  constructor(
    state,
    dp = 0.0,
    dan_label = '',
    dan_short = '',
    sublevel = '',
    confidence = 1.0,
    dan_range = null,
    sr = 0.0,
    family = '',
    mod = '',
    corrections = [],
    nps = 0.0,
    peak_nps = 0.0,
    nps_curve = [],
    duration_s = 0.0,
    note_count = 0,
    warnings = [],
    error = null,
    debug = {},
    // MinaCalc role analysis
    overall_msd = 0.0,
    primary_role = '',
    role_estimates = {},
    skillsets = {},
    composite_dan = '',
    bottleneck_role = '',
    is_generalist = false,
    role_breakdown_text = '',
    // Dan Celestial (CelestialResult.to_dict())
    celestial = null,
    // Dan Signicial (SignicialResult.to_dict())
    signicial = null,
    // Dan Shoegazer (ShoegazerResult.to_dict())
    shoegazer = null,
    // LN Dan Course (LnCourseResult.to_dict()) and per-map mode routing
    ln_course = null,
    ln_route = 'rice',
    // Strain graph data
    strain_graph = null,
    // Map metadata
    bpm = 0.0,
    bpm_min = 0,
    bpm_max = 0,
    bpm_common = 0,
    osu_sr = 0.0,
    od = 0.0,
    // 7K mode
    mode_7k = false,
    tier_7k = '',
    sublevel_7k = '',
    dp_7k = 0.0,
    engine = 'isor',
  ) {
    this.state = state;
    this.dp = dp;
    this.dan_label = dan_label;
    this.dan_short = dan_short;
    this.sublevel = sublevel;
    this.confidence = confidence;
    this.dan_range = dan_range;
    this.sr = sr;
    this.family = family;
    this.mod = mod;
    this.corrections = corrections;
    this.nps = nps;
    this.peak_nps = peak_nps;
    this.nps_curve = nps_curve;
    this.duration_s = duration_s;
    this.note_count = note_count;
    this.warnings = warnings;
    this.error = error;
    this.debug = debug;
    this.overall_msd = overall_msd;
    this.primary_role = primary_role;
    this.role_estimates = role_estimates;
    this.skillsets = skillsets;
    this.composite_dan = composite_dan;
    this.bottleneck_role = bottleneck_role;
    this.is_generalist = is_generalist;
    this.role_breakdown_text = role_breakdown_text;
    this.celestial = celestial;
    this.signicial = signicial;
    this.shoegazer = shoegazer;
    this.ln_course = ln_course;
    this.ln_route = ln_route;
    this.strain_graph = strain_graph;
    this.bpm = bpm;
    this.bpm_min = bpm_min;
    this.bpm_max = bpm_max;
    this.bpm_common = bpm_common;
    this.osu_sr = osu_sr;
    this.od = od;
    this.mode_7k = mode_7k;
    this.tier_7k = tier_7k;
    this.sublevel_7k = sublevel_7k;
    this.dp_7k = dp_7k;
    this.engine = engine;
  }

  to_dict() {
    // (self.debug.get("sr_result", {}) if isinstance(self.debug, dict) else {}).get(key, 0.0) — evaluated per key
    const srComponent = (key) => {
      const sr_result = _isDict(this.debug) ? py.get(this.debug, 'sr_result', {}) : {};
      return py.round(py.float(_or(py.get(sr_result, key, 0.0), 0.0)), 2);
    };
    return {
      type: 'analysis',
      state: this.state,
      engine: this.engine,
      mode: py.truthy(this.mode_7k) ? '7k' : '',
      tier_7k: this.tier_7k,
      sublevel_7k: this.sublevel_7k,
      dp_7k: py.round(this.dp_7k, 2),
      dp: py.round(this.dp, 3),
      dan_label: this.dan_label,
      dan_short: this.dan_short,
      sublevel: this.sublevel,
      confidence: py.round(this.confidence, 3),
      dan_range: this.dan_range,
      sr: py.round(this.sr, 3),
      family: this.family,
      mod: this.mod,
      corrections: this.corrections,
      nps: py.round(this.nps, 1),
      peak_nps: py.round(this.peak_nps, 1),
      nps_curve: this.nps_curve,
      duration_s: py.round(this.duration_s, 1),
      note_count: this.note_count,
      warnings: this.warnings,
      // MinaCalc role analysis / course estimators
      overall_msd: py.round(this.overall_msd, 3),
      primary_role: this.primary_role,
      role_estimates: this.role_estimates,
      skillsets: this.skillsets,
      composite_dan: this.composite_dan,
      bottleneck_role: this.bottleneck_role,
      is_generalist: this.is_generalist,
      role_breakdown: this.role_breakdown_text,
      celestial: this.celestial,
      signicial: this.signicial,
      shoegazer: this.shoegazer,
      ln_course: this.ln_course,
      ln_route: this.ln_route,
      strain_graph: this.strain_graph,
      bpm: py.round(this.bpm, 1),
      bpm_min: this.bpm_min,
      bpm_max: this.bpm_max,
      bpm_common: this.bpm_common,
      osu_sr: py.round(this.osu_sr, 2),
      od: py.round(this.od, 1),
      debug: this.debug,
      jbar_max: srComponent('jbar_max'),
      pbar_max: srComponent('pbar_max'),
      xbar_max: srComponent('xbar_max'),
      abar_mean: srComponent('abar_mean'),
    };
  }

  /** Build an AnalysisResult from pipeline.analyze_map() output. */
  static from_pipeline(raw, mod_label = '') {
    if (py.truthy(py.get(raw, 'error')) && py.get(raw, 'dp') == null) { // `is None` (undefined counts too)
      return _analysisResultKw({
        state: STATE_ERROR,
        error: _str(py.getitem(raw, 'error')),
        warnings: py.get(raw, 'warnings', []),
        engine: _str(_or(py.get(raw, 'engine', 'isor'), 'isor')),
      });
    }

    const dp = py.float(_or(py.get(raw, 'dp', 0), 0));
    const confidence = py.float(_or(py.get(raw, 'confidence', 0), 0));
    const warnings = _list(py.get(raw, 'warnings', []));

    if (py.truthy(py.get(raw, 'error'))) warnings.push(_str(py.getitem(raw, 'error')));

    const state = confidence < RANGE_CONFIDENCE_THRESHOLD ? STATE_WARNING : STATE_READY;
    const dan_range = _compute_dan_range(dp, confidence);

    const overall_msd = py.float(_or(py.get(raw, 'overall_msd', 0), 0));
    const sr_value = py.float(_or(py.get(raw, 'sr', 0), 0));
    const duration_s = py.float(_or(py.get(raw, 'duration_s', 0), 0));
    const note_count = py.int(_or(py.get(raw, 'note_count', 0), 0));

    const avg_nps = duration_s > 0 ? note_count / duration_s : 0.0;
    const peak_nps = py.float(_or(py.get(raw, 'peak_nps', 0), 0));

    return _analysisResultKw({
      state,
      dp,
      dan_label: py.get(raw, 'dan_label', ''),
      dan_short: py.get(raw, 'dan_short', ''),
      sublevel: py.get(raw, 'sublevel', ''),
      confidence,
      dan_range,
      sr: _or(sr_value, overall_msd),
      family: py.get(raw, 'family', ''),
      mod: mod_label,
      corrections: py.get(raw, 'corrections', []),
      nps: avg_nps,
      peak_nps,
      nps_curve: _list(_or(py.get(raw, 'nps_curve'), [])),
      duration_s,
      note_count,
      warnings,
      debug: py.get(raw, 'debug', {}),
      overall_msd,
      primary_role: _str(_or(py.get(raw, 'primary_role', ''), '')),
      role_estimates: _dict(_or(py.get(raw, 'role_estimates', {}), {})),
      skillsets: _dict(_or(py.get(raw, 'skillsets', {}), {})),
      composite_dan: _str(_or(py.get(raw, 'composite_dan', ''), '')),
      bottleneck_role: _str(_or(py.get(raw, 'bottleneck_role', ''), '')),
      is_generalist: py.truthy(py.get(raw, 'is_generalist', false)),
      role_breakdown_text: _str(_or(py.get(raw, 'role_breakdown_text', ''), '')),
      celestial: py.get(raw, 'celestial'),
      signicial: py.get(raw, 'signicial'),
      shoegazer: py.get(raw, 'shoegazer'),
      ln_course: py.get(raw, 'ln_course'),
      ln_route: _str(_or(py.get(raw, 'ln_route', 'rice'), 'rice')),
      strain_graph: py.get(raw, 'strain_graph'),
      bpm: py.float(_or(py.get(raw, 'bpm', 0), 0)),
      bpm_min: py.int(_or(py.get(raw, 'bpm_min', 0), 0)),
      bpm_max: py.int(_or(py.get(raw, 'bpm_max', 0), 0)),
      bpm_common: py.int(_or(py.get(raw, 'bpm_common', 0), 0)),
      osu_sr: py.float(_or(py.get(raw, 'osu_sr', 0), 0)),
      od: py.float(_or(py.get(raw, 'od', 0), 0)),
      mode_7k: py.get(raw, 'mode') === '7k',
      tier_7k: _str(_or(py.get(raw, 'tier_7k', ''), '')),
      sublevel_7k: _str(_or(py.get(raw, 'sublevel_7k', ''), '')),
      dp_7k: py.float(_or(py.get(raw, 'dp_7k', 0), 0)),
      engine: _str(_or(py.get(raw, 'engine', 'isor'), 'isor')),
    });
  }
}
