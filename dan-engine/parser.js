// parser.js — osu!mania .osu parser: notes, LN events, rows, metadata and timing.
//
// JavaScript port of engine/dan_engine/parser.py (DanOverlay 3.1.0 by 8doul).
import * as py from './py.js';
import * as vfs from './vfs.js';

// ── JS-only helpers (Python builtins the parser relies on) ────────────────
/** str(value) for the values the parser can see (str, int, float, None, bool). */
function _pyStr(value) {
  if (typeof value === 'string') return value;
  if (value === null || value === undefined) return 'None';
  if (typeof value === 'boolean') return value ? 'True' : 'False';
  if (typeof value === 'number') return Number.isInteger(value) ? _strInt(value) : py.reprFloat(value);
  return String(value);
}

/** str() of a Python int (exact digits also beyond 2**53). */
function _strInt(n) {
  return Number.isSafeInteger(n) ? py.reprInt(n) : BigInt(n).toString();
}

/** s[:n] — Python slices by code point, JS strings are UTF-16. */
function _cpSlice(s, n) {
  let i = 0;
  for (let k = 0; k < n && i < s.length; k++) {
    const c = s.charCodeAt(i);
    const pair = c >= 0xd800 && c <= 0xdbff && i + 1 < s.length && (s.charCodeAt(i + 1) & 0xfc00) === 0xdc00;
    i += pair ? 2 : 1;
  }
  return s.slice(0, i);
}

const _VIDEO_EXTS = ['.mp4', '.avi', '.flv', '.mkv'];

// ── parser.py ─────────────────────────────────────────────────────────────
export function _safe_int(value, dflt = null) {
  try {
    return py.int(py.float(py.strip(_pyStr(value))));
  } catch (e) {
    if (e instanceof py.TypeError_ || e instanceof py.ValueError) return dflt;
    throw e;
  }
}

export function _safe_float(value, dflt = null) {
  try {
    return py.float(py.strip(_pyStr(value)));
  } catch (e) {
    if (e instanceof py.TypeError_ || e instanceof py.ValueError) return dflt;
    throw e;
  }
}

export function _safe_str(value, dflt = '') {
  if (value === null || value === undefined) return dflt;
  return py.strip(_pyStr(value));
}

export function _map_x_to_col(x, keycount, warnings) {
  if (keycount <= 0) return 0;
  let x_clamped = x;
  if (x < 0 || x >= 512) {
    warnings.push(`x out of bounds (${_strInt(x)}), clamped to [0,511]`);
    x_clamped = py.max2(0, py.min2(511, x));
  }
  const col = py.int(py.ifloordiv(x_clamped * keycount, 512));
  return py.max2(0, py.min2(keycount - 1, col));
}

export function _remap_col(col_in, keycount_in, keycount_out) {
  if (keycount_in <= 0 || keycount_out <= 0) return 0;
  const out = py.int(py.ifloordiv(col_in * keycount_out, keycount_in));
  return py.max2(0, py.min2(keycount_out - 1, out));
}

export function _build_rows(notes, tolerance_ms) {
  if (!py.truthy(notes)) return [];

  const tol = py.max2(0, py.int(tolerance_ms));
  const ordered = py.sortedBy(notes, (n) => [n[0], n[1]]);
  const rows = [];

  let t0 = py.int(ordered[0][0]);
  let cols = new Set([py.int(ordered[0][1])]);

  for (let i = 1; i < ordered.length; i++) {
    const t = py.int(ordered[i][0]);
    const c = py.int(ordered[i][1]);
    if (Math.abs(t - t0) <= tol) {
      cols.add(c);
    } else {
      rows.push({ t: t0, cols: py.sorted(cols) });
      t0 = t;
      cols = new Set([c]);
    }
  }

  rows.push({ t: t0, cols: py.sorted(cols) });
  return rows;
}

/**
 * Parses an osu!mania .osu file and returns a parsed dictionary for the pipeline.
 *
 * Key fields: notes (sorted [time_ms, col]), note_events (tap/ln_start/ln_end), bpm (first
 * uninherited timing point), metadata, od, hp, note_count, ln_count, ln_ratio, drain_time_s,
 * warnings, rejected (mode or keycount filter failed).
 */
export function parsear_osu_v2(
  ruta_archivo,
  output_keycount = null,
  include_ln_tails = true,
  enforce_mode_mania = true,
  enforce_keycount = null,
  row_tolerance_ms = 0,
  strict = false,
) {
  const lineas = py.textLines(vfs.readText(ruta_archivo, 'ignore'));

  const warnings = [];
  let bpm = 120.0;
  const _all_bpms = [];
  let mode = 3;
  let keycount_native = 4;
  let od = 8.0;
  let hp = 8.0;

  // Metadata
  let title = '';
  let artist = '';
  let version = '';
  let creator = '';
  let beatmap_id = '';
  let bg_file = '';

  let section = '';
  const notes = [];
  let note_events = [];
  let object_count = 0;
  let ln_count = 0;

  for (const raw of lineas) {
    const line = py.strip(raw);
    if (!line) continue;

    if (line.startsWith('[') && line.endsWith(']')) {
      section = line;
      continue;
    }

    if (line.startsWith('//')) continue;

    if (section === '[General]') {
      if (line.startsWith('Mode:')) {
        const val = _safe_int(py.split(line, ':', 1)[1], mode);
        mode = val === null ? mode : val;
      }
    } else if (section === '[Metadata]') {
      if (line.startsWith('Title:')) {
        title = _safe_str(py.split(line, ':', 1)[1]);
      } else if (line.startsWith('Artist:')) {
        artist = _safe_str(py.split(line, ':', 1)[1]);
      } else if (line.startsWith('Version:')) {
        version = _safe_str(py.split(line, ':', 1)[1]);
      } else if (line.startsWith('Creator:')) {
        creator = _safe_str(py.split(line, ':', 1)[1]);
      } else if (line.startsWith('BeatmapID:')) {
        beatmap_id = _safe_str(py.split(line, ':', 1)[1]);
      }
    } else if (section === '[Events]') {
      if (line.startsWith('0,0,') || line.startsWith('1,0,')) {
        const parts = line.split(',');
        if (parts.length >= 3) {
          const cand = py.strip(parts[2]).replaceAll('"', '');
          if (cand && !bg_file) {
            const low = cand.toLowerCase();
            if (!_VIDEO_EXTS.some((ext) => low.endsWith(ext))) bg_file = cand;
          }
        }
      }
    } else if (section === '[Difficulty]') {
      if (line.startsWith('CircleSize:')) {
        const cs = _safe_float(py.split(line, ':', 1)[1], py.float(keycount_native));
        if (cs !== null) keycount_native = py.max2(1, py.int(py.roundInt(cs)));
      } else if (line.startsWith('OverallDifficulty:')) {
        const val = _safe_float(py.split(line, ':', 1)[1], null);
        if (val !== null) od = val;
      } else if (line.startsWith('HPDrainRate:')) {
        const val = _safe_float(py.split(line, ':', 1)[1], null);
        if (val !== null) hp = val;
      }
    } else if (section === '[TimingPoints]') {
      const parts = line.split(',');
      if (parts.length >= 7) {
        const uninherited = _safe_int(parts[6], 0);
        const beat_length = _safe_float(parts[1], null);
        if (uninherited === 1 && beat_length !== null && beat_length > 0) {
          const _tp_bpm = 60000.0 / beat_length;
          if (!_all_bpms.length) bpm = _tp_bpm;
          _all_bpms.push(_tp_bpm);
        }
      }
    } else if (section === '[HitObjects]') {
      const parts = line.split(',');
      if (parts.length < 5) {
        warnings.push(`hitobject invalido: '${_cpSlice(line, 48)}'`);
        continue;
      }

      const x = _safe_int(parts[0], null);
      const t = _safe_int(parts[2], null);
      const obj_type = _safe_int(parts[3], 0);

      if (x === null || t === null) {
        warnings.push(`hitobject no parseable: '${_cpSlice(line, 48)}'`);
        continue;
      }

      const col_native = _map_x_to_col(x, keycount_native, warnings);
      const keycount_output = output_keycount !== null ? py.max2(1, py.int(output_keycount)) : keycount_native;
      const col = _remap_col(col_native, keycount_native, keycount_output);

      // obj_type is an integer-valued double: & works on its low 32 bits, like Python's & here
      const is_ln = (obj_type & 128) !== 0;
      const is_tap = (obj_type & 1) !== 0;
      object_count += 1;

      if (is_ln) {
        ln_count += 1;
        if (parts.length < 6) {
          warnings.push('LN missing endTime, treated as tap');
          notes.push([t, col]);
          note_events.push({ time_ms: t, col, event_type: 'tap', is_ln: false });
          continue;
        }

        const end_raw = py.split(parts[5], ':', 1)[0];
        const t_end = _safe_int(end_raw, null);
        if (t_end === null) {
          warnings.push('LN with invalid endTime, treated as tap');
          notes.push([t, col]);
          note_events.push({ time_ms: t, col, event_type: 'tap', is_ln: false });
          continue;
        }

        if (t_end <= t) {
          warnings.push(`LN endTime <= start (${_strInt(t_end)} <= ${_strInt(t)}), treated as tap`);
          notes.push([t, col]);
          note_events.push({ time_ms: t, col, event_type: 'tap', is_ln: false });
          continue;
        }

        notes.push([t, col]);
        note_events.push({
          time_ms: t,
          col,
          event_type: 'ln_start',
          is_ln: true,
          ln_start_ms: t,
          ln_end_ms: t_end,
          duration_ms: t_end - t,
        });

        if (py.truthy(include_ln_tails)) {
          notes.push([t_end, col]);
          note_events.push({
            time_ms: t_end,
            col,
            event_type: 'ln_end',
            is_ln: true,
            ln_start_ms: t,
            ln_end_ms: t_end,
            duration_ms: t_end - t,
          });
        }
      } else if (is_tap) {
        notes.push([t, col]);
        note_events.push({ time_ms: t, col, event_type: 'tap', is_ln: false });
      }
    }
  }

  const notes_sorted = py.sortedBy(notes, (n) => [n[0], n[1]]);

  let deduped = [];
  let dup_count = 0;
  let prev = null;
  for (const n of notes_sorted) {
    if (prev !== null && prev[0] === n[0] && prev[1] === n[1]) {
      dup_count += 1;
      continue;
    }
    deduped.push(n);
    prev = n;
  }
  if (dup_count) warnings.push(`${dup_count} duplicate notes deduplicated`);

  let rows = _build_rows(deduped, row_tolerance_ms);

  let rejected = false;
  if (py.truthy(enforce_mode_mania) && mode !== 3) rejected = true;
  if (enforce_keycount !== null && py.int(enforce_keycount) !== py.int(keycount_native)) rejected = true;

  if (rejected) {
    deduped = [];
    note_events = [];
    rows = [];
  }

  const note_count = deduped.length;
  const first_ms = deduped.length ? deduped[0][0] : 0;
  const last_ms = deduped.length ? deduped[deduped.length - 1][0] : 0;
  const drain_time_s = py.max2(0.0, (last_ms - first_ms) / 1000.0);
  const ln_ratio = ln_count / py.max2(object_count, 1);

  // BPM stats (rounded): min / max / most common
  const _bpms_rounded = _all_bpms.length ? _all_bpms.map((b) => py.roundInt(b)) : [py.roundInt(bpm)];
  const _bpm_min = py.min(_bpms_rounded);
  const _bpm_max = py.max(_bpms_rounded);
  const _bpm_common = new py.Counter(_bpms_rounded).mostCommon(1)[0][0];

  const out = {
    notes: deduped,
    note_events: py.sortedBy(note_events, (e) => [py.int(py.get(e, 'time_ms', 0)), py.int(py.get(e, 'col', 0))]),
    rows,
    bpm: py.float(bpm),
    bpm_min: py.int(_bpm_min),
    bpm_max: py.int(_bpm_max),
    bpm_common: py.int(_bpm_common),
    od: py.float(od),
    hp: py.float(hp),
    metadata: {
      title,
      artist,
      version,
      creator,
      beatmap_id,
      bg_file,
    },
    note_count,
    ln_count,
    ln_ratio,
    drain_time_s,
    warnings,
    rejected: Boolean(rejected),
    mode,
    keycount_native,
    keycount_output: output_keycount !== null ? py.max2(1, py.int(output_keycount)) : keycount_native,
  };

  if (py.truthy(strict) && warnings.length) {
    throw new py.ValueError('Parser strict mode: se detectaron warnings');
  }

  return out;
}

/** Legacy API: returns [notes, bpm]. */
export function parsear_osu(ruta_archivo) {
  const parsed = parsear_osu_v2(ruta_archivo, 4, true, false, null, 0, false);
  return [parsed.notes, parsed.bpm];
}
