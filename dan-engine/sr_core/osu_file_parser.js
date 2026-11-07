// osu_file_parser.js — minimal .osu reader (key count, columns, note times) for the Sunny SR core.
//
// JavaScript port of engine/dan_engine/sr_core/osu_file_parser.py (DanOverlay 3.1.0 by 8doul).
import * as py from '../py.js';
import * as vfs from '../vfs.js';

/** Python's StopIteration, raised by next(f) at the end of the file. */
export class StopIteration extends py.PyError {}

/**
 * A text file opened with open(path, 'r', encoding='utf-8'): `for line in f` / next(f) share one
 * position. Invalid UTF-8 raises UnicodeDecodeError from the next(f) that needs the bad chunk
 * (see decodeLinesChunked); valid files (the normal case) are decoded in one go.
 */
class TextFile {
  constructor(path) {
    this.error = null;
    try {
      this.lines = py.textLines(vfs.readText(path));
    } catch (e) {
      // CPython's position/timing of the error needs the raw bytes (vfs.readBytes, when available)
      if (!(e instanceof py.UnicodeDecodeError) || typeof vfs.readBytes !== 'function') throw e;
      ({ lines: this.lines, error: this.error } = decodeLinesChunked(vfs.readBytes(path)));
    }
    this.pos = 0;
  }

  next() {
    if (this.pos < this.lines.length) return this.lines[this.pos++];
    if (this.error) throw this.error;
    throw new StopIteration();
  }
}

/** io.TextIOWrapper._CHUNK_SIZE: text files are decoded 8192 bytes at a time. */
const TEXT_CHUNK = 8192;

/** Bytes at the end of b[start:end) forming an incomplete but valid UTF-8 prefix (kept pending). */
function incompleteTail(b, start, end) {
  for (let k = 1; k <= 3 && end - k >= start; k++) {
    const c = b[end - k];
    if (c >= 0x80 && c <= 0xbf) continue;
    const need = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : 0;
    if (need <= k - 1) return 0; // ASCII, invalid, or a complete sequence
    for (let i = 1; i < k; i++) {
      const cc = b[end - k + i];
      let lo = 0x80;
      let hi = 0xbf;
      if (i === 1) {
        if (c === 0xe0) lo = 0xa0;
        else if (c === 0xed) hi = 0x9f;
        else if (c === 0xf0) lo = 0x90;
        else if (c === 0xf4) hi = 0x8f;
      }
      if (cc < lo || cc > hi) return 0; // invalid: the decoder reports it right away
    }
    return k;
  }
  return 0;
}

/**
 * What `for line in open(p, encoding='utf-8')` yields for invalid UTF-8 bytes: TextIOWrapper feeds
 * 8192-byte chunks (plus the pending bytes of a sequence split by the previous chunk) to the
 * incremental decoder, so the UnicodeDecodeError positions are relative to that input, and the
 * error surfaces only when a line from the bad chunk is needed. Returns the complete lines before
 * the failing decode call and the error.
 */
function decodeLinesChunked(bytes) {
  const n = bytes.length;
  let start = 0; // first byte not decoded yet (the pending bytes start here)
  let error = null;
  for (let off = 0; off < n && !error; off += TEXT_CHUNK) {
    const end = Math.min(off + TEXT_CHUNK, n);
    const tail = incompleteTail(bytes, start, end);
    try {
      vfs.decodeUtf8(bytes.subarray(start, end - tail), 'strict');
      start = end - tail;
    } catch (e) {
      if (!(e instanceof py.UnicodeDecodeError)) throw e;
      error = e;
    }
  }
  if (!error && start < n) {
    // end of file inside a sequence: decode(b'', final=True) → 'unexpected end of data'
    try {
      vfs.decodeUtf8(bytes.subarray(start, n), 'strict');
    } catch (e) {
      if (!(e instanceof py.UnicodeDecodeError)) throw e;
      error = e;
    }
  }
  let text = vfs.decodeUtf8(bytes.subarray(0, start), 'strict');
  if (!error) return { lines: py.textLines(text.replace(/\r\n?/g, '\n')), error: null };
  if (text.endsWith('\r')) text = text.slice(0, -1); // held back by the newline decoder
  const lines = py.textLines(text.replace(/\r\n?/g, '\n'));
  if (lines.length && !lines[lines.length - 1].endsWith('\n')) lines.pop(); // incomplete last line
  return { lines, error };
}

/** Last character (code point) of a string: s[-1] (IndexError when empty). */
function lastChar(s) {
  if (s.length === 0) throw new py.IndexError('string index out of range');
  const c = s.charCodeAt(s.length - 1);
  if (c >= 0xdc00 && c <= 0xdfff && s.length >= 2) {
    const h = s.charCodeAt(s.length - 2);
    if (h >= 0xd800 && h <= 0xdbff) return s.slice(-2);
  }
  return s[s.length - 1];
}

export class parser {
  constructor(file_path) {
    this.file_path = file_path;
    this.od = -1;
    this.column_count = -1;
    this.columns = [];
    this.note_starts = [];
    this.note_ends = [];
    this.note_types = [];
  }

  process() {
    const f = new TextFile(this.file_path); // strict utf-8: UnicodeDecodeError propagates
    try {
      for (;;) {
        const line = f.next();
        this.read_metadata(f, line);

        const cc = this._read_column_count(line);
        if (cc !== -1) this.column_count = cc;

        this.od = 9;

        if (this.column_count !== -1) this._read_notes(f, line, this.column_count);
      }
    } catch (e) {
      if (!(e instanceof StopIteration)) throw e;
    }
  }

  read_metadata(f, line) {
    if (line.includes('[Metadata]')) {
      while (!line.includes('Source:')) {
        line = f.next();
        if (line.startsWith('[')) return;
      }
    }
  }

  _read_column_count(line) {
    if (!line.includes('CircleSize:')) return -1;
    let val = lastChar(py.strip(line));
    if (val === '0') val = '10';
    return py.int(py.float(val));
  }

  _read_notes(f, line, column_count) {
    if (!line.includes('[HitObjects]')) return;
    line = f.next();
    while (line !== null) {
      line = py.strip(line);
      if (!line) {
        line = f.next();
        continue;
      }
      this._parse_hit_object(line, column_count);
      line = f.next();
    }
  }

  _parse_hit_object(line, column_count) {
    const params = line.split(',');
    const column_width = py.ifloordiv(512, column_count);
    this.columns.push(py.ifloordiv(py.int(py.float(params[0])), column_width));
    this.note_starts.push(py.int(py.at(params, 2)));
    this.note_types.push(py.int(py.at(params, 3)));
    this.note_ends.push(py.int(py.at(params, 5).split(':')[0]));
  }

  get_parsed_data() {
    return [
      this.column_count,
      this.columns,
      this.note_starts,
      this.note_ends,
      this.note_types,
      this.od,
    ];
  }
}
