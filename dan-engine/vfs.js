// vfs.js — in-memory stand-in for the file-system calls the DanOverlay engine makes
// (open/read, os.path.exists/isfile, os.stat mtime, tempfile, os.remove).
//
// The overlay puts the current .osu text here (fetched from tosu) and the config
// JSONs (fetched from the overlay folder) under RES_ROOT, which plays the role of
// sys._MEIPASS for the ported modules. Paths are plain strings; Windows paths
// coming from tosu are normalised to forward slashes.
import { OSError_, UnicodeDecodeError, reprStr } from './py.js';

export const RES_ROOT = '/res';
const files = new Map(); // normalised path -> { data: string | Uint8Array, mtimeNs }
let tempSeq = 0;
let clock = 1;

const fatalDecoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
const replaceDecoder = new TextDecoder('utf-8', { fatal: false, ignoreBOM: true });

/**
 * bytes.decode('utf-8', errors) with CPython's error ranges (maximal subparts — the same
 * ranges the WHATWG decoder uses) for errors = 'strict' | 'replace' | 'ignore'.
 * The BOM is kept as U+FEFF, like Python's 'utf-8' codec.
 */
export function decodeUtf8(bytes, errors = 'strict') {
  try {
    return fatalDecoder.decode(bytes);
  } catch {
    /* invalid UTF-8: handled below */
  }
  if (errors === 'replace') return replaceDecoder.decode(bytes);
  const parts = [];
  let cps = [];
  const push = (cp) => {
    cps.push(cp);
    if (cps.length >= 4096) {
      parts.push(String.fromCodePoint(...cps));
      cps = [];
    }
  };
  const fail = (start, end, reason) => {
    if (errors === 'ignore') return;
    const what = end - start === 1
      ? `byte 0x${bytes[start].toString(16).padStart(2, '0')} in position ${start}`
      : `bytes in position ${start}-${end - 1}`;
    throw new UnicodeDecodeError(`'utf-8' codec can't decode ${what}: ${reason}`);
  };
  const n = bytes.length;
  let i = 0;
  while (i < n) {
    const b = bytes[i];
    if (b < 0x80) {
      push(b);
      i++;
      continue;
    }
    let need;
    let cp;
    let lower = 0x80;
    let upper = 0xbf;
    if (b >= 0xc2 && b <= 0xdf) {
      need = 1;
      cp = b & 0x1f;
    } else if (b >= 0xe0 && b <= 0xef) {
      need = 2;
      cp = b & 0x0f;
      if (b === 0xe0) lower = 0xa0;
      if (b === 0xed) upper = 0x9f;
    } else if (b >= 0xf0 && b <= 0xf4) {
      need = 3;
      cp = b & 0x07;
      if (b === 0xf0) lower = 0x90;
      if (b === 0xf4) upper = 0x8f;
    } else {
      fail(i, i + 1, 'invalid start byte');
      i++;
      continue;
    }
    let j = i + 1;
    let bad = false;
    for (let k = 0; k < need; k++, j++) {
      if (j >= n) {
        fail(i, n, 'unexpected end of data');
        bad = true;
        j = n;
        break;
      }
      const c = bytes[j];
      if (c < lower || c > upper) {
        fail(i, j, 'invalid continuation byte');
        bad = true;
        break;
      }
      lower = 0x80;
      upper = 0xbf;
      cp = (cp << 6) | (c & 0x3f);
    }
    if (!bad) push(cp);
    i = j;
  }
  if (cps.length) parts.push(String.fromCodePoint(...cps));
  return parts.join('');
}

export function norm(p) {
  let s = String(p).replace(/\\/g, '/');
  s = s.replace(/\/+/g, '/');
  const parts = [];
  for (const seg of s.split('/')) {
    if (seg === '.' || (seg === '' && parts.length)) continue;
    const last = parts[parts.length - 1];
    if (seg === '..' && parts.length === 1 && last === '') continue; // '/..' is '/' (normpath)
    if (seg === '..' && parts.length && last !== '..' && last !== '') parts.pop();
    else parts.push(seg);
  }
  return parts.join('/') || '/';
}

/** Store a file: a string (already decoded text) or raw bytes (Uint8Array / ArrayBuffer). */
export function putFile(p, data, mtimeNs) {
  const stored = data instanceof ArrayBuffer ? new Uint8Array(data) : ArrayBuffer.isView(data) ? data : String(data);
  files.set(norm(p), { data: stored, mtimeNs: mtimeNs ?? clock++ });
}
export function removeFile(p) {
  files.delete(norm(p));
}
export function hasFile(p) {
  return files.has(norm(p));
}
/**
 * open(p, encoding='utf-8', errors=errors).read(): decodes stored bytes like Python and applies
 * text mode's universal newline translation ('\r\n' and '\r' → '\n').
 */
export function readText(p, errors = 'strict') {
  const f = files.get(norm(p));
  if (!f) throw new OSError_(`[Errno 2] No such file or directory: ${reprStr(String(p))}`);
  const text = typeof f.data === 'string' ? f.data : decodeUtf8(f.data, errors);
  return text.includes('\r') ? text.replace(/\r\n?/g, '\n') : text;
}
/** open(p, 'rb').read(): the stored bytes (text stored as a string comes back UTF-8 encoded). */
export function readBytes(p) {
  const f = files.get(norm(p));
  if (!f) throw new OSError_(`[Errno 2] No such file or directory: ${reprStr(String(p))}`);
  return typeof f.data === 'string' ? new TextEncoder().encode(f.data) : f.data;
}
export function statMtimeNs(p) {
  const f = files.get(norm(p));
  if (!f) throw new OSError_(`[Errno 2] No such file or directory: ${reprStr(String(p))}`);
  return f.mtimeNs;
}
/** tempfile.mkstemp-like unique path (the ported code writes/reads/removes it through this module). */
export function tempPath(prefix = 'tmp', suffix = '') {
  tempSeq += 1;
  return `/tmp/${prefix}${tempSeq}${suffix}`;
}
export function clear(pred = () => true) {
  for (const k of [...files.keys()]) if (pred(k)) files.delete(k);
}

/** os.path subset with forward-slash semantics (enough for the engine's path handling). */
export const path = {
  sep: '/',
  join: (...parts) => norm(parts.filter((x) => x !== '' && x !== null && x !== undefined).join('/')),
  dirname: (p) => {
    const s = norm(p);
    const k = s.lastIndexOf('/');
    return k <= 0 ? (k === 0 ? '/' : '') : s.slice(0, k);
  },
  basename: (p) => {
    const s = String(p).replace(/\\/g, '/');
    return s.slice(s.lastIndexOf('/') + 1);
  },
  splitext: (p) => {
    const b = path.basename(p);
    const k = b.lastIndexOf('.');
    if (k <= 0) return [String(p), ''];
    const s = String(p);
    return [s.slice(0, s.length - (b.length - k)), b.slice(k)];
  },
  abspath: (p) => norm(p),
  normpath: (p) => norm(p),
  exists: (p) => files.has(norm(p)) || [...files.keys()].some((k) => k.startsWith(`${norm(p)}/`)),
  isfile: (p) => files.has(norm(p)),
  isdir: (p) => [...files.keys()].some((k) => k.startsWith(`${norm(p)}/`)),
};

/** Python `sys` as the ported modules see it (frozen build with resources under RES_ROOT). */
export const sys = { frozen: true, _MEIPASS: RES_ROOT, platform: 'win32' };

/** Config files the engine reads (relative to RES_ROOT). */
export const CONFIG_FILES = [
  'celestial_profiles.json', 'celestial_ruler.json', 'family_profiles.json', 'ln_course_profiles.json',
  'role_scales.json', 'shoegazer_profiles.json', 'signicial_profiles.json', 'sr_means.json',
  'sr_means_7k.json', 'vsrg_ridge_model.json',
];
