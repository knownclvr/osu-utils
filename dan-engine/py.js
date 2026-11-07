// py.js — CPython 3.13/3.14 semantics the DanOverlay engine relies on, reproduced
// bit for bit in JavaScript: round(), float formatting, sum() (Neumaier),
// // and %, int()/float() parsing, str.strip()/split(), math domain errors,
// sorted()/max()/min() with keys, and Python's exception classes.
//
// Rule of thumb for ported code: every Python operation whose result or error
// behaviour differs from plain JS goes through a helper from this file.
//
// math.exp / cos / log / log2 / log1p / pow and float ** float return the Windows UCRT's
// exact bits (np.crt* — CPython calls the CRT on the reference machine), with CPython 3.14's
// special cases and error messages on top.
import { crtExp, crtCos, crtPow, crtLog, crtLog2, crtLog1p } from './np.js';

// ── exceptions ────────────────────────────────────────────────────────
export class PyError extends Error {
  constructor(message = '') {
    super(message);
    this.name = new.target.name;
  }
}
export class ValueError extends PyError {}
export class ZeroDivisionError extends PyError {}
export class KeyError extends PyError {}
export class IndexError extends PyError {}
export class TypeError_ extends PyError {}  // Python TypeError (JS already has one)
export class AttributeError extends PyError {}
export class OverflowError extends PyError {}
export class OSError_ extends PyError {}    // Python OSError / FileNotFoundError
export class JSONDecodeError extends ValueError {}
export class UnicodeDecodeError extends ValueError {}

/** `except Exception:` in Python also catches JS runtime errors raised by ported code. */
export const isPyException = (e) => e instanceof Error;

// ── exact float decomposition (BigInt) ────────────────────────────────
const F64 = new Float64Array(1);
const U64 = new BigUint64Array(F64.buffer);

/** x (finite, non-zero magnitude handled) → { neg, mant: BigInt, exp } with |x| = mant * 2**exp */
function decompose(x) {
  F64[0] = x;
  const bits = U64[0];
  const neg = (bits >> 63n) === 1n;
  const e = Number((bits >> 52n) & 0x7ffn);
  let mant = bits & 0xfffffffffffffn;
  let exp;
  if (e === 0) {
    exp = -1074;
  } else {
    mant |= 0x10000000000000n;
    exp = e - 1075;
  }
  return { neg, mant, exp };
}

const pow10 = (n) => 10n ** BigInt(n);

/**
 * Round |x| * 10**nd to an integer with round-half-even on the exact binary value
 * (what CPython's dtoa mode 3 does). Returns { neg, q } with q a BigInt.
 */
function scaledRoundHalfEven(x, nd) {
  const { neg, mant, exp } = decompose(x);
  let num = mant;
  let den = 1n;
  if (exp >= 0) num <<= BigInt(exp);
  else den <<= BigInt(-exp);
  if (nd >= 0) num *= pow10(nd);
  else den *= pow10(-nd);
  let q = num / den;
  const r = num - q * den;
  const twice = 2n * r;
  if (twice > den || (twice === den && (q & 1n) === 1n)) q += 1n;
  return { neg, q };
}

/** Python's type name of a JS value, for error messages. */
export function typeName(x) {
  if (x === null || x === undefined) return 'NoneType';
  if (typeof x === 'string') return 'str';
  if (typeof x === 'boolean') return 'bool';
  if (typeof x === 'number') return Number.isInteger(x) ? 'int' : 'float';
  if (Array.isArray(x)) return 'list';
  if (x instanceof Set) return 'set';
  if (typeof x === 'function') return 'function';
  return 'dict';
}

/** round() of something that isn't a number → TypeError like Python. */
function roundOperand(x) {
  if (typeof x === 'number') return x;
  if (typeof x === 'boolean') return x ? 1 : 0;
  throw new TypeError_(`type ${typeName(x)} doesn't define __round__ method`);
}

/** Python round(x) (no ndigits): nearest integer, ties to even. Returns a JS number. */
export function roundInt(x) {
  x = roundOperand(x);
  if (!Number.isFinite(x)) {
    if (Number.isNaN(x)) throw new ValueError('cannot convert float NaN to integer');
    throw new OverflowError('cannot convert float infinity to integer');
  }
  const t = Math.trunc(x);
  const diff = x - t; // exact
  let r = t;
  if (Math.abs(diff) > 0.5) r = t + Math.sign(x);
  else if (Math.abs(diff) === 0.5) r = t % 2 === 0 ? t : t + Math.sign(x);
  return r === 0 ? 0 : r; // Python ints have no -0
}

/** Python round(x, ndigits) for floats: correctly rounded, ties to even, keeps -0.0. */
export function round(x, ndigits) {
  if (ndigits === undefined || ndigits === null) return roundInt(x);
  x = roundOperand(x);
  if (!Number.isFinite(x) || x === 0) return x;
  if (ndigits > 323) return x;
  if (ndigits < -308) return x < 0 ? -0 : 0;
  if (ndigits >= 0 && ndigits <= 22) {
    // fast path: when |x|*10**n is safely away from a .5 tie, the double product
    // decides the direction and q / 10**n is the correctly rounded result
    const p = 10 ** ndigits; // exact for n <= 22
    const y = Math.abs(x) * p;
    if (y < 4503599627370496) {
      const fl = Math.floor(y);
      const frac = y - fl;
      if (Math.abs(frac - 0.5) > y * 8.881784197001252e-16 + 1e-300) {
        const v = (frac > 0.5 ? fl + 1 : fl) / p;
        return x < 0 ? -v : v;
      }
    }
  }
  const { neg, q } = scaledRoundHalfEven(x, ndigits);
  const s = `${neg ? '-' : ''}${q.toString()}e${-ndigits}`;
  const v = Number(s);
  return v === 0 ? (neg ? -0 : 0) : v;
}

/** Python round(int_value, ndigits) for ints stays an int. */
export function roundIntValue(n, ndigits) {
  if (ndigits >= 0) return n;
  const f = 10 ** -ndigits;
  const q = Math.floor(n / f);
  const r = n - q * f;
  let res = q;
  if (2 * r > f || (2 * r === f && q % 2 !== 0)) res = q + 1;
  return res * f;
}

// ── formatting ────────────────────────────────────────────────────────
/** Digits of |x| rounded to `prec` decimals (Python 'f' format, ties to even). */
function fixedDigits(x, prec) {
  const { neg, q } = scaledRoundHalfEven(x, prec);
  let s = q.toString();
  if (prec > 0) {
    if (s.length <= prec) s = '0'.repeat(prec - s.length + 1) + s;
    s = `${s.slice(0, s.length - prec)}.${s.slice(s.length - prec)}`;
  }
  return { neg, s };
}

/** format(x, spec) for the specs the engine uses: [+][width]'.Nf' / '.N%' / 'd'. */
export function format(x, spec) {
  const m = /^(\+)?(\d+)?(?:\.(\d+))?([f%d])$/.exec(spec);
  if (!m) throw new ValueError(`format spec not supported by py.js: ${spec}`);
  const plus = !!m[1];
  const width = m[2] ? Number(m[2]) : 0;
  const type = m[4];
  let out;
  if (type === 'd') {
    out = String(x);
    if (plus && x >= 0) out = `+${out}`;
  } else {
    let v = type === '%' ? x * 100 : x;
    const prec = m[3] !== undefined ? Number(m[3]) : 6;
    if (Number.isNaN(v)) out = 'nan';
    else if (!Number.isFinite(v)) out = v > 0 ? 'inf' : '-inf';
    else {
      const neg = v < 0 || Object.is(v, -0);
      const { s } = fixedDigits(Math.abs(v), prec);
      out = (neg ? '-' : plus ? '+' : '') + s;
    }
    if (type === '%') out += '%';
  }
  return out.length < width ? ' '.repeat(width - out.length) + out : out;
}

/** repr()/str() of a Python float (shortest round-trip, Python's layout). */
export function reprFloat(x) {
  if (Number.isNaN(x)) return 'nan';
  if (!Number.isFinite(x)) return x > 0 ? 'inf' : '-inf';
  if (x === 0) return Object.is(x, -0) ? '-0.0' : '0.0';
  const neg = x < 0;
  const ax = Math.abs(x);
  // shortest round-trip digits via JS, then re-layout the way Python does
  let [mant, e] = ax.toExponential().split('e');
  const digits = mant.replace('.', '');
  const exp = Number(e); // value = d.ddd * 10**exp
  let out;
  if (exp < -4 || exp >= 16) {
    const d = digits.length > 1 ? `${digits[0]}.${digits.slice(1)}` : digits;
    out = `${d}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`;
  } else if (exp >= digits.length - 1) {
    out = `${digits}${'0'.repeat(exp - digits.length + 1)}.0`;
  } else if (exp >= 0) {
    out = `${digits.slice(0, exp + 1)}.${digits.slice(exp + 1)}`;
  } else {
    out = `0.${'0'.repeat(-exp - 1)}${digits}`;
  }
  return neg ? `-${out}` : out;
}

/** str() of a Python int. */
export const reprInt = (n) => String(n === 0 ? 0 : n);

// characters str.isprintable() rejects (besides ASCII controls): Cc Cf Cs Co Cn Zl Zp Zs (except ' ')
const NON_PRINTABLE = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

/** repr() of a Python str (quote choice and escapes as CPython's unicode_repr). */
export function reprStr(s) {
  const quote = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = quote;
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (ch === quote || ch === '\\') out += `\\${ch}`;
    else if (ch === '\t') out += '\\t';
    else if (ch === '\n') out += '\\n';
    else if (ch === '\r') out += '\\r';
    else if (cp < 0x20 || cp === 0x7f) out += `\\x${cp.toString(16).padStart(2, '0')}`;
    else if (cp < 0x7f || ch === ' ' || !NON_PRINTABLE.test(ch)) out += ch;
    else if (cp <= 0xff) out += `\\x${cp.toString(16).padStart(2, '0')}`;
    else if (cp <= 0xffff) out += `\\u${cp.toString(16).padStart(4, '0')}`;
    else out += `\\U${cp.toString(16).padStart(8, '0')}`;
  }
  return out + quote;
}

// ── float32 → the double Python sees after json.loads(msd.exe output) ─
const F32 = new Float32Array(1);
/**
 * Shortest decimal that round-trips as float32, picked like ryu (serde_json's f32 printer:
 * shortest length, then closest to the exact value, ties to even), parsed back as a double
 * — i.e. the value Python sees after json.loads(msd.exe output).
 */
export function f32JsonValue(x) {
  F32[0] = x;
  const f = F32[0];
  if (!Number.isFinite(f) || f === 0) return f;
  const neg = f < 0;
  const af = Math.abs(f);
  const mag = Math.floor(Math.log10(af));
  for (let p = 1; p <= 9; p++) {
    for (const m of [mag, mag + 1]) {           // guard against log10 rounding at powers of 10
      const nd = p - 1 - m;                      // decimal places giving p significant digits
      const { q } = scaledRoundHalfEven(af, nd); // nearest p-digit decimal, ties to even
      if (q.toString().length !== p) continue;
      for (const cand of [q, q - 1n, q + 1n]) {  // neighbours cover asymmetric intervals at 2**k
        const v = Number(`${cand.toString()}e${-nd}`);
        if (Math.fround(v) === af) return neg ? -v : v;
      }
    }
  }
  return f;
}

// ── arithmetic with Python semantics ──────────────────────────────────
/** a / b raising ZeroDivisionError like Python (ints and floats). */
export function div(a, b) {
  if (b === 0) throw new ZeroDivisionError('division by zero');
  return a / b;
}

/** CPython float_divmod: returns [floordiv, mod] exactly like Python for floats and ints. */
export function divmod(vx, wx) {
  if (wx === 0) throw new ZeroDivisionError('integer division or modulo by zero');
  let mod = vx % wx; // C fmod
  let d = (vx - mod) / wx;
  if (mod) {
    if ((wx < 0) !== (mod < 0)) {
      mod += wx;
      d -= 1.0;
    }
  } else {
    mod = wx < 0 ? -0 : 0;
  }
  let floordiv;
  if (d) {
    floordiv = Math.floor(d);
    if (d - floordiv > 0.5) floordiv += 1.0;
  } else {
    floordiv = (vx / wx) < 0 || Object.is(vx / wx, -0) ? -0 : 0;
  }
  return [floordiv, mod];
}
export const floordiv = (a, b) => divmod(a, b)[0];
export const mod = (a, b) => divmod(a, b)[1];
/** Python int // int (both integers): exact floor division. */
export function ifloordiv(a, b) {
  if (b === 0) throw new ZeroDivisionError('integer division or modulo by zero');
  return Math.floor(a / b) === 0 ? 0 : Math.floor(a / b);
}

/** DOUBLE_IS_ODD_INTEGER(x): fmod(fabs(x), 2.0) == 1.0 */
const isOddInteger = (x) => Math.abs(x) % 2 === 1;

/**
 * Python float ** float (CPython float_pow; also int ** negative int): its special cases and
 * errors, then the CRT's pow() on |v| (np.crtPow). Python returns a complex number for a
 * negative base with a fractional exponent; that raises ValueError here (never relied on).
 */
export function pow(v, w) {
  if (w === 0) return 1.0; // v**0 is 1, even 0**0 and nan**0
  if (v !== v) return v; // nan**w = nan
  if (w !== w) return v === 1 ? 1.0 : w; // v**nan = nan, but 1**nan = 1
  if (w === Infinity || w === -Infinity) {
    const av = Math.abs(v);
    if (av === 1) return 1.0;
    return (w > 0) === (av > 1) ? Infinity : 0.0;
  }
  if (v === Infinity || v === -Infinity) {
    const odd = isOddInteger(w);
    if (w > 0) return odd ? v : Math.abs(v);
    return odd ? (v < 0 ? -0.0 : 0.0) : 0.0;
  }
  if (v === 0) {
    if (w < 0) throw new ZeroDivisionError('zero to a negative power');
    return isOddInteger(w) ? v : 0.0;
  }
  let negate = false;
  if (v < 0) {
    if (w !== Math.floor(w)) {
      throw new ValueError('negative number cannot be raised to a fractional power (complex result)');
    }
    v = -v;
    negate = isOddInteger(w);
  }
  if (v === 1) return negate ? -1.0 : 1.0;
  const r = crtPow(v, w);
  if (r === Infinity) throw new OverflowError("(34, 'Result too large')"); // ERANGE
  return negate ? -r : r;
}

/** Python sum(iterable, start=0) — CPython >= 3.12 (Neumaier-compensated float path). */
export function sum(iterable, start = 0) {
  let first = true;
  let f = start;
  let c = 0.0;
  let started = false;
  for (const x of iterable) {
    if (first && start === 0) {
      // int 0 + first item: exact, becomes the running float value
      f = 0 + x;
      first = false;
      started = true;
      continue;
    }
    first = false;
    started = true;
    const t = f + x;
    if (Math.abs(f) >= Math.abs(x)) c += (f - t) + x;
    else c += (x - t) + f;
    f = t;
  }
  if (!started) return start;
  if (c && Number.isFinite(c)) f += c;
  return f;
}

/** math.fsum — exact sum, correctly rounded (via BigInt). */
export function fsum(iterable) {
  let acc = 0n; // scaled by 2**1074
  let special = null;
  for (const x of iterable) {
    if (!Number.isFinite(x)) {
      special = special === null ? x : special + x;
      continue;
    }
    if (x === 0) continue;
    const { neg, mant, exp } = decompose(x);
    const v = mant << BigInt(exp + 1074);
    acc += neg ? -v : v;
  }
  if (special !== null) {
    if (Number.isNaN(special)) throw new ValueError('-inf + inf in fsum');
    return special;
  }
  return bigScaledToDouble(acc, 1074);
}

/** Correctly rounded double of n / 2**shift (n BigInt, may be negative). */
function bigScaledToDouble(n, shift) {
  if (n === 0n) return 0;
  const neg = n < 0n;
  let a = neg ? -n : n;
  const bitlen = a.toString(2).length;
  // keep 53 significant bits (plus guard) relative to the value's exponent
  const e2 = bitlen - 1 - shift; // floor(log2(value))
  let keep = 53;
  if (e2 < -1022) keep = 53 - (-1022 - e2); // subnormal range
  let result;
  if (keep <= 0) {
    // below half of the smallest subnormal → rounds to 0 or min subnormal
    result = 0;
    const halfMin = 1n << BigInt(shift - 1075);
    if (shift - 1075 >= 0 && a > halfMin) result = 5e-324;
  } else {
    const drop = bitlen - keep;
    let q = drop > 0 ? a >> BigInt(drop) : a << BigInt(-drop);
    if (drop > 0) {
      const rem = a - (q << BigInt(drop));
      const half = 1n << BigInt(drop - 1);
      if (rem > half || (rem === half && (q & 1n) === 1n)) q += 1n;
    }
    result = Number(q) * 2 ** (drop - shift);
  }
  return neg ? -result : result;
}

// ── math module with Python's domain errors ───────────────────────────
export const math = {
  pi: Math.PI,
  e: Math.E,
  inf: Infinity,
  floor(x) {
    if (Number.isNaN(x)) throw new ValueError('cannot convert float NaN to integer');
    if (!Number.isFinite(x)) throw new OverflowError('cannot convert float infinity to integer');
    const r = Math.floor(x);
    return r === 0 ? 0 : r;
  },
  ceil(x) {
    if (Number.isNaN(x)) throw new ValueError('cannot convert float NaN to integer');
    if (!Number.isFinite(x)) throw new OverflowError('cannot convert float infinity to integer');
    const r = Math.ceil(x);
    return r === 0 ? 0 : r;
  },
  sqrt(x) {
    const r = Math.sqrt(x); // correctly rounded, like the CRT
    if (r !== r && x === x) throw new ValueError(`expected a nonnegative input, got ${reprFloat(x)}`);
    return r;
  },
  /** math.log(x[, base]): num / den with the CRT log (math_log_impl + loghelper for float args). */
  log(x, base) {
    const num = mathLogOf(x, crtLog);
    if (base === undefined) return num;
    const den = mathLogOf(base, crtLog);
    if (den === 0) throw new ZeroDivisionError('division by zero');
    return num / den;
  },
  log2(x) {
    return mathLogOf(x, crtLog2);
  },
  /** (not CRT-exact: Math.log10 — the engine never calls math.log10) */
  log10(x) {
    return mathLogOf(x, Math.log10);
  },
  log1p(x) {
    if (x !== x || x === Infinity) return x;
    if (!(x > -1)) throw new ValueError(`expected argument value > -1, got ${reprFloat(x)}`);
    return crtLog1p(x); // log1p(±0) = ±0 like CPython's m_log1p
  },
  exp(x) {
    const r = crtExp(x);
    if (r === Infinity && Number.isFinite(x)) throw new OverflowError('math range error');
    return r;
  },
  cos(x) {
    if (x === Infinity || x === -Infinity) throw new ValueError(`expected a finite input, got ${reprFloat(x)}`);
    return crtCos(x);
  },
  // sin / tanh / atan / log10 are not CRT-exact (Math.*): the engine never calls them
  sin(x) {
    if (x === Infinity || x === -Infinity) throw new ValueError(`expected a finite input, got ${x > 0 ? 'inf' : '-inf'}`);
    return Math.sin(x);
  },
  tanh: Math.tanh,
  atan: Math.atan,
  fabs: Math.abs,
  isfinite: Number.isFinite,
  isnan: Number.isNaN,
  isinf: (x) => x === Infinity || x === -Infinity,
  copysign: (a, b) => ((b < 0 || Object.is(b, -0)) ? -Math.abs(a) : Math.abs(a)),
  /** math.pow(x, y): CPython's math_pow_impl (IEEE specials itself, else the CRT pow()). */
  pow(x, y) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      if (x !== x) return y === 0 ? 1.0 : x; // nan**0 = 1
      if (y !== y) return x === 1 ? 1.0 : y; // 1**nan = 1
      if (x === Infinity || x === -Infinity) {
        const odd = Number.isFinite(y) && isOddInteger(y);
        if (y > 0) return odd ? x : Math.abs(x);
        if (y === 0) return 1.0;
        return odd ? (x < 0 ? -0.0 : 0.0) : 0.0;
      }
      const ax = Math.abs(x); // y is ±inf
      if (ax === 1) return 1.0;
      if (y > 0 && ax > 1) return y;
      if (y < 0 && ax < 1) return -y;
      return 0.0;
    }
    const r = crtPow(x, y);
    if (r !== r) throw new ValueError('math domain error'); // (-ve) ** non-integer
    if (r === Infinity || r === -Infinity) {
      if (x === 0) throw new ValueError('math domain error'); // 0 ** negative
      throw new OverflowError('math range error');
    }
    return r;
  },
};

/** loghelper(x, m_log / m_log2 / m_log10) for a float x (math_1 with CPython 3.14's message). */
function mathLogOf(x, fn) {
  if (x !== x || x === Infinity) return x;
  if (!(x > 0)) throw new ValueError(`expected a positive input, got ${reprFloat(x)}`);
  return fn(x);
}

// ── conversions ───────────────────────────────────────────────────────
const PY_WS = ' \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f\x85\xa0                　';
const isWs = (ch) => PY_WS.includes(ch);

/** str.strip() with no arguments (Python's whitespace set; unlike JS trim it keeps U+FEFF). */
export function strip(s, chars) {
  if (chars !== undefined) {
    let a = 0;
    let b = s.length;
    while (a < b && chars.includes(s[a])) a++;
    while (b > a && chars.includes(s[b - 1])) b--;
    return s.slice(a, b);
  }
  let a = 0;
  let b = s.length;
  while (a < b && isWs(s[a])) a++;
  while (b > a && isWs(s[b - 1])) b--;
  return s.slice(a, b);
}
export function lstrip(s, chars) {
  let a = 0;
  while (a < s.length && (chars === undefined ? isWs(s[a]) : chars.includes(s[a]))) a++;
  return s.slice(a);
}
export function rstrip(s, chars) {
  let b = s.length;
  while (b > 0 && (chars === undefined ? isWs(s[b - 1]) : chars.includes(s[b - 1]))) b--;
  return s.slice(0, b);
}

/** str.split() with no separator: runs of whitespace, no empty strings. */
export function splitWs(s, maxsplit = -1) {
  const out = [];
  let i = 0;
  const n = s.length;
  while (i < n) {
    while (i < n && isWs(s[i])) i++;
    if (i >= n) break;
    if (maxsplit >= 0 && out.length === maxsplit) {
      out.push(rstrip(s.slice(i)));
      return out;
    }
    let j = i;
    while (j < n && !isWs(s[j])) j++;
    out.push(s.slice(i, j));
    i = j;
  }
  return out;
}

/** str.split(sep, maxsplit) with an explicit separator. */
export function split(s, sep, maxsplit = -1) {
  if (sep === undefined || sep === null) return splitWs(s, maxsplit);
  if (maxsplit < 0) return s.split(sep);
  const out = [];
  let rest = s;
  while (out.length < maxsplit) {
    const k = rest.indexOf(sep);
    if (k < 0) break;
    out.push(rest.slice(0, k));
    rest = rest.slice(k + sep.length);
  }
  out.push(rest);
  return out;
}

/** Text of a file read in Python text mode (universal newlines) as a list of lines with '\n'. */
export function textLines(text) {
  const norm = text.replace(/\r\n?/g, '\n');
  const lines = [];
  let start = 0;
  for (let i = 0; i < norm.length; i++) {
    if (norm[i] === '\n') {
      lines.push(norm.slice(start, i + 1));
      start = i + 1;
    }
  }
  if (start < norm.length) lines.push(norm.slice(start));
  return lines;
}

/** int(x): from a float (truncates, errors on nan/inf) or from a str (Python literal rules). */
export function int(x, base = 10) {
  if (typeof x === 'number') {
    if (Number.isNaN(x)) throw new ValueError('cannot convert float NaN to integer');
    if (!Number.isFinite(x)) throw new OverflowError('cannot convert float infinity to integer');
    const t = Math.trunc(x);
    return t === 0 ? 0 : t;
  }
  if (typeof x === 'boolean') return x ? 1 : 0;
  if (typeof x !== 'string') throw new TypeError_(`int() argument must be a string, a bytes-like object or a real number, not '${typeName(x)}'`);
  const s = asciiNumber(x).replace(ASCII_WS_EDGES, '');
  const re = base === 10 ? /^[+-]?\d+(_\d+)*$/ : null;
  if (!re || !re.test(s)) throw new ValueError(`invalid literal for int() with base ${base}: ${reprStr(x).slice(0, 200)}`);
  const v = Number(s.replace(/_/g, '')); // beyond 2**53: the double nearest to Python's exact int
  return v === 0 ? 0 : v;
}

// int()/float() first turn Unicode decimal digits into ASCII and Unicode spaces into ' '
// (_PyUnicode_TransformDecimalAndSpaceToASCII), then parse with ASCII rules.
const ND = /\p{Nd}/u;
const ASCII_WS_EDGES = /^[ \t\n\r\x0b\x0c]+|[ \t\n\r\x0b\x0c]+$/g;
function asciiNumber(s) {
  if (!/[^\x00-\x7e]/.test(s)) return s;
  let out = '';
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (cp < 127) out += ch;
    else if (isWs(ch)) out += ' ';
    else if (ND.test(ch)) {
      // decimal digits are encoded in runs of ten, 0..9, one run after another
      let start = cp;
      while (ND.test(String.fromCodePoint(start - 1))) start--;
      out += String((cp - start) % 10);
    } else return `${out}?`;
  }
  return out;
}

/** float(x): from a str with Python's grammar (inf/nan/underscores/whitespace), or a number. */
export function float(x) {
  if (typeof x === 'number') return x;
  if (typeof x === 'boolean') return x ? 1.0 : 0.0;
  if (typeof x !== 'string') throw new TypeError_(`float() argument must be a string or a real number, not '${typeName(x)}'`);
  const s = asciiNumber(x).replace(ASCII_WS_EDGES, '');
  const low = s.toLowerCase();
  if (/^[+-]?(inf|infinity)$/.test(low)) return low.startsWith('-') ? -Infinity : Infinity;
  if (/^[+-]?nan$/.test(low)) return NaN;
  const re = /^[+-]?((\d+(_\d+)*)(\.(\d+(_\d+)*)?)?|\.\d+(_\d+)*)([eE][+-]?\d+(_\d+)*)?$/;
  if (!re.test(s)) throw new ValueError(`could not convert string to float: ${reprStr(x)}`);
  return Number(s.replace(/_/g, ''));
}

// ── containers ────────────────────────────────────────────────────────
/** Python list indexing (negative indices, IndexError). */
export function at(list, i) {
  const n = list.length;
  const k = i < 0 ? n + i : i;
  if (k < 0 || k >= n) throw new IndexError('list index out of range');
  return list[k];
}

/** Python slice list[a:b] (negative/None bounds). step not supported. */
export function slice(list, a, b) {
  const n = list.length;
  const norm = (v, d) => {
    if (v === undefined || v === null) return d;
    if (v < 0) return Math.max(0, n + v);
    return Math.min(v, n);
  };
  return list.slice(norm(a, 0), norm(b, n));
}

/** dict[key] with KeyError. Works on Map and plain objects. */
export function getitem(d, key) {
  if (d instanceof Map) {
    if (!d.has(key)) throw new KeyError(String(key));
    return d.get(key);
  }
  if (d === null || d === undefined) throw new TypeError_("'NoneType' object is not subscriptable");
  if (!Object.prototype.hasOwnProperty.call(d, key)) throw new KeyError(String(key));
  return d[key];
}

/** dict.get(key, default) on Map or object; raises AttributeError for None like Python. */
export function get(d, key, dflt = null) {
  if (d === null || d === undefined) throw new AttributeError("'NoneType' object has no attribute 'get'");
  if (d instanceof Map) return d.has(key) ? d.get(key) : dflt;
  return Object.prototype.hasOwnProperty.call(d, key) ? d[key] : dflt;
}

/** collections.defaultdict(factory) backed by a Map (keeps insertion order for any key type). */
export class DefaultDict extends Map {
  constructor(factory) {
    super();
    this.factory = factory;
  }
  getOrCreate(key) {
    if (!this.has(key)) this.set(key, this.factory());
    return this.get(key);
  }
}

/** collections.Counter(iterable) as a Map; mostCommon() matches Counter.most_common(). */
export class Counter extends Map {
  constructor(iterable) {
    super();
    if (iterable) for (const x of iterable) this.set(x, (this.get(x) || 0) + 1);
  }
  count(key) {
    return this.get(key) || 0;
  }
  mostCommon(n) {
    const items = [...this.entries()];
    const sorted = sortedBy(items, (kv) => kv[1], true);
    return n === undefined ? sorted : sorted.slice(0, n);
  }
}

// ── ordering ──────────────────────────────────────────────────────────
/** Python comparison for numbers, strings, booleans, null-free arrays (tuples/lists). */
export function cmp(a, b) {
  if (Array.isArray(a) && Array.isArray(b)) {
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
      const c = cmp(a[i], b[i]);
      if (c) return c;
    }
    return a.length - b.length;
  }
  if (a === null || b === null || a === undefined || b === undefined) {
    throw new TypeError_("'<' not supported between instances involving 'NoneType'");
  }
  if (typeof a === 'string' && typeof b === 'string') {
    if (a === b) return 0;
    // Python compares code points; JS < compares UTF-16 units (same for the BMP)
    return a < b ? -1 : 1;
  }
  if (a < b) return -1;
  if (a > b) return 1;
  return 0; // equal or NaN (Python treats NaN comparisons as False → keeps order)
}

/** sorted(iterable, key=..., reverse=...) — stable, Python's reverse semantics. */
export function sortedBy(iterable, key = (x) => x, reverse = false) {
  const decorated = [...iterable].map((v, i) => ({ k: key(v), v, i }));
  decorated.sort((p, q) => {
    const c = reverse ? cmp(q.k, p.k) : cmp(p.k, q.k);
    return c || p.i - q.i;
  });
  return decorated.map((d) => d.v);
}
export const sorted = (iterable, reverse = false) => sortedBy(iterable, (x) => x, reverse);

/** max(iterable, key=...) — first maximal element; ValueError when empty (unless default given). */
export function max(iterable, key = (x) => x, dflt) {
  let best;
  let bestK;
  let any = false;
  for (const v of iterable) {
    const k = key(v);
    if (!any || cmp(k, bestK) > 0) {
      best = v;
      bestK = k;
      any = true;
    }
  }
  if (!any) {
    if (dflt !== undefined) return dflt;
    throw new ValueError('max() iterable argument is empty');
  }
  return best;
}

/** min(iterable, key=...) — first minimal element; ValueError when empty (unless default given). */
export function min(iterable, key = (x) => x, dflt) {
  let best;
  let bestK;
  let any = false;
  for (const v of iterable) {
    const k = key(v);
    if (!any || cmp(k, bestK) < 0) {
      best = v;
      bestK = k;
      any = true;
    }
  }
  if (!any) {
    if (dflt !== undefined) return dflt;
    throw new ValueError('min() iterable argument is empty');
  }
  return best;
}

/** max(a, b, ...) / min(a, b, ...) on numbers with Python's first-wins tie rule and NaN behaviour. */
export function max2(...args) {
  return max(args);
}
export function min2(...args) {
  return min(args);
}

/** Python truthiness. */
export function truthy(v) {
  if (v === null || v === undefined || v === false) return false;
  if (typeof v === 'number') return v !== 0 && !Number.isNaN(v) ? true : Number.isNaN(v);
  if (typeof v === 'string' || Array.isArray(v)) return v.length > 0;
  if (v instanceof Map || v instanceof Set) return v.size > 0;
  if (typeof v === 'object') return Object.keys(v).length > 0;
  return true;
}

/** range(start, stop, step) as an array. */
export function range(a, b, step = 1) {
  if (b === undefined) {
    b = a;
    a = 0;
  }
  const out = [];
  if (step > 0) for (let i = a; i < b; i += step) out.push(i);
  else for (let i = a; i > b; i += step) out.push(i);
  return out;
}

/** abs() */
export const abs = Math.abs;

/** len() that also works on Map/Set/strings/objects. */
export function len(x) {
  if (x === null || x === undefined) throw new TypeError_("object of type 'NoneType' has no len()");
  if (typeof x === 'string' || Array.isArray(x) || ArrayBuffer.isView(x)) return x.length;
  if (x instanceof Map || x instanceof Set) return x.size;
  return Object.keys(x).length;
}

/** Python `in` for arrays, strings, Map/Set and object keys. */
export function contains(container, item) {
  if (Array.isArray(container)) return container.some((v) => (Array.isArray(v) ? cmpEq(v, item) : v === item));
  if (typeof container === 'string') return container.includes(item);
  if (container instanceof Map || container instanceof Set) return container.has(item);
  return Object.prototype.hasOwnProperty.call(container, item);
}
function cmpEq(a, b) {
  return Array.isArray(b) && a.length === b.length && a.every((v, i) => v === b[i]);
}

const lt = (a, b) => (typeof a === 'number' && typeof b === 'number' ? a < b : cmp(a, b) < 0);

/** bisect.bisect_left(a, x, lo, hi) — same comparisons as CPython's C implementation. */
export function bisect_left(a, x, lo = 0, hi = a.length) {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (lt(a[mid], x)) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** bisect.bisect_right(a, x, lo, hi) (= bisect.bisect). */
export function bisect_right(a, x, lo = 0, hi = a.length) {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (lt(x, a[mid])) hi = mid;
    else lo = mid + 1;
  }
  return lo;
}

/**
 * copy.deepcopy(): arrays, typed arrays, Map (DefaultDict/Counter kept), Set, plain objects and
 * class instances (prototype kept). Shared references and cycles are preserved like Python's memo.
 */
export function deepcopy(x, memo = new Map()) {
  if (x === null || typeof x !== 'object') return x;
  if (memo.has(x)) return memo.get(x);
  let y;
  if (Array.isArray(x)) {
    y = new Array(x.length);
    memo.set(x, y);
    for (let i = 0; i < x.length; i++) y[i] = deepcopy(x[i], memo);
  } else if (ArrayBuffer.isView(x)) {
    y = x.slice();
    memo.set(x, y);
  } else if (x instanceof Map) {
    y = x instanceof DefaultDict ? new DefaultDict(x.factory) : x instanceof Counter ? new Counter() : new Map();
    memo.set(x, y);
    for (const [k, v] of x) y.set(deepcopy(k, memo), deepcopy(v, memo));
  } else if (x instanceof Set) {
    y = new Set();
    memo.set(x, y);
    for (const v of x) y.add(deepcopy(v, memo));
  } else {
    y = Object.create(Object.getPrototypeOf(x));
    memo.set(x, y);
    for (const k of Object.keys(x)) y[k] = deepcopy(x[k], memo);
  }
  return y;
}
