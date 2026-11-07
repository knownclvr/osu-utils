// np.js — the numpy 2.x (float64) behaviour the DanOverlay engine depends on,
// reproduced bit for bit where numpy's algorithm is deterministic:
//   * add.reduce / sum / mean on 1-D data use numpy's pairwise summation
//   * dot / convolve use OpenBLAS 0.3.34 (Haswell kernel) ddot: 16-wide FMA
//     accumulation with its exact lane-reduction order, plain multiply-add tail
//   * interp / searchsorted / arange / diff / cumsum follow numpy's C code
//   * exp / cos / pow / log / log2 / log1p: crtExp, crtCos, crtPow, crtLog, crtLog2,
//     crtLog1p (end of file) return the Windows UCRT's exact bits, like numpy and
//     CPython's math module on the reference machine; powScalar / power use crtPow
//     outside numpy's fast paths. Only crtCos with |x| >= 2e7 falls back to Math.cos.
// Known deliberate deviation: argsort ties are ordered by index (numpy's
// AVX2 sort orders ties differently; only affects summation order of equal keys).

// ── exact fused multiply-add (Boldo & Melquiond, round-to-odd emulation) ──
const SPLITTER = 134217729; // 2**27 + 1
const F64 = new Float64Array(1);
const I64 = new BigInt64Array(F64.buffer);

function twoProd(a, b) {
  const p = a * b;
  let t = SPLITTER * a;
  const ah = t - (t - a);
  const al = a - ah;
  t = SPLITTER * b;
  const bh = t - (t - b);
  const bl = b - bh;
  const e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
  return [p, e];
}

function twoSum(a, b) {
  const s = a + b;
  const bb = s - a;
  const e = (a - (s - bb)) + (b - bb);
  return [s, e];
}

/** x + e rounded to odd, where s = RN(x) and e is the exact remainder (s + e exact). */
function roundToOdd(s, e) {
  if (e === 0 || !Number.isFinite(s)) return s;
  F64[0] = s;
  if ((I64[0] & 1n) === 1n) return s; // already odd
  // move one ulp towards e
  const towardsUp = (e > 0) === (s > 0) || s === 0;
  if (s === 0) return e > 0 ? 5e-324 : -5e-324;
  I64[0] += towardsUp ? 1n : -1n;
  return F64[0];
}

/**
 * Correctly rounded a*b + c (like C fma / Python math.fma) for finite, non-extreme inputs.
 * Boldo & Melquiond, "Emulation of FMA and correctly rounded sums: proved algorithms
 * using rounding to odd" (Algorithm "emulated FMA"):
 *   (uh, ul) = ExactMult(a, b); (th, tl) = ExactAdd(c, uh); v = RO(tl + ul); RN(th + v)
 */
export function fma(a, b, c) {
  if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return a * b + c;
  if (a === 0 || b === 0) return a * b + c;
  const [uh, ul] = twoProd(a, b);
  if (!Number.isFinite(uh)) return a * b + c;
  const [th, tl] = twoSum(c, uh);
  const [s, e] = twoSum(tl, ul);
  const v = roundToOdd(s, e);
  return th + v;
}

// ── reductions ─────────────────────────────────────────────────────────
/** numpy pairwise_sum_DOUBLE (PW_BLOCKSIZE 128, 8-way unroll) over a[lo..lo+n) with stride. */
export function pairwiseSum(a, lo = 0, n = a.length - lo, stride = 1) {
  if (n < 8) {
    let res = -0.0; // numpy seeds the short loop with -0.0 (keeps sign of all -0 inputs)
    for (let i = 0; i < n; i++) res += a[lo + i * stride];
    return res;
  }
  if (n <= 128) {
    const r = new Float64Array(8);
    for (let j = 0; j < 8; j++) r[j] = a[lo + j * stride];
    let i = 8;
    const lim = n - (n % 8);
    for (; i < lim; i += 8) {
      for (let j = 0; j < 8; j++) r[j] += a[lo + (i + j) * stride];
    }
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[lo + i * stride];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwiseSum(a, lo, n2, stride) + pairwiseSum(a, lo + n2 * stride, n - n2, stride);
}

/** np.sum(a) / np.add.reduce for a 1-D array. */
export const sum = (a) => pairwiseSum(a, 0, a.length, 1);
/** np.mean(a) for a 1-D array. */
export const mean = (a) => pairwiseSum(a, 0, a.length, 1) / a.length;

/** np.sum(rows, axis=0) for a C-contiguous 2-D array given as a list of equal-length rows. */
export function sumAxis0(rows) {
  const n = rows[0].length;
  const out = new Float64Array(n);
  for (let j = 0; j < n; j++) out[j] = rows[0][j];
  for (let k = 1; k < rows.length; k++) {
    const r = rows[k];
    for (let j = 0; j < n; j++) out[j] += r[j];
  }
  return out;
}

/** np.sum(m, axis=1) for a list of rows: pairwise along each row. */
export function sumAxis1(rows) {
  return Float64Array.from(rows, (r) => pairwiseSum(r, 0, r.length, 1));
}

// ── BLAS ddot (OpenBLAS Haswell) ──────────────────────────────────────
/** cblas_ddot(n, x, 1, y, 1) exactly as OpenBLAS 0.3.34's Haswell kernel computes it. */
export function ddot(x, y, xo = 0, yo = 0, n = x.length - xo) {
  const n1 = n & -16;
  let dot = 0.0;
  if (n1) {
    const A = new Float64Array(16); // 4 ymm accumulators x 4 lanes
    for (let i = 0; i < n1; i += 16) {
      for (let k = 0; k < 16; k++) A[k] = fma(x[xo + i + k], y[yo + i + k], A[k]);
    }
    // vextractf128 + vaddpd: lanes (0+2), (1+3) of each accumulator
    const h = [];
    for (let k = 0; k < 4; k++) h.push(A[4 * k] + A[4 * k + 2], A[4 * k + 1] + A[4 * k + 3]);
    const a45p0 = h[0] + h[2];
    const a45p1 = h[1] + h[3];
    const a67p0 = h[4] + h[6];
    const a67p1 = h[5] + h[7];
    dot = (a45p0 + a67p0) + (a45p1 + a67p1); // vaddpd then vhaddpd
  }
  for (let i = n1; i < n; i++) dot += y[yo + i] * x[xo + i];
  return dot;
}

/** np.dot / @ for two 1-D float64 arrays. */
export const dot = (x, y) => ddot(x, y, 0, 0, x.length);

/** np.convolve(a, v, mode='valid') (numpy correlates with the reversed kernel via DOUBLE_dot). */
export function convolveValid(a, v) {
  let A = a;
  let V = v;
  if (V.length > A.length) [A, V] = [V, A];
  const m = V.length;
  const vr = Float64Array.from(V).reverse();
  const out = new Float64Array(A.length - m + 1);
  for (let i = 0; i < out.length; i++) out[i] = ddot(A, vr, i, 0, m);
  return out;
}

// ── construction ───────────────────────────────────────────────────────
/** np.arange(start, stop, step) for floats (numpy computes element i as start + i*delta, delta = (start+step)-start). */
export function arange(start, stop, step = 1) {
  const len = Math.max(0, Math.ceil((stop - start) / step));
  const out = new Float64Array(len);
  if (len > 0) out[0] = start;
  if (len > 1) {
    out[1] = start + step;
    const delta = out[1] - out[0];
    for (let i = 2; i < len; i++) out[i] = start + i * delta;
  }
  return out;
}

export const zeros = (n) => new Float64Array(n);
export function full(n, v) {
  const out = new Float64Array(n);
  out.fill(v);
  return out;
}
export const ones = (n) => full(n, 1);
export const array = (list) => Float64Array.from(list, Number);

/** np.pad(data, r, mode) for 1-D arrays: 'constant' (cval), 'edge'/'nearest', 'reflect', 'symmetric', 'wrap'. */
export function pad(data, r, mode = 'constant', cval = 0.0) {
  const n = data.length;
  const out = new Float64Array(n + 2 * r);
  out.set(data, r);
  for (let i = 0; i < r; i++) {
    let left;
    let right;
    switch (mode) {
      case 'constant':
        left = cval;
        right = cval;
        break;
      case 'edge':
      case 'nearest':
        left = data[0];
        right = data[n - 1];
        break;
      case 'reflect': {
        // numpy 'reflect': mirror without repeating the edge
        const period = 2 * (n - 1) || 1;
        const idxL = mirror(-(i + 1), n, period, false);
        const idxR = mirror(n + i, n, period, false);
        left = data[idxL];
        right = data[idxR];
        break;
      }
      case 'symmetric': {
        const period = 2 * n;
        left = data[mirror(-(i + 1), n, period, true)];
        right = data[mirror(n + i, n, period, true)];
        break;
      }
      case 'wrap':
        left = data[((-(i + 1)) % n + n) % n];
        right = data[(n + i) % n];
        break;
      default:
        throw new Error(`np.pad mode not supported: ${mode}`);
    }
    out[r - 1 - i] = left;
    out[r + n + i] = right;
  }
  return out;
}
function mirror(k, n, period, symmetric) {
  let m = ((k % period) + period) % period;
  if (symmetric) return m < n ? m : period - 1 - m;
  return m < n ? m : period - m;
}

// ── element-wise helpers (numpy NaN semantics) ────────────────────────
const toArr = (x, n) => (typeof x === 'number' ? { get: () => x } : { get: (i) => x[i] });
function binary(a, b, f) {
  const n = typeof a === 'number' ? b.length : a.length;
  const A = toArr(a, n);
  const B = toArr(b, n);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = f(A.get(i), B.get(i));
  return out;
}
const npMax = (x, y) => (Number.isNaN(x) || Number.isNaN(y) ? NaN : x >= y ? x : y);
const npMin = (x, y) => (Number.isNaN(x) || Number.isNaN(y) ? NaN : x <= y ? x : y);
export const maximum = (a, b) => binary(a, b, npMax);
export const minimum = (a, b) => binary(a, b, npMin);
/** np.clip(a, lo, hi) = minimum(maximum(a, lo), hi). */
export function clip(a, lo, hi) {
  if (typeof a === 'number') return npMin(npMax(a, lo), hi);
  const out = new Float64Array(a.length);
  const L = toArr(lo, a.length);
  const H = toArr(hi, a.length);
  for (let i = 0; i < a.length; i++) out[i] = npMin(npMax(a[i], L.get(i)), H.get(i));
  return out;
}
export const scalarMax = npMax;
export const scalarMin = npMin;

/**
 * ndarray ** python-scalar: numpy's fast paths 2 → square, 0.5 → sqrt, 1 → copy, -1 → reciprocal,
 * 0 → ones; any other exponent goes through the power loop, i.e. the CRT pow() (crtPow).
 * (np.float64 ** scalar and array ** array have no fast paths: use crtPow directly.)
 */
export function powScalar(x, e) {
  if (e === 2) return x * x;
  if (e === 0.5) return Math.sqrt(x);
  if (e === 1) return x;
  if (e === -1) return 1 / x;
  if (e === 0) return 1;
  return crtPow(x, e);
}
export function power(a, e) {
  if (typeof a === 'number') return powScalar(a, e);
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = powScalar(a[i], e);
  return out;
}

/** np.cumsum (sequential). */
export function cumsum(a) {
  const out = new Float64Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s = i === 0 ? a[0] : s + a[i];
    out[i] = s;
  }
  return out;
}

/** np.diff (1-D, n=1). */
export function diff(a) {
  const out = new Float64Array(Math.max(0, a.length - 1));
  for (let i = 0; i < out.length; i++) out[i] = a[i + 1] - a[i];
  return out;
}

// ── searching / interpolation ─────────────────────────────────────────
/** np.searchsorted(a, v, side) for a sorted 1-D array and a scalar. */
export function searchsorted1(a, v, side = 'left') {
  let lo = 0;
  let hi = a.length;
  if (side === 'left') {
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (a[mid] < v) lo = mid + 1;
      else hi = mid;
    }
  } else {
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (a[mid] <= v) lo = mid + 1;
      else hi = mid;
    }
  }
  return lo;
}
/** np.searchsorted(a, values, side) element-wise (returns plain numbers). */
export function searchsorted(a, values, side = 'left') {
  if (typeof values === 'number') return searchsorted1(a, values, side);
  const out = new Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = searchsorted1(a, values[i], side);
  return out;
}

/** np.interp(x, xp, fp) — numpy's arr_interp (default left/right = fp[0]/fp[-1]). */
export function interp(x, xp, fp, left, right) {
  const lenxp = xp.length;
  const lval = left === undefined ? fp[0] : left;
  const rval = right === undefined ? fp[lenxp - 1] : right;
  const scalar = typeof x === 'number';
  const xs = scalar ? [x] : x;
  const out = new Float64Array(xs.length);
  let slopes = null;
  if (lenxp === 1) {
    for (let i = 0; i < xs.length; i++) {
      const xv = xs[i];
      out[i] = xv < xp[0] ? lval : xv > xp[0] ? rval : fp[0];
    }
    return scalar ? out[0] : out;
  }
  if (lenxp <= xs.length) {
    slopes = new Float64Array(lenxp - 1);
    for (let i = 0; i < lenxp - 1; i++) slopes[i] = (fp[i + 1] - fp[i]) / (xp[i + 1] - xp[i]);
  }
  for (let i = 0; i < xs.length; i++) {
    const xv = xs[i];
    if (Number.isNaN(xv)) {
      out[i] = xv;
      continue;
    }
    let j;
    if (xv > xp[lenxp - 1]) j = lenxp;
    else if (xv < xp[0]) j = -1;
    else j = searchsorted1(xp, xv, 'right') - 1;
    if (j === -1) out[i] = lval;
    else if (j === lenxp) out[i] = rval;
    else if (j === lenxp - 1) out[i] = fp[j];
    else if (xp[j] === xv) out[i] = fp[j];
    else {
      const slope = slopes ? slopes[j] : (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
      let r = slope * (xv - xp[j]) + fp[j];
      if (Number.isNaN(r)) {
        r = slope * (xv - xp[j + 1]) + fp[j + 1];
        if (Number.isNaN(r) && fp[j] === fp[j + 1]) r = fp[j];
      }
      out[i] = r;
    }
  }
  return scalar ? out[0] : out;
}

// ── sorting ────────────────────────────────────────────────────────────
/** np.sort for 1-D float64 (NaNs last). */
export function sort(a) {
  const out = Float64Array.from(a);
  out.sort((x, y) => (Number.isNaN(x) ? 1 : Number.isNaN(y) ? -1 : x - y));
  return out;
}
/** np.argsort (ties by index — see header note). */
export function argsort(a) {
  const idx = Array.from({ length: a.length }, (_, i) => i);
  idx.sort((i, j) => {
    const x = a[i];
    const y = a[j];
    if (Number.isNaN(x)) return Number.isNaN(y) ? i - j : 1;
    if (Number.isNaN(y)) return -1;
    return x - y || i - j;
  });
  return idx;
}

/** np.add.at(arr, idx, v) — unbuffered in-place add (repeated indices accumulate in order). */
export function addAt(arr, idx, v) {
  for (let k = 0; k < idx.length; k++) arr[idx[k]] += typeof v === 'number' ? v : v[k];
}

/** np.trapezoid(y_rows, x, axis=1): per row add.reduce(d * (y[1:] + y[:-1]) / 2.0). */
export function trapezoidRows(rows, x) {
  const d = diff(x);
  return rows.map((y) => {
    const t = new Float64Array(d.length);
    for (let i = 0; i < d.length; i++) t[i] = (d[i] * (y[i + 1] + y[i])) / 2.0;
    return pairwiseSum(t, 0, t.length, 1);
  });
}

// ── Windows UCRT libm ─────────────────────────────────────────────────────
// On the reference machine numpy's float64 exp / cos / power loops and CPython's math.exp,
// math.cos, math.log, math.log2, math.log1p, math.pow and float ** float all call
// ucrtbase.dll (x64; the FMA3 code paths where the CRT has them — the default on any CPU
// with FMA3). V8's fdlibm-based Math.* differ from it in the last bit for ~10% (exp),
// ~3% (cos), ~0.2% (pow) of arguments, and the CRT is not correctly rounded either
// (pow(x, 2.0) != x*x for ~0.2% of x). crtExp / crtCos / crtPow / crtLog / crtLog2 /
// crtLog1p replay the CRT's algorithms operation by operation (same tables, constants, FMA
// contractions and rounding steps), so they return the CRT's exact bits. Verified against
// ucrtbase (via numpy / CPython / ctypes) on 3.7M exp, 4.3M cos, 9.6M pow, 4.8M log,
// 4.8M log2 and 5.5M log1p arguments (random, boundary, subnormal and special values):
// all identical.
//   * These are the C functions (C99 special values, no exceptions). py.math.* and py.pow
//     add CPython's special cases and errors on top (math_1 / float_pow / math.pow).
//   * ndarray ** python-scalar takes numpy fast paths for 2 (x*x), 0.5 (sqrt), -1 (1/x),
//     1 and 0 (see powScalar); np.float64 ** scalar and array ** array always call pow().
//   * crtCos falls back to Math.cos for |x| >= 2e7 (CRT Payne-Hanek reduction not reproduced).

const _crtDV = new DataView(new ArrayBuffer(8));
function _crtTable(hex) {
  const n = hex.length / 16;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    _crtDV.setUint32(0, parseInt(hex.substr(16 * i, 8), 16));
    _crtDV.setUint32(4, parseInt(hex.substr(16 * i + 8, 8), 16));
    out[i] = _crtDV.getFloat64(0);
  }
  return out;
}
// 2^(j/64) (T1), its tail (T2) and head (T3); 1/F head/tail and log(2F) head/tail, F = (1 + j/256)/2
const _CRT_EXP_T1 = _crtTable(
  '3ff00000000000003ff02c9a3e7780613ff059b0d31585743ff0874518759bc83ff0b5586cf9890f3ff0e3ec32d3d1a23ff11301d0125b513ff1429aaea92de0' +
  '3ff172b83c7d517b3ff1a35beb6fcb753ff1d4873168b9aa3ff2063b88628cd63ff2387a6e7562383ff26b4565e27cdd3ff29e9df51fdee13ff2d285a6e4030b' +
  '3ff306fe0a31b7153ff33c08b26416ff3ff371a7373aa9cb3ff3a7db34e59ff73ff3dea64c1234223ff4160a21f72e2a3ff44e086061892d3ff486a2b5c13cd0' +
  '3ff4bfdad5362a273ff4f9b2769d2ca73ff5342b569d4f823ff56f4736b527da3ff5ab07dd4854293ff5e76f15ad21483ff6247eb03a55853ff6623882552225' +
  '3ff6a09e667f3bcd3ff6dfb23c651a2f3ff71f75e8ec5f743ff75feb564267c93ff7a11473eb01873ff7e2f336cf4e623ff82589994cce133ff868d99b4492ed' +
  '3ff8ace5422aa0db3ff8f1ae991577363ff93737b0cdc5e53ff97d829fde4e503ff9c49182a3f0903ffa0c667b5de5653ffa5503b23e255d3ffa9e6b5579fdbf' +
  '3ffae89f995ad3ad3ffb33a2b84f15fb3ffb7f76f2fb5e473ffbcc1e904bc1d23ffc199bdd85529c3ffc67f12e57d14b3ffcb720dcef90693ffd072d4a07897c' +
  '3ffd5818dcfba4873ffda9e603db32853ffdfc97337b9b5f3ffe502ee78b3ff63ffea4afa2a490da3ffefa1bee615a273fff50765b6e45403fffa7c1819e90d8');
const _CRT_EXP_T2 = _crtTable(
  '00000000000000003e6cef00c1dcdef93e48ac2ba1d73e2a3e60eb37901186be3e69f3121ec531723e469e8d10103a173df25b50a4ebbf1a3e6d525bbf668203' +
  '3e68faa2f5b9bef93e66df96ea796d313e368b9aa7805b803e60c519ac771dd63e6ceac470cd83f53e5789f37495e99c3e547f7b84b097453e5b900c2d002475' +
  '3e64636e2a5bd1ab3e4320b7fa64e4303e5ceaa72a9c51543e53967fdba86f243e682468446b68243e3f72e29f84325b3e18624b40c4dbd03e5704f3404f068e' +
  '3e54d8a89c750e5e3e5a74b29ab4cf623e5a753e077c2a0f3e5ad49f699bb2c03e6a90a852b192603e56b48521ba6f933e0d2ac258f87d033e42a91124893ecf' +
  '3e59fcef32422cbe3e68ca345de441c53e61d8bee7ba46e13e59099f22fdba6a3e4f580c36bea8813e5b3d398841740a3e62999c25159f113e668925d901c83b' +
  '3e415506dadd3e2a3e622aee6c57304e3e29b8bc9e8a03873e6fbc9c9f173d243e451f8480e3e2353e66bbcac96535b53e41f12ae45a12243e55e7f6fd0fac90' +
  '3e62b5a75abd0e693e609e2bf5ed7fa13e47daf237553d843e12f074891ee83d3e6b0aa5384441963e6cafa29694426f3e69df20d22a07973e640f12f71a1e45' +
  '3e69f7490e4bb40b3e4ed9942b84600d3e4bdcdaf5cb46563e5e2cffd89cf44c3e452486cc2c7b9d3e6cc2b44eee3fa43e66dc8a80ce9f093e39e90d82e90a7e');
const _CRT_EXP_T3 = _crtTable(
  '3ff00000000000003ff02c9a300000003ff059b0d00000003ff08745100000003ff0b558600000003ff0e3ec300000003ff11301d00000003ff1429aa0000000' +
  '3ff172b8300000003ff1a35be00000003ff1d487300000003ff2063b800000003ff2387a600000003ff26b45600000003ff29e9df00000003ff2d285a0000000' +
  '3ff306fe000000003ff33c08b00000003ff371a7300000003ff3a7db300000003ff3dea6400000003ff4160a200000003ff44e08600000003ff486a2b0000000' +
  '3ff4bfdad00000003ff4f9b2700000003ff5342b500000003ff56f47300000003ff5ab07d00000003ff5e76f100000003ff6247eb00000003ff6623880000000' +
  '3ff6a09e600000003ff6dfb2300000003ff71f75e00000003ff75feb500000003ff7a114700000003ff7e2f3300000003ff82589900000003ff868d990000000' +
  '3ff8ace5400000003ff8f1ae900000003ff93737b00000003ff97d82900000003ff9c491800000003ffa0c66700000003ffa5503b00000003ffa9e6b50000000' +
  '3ffae89f900000003ffb33a2b00000003ffb7f76f00000003ffbcc1e900000003ffc199bd00000003ffc67f1200000003ffcb720d00000003ffd072d40000000' +
  '3ffd5818d00000003ffda9e6000000003ffdfc97300000003ffe502ee00000003ffea4afa00000003ffefa1be00000003fff5076500000003fffa7c180000000');
const _CRT_LOG_INV_HI = _crtTable(
  '40000000000000003fffe000000000003fffc000000000003fffa000000000003fff8000000000003fff6000000000003fff4000000000003fff200000000000' +
  '3fff0000000000003ffee000000000003ffec000000000003ffea000000000003ffe9000000000003ffe7000000000003ffe5000000000003ffe300000000000' +
  '3ffe1000000000003ffe0000000000003ffde000000000003ffdc000000000003ffda000000000003ffd9000000000003ffd7000000000003ffd500000000000' +
  '3ffd4000000000003ffd2000000000003ffd0000000000003ffcf000000000003ffcd000000000003ffcb000000000003ffca000000000003ffc800000000000' +
  '3ffc7000000000003ffc5000000000003ffc3000000000003ffc2000000000003ffc0000000000003ffbf000000000003ffbd000000000003ffbc00000000000' +
  '3ffba000000000003ffb9000000000003ffb7000000000003ffb6000000000003ffb4000000000003ffb3000000000003ffb2000000000003ffb000000000000' +
  '3ffaf000000000003ffad000000000003ffac000000000003ffaa000000000003ffa9000000000003ffa8000000000003ffa6000000000003ffa500000000000' +
  '3ffa4000000000003ffa2000000000003ffa1000000000003ffa0000000000003ff9e000000000003ff9d000000000003ff9c000000000003ff9a00000000000' +
  '3ff99000000000003ff98000000000003ff97000000000003ff95000000000003ff94000000000003ff93000000000003ff92000000000003ff9000000000000' +
  '3ff8f000000000003ff8e000000000003ff8d000000000003ff8b000000000003ff8a000000000003ff89000000000003ff88000000000003ff8700000000000' +
  '3ff86000000000003ff84000000000003ff83000000000003ff82000000000003ff81000000000003ff80000000000003ff7f000000000003ff7e00000000000' +
  '3ff7d000000000003ff7b000000000003ff7a000000000003ff79000000000003ff78000000000003ff77000000000003ff76000000000003ff7500000000000' +
  '3ff74000000000003ff73000000000003ff72000000000003ff71000000000003ff70000000000003ff6f000000000003ff6e000000000003ff6d00000000000' +
  '3ff6c000000000003ff6b000000000003ff6a000000000003ff69000000000003ff68000000000003ff67000000000003ff66000000000003ff6500000000000' +
  '3ff64000000000003ff63000000000003ff62000000000003ff61000000000003ff60000000000003ff5f000000000003ff5e000000000003ff5d00000000000' +
  '3ff5c000000000003ff5b000000000003ff5a000000000003ff59000000000003ff58000000000003ff58000000000003ff57000000000003ff5600000000000' +
  '3ff55000000000003ff54000000000003ff53000000000003ff52000000000003ff51000000000003ff50000000000003ff50000000000003ff4f00000000000' +
  '3ff4e000000000003ff4d000000000003ff4c000000000003ff4b000000000003ff4a000000000003ff4a000000000003ff49000000000003ff4800000000000' +
  '3ff47000000000003ff46000000000003ff46000000000003ff45000000000003ff44000000000003ff43000000000003ff42000000000003ff4200000000000' +
  '3ff41000000000003ff40000000000003ff3f000000000003ff3e000000000003ff3e000000000003ff3d000000000003ff3c000000000003ff3b00000000000' +
  '3ff3b000000000003ff3a000000000003ff39000000000003ff38000000000003ff38000000000003ff37000000000003ff36000000000003ff3500000000000' +
  '3ff35000000000003ff34000000000003ff33000000000003ff32000000000003ff32000000000003ff31000000000003ff30000000000003ff3000000000000' +
  '3ff2f000000000003ff2e000000000003ff2e000000000003ff2d000000000003ff2c000000000003ff2b000000000003ff2b000000000003ff2a00000000000' +
  '3ff29000000000003ff29000000000003ff28000000000003ff27000000000003ff27000000000003ff26000000000003ff25000000000003ff2500000000000' +
  '3ff24000000000003ff23000000000003ff23000000000003ff22000000000003ff21000000000003ff21000000000003ff20000000000003ff2000000000000' +
  '3ff1f000000000003ff1e000000000003ff1e000000000003ff1d000000000003ff1c000000000003ff1c000000000003ff1b000000000003ff1b00000000000' +
  '3ff1a000000000003ff19000000000003ff19000000000003ff18000000000003ff18000000000003ff17000000000003ff16000000000003ff1600000000000' +
  '3ff15000000000003ff15000000000003ff14000000000003ff13000000000003ff13000000000003ff12000000000003ff12000000000003ff1100000000000' +
  '3ff11000000000003ff10000000000003ff0f000000000003ff0f000000000003ff0e000000000003ff0e000000000003ff0d000000000003ff0d00000000000' +
  '3ff0c000000000003ff0c000000000003ff0b000000000003ff0a000000000003ff0a000000000003ff09000000000003ff09000000000003ff0800000000000' +
  '3ff08000000000003ff07000000000003ff07000000000003ff06000000000003ff06000000000003ff05000000000003ff05000000000003ff0400000000000' +
  '3ff04000000000003ff03000000000003ff03000000000003ff02000000000003ff02000000000003ff01000000000003ff01000000000003ff0000000000000' +
  '3ff0000000000000');
const _CRT_LOG_INV_LO = _crtTable(
  '00000000000000003effe01fe01fe0203f1fc07f01fc07f03f31caa01fa11caa3f3f81f81f81f8203f48856506ddaba63f5196792909c5603f57d9108c2ad433' +
  '3f5f07c1f07c1f083f638ff08b1c03dd3f680f6603d980f63f6d00f57403d5d03f331abf0b7672a03f506a965d43919b3f5ceb240795ceb23f6522f3b834e67f' +
  '3f6c3c3c3c3c3c3c3f3e01e01e01e01e3f575b8fe21a291c3f6403b9403b94043f6cc0ed7303b5cc3f479118f3fc4da23f5ed952e0b0ce463f695900eae56404' +
  '3f3d41d41d41d41d3f5cb28ff16c69ae3f696b1edd80e8663f4372e225fe30d93f60ad12073615a23f6cdb2c0397cdb33f52cc157b8644073f664cb5f7148404' +
  '3f3c71c71c71c71c3f6129a21a930b843f6f1e0387f1e0383f5ad4e4ba80709b3f6c0e070381c0e03f560fba1a362bb03f6a5713280dee963f53f59620f9ece9' +
  '3f69f22983759f233f5478ac63fc8d5c3f6ad87bb46716563f578b8efbb8148c3f6d0369d0369d033f5d212b601b37483f0b2036406c80d93f629663b24547d1' +
  '3f4435e50d79435e3f67d0ff2920bc033f55c06b15c06b163f6e3a5f0fd7f9543f61dec0d4c77b033f473289870ac52e3f6a034da034da033f5d041da2292856' +
  '3f3a41a41a41a41a3f68550f8a39409d3f5b4fe5e92c06863f3a01a01a01a01a3f691d2a2067b23a3f5e7c5dada0b4e53f468a7725080ce13f6c49d4aa21b490' +
  '3f633333333333333f54bc363b03fccf3f2c9f01970e4f813f697617c6ef5b253f6161f9add3c0ca3f5319fe6cb398063f2f693a1c451ab33f6a9e240321a9e2' +
  '3f63831f3831f3833f5949ebc4dcfc1c3f480c6980c6980c3f6f9d00c5fe74033f69721ed7e753473f6381ec0313381f3f5b97c2aec126533f509ef3024ae3ba' +
  '3f386186186186183f6e0184f00c27803f692ef5657dba523f649403054940303f603030303030303f580601806018063f5017f405fd017f3f412a8ad278e8dd' +
  '3f17d05f417d05f43f6d67245c02f7d63f6a4411c1d986a93f6754d76c7316df3f649902f149902f3f621023358c1a683f5f7390d2a6c4063f5b2b0805d5b2b1' +
  '3f5745d1745d17463f53c31507fa32c43f50a1fd1b7af0173f4bc36ce3e0453a3f4702e05c0b81703f4300b79300b7933f3f76b4337c6cb13f3a62681c860fb0' +
  '3f36c16c16c16c173f3490aa31a3cfc73f33cd153729043e3f3473a88d0bfd2e3f368168168168173f39f36016719f363f3ec6a5122f90163f427c29da5519cf' +
  '3f4642c8590b21643f4ab5c45606f00b3f4fd3b80b11fd3c3f52cda0c6ba4eaa3f560581605816063f5990d0a4b7ef873f5d6ee340579d6f3f60cf87d9c54a69' +
  '3f6310572620ae4c3f65798c8ff522a23f680ad602b580ad3f6ac3e24799546f3f6da46102b1da463f158056015805603f3ed3c506b39a233f4cbdd3e2970f60' +
  '3f555555555555553f5c979aee0bf8053f621291e81fd58e3f65fead500a95803f6a0fd5c5f02a3a3f6e45c223898adc3f350150150150153f4c7b16ea64d422' +
  '3f57829cbc14e5e13f60877db85897203f65710e4b5edcea3f6a7dbb4d1fc1c83f6fad40a57eb5033f43fd6bb00a51403f54e78ecb419ba93f600a44029100a4' +
  '3f65c28f5c28f5c33f6b9c68b2c0cc4a3f2978feb9f343813f4ecf163bb6500a3f5be1958b67ebb93f644e6157dc9a3b3f6acc4baa3f0ddf3f26a4cbcb2a247b' +
  '3f505050505050503f5e0b44399598193f66027f6027f6023f6d1e854b5e0db43f4165e7254813e23f576646a9d716ef3f632b48f757ce883f6ac1b24652a906' +
  '3f33b13b13b13b143f5490e1eb2089843f62385830fec66e3f6a45a6cc111b7e3f338138138138143f556f472517b7083f631be7bc0e8f2a3f6b9cbf3e55f044' +
  '3f40e7d95bc609a93f59e6b3804d19e73f65c8b6af7963c23f6eb9dad43bf4023f4f1a515885fb373f60eeb1d3d76c023f6a320261a320263f3c82ac40260390' +
  '3f5a12f684bda12f3f669d43fda2962c3f02e025c04b80973f542804b542804b3f63f69b02593f6a3f6df31cb46e21fa3f5012b404ad012b3f623925e7820a7f' +
  '3f6c8253c8253c823f4b92ddc02526e53f616025116025113f6bf471439c9adf3f4a85c40939a85c3f6166f9ac024d163f6c44e10125e2273f4cebf48bbd90e5' +
  '3f624924924924923f6d6f2e2ec0b6733f5159e26af37c053f640245402454023f6f6f0243f6f0243f55e601215798053f668e18cf81b10f3f32012012012012' +
  '3f5c11f7047dc11f3f69e878ff70985e3f4779d9fdc3a2193f61eace5c9579073f6e0d5b450239e13f548bf0738163673f6694808dda52023f37c67f2bae2b21' +
  '3f5ee58469ee58473f6c0233c0233c023f514e02328a70123f6561072057b5733f318118118118123f5e28646f5a10603f6c0d1284e6f1d73f523543f0c80459' +
  '3f663cbeea4e1a093f3b9a3fdd5c8cb83f60be1c159a76d23f6e1d1a688e48383f572044d72044d73f691713db81577b3f4ac73ae9819b503f6460334e904cf6' +
  '3f311111111111113f5feef80441fef03f6de021fde021fe3f57b7eacc9686a03f69ead7cd391fbc3f501956098043903f6641511e8d2b323f4222b1acf1ce96' +
  '3f62e29f79b475823f24f0d1682e11cd3f5f9bb096771e4d3f6e5ee45dd96ae23f5a0429a0429a043f6bb74d5f06c0213f54fce404254fce3f695766eacbc402' +
  '3f508421084210843f673e5371d5c3383f4930523fbe33683f656b38f225f6c43f426e978d4fdf3b3f63dd40e4eb0cc63f397f7d734041463f6293982cc98af1' +
  '3f304104104104103f618d6f048ff7e43f2236a3ebc349de3f60c9f8ee53d18c3f102040810204083f60486ca2f46ea63ef01010101010103f60080402010080' +
  '0000000000000000');
const _CRT_LOG_HI = _crtTable(
  '00000000000000003f6ff00aa00000003f7fe02a600000003f87dc47500000003f8fc0a8b00000003f93cea4400000003f97b91b000000003f9b9fc020000000' +
  '3f9f829b000000003fa1b0d9800000003fa39e87b00000003fa58a5ba00000003fa77458f00000003fa95c83000000003fab42dd700000003fad276b80000000' +
  '3faf0a30c00000003fb07598300000003fb16536e00000003fb253f6200000003fb341d7900000003fb42edcb00000003fb51b07300000003fb60658a0000000' +
  '3fb6f0d2800000003fb7da76600000003fb8c345d00000003fb9ab42400000003fba926d300000003fbb78c8200000003fbc5e54800000003fbd4313d0000000' +
  '3fbe2707600000003fbf0a30c00000003fbfec91300000003fc06715100000003fc0d77e700000003fc14785800000003fc1b72ad00000003fc2266f10000000' +
  '3fc29552f00000003fc303d7100000003fc371fc200000003fc3dfc2b00000003fc44d2b600000003fc4ba36f00000003fc526e5e00000003fc59338d0000000' +
  '3fc5ff30700000003fc66acd400000003fc6d60fe00000003fc740f8f00000003fc7ab89000000003fc815c0a00000003fc87fa0600000003fc8e928d0000000' +
  '3fc9525a900000003fc9bb36200000003fca23bc100000003fca8becf00000003fcaf3c9400000003fcb5b51900000003fcbc286700000003fcc296850000000' +
  '3fcc8ff7c00000003fccf635400000003fcd5c21600000003fcdc1bca00000003fce2707600000003fce8c02500000003fcef0adc00000003fcf550a50000000' +
  '3fcfb918600000003fd00e6c400000003fd04025900000003fd071b8500000003fd0a324e00000003fd0d46b500000003fd1058bf00000003fd1368700000000' +
  '3fd1675ca00000003fd1980d200000003fd1c898c00000003fd1f8ff900000003fd22941f00000003fd25960100000003fd2895a100000003fd2b93030000000' +
  '3fd2e8e2b00000003fd31871c00000003fd347dd900000003fd37726600000003fd3a64c500000003fd3d54fa00000003fd40430800000003fd432ef20000000' +
  '3fd4618bc00000003fd49006800000003fd4be5f900000003fd4ec97300000003fd51aad800000003fd548a2c00000003fd57677100000003fd5a42ab0000000' +
  '3fd5d1bdb00000003fd5ff30700000003fd62c82f00000003fd659b5700000003fd686c8100000003fd6b3bb200000003fd6e08ea00000003fd70d42e0000000' +
  '3fd739d7f00000003fd7664e100000003fd792a5500000003fd7bede000000003fd7eaf8300000003fd816f4100000003fd842d1d00000003fd86e9190000000' +
  '3fd89a33800000003fd8c5b7c00000003fd8f11e800000003fd91c67e00000003fd94794100000003fd972a3400000003fd99d95800000003fd9c86b00000000' +
  '3fd9f323e00000003fda1dc0600000003fda4840900000003fda72a4900000003fda9cec900000003fdac718c00000003fdaf129300000003fdb1b1e00000000' +
  '3fdb44f7700000003fdb6eb5900000003fdb9858900000003fdbc1e0800000003fdbeb4d900000003fdc149ff00000003fdc3dd7a00000003fdc66f4e0000000' +
  '3fdc8ff7c00000003fdcb8e0700000003fdce1af000000003fdd0a63a00000003fdd32fe700000003fdd5b7f900000003fdd83e7200000003fddac3530000000' +
  '3fddd46a000000003fddfc85900000003fde2488100000003fde4c71a00000003fde7442600000003fde9bfa600000003fdec399d00000003fdeeb20c0000000' +
  '3fdf128f500000003fdf39e5b00000003fdf6123f00000003fdf884a300000003fdfaf58800000003fdfd64f200000003fdffd2e000000003fe011fab0000000' +
  '3fe02552a00000003fe0389ee00000003fe04bdf900000003fe05f14b00000003fe0723e500000003fe0855c800000003fe0986f400000003fe0ab76b0000000' +
  '3fe0be72e00000003fe0d163c00000003fe0e449800000003fe0f724100000003fe109f3900000003fe11cb8100000003fe12f71900000003fe1422020000000' +
  '3fe154c3d00000003fe1675ca00000003fe179eab00000003fe18c6e000000003fe19ee6b00000003fe1b154b00000003fe1c3b8100000003fe1d610f0000000' +
  '3fe1e85f500000003fe1faa3400000003fe20cdcd00000003fe21f0bf00000003fe23130d00000003fe2434b600000003fe2555bc00000003fe2676200000000' +
  '3fe2795e100000003fe28b50000000003fe29d37f00000003fe2af15f00000003fe2c0e9e00000003fe2d2b4000000003fe2e474300000003fe2f62a90000000' +
  '3fe307d7300000003fe3197a000000003fe32b13300000003fe33ca2b00000003fe34e28900000003fe35fa4e00000003fe37117b00000003fe38280f0000000' +
  '3fe393e0d00000003fe3a537300000003fe3b684400000003fe3c7c7f00000003fe3d902600000003fe3ea33900000003fe3fb5b800000003fe40c7a40000000' +
  '3fe41d8fe00000003fe42e9c600000003fe43f9fe00000003fe4509a500000003fe4618bc00000003fe47274300000003fe48353d00000003fe4942a80000000' +
  '3fe4a4f8500000003fe4b5bd600000003fe4c679a00000003fe4d72d300000003fe4e7d8100000003fe4f87a300000003fe50913c00000003fe519a4c0000000' +
  '3fe52a2d200000003fe53aad000000003fe54b24600000003fe55b93500000003fe56bf9d00000003fe57c57f00000003fe58cadb00000003fe59cfb20000000' +
  '3fe5ad40400000003fe5bd7d300000003fe5cdb1d00000003fe5ddde500000003fe5ee02a00000003fe5fe1ed00000003fe60e32f00000003fe61e3ef0000000' +
  '3fe62e42e0000000');
const _CRT_LOG_LO = _crtTable(
  '00000000000000003db5885e0250435a3de620cf11f86ed23dff0214edba4a253dbf807c79f3db4e3dea352ba779a52b3dff56c46aa49fd53dfebe465fef5196' +
  '3e0cf0660099f1f83e1247b2ff85945d3e13fd7abf5202b63e1f91c9a918d51e3e08cb73f118d3ca3e1d91c7d6fad0743de1971bec28d14c3e15b616a423c78a' +
  '3da162a6617cc9713e166391c4c06d293e2d46f5c1d0c4b83e2e14282df1f6d33e186f47424a660d3e2d4c8de077753e3e2e0c307ed24f1c3e226ea18763bdd3' +
  '3e25cad69737c9333e2af625990889013e18c66c83d6b2d03e1880ceb36fb30f3e2495aac6ca17a43e2761db4210878c3e2eb78e862bac2f3e19b2cd75790dd9' +
  '3e2c55e5cbd3d50f3db162a6617cc9713dfdbeabaaa2e5193e1652cb7150c6473e39a11cb2cd2ee23e219d0ab1a288133e24bd9e80a418113e3214b596faa3df' +
  '3e303fea46980bb83e31c8ffa5fd28c73dce8f743bcd96c53dfd98c5395315c63e3996fa3ccfa7b23e1cd2af2ad130373e1d0da1bd17200e3e3330410ba68b75' +
  '3df4f27a790e7c413e13956a86f6ff1b3e2c6748723551d93e2500de9326cdfc3e1086c848df1b593e04357ead6836ff3e248324424080243e3d10da8154b13d' +
  '3e39e8ad68ec82603e3cfbf706abaf183e3fc56ac6326e233e39105e3185cf213e3d017fe5b19cc03e3d1f6b48dd13fe3e20b63358a7e73a3e263063028c211c' +
  '3e2e6a6886b097603e3c138bb891cd033e369f7722b7221a3df57d8fac1a628c3e3c55e5cbd3d50f3e1552d2ff48fe2e3e37b8b26ca431bc3e292decdc1c5f6d' +
  '3e3abc7c551aaa8c3e36b540731a354b3e32d341036b89ef3e4f9ab21a3a2e0f3e239c871afb9fbd3e3e6add2c81f6403e435c95aa313f413e249d4582f6cc53' +
  '3e47574c1c07398f3e4ba846dece9e8d3e16999fafbc68e73e4c9145e51b01033e479ef2cb44850a3e0beec73de112753e2ef4351af5a4983e45713a493b4a50' +
  '3e45c23a613859923e42a88309f572993e4530faa9ac8ace3e25fec2d792a7583e35a517a71cbcd73e3707dc3e1cd9a33e3a1a9f8ef430493e4409d0276b3674' +
  '3e20e2f613e85bd93df0027433001e5f3e35dde2836d32653e2300134d7aaf043e3cb7e0b42724f53e2d6e93167e63083e3d1569b1526adb3e0e99fc338a1a41' +
  '3e4eb01394a11b1c3e04f27a790e7c413e25ce3ca97b7af93e281f0f940ed8573e4d36295d88857c3e21aca1ec4af5263e445743c71827263e23c491aead337e' +
  '3e3aef401a7389313e21cede76092a293e4fba8f44f82bb43e446f5f7f3c3e1a3e47055f86c9674b3e4b41a92b6b6e1a3e443d162e9276283e4466174013f9b1' +
  '3e3b05096ad69c623e40b169150faa583e3cd98b1df85da73e468b507b0f8fa83e48422df57499ba3e113515869702743e117e08acba92ee3e26e04314dd0229' +
  '3e497f3097e56d1a3e3356e6559012863e0cb761457f94d63e39af67a85a9dac3e453410931a909f3e22c587206058f53e223bc358899c223e4d7bf8b6d223cb' +
  '3e47991ec5197ddb3e4a79e6bb3a92193e3a4c43ed663ec53e461b5a1484f4383e4b4e36f7ef0c3a3e115f026acd0d1b3e3f36b535cecf053e2ffb7fbf3eb5c6' +
  '3e3e6a6886b097603e3135eb27f5bbc33e470be7d6f6fa573e4ce43cc84ab3383e4c01d7aac3bd913e45c58d079610603e3628bcf941456e3e4c58b2a8461cd2' +
  '3e33071282fb989a3e420dab6a80f09c3e44f8d84c397b1e3e40d0ee08599e483e1d68787e37da363e366187d591bafc3e22346600bae7723e390377d0d61b8e' +
  '3e4f5e0dd966b9073e49023cb79a00e23e44e05158c28ad83e3bfa7b08b18ae43e4ef1e63db35f673e0ec2ae39493d4f3e40afe930ab2fa03e225ff8a1810dd4' +
  '3e469743fb1a71a53e5f9cc6767855713e5b524da4cbf9823e5a4c8b381535b83e5839be809caf2c3e50968a1cb82c133e5eae6a41723fb53e5d9c29a380a4db' +
  '3e4094aa0ada625e3e5973ad6fc108ca3e4747322fdbab973e593692fa9d42213e5c5a992dfbc7d93e4e1f33e102387a3e464fbef14c048c3e4490f513ca5e3b' +
  '3e37a6af4d4c799d3e57574c1c07398f3e57b133417f8c1c3e5feb9e0c1765143e419f25bb3172f73e45f68a7bbfb8523e5ee278497929f13e5ccee006109d58' +
  '3e5ce081a07bd8b33e570e12981817b83e292ab6d93503d03e58cb7dd7c3b61e3e4efafd0a0b78da3e5e907267c4288e3e5d31ef967808753e23430dfcd2ad50' +
  '3e344d88d75bc1f93e5bec0f055e04fc3e5d85611590b9ad3df320568e5832293e5a891d1772f5383e22edc9dabba74d3e4b9009a10150863e52a12a8c5b1a19' +
  '3e3a7885f0fdac853e5f4ffcd43ac6913e52243ae2640aad3e546513299035d33e5b39c3a62dd7253e5ba6dd40049f513e451d1ed71774093e5cb0f2fd7f5216' +
  '3e3ab150cd4e22133e5cfd7bf31938443e53fff8455f1dbd3e5fee640b905fc93e54e2adf548084c3e3b597adc1ecdd23e4345bd096d3a753e5101b9d2453c8b' +
  '3e508ce55cc8c9793e5bbf017e595f713e37ce733bd393dc3e233bb0a503f8a13e30e2f613e85bd93e5e67555a635b3c3e2ea88df73d5e8b3e3d17e03bda18a8' +
  '3e5b607d76044f7e3e52adc4e71bc2fc3e5f99dc7362d1d93e5473fa008e6a6a3e2b75bb09cb09853e5ea04dd10b9aba3e5802d0d69796743e174688ccd99094' +
  '3e496f16abb9df223e46e66df2aa374f3e4e66525ea4550a3e42d02f34f20cbd3e46cfce650471883e39b78c842d58b83e4735e624c24bc93e47eba1f7dd1adf' +
  '3e586b3e59f653553e1ce38e637f1b4d3e58d82ec919edc73e4c52648ddcfa373e52482ceae1ac123e55a312311aba4f3e411e236329f2253e5b48c8cd2f246c' +
  '3e6efa39ef35793c');

const _CRT_F = new Float64Array(1);
const _CRT_U = new Uint32Array(_CRT_F.buffer);
const _CRT_LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const _CRT_LO = _CRT_LE ? 0 : 1;
const _CRT_HI = _CRT_LE ? 1 : 0;

function _crtFromBits(hi, lo) {
  _CRT_U[_CRT_HI] = hi;
  _CRT_U[_CRT_LO] = lo;
  return _CRT_F[0];
}
/** 2^m exactly, -1074 <= m <= 1023. */
function _crtPow2(m) {
  if (m >= -1022) return _crtFromBits(((m + 1023) << 20) >>> 0, 0);
  const k = m + 1074; // subnormal: bit pattern 1 << k
  return k >= 32 ? _crtFromBits((1 << (k - 32)) >>> 0, 0) : _crtFromBits(0, (2 ** k) >>> 0);
}

/** Correctly rounded a*b + c (vfmadd): Dekker product + Boldo-Melquiond round-to-odd, no allocation. */
function _crtFma(a, b, c) {
  const p = a * b;
  if (!((p > 2.004168360008973e-292 && p < 1e300) || (p < -2.004168360008973e-292 && p > -1e300))
      || !(c < 1e300 && c > -1e300) || !(a < 1e299 && a > -1e299 && b < 1e299 && b > -1e299)) {
    return _crtFmaSlow(a, b, c);
  }
  let t = 134217729 * a;
  const ah = t - (t - a);
  const al = a - ah;
  t = 134217729 * b;
  const bh = t - (t - b);
  const bl = b - bh;
  const pe = ((ah * bh - p) + ah * bl + al * bh) + al * bl; // p + pe = a*b exactly
  const th = c + p;
  let bb = th - c;
  const tl = (c - (th - bb)) + (p - bb); // th + tl = c + p exactly
  const s = tl + pe;
  bb = s - tl;
  const e = (tl - (s - bb)) + (pe - bb); // s + e = tl + pe exactly
  if (e === 0) return th + s;
  _CRT_F[0] = s;
  const lo = _CRT_U[_CRT_LO];
  if (lo & 1) return th + s; // s odd: round-to-odd(tl + pe) = s
  if (s === 0) return th + (e > 0 ? 5e-324 : -5e-324);
  if ((e > 0) === (s > 0)) _CRT_U[_CRT_LO] = lo + 1; // one ulp towards s + e
  else if (lo === 0) {
    _CRT_U[_CRT_LO] = 0xffffffff;
    _CRT_U[_CRT_HI] -= 1;
  } else _CRT_U[_CRT_LO] = lo - 1;
  return th + _CRT_F[0];
}
function _crtFmaSlow(a, b, c) {
  if (a === 0 || b === 0 || !Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) return a * b + c;
  // extreme magnitudes: exact BigInt sum, rounded to nearest even
  const da = _crtDecompose(a);
  const db = _crtDecompose(b);
  const dc = _crtDecompose(c);
  const pm = da.neg !== db.neg ? -(da.m * db.m) : da.m * db.m;
  const pe = da.e + db.e;
  const cm = dc.neg ? -dc.m : dc.m;
  const e = Math.min(pe, dc.e);
  const sum = (pm << BigInt(pe - e)) + (cm << BigInt(dc.e - e));
  if (sum === 0n) return a * b + c;
  const neg = sum < 0n;
  const mag = neg ? -sum : sum;
  const bitlen = mag.toString(2).length;
  const topExp = bitlen - 1 + e;
  const keep = topExp < -1022 ? 53 - (-1022 - topExp) : 53;
  let res;
  if (keep <= 0) {
    res = keep === 0 && mag !== 1n << BigInt(bitlen - 1) ? 5e-324 : 0;
  } else {
    const drop = bitlen - keep;
    let q = mag;
    if (drop > 0) {
      q = mag >> BigInt(drop);
      const rem = mag - (q << BigInt(drop));
      const half = 1n << BigInt(drop - 1);
      if (rem > half || (rem === half && (q & 1n) === 1n)) q += 1n;
    } else q = mag << BigInt(-drop);
    const sh = drop + e;
    res = sh > 960 ? Number(q) * 2 ** 960 * 2 ** (sh - 960) : sh < -960 ? Number(q) * 2 ** -960 * 2 ** (sh + 960) : Number(q) * 2 ** sh;
  }
  return neg ? -res : res;
}
function _crtDecompose(x) {
  _CRT_F[0] = x;
  const hi = _CRT_U[_CRT_HI];
  const ex = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(_CRT_U[_CRT_LO]);
  if (ex !== 0) m |= 1n << 52n;
  return { neg: hi >>> 31 === 1, m, e: ex === 0 ? -1074 : ex - 1075 };
}

/** exp(x) exactly as ucrtbase computes it (2^(j/64) table + degree-6 polynomial). */
export function crtExp(x) {
  if (x !== x) return x;
  if (!(x <= 709.782712893384)) return Infinity; // overflow (errno ERANGE)
  if (!(x >= -744.0346068132731)) return 0; // underflow, -inf
  if (x <= 1.4901161193847656e-08 && x >= -1.4901161193847656e-08) return 1.0 + x;
  const nf = Math.trunc(x * 92.33248261689366); // x * 64/ln2, truncated
  const n = nf | 0;
  const r = nf * -2.5728046223276688e-14 + (nf * -0.010830424696223417 + x); // nf * ln2/64 (hi) is exact
  let p = _crtFma(r, 0.001388888888888889, 0.008333333333333333);
  p = _crtFma(r, p, 0.041666666666666664);
  p = _crtFma(r, p, 0.16666666666666666);
  p = _crtFma(r, p, 0.5);
  const q = _crtFma(p, r * r, r);
  const j = n & 63;
  const m = n >> 6;
  const y = (q * _CRT_EXP_T1[j] + _CRT_EXP_T2[j]) + _CRT_EXP_T3[j];
  if (m === 1024) return (y * 2) * 8.98846567431158e307; // exponent add past 2^1023 (x == 709.782712893384)
  return y * _crtPow2(m); // exponent add (exact), or one rounding for subnormal results
}

/** cos(x) exactly as ucrtbase computes it (|x| < 2e7: Cody-Waite reduction by pi/2 with FMA). */
export function crtCos(x) {
  const ax = x < 0 ? -x : x;
  if (ax > 0.7853981633974483 || ax !== ax) {
    if (!(ax < Infinity)) return NaN;
    if (ax >= 20000000.0) return Math.cos(x); // Payne-Hanek range: not reproduced
    const n = _crtFma(ax, 0.6366197723675814, 6755399441055744.0) - 6755399441055744.0;
    const r1 = _crtFma(-n, 1.5707963267948966, ax);
    const p2 = n * 6.123233995736757e-17;
    const e1 = _crtFma(6.123233995736757e-17, n, -p2);
    let t5 = r1 - p2;
    const t4 = (r1 - t5) - p2;
    const r = _crtFma(-n, 6.123233995736757e-17, r1);
    t5 = ((t5 - r) + t4) - e1;
    const rr = _crtFma(-n, 8.478427660368898e-32, t5);
    const region = n & 3;
    const x2 = r * r;
    let res;
    if (region & 1) {
      let p = _crtFma(x2, 1.5918144304485914e-10, -2.5051132068021698e-08);
      p = _crtFma(x2, p, 2.7557316103728802e-06);
      p = _crtFma(x2, p, -0.00019841269836761127);
      p = _crtFma(x2, p, 0.00833333333333095);
      const x3 = r * x2;
      res = r - _crtFma(-x3, -0.16666666666666666, x2 * (rr * 0.5 - x3 * p) - rr);
    } else {
      const hx2 = x2 * 0.5;
      const t = 1.0 - hx2;
      const corr = _crtFma(-r, rr, (1.0 - t) - hx2);
      let p = _crtFma(x2, -1.138263981623609e-11, 2.0876146382372144e-09);
      p = _crtFma(x2, p, -2.755731727234489e-07);
      p = _crtFma(x2, p, 2.4801587298767044e-05);
      p = _crtFma(x2, p, -0.0013888888888887398);
      p = _crtFma(x2, p, 0.041666666666666664);
      res = _crtFma(x2 * x2, p, corr) + t;
    }
    return (region + 1) & 2 ? -res : res;
  }
  if (ax >= 0.0001220703125) {
    const x2 = x * x;
    let p = _crtFma(x2, -1.138263981623609e-11, 2.0876146382372144e-09);
    p = _crtFma(x2, p, -2.755731727234489e-07);
    p = _crtFma(x2, p, 2.4801587298767044e-05);
    p = _crtFma(x2, p, -0.0013888888888887398);
    p = _crtFma(x2, p, 0.041666666666666664);
    p = _crtFma(x2, p, -0.5);
    return _crtFma(x2, p, 1.0);
  }
  if (ax >= 7.450580596923828e-09) return _crtFma(-(x * 0.5), x, 1.0);
  return 1.0;
}

/** pow(x, y) exactly as ucrtbase computes it (C99 special cases; double-double log, then exp). */
export function crtPow(x, y) {
  if (x > 0 && x < Infinity && x !== 1) {
    const ay = y < 0 ? -y : y;
    if (ay >= 1.0842021724855044e-19 && ay < 18446744073709551616 && y !== 1) return _crtPowCore(x, y, false);
  }
  if (y === 0) return 1.0;
  if (y === 1) return x;
  if (x !== x || y !== y) return x === 1 ? 1.0 : NaN;
  const ay = y < 0 ? -y : y;
  if (x < 0 || Object.is(x, -0)) {
    if (ay >= 18446744073709551616) { // |y| >= 2^64 or inf: even
      if (x === -Infinity) return y > 0 ? Infinity : 0;
      if (x === 0) return y > 0 ? 0 : Infinity;
      if (x === -1) return 1.0;
      return (-x < 1) === (y < 0) ? Infinity : 0;
    }
    if (ay < 1 || !Number.isInteger(y)) {
      if (x === -Infinity) return y > 0 ? Infinity : 0;
      if (x === 0) return y > 0 ? 0 : Infinity;
      return NaN;
    }
    const odd = ay < 9007199254740992 && ay % 2 === 1;
    if (x === 0) return y > 0 ? (odd ? -0 : 0) : (odd ? -Infinity : Infinity);
    if (x === -1) return odd ? -1.0 : 1.0;
    if (x === -Infinity) return y > 0 ? (odd ? -Infinity : Infinity) : (odd ? -0 : 0);
    return _crtPowCore(-x, y, odd);
  }
  if (x === 1) return 1.0;
  if (x === 0) return y > 0 ? 0 : Infinity;
  if (x === Infinity) return y > 0 ? Infinity : 0;
  if (ay >= 18446744073709551616) return (x < 1) === (y < 0) ? Infinity : 0;
  return 1.0 + y; // |y| < 2^-63
}

/** The CRT's main pow path: x > 0 finite, 2^-63 <= |y| < 2^64; neg = negate the result. */
function _crtPowCore(x, y, neg) {
  _CRT_F[0] = x;
  let hi = _CRT_U[_CRT_HI];
  let lo = _CRT_U[_CRT_LO];
  let e = (hi >>> 20) - 1023;
  if (e === -1023) { // subnormal x: (mantissa | 1.0) - 1.0 = x * 2^1022
    _CRT_F[0] = _crtFromBits((hi & 0xfffff) | 0x3ff00000, lo) - 1.0;
    hi = _CRT_U[_CRT_HI];
    lo = _CRT_U[_CRT_LO];
    e = (hi >>> 20) - 2045;
  }
  // log(x) = e*ln2 + log(2F) + log(1 - u), F = mantissa rounded to 8 bits, u = (F - f)/F
  const j = ((hi >>> 12) & 0xff) + ((hi >>> 11) & 1);
  const d = (0.5 + j / 512) - _crtFromBits((hi & 0xfffff) | 0x3fe00000, lo);
  const a = d * _CRT_LOG_INV_HI[j];
  const b = d * _CRT_LOG_INV_LO[j];
  const u = a + b;
  let p = _crtFma(u, 0.1428571428571429, 0.16666666666666666);
  p = _crtFma(u, p, 0.2);
  p = _crtFma(u, p, 0.25);
  p = _crtFma(u, p, 0.3333333333333333);
  p = _crtFma(u, p, 0.5);
  p = _crtFma(u * u, p, (a - u) + b);
  const s = _crtFma(e, 5.7699990475432854e-08, -p) + _CRT_LOG_LO[j];
  const t3 = s - u;
  const h = e * 0.6931471228599548 + _CRT_LOG_HI[j]; // e * ln2 (hi) is exact
  const L = h + t3;
  _CRT_F[0] = L;
  _CRT_U[_CRT_LO] &= 0xf8000000;
  const Lh = _CRT_F[0];
  _CRT_F[0] = y;
  _CRT_U[_CRT_LO] &= 0xf8000000;
  const yh = _CRT_F[0];
  const Ll = (((h - L) + t3) + (s - (u + t3))) + (L - Lh);
  // z + zlo = y * log(x)
  const yl = y - yh;
  const c3 = ((yl * Ll) + (yl * Lh)) + (Ll * yh);
  const c6 = Lh * yh;
  const z = c6 + c3;
  const zlo = (c6 - z) + c3;
  // exp(z + zlo)
  const t = z * 92.33248261689366;
  if (t > 65536.0) return neg ? -Infinity : Infinity;
  if (!(t >= -68800.0)) return neg ? -0 : 0;
  let n = Math.floor(t); // round half to even (cvtpd2dq)
  const fr = t - n;
  if (fr > 0.5 || (fr === 0.5 && n % 2 !== 0)) n += 1;
  const j2 = n & 63;
  const m = (n - j2) >> 6;
  const r = ((n * -0.010830424260348082 + z) + n * -4.359010638708991e-10) + zlo; // n * ln2/64 (hi) is exact
  let q = _crtFma(r, 0.001388888888888889, 0.008333333333333333);
  q = _crtFma(r, q, 0.041666666666666664);
  q = _crtFma(r, q, 0.16666666666666666);
  q = _crtFma(r, q, 0.5);
  q = _crtFma(r, q, 1.0) * r;
  const w = (q * _CRT_EXP_T3[j2] + (q * _CRT_EXP_T2[j2] + _CRT_EXP_T2[j2])) + _CRT_EXP_T3[j2];
  let res;
  if (m === 1024) {
    if (w >= 1.0) return neg ? -Infinity : Infinity;
    res = (w * 2) * 8.98846567431158e307;
  } else if (m > -1022 || (m === -1022 && w >= 1.0)) {
    res = w * _crtPow2(m);
  } else if (z < -744.0346068132731) {
    res = 5e-324;
  } else {
    res = w * (m < -1074 ? 5e-324 : _crtPow2(m));
  }
  return neg ? -res : res;
}

// ── Windows UCRT log / log2 / log1p (appended) ─────────────────────────
// Same method as crtExp / crtCos / crtPow above: ucrtbase's code replayed operation by operation.
// log has an FMA3 variant (used here); log2 has none; log1p is log(1+x) - ((1+x)-1-x)/(1+x).
/** _CRT_P2[k] = 2^(k - 1074), k = 0..2097 (every power of two representable as a double). */
const _CRT_P2 = new Float64Array(2098);
_CRT_P2[0] = 5e-324;
for (let i = 1; i < 2098; i++) _CRT_P2[i] = _CRT_P2[i - 1] * 2;
// 1/F (F = (1 + j/256)/2), and the head/tail of log(j/64) used by log2
const _CRT_LOG_INV = _crtTable(
  '40000000000000003fffe01fe01fe0203fffc07f01fc07f03fffa11caa01fa123fff81f81f81f8203fff6310aca0dbb53fff44659e4a42713fff25f644230ab5' +
  '3fff07c1f07c1f083ffee9c7f8458e023ffecc07b301ecc03ffeae807aba01eb3ffe9131abf0b7673ffe741aa59750e43ffe573ac901e5743ffe3a9179dc1a73' +
  '3ffe1e1e1e1e1e1e3ffe01e01e01e01e3ffde5d6e3f8868a3ffdca01dca01dca3ffdae6076b981db3ffd92f2231e7f8a3ffd77b654b82c343ffd5cac807572b2' +
  '3ffd41d41d41d41d3ffd272ca3fc5b1a3ffd0cb58f6ec0743ffcf26e5c44bfc63ffcd85689039b0b3ffcbe6d9601cbe73ffca4b3055ee1913ffc8b265afb8a42' +
  '3ffc71c71c71c71c3ffc5894d10d49863ffc3f8f01c3f8f03ffc26b5392ea01c3ffc0e070381c0e03ffbf583ee868d8b3ffbdd2b899406f73ffbc4fd65883e7b' +
  '3ffbacf914c1bad03ffb951e2b18ff233ffb7d6c3dda338b3ffb65e2e3beee053ffb4e81b4e81b4f3ffb37484ad806ce3ffb2036406c80d93ffb094b31d922a4' +
  '3ffaf286bca1af283ffadbe87f94905e3ffac5701ac5701b3ffaaf1d2f87ebfd3ffa98ef606a63be3ffa82e65130e1593ffa6d01a6d01a6d3ffa574107688a4a' +
  '3ffa41a41a41a41a3ffa2c2a87c51ca03ffa16d3f97a4b023ffa01a01a01a01a3ff9ec8e951033d93ff9d79f176b682d3ff9c2d14ee4a1023ff9ae24ea5510da' +
  '3ff999999999999a3ff9852f0d8ec0ff3ff970e4f80cb8723ff95cbb0be377ae3ff948b0fcd6e9e03ff934c67f9b2ce63ff920fb49d0e2293ff90d4f120190d5' +
  '3ff8f9c18f9c18fa3ff8e6527af1373f3ff8d3018d3018d33ff8bfce8062ff3a3ff8acb90f6bf3aa3ff899c0f601899c3ff886e5f0abb04a3ff87427bcc092b9' +
  '3ff86186186186183ff84f00c27806143ff83c977ab2bedd3ff82a4a0182a4a03ff81818181818183ff80601806018063ff7f405fd017f403ff7e225515a4f1d' +
  '3ff7d05f417d05f43ff7beb3922e017c3ff7ad2208e0ecc33ff79baa6bb6398b3ff78a4c8178a4c83ff77908119ac60d3ff767dce434a9b13ff756cac201756d' +
  '3ff745d1745d17463ff734f0c541fe8d3ff724287f46debc3ff713786d9c7c093ff702e05c0b81703ff6f26016f260173ff6e1f76b4337c73ff6d1a62681c861' +
  '3ff6c16c16c16c173ff6b1490aa31a3d3ff6a13cd15372903ff691473a88d0c03ff68168168168173ff6719f3601671a3ff661ec6a5122f93ff6524f853b4aa3' +
  '3ff642c8590b21643ff63356b88ac0de3ff623fa770162403ff614b36831ae943ff60581605816063ff5f66434292dfc3ff5e75bb8d015e73ff5d867c3ece2a5' +
  '3ff5c9882b9310573ff5babcc647fa913ff5ac056b015ac03ff59d61f123ccaa3ff58ed2308158ed3ff58056015805603ff571ed3c506b3a3ff56397ba7c52e2' +
  '3ff55555555555553ff54725e6bb82fe3ff5390948f40feb3ff52aff56a8054b3ff51d07eae2f8153ff50f22e111c4c53ff50150150150153ff4f38f62dd4c9b' +
  '3ff4e5e0a72f05393ff4d843bedc2c4c3ff4cab88725af6e3ff4bd3edda68fe13ff4afd6a052bf5b3ff4a27fad76014a3ff49539e3b2d0673ff4880522014880' +
  '3ff47ae147ae147b3ff46dce345960663ff460cbc7f5cf9a3ff453d9e2c776ca3ff446f86562d9fb3ff43a2730abee4d3ff42d6625d51f873ff420b5265e5951' +
  '3ff41414141414143ff40782d10e65663ff3fb013fb013fb3ff3ee8f42a5af073ff3e22cbce4a9023ff3d5d991aa75c63ff3c995a47babe73ff3bd60d9232955' +
  '3ff3b13b13b13b143ff3a524387ac8223ff3991c2c187f633ff38d22d366088e3ff38138138138143ff3755bd1c945ee3ff3698df3de07483ff35dce5f9f2af8' +
  '3ff3521cfb2b78c13ff34679ace013463ff33ae45b57bcb23ff32f5ced6a1dfa3ff323e34a2b10bf3ff3187758e9ebb63ff30d190130d1903ff301c82ac40260' +
  '3ff2f684bda12f683ff2eb4ea1fed14b3ff2e025c04b80973ff2d50a012d50a03ff2c9fb4d812ca03ff2bef98e5a37113ff2b404ad012b403ff2a91c92f3c105' +
  '3ff29e4129e4129e3ff293725bb804a53ff288b01288b0133ff27dfa38a1ce4d3ff27350b88127353ff268b37cd601273ff25e22708092f13ff2539d7e9177b2' +
  '3ff24924924924923ff23eb79717605b3ff23456789abcdf3ff22a0122a0122a3ff21fb78121fb783ff21579804855e63ff20b470c67c0d93ff2012012012012' +
  '3ff1f7047dc11f703ff1ecf43c7fb84c3ff1e2ef3b3fb8743ff1d8f5672e4abd3ff1cf06ada2811d3ff1c522fc1ce0593ff1bb4a4046ed293ff1b17c67f2bae3' +
  '3ff1a7b9611a7b963ff19e0119e0119e3ff19453808ca29c3ff18ab083902bdb3ff18118118118123ff1778a191bd6843ff16e06894273793ff1648d50fc3201' +
  '3ff15b1e5f75270d3ff151b9a3fdd5c93ff1485f0e0acd3b3ff13f0e8d3447243ff135c81135c8113ff12c8b89edc0ac3ff12358e75d30333ff11a3019a74826' +
  '3ff11111111111113ff107fbbe0110803ff0fef010fef0113ff0f5edfab325a23ff0ecf56be69c903ff0e406558260113ff0db20a88f46963ff0d24456359e3a' +
  '3ff0c9714fbcda3b3ff0c0a7868b41713ff0b7e6ec259dc83ff0af2f722eecb53ff0a6810a6810a73ff09ddba6af83603ff0953f390109543ff08cabb37565e2' +
  '3ff08421084210843ff07b9f29b8eae23ff073260a47f7c63ff06ab59c7912fb3ff0624dd2f1a9fc3ff059eea07275863ff05197f7d734043ff04949cc1664c5' +
  '3ff04104104104103ff038c6b78247fc3ff03091b51f5e1a3ff02864fc7729e93ff02040810204083ff0182436517a373ff01010101010103ff0080402010080' +
  '3ff0000000000000');
const _CRT_LOG2_A = _crtTable(
  '00000000000000003f8fc0a8000000003f9f8298000000003fa77458000000003faf0a30000000003fb341d7000000003fb6f0d2000000003fba926d00000000' +
  '3fbe2707000000003fc0d77e000000003fc29552800000003fc44d2b000000003fc5ff30000000003fc7ab89000000003fc9525a800000003fcaf3c900000000' +
  '3fcc8ff7800000003fce2707000000003fcfb918000000003fd0a324c00000003fd1675c800000003fd22941c00000003fd2e8e2800000003fd3a64c40000000' +
  '3fd4618bc00000003fd51aad800000003fd5d1bd800000003fd686c8000000003fd739d7c00000003fd7eaf8000000003fd89a33800000003fd9479400000000' +
  '3fd9f323c00000003fda9cec800000003fdb44f7400000003fdbeb4d800000003fdc8ff7c00000003fdd32fe400000003fddd46a000000003fde744240000000' +
  '3fdf128f400000003fdfaf58800000003fe02552a00000003fe0723e400000003fe0be72e00000003fe109f3800000003fe154c3c00000003fe19ee6a0000000' +
  '3fe1e85f400000003fe23130c00000003fe2795e000000003fe2c0e9e00000003fe307d7200000003fe34e28800000003fe393e0c00000003fe3d90260000000' +
  '3fe41d8fe00000003fe4618bc00000003fe4a4f8400000003fe4e7d8000000003fe52a2d200000003fe56bf9c00000003fe5ad40400000003fe5ee02a0000000' +
  '3fe62e42e0000000');
const _CRT_LOG2_B = _crtTable(
  '00000000000000003e361f807c79f3db3e6873c1980267c83e5ec65b9f88c69e3e58022c54cc2f993e62c37a3a1253303e615cad69737c933e4d256ab1b285e9' +
  '3e5b8abcb97a7aa23e6f34239659a5dc3e6e07fd48d301773e6b32df4799f4f63e6c29e4f4f21cf83e1086c848df1b593e4cf456b47641303e63a02ffcb63398' +
  '3e61e6a6886b09763e6b8abcb97a7aa23e6b578f8aa355523e6139c871afb9fc3e65d5d30701ce643e6de7bcb2d121423e6d708e984e16643e556945e9c72f36' +
  '3e20e2f613e85bda3e3cb7e0b42724f63e6fac04e52846c73e5e9b14aec442be3e6b5de8034e71263e6dc157e1b259d33e3b05096ad69c623e5c2116faba4cdd' +
  '3e665fcc25f95b473e5a9a08498d48503e6de647b1465f773e5da71b7bf7861d3e3e6a6886b097603e6f0075eab0ef643e33071282fb989b3e60eb43c3f1bed2' +
  '3e5faf06ecb35c843e4ef1e63db35f683e469743fb1a71a53e6c1cdf404e57963e4094aa0ada625e3e6e2d4c96fde3ec3e62f4d5e9a98f343e6467c96ecc5cbe' +
  '3e6e7040d03dec5a3e67bebf4282de363e6289b11aeb783f3e5a891d1772f5383e634f10be1fb5913e6d9ce1d316eb933e63562a19a9c4423e54e2adf548084c' +
  '3e508ce55cc8c97a3e30e2f613e85bda3e6db03ebb0227bf3e61b75bb09cb0983e496f16abb9df223e65b3f399411c623e586b3e59f653553e52482ceae1ac12' +
  '3e6efa39ef35793c');

/** log(x) exactly as ucrtbase computes it (FMA3 path: 8-bit table reduction, or r/(2+r) near 1). */
export function crtLog(x) {
  if (x !== x) return x;
  if (x === Infinity) return x;
  if (!(x > 0)) return x === 0 ? -Infinity : NaN; // log(±0) = -inf (ERANGE), log(<0) = log(-inf) = nan (EDOM)
  _CRT_F[0] = x;
  let hi = _CRT_U[_CRT_HI];
  let e = (hi >>> 20) - 1023;
  const r = x - 1.0;
  let f; // the mantissa scaled to [0.5, 1)
  if (e === -1023) { // subnormal x: v = x * 2^1022 = (mantissa | 1.0) - 1.0 is normal
    const v = x * 4.49423283715579e307;
    _CRT_F[0] = v;
    hi = _CRT_U[_CRT_HI];
    const ev = (hi >>> 20) - 1023;
    e = ev - 1022;
    f = v * _CRT_P2[1073 - ev];
  } else {
    if ((r < 0 ? -r : r) < 0.0625) {
      // log(1 + r) = r - r*s + (2s)^3 * (...) with s = r / (2 + r)
      const s = r / (2.0 + r);
      const rs = r * s;
      const s2 = s + s;
      const w = s2 * s2;
      let a = _crtFma(w, 0.012500000003771751, 0.08333333333333179);
      const b = _crtFma(w, 0.0004348877777076146, 0.0022321399879194482);
      const w3 = w * s2;
      a *= w3;
      a = _crtFma(b, (w3 * w3) * s2, a);
      return r + (a - rs);
    }
    f = x * _CRT_P2[1073 - e];
  }
  // log(x) = e*ln2 + log(2F) + log(1 - u), F = mantissa rounded to 8 bits, u = (F - f)/F
  const j = ((hi >>> 12) & 0xff) + ((hi >>> 11) & 1);
  const u = ((0.5 + j / 512) - f) * _CRT_LOG_INV[j];
  const u2 = u * u;
  let p3 = _crtFma(u, 0.16666666666666666, 0.2);
  const p5 = _crtFma(u, 0.3333333333333333, 0.5);
  p3 = _crtFma(u, p3, 0.25);
  let q = _crtFma(p5, u2, u); // -log(1 - u) = u + u^2/2 + ... + u^6/6
  q = _crtFma(p3, u2 * u2, q);
  const t = _CRT_LOG_LO[j] + _crtFma(e, 5.7699990475432854e-08, -q);
  return (e * 0.6931471228599548 + _CRT_LOG_HI[j]) + t; // e * ln2 (hi) is exact
}

/** log2(x) exactly as ucrtbase computes it (no FMA variant: 7-bit table reduction, or r/(2+r) near 1). */
export function crtLog2(x) {
  if (x !== x) return x;
  if (x === Infinity) return x;
  if (!(x > 0)) return x === 0 ? -Infinity : NaN; // log2(±0) = -inf (ERANGE), log2(<0) = nan (EDOM)
  _CRT_F[0] = x;
  let hi = _CRT_U[_CRT_HI];
  const lo = _CRT_U[_CRT_LO];
  const LOG2E_HI = 1.4426918029785156;
  const LOG2E_LO = 3.2379104477823597e-06;
  const d = (hi - 0x3fee0faa) >>> 0;
  if (d < 0x2f882 || (d === 0x2f882 && lo === 0)) { // x in [0x3fee0faa00000000, 0x3ff1082c00000000]
    const r = x - 1.0;
    _CRT_F[0] = r;
    _CRT_U[_CRT_LO] = 0;
    const rh = _CRT_F[0]; // r with the low 32 bits of the pattern cleared
    const s = r / (r + 2.0);
    const s2 = s + s;
    const sr = s * r;
    const rl = r - rh;
    const w = s2 * s2;
    let p = w * 0.0004348877777076146 + 0.0022321399879194482;
    p = p * w + 0.012500000003771751;
    p = (p * w + 0.08333333333333179) * (w * s2) - sr;
    p += rl;
    return ((p * LOG2E_LO + rh * LOG2E_LO) + p * LOG2E_HI) + rh * LOG2E_HI;
  }
  let e = (hi >>> 20) - 1023;
  let v = x;
  if (e === -1023) { // subnormal x: v = x * 2^60 (the CRT ORs 0x03d exponent bits in, then subtracts 2^-962)
    v = x * 1152921504606846976;
    _CRT_F[0] = v;
    hi = _CRT_U[_CRT_HI];
    e = (hi >>> 20) - 1083;
  }
  const f = v * _CRT_P2[1073 - ((hi >>> 20) - 1023)]; // mantissa scaled to [0.5, 1)
  const j = 64 + ((hi >>> 14) & 0x3f) + ((hi >>> 13) & 1); // F = j/128
  const F = j * 0.0078125;
  const dd = f - F;
  const u = dd / (dd * 0.5 + F); // 2 atanh(u/2) = log(f/F)
  const w = u * u;
  let p = w * 0.0022321981075855983 + 0.012499999997813867;
  p = (p * w + 0.08333333333333336) * w;
  p = (p * u + u) + _CRT_LOG2_B[j - 64];
  const A = _CRT_LOG2_A[j - 64];
  return (e + A * LOG2E_HI) + ((p * LOG2E_LO + A * LOG2E_LO) + p * LOG2E_HI);
}

/** log1p(x) as ucrtbase computes it: log(1 + x) - ((1 + x) - 1 - x) / (1 + x), with the CRT log. */
export function crtLog1p(x) {
  if (x === 0 || x !== x || x === Infinity) return x;
  if (!(x >= -1)) return NaN; // x < -1, -inf (EDOM)
  if (x === -1) return -Infinity; // (ERANGE)
  const y = x + 1.0;
  return crtLog(y) - ((y - 1.0) - x) / y;
}
