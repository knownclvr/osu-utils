// msd.js — in-process replacement for DanOverlay's msd.exe.
//
// msd.exe is a small Rust CLI over minacalc-rs 0.2.x: it reads [{"notes","time"}]
// rows from stdin, validates them (minacalc-rs Note::validate), runs
// Calc::calc_msd (MinaCalc 505) and prints the 1.0x skillsets as JSON (f32 via ryu).
// Here MinaCalc runs as WebAssembly (minacalc.mjs/.wasm, built by
// tools/minacalc_wasm/build.ps1 with an exact emulation of the CPU's RSQRTSS) and
// the result is converted to the doubles Python gets from json.loads(stdout).
// Verified bit-exact against msd.exe on 114 map/rate cases (tools/msd_compare.mjs).
import { f32JsonValue } from './py.js';

const KEYS = ['overall', 'stream', 'jumpstream', 'handstream', 'stamina', 'jackspeed', 'chordjack', 'technical'];
let M = null;
let calc = 0;

/** Load the WebAssembly module once. `moduleOptions` is passed to createMinaCalc (e.g. locateFile). */
export async function initMinaCalc(moduleOptions = {}) {
  if (M) return;
  const { default: createMinaCalc } = await import('./minacalc.mjs');
  M = await createMinaCalc(moduleOptions);
  calc = M._create_calc();
}
export const isReady = () => M !== null;

/**
 * Behaves like `subprocess.run([msd.exe], input=json.dumps(rows))` as used by
 * minacalc_bridge._calc_native: returns { returncode, data } where data is the parsed
 * stdout object (keys as msd.exe prints them) or null on failure.
 */
export function runMsd(rows) {
  if (!M) return { returncode: 1, data: null, error: 'MinaCalc not initialised' };
  if (!rows.length) return { returncode: 1, data: null, error: 'No notes provided for calculation' };
  for (const r of rows) {
    if (r.notes === 0) return { returncode: 1, data: null, error: 'Note must have at least one column' };
    if (r.notes > 0b1111) return { returncode: 1, data: null, error: 'Note bitflags exceed 4K limit' };
    if (Math.fround(r.time) < 0) return { returncode: 1, data: null, error: 'Row time cannot be negative' };
  }
  const n = rows.length;
  const ptr = M._malloc(n * 8);
  const out = M._malloc(32);
  try {
    const u32 = new Uint32Array(M.HEAPU32.buffer, ptr, n * 2);
    const f32 = new Float32Array(M.HEAPF32.buffer, ptr, n * 2);
    for (let i = 0; i < n; i++) {
      u32[i * 2] = rows[i].notes >>> 0;
      f32[i * 2 + 1] = rows[i].time;
    }
    M._mc_calc_msd_rate(calc, ptr, n, 10, out);
    const vals = new Float32Array(M.HEAPF32.buffer, out, 8);
    const data = {};
    for (let i = 0; i < 8; i++) {
      const v = vals[i];
      if (v < 0 || v > 1000) return { returncode: 1, data: null, error: `Score ${v} is out of reasonable bounds` };
      data[KEYS[i]] = Number.isNaN(v) ? null : f32JsonValue(v);
    }
    return { returncode: 0, data };
  } finally {
    M._free(ptr);
    M._free(out);
  }
}
