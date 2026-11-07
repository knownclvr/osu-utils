// setup.js — load the engine's resources into the virtual file system and start MinaCalc.
//
// Browser / Web Worker:  await setupEngine({ baseUrl: new URL('./', import.meta.url) })
// Node (tests):          await setupEngine({ readFile: (rel) => fs.readFileSync(...) })
import { putFile, RES_ROOT, CONFIG_FILES } from './vfs.js';
import { initMinaCalc } from './msd.js';

let ready = null;

export function setupEngine({ baseUrl, readFile } = {}) {
  if (ready) return ready;
  ready = (async () => {
    const get = async (rel) => {
      const res = await fetch(new URL(rel, baseUrl));
      if (!res.ok) throw new Error(`cannot load ${rel}: HTTP ${res.status}`);
      return res;
    };
    const load = readFile ? async (rel) => readFile(rel) : async (rel) => (await get(rel)).text();
    // tosu serves .wasm as "application/wasm; charset=utf-8", which WebAssembly's streaming
    // compile rejects (noisy console errors before Emscripten's fallback) — hand MinaCalc a
    // blob: URL with the exact MIME type instead.
    const wasm = readFile ? null : get('minacalc.wasm').then(async (res) => new Blob([await res.arrayBuffer()], { type: 'application/wasm' }));
    await Promise.all(CONFIG_FILES.map(async (name) => {
      putFile(`${RES_ROOT}/config/${name}`, await load(`resources/config/${name}`));
    }));
    putFile(`${RES_ROOT}/msd.exe`, ''); // minacalc_bridge._find_cli() looks for it
    if (!wasm) {
      await initMinaCalc({});
      return;
    }
    const wasmUrl = URL.createObjectURL(await wasm);
    try {
      await initMinaCalc({ locateFile: (f) => (f.endsWith('.wasm') ? wasmUrl : new URL(f, baseUrl).href) });
    } finally {
      URL.revokeObjectURL(wasmUrl);
    }
  })();
  return ready;
}
