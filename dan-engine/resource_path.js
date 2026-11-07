// resource_path.js — resolve bundled resources (config/*.json, msd.exe) in the virtual file system.
//
// JavaScript port of engine/dan_engine/resource_path.py (DanOverlay 3.1.0 by 8doul).
// The engine always runs "frozen" here: sys._MEIPASS is vfs.RES_ROOT, filled by setup.js.
import { path, sys } from './vfs.js';

/** Project root: sys._MEIPASS in the frozen build (always the case in the overlay). */
export function get_root() {
  return sys._MEIPASS;
}

/** Build a resource path relative to the root, e.g. resource_path('config', 'sr_means.json'). */
export function resource_path(...parts) {
  return path.join(get_root(), ...parts);
}
