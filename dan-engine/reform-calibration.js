// ISOR -> REFORM scale, calibrated against the user's 2nd Pack - v1 references.
// Source: DDMythical, ~ EXTRA-ALPHA/BETA/GAMMA/DELTA/EPSILON ~ (Marathon), NM.
// Only course tiers are known, not exact sublevels. Keep anchors already inside
// their tier; move out-of-tier anchors to the nearest interior point (.1 margin).
// This avoids forcing every reference to Mid and distorting adjacent courses.
// Local excerpts use the SAME curve, never a course label. Jack/Speed/Tech - v1
// examples are regression checks, not additional numerical fitting anchors.
export const REFORM_CALIBRATION = {
  id: 'reform-2nd-v1-2026-11-06',
  anchors: [
    [11.00, 11.00], // Fade in above the unchanged 1st-10th Dan scale.
    [12.20, 11.90], // EXTRA-ALPHA: previously in Beta.
    [13.07, 12.90], // EXTRA-BETA: previously in Gamma.
    [13.71, 13.71], // EXTRA-GAMMA already matches.
    [14.71, 14.71], // EXTRA-DELTA already matches.
    [15.80, 15.80], // EXTRA-EPSILON already matches.
    [16.50, 16.50], // Fade out; no new evidence for Zeta-Kappa.
  ],
};

export function calibratedReformDp(dp) {
  const anchors = REFORM_CALIBRATION.anchors;
  if (!Number.isFinite(dp) || dp <= anchors[0][0] || dp >= anchors.at(-1)[0]) return dp;
  for (let i = 1; i < anchors.length; i++) {
    const [x1, y1] = anchors[i], [x0, y0] = anchors[i - 1];
    if (dp <= x1) return Math.round((y0 + (dp - x0) / (x1 - x0) * (y1 - y0)) * 100) / 100;
  }
  return dp;
}

export function calibrateReformResult(raw) {
  // Celestial/Signicial/LN have their own scales; only the 4K ISOR Reform DP
  // and its associated label/sublevel are changed. No title, path or hash matching.
  if (raw.engine !== 'isor' || raw.mode === '7k' || raw.ln_route === 'ln'
      || raw.error || !(Number(raw.dp) > 0)) return raw;
  const before = Number(raw.dp), dp = calibratedReformDp(before);
  if (before === dp) return raw;
  const tiers = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
    'Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta', 'Eta', 'Theta', 'Iota', 'Kappa'];
  const short = tiers[Math.min(tiers.length - 1, Math.max(0, Math.floor(dp) - 1))];
  const fraction = Math.round((dp - Math.floor(dp)) * 100) / 100;
  const sublevel = fraction <= .2 ? 'Low' : fraction <= .4 ? 'Mid-Low'
    : fraction <= .6 ? 'Mid' : fraction <= .8 ? 'Mid-High' : 'High';
  return { ...raw, dp, dan_short: short, dan_label: dp < 11 ? `${short} Dan` : short, sublevel,
    corrections: [...(raw.corrections || []), `reform_reference:${(dp - before).toFixed(2)}`],
    debug: { ...raw.debug, reform_calibration: { id: REFORM_CALIBRATION.id, raw_dp: before, calibrated_dp: dp } },
  };
}
