import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';

// stdout은 본체와 주고받는 JSON 전용이다.
console.log = (...args) => console.error(...args);

const appDir = path.dirname(fileURLToPath(import.meta.url));
const { setupEngine } = await import('./dan-engine/setup.js');
await setupEngine({
  readFile: relativePath => fs.readFileSync(path.join(appDir, 'dan-engine', relativePath), 'utf8'),
});

const { runAnalysisJob } = await import('./dan-engine/job.js');
const { readSegmentSource, analyzeSegment } = await import('./dan-engine/live-segment.js');

const danNames = {
  Alpha: '알파', Beta: '베타', Gamma: '감마', Delta: '델타', Epsilon: '엡실론',
  Zeta: '제타', Eta: '에타', Theta: '세타', Iota: '이오타', Kappa: '카파',
};
const sublevelNames = {
  Low: '로우', 'Mid-Low': '미드로우', Mid: '미드', 'Mid-High': '미드하이', High: '하이',
};

function formatTier(result) {
  if (result?.error || !(Number(result?.dp) > 0)) return '계산 불가';
  const tier = danNames[result.dan_label]
    || String(result.dan_label || '').replace(/^(\d+)(st|nd|rd|th) Dan$/, '$1단');
  const sublevel = sublevelNames[result.sublevel] || result.sublevel || '';
  return `${tier} ${sublevel}`.trim();
}

let activeMap;
const requests = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

for await (const line of requests) {
  let request;
  try {
    request = JSON.parse(line);
    const modifiedAt = fs.statSync(request.path).mtimeMs;
    const cacheKey = JSON.stringify([request.path, modifiedAt, request.rate]);

    // 맵과 배속이 같으면 전체 분석은 다시 하지 않는다.
    if (activeMap?.cacheKey !== cacheKey) {
      const bytes = fs.readFileSync(request.path);
      const difficulty = bytes.toString('utf8').match(/^Version:(.*)$/m)?.[1].trim() || '';
      const rate = Number(request.rate) || 1;
      const job = {
        path: request.path,
        md5: crypto.createHash('md5').update(bytes).digest('hex'),
        bytes,
        mod: rate > 1 ? 'DT' : rate < 1 ? 'HT' : 'NM',
        rate,
        engine: 'isor',
        difficulty,
        modLabel: '',
        srOfficial: 0,
      };

      activeMap = {
        cacheKey,
        job,
        whole: runAnalysisJob(job),
        source: readSegmentSource(bytes),
        segments: new Map(),
      };
    }

    const sectionIndex = Math.floor(Math.max(0, request.timeMs) / (500 * activeMap.job.rate));
    let sectionLabel = activeMap.segments.get(sectionIndex);
    if (!sectionLabel) {
      const result = analyzeSegment(activeMap.job, activeMap.source, request.timeMs, activeMap.whole);
      sectionLabel = result.analysis_scope === 'unavailable'
        ? '계산 불가'
        : result.segment?.rest ? '휴식' : formatTier(result);
      activeMap.segments.set(sectionIndex, sectionLabel);
      if (activeMap.segments.size > 90) {
        activeMap.segments.delete(activeMap.segments.keys().next().value);
      }
    }

    process.stdout.write(JSON.stringify({
      id: request.id,
      whole: formatTier(activeMap.whole),
      segment: sectionLabel,
    }) + '\n');
  } catch (error) {
    process.stdout.write(JSON.stringify({
      id: request?.id,
      whole: '계산 불가',
      segment: '계산 불가',
      error: String(error.message || error),
    }) + '\n');
  }
}
