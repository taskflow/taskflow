'use strict';
// Parallelism sweep worker (inlined in index.html as #src-sweep — keep in sync).
//
// init  : {type:'init', beg, end, workers, minUs, maxUs}
//         Builds, per physical worker (executor, worker), the union of its busy
//         intervals across all nesting levels. Child tasks of a subflow/recursive
//         task overlap their parent on the same worker, so the union avoids
//         counting one thread twice.
// query : {type:'query', id, x0, x1, bins, phys}
//         Returns avg[b] = average number of busy workers in bin b (exact, by
//         time-weighted overlap, at any zoom level), and the total busy time.
//         phys: optional array of physical-worker indices to include.
// After init the worker also posts {type:'summary'} for the whole run:
//         busy[p]    total busy time of physical worker p
//         keys[p]    its label, e.g. "E0.W3"
//         low / idle time during which <=1 / 0 workers were busy
//         stretches  longest [begin,end] stretches with <=1 worker busy

let U = [];   // per physical worker: {b:Float64Array, e:Float64Array} sorted, disjoint
// Precomputed busy time of all workers over FINE equal slots of the whole run,
// as a prefix sum. Unfiltered queries whose bins span at least FINE_MIN slots
// are answered from it in O(bins) — independent of the number of tasks.
const FINE = 1 << 20, FINE_MIN = 16;
let fineP = null, fineMin = 0, fineW = 0;
let K = [];   // per physical worker: label "E<executor>.W<worker>"

function buildUnions(B, E, W) {
  const groups = new Map();
  for (const w of W) {
    const k = w.executorId + '_' + w.workerIdx;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(w);
  }
  K = [...groups.values()].map(rows => 'E'+rows[0].executorId+'.W'+rows[0].workerIdx);
  U = [...groups.values()].map(rows => {
    let n = 0;
    for (const r of rows) n += r.length;
    const idx = new Uint32Array(n);
    let c = 0;
    for (const r of rows) for (let s = 0; s < r.length; s++) idx[c++] = r.offset + s;
    // Segments within one row are already sorted by begin; only merge when
    // the worker has several levels.
    if (rows.length > 1) idx.sort((a, b) => B[a] - B[b]);
    const ub = new Float64Array(n), ue = new Float64Array(n);
    let m = 0;
    for (let k = 0; k < n; k++) {
      const i = idx[k], b = B[i], e = E[i];
      if (m && b <= ue[m - 1]) { if (e > ue[m - 1]) ue[m - 1] = e; }
      else { ub[m] = b; ue[m] = e; m++; }
    }
    return { b: ub.slice(0, m), e: ue.slice(0, m) };
  });
}

function query(x0, x1, bins, phys) {
  bins = Math.max(1, Math.floor(bins));
  const avg = new Float32Array(bins);
  const list = phys || U.map((_, i) => i);
  const span = x1 - x0;
  if (!(span > 0)) return { avg, busy: 0, n: list.length };
  const bw = span / bins;
  const acc = new Float64Array(bins), diff = new Float64Array(bins + 1);
  let busy = 0;

  for (const p of list) {
    const u = U[p];
    if (!u) continue;
    const b = u.b, e = u.e;
    // first interval ending after x0
    let lo = 0, hi = e.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (e[m] <= x0) lo = m + 1; else hi = m; }
    for (let k = lo; k < b.length && b[k] < x1; k++) {
      const s = b[k] > x0 ? b[k] : x0, t = e[k] < x1 ? e[k] : x1;
      if (t <= s) continue;
      busy += t - s;
      const fb = Math.min(bins - 1, Math.floor((s - x0) / bw));
      const lb = Math.min(bins - 1, Math.floor((t - x0) / bw));
      if (fb === lb) { acc[fb] += t - s; continue; }
      acc[fb] += x0 + (fb + 1) * bw - s;
      acc[lb] += t - (x0 + lb * bw);
      if (lb > fb + 1) { diff[fb + 1] += bw; diff[lb] -= bw; }  // full middle bins
    }
  }
  let run = 0;
  for (let i = 0; i < bins; i++) { run += diff[i]; avg[i] = (acc[i] + run) / bw; }
  return { avg, busy, n: list.length };
}

// Whole-run summary: sweep over every worker's busy intervals, tracking how
// many workers are busy, and record the stretches where at most one is.
function summarize(minUs, maxUs) {
  const P = U.length, busy = new Float64Array(P);
  let n = 0;
  for (const u of U) n += u.b.length;
  const S = new Float64Array(n), E = new Float64Array(n);
  let c = 0;
  U.forEach((u, p) => {
    for (let k = 0; k < u.b.length; k++) {
      busy[p] += u.e[k] - u.b[k];
      S[c] = u.b[k]; E[c] = u.e[k]; c++;
    }
  });
  S.sort(); E.sort();
  let i = 0, j = 0, cnt = 0, t = minUs, lowStart = minUs, low = 0, idle = 0;
  const stretches = [];
  while (i < n || j < n) {
    const isStart = i < n && (j >= n || S[i] <= E[j]);  // starts first on ties
    const tt = isStart ? S[i] : E[j];
    if (cnt <= 1) low += tt - t;
    if (cnt === 0) idle += tt - t;
    const prev = cnt;
    if (isStart) { cnt++; i++; } else { cnt--; j++; }
    if (prev <= 1 && cnt >= 2 && tt > lowStart) stretches.push([lowStart, tt]);
    if (prev >= 2 && cnt <= 1) lowStart = tt;
    t = tt;
  }
  if (maxUs > t) { low += maxUs - t; idle += maxUs - t; }
  if (cnt <= 1 && maxUs > lowStart) stretches.push([lowStart, maxUs]);
  stretches.sort((a, b) => (b[1] - b[0]) - (a[1] - a[0]));
  return { busy, keys: K, low, idle, stretches: stretches.slice(0, 5) };
}

function buildFine(minUs, maxUs) {
  fineP = null;
  if (!(maxUs > minUs)) return;
  fineMin = minUs; fineW = (maxUs - minUs) / FINE;
  const { avg } = query(minUs, maxUs, FINE, null);   // exact busy workers per slot
  fineP = new Float64Array(FINE + 1);
  for (let i = 0; i < FINE; i++) fineP[i + 1] = fineP[i] + avg[i] * fineW;
}
// Busy time in [fineMin, t], interpolating linearly inside a slot.
function fineAt(t) {
  let u = (t - fineMin) / fineW;
  if (u <= 0) return 0;
  if (u >= FINE) return fineP[FINE];
  const k = Math.floor(u);
  return fineP[k] + (u - k) * (fineP[k + 1] - fineP[k]);
}
function queryFast(x0, x1, bins, phys) {
  bins = Math.max(1, Math.floor(bins));
  if (phys || !fineP || !((x1 - x0) / bins >= FINE_MIN * fineW)) return query(x0, x1, bins, phys);
  const avg = new Float32Array(bins), bw = (x1 - x0) / bins;
  let prev = fineAt(x0);
  for (let i = 0; i < bins; i++) {
    const cur = fineAt(x0 + (i + 1) * bw);
    avg[i] = (cur - prev) / bw; prev = cur;
  }
  return { avg, busy: fineAt(x1) - fineAt(x0), n: U.length };
}

self.onmessage = ev => {
  const m = ev.data;
  if (m.type === 'init') {
    buildUnions(m.beg, m.end, m.workers);
    buildFine(m.minUs, m.maxUs);
    self.postMessage({ type: 'ready', numPhys: U.length });
    self.postMessage({ type: 'summary', ...summarize(m.minUs, m.maxUs) });
  } else if (m.type === 'query') {
    const r = queryFast(m.x0, m.x1, m.bins, m.phys);
    self.postMessage({ type: 'result', id: m.id, x0: m.x0, x1: m.x1, ...r }, [r.avg.buffer]);
  }
};
