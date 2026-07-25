// worker.js — the runtime lives here and nowhere else.
//
// SPEC §10 (I10): a dedicated Web Worker, respawned on trap. The main thread
// never holds an instance, so a `RuntimeError` out of wasm takes down one
// worker and arrives on the page as a message — not a white screen.
//
// Every failure reaches the page as data. There are three kinds and they are
// not the same thing:
//
//   negative status   the runtime declined: src_too_long, out_too_small, ...
//   RuntimeError      a trap. The worker is dead after this; the page respawns.
//   OOM               memory.grow refused. Expected with two 33 MB artifacts
//                     resident, and a *result* rather than a crash.

import { boot, translate, lang } from "./fizh.js";

let ctx = null;
let loaded = [];

const reply = (id, ok, body) => postMessage({ id, ok, ...body });

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !onProgress) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress(got, total);
  }
  const all = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { all.set(c, at); at += c.length; }
  return all;
}

async function sha256(bytes) {
  const d = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

/** Percentiles, never a mean (SPEC §14). */
const pct = (sorted, p) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length * p) / 100))];

/** A p99 needs enough samples for the 99th percentile to be an estimate rather
 *  than a synonym for the worst one. Below this it is reported as `max`,
 *  labelled as max, and `p99_ms` is null. */
const P99_MIN_RUNS = 300;

function timeMany(text, from, to, runs) {
  const samples = [];
  let last = null;
  const wall0 = performance.now();
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    last = translate(ctx, text, from, to);
    samples.push(performance.now() - t0);
    if (last.error) return { error: last.error, code: last.code, runs: i + 1 };
  }
  const wall = performance.now() - wall0;
  const sorted = [...samples].sort((a, b) => a - b);

  // Burst and steady state are different numbers on a heterogeneous phone: the
  // scheduler starts on a big core and migrates, or DVFS ramps. Reporting one
  // p50 over both hides which of the two any budget is describing.
  const q = Math.max(1, runs >> 2);
  const burst = [...samples.slice(0, q)].sort((a, b) => a - b);
  const steady = [...samples.slice(runs >> 1)].sort((a, b) => a - b);

  return {
    runs,
    wall_ms: wall,
    min_ms: sorted[0],
    p50_ms: pct(sorted, 50),
    p90_ms: pct(sorted, 90),
    p99_ms: runs >= P99_MIN_RUNS ? pct(sorted, 99) : null,
    max_ms: sorted[sorted.length - 1],
    burst_p50_ms: pct(burst, 50),
    steady_p50_ms: pct(steady, 50),
    sample: last.text,
  };
}

const handlers = {
  async load({ wasmUrl, models }) {
    const wasm = await fetchBytes(wasmUrl, (got, total) =>
      postMessage({ progress: { what: "runtime", got, total } }));
    const blobs = [];
    const meta = [];
    for (const m of models) {
      const bytes = await fetchBytes(m.url, (got, total) =>
        postMessage({ progress: { what: m.pair, got, total } }));
      blobs.push(bytes);
      meta.push({ pair: m.pair, bytes: bytes.length, sha256: await sha256(bytes) });
    }
    // Cold start is init + load + repack, timed exactly as bench.zig does it.
    const t0 = performance.now();
    ctx = await boot(wasm, blobs);
    const cold = performance.now() - t0;
    loaded = models;
    return {
      cold_start_ms: cold,
      arena_bytes: ctx.arenaBytes,
      wasm_bytes: wasm.length,
      models: meta,
      heap_bytes: ctx.f.memory.buffer.byteLength,
    };
  },

  bench({ cases, runs, warmup }) {
    if (!ctx) throw new Error("no runtime loaded");
    const out = [];
    for (const c of cases) {
      // Warm and cold are reported separately and never blended (SPEC §14).
      for (let i = 0; i < warmup; i++) translate(ctx, c.text, c.from, c.to);
      const r = timeMany(c.text, c.from, c.to, runs);
      // A paragraph's cost is per sentence; §14 budgeted a total without ever
      // saying how many sentences were in it, which is why it could not be
      // passed or failed.
      if (c.sentences && r.p50_ms) r.per_sentence_ms = r.p50_ms / c.sentences;
      out.push({ name: c.name, sentences: c.sentences ?? null, ...r });
    }
    return { cases: out, heap_bytes: ctx.f.memory.buffer.byteLength };
  },

  translate({ text, from, to }) {
    if (!ctx) throw new Error("no runtime loaded");
    const t0 = performance.now();
    const r = translate(ctx, text, from, to);
    return { ...r, ms: performance.now() - t0 };
  },

  can({ from, to }) {
    if (!ctx) throw new Error("no runtime loaded");
    return { route: ctx.f.e.fizh_can_translate(ctx.handle, lang(from), lang(to)) };
  },
};

onmessage = async (e) => {
  const { id, op, args } = e.data;
  const fn = handlers[op];
  if (!fn) return reply(id, false, { error: `unknown op ${op}` });
  try {
    reply(id, true, await fn(args ?? {}));
  } catch (err) {
    // A trap arrives here as RuntimeError and the worker is not reusable after
    // it; the page decides whether to respawn. Either way it is reported.
    reply(id, false, {
      error: String(err && err.message ? err.message : err),
      fatal: err instanceof WebAssembly.RuntimeError,
    });
  }
};
