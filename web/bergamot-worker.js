// bergamot-worker.js — browsermt/bergamot-translator, on the device.
//
// This is the measurement invariant **I1** rests on. I1 says fizh exists
// because bergamot's engine is intgemm-based and x86-tuned while phones are
// ARM. Comparing fizh's relaxed build against its own baseline build measures
// relaxed SIMD; it says nothing about that claim. Only bergamot running on the
// same phone, on the same model files, over the same corpus, does.
//
// A classic worker rather than a module one: the Emscripten glue is a script
// that declares `var Module` at top level and expects to be loaded, not
// imported. Setting `self.Module` first works because a redeclaring `var` at
// global scope keeps the existing value — the same trick
// `tools/eval/fetch-reference.sh` uses for Node.

let M = null;
let service = null;
let model = null;

// Firefox's own decoder settings, byte-for-byte what tools/eval/reference_engine.mjs
// passes, so a phone number is comparable to the desktop corpus runs.
const CONFIG = [
  "beam-size: 1",
  "normalize: 1.0",
  "word-penalty: 0",
  "max-length-break: 128",
  "mini-batch-words: 1024",
  "workspace: 128",
  "max-length-factor: 2.0",
  "skip-cost: true",
  "cpu-threads: 0",
  "quiet: true",
  "quiet-translation: true",
  "gemm-precision: int8shiftAlphaAll",
  "alignment: soft",
  // Marian builds one lexical shortlist per batch, so batch size changes the
  // answer (ADR 0018). Every translate() below is one sentence.
  "shortlist:\n    - dummy\n    - false",
].join("\n");

const reply = (id, ok, body) => postMessage({ id, ok, ...body });

async function bytes(url, what) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    postMessage({ progress: { what, got, total } });
  }
  const all = new Uint8Array(got);
  let at = 0;
  for (const c of chunks) { all.set(c, at); at += c.length; }
  return all;
}

function aligned(view, alignment) {
  const mem = new M.AlignedMemory(view.byteLength, alignment);
  mem.getByteArrayView().set(new Int8Array(view.buffer, view.byteOffset, view.byteLength));
  return mem;
}

const pct = (sorted, p) =>
  sorted[Math.min(sorted.length - 1, Math.floor((sorted.length * p) / 100))];

function translateOne(text) {
  const input = new M.VectorString();
  const options = new M.VectorResponseOptions();
  input.push_back(text);
  options.push_back({ qualityScores: false, alignment: false, html: false });
  const responses = service.translate(model, input, options);
  return responses.get(0).getTranslatedText().replace(/\n/g, " ");
}

const handlers = {
  async load({ engineJs, engineWasm, models }) {
    const wasm = await bytes(engineWasm, "bergamot engine");
    // Emscripten will not re-fetch when handed the bytes, so the download is
    // ours to time and ours to report.
    self.Module = {
      wasmBinary: wasm.buffer,
      print: () => {},
      printErr: () => {},
      locateFile: (p) => p,
    };
    const t0 = performance.now();
    importScripts(engineJs);
    M = self.Module;
    await new Promise((r) => { M.onRuntimeInitialized = r; });
    const instantiate = performance.now() - t0;

    // One model only. A bergamot pivot is two TranslationModels and a
    // different call shape; fizh's pivot is measured against a single-hop
    // bergamot only where that is stated.
    const m = models[0];
    const modelBytes = await bytes(m.model, `${m.pair} weights`);
    const vocabBytes = await bytes(m.vocab, `${m.pair} vocab`);
    const lexBytes = await bytes(m.lex, `${m.pair} shortlist`);

    const t1 = performance.now();
    const vocabs = new M.AlignedMemoryList();
    vocabs.push_back(aligned(vocabBytes, 64));
    model = new M.TranslationModel(CONFIG, aligned(modelBytes, 256),
                                   aligned(lexBytes, 64), vocabs, null);
    service = new M.BlockingService({ cacheSize: 0 });
    const build = performance.now() - t1;

    return {
      // Reported separately: instantiating 5 MB of engine is a different cost
      // from building a model out of already-downloaded bytes, and fizh's
      // "cold start" is only the second kind.
      instantiate_ms: instantiate,
      model_build_ms: build,
      cold_start_ms: instantiate + build,
      engine_bytes: wasm.length,
      model_bytes: modelBytes.length,
      heap_bytes: M.HEAPU8 ? M.HEAPU8.byteLength : null,
    };
  },

  bench({ cases, runs, warmup }) {
    if (!service) throw new Error("engine not loaded");
    const out = [];
    for (const c of cases) {
      for (let i = 0; i < warmup; i++) translateOne(c.text);
      const samples = [];
      let last = null;
      const t0 = performance.now();
      for (let i = 0; i < runs; i++) {
        const s = performance.now();
        last = translateOne(c.text);
        samples.push(performance.now() - s);
      }
      const wall = performance.now() - t0;
      const sorted = [...samples].sort((a, b) => a - b);
      out.push({
        name: c.name,
        runs,
        wall_ms: wall,
        min_ms: sorted[0],
        p50_ms: pct(sorted, 50),
        p90_ms: pct(sorted, 90),
        p99_ms: runs >= 300 ? pct(sorted, 99) : null,
        max_ms: sorted[sorted.length - 1],
        burst_p50_ms: pct([...samples.slice(0, Math.max(1, runs >> 2))].sort((a, b) => a - b), 50),
        steady_p50_ms: pct([...samples.slice(runs >> 1)].sort((a, b) => a - b), 50),
        sample: last,
      });
    }
    return { cases: out, heap_bytes: M.HEAPU8 ? M.HEAPU8.byteLength : null };
  },
};

onmessage = async (e) => {
  const { id, op, args } = e.data;
  const fn = handlers[op];
  if (!fn) return reply(id, false, { error: `unknown op ${op}` });
  try {
    reply(id, true, await fn(args ?? {}));
  } catch (err) {
    reply(id, false, {
      error: String(err && err.message ? err.message : err),
      fatal: err instanceof WebAssembly.RuntimeError,
    });
  }
};
