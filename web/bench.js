// bench.js — main thread. Owns the UI and the workers; never instantiates wasm.
//
// Two questions, two tables, deliberately not blended:
//
//   I1        fizh against *bergamot* on this phone. That is the claim the
//             project rests on: bergamot's engine is intgemm-based and x86-tuned
//             while phones are ARM. Comparing fizh to itself cannot test it.
//   SIMD      fizh relaxed against fizh baseline. That measures relaxed SIMD,
//             and it is a different question with a different answer.

const $ = (id) => document.getElementById(id);

// The SPEC §14 corpus, byte-for-byte what tools/bench.zig measures, so a phone
// number sits beside the desktop one and means the same thing.
const CLAUSES = [
  "el gato negro duerme en la mesa de la cocina",
  "no puedo ir contigo porque tengo que trabajar",
  "me gusta mucho la comida que preparaste ayer",
  "vamos a la playa si hace buen tiempo el domingo",
  "ella dijo que llegaria tarde a la reunion de hoy",
];

function message(words) {
  const parts = [];
  let n = 0;
  for (let i = 0; n < words; i++) {
    const c = CLAUSES[i % CLAUSES.length];
    parts.push(c);
    n += c.split(" ").length;
  }
  return parts.join(", ");
}

const PARA_SENTENCES = 8;
const paragraph = (n) =>
  Array.from({ length: n }, (_, i) => CLAUSES[i % CLAUSES.length] + ".").join(" ");

const SHORT = message(12);
const LONG = message(120);
const PARA = paragraph(PARA_SENTENCES);

// A p99 needs the samples to support it. The 12-token case is cheap enough to
// run 300 times; the others are not, and report max instead — labelled as max.
const RUNS = { short: 400, long: 40, para: 60 };
const WARMUP = 10;

let manifest = null;
let selected = null;
let workers = { fizh: null, berg: null };
const pending = new Map();
let nextId = 1;

function spawn(kind) {
  if (workers[kind]) workers[kind].terminate();
  const w = kind === "fizh"
    ? new Worker("./worker.js", { type: "module" })
    : new Worker("./bergamot-worker.js");
  w.onmessage = (e) => {
    const { id, ok, progress, ...body } = e.data;
    if (progress) return onProgress(progress);
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    ok ? p.resolve(body) : p.reject(Object.assign(new Error(body.error), body));
  };
  w.onerror = (e) => {
    const err = Object.assign(new Error(e.message || "worker died"), { fatal: true });
    for (const [, p] of pending) p.reject(err);
    pending.clear();
  };
  workers[kind] = w;
  return w;
}

function call(kind, op, args) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    workers[kind].postMessage({ id, op, args });
  });
}

function onProgress({ what, got, total }) {
  $("status").textContent = total
    ? `${what}: ${(got / 2 ** 20).toFixed(1)} / ${(total / 2 ** 20).toFixed(1)} MiB`
    : `${what}: ${(got / 2 ** 20).toFixed(1)} MiB`;
  $("bar").style.width = total ? `${(100 * got) / total}%` : "0%";
}

async function detectRelaxed() {
  try {
    const probe = await fetch("./fizh.probe.wasm");
    if (!probe.ok) return false;
    return WebAssembly.validate(await probe.arrayBuffer());
  } catch { return false; }
}

const round = (x) => (x === null || x === undefined ? null : Math.round(x * 100) / 100);

function casesFor(set) {
  const c = [
    { name: "12-token, direct", text: SHORT, from: set.from, to: set.to, runs: RUNS.short },
    { name: "120-token, direct", text: LONG, from: set.from, to: set.to, runs: RUNS.long },
    { name: "8-sentence paragraph", text: PARA, from: set.from, to: set.to,
      sentences: PARA_SENTENCES, runs: RUNS.para },
  ];
  if (set.pivot) {
    c.push({ name: "12-token, pivot", text: SHORT, from: set.from, to: set.pivot, runs: RUNS.long });
    c.push({ name: "120-token, pivot", text: LONG, from: set.from, to: set.pivot, runs: RUNS.long });
  }
  return c;
}

/** Cases are run one at a time so each can carry its own run count. */
async function benchAll(kind, cases) {
  const out = [];
  for (const c of cases) {
    $("status").textContent = `${kind}: ${c.name} (${c.runs} runs)`;
    const r = await call(kind, "bench", { cases: [c], runs: c.runs, warmup: WARMUP });
    out.push({ ...r.cases[0], heap_bytes: r.heap_bytes });
  }
  return out;
}

async function runFizh(set, relaxed) {
  spawn("fizh");
  const wasmUrl = relaxed ? "./fizh.relaxed.wasm" : "./fizh.baseline.wasm";
  const info = await call("fizh", "load", {
    wasmUrl,
    models: set.models.map((m) => ({ pair: m.pair, url: `./models/${m.file}` })),
  });
  const cases = await benchAll("fizh", casesFor(set));
  return {
    engine: relaxed ? "fizh-relaxed" : "fizh-baseline",
    wasm_bytes: info.wasm_bytes,
    cold_start_ms: round(info.cold_start_ms),
    arena_bytes: info.arena_bytes,
    peak_heap_bytes: Math.max(info.heap_bytes, ...cases.map((c) => c.heap_bytes || 0)),
    models: info.models,
    cases: cases.map(clean),
  };
}

async function runBergamot(set) {
  const raw = set.bergamot;
  if (!raw) return { engine: "bergamot", skipped: "no raw bundle published for this set" };
  spawn("berg");
  const info = await call("berg", "load", {
    engineJs: "./bergamot/bergamot-translator-worker.js",
    engineWasm: "./bergamot/bergamot-translator-worker.wasm",
    models: [{
      pair: raw.pair,
      model: `./bergamot/${raw.model}`,
      vocab: `./bergamot/${raw.vocab}`,
      lex: `./bergamot/${raw.lex}`,
    }],
  });
  // Only the direct cases: a bergamot pivot is two TranslationModels and a
  // different call shape, so pivot rows would not be comparing like with like.
  const cases = await benchAll("berg", casesFor(set).filter((c) => !c.name.includes("pivot")));
  return {
    engine: "bergamot",
    wasm_bytes: info.engine_bytes,
    instantiate_ms: round(info.instantiate_ms),
    model_build_ms: round(info.model_build_ms),
    cold_start_ms: round(info.cold_start_ms),
    peak_heap_bytes: Math.max(info.heap_bytes || 0, ...cases.map((c) => c.heap_bytes || 0)),
    models: info.models,
    cases: cases.map(clean),
  };
}

const clean = (c) => ({
  name: c.name,
  runs: c.runs,
  sentences: c.sentences ?? null,
  min_ms: round(c.min_ms),
  p50_ms: round(c.p50_ms),
  p90_ms: round(c.p90_ms),
  p99_ms: round(c.p99_ms),
  max_ms: round(c.max_ms),
  burst_p50_ms: round(c.burst_p50_ms),
  steady_p50_ms: round(c.steady_p50_ms),
  per_sentence_ms: round(c.per_sentence_ms),
  wall_ms: round(c.wall_ms),
  error: c.error,
});

async function run() {
  $("run").disabled = true;
  const set = manifest.sets.find((s) => s.id === selected);
  const started = new Date().toISOString();
  const t0 = performance.now();
  const wantBerg = $("with-bergamot").checked;
  const relaxedAvailable = await detectRelaxed();

  const engines = [];
  const failures = [];
  const attempt = async (label, fn) => {
    try { engines.push(await fn()); }
    catch (err) { failures.push({ engine: label, error: err.message, fatal: !!err.fatal }); }
  };

  if (relaxedAvailable) await attempt("fizh-relaxed", () => runFizh(set, true));
  await attempt("fizh-baseline", () => runFizh(set, false));
  if (wantBerg) await attempt("bergamot", () => runBergamot(set));

  const result = {
    started,
    duration_s: round((performance.now() - t0) / 1000),
    build: manifest.build,
    set: set.id,
    corpus: {
      short_words: 12, long_words: 120, paragraph_sentences: PARA_SENTENCES,
      warmup: WARMUP, runs: RUNS,
    },
    relaxed_simd_supported: relaxedAvailable,
    engines,
    failures,
    conditions: conditions(),
    device: device(),
  };
  render(result);
  $("run").disabled = false;
  $("bar").style.width = "0%";
  $("status").textContent = failures.length
    ? `done, ${failures.length} engine(s) failed — reported below`
    : "done";
}

/** Thermal state is not exposed to the web. Duration and whether the page
 *  stayed visible are, and they are what a reader needs to judge a number
 *  whose variance is the CPU scheduler. */
function conditions() {
  return {
    page_visible_throughout: document.visibilityState === "visible" && !sawHidden,
    reduced_motion: matchMedia("(prefers-reduced-motion: reduce)").matches,
    note: "thermal state and CPU affinity are not observable from the web; "
        + "min/p50 spread on a big.LITTLE device is the scheduler, not the runtime",
  };
}
let sawHidden = false;
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") sawHidden = true;
});

function device() {
  const d = {
    user_agent: navigator.userAgent,
    platform: navigator.platform,
    hardware_concurrency: navigator.hardwareConcurrency ?? null,
    device_memory_gb: navigator.deviceMemory ?? null,
    language: navigator.language,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
  };
  if (performance.memory) {
    d.js_heap_limit_bytes = performance.memory.jsHeapSizeLimit;
    d.js_heap_used_bytes = performance.memory.usedJSHeapSize;
  }
  return d;
}

// ---- rendering ------------------------------------------------------------

function table(el, head, rows) {
  el.innerHTML = "";
  const h = el.insertRow();
  for (const t of head) {
    const th = document.createElement("th");
    th.textContent = t;
    h.appendChild(th);
  }
  for (const r of rows) {
    const tr = el.insertRow();
    if (r.cls) tr.className = r.cls;
    for (const c of r.cells) tr.insertCell().textContent = c;
  }
}

const ms = (x) => (x === null || x === undefined ? "—" : `${x} ms`);

function render(r) {
  $("results-section").hidden = false;
  const find = (n) => r.engines.find((e) => e.engine === n);
  const fr = find("fizh-relaxed"), fb = find("fizh-baseline"), bg = find("bergamot");

  // --- Table 1: I1, three engines on one phone ---
  const names = [...new Set(r.engines.flatMap((e) => (e.cases || []).map((c) => c.name)))];
  const cell = (e, n, k) => {
    const c = (e?.cases || []).find((x) => x.name === n);
    return c ? ms(c[k]) : "—";
  };
  const rows = [{
    cells: ["cold start", ms(fr?.cold_start_ms), ms(fb?.cold_start_ms), ms(bg?.cold_start_ms)],
  }, {
    // `WebAssembly.Memory.buffer.byteLength` at its maximum during the run —
    // *linear memory demanded*, for both engines, measured the same way. It is
    // not RSS: growth commits pages but the OS need not keep them resident, and
    // neither engine reserves up front (bergamot's initial is 16 MiB against a
    // 2 GiB maximum). Compare it as "how much address space each engine asked
    // for", not as "how much RAM it used".
    cells: ["linear memory demanded",
      fr?.peak_heap_bytes ? `${(fr.peak_heap_bytes / 2 ** 20).toFixed(0)} MiB` : "—",
      fb?.peak_heap_bytes ? `${(fb.peak_heap_bytes / 2 ** 20).toFixed(0)} MiB` : "—",
      bg?.peak_heap_bytes ? `${(bg.peak_heap_bytes / 2 ** 20).toFixed(0)} MiB` : "—"],
  }, {
    cells: ["engine size",
      fr ? `${(fr.wasm_bytes / 1024).toFixed(0)} KiB` : "—",
      fb ? `${(fb.wasm_bytes / 1024).toFixed(0)} KiB` : "—",
      bg ? `${(bg.wasm_bytes / 2 ** 20).toFixed(2)} MiB` : "—"],
  }];
  for (const n of names) {
    rows.push({ cells: [`${n} — p50`, cell(fr, n, "p50_ms"), cell(fb, n, "p50_ms"), cell(bg, n, "p50_ms")] });
    rows.push({ cells: [`${n} — steady p50`, cell(fr, n, "steady_p50_ms"), cell(fb, n, "steady_p50_ms"), cell(bg, n, "steady_p50_ms")] });
  }
  table($("t-engines"), ["metric", "fizh relaxed", "fizh baseline", "bergamot"], rows);

  // --- Table 2: distribution, and what is honestly a p99 ---
  const drows = [];
  for (const e of r.engines) {
    for (const c of e.cases || []) {
      drows.push({
        cells: [`${e.engine} · ${c.name}`, String(c.runs), ms(c.min_ms), ms(c.burst_p50_ms),
                ms(c.steady_p50_ms), ms(c.p90_ms),
                c.p99_ms === null ? `max ${c.max_ms}` : `p99 ${c.p99_ms}`,
                c.per_sentence_ms ? `${c.per_sentence_ms}/sent` : "—"],
      });
    }
  }
  table($("t-dist"),
    ["engine · case", "runs", "min", "burst p50", "steady p50", "p90", "tail", "per sentence"],
    drows);

  const mem = r.engines.filter((e) => e.peak_heap_bytes).map((e) =>
    `${e.engine} peak ${(e.peak_heap_bytes / 2 ** 20).toFixed(1)} MiB` +
    (e.arena_bytes ? ` (arena ${(e.arena_bytes / 2 ** 20).toFixed(1)} MiB)` : ""));
  $("mem").textContent = mem.join(" · ");
  $("fails").textContent = r.failures.length
    ? r.failures.map((f) => `${f.engine}: ${f.error}`).join(" · ") : "";
  $("json").value = JSON.stringify(r, null, 2);
}

// ---- free-text box --------------------------------------------------------

async function go() {
  const [from, to] = $("dir").value.split(">");
  $("out").textContent = "…";
  try {
    const r = await call("fizh", "translate", { text: $("in").value, from, to });
    $("out").textContent = r.error ? r.error : `${r.text}\n\n(${round(r.ms)} ms)`;
  } catch (err) {
    $("out").textContent = `worker died: ${err.message}`;
  }
}

// ---- boot -----------------------------------------------------------------

async function main() {
  manifest = await (await fetch("./manifest.json")).json();
  const b = manifest.build;
  $("build").innerHTML =
    `<dt>commit</dt><dd><code>${b.sha}</code></dd>` +
    `<dt>built</dt><dd>${b.built}</dd><dt>zig</dt><dd>${b.zig}</dd>`;
  if (b.repo) $("repo").href = b.repo;
  $("probe").textContent = (await detectRelaxed())
    ? "supported — both builds will be measured"
    : "not supported — baseline only";

  const box = $("models");
  for (const s of manifest.sets) {
    const size = s.models.reduce((a, m) => a + m.bytes, 0) / 2 ** 20;
    box.insertAdjacentHTML("beforeend",
      `<label class="set"><input type="radio" name="set" value="${s.id}">
       <span><strong>${s.label}</strong> — ${size.toFixed(0)} MiB,
       ${s.models.map((m) => `${m.pair} (d=${m.d_model})`).join(" + ")}
       ${s.bergamot ? "" : "<em>· no bergamot bundle</em>"}</span></label>`);
  }
  box.addEventListener("change", (e) => {
    selected = e.target.value;
    $("run").disabled = false;
    const set = manifest.sets.find((x) => x.id === selected);
    $("dir").innerHTML = [`${set.from}>${set.to}`]
      .concat(set.pivot ? [`${set.from}>${set.pivot}`] : [])
      .map((d) => `<option value="${d}">${d.replace(">", " → ")}</option>`).join("");
    $("go").disabled = false;
  });

  $("run").onclick = run;
  $("go").onclick = go;
  $("copy").onclick = () => navigator.clipboard.writeText($("json").value);
}

main().catch((e) => { $("status").textContent = `page failed: ${e.message}`; });
