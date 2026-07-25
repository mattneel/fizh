// bench.js — main thread. Owns the UI and exactly one worker at a time.
//
// SPEC §10: the runtime lives in a dedicated worker. This file never
// instantiates wasm, so a trap kills the worker and lands here as a message.

const $ = (id) => document.getElementById(id);

// The SPEC §14 corpus, byte-for-byte what tools/bench.zig measures, so a number
// from a phone sits next to the desktop proxy and means the same thing.
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

const paragraph = (n) =>
  Array.from({ length: n }, (_, i) => CLAUSES[i % CLAUSES.length] + ".").join(" ");

const SHORT = message(12);
const LONG = message(120);
const PARA = paragraph(8);

let manifest = null;
let worker = null;
let nextId = 1;
const pending = new Map();
let selected = null;
let ready = false;

function spawn() {
  if (worker) worker.terminate();
  worker = new Worker("./worker.js", { type: "module" });
  worker.onmessage = (e) => {
    const { id, ok, progress, ...body } = e.data;
    if (progress) return onProgress(progress);
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    ok ? p.resolve(body) : p.reject(Object.assign(new Error(body.error), body));
  };
  // A worker that dies without replying is a trap or an OOM kill. Every
  // outstanding call has to hear about it, or the page just stops.
  worker.onerror = (e) => {
    const err = Object.assign(new Error(e.message || "worker died"), { fatal: true });
    for (const [, p] of pending) p.reject(err);
    pending.clear();
  };
  ready = false;
}

function call(op, args) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, op, args });
  });
}

function onProgress({ what, got, total }) {
  $("status").textContent = total
    ? `${what}: ${(got / 2 ** 20).toFixed(1)} / ${(total / 2 ** 20).toFixed(1)} MiB`
    : `${what}: ${(got / 2 ** 20).toFixed(1)} MiB`;
  $("bar").style.width = total ? `${(100 * got) / total}%` : "0%";
}

// ---- relaxed SIMD detection (SPEC §3, ADR 0006) ---------------------------

async function detectRelaxed() {
  try {
    const probe = await fetch("./fizh.probe.wasm");
    if (!probe.ok) return false;
    return WebAssembly.validate(await probe.arrayBuffer());
  } catch {
    return false;
  }
}

// ---- run ------------------------------------------------------------------

async function load(set) {
  spawn();
  const relaxed = !$("force-baseline").checked && (await detectRelaxed());
  const wasmUrl = relaxed ? "./fizh.relaxed.wasm" : "./fizh.baseline.wasm";
  $("probe").textContent = relaxed ? "supported — using relaxed build" : "using baseline build";

  const info = await call("load", {
    wasmUrl,
    models: set.models.map((m) => ({ pair: m.pair, url: `./models/${m.file}` })),
  });
  ready = true;
  return { ...info, relaxed, wasm: wasmUrl.replace("./", "") };
}

async function run() {
  $("run").disabled = true;
  $("go").disabled = true;
  const set = manifest.sets.find((s) => s.id === selected);
  const started = new Date().toISOString();

  let result;
  try {
    const info = await load(set);
    $("status").textContent = "warming up…";

    const cases = [
      { name: "12-token, direct", text: SHORT, from: set.from, to: set.to },
      { name: "120-token, direct", text: LONG, from: set.from, to: set.to },
      { name: "8-sentence paragraph", text: PARA, from: set.from, to: set.to },
    ];
    if (set.pivot) {
      cases.push({ name: "12-token, pivot", text: SHORT, from: set.from, to: set.pivot });
      cases.push({ name: "120-token, pivot", text: LONG, from: set.from, to: set.pivot });
    }

    $("status").textContent = "measuring…";
    const bench = await call("bench", { cases, runs: 30, warmup: 5 });

    result = {
      ok: true,
      started,
      build: manifest.build,
      set: set.id,
      runtime: { wasm: info.wasm, relaxed: info.relaxed, bytes: info.wasm_bytes },
      models: info.models,
      arena_bytes: info.arena_bytes,
      cold_start_ms: round(info.cold_start_ms),
      warm: bench.cases.map((c) => ({
        name: c.name,
        p50_ms: round(c.p50),
        p99_ms: round(c.p99),
        min_ms: round(c.min),
        runs: c.runs,
        error: c.error,
      })),
      peak_wasm_heap_bytes: bench.heap_bytes,
      device: device(),
    };
  } catch (err) {
    // An OOM on a two-slot pivot, or a trap, is a result. Report it with
    // everything known at the time rather than losing the run.
    result = {
      ok: false,
      started,
      build: manifest.build,
      set: set.id,
      failure: err.message,
      fatal: !!err.fatal,
      device: device(),
    };
    $("status").textContent = err.fatal
      ? `worker died: ${err.message} — this is a result, not a crash`
      : `failed: ${err.message}`;
  }

  render(result);
  $("run").disabled = false;
  $("go").disabled = !ready;
  $("bar").style.width = "0%";
  if (result.ok) $("status").textContent = "done";
}

const round = (x) => Math.round(x * 100) / 100;

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

function render(r) {
  $("results-section").hidden = false;
  const t = $("results");
  t.innerHTML = "";
  const head = t.insertRow();
  for (const h of ["metric", "measured", "§14 budget"]) {
    const th = document.createElement("th");
    th.textContent = h;
    head.appendChild(th);
  }

  const budgets = {
    "cold start": 10,
    "12-token, direct": 22,
    "120-token, direct": 200,
    "8-sentence paragraph": 100,
    "12-token, pivot": 44,
    "120-token, pivot": 400,
  };

  const add = (name, got, budget, unit = "ms") => {
    const row = t.insertRow();
    row.insertCell().textContent = name;
    const v = row.insertCell();
    v.textContent = got === null ? "—" : `${got} ${unit}`;
    const b = row.insertCell();
    b.textContent = budget ? `${budget} ${unit}` : "—";
    if (budget && got !== null && got > budget) row.className = "over";
  };

  if (!r.ok) {
    const row = t.insertRow();
    row.className = "over";
    const c = row.insertCell();
    c.colSpan = 3;
    c.textContent = `${r.fatal ? "worker died" : "failed"}: ${r.failure}`;
  } else {
    add("cold start (init+load+repack)", r.cold_start_ms, budgets["cold start"]);
    for (const c of r.warm) {
      if (c.error) {
        add(`warm p50, ${c.name}`, null, budgets[c.name]);
        continue;
      }
      add(`warm p50, ${c.name}`, c.p50_ms, budgets[c.name]);
      add(`warm p99, ${c.name}`, c.p99_ms, null);
    }
    $("mem").textContent =
      `peak wasm heap ${(r.peak_wasm_heap_bytes / 2 ** 20).toFixed(1)} MiB, ` +
      `arena ${(r.arena_bytes / 2 ** 20).toFixed(1)} MiB, ` +
      `runtime ${(r.runtime.bytes / 1024).toFixed(0)} KiB ${r.runtime.relaxed ? "(relaxed)" : "(baseline)"}`;
  }
  $("json").value = JSON.stringify(r, null, 2);
}

// ---- free-text box --------------------------------------------------------

async function go() {
  const set = manifest.sets.find((s) => s.id === selected);
  const [from, to] = $("dir").value.split(">");
  $("out").textContent = "…";
  try {
    const r = await call("translate", { text: $("in").value, from, to });
    $("out").textContent = r.error ? `${r.error}` : `${r.text}\n\n(${round(r.ms)} ms)`;
  } catch (err) {
    $("out").textContent = `worker died: ${err.message}`;
    spawn();
    ready = false;
    $("go").disabled = true;
  }
}

// ---- boot -----------------------------------------------------------------

async function main() {
  manifest = await (await fetch("./manifest.json")).json();

  const b = manifest.build;
  $("build").innerHTML =
    `<dt>commit</dt><dd><code>${b.sha}</code></dd>` +
    `<dt>built</dt><dd>${b.built}</dd>` +
    `<dt>zig</dt><dd>${b.zig}</dd>`;
  if (b.repo) $("repo").href = b.repo;

  $("probe").textContent = (await detectRelaxed())
    ? "supported"
    : "not supported — baseline only";

  const box = $("models");
  for (const s of manifest.sets) {
    const id = `set-${s.id}`;
    const size = s.models.reduce((a, m) => a + m.bytes, 0) / 2 ** 20;
    box.insertAdjacentHTML("beforeend",
      `<label class="set"><input type="radio" name="set" value="${s.id}" id="${id}">
       <span><strong>${s.label}</strong> — ${size.toFixed(0)} MiB,
       ${s.models.map((m) => `${m.pair} (d=${m.d_model})`).join(" + ")}</span></label>`);
  }
  box.addEventListener("change", (e) => {
    selected = e.target.value;
    $("run").disabled = false;
    const set = manifest.sets.find((x) => x.id === selected);
    $("dir").innerHTML = [`${set.from}>${set.to}`]
      .concat(set.pivot ? [`${set.from}>${set.pivot}`] : [])
      .map((d) => `<option value="${d}">${d.replace(">", " → ")}</option>`).join("");
  });

  $("run").onclick = run;
  $("go").onclick = go;
  $("copy").onclick = () => navigator.clipboard.writeText($("json").value);
  $("force-baseline").onchange = () => { ready = false; $("go").disabled = true; };
}

main().catch((e) => { $("status").textContent = `page failed: ${e.message}`; });
