// host.mjs — the host side of SPEC §4 and §9, in about a hundred lines.
//
// It is a test, and it is documentation: this is exactly what the PWA's Web
// Worker has to do: load the artifact, ask for
// the arena size, grow memory, initialize, load a direction per slot, translate.
//
//   node tools/host.mjs zig-out/wasm zig-out/selftest.fzm
//
// SPEC §10 says the runtime must live in a dedicated Web Worker and be
// respawned on trap. Nothing here does that, because nothing here is a worker —
// but note that every failure below arrives as a negative status, and a trap
// would arrive as a thrown RuntimeError. Those are the two cases a real host
// has to tell apart.

import { readFileSync } from "node:fs";
import { join } from "node:path";

const PAGE = 65536;

// SPEC §9. Anything not in this list is not part of the ABI.
const EXPORTS = [
  "fizh_abi_version",
  "fizh_arena_bytes",
  "fizh_init",
  "fizh_model_load",
  "fizh_translate",
  "fizh_can_translate",
  "fizh_status_str",
];

// abi.Lang: up to four lowercase ASCII bytes, big-endian, zero-padded left.
const lang = (s) => [...s].reduce((a, c) => (a << 8) | c.charCodeAt(0), 0) >>> 0;

/** One artifact. The relaxed variant and its probe are gone (ADR 0025). */
function pickArtifact(dir) {
  return { path: join(dir, "fizh.wasm") };
}

class Fizh {
  constructor(instance) {
    this.e = instance.exports;
    this.memory = this.e.memory;
    // The host owns everything past the module's initial memory. The runtime
    // only ever touches the arena it was handed.
    this.top = this.memory.buffer.byteLength;
    for (const name of EXPORTS) {
      if (typeof this.e[name] !== "function") throw new Error(`missing export ${name}`);
    }
  }

  /** Reserves `n` bytes of host scratch, 64-byte aligned (SPEC §4). */
  reserve(n) {
    const at = (this.top + 63) & ~63;
    const need = at + n;
    const have = this.memory.buffer.byteLength;
    if (need > have) this.memory.grow(Math.ceil((need - have) / PAGE));
    this.top = need;
    return at;
  }

  bytes() {
    // Re-acquired after every grow: the old view is detached.
    return new Uint8Array(this.memory.buffer);
  }

  write(data) {
    const at = this.reserve(data.length);
    this.bytes().set(data, at);
    return at;
  }

  status(code) {
    const at = this.e.fizh_status_str(code);
    const mem = this.bytes();
    let end = at;
    while (mem[end] !== 0) end += 1;
    return new TextDecoder().decode(mem.subarray(at, end));
  }

  check(code, what) {
    if (code < 0) throw new Error(`${what}: ${this.status(code)} (${code})`);
    return code;
  }
}

/** Reads SPEC §6's header so the host can size the arena for whatever it was
 *  handed, rather than hard-coding one model's shape. */
function hparams(blob) {
  const v = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  if (String.fromCharCode(...blob.subarray(0, 4)) !== "FIZH") throw new Error("not a .fzm");
  return {
    d_model: v.getUint16(16, true),
    ffn_dim: v.getUint16(18, true),
    n_enc_layers: blob[20],
    n_dec_layers: blob[21],
    n_heads: blob[22],
    vocab_size: v.getUint32(24, true),
    max_pos: v.getUint16(28, true),
  };
}

/** SPEC §9. 64 bytes, little-endian, fourteen limits and three reserved words. */
function config(overrides = {}) {
  const c = {
    abi_version: 1,
    max_models: 1,
    max_model_bytes: 22 << 20,
    max_src_bytes: 4096,
    max_src_tokens: 256,
    max_tgt_tokens: 384,
    max_shortlist: 2048,
    max_d_model: 256,
    max_ffn_dim: 1536,
    max_enc_layers: 6,
    max_dec_layers: 2,
    max_heads: 8,
    max_vocab: 32768,
    ...overrides,
  };
  const buf = new Uint8Array(64);
  const view = new DataView(buf.buffer);
  Object.values(c).forEach((v, i) => view.setUint32(i * 4, v, true));
  return buf;
}

async function main() {
  const dir = process.argv[2] ?? "zig-out/wasm";
  const model = process.argv[3] ?? "zig-out/selftest.fzm";

  const { path } = pickArtifact(dir);
  console.log(`artifact: ${path}`);

  const { instance } = await WebAssembly.instantiate(readFileSync(path), {});
  const f = new Fizh(instance);

  const version = f.e.fizh_abi_version();
  if (version !== 1) throw new Error(`ABI version ${version}, expected 1`);

  const blob = readFileSync(model);
  const hp = hparams(blob);
  // The positional table is generated up to max_pos, so the token limits may
  // not exceed it.
  const steps = Math.min(384, hp.max_pos);

  // SPEC §4, the four steps.
  const cfgPtr = f.write(config({
    max_model_bytes: Math.max(1 << 20, (blob.length + (1 << 20)) & ~0xffff),
    max_src_tokens: Math.min(256, steps),
    max_tgt_tokens: steps,
    max_d_model: hp.d_model,
    max_ffn_dim: hp.ffn_dim,
    max_enc_layers: hp.n_enc_layers,
    max_dec_layers: hp.n_dec_layers,
    max_heads: hp.n_heads,
    max_vocab: hp.vocab_size,
    max_shortlist: Math.min(2048, hp.vocab_size),
  }));
  const arenaBytes = f.e.fizh_arena_bytes(cfgPtr, 64);
  if (arenaBytes === 0) throw new Error("config rejected");
  console.log(`arena: ${arenaBytes} bytes`);

  const arenaPtr = f.reserve(arenaBytes);
  const handle = f.check(f.e.fizh_init(arenaPtr, arenaBytes, cfgPtr, 64), "fizh_init");

  const blobPtr = f.write(blob);
  f.check(f.e.fizh_model_load(handle, 0, blobPtr, blob.length), "fizh_model_load");
  console.log(`loaded: ${model} (${blob.length} bytes)`);

  const es = lang("es"), en = lang("en"), de = lang("de");
  console.log(`can_translate es->en: ${f.e.fizh_can_translate(handle, es, en)} (1 = direct)`);
  console.log(`can_translate es->de: ${f.e.fizh_can_translate(handle, es, de)} (0 = none)`);

  console.log(`model: d=${hp.d_model} ffn=${hp.ffn_dim} enc=${hp.n_enc_layers} ` +
              `dec=${hp.n_dec_layers} heads=${hp.n_heads} vocab=${hp.vocab_size}`);
  const src = new TextEncoder().encode(process.argv[4] ?? "El gato está en la mesa.");
  const srcPtr = f.write(src);
  const outPtr = f.reserve(4096);
  const n = f.check(
    f.e.fizh_translate(handle, srcPtr, src.length, es, en, outPtr, 4096),
    "fizh_translate",
  );
  const text = new TextDecoder().decode(f.bytes().subarray(outPtr, outPtr + n));
  console.log(`translate: ${JSON.stringify(text)} (${n} bytes)`);

  // Timing through the real wasm, which is the thing that ships.
  const t0 = performance.now();
  const runs = 20;
  for (let i = 0; i < runs; i++) f.e.fizh_translate(handle, srcPtr, src.length, es, en, outPtr, 4096);
  console.log(`warm: ${((performance.now() - t0) / runs).toFixed(1)} ms/translation (wasm)`);

  // I9: same artifact, same build, same bytes.
  const again = f.check(
    f.e.fizh_translate(handle, srcPtr, src.length, es, en, outPtr, 4096),
    "fizh_translate",
  );
  const text2 = new TextDecoder().decode(f.bytes().subarray(outPtr, outPtr + again));
  if (text !== text2) throw new Error("not deterministic across calls");

  // SPEC §11: hostile input comes back as a status, not a trap.
  const bad = new Uint8Array([0xff, 0xfe]);
  const badPtr = f.write(bad);
  const badStatus = f.e.fizh_translate(handle, badPtr, bad.length, es, en, outPtr, 4096);
  if (badStatus >= 0) throw new Error("invalid UTF-8 was accepted");
  console.log(`invalid UTF-8: ${f.status(badStatus)}`);

  console.log("host: ok");
}

main().catch((e) => {
  console.error(`host: FAILED — ${e.message}`);
  process.exit(1);
});
