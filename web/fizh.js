// fizh.js — the host side of SPEC §4 and §9, for the browser.
//
// The same four steps `tools/host.mjs` documents: feature-detect with the
// probe, ask for the arena size, grow memory, initialize, load a direction per
// slot, translate. This module is imported by the worker and by nothing else —
// SPEC §10 puts the runtime in a dedicated worker, so the main thread never
// holds an instance.

const PAGE = 65536;

// SPEC §9. Anything not in this list is not part of the ABI.
export const EXPORTS = [
  "fizh_abi_version",
  "fizh_arena_bytes",
  "fizh_init",
  "fizh_model_load",
  "fizh_translate",
  "fizh_can_translate",
  "fizh_status_str",
];

export // abi.Lang: up to four lowercase ASCII bytes, big-endian, zero-padded left.
const lang = (s) => [...s].reduce((a, c) => (a << 8) | c.charCodeAt(0), 0) >>> 0;

/** SPEC §6's header, so the arena is sized from the artifact rather than from
 *  a guess. Firefox ships d_model 256 and 384; guessing rejects one of them. */
export function hparams(blob) {
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
export function config(overrides = {}) {
  const c = {
    abi_version: 1,
    max_models: 1,
    max_model_bytes: 36 << 20,
    max_src_bytes: 4096,
    max_src_tokens: 256,
    max_tgt_tokens: 768,
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

export class Fizh {
  constructor(instance) {
    this.e = instance.exports;
    this.memory = this.e.memory;
    // The host owns everything past the module's initial memory. The runtime
    // only ever touches the arena it was handed.
    this.top = this.memory.buffer.byteLength;
    this.peak = this.top;
    for (const name of EXPORTS) {
      if (typeof this.e[name] !== "function") throw new Error(`missing export ${name}`);
    }
  }

  /** Reserves `n` bytes of host scratch, 64-byte aligned (SPEC §4).
   *  Throws rather than returning garbage when the browser refuses to grow —
   *  a two-slot pivot of 33 MB artifacts is exactly where that happens, and it
   *  is a result, not a crash. */
  reserve(n) {
    const at = (this.top + 63) & ~63;
    const need = at + n;
    const have = this.memory.buffer.byteLength;
    if (need > have) {
      const pages = Math.ceil((need - have) / PAGE);
      try {
        if (this.memory.grow(pages) < 0) throw new Error("grow refused");
      } catch (e) {
        throw new Error(`out of memory growing to ${(need / 2 ** 20).toFixed(1)} MiB: ${e.message}`);
      }
    }
    this.top = need;
    this.peak = Math.max(this.peak, this.memory.buffer.byteLength);
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

/** Instantiate, size the arena for every blob, load each into its own slot. */
export async function boot(wasmBytes, blobs) {
  const { instance } = await WebAssembly.instantiate(wasmBytes, {});
  const f = new Fizh(instance);
  if (f.e.fizh_abi_version() !== 1) throw new Error(`ABI version ${f.e.fizh_abi_version()}`);

  const hps = blobs.map(hparams);
  const maxPos = Math.min(...hps.map((h) => h.max_pos));
  const biggest = Math.max(...blobs.map((b) => b.length));
  const cfg = config({
    max_models: blobs.length,
    max_model_bytes: (((biggest * 1.25) | 0) + 63) & ~63,
    max_src_tokens: Math.min(256, maxPos),
    max_tgt_tokens: Math.min(768, maxPos),
    max_d_model: Math.max(...hps.map((h) => h.d_model)),
    max_ffn_dim: Math.max(...hps.map((h) => h.ffn_dim)),
    max_enc_layers: Math.max(...hps.map((h) => h.n_enc_layers)),
    max_dec_layers: Math.max(...hps.map((h) => h.n_dec_layers)),
    max_heads: Math.max(...hps.map((h) => h.n_heads)),
    max_vocab: Math.max(...hps.map((h) => h.vocab_size)),
  });

  const cfgPtr = f.write(cfg);
  const arenaBytes = f.e.fizh_arena_bytes(cfgPtr, 64);
  if (arenaBytes === 0) throw new Error("config rejected");
  const base = f.reserve(arenaBytes);
  const handle = f.check(f.e.fizh_init(base, arenaBytes, cfgPtr, 64), "init");

  // One staging buffer, reused for every slot.
  //
  // `fizh_model_load` repacks the blob into the slot, so between the host
  // writing it and that call returning, the same weights exist twice. Writing
  // each blob at a fresh offset makes the transient peak `arena + sum(blobs)`;
  // reusing one buffer makes it `arena + max(blob)`. On the measured Android
  // run that delta was 21.4 MB for a single 20 MB model, and a two-slot pivot
  // of 33 MB artifacts would have carried 66 MB of it. SPEC §4.3.
  //
  // Freeing it afterwards is not possible: the host allocator here is a bump
  // pointer into wasm memory, and wasm memory never shrinks. The buffer stays
  // reserved and is reused by nothing else, which is why it is placed last.
  const staging = f.reserve(Math.max(...blobs.map((b) => b.length)));
  for (const [slot, blob] of blobs.entries()) {
    f.bytes().set(blob, staging);
    f.check(f.e.fizh_model_load(handle, slot, staging, blob.length), `load slot ${slot}`);
  }

  // Source and destination are reserved once. A benchmark that reserves per
  // call grows the heap sixty times and measures the growth.
  const srcLen = 4096;
  const src = f.reserve(srcLen);
  // Output is not bounded by input (SPEC §4.2), so it gets the 2x the arena
  // gives io_dst.
  const outLen = 8192;
  const out = f.reserve(outLen);
  return { f, handle, src, srcLen, out, outLen, arenaBytes };
}

export function translate(ctx, text, from, to) {
  const { f, handle, src, srcLen, out, outLen } = ctx;
  const bytes = new TextEncoder().encode(text);
  if (bytes.length > srcLen) return { error: `source exceeds ${srcLen} bytes`, code: -11 };
  f.bytes().set(bytes, src);
  const n = f.e.fizh_translate(handle, src, bytes.length, lang(from), lang(to), out, outLen);
  if (n < 0) return { error: f.status(n), code: n };
  return { text: new TextDecoder().decode(f.bytes().subarray(out, out + n)) };
}
