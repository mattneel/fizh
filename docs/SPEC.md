# fizh — SPEC

Babel fish in Zig. On-device neural machine translation for a direct messaging app.

Text in, language code in, translated text out. Nothing else.

Zig 0.16, `wasm32-freestanding`, SIMD128. Status: **pre-M0.**

### Conventions

| Thing | Value |
|---|---|
| ABI prefix | `fizh_` |
| Model artifact | `.fzm`, magic `FIZH` |
| Wasm artifacts | `fizh.{baseline,relaxed}.wasm` |
| Language codes | `u16`, two ASCII bytes packed big-endian: `'e'<<8 \| 's'` = `es` |

Lowercase everywhere. Not an acronym.

---

## 1. Scope

**In:** artifact loading, tokenization, quantized seq2seq inference, greedy decode, route resolution and English pivoting, a C ABI.

**Out:** training, quantization research, language identification (host's job — CLD2 or the browser's detector), the messaging app, transport, crypto, WebGPU (v2).

The host never sees a tensor. The runtime never sees a socket.

---

## 2. Design invariants

Decided. Changing one needs an ADR in `docs/adr/`.

| # | Invariant | Rationale |
|---|---|---|
| I1 | SIMD128 CPU backend, hand-written, architecture-neutral | Bergamot's own engine is intgemm-based and x86-tuned; Mozilla flagged ARM as a known weak spot. Phones are ARM. This is the reason fizh exists. |
| I2 | A scalar `f32` reference implementation lives in-repo, in Zig, forever | In-process differential testing with no FFI. It is the only thing that makes hand-written SIMD debuggable. |
| I3 | Zero heap allocation. `std.mem.Allocator` does not appear in `src/` | Every shape is known at init. |
| I4 | int8 only. No int4, no mixed precision | At ~17M parameters the decoder is ~3 MB; there is no bandwidth problem to solve. |
| I5 | Weights clamp to `[-127, 127]` | One line in the converter, keeps the artifact valid for a future WGSL backend without requantizing. |
| I6 | Shortlisted output projection, always | Bergamot ships the shortlists. Full-vocab projection is the single largest per-token cost. |
| I7 | Greedy decode, beam width 1 | What Bergamot ships. Removes a KV dimension and a whole scratch region. |
| I8 | Pivoting is internal | The host asks for `es → de` and gets German. Model topology does not leak into the PWA. |
| I9 | Bit-exact determinism for a given artifact + build | Golden vectors, reproducible bugs. |
| I10 | Runs in a dedicated Web Worker | Invariant violations trap; a trap must kill a worker, not the app. |

---

## 3. Target and build

`wasm32-freestanding`. No libc, no WASI, no `std.Io`. The project performs no filesystem, network, or timer I/O, which insulates it almost entirely from 0.16's `std.Io` migration. **Adding an import to the module requires an ADR.**

Note 0.16's tightened vector-index restrictions: `src/kernel/` uses `@Vector`, `@shuffle`, `@select`, `@reduce` with comptime-known indices only.

| Artifact | CPU features |
|---|---|
| `fizh.baseline.wasm` | `+simd128,+bulk_memory,+sign_ext,+nontrapping_fptoint,+mutable_globals` |
| `fizh.relaxed.wasm` | above, `+relaxed_simd` |

Host picks via probe-module feature detection.

**Ship build is `ReleaseSafe`** — assertions stay on. Measure the `ReleaseFast` delta at M7; the number goes in an ADR either way.

`@setFloatMode(.optimized)` is **forbidden** in `src/kernel/` and `src/graph/`. Vectorization comes from explicit `@Vector` ops, not compiler reassociation freedom. Fixed reduction order is what makes I9 hold.

| Budget | Target |
|---|---|
| `fizh.baseline.wasm`, gzipped | ≤ 200 KB |
| Module imports, ship build | 0 |
| Exported symbols | ≤ 10 |

For scale: Bergamot's wasm engine is ~4.94 MB because it carries most of Marian. fizh implements one architecture.

---

## 4. Memory

Two-phase init. The runtime never grows memory.

```
1. host: n = fizh_arena_bytes(cfg)
2. host: memory.grow to cover n + all model blobs
3. host: h = fizh_init(base, n, cfg)
4. host: fizh_model_load(h, slot, blob, len)   -- once per direction
```

One arena, carved at init, never resized, no sub-allocators.

### 4.1 Key property: scratch is shared across a pivot

A pivot is two sequential passes. Pass one completes before pass two starts, so **scratch is sized by the max over loaded models, not the sum.** Only the weight slots are additive.

### 4.2 Regions

| Region | Size |
|---|---|
| `weights[slot]` | from artifact header, one per loaded direction |
| `xattn_kv` | `2 · max(n_dec_layers) · max_src_tokens · max(d_model) · 4` |
| `self_kv` | `2 · max(n_dec_layers) · max_tgt_tokens · max(d_model) · 4` |
| `enc_states` | `max_src_tokens · max(d_model) · 4` |
| `act_a`, `act_b` | `max_src_tokens · max(ffn_dim) · 4` each |
| `qact` | same element count as `act_a`, 1 byte each |
| `shortlist_rows` | `max_shortlist · max(d_model) · 1` |
| `shortlist_ids`, `logits` | `max_shortlist · 4` each |
| `tok_lattice` | `max_src_bytes · sizeof(LatticeNode)` |
| `io_src`, `io_pivot`, `io_dst` | `max_src_bytes` each |

### 4.3 Worked example

Bergamot-student hyperparameters (`d_model=256`, `ffn=1536`, `n_enc=6`, `n_dec=2`, `vocab=32k`), `max_src_tokens=256`, `max_tgt_tokens=384`, `max_shortlist=2048`, `max_src_bytes=4096`. Verify per artifact at M2.

| | |
|---|---|
| Shared scratch | ≈ 6.6 MB |
| Weights, per direction (int8, ~17M params) | ≈ 17 MB |
| Two directions resident (pivot pair) | ≈ 40 MB total |

---

## 5. Repository layout

```
build.zig
build.zig.zon
SPEC.md
docs/adr/
src/
  root.zig            ABI exports, nothing else
  abi.zig             status codes, C-layout structs
  arena.zig           region carving, alignment asserts
  route.zig           direct / pivot resolution (§10)
  model/
    format.zig        artifact parsing — THE validation boundary (§11)
    repack.zig        canonical -> SIMD128 layout
  tok/
    unigram.zig       Viterbi over the lattice, iterative
    trie.zig
  kernel/
    ref/              scalar f32 oracle (I2)
    simd128/
    relaxed/
  graph/
    encoder.zig
    decoder.zig
    attention.zig
    shortlist.zig
test/{golden,fuzz,perf}/
tools/
  convert.py          Marian intgemm -> .fzm
  reference.py        the oracle that generates test/golden/
  eval/
```

---

## 6. Artifact format (`.fzm`)

Little-endian throughout. wasm is LE; no byte swapping exists anywhere.

```
magic         [4]u8   "FIZH"
version       u32     = 1
src_lang      u16     packed ASCII pair
tgt_lang      u16
hparams       [48]u8  packed
tensor_count  u32
tensors       [tensor_count]TensorDesc
payload       [...]   64-byte-aligned tensor bytes
```

```zig
const TensorDesc = extern struct {
    name_hash: u64,   // comptime FNV-1a; no string compare at load
    dtype: u8,        // 0=f32 1=i8
    rank: u8,
    _pad: u16,
    dims: [4]u32,
    offset: u64,
    nbytes: u64,
    scale_offset: u64,
};
```

- All offsets 64-byte aligned.
- Canonical matrix layout is row-major `[N][K]`, K contiguous. The backend repacks at load.
- One direction per file. Vocabulary and shortlist ship as tensors with reserved name hashes — one download per direction.
- Lookup by `name_hash` against a comptime perfect hash. A missing required tensor is a load *error*, not an assertion.

`tools/convert.py` reads Marian `.intgemm.alphas.bin` plus its `.spm` vocab and `lex.*.s2t.bin` shortlist and emits one `.fzm`. **M2 task: confirm whether Marian's on-disk int8 layout is already register-tiled for intgemm and unshuffle to canonical if so.**

---

## 7. Quantization contract

Artifact ABI. Specified before any kernel is written.

- **Weights:** symmetric, no zero point, per-output-channel `f32` scale, values in `[-127, 127]`. `assert(w != -128)` is a real assertion in the load path — see I5.
- **Activations:** dynamic per-row absmax → `i8` in `[-127, 127]`, `f32` scale, computed at runtime.
- **Accumulate in `i32`.** Never `f32` accumulation of integer products.
- **Dequant** is one multiply at the end of the reduction: `@floatFromInt(acc) * (act_scale * w_scale)`.

---

## 8. Kernels

Every op gets a `ref/` implementation first (I2); `simd128/` and `relaxed/` are optimizations validated against it.

| Op | Regime |
|---|---|
| `embed_gather` | random access, dequant on gather |
| `layer_norm` | row-wise, fixed reduction order |
| `qgemm_i8` | M×K×N, M = src_len — encoder workhorse |
| `qgemv_i8` | 1×K×N — decoder workhorse |
| `attn_prefill` | batched, bidirectional |
| `attn_decode` | batch 1, append to `self_kv`, attend over prefix |
| `xattn_decode` | batch 1, reads precomputed `xattn_kv` |
| `softmax` | row-wise, max-subtract, fixed order |
| `activation` | elementwise |
| `residual_add` | elementwise |
| `shortlist_build` | union of per-source-token candidates + top-N frequent |
| `logits_project` | 1×K×S, S = shortlist size |
| `argmax` | over S |

---

## 9. Public ABI

`extern "C"`, exported from `src/root.zig` only. No `usize` crosses the boundary.

```zig
export fn fizh_abi_version() u32;
export fn fizh_arena_bytes(cfg: [*]const u8, cfg_len: u32) u32;
export fn fizh_init(base: [*]u8, len: u32, cfg: [*]const u8, cfg_len: u32) i32;
export fn fizh_model_load(h: i32, slot: u32, blob: [*]const u8, len: u32) i32;

export fn fizh_translate(
    h: i32,
    src: [*]const u8, src_len: u32,
    src_lang: u16, tgt_lang: u16,
    out: [*]u8, out_cap: u32,
) i32;   // bytes written, or negative status

export fn fizh_can_translate(h: i32, src_lang: u16, tgt_lang: u16) i32;  // 0 no, 1 direct, 2 pivot

export fn fizh_status_str(code: i32) [*:0]const u8;
```

One call does the work. Pivoting, tokenization, and detokenization are internal. `fizh_can_translate` lets the host grey out a language before the user picks it.

---

## 10. Routing and pivot

Bergamot models are single-direction and one-to-one (`en⟶es`, not `es⟷en`), and non-English pairs have no direct model. Firefox resolves these by pivoting through English, at most once. fizh does the same, internally.

- `route.zig` holds a static table of loaded slots keyed by `(src_lang, tgt_lang)`.
- Direct hit → one pass.
- Miss → look for `(src, en)` and `(en, tgt)`. Found → two passes through `io_pivot`.
- No path → `STATUS_NO_ROUTE`.
- **Never pivot more than once.** Asserted, not merely intended.

The pivot boundary is UTF-8 text, not tokens — the two models have different vocabularies, so pass one detokenizes fully and pass two tokenizes fresh.

Two consequences the host should know: a pivot pair needs both models resident, and it roughly doubles latency. §13 budgets them separately.

---

## 11. Error model

**Validation errors** — anything derived from bytes the runtime did not produce: artifact blobs, source text, language codes, host lengths. Negative status codes. Must never trap, never read out of bounds. Only `model/format.zig` and `tok/` handle untrusted input.

**Invariant violations** — shape mismatches, alignment failures, out-of-range quantized values, non-finite scales, exhausted regions, double pivot. These `@panic` and trap. Correct: the instance is undefined and must die.

The boundary is the design. Once `format.zig` has validated an artifact, everything downstream asserts freely, because the invariants were established once, in one place, by code that returns errors instead of asserting.

Because traps are unrecoverable within an instance, I10 is a hard requirement on the host: dedicated Web Worker, respawn on trap.

---

## 12. Tiger Style rules

Enforced in CI where mechanically checkable.

1. **No allocator.** `std.mem.Allocator` must not appear in `src/`. CI greps.
2. **No recursion.** Anywhere. Viterbi is iterative by construction.
3. **All loops bounded.** No `while (true)`. Generation is bounded by *both* `max_tgt_tokens` and `max_length_factor · src_len`, and asserts both.
4. **Two assertions minimum** per non-trivial function: one on arguments, one on results.
5. **Assert at every kernel entry:** shapes, strides, 64-byte alignment, finite scales, quantized values in range.
6. **Negative-space assertions.** `assert(w != -128)`, `assert(!isNan(scale))`, `assert(pivot_depth == 0)`.
7. **Explicitly sized types.** `u32`, `i32`, `f32`. `usize` never crosses the ABI or the artifact format.
8. **Functions ≤ 70 lines.**
9. **Assertions ship** (§3).
10. **Zero allocation in the decode loop** is separately tested: a step counter asserts the arena high-water mark is unchanged across a decode step.

---

## 13. Testing

| Tier | What | Gate |
|---|---|---|
| T0 | Per-op golden vectors from `tools/reference.py`, checked in | every commit |
| T1 | Differential: `ref/` vs `simd128/` vs `relaxed/`, max-abs-error per tensor | every commit |
| T2 | Full-model differential against the Python oracle, error profile per layer | every commit |
| T3 | Fuzz (0.16 integrated fuzzer) | nightly |
| T4 | End-to-end quality, chrF++ | per artifact |
| T5 | Perf | nightly + device matrix |

**T1 is why I2 exists.** A jump in per-tensor error at layer N localizes a transposed stride in minutes rather than days. Build it at M3 while there is exactly one backend and it is nearly free.

**T3 targets:** tokenizer against arbitrary bytes (never trap, round-trip within vocab), `format.zig` against corrupted and truncated blobs (return a status, never trap).

**T4 uses two corpora, reported separately, never averaged.** A FLORES subset for comparability, and a chat-register set: short messages, emoji, code-switching, typos, missing punctuation. A model that gains on FLORES and loses on chat register is a regression. Pivot pairs are evaluated end-to-end, not per-hop.

---

## 14. Performance budget

Reference device: 2022-class mid-tier Android (A55-class cores), single-threaded, Chrome stable. Pin one physical unit; CI numbers are for trend detection only.

| Metric | Budget |
|---|---|
| Cold start: instantiate + load + repack, one direction | ≤ 300 ms |
| Warm p50, 12-token message, **direct** | ≤ 80 ms |
| Warm p50, 12-token message, **pivot** | ≤ 160 ms |
| Warm p99, 120-token message, direct | ≤ 800 ms |
| Shared scratch | ≤ 16 MB |
| Weights, per direction | ≤ 20 MB |

p50 and p99 are separate budgets, never a mean. p99 is a long message on the slowest supported device and it decides whether the UI feels broken.

---

## 15. Milestones

| M | Deliverable |
|---|---|
| M0 | Skeleton: `build.zig`, both variants, ABI surface, arena carving, passthrough echo. No math. |
| M1 | Tokenizer: unigram Viterbi + trie. Fuzz-clean, round-trips the full vocabulary. |
| M2 | `tools/convert.py`, `.fzm` format, loader, validation boundary, repack. Golden load test. |
| M3 | `kernel/ref/` — scalar f32, slow, correct. **T1 harness lands here.** |
| M4 | Encoder matching the oracle within epsilon. |
| M5 | Decoder, KV cache, shortlist, greedy decode. First end-to-end translation, one direction. |
| M6 | `simd128/` int8 path. |
| M7 | `relaxed/` variant, host feature detection, `ReleaseFast` measurement. |
| M8 | Routing + pivot. Perf harness, chat-register eval, §14 enforcement. |

M3 before M4 is not negotiable. Correct first, with the oracle in place before there is anything to optimize.

---

## 16. Deferred to v2

WebGPU backend (behind the §8 op seam). Threads (`+atomics` needs COOP/COEP on the PWA; near-zero gain on batch-1 decode). int8 KV cache. Neither blocks anything above.
