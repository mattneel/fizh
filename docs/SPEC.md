# fizh — SPEC

Babel fish in Zig. On-device neural machine translation for a direct messaging app.

Text in, language code in, translated text out. Nothing else.

Zig 0.16, `wasm32-freestanding`, SIMD128. Status: **M0–M8 complete and translating real Mozilla Bergamot models, at parity with `bergamot-translator`.** `tools/fetch-model.sh es en` downloads and converts one; `zig build ci` is green.

On 500 FLORES devtest segments, chrF++ against gold, paired bootstrap over 1000 resamples: es→en −0.08 [−0.37, +0.20], en→de −0.33 [−0.74, +0.03], es→de end to end −0.23 [−0.57, +0.10]. Every interval covers zero. See ADR 0015 for how the last 1.5 points were found.

### Conventions

| Thing | Value |
|---|---|
| ABI prefix | `fizh_` |
| Model artifact | `.fzm`, magic `FIZH`, format version 3 |
| Wasm artifact | `fizh.wasm` |
| Language codes | `u32`, one to four lowercase ASCII bytes packed big-endian, zero-padded on the left: `"es"` is `0x0000_6573`. Codes the registry ships that are not ISO 639-1 get a fizh short form — `zh-Hans` → `zhs`, `zh-Hant` → `zht` (`convert.LANG_SHORT`). This was `u16`, which is a convention this project invented and the registry does not honour. |

Lowercase everywhere. Not an acronym.

---

## 1. Scope

**In:** artifact loading, **sentence segmentation**, **`nmt_nfkc` normalization**, tokenization, quantized seq2seq inference, greedy decode, route resolution and English pivoting, a C ABI.

Normalization is not reimplemented: the model's own `precompiled_charsmap` ships in the artifact and the runtime interprets it (ADR 0017). An NFKC implementation that is correct and disagrees with the model's table is worse than a table interpreter that agrees with it, because the model's segmentation was trained on the table.

Segmentation is not optional. Bergamot's models are trained on single sentences and stop at the end of one; handing them a paragraph returns its first sentence and silently discards the rest, measured at −24 chrF++ (ADR 0011).

**Out:** training, quantization research, language identification (host's job — CLD2 or the browser's detector), the messaging app, transport, crypto. **WebGPU is out entirely** — not deferred (ADR 0014).

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
| I5 | Weights clamp to `[-127, 127]` | Symmetric int8 has no positive counterpart to `-128`. One line in the converter, and the real Bergamot weights already satisfy it: 0 of 16,842,753. (This once cited a future WGSL backend; ADR 0014 deleted that backend, the clamp stays on its own merit.) |
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
| `fizh.wasm` | `+simd128,+bulk_memory,+sign_ext,+nontrapping_fptoint,+mutable_globals` |


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

That sentence is about *sizing*, and it was silent about *content*. Two kinds of region get carved and they are not interchangeable:

| kind | lifetime | may be shared |
|---|---|---|
| **scratch** | one pass; overwritten by the next | yes — that is the point above |
| **per-model derived data** | one model; computed at load from *its* parameters | **no** |

**Per-model derived data lives in that model's slot, never in shared scratch.** A shared region holding something derived from one model's parameters means load order decides correctness, and it fails silently: fluent output, wrong words, no assertion. `pos_enc` was exactly this — sinusoidal encodings depend on `d_model`, a 256-wide table is not a prefix of a 384-wide one, and Firefox ships both widths (ADR 0020).

The rule is enforced structurally rather than by review: derived regions are carved by `model/layout.zig` into the slot, so there is no shared region for them to be written to, and `runtime.buildCtx` asserts the view it hands a pass lies inside that pass's own slot.

Deciding which kind a new region is: *if two loaded models would write different bytes into it, it is derived.* Scratch does not care which model wrote last, because the next pass overwrites it before reading.

### 4.2 Regions

| Region | Size |
|---|---|
| `weights[slot]` | from artifact header, one per loaded direction |
| `xattn_kv` | `2 · max(n_dec_layers) · max_src_tokens · max(d_model) · 4` |
| `ssru_state` | `max(n_dec_layers) · max(d_model) · 4` — the decoder is an SSRU, so its whole recurrent state is one vector per layer (ADR 0008). This replaced a `self_kv` cache of `2 · n_dec · max_tgt · d_model · 4`; at the §4.3 configuration that is 2 KB where the cache was 1.5 MB. |
| `enc_states` | `max_src_tokens · max(d_model) · 4` |
| `act_a`, `act_b` | `max_src_tokens · max(ffn_dim) · 4` each |
| `qact` | same element count as `act_a`, 1 byte each |
| `shortlist_rows` | `max_shortlist · max(d_model) · 1` |
| `shortlist_ids`, `logits` | `max_shortlist · 4` each |
| `sent_spans` | `(max_src_bytes / 2 + 2) · 8` — a sentence needs at least a terminator and a separator |
| `qact_scales`, `attn_scores` | `max(max_src_tokens, max_tgt_tokens) · 4` and `heads · that · 4`. Raising `max_tgt_tokens` from 1.5× to 3× `max_src_tokens` (§4.3, measured ratios) cost ~390 KB across these and `tgt_ids` — recorded here because it went in uncosted. |
| `pos_enc` | **not here.** Carved in the slot, `max(max_src_tokens, max_tgt_tokens) · d_model · 4` per loaded model — it is derived from that model's `d_model` (§4.1, ADR 0020). |
| `tok_raw`, `tok_norm` | `max_src_bytes + 8` each — the `nmt_nfkc` rewrite runs into `tok_raw`, whitespace handling then moves it into `tok_norm`. Two buffers because the rewrite can grow the text and the whitespace pass prepends a marker (ADR 0017). |
| `tok_lattice` | `max_src_bytes · sizeof(LatticeNode)` |
| `io_src` | `max_src_bytes` |
| `io_pivot`, `io_dst` | `2 · max_src_bytes` each — output is not bounded by input. German compounding and English article expansion both exceed the source routinely, and a pivot's English waypoint can exceed both endpoints. Past the factor the pass returns `out_too_small`; it never truncates. |

### 4.3 Worked example

Bergamot-student hyperparameters (`d_model=256`, `ffn=1536`, `n_enc=6`, `n_dec=2`, `vocab=32k`), `max_src_tokens=256`, `max_tgt_tokens=768`, `max_shortlist=2048`, `max_src_bytes=4096`. Verify per artifact at M2.

`max_tgt_tokens` is `3 × max_src_tokens`, which is measured rather than guessed. Across 36 language pairs and 100 FLORES segments each, tokenized with each model's own vocabulary, the worst per-segment target/source token ratio is **2.412** (ko→en) and the median of per-pair medians is 1.032. It was `1.5 ×`, which no measurement supported and which sits below the observed maximum — a source at the limit could have had its tail silently dropped. Three also makes the two bounds in §12.3 agree: the artifacts ship `max_length_factor = 3.0`, so neither bound is dead weight.

| | |
|---|---|
| Shared scratch | 6.74 MB, measured |
| Weights, per direction (int8, ~17M params) | 19.01 MB measured, including vocabulary and shortlist |
| Two directions resident (pivot pair) | ≈ 45 MB total |

A pivot between two `d=384` directions is the expensive case and it is real — 23 of the 105 pairs Firefox ships are that width, so a non-English pivot has a good chance of crossing two of them:

| | |
|---|---|
| Weights, largest selected artifact | 56.7 MB |
| Two resident | 113 MB |
| Shared scratch | ~9 MB |
| **Total before the host's own allocations** | **≈ 122 MB** |

Whether a phone tolerates that is not answerable from a desktop, which is what SPEC §13 T5 and the Pages harness exist to settle.

### The transient peak, which is larger than the arena

**`fizh_model_load` repacks; it does not adopt.** Between the host writing a blob into memory and that call returning, the same weights exist twice — once as the host's copy, once repacked into the slot. The arena figure above does not include the host's copy, and the first Android run measured the difference precisely: a 31.9 MB arena against a 53.3 MB peak heap, a 21.4 MB delta that is the 20.2 MB blob plus buffers.

Two things follow, and a host that ignores either will hit them:

1. **Stage every model through one reused buffer.** Writing each blob at a fresh offset makes the transient `arena + Σ blobs`; reusing one buffer makes it `arena + max(blob)`. On a two-slot pivot of 33 MB artifacts that is the difference between 66 MB and 33 MB of transient. `web/fizh.js` does this and is the reference for it.
2. **The buffer cannot be freed.** wasm memory never shrinks, so the staging region stays reserved for the process lifetime. Size it once, deliberately, and reuse it.

So the worked case above is a *resident* figure, and the number a device actually has to survive is:

| | |
|---|---|
| Resident: two directions + scratch | ≈ 122 MB |
| Transient at load: + the largest blob | **≈ 178 MB** |

That is where iOS jetsam and low-end Android tab-killing live. A host that fetches two 33 MB models into memory *before* calling `fizh_model_load` adds both blobs on top again; fetch and load one at a time.

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
    ssplit.zig        sentence segmentation, Moses rules (ADR 0011)
    unigram.zig       Viterbi over the lattice, iterative
    trie.zig
  kernel/
    ref/              scalar f32 oracle (I2)
    vector/
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

`tools/bergamot.py` reads Marian `.intgemm.alphas.bin` plus its `.spm` vocab and `lex.*.s2t.bin` shortlist and emits one `.fzm`. `tools/fetch-model.sh <src> <tgt>` does the download and the conversion in one step.

**M2 task — answered (ADR 0009). The on-disk int8 is *not* register-tiled.** Type `0x4101` is Marian's `intgemm8`, "quantized (not packed)", architecture-agnostic; tiling happens at load on the target CPU. It *is* already stored `[out][in]`, which is this format's canonical order, so the conversion is a reshape and not a transpose. Doing both collapses the encoder to a single vector by layer six, silently.

Special-token ids are read from the vocabulary, never assumed: `es-en` has `</s>`=0, `en-de` has `</s>`=2 (ADR 0010).

---

## 7. Quantization contract

Artifact ABI. Specified before any kernel is written.

- **Weights:** symmetric, no zero point, per-output-channel `f32` scale, values in `[-127, 127]`. `assert(w != -128)` is a real assertion in the load path — see I5. Marian ships one scale per *tensor*; the converter broadcasts it across the channels, so no weight is ever requantized. Measured: 0 of 16,842,753 int8 weights in the real `es-en` model are `-128`.
- **Activations:** dynamic per-row absmax → `i8` in `[-127, 127]`, `f32` scale, computed at runtime. Bergamot artifacts also ship 53 *static* per-matmul activation multipliers (the `.alphas.` in the filename). **Both paths are implemented and the choice was measured** (ADR 0012): on 500 FLORES segments the byte-identical rate against bergamot-translator differs by ±2 segments in opposite directions between them, and chrF++ by under 0.1. Dynamic stays, and `--activation-quant static` keeps the alternative available for re-testing on a future model.
- **Accumulate in `i32`.** Never `f32` accumulation of integer products.
- **Dequant** is one multiply at the end of the reduction: `@floatFromInt(acc) * (act_scale * w_scale)`.

---

## 8. Kernels

Every op gets a `ref/` implementation first (I2); `vector/` is the optimization validated against it. There was a `relaxed/` for `+relaxed_simd`; it is gone (ADR 0025) — the artifact was byte-identical, `f32x4.relaxed_madd` was never emitted, and the integer dot the variant existed for is unreachable from Zig source.

`vector/` was called `simd128/` until an audit counted what was actually in it: 245 lines of `@Vector`, `@shuffle`, `@select` and `@reduce`, with no wasm builtins, no inline assembly and no target branches. The only target-specific code in the whole kernel tree is three lines of feature detection in `backend.zig` — 867 of 870 lines are target-neutral. The old name encoded a constraint that was never real (ADR 0023).

Integer lane count comes from `std.simd.suggestVectorLength(i8)`, which is 16 on aarch64, 32 on an x86_64 with AVX2, and null on wasm (no CPU model to ask), where it falls back to 16. Widening the integer path is free of consequence because SPEC §7 accumulates in `i32` and integer addition is associative, so any lane count reduces to the identical sum — which is what keeps `vector/` bit-exact against `ref/` on every target.

**The float lane count is deliberately fixed at 4.** Float addition is not associative, so lane count decides reduction order and reduction order decides the last bits; taking the target's suggestion would make a native build round differently from a wasm one. It would also buy nothing where it matters, since the suggestion is 4 on aarch64 and unavailable on wasm — only an AVX2 desktop would ever widen, and no phone would.

| Op | Regime |
|---|---|
| `embed_gather` | random access, dequant on gather. **Step 0 of decode gathers nothing:** Marian shifts the target embeddings right by one with a zero fill, so the first decoder input is the positional encoding alone (ADR 0015). `bos_id` is header metadata; nothing seeds the loop with it. |
| `layer_norm` | row-wise, fixed reduction order |
| `qgemm_i8` | M×K×N, M = src_len — encoder workhorse |
| `qgemv_i8` | 1×K×N — decoder workhorse |
| `attn_prefill` | batched, bidirectional |
| `ssru_step` | batch 1, two `qgemv_i8` and a pointwise gate over one `d_model` cell (ADR 0008). Constant work per token, where attention over a prefix is linear in it. |
| `xattn_decode` | batch 1, reads precomputed `xattn_kv`. The only attention left in the decode loop. |
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
10. **Assert on the shape of the representation, not only the numbers in it.** Every rule above checks magnitude, and layer norm satisfies all of them under total representational collapse — a wrong weight layout drove mean pairwise cosine between encoder positions to 0.9996 with every assertion passing. `graph/encoder.zig` asserts the encoder's positions stay distinct — on the *mean* pairwise cosine, because the max reaches 0.99 in a healthy encoder and discriminates nothing. Sample rather than sweep: a bounded number of pairs is `O(d_model)` and free.
11. **Zero allocation in the decode loop** is separately tested: a step counter asserts the arena high-water mark is unchanged across a decode step.

---

### 12.11 A per-language quality claim requires ruling out per-language tooling failure, in writing

Before recording that fizh is worse at a language, name the tooling that is per-language and say how each was excluded. At minimum: which registry version was selected and why, which vocabulary file was used, whether the shortlist covers the reference's tokens, and whether the tokenizer round-trips.

This is a rule rather than a habit because the project keeps producing the same wrong sentence. Of the seven findings in the ADR 0019 sweep, five were tooling failures presenting as language properties — the worst being a fetcher that selected pre-release models for 21 of 105 pairs, which made "fizh is bad at Czech" look like a measurement when the truth was "we downloaded the wrong file". It is the same shape as the latin-1 vocabulary (ADR 0009), the hardcoded `</s>` (ADR 0010), and the synthetic decoder (ADR 0008): a defect in what surrounds the model, wearing the model's face.

The asymmetry is what makes it worth a rule. A false "this language is hard" costs nothing visible and is never revisited; a false "our tooling is wrong" gets found in an afternoon. Default to the second.

---

## 13. Testing

| Tier | What | Gate |
|---|---|---|
| T0 | Per-op golden vectors from `tools/reference.py`, checked in | every commit |
| T1 | Differential: `ref/` vs `vector/`, max-abs-error per tensor | every commit |
| T2 | Full-model differential against the Python oracle, error profile per layer | every commit |
| T3 | Fuzz (0.16 integrated fuzzer) | nightly |
| T4 | End-to-end quality, chrF++ | per artifact |
| T5 | Perf | nightly + device matrix |

**T1 is why I2 exists.** A jump in per-tensor error at layer N localizes a transposed stride in minutes rather than days. Build it at M3 while there is exactly one backend and it is nearly free.

**T3 targets:** tokenizer against arbitrary bytes (never trap, round-trip within vocab), `format.zig` against corrupted and truncated blobs (return a status, never trap).

**A benchmark asserts its own configuration, or it is not a measurement.** Every harness that emits a number reads its build configuration *from the runtime module* — optimize mode, backend, lane count, whether the hot interiors are unchecked — prints it, and refuses to run when asked for a configuration it did not get (`--expect-mode`). Four gates in this project have failed silently and each produced a plausible number rather than an error: a synthetic decoder architecture (ADR 0008), a decode loop that terminated at step 0 (ADR 0016), a reference engine running with no shortlist (ADR 0018), and `standardOptimizeOption` swallowing `--release=fast` so a `ReleaseSafe` build reported itself as fast (ADR 0026). All four were detectable from inside the harness.

**A T4 delta without a confidence interval is not a measurement.** Report deltas against a reference engine on identical input, never absolute scores across differing input, and run a paired bootstrap (`tools/eval/bootstrap.py`) before treating a gap as real — a −0.42 that straddles zero is the corpus, and hunting a cause for it is hunting noise. When a gap *is* real, `tools/eval/divergence.py` locates it: a per-token hazard that is flat across position is compounding arithmetic, and one that clusters early is a defect at the start of decode (ADR 0015).

**T4 uses three corpora, reported separately, never averaged.** A FLORES subset for comparability, a **multi-sentence** set — FLORES is one sentence per line and structurally cannot measure segmentation, which hid a 24-point defect behind a parity result (ADR 0011) — and a chat-register set: short messages, emoji, code-switching, typos, missing punctuation. A model that gains on FLORES and loses on chat register is a regression. Pivot pairs are evaluated end-to-end, not per-hop.

---

## 14. Performance budget

Reference device: 2022-class mid-tier Android (A55-class cores), single-threaded, Chrome stable. Pin one physical unit.

**Build mode:** the ship build is `ReleaseSafe`, and the two loop interiors that dominate it — `qgemmTile` and `dotI8` — set `@setRuntimeSafety(false)`. That is worth **2.02x** (109.3 ms → 54.2 at 164 source tokens, against 48.2 for a whole-program `ReleaseFast`), and it leaves the load path and every other function checked. Check once at the boundary, not once per element (ADR 0026).

**T5 has run.** The budgets below are mobile measurements, not desktop proxies. The first run was an 8-core armv81 Android 10 device, 8 GB, Chrome 150, on `es→en`. (It used the then-current relaxed artifact, which has since been shown byte-identical to the one that ships — ADR 0025 — so the numbers stand.)

Work order 5 retightened this table to ~1.5× *desktop* measurements. That was a mistake with a specific cost: a desktop is not a basis for a mobile budget, and the retightened paragraph row failed on the phone at 287.7 ms against 100 ms — a false failure produced by the budget, not by the runtime. The numbers below return to the mobile-targeted scale the table originally had, which the device says was closer to right.

| Metric | Budget | Measured (Android, es→en) |
|---|---|---|
| Cold start: instantiate + load + repack, one direction | ≤ 300 ms | **29.3** |
| Warm p50, 12-token message, **direct** | ≤ 80 ms | **59.5** |
| Warm p50, 12-token message, **pivot** | ≤ 160 ms | not yet run |
| Warm p99, 120-token message, direct | ≤ 900 ms | **847.8** ⚠ see below |
| Warm p50, 8-sentence paragraph | ≤ 400 ms | **287.7** (35.9 per sentence) |
| Shared scratch | ≤ 12 MB | 6.4 at the §4.3 config |
| Weights, per direction | ≤ 64 MB | 19.2 (tiny), 33.4 (base-memory), up to 56.7 (base) |

Three things this table now states that the desktop one could not:

**The 120-token p99 is marginal and its first measurement was not a p99.** 847.8 ms came from 30 samples, where the 99th percentile is definitionally the worst one. The harness now refuses to print a p99 below 300 runs and reports `max` instead, labelled as max. Treat 847.8 as a max until a 300-run figure replaces it.

**The paragraph row has a unit.** Eight sentences, fixed, and per-sentence cost is reported beside the total. Without that the row could be neither passed nor failed: 287.7 ms for eight sentences is *faster per sentence* than the 12-token case, because the encoder amortizes across a warm cache.

**Burst and steady state are separate numbers.** The first run showed min 39.6 against p50 59.5 with no tail above it — a 50% spread that is DVFS or big.LITTLE migration on an 8-core heterogeneous device, not the runtime. Web code cannot pin affinity, so the harness reports burst (first quarter of samples) and steady (second half) as distinct rows and records run duration and page visibility. **Do not tune against a number whose variance is the CPU scheduler.**

### Against bergamot, measured

Same page, same machine, same upstream weights, foreground (ADR 0027). Both engines single-threaded.

| | fizh | bergamot |
|---|---|---|
| 120-token p50 | 222.9 ms | 208.3 ms |
| per-token throughput | 1.3657 ms/tok | 1.3371 ms/tok |
| cold start | **20.5 ms** | 128.8 ms |
| engine | **102 KiB** | 5,120 KiB |
| peak linear memory | **86 MiB** | 522 MiB |

Throughput ratio **1.021** — parity. What remains is ~10 ms of fixed per-call cost, which natively is 0.09 ms and is therefore wasm-specific. This is x86, where intgemm has AVX2 and VNNI paths ARM does not; **I1 is a claim about ARM and is still unmeasured.**

### Desktop figures, which are not budgets

`zig build bench` still measures on the build machine. Those numbers are useful for catching a 3× regression between commits and for nothing else; they are not a budget and passing them means nothing about a phone. At the §4.3 config on the development desktop: cold start 5.8 ms, p50 12-token 14.8, p99 120-token 130, paragraph 64.8, scratch 6.4 MB.

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
| M5 | Decoder (SSRU), cross-attention KV, shortlist, greedy decode. First end-to-end translation, one direction. |
| M6 | `simd128/` int8 path. |
| M7 | `relaxed/` variant, host feature detection, `ReleaseFast` measurement. **Reverted** in M11: the variant bought nothing and the feature was a pessimization (ADR 0025). |
| M8 | Routing + pivot. Perf harness, chat-register eval, §14 enforcement. |

M3 before M4 is not negotiable. Correct first, with the oracle in place before there is anything to optimize.

---

## 16. Deferred to v2

Threads (`+atomics` needs COOP/COEP on the PWA; near-zero gain on batch-1 decode). int8 cross-attention KV — the self-attention cache this once targeted no longer exists (ADR 0008), so the remaining prize is the 1 MB `xattn_kv` rather than the 2.5 MB implied here. None of it blocks anything above.
