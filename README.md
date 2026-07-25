# fizh

Babel fish in Zig. On-device neural machine translation for a direct messaging
app.

Text in, language code in, translated text out. Nothing else.

`docs/SPEC.md` is the specification. This file is the map.

| | |
|---|---|
| `docs/SPEC.md` | what fizh is, and every budget it is held to |
| `docs/ABI.md` | the public surface, and what a version bump means for artifacts you already have |
| `docs/COVERAGE.md` | all 105 language pairs: what works, what does not, and by how much |
| `docs/adr/` | why, including the negative results |

```
zig build                                        # Zig 0.16
tools/fetch-model.sh es en                       # 19 MB, downloads and converts
echo 'El gato negro duerme en la mesa.' \
  | ./zig-out/bin/translate --model zig-out/esen.fzm --src es --tgt en
# The black cat sleeps on the table.
```

Those three lines are the whole path from an empty checkout to a translation,
and they are tested as such — `git clone`, build, fetch, translate.

## It works

Real Mozilla Bergamot models, real translation, through the shipped wasm.

| | |
|---|---|
| es→en | `Hola, ¿qué tal?` → `Hey, how are you?` |
| en→de | `Good morning, how are you?` → `Guten Morgen, wie geht es dir?` |
| es→de, pivoted | `Buenos días, ¿cómo estás?` → `Guten Morgen, wie geht es dir?` |
| chrF++, chat corpus | **46.73** (bergamot-translator: 45.86) |
| warm p50, 12 tokens | **59.5 ms** on Android (budget 80) |
| warm p50, 8-sentence paragraph | **287.7 ms**, 35.9 per sentence (budget 400) |
| cold start | **29.3 ms** on Android (budget 300) |
| slot, per direction | **19.23 MB** (budget 20) |
| shared scratch | **7.16 MB** (budget 10) |
| `fizh.baseline.wasm` | **32 KB gzipped**, 0 imports, 8 exports (budget 200 KB) |

**§14's budgets are mobile and T5 has run**: an 8-core armv81 Android 10
device, Chrome 150, `es→en`. Timing rows above are that device; size rows are
build-time facts. The desktop figures `zig build bench` prints are for
catching a regression between commits and are not a budget.

One caveat travels with the first run: its p99 came from 30 samples, where the
99th percentile is the worst sample by definition. The harness now refuses to
print a p99 below 300 runs.

### Against the reference implementation

500 FLORES devtest segments, chrF++ against gold, paired bootstrap over 1000
resamples. Deltas against `bergamot-translator` on identical input, run
`--per-line` — one sentence per batch, the only configuration comparable to a
library that translates one message at a time (ADR 0018). Never absolute scores
across differing input.

| direction | delta | 95% CI | |
|---|---|---|---|
| es→en | −0.10 | [−0.35, +0.15] | indistinguishable |
| en→de | −0.16 | [−0.47, +0.14] | indistinguishable |
| es→de, end to end | −0.17 | [−0.50, +0.16] | indistinguishable |
| chat register (n=40) | +0.71 | [−2.04, +3.04] | indistinguishable |

Every interval covers zero. 45% of segments are byte-identical to the
reference; the rest diverge at a per-token hazard of ~0.025, flat across
position, which is what two int8 implementations rounding differently under
greedy decode look like.

### Coverage: all 105 pairs Firefox ships

`docs/COVERAGE.md` has the matrix. In summary:

| | |
|---|---|
| usable | **99 / 105** |
| within ±1.0 chrF++ of the reference | **87 / 99** |
| median delta | **+0.17** |
| architectures | `d=256, 6+2` (76 pairs), `d=384, 6+4` (23) |

Six are refused with a stated reason: five have separate source and target
vocabularies, one has a script-qualified language code that two ASCII letters
cannot hold. Four more (`bg` and `fr`, both directions) are usable but score
10–22 low, because their v1.0 "tiny" artifact is a poor one — their v2.0
translates correctly through fizh but is 33 MB, over the §14 weights budget.

Sweeping the registry is where most of the recent bugs came from, and every one
of them looked like a property of a language until it was a line in a table
next to a hundred that worked (ADR 0019).

Every SPEC §14 budget met, on the real thing.

## Build

```
zig build ci        everything below
zig build wasm      fizh.{baseline,relaxed}.wasm + fizh.probe.wasm
zig build test      T0/T1 and the unit suite, all three backends
zig build real      recorded output of a real Bergamot model (skips if absent)
zig build tiger     Tiger Style over src/ (SPEC §12)
zig build check     wasm budgets: imports, exports, gzipped size (SPEC §3)
zig build oracle    T2 full-model differential against tools/reference.py
zig build host      drives the shipped wasm from a JS host, as the PWA would
zig build bench     SPEC §14 budgets
zig build eval      T4 chrF++, per corpus, never averaged
```

## Where things are

| Path | |
|---|---|
| `src/root.zig` | The eight ABI exports (SPEC §9). Nothing else. |
| `src/runtime.zig` | The instance: arena, slot table, the body of `fizh_translate`. |
| `src/abi.zig` `src/arena.zig` `src/route.zig` | Status codes and config; region carving; direct/pivot resolution. |
| `src/model/` | `format.zig` is **the** validation boundary; `layout.zig` places every weight in a slot; `repack.zig` gets it there; `names.zig` hashes names. |
| `src/tok/` | `trie.zig` walks the vocabulary; `unigram.zig` runs Viterbi over the byte lattice. |
| `src/kernel/` | `ref/` is the scalar oracle (I2); `simd128/` and `relaxed/` are validated against it. `math.zig` is the libm that isn't there. |
| `src/graph/` | `encoder.zig`, `decoder.zig` (SSRU), `attention.zig`, `shortlist.zig`, `pass.zig`. |
| `tools/marian.py` | Readers for Marian's binary model and lexical shortlist. |
| `tools/bergamot.py` | A Bergamot bundle → `.fzm`. |
| `tools/fetch-model.sh` | Download and convert, from nothing, in one command. |
| `tools/eval/sweep.py` | Every registry pair: fetch, convert, load, translate, score. |
| `tools/eval/bootstrap.py` | Paired bootstrap. A delta without an interval is not a measurement. |
| `tools/eval/divergence.py` | Per-token hazard and where the first divergence lands. |
| `test/` | `golden/` T0 vectors and end-to-end, `real/` translations recorded **from bergamot**, `fuzz/` T3. `differential_test.zig` is T1. |

## What the real model taught us

The runtime was built to the SPEC and passed 98 tests against synthetic
artifacts. Then a real Bergamot model was loaded, and three things the tests
could not have known came out:

**The decoder is an SSRU, not self-attention** (ADR 0008). SPEC §8's
`attn_decode` and §4.2's `self_kv` describe an architecture Mozilla does not
ship. The real fix made things *better*: the KV cache became one 2 KB vector
instead of 1.5 MB, and per-token work stopped growing with output length.

**Marian's int8 is not register-tiled, and is already transposed** (ADR 0009) —
SPEC §6's open M2 question, finally answered against the bytes. Getting it
wrong was silent: the encoder collapsed to a single vector by layer six and
every summary statistic still looked healthy, because layer norm hides it.

**Special-token ids are not constants** (ADR 0010). `es-en` has `</s>`=0,
`en-de` has `</s>`=2. Hardcoding it gave a decoder that could never stop, which
presents as a quality problem and is a configuration problem.

The lesson is in ADR 0008 and it is the reason `zig build real` exists: *the
artifact is the specification; the SPEC is a description of it, and descriptions
can be wrong.*

## The decisions

| ADR | |
|---|---|
| 0001 | The arena owns the weight slots. |
| 0002 | Regions and files beyond the SPEC tables. |
| 0003 | fizh brings its own transcendentals — zero imports means no libm. |
| 0004 | The trie is implicit; the boundary marker is one byte; the unknown edge only appears when nothing matched. |
| 0005 | Converter status. Its intgemm claim is superseded by 0009. |
| 0006 | What relaxed SIMD is for, the `pmaddubsw` trap, and the ReleaseFast number. |
| 0007 | What a passing T2 looks like when the function under test is discontinuous. |
| 0008 | The decoder is an SSRU. |
| 0009 | Marian's on-disk int8: not tiled, already transposed. |
| 0010 | Special ids come from the vocabulary; shortlist ids are `u16`. |
| 0011 | fizh splits sentences. Bergamot's models stop at the end of one. |
| 0012 | Static activation alphas, measured and rejected. A negative result. |
| 0013 | Rejoin fidelity: what segmentation costs at the seams. |
| 0014 | No WebGPU. |
| 0015 | The decoder starts from a zero vector, not `emb(</s>)`. Worth 1.5 chrF++ on en→de, and the method — teacher forcing — matters more than the fix. |
| 0016 | Benchmark a model that terminates. The §14 numbers were timing an empty decode loop. |
| 0017 | Ship the model's own `nmt_nfkc` table; do not reimplement NFKC. |
| 0018 | The shortlist is not free — and the reference was not using one. Two harness bugs, each producing a confident wrong conclusion. |
| 0019 | Coverage is not two pairs. What sweeping all 105 found. |

## Known gaps

- **`¿` and `¡` are not in Bergamot's `es-en` vocabulary**, so they tokenize to
  `<unk>`. That is Bergamot's behaviour too, not a fizh bug.
- **No FLORES corpus is vendored** (CC-BY-SA); `tools/eval/corpora/README.md`
  has the three lines that build one. Only the chat register ships.
- **I1 is not yet tested.** The project's premise is that bergamot's engine is
  intgemm-based and x86-tuned while phones are ARM. Comparing fizh's relaxed
  build to its own baseline measures relaxed SIMD, not that. The benchmark page
  now runs bergamot itself on the device; until that number exists, I1 is an
  assumption.
- **The transient load peak is ~178 MB for a two-slot pivot** of the largest
  artifacts, because `fizh_model_load` repacks rather than adopts. Stage
  through one reused buffer and load one model at a time (SPEC §4.3).
- **fizh's shortlist costs it ~0.27 chrF++ on en→de** against its own
  full-vocabulary projection. Bergamot pays the same tax at the same batch
  size; the per-sentence scope is correct for a library that translates one
  message at a time (ADR 0018).
- **The shipped `.fzm` is 19.23 MB against a 20 MB budget.** The `nmt_nfkc`
  table is 237 KB of that and is the first thing to drop if a future artifact
  needs the headroom (ADR 0017).
- **Six pairs are unsupported**, with stated reasons: `en-ja`, `en-ko`,
  `en-zh-Hans`, `en-zh-Hant` and `zh-Hant-en` have separate source and target
  vocabularies; `zh-Hans-en` has a language code two ASCII letters cannot hold.
  Supporting either is a different runtime, not a flag.
- **`bg` and `fr` score 10–22 low in both directions** because their v1.0 tiny
  artifact is a poor one, not because of fizh. See `docs/COVERAGE.md`.
- **Pivoting is only through English**, and only where both hops exist.
