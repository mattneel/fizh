# fizh

Babel fish in Zig. On-device neural machine translation for a direct messaging
app.

Text in, language code in, translated text out. Nothing else.

`docs/SPEC.md` is the specification. This file is the map.

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
| warm p50, 12 tokens | **15.6 ms** (budget 22) |
| warm p99, 120 tokens | **134 ms** (budget 200) |
| warm p50, 8-sentence paragraph | **66 ms** (budget 100) |
| cold start | **6.5 ms** (budget 10) |
| slot, per direction | **19.23 MB** (budget 20) |
| shared scratch | **6.77 MB** (budget 10) |
| `fizh.baseline.wasm` | **32 KB gzipped**, 0 imports, 8 exports (budget 200 KB) |

Budgets are SPEC §14 as retightened to ~1.5x measured — the originals were
sized for a 600M-parameter model and left 4-45x of headroom, which cannot catch
a 3x regression. Every number is a desktop proxy; see the last section.

### Against the reference implementation

500 FLORES devtest segments, chrF++ against gold, paired bootstrap over 1000
resamples. Deltas against `bergamot-translator` on identical input, never
absolute scores:

| direction | delta | 95% CI | |
|---|---|---|---|
| es→en | −0.08 | [−0.37, +0.20] | indistinguishable |
| en→de | −0.33 | [−0.74, +0.03] | indistinguishable |
| es→de, end to end | −0.23 | [−0.57, +0.10] | indistinguishable |

Every interval covers zero. 46% of segments are byte-identical to the
reference; the rest diverge at a per-token hazard of ~0.025, flat across
position, which is what two int8 implementations rounding differently under
greedy decode look like.

Getting there took finding a real bug: fizh seeded the decoder with
`emb(</s>)` where Marian zero-fills, which got roughly half of all first
tokens wrong and cost 1.5 chrF++ on en→de. ADR 0015 has the method — teacher
forcing, which is what separates a defect at position 0 from its own blast
radius.

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
| `test/` | `golden/` T0 vectors and end-to-end, `real/` recorded translations, `fuzz/` T3. `differential_test.zig` is T1. |

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

## Known gaps

- **`¿` and `¡` are not in Bergamot's `es-en` vocabulary**, so they tokenize to
  `<unk>`. That is Bergamot's behaviour too, not a fizh bug.
- **No FLORES corpus is vendored** (CC-BY-SA); `tools/eval/corpora/README.md`
  has the three lines that build one. Only the chat register ships.
- **T5 has never run on the reference device.** SPEC §14 pins a 2022-class
  mid-tier Android; every number above is a desktop, which proves less than it
  looks like.
- **The shortlist costs en→de about 0.27 chrF++** against full-vocabulary
  projection, which is most of the remaining gap. Not the size cap and not the
  frequent-word set — the per-source-token candidate lists (ADR 0018).
- **The reference engine was not running with a shortlist enabled**, so the
  deltas above compare fizh-with-shortlist against bergamot-at-full-vocab.
  Conservative rather than wrong, but not the comparison claimed. ADR 0018.
- **The shipped `.fzm` is 19.23 MB against a 20 MB budget.** The `nmt_nfkc`
  table is 237 KB of that and is the first thing to drop if a future artifact
  needs the headroom (ADR 0017).
