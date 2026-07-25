# fizh

Babel fish in Zig. On-device neural machine translation for a direct messaging
app.

Text in, language code in, translated text out. Nothing else.

`docs/SPEC.md` is the specification. This file is the map.

```
tools/fetch-model.sh es en                       # 19 MB, one command
echo 'El gato negro duerme en la mesa.' \
  | ./zig-out/bin/translate --model zig-out/esen.fzm --src es --tgt en
# The black cat sleeps on the table.
```

## It works

Real Mozilla Bergamot models, real translation, through the shipped wasm.

| | |
|---|---|
| es→en | `Hola, ¿qué tal?` → `Hey, how are you?` |
| en→de | `Good morning, how are you?` → `Guten Morgen, wie geht es dir?` |
| es→de, pivoted | `Buenos días, ¿cómo estás?` → `Guten Morgen, wie geht es dir?` |
| chrF++, chat corpus | **42.51** |
| warm p50, 12 tokens | **15.2 ms** native, **22 ms** in wasm (budget 80) |
| warm p99, 120 tokens | **158 ms** (budget 800) |
| cold start | **6.4 ms** (budget 300) |
| slot, per direction | **19.01 MB** (budget 20) |
| shared scratch | **6.74 MB** (budget 16) |
| `fizh.baseline.wasm` | **27 KB gzipped**, 0 imports, 8 exports (budget 200 KB) |

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
- **SentencePiece's `nmt_nfkc` normalization is not implemented** — only the
  structural preprocessing (ADR 0004). Worth measuring on FLORES.
- **No FLORES corpus is vendored** (CC-BY-SA); `tools/eval/corpora/README.md`
  has the three lines that build one. Only the chat register ships.
- **T5 has never run on the reference device.** SPEC §14 pins a 2022-class
  mid-tier Android; every number above is a desktop, which proves less than it
  looks like.
- **A first-token artifact** shows up occasionally (`of the meeting has been
  postponed…`). Recorded in `test/real/golden_translations.tsv` rather than
  hidden; worth chasing against bergamot-translator's own output.
