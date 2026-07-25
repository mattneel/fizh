# ADR 0005 — What `tools/convert.py` can and cannot do yet

Status: superseded
Date: 2026-07-24
Milestone: M2

## Context

SPEC §6 sets an M2 task: "confirm whether Marian's on-disk int8 layout is
already register-tiled for intgemm and unshuffle to canonical if so." Answering
it needs a real Bergamot artifact and a known-good translation to check against.
This repository has neither.

## Decision

`tools/convert.py` has two input paths and is explicit about which is which.

**`--npz` — the working path.** Marian writes numpy archives; numpy reads them;
the weights are `f32`. The converter does the SPEC §7 quantization itself:
symmetric, per-output-channel, `absmax / 127`, clipped to `[-127, 127]`. That
clip is the one line SPEC §7 asks for, and `model/repack.zig` re-checks it at
load, so I5 holds whatever the input was.

**`--intgemm` — refused, loudly.** intgemm stores B matrices column-major and
register-tiled, and the tiling depends on which width packed them (SSSE3, AVX2
and AVX512 differ). `model.intgemm.alphas.bin` carries no record of which. The
converter therefore refuses rather than guessing, and prints why.

Quantizing from `f32` is also the better artifact even once the intgemm path
works: the `.intgemm` file has already been through someone else's rounding, and
`[-127, 127]` was not necessarily their constraint.

**The lexical shortlist is not read yet.** `read_lex` checks the header against
the vocabulary size and refuses if they disagree, rather than emitting a
shortlist that silently degrades every translation. Marian's `lex.s2t.bin`
layout has changed across releases and the same "verify against a real file"
problem applies. Without `--lex`, the converter emits an empty per-source table
and a frequent-token list, which SPEC §8's `shortlist_build` handles correctly —
it just makes for a bad shortlist.

## How this stays honest

`zig build convert-selftest` runs `convert.py --selftest` to build an artifact
with no Marian input, then loads it with `tools/fzm_load.zig`, which drives the
*real* `src/model/format.zig`. Two independent implementations of SPEC §6,
checked against each other on every CI run. `test/artifact.zig` is a third,
in Zig, used by the golden load tests.

## What remains

1. Obtain a Bergamot `es-en` bundle. Answer the §6 intgemm question against it.
2. Implement `read_lex` against that release's format, and measure the T4 delta
   between a real shortlist and the frequent-only fallback. SPEC §14 gives the
   shortlist roughly 2 MB of the 20 MB per-direction budget, which is about 18
   candidates per source piece — Bergamot itself uses 50, so this is a real
   tradeoff and it should be measured, not assumed.
3. Decide whether SentencePiece's `nmt_nfkc` normalization table needs to ship
   (see ADR 0004).
