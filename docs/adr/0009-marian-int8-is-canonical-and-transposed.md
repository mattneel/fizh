# ADR 0009 — Marian's on-disk int8: not tiled, already transposed

Status: accepted
Date: 2026-07-25
Milestone: M2 (the task SPEC §6 set)
Supersedes: the intgemm section of ADR 0005

## The question

SPEC §6, verbatim:

> **M2 task: confirm whether Marian's on-disk int8 layout is already
> register-tiled for intgemm and unshuffle to canonical if so.**

ADR 0005 guessed "yes, tiled, and width-dependent, so refuse to read it". That
guess was wrong on both counts, and it cost the project the ability to load any
real model.

## The answer

**Not tiled.** Every weight tensor in the shipped model carries type `0x4101`.
In `browsermt/marian-dev`, `src/common/types.h`:

```cpp
intgemm_type = 0x4000,   // intgemm quantized architecture agnostic models
intgemm8      = TypeClass::signed_type + 1u + TypeClass::intgemm_type,
                // Int8 quantized (not packed) matrices for intgemm
intgemm8ssse3 = ... + TypeClass::ssse3_type,   // quantized *and packed*
intgemm8avx2  = ... + TypeClass::avx2_type,
intgemm8avx512= ... + TypeClass::avx512_type,
```

`0x0100 + 1 + 0x4000` = `0x4101` exactly. The shipped artifact uses the plain,
**unpacked**, architecture-agnostic type; the register-tiled variants are
distinct types and Bergamot does not ship them. bergamot-translator calls
intgemm's `PrepareB` at load, on the target machine, which is the whole point of
keeping the file neutral.

**But it is already transposed.** A tensor whose header shape is `[K, N]` — the
logical `[in][out]` of Marian's `x @ W` — has its bytes laid out as `N`
contiguous rows of `K` int8. That is `[out][in]` row-major, which is exactly
SPEC §6's canonical form.

So the correct conversion is a **reshape, not a transpose**. Doing both — which
is what `[in][out]` in the header naively implies — is a genuine transpose, and
it is silent: the model loads, the arithmetic runs, the encoder output is
layer-normed and therefore *looks* healthy at every summary statistic.

The way it actually presented was representation collapse. Mean pairwise cosine
between encoder positions, layer by layer:

    transposed (wrong)  0.660  0.849  0.955  0.992  0.998  1.000
    reshaped   (right)  0.619  0.364  0.587  0.678  0.642  0.800

By layer six every source position had the same vector, cross-attention went
uniform (0.10–0.13 across nine positions), and the decoder emitted `ly` forever.

`Wemb` is the exception: genuinely `[vocab][dim]`, not transposed, because
Marian uses it as an embedding gather and transposes it explicitly at the tied
output projection.

## Block sizing

Marian's `TensorAllocator` aligns to 256 bytes, and intgemm tensors carry their
quantization multiplier as one `f32` immediately after the data:

    dataLength = roundUp(elems * sizeof(type) + (isIntgemm ? 4 : 0), 256)

Verified against the real file for `elems` = 1, 65536 and 8192000; the reader
consumes 17,140,755 bytes exactly. The padding is **uninitialized** — stale
weight bytes, not zeros — so nothing may read or checksum past the data.

## Quantization convention

`w = q / quantMult`, a divisor, one scalar per tensor. The separate
`*_QuantMultA` tensors are *activation* multipliers — the "alphas" in the
filename — and fizh ignores them: SPEC §7 specifies dynamic per-row absmax,
which is computed at runtime and is at least as accurate.

The per-tensor weight scale is broadcast to SPEC §7's per-output-channel array.
Broadcasting is exact, so no weight is requantized: the int8 Mozilla trained is
the int8 fizh multiplies.

## Consequences

- `tools/marian.py` reads the model, the shortlist and the config; `tools/
  bergamot.py` converts a whole bundle; `tools/fetch-model.sh` does both from
  nothing in one command.
- SPEC §6's M2 task is closed.
- I5 survives contact with reality: **0 of 16,842,753** int8 weights in the real
  model are `-128`, so `repack.zig`'s rejection never fires on a valid artifact.

## The lesson

The failure was not "we got the layout wrong" — that is an ordinary bug. The
failure was that a *guess* about an external format was written into an ADR in
the language of a decision, and then believed. ADR 0005 said the path was
"unverified" and stopped there. Verifying it took one afternoon and a
`curl`.
