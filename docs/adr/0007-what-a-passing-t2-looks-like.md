# ADR 0007 — What a passing T2 looks like

Status: decided
Date: 2026-07-24
Milestone: M4/M5

## Context

SPEC §13 T2 is "full-model differential against the Python oracle, error profile
per layer", gated on every commit. The obvious implementation is: compute the
forward pass twice, assert the outputs agree to some epsilon. The first run of
it looked like a bug:

    encoder    FAIL max-abs 8.194e-02 over 3.091e+00 = 2.650e-02

Two and a half percent, from an oracle written specifically to mirror the
runtime's quantization, in float32, with the same per-row absmax and the same
`i32` accumulation. That is far too large for arithmetic.

## What it actually was

The per-layer profile — which is exactly why SPEC §13 asks for one — said:

    stage      max-abs     magnitude   relative
      embed    0.000e+00   3.127e+00   0.000e+00
      layer 0  4.768e-07   3.344e+00   1.426e-07
      layer 1  2.917e-02   3.580e+00   8.150e-03   <-- jump
      layer 2  5.066e-02   3.752e+00   1.350e-02
      ...
      layer 5  8.194e-02   3.091e+00   2.650e-02

The embedding is bit-identical. Layer 0 agrees to one ulp. Then it jumps five
orders of magnitude in a single layer and grows slowly after.

`reference.py --selfcheck` settles it. It runs the encoder twice through the
*same* code, once with float32 intermediates and once with float64 — a
perturbation the size of one ulp, and nothing else:

    oracle(float32) vs oracle(float64)
      embed    relative 7.205e-08
      layer 0  relative 7.115e-03
      layer 5  relative 2.626e-02

The oracle diverges from **itself** by 2.6e-2. The same magnitude. The
implementations are not disagreeing; the *function* is discontinuous.

Dynamic int8 quantization is a step function. A one-ulp difference in an
activation that happens to sit on a rounding boundary flips a code by one, which
moves that channel by `absmax / 127` — about 0.8%. Six encoder layers, four
matmuls each, 256 channels: one flip somewhere is close to certain, and it
propagates.

## Decision

T2's verdict is **structural**, not a single epsilon.

| Check | Bound | Why |
|---|---|---|
| Tokenizer ids | exact | No floats involved. Any difference is a bug. |
| Embedding | 1e-5 | Gather, scale, add. No quantization, so no discontinuity. |
| Encoder **layer 0** | 1e-5 | One layer of quantization: the codes still agree, so this is plain float32 rounding. **This is the diagnostic stage.** A transposed stride or a swapped weight lands here at ~1e0, five orders of magnitude clear of the noise. |
| Encoder layers 1..N | 2.5e-1 | Above the measured 2.6e-2 noise floor, far below the ~1e0 a real bug produces. |
| Decoder ids | exact | See below. |

The decoder check is the strongest one in the project and it costs nothing.
Seventeen consecutive `argmax` operations over a 1042-entry shortlist, each one
downstream of two decoder layers, a KV cache, cross-attention over the encoder
states, and a shortlisted projection — agreeing exactly with an independent
numpy implementation. A graph with a transposed weight anywhere does not survive
that.

`zig build oracle` runs `--selfcheck` alongside the comparison, so the noise
floor is *measured* on every CI run rather than asserted once in this document.
If it ever exceeds the bound, `--selfcheck` fails and says so.

## Consequences

- `src/graph/encoder.zig` carries a `layer_sink` function pointer, null except
  under `tools/trace.zig`. One null check per layer, in exchange for an error
  profile that says *where*.
- The same reasoning applies to T4: a chrF++ regression of a fraction of a point
  between two builds may be one code flip rather than a quality change. Compare
  artifacts, not builds.
- If quantization sensitivity ever needs to come *down* rather than be reasoned
  around, the lever is per-channel activation scales or a wider accumulator —
  both of which are I4 changes and need their own ADR.
