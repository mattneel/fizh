# ADR 0014 — No WebGPU backend

Status: decided
Date: 2026-07-25
Milestone: v2 scope
Supersedes: SPEC §16's WebGPU line

## Decision

Delete WebGPU from §16. It is not deferred; it is not planned.

## Why the case collapsed

§16 deferred "WebGPU backend (behind the §8 op seam)". The argument had two
halves, and ADR 0008 removed the larger one.

**Decode.** With an SSRU decoder, a step is two `qgemv`s of 256×256 and a
pointwise gate, per layer, twice. That is roughly 0.13 MFLOP per token against
a dispatch overhead measured in tens of microseconds on any WebGPU
implementation. There is no parallelism left to find: the recurrence is
sequential by construction, so the only axis a GPU could exploit is the one the
model does not have. Decode was already the weaker half of the argument when it
was self-attention with a growing KV cache; SSRU removes it entirely.

**Encode.** The encoder is a real `qgemm` workload and would benefit. But it is
one part of a pass that measures 15.2 ms warm p50 native and 22 ms through
wasm, against an 80 ms budget — 3.6× of headroom before a second backend buys
anything a user can perceive. Against that: a WGSL kernel set, a second
numerical path for T1 and T2 to validate, a device-capability matrix, and an
adapter-loss failure mode the CPU path does not have.

A second backend is justified when the first one misses its budget. This one
does not.

## The stale rationale it leaves behind

I5 clamps weights to `[-127, 127]`, and §2 justifies that as keeping "the
artifact valid for a future WGSL backend without requantizing". That
justification is now void.

**Keep the clamp.** It costs one line in the converter, it is the natural range
for symmetric int8, `-128` has no positive counterpart, and the real Bergamot
weights already satisfy it — 0 of 16,842,753 are `-128` (ADR 0009). But the
reason recorded in §2 should be the honest one — symmetry — not a backend that
is not coming.

## Revisit when

The reference device is pinned and T5 runs on it. If an A55-class phone puts
the encoder over budget where a desktop had 3.6× of headroom, the encode half
of this argument reopens. The decode half does not.
