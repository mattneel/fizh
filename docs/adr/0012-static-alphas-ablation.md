# ADR 0012 — Static activation alphas: measured, and rejected

Status: decided
Date: 2026-07-25
Milestone: M2/M5
Amends: SPEC §7

## The hypothesis

The artifact is `model.<pair>.intgemm.alphas.bin` and Bergamot's precision mode
is `int8shiftAlphaAll`. Those alphas are precomputed **activation**
quantization multipliers, shipped in the file. SPEC §7 has fizh recompute
activation scales dynamically per row instead — a divergence specified before
anyone knew the artifact carried the alternative.

That predicted the residual: a small systematic gap with no localizable cause,
near-ties flipping at step 0, and a magnitude varying by direction because the
alphas are calibrated per model.

## What the alphas are

Confirmed before any kernel was touched.

`*_QuantMultA`, 54 per model, `f32` scalars. **53 int8 weights, 53 matching
alphas** — a 1:1 mapping keyed by the weight tensor, plus a `none_QuantMultA`
fallback for the unnamed matmul. From `integer_common.h`:

```cpp
fetchAlphaFromModelNodeOp(Expr b) { ... aQuantKey = b->name() + "_QuantMultA"; }
```

The key is the **weight**, so the multiplier is per *matmul*, not per layer.
Their implied activation magnitudes — `127 / alpha` — land between 4.5 and 12.4,
exactly layer-norm scale, which is the corroboration that they are activation
multipliers and not something else.

fizh's dynamic scales run 0.76–0.95× of them on real input: *finer*, therefore
slightly more precise, but different.

## The ablation

Both paths implemented and kept. `--activation-quant {dynamic,static}` in the
converter, `act_quant` in the header, `quantizeRowsWith` in all three backends.
An artifact that asks for static without shipping alphas is rejected at load
rather than silently falling back, because the two paths produce different
numbers.

FLORES-200 devtest, 500 segments, against bergamot-translator on identical
input. **Primary metric is byte-identical rate**, because chrF++ forgives a
flipped token that lands on a synonym:

| direction | mode | byte-identical | chrF++ | delta |
|---|---|---|---|---|
| es→en | dynamic | 108/500 (21.6%) | 54.07 | −0.42 |
| es→en | static  | 106/500 (21.2%) | 54.10 | −0.38 |
| en→de | dynamic | 106/500 (21.2%) | 61.00 | −1.79 |
| en→de | static  | 108/500 (21.6%) | 61.08 | −1.72 |

**±2 segments out of 500, in opposite directions.** Noise. chrF++ moves +0.03
and +0.08, also noise.

The first-token case (`The` vs `of`) does flip to match bergamot under static.
That is not evidence: ADR 0007 measured the margin at 0.054 against a ~0.26
noise floor, so *any* perturbation flips it. A coin landing heads is not a
theory of coins.

**The hypothesis is dead.** Static alphas are not the residual.

## Decision

SPEC §7 keeps dynamic per-row absmax, and now carries the measurement that
justifies it, which it previously lacked. The static path stays in the tree
behind `--activation-quant static`: it cost little, it is the only way to
re-run this ablation on a future model, and deleting it would leave the ADR
unfalsifiable.

## Consequence for ADR 0007

Its noise floor stands. The activation path did not change, and the measured
±2/500 says the two quantization schemes sit inside that floor rather than
outside it — which is itself a small confirmation of the number.

## What the residual is not

Not static alphas (this ADR). Not segmentation (ADR 0013). Not `nmt_nfkc` on
the available evidence: es→en has the accent-heavy source and the *smaller* gap
(−0.42) while en→de has a near-ASCII source and the larger one (−1.79); if
normalization were the cause that pattern would invert.

The gap is per-sentence and larger for en→de. It remains open.
