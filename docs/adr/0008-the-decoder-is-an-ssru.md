# ADR 0008 — The decoder is an SSRU, not self-attention

Status: decided
Date: 2026-07-25
Milestone: M5 (rework)
Supersedes: SPEC §8's `attn_decode` row, SPEC §4.2's `self_kv` region

## Context

SPEC §8 lists `attn_decode` — "batch 1, append to `self_kv`, attend over prefix"
— and SPEC §4.2 carves `self_kv` at `2 · n_dec_layers · max_tgt_tokens ·
d_model · 4`. The decoder was built to that.

The real Bergamot `es-en` tiny model does not contain those weights. Its 195
tensors include

    decoder_l{1,2}_rnn_W       int8 [256,256]
    decoder_l{1,2}_rnn_Wf      int8 [256,256]
    decoder_l{1,2}_rnn_bf      f32  [256]
    decoder_l{1,2}_rnn_ffn_ln_{scale,bias}

and **no `decoder_lN_self_W{q,k,v,o}` at all**. The config embedded in the
artifact as `special:model.yml` says `dec-cell: ssru` and
`transformer-decoder-autoreg: rnn`.

Bergamot replaced decoder self-attention with a Simpler Simple Recurrent Unit
for speed — it is the central trick of Kim et al., *From Research to Production
and Back* (2019), which is the paper the Bergamot student models come from.
SPEC §8 describes an architecture Mozilla does not ship.

## What SSRU computes

From `browsermt/marian-dev`, `src/rnn/cells.h`, class `SSRU`:

```cpp
x = dot(input, W_);                    // no bias
f = affine(input, Wf_, bf_);
auto nextCellState = highway(cellState, x, f);
auto nextState = relu(nextCellState);
```

with `highway(y, x, t) = σ(t)·y + (1 − σ(t))·x`. So, per layer, per step:

    f  = Wf·xₜ + bf
    cₜ = σ(f) ⊙ cₜ₋₁ + (1 − σ(f)) ⊙ (W·xₜ)
    hₜ = relu(cₜ)
    out = LayerNorm(xₜ + hₜ)          // post-norm, `transformer-postprocess: dan`

## Decision

`graph/decoder.zig` implements `ssruStep` in place of `selfAttention`.
`kernel/*/kernels.zig` gains one op, `ssruGate`. Cross-attention is untouched
and is now the only attention in the decode loop.

## Consequences — all of them good

**The arena shrinks by three orders of magnitude.** `self_kv` was 1,572,864
bytes at the SPEC §4.3 configuration. `ssru_state` is `n_dec_layers · d_model ·
4` = **2,048 bytes**. Measured shared scratch went from ~8.2 MB to **6.74 MB**.

**Per-step work stops growing with the output.** Self-attention at step *t*
scores *t* keys; SSRU does two `qgemv`s and a pointwise gate regardless. The
decode loop is now O(1) per token instead of O(t), which is why the 120-token
p50 is 126 ms rather than something that curves upward.

**Two matmuls per layer instead of four.** No Q, K, V and output projection —
just `W` and `Wf`.

**The state must be cleared per pass.** A KV cache is written before it is read
at every position; a recurrent cell is not. `graph/pass.zig` zeroes
`ssru_state` before each decode, or a message would inherit the last one's
final state — a bug that would only show up as a quality drift on the second
translation, which is the worst kind.

## What this cost, and the lesson

M5 was built, tested and reported complete against synthetic artifacts whose
shape came from the SPEC rather than from a real model. The synthetic builder
generated `dec.{i}.sa.*` because the SPEC said so, the loader loaded it, and
ninety-eight tests passed. None of that was evidence about Bergamot.

SPEC §6 had already set the M2 task — "confirm ... against a real artifact" —
and it went unanswered until after M8. Ten minutes with the real file would have
caught this before the decoder was written. **The artifact is the
specification; the SPEC is a description of it, and descriptions can be wrong.**

## If a self-attention decoder ever ships

Marian supports `transformer-decoder-autoreg: self-attention`, and
`tools/bergamot.py` refuses such a model by name rather than mis-loading it. The
path back is: restore the `self_kv` region in `arena.zig`, restore `Attn` in
`DecLayer`, and call `attention.decodeStep` against the target-side cache —
`graph/attention.zig` still has exactly that function, because cross-attention
needs it.
