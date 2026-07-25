# ADR 0015 — The decoder starts from a zero vector, not `emb(</s>)`

Status: decided
Date: 2026-07-25
Milestone: M8 (calibration)
Amends: SPEC §8's decode loop

## Context

fizh seeded the decoder with the embedding of `bos_id`, which the converter
sets to the end-of-sequence id because Marian has no separate start token:

```zig
var prev: u32 = ctx.hp.bos_id;      // = eos_id
...
embedStep(ctx, prev, t, x);         // emb(prev) + pos(t)
```

That is the obvious reading of "Marian starts the decoder with EOS", and it is
wrong. Marian does not embed a start token at all. `transformer.h` builds the
decoder input by shifting the *target* embeddings right by one with a zero
fill — `shift(embeddings, {1, 0, 0})` — so step 0 sees the positional encoding
alone against a zero embedding, and step *t* sees `emb(y_{t-1}) + pos(t)`.

## How it was found, which matters more than the fix

The residual against `bergamot-translator` was −0.42 chrF++ on es→en and −1.79
on en→de over 500 FLORES devtest segments. Three prior rounds of work had
failed to explain it, and it had twice been written off as quantization noise.

The measurement that broke it open was **teacher forcing**. Sentence-level
byte-identity and even first-divergence position both conflate the defect with
its own downstream compounding: under greedy decode one wrong token corrupts
every token after it, so any early error looks like a diffuse whole-sequence
disagreement. Feeding bergamot's own output back as the decoder prefix and
scoring only the next-token argmax removes the compounding entirely:

| position | es→en agreement | en→de agreement |
|---|---|---|
| 0 | **0.547** | **0.473** |
| 1 | 0.967 | 0.933 |
| 2–10 | 0.940–0.973 | 0.960–0.987 |
| 11+ | 0.964 | 0.965 |

One position wrong at chance-like rates, every other position at 96%. The
encoder, cross-attention, FFN, layer norms, shortlist and output projection
were all exonerated by the same table, and only three things are unique to
step 0: the start embedding, position index 0, and the zeroed SSRU cell.

Sweeping them over 150 segments, scoring first-token agreement:

| step-0 decoder input | es→en | en→de |
|---|---|---|
| `emb(</s>) + pos(0)` — what fizh did | 0.547 | 0.473 |
| no positional term | 0.060 | 0.040 |
| `emb(</s>) + pos(1)` | 0.560 | 0.493 |
| **zero embedding + pos(0)** | **0.960** | **0.980** |
| all zeros | 0.000 | 0.033 |
| `emb(<unk>) + pos(0)` | 0.747 | 0.193 |
| `emb(</s>)/√d + pos(0)` | 0.940 | 0.960 |

The zero fill lands exactly on the ~0.96 that every later position already
achieved. The unscaled-embedding row is a near miss for the same reason it is
not the answer: dividing by √256 makes the embedding small, which approximates
zero without being it.

## Decision

`graph/decoder.zig` gains `embedStart`, which writes `pos_enc[0..d]` and no
embedding, and the loop carries `?u32` so "no previous token" is a state the
type system knows about rather than a sentinel id:

```zig
var prev: ?u32 = null;
...
if (prev) |id| embedStep(ctx, id, t, x) else embedStart(ctx, x);
```

`tools/reference.py` gets the same change, because an oracle that mirrors the
bug cannot detect it — and did not, for three rounds.

## What it was worth

500 FLORES devtest segments, chrF++ against gold, paired bootstrap over 1000
resamples:

| | before | after | 95% CI after |
|---|---|---|---|
| es→en vs bergamot | −0.42 | **−0.08** | [−0.37, +0.20] |
| en→de vs bergamot | −1.79 | **−0.33** | [−0.74, +0.03] |
| es→de end to end | −1.09 | **−0.23** | [−0.57, +0.10] |
| es→en, 4-sentence paragraphs | −0.59 | **−0.10** | |
| en→de, 4-sentence paragraphs | −2.26 | **−0.43** | |
| chat register (n=40) | | **+0.86** | [−1.88, +3.53] |

Every interval covers zero. Sentence-level byte-identity with bergamot went
from 21.6% to 46.4% (es→en) and 21.2% to 46.0% (en→de); per-token divergence
hazard fell from 0.0628 to 0.0251 and from 0.0777 to 0.0260.

The positional signature that started the hunt is gone. Divergence hazard by
relative position, after:

    es→en  0.172  0.157  0.158  0.150  0.072
    en→de  0.178  0.151  0.143  0.144  0.102

Flat, which is what compounding rounding differences are supposed to look like.
Before the fix the first bucket was 0.528 and 0.606.

## Consequences

- `bos_id` is now unused by the decode loop. It stays in the format header
  (§6) because it is real metadata about the artifact, and the converter still
  validates it, but nothing reads it to seed anything.
- The `known_divergences.tsv` row resolved for cause. Its recorded diagnosis —
  an irreducible step-0 near-tie at a 0.054 margin against ADR 0007's 0.26
  noise floor — was wrong, and the file now carries why, because the next
  near-tie will look identical. **A small margin measured on your own logits
  says your two candidates are close. It says nothing about whether your
  logits are the reference's.**
- One golden translation was re-recorded: `Me llamo Ana y vivo en Madrid.` had
  `I'm Ana and I live in Madrid.` in the golden file, which was fizh's own
  wrong output. It is now `My name is Ana and I live in Madrid.`, matching
  bergamot.
- SPEC §8's decode loop gains the zero-fill.

## The lesson, which is ADR 0008's lesson again

ADR 0008 concluded: *the artifact is the specification; the SPEC is a
description of it, and descriptions can be wrong.* This is the same failure one
level down. "Marian starts the decoder with EOS" is true as a statement about
Marian's *vocabulary* — there is no separate BOS symbol — and false as a
statement about what the decoder embeds. The code followed a sentence about the
model instead of the model.

The generalisable part is the instrument, not the bug. **Greedy decode makes
every early defect look diffuse.** Any metric computed on free-running output —
byte-identity, chrF++, even first-divergence position — measures the defect
convolved with its own blast radius. Teacher forcing deconvolves it, costs one
afternoon, and would have found this in the first round.
