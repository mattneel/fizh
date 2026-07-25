# ADR 0013 — Rejoin fidelity, and how to read a segmentation number

Status: decided
Date: 2026-07-25
Milestone: M5
Amends: ADR 0011

## The bug

`graph/pass.zig` joined translated sentences with a single space. bergamot
preserves the source separator byte-for-byte. `cat -A` on identical input:

    bergamot   One·thing·happened.··Two·spaces·before·this.├──┤Tab·before·this.
    fizh       One·thing·happened.·Two·spaces·before·this.·Tab·before·this.

Two spaces collapsed to one; a tab collapsed to a space. chrF++ counts word
2-grams, so every seam that differs generates spurious n-grams, and the cost
scales with sentence count.

Fixed: the separator is copied from `in[prev_end .. span.start]` verbatim. The
splitter already trims spans, so that slice is exactly the original whitespace.

## The measurement mistake, which was worse

The previous report concluded segmentation was closed because multi-sentence
lines "scored closer". That read **absolute** chrF++. Longer segments score
higher on chrF++ regardless of quality; the *deltas* against bergamot had
widened, in both directions, roughly proportionally.

The right instrument separates the splitter from everything else: translate
each sentence individually, join with the same rule, and score that against
what each engine produces when it splits for itself. The difference is the
segmentation cost, with per-sentence quality held constant.

100 paragraphs of four FLORES sentences:

| | es→en | en→de |
|---|---|---|
| delta with perfect (gold) splits | −0.59 | −2.04 |
| delta with each engine splitting | −0.59 | −2.26 |
| **fizh's segmentation cost** | **0.00** | **−0.23** |
| bergamot's segmentation cost | 0.00 | 0.00 |

So the widening from −0.42 (single) to −0.59 (paragraph) on es→en is not
segmentation at all — it is the same per-sentence gap measured over longer
segments. On en→de there is a real but small residual splitter cost of 0.23.

One trap found on the way: the paragraph corpus was built with `" ".join()`
over FLORES lines, **20 of which carry trailing whitespace**. That produced
genuine double spaces in the source, which fizh now correctly preserves — so
fizh's output legitimately differs from the naive join, and the naive join was
the thing that was wrong.

## Rules this leaves behind

- **Report deltas against a reference on identical input, never absolute
  chrF++.** Absolute scores move with segment length.
- **Isolate segmentation by holding per-sentence output constant.** Comparing a
  paragraph score to a sentence score measures both at once and attributes the
  sum to whichever one you were thinking about.
- A corpus built by joining lines inherits whatever whitespace those lines end
  with. Check before concluding the runtime is wrong.
