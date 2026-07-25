# ADR 0016 — Benchmark against a model that terminates

Status: decided
Date: 2026-07-25
Milestone: M8 (calibration)
Amends: SPEC §4.2 `io_dst`, SPEC §14 measurement conditions

## Context

`zig build bench` measured the SPEC §14 budgets against a synthetic artifact
built by `tools/convert.py --selftest --big`: correct §4.3 shape, random
weights. Shape is what the timings depend on, so that looked sound.

It was not. A model with random weights has no reason to ever emit `</s>`, and
which token it *does* emit depends on the decoder's step-0 input. Seeded with
`emb(</s>)` the synthetic model predicted `</s>` immediately and every
translation returned almost nothing. The §14 numbers were measuring an empty
decode loop.

ADR 0015 changed the step-0 input, and the fixture stopped terminating. The
failure surfaced as `out_too_small`, then as this:

| metric | synthetic | real `esen.fzm` | budget |
|---|---|---|---|
| warm p50, 12-token | 90.87 ms | 15.61 ms | 22 ms |
| warm p50, 120-token | 651.55 ms | 128.61 ms | 200 ms |
| warm p50, paragraph | 381.21 ms | 67.63 ms | 100 ms |

The synthetic column is every sentence running to the 384-token step limit. The
real column is the workload. Both are "correct" measurements of different
things, and only one of them is the thing §14 budgets.

The budgets themselves were never wrong — the ~1.5x retightening holds against
the real model with room. They were being compared against a fixture that had
previously measured near-zero work, which is the more dangerous direction: a
budget that passes because nothing ran.

## Decision

**`zig build bench` uses a real Bergamot artifact when one has been fetched.**
`build.zig` checks for `zig-out/esen.fzm` and falls back to building the
synthetic one only so the step works in a bare checkout — where the step name
says so:

    bench  SPEC §14 budgets against a synthetic artifact
           (run tools/fetch-model.sh for real timings)

The synthetic artifact is still built for `trace`, `selfcheck` and `eval`,
which want a fixture with known contents rather than realistic behaviour.

**`io_pivot` and `io_dst` are sized at `2 · max_src_bytes`.** They were
`max_src_bytes`, which asserts that no translation is longer than its source.
German compounding and English article insertion both break that routinely, and
a pivot's English waypoint can exceed both endpoints. A *correct* translation
of a full-length source could return `out_too_small`. Two is room, not a
guarantee: past it the pass still returns `out_too_small` rather than
truncating, which is the one behaviour worth preserving.

Cost is 8 KB against 6.76 MB of shared scratch.

## Consequences

- SPEC §4.2's region table splits `io_src` from `io_pivot`/`io_dst`.
- `arena.io_expansion` is the single place the factor lives, and `bench.zig`
  sizes its own output buffer from it rather than from a literal.
- Every §14 budget passes against the real artifact, on this desktop, with the
  retightened numbers: cold start 6.28 ms / 10, p50 direct 15.61 / 22, p99 long
  132.54 / 200, paragraph 67.63 / 100, scratch 6.76 MB / 10, weights 19.00 MB /
  20. Still a desktop proxy — §14's reference device is a pinned 2022-class
  mid-tier Android and T5 has not run on it.

## The lesson

A fixture that cannot fail the way production fails is not a fixture. This one
could not even *run* the way production runs: it exercised the load path, the
tokenizer and the encoder, then skipped the decode loop entirely, and reported
a number in milliseconds that looked like a pass.

It is the same shape as the defect ADR 0008 records — synthetic artifacts
whose properties came from the SPEC rather than from a real model — and it
survived a milestone longer because its output was a plausible number rather
than a crash. **Prefer a fixture that is real over one that is convenient, and
when you cannot, assert on the property you are relying on** — here, that the
decode loop produced a sequence that ended because the model said so.
