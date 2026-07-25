# ADR 0027 — Throughput parity on x86, and what is left is per-call

Status: decided
Date: 2026-07-25
Milestone: M11

## The measurement

`bergamot-translator`'s own wasm engine and fizh, run from the same page, on the
same machine, over the same corpus, reading the same upstream `es-en` weights.
Win64, Chrome 150, 32 cores, foreground, page visible throughout. 400 runs on
the short case, 40 and 60 on the others, 10 warmup each.

| | fizh | bergamot | ratio |
|---|---|---|---|
| 12-token p50 | 31.7 ms | 21.1 ms | 1.50× |
| 120-token p50 | 222.9 ms | 208.3 ms | **1.07×** |
| 8-sentence paragraph | 139.5 ms | 88.4 ms | 1.58× |
| cold start | **20.5 ms** | 128.8 ms | **0.16×** |
| engine | **102 KiB** | 5,120 KiB | **0.020×** |
| peak linear memory | **86 MiB** | 522 MiB | **0.16×** |

Both engines are single-threaded (ADR 0025), so this is per-core.

## The 3× is gone, and the shape says why

An earlier reading put fizh at ~3×. That number came from a backgrounded tab, a
different model, and a build whose loop interiors were bounds-checked against
bergamot's `-O3`. With those corrected (ADR 0026, and the row blocking before
it), the same comparison is 1.07–1.58×.

More useful than the ratio is its shape. Fitting both engines across the same
two points, on the token axis:

| | ms/token | intercept |
|---|---|---|
| fizh | 1.3657 | −1.1 ms |
| bergamot | 1.3371 | −11.0 ms |

**Per-token throughput ratio: 1.021.** Both fits carry the same curvature —
same architecture, same attention — so the *slope ratio* is robust even where
the intercepts are not (ADR 0025 §4).

Reading it as a gap instead, across three input sizes:

    paragraph sentence  ~18 tok    6.39 ms
    12-word              24 tok   10.60 ms
    120-word            164 tok   14.60 ms

which fits **9.9 ms + 0.0286 ms/token**. The per-token term is 2.1% of
bergamot's rate. **Essentially all of the remaining difference is a fixed ~10 ms
per call.**

That is the opposite conclusion from every earlier reading, and it is good news:
the `@Vector` kernels match hand-tuned register-blocked intgemm on throughput.
What does not match is something that happens once per call.

## Where the 10 ms is not

Natively the same phases total **0.09 ms** at 24 tokens (`zig build profile`),
so this is wasm-specific rather than algorithmic. Ruled out by inspection:

- `shortlist.build` runs once per sentence and profiles at 0.025–0.086 ms
  natively, including its `gatherRows`.
- `project`'s shortlisted `qgemv` runs per decode step, so it is in the slope,
  not the intercept.
- Arena zeroing is in `fizh_init`, not per call.

A plausible remaining candidate is fixed weight traffic: the encoder streams
~10.6 MB of weight matrices per call regardless of source length, because for
small `M` the matmuls are bound by reading B rather than by the arithmetic. That
is a hypothesis and it is **not** measured — the project has been wrong four
times by reasoning where it could have measured.

So `web/bench.js` gains a **2-word floor case**. A two-point fit cannot separate
a per-call floor from curvature; a low anchor reads the floor directly. The next
run answers it.

## What this does and does not say about I1

It says nothing. x86 is intgemm's home turf: it has AVX2 and VNNI paths that do
not exist on ARM, while fizh's `@Vector` code lowers to NEON on ARM exactly as
it lowers to AVX here. **I1 predicts fizh does relatively better on ARM, and
that is still unmeasured.**

What the run does establish independently of I1 is that fizh wins the axes it
was actually designed for: **6.3× faster cold start, a 50× smaller engine, and
6.1× less linear memory** — on a phone, plausibly the axes that decide whether
the thing runs at all.
