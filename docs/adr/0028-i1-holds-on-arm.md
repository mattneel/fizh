# ADR 0028 — I1 holds, and the asymmetry is 1.84×

Status: decided
Date: 2026-07-25
Milestone: M11

## The claim, and what would have falsified it

Invariant **I1**, written before a line of code: *fizh exists because Mozilla's
engine is intgemm-based and x86-tuned, while phones are ARM.*

That is a claim about an **asymmetry**, not about absolute speed. It predicts
bergamot degrades on ARM relative to x86 while fizh does not, because intgemm
has hand-written AVX2 and VNNI paths that have no NEON equivalent, whereas
fizh's `@Vector` kernels lower to NEON exactly as they lower to AVX.

It went eleven milestones unmeasured. On x86 alone it looked bad: ADR 0027 put
fizh at throughput *parity* there, which is neither a confirmation nor a
refutation — x86 is intgemm's home turf and I1 never claimed a win on it.

## The measurement

Same page, same corpus, same upstream `es-en` weights, both engines
single-threaded, foreground, page visible throughout.

**Android 10, armv81, 8 cores, Chrome 150:**

| | fizh | bergamot | |
|---|---|---|---|
| 12-token p50 | 27.0 ms | 31.2 ms | fizh **1.16×** faster |
| 12-token *steady* p50 | 21.8 ms | 31.3 ms | fizh **1.44×** faster |
| 120-token p50 | 197.8 ms | 339.0 ms | fizh **1.71×** faster |
| 8-sentence paragraph | 120.1 ms | 146.7 ms | fizh **1.22×** faster |
| cold start | 52.0 ms | 266.5 ms | fizh **5.12×** faster |
| peak linear memory | 86 MiB | 522 MiB | fizh **6.1×** less |

**fizh wins every case on the phone.**

## The asymmetry, which is the actual claim

Per-token throughput, fitted across the same two points on the token axis, on
both machines:

| | fizh | bergamot | |
|---|---|---|---|
| x86_64 | 1.3657 ms/tok | 1.3371 ms/tok | parity (fizh 0.98×) |
| **aarch64** | **1.2200 ms/tok** | **2.1986 ms/tok** | **fizh 1.80× faster** |

Read down the columns rather than across:

- **bergamot is 1.64× slower per token on ARM than on x86.**
- **fizh is 0.89× — slightly *faster* on ARM than on x86.**

The ratio of those is **1.84×**, and it is the number I1 predicted the shape of
without predicting the size. A portable `@Vector` kernel carries its performance
across architectures; a hand-tuned x86 kernel does not.

## Caveats, stated because they are load-bearing

**Engine order biases the short case.** fizh ran first and absorbed the DVFS
ramp: its burst p50 is 32.2 ms against a steady 21.8, while bergamot's burst and
steady are flat at 31.0 and 31.3 because the CPU was already warm by the time it
ran. That makes fizh's *p50* pessimistic and the **steady** comparison the fair
one. It is also a harness defect, and it is fixed rather than annotated —
`web/bench.js` now warms the device before the first engine runs.

**Two devices, one run each.** The x86 and ARM machines differ in more than ISA
— different browsers on different OSes, different memory systems. The
*within-device* ratios are sound; the cross-device per-token comparison
(bergamot 1.64× worse on ARM) mixes ISA with everything else about the two
machines. What survives that objection is the within-device flip: parity on one,
1.80× on the other, same code and same corpus on both.

**One language pair.** `es-en`, `d=384`. The wide architecture and the pivot
case have not been run on the phone.

## What it settles

The premise holds, and the desktop number that looked discouraging was
measuring the wrong machine. fizh is at parity with hand-tuned intgemm on
intgemm's home turf and ahead of it by 1.44–1.71× where the product actually
runs — on top of a 5.1× faster cold start, a 50× smaller engine, and 6.1× less
memory.

The remaining ~10 ms of fixed per-call cost (ADR 0027) is now the *only* axis
where fizh trails on any machine, and on ARM it is small enough that fizh wins
the 12-token case anyway.
