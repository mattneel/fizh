# ADR 0024 — The relaxed artifact buys nothing, and the 38 ms was an axis

Status: decided
Date: 2026-07-25
Milestone: M11
Amends: ADR 0006

Two findings from the desktop three-engine run, both of which say the same
thing: **a number that was never checked against the bytes.**

## 1. `f32x4.relaxed_madd` is not emitted, and never was

ADR 0006 scoped `relaxed/` to exactly one instruction. `dot` and `axpy` used
`@mulAdd` on a `@Vector`, on the stated premise that it "lowers to
`f32x4.relaxed_madd` under `+relaxed_simd`".

Scanning both shipped artifacts for the entire relaxed opcode range
(`0x100`–`0x115`) finds **zero occurrences in either binary**, with
`+relaxed_simd` enabled in the target features. The two artifacts carry the same
57 distinct SIMD opcodes and no opcode unique to relaxed.

What LLVM emits instead is an *expansion*: the relaxed binary carried 76 more of
one f32 opcode and 21 more of another. `@mulAdd` is a **correctly rounded** fused
multiply-add, and with no hardware instruction to lower to, it must be emulated
rather than relaxed into a multiply and an add.

So the relaxed build was paying for a software FMA in order to avoid a multiply
and an add. On the desktop run it was **5% slower on a short message and 24%
slower on a long one**, scaling with per-token work exactly as a kernel
regression should — and the artifact-selection logic was choosing it wherever
relaxed SIMD is supported.

`relaxed/` now aliases `vector/` for both functions. The two artifacts are
**byte-identical**.

**That is the finding, stated plainly: the second artifact, the probe module,
and the host-side feature detection currently buy nothing.** They are not
removed, because the machinery is correct and cheap and the instruction may
land; but nothing should be claimed for them until a scan finds the opcode.
ADR 0006's reasoning was sound and its premise was unverified, which is the
same failure as ADR 0019's version selection.

## 2. The 38 ms "fixed cost per call" is mostly an axis artifact

The desktop run was read with a two-point linear fit across the 12-word and
120-word cases, giving fizh 38.1 ms of fixed cost against bergamot's 0.27 — a
140x gap that looked structural.

`tools/profile.zig` (`zig build profile -Dprofile`) measures the phases
directly. Native ReleaseSafe, es-en:

| words | src tokens | total | encoder | decoder | **setup** |
|---|---|---|---|---|---|
| 6 | 13 | 6.95 | 5.70 | 1.23 | **0.030** |
| 12 | 24 | 14.51 | 10.71 | 3.71 | **0.090** |
| 24 | 34 | 21.07 | 15.45 | 5.49 | **0.133** |
| 48 | 72 | 46.44 | 33.78 | 12.43 | **0.237** |
| 96 | 131 | 92.13 | 66.87 | 24.94 | **0.315** |
| 120 | 164 | 119.20 | 86.38 | 32.48 | **0.342** |

**Setup — normalize, tokenize, shortlist build, detokenize — is 0.03 to 0.34 ms,
at most 0.3% of a call, and it grows with length rather than being fixed.** That
measurement does not depend on any fit, and it rules out all three ranked
suspects:

- Arena zeroing is in `fizh_init`, not per call.
- `shortlist.build` is 0.025–0.086 ms.
- Nothing per-model is recomputed per call; `pos_enc` moved into the slot in
  ADR 0020 and is built once at load.

The apparent intercept comes from the x-axis. `message(words)` does not produce
a constant tokens-per-word: it is 2.17 at 6 words and 1.37 at 120, a 37% drop
across the range. Fitting milliseconds against *words* on two points therefore
manufactures a positive intercept out of a curve.

Six-point least squares, native:

| axis | slope | intercept |
|---|---|---|
| vs words | 0.974 ms/word | **+0.37 ms** |
| vs source tokens | 0.741 ms/token | **−4.01 ms** |

Re-fitting the *browser* numbers on the token axis (24 and 164 tokens) moves
fizh-baseline's intercept from **+38.1 ms to +4.6 ms**.

**The decomposition method was right; the axis was wrong.** There is no
structural per-call overhead to remove. The gap to bergamot is per-token, which
is the honest engineering gap: intgemm is hand-tuned register-blocked x86 with
prepared-B layouts, and a portable `@Vector` kernel being a few times off that
is a fair result rather than a defect.

## What this changes about measurement

Report cost against **source tokens**, not words. A words axis is a tokenizer
artifact wearing a performance claim, and it will keep producing spurious
intercepts on any corpus whose tokens-per-word varies with length.
