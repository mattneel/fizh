# ADR 0026 — Check at the boundary, not per element

Status: decided
Date: 2026-07-25
Milestone: M11
Amends: SPEC §3 build mode, SPEC §14

## The question that should have been asked first

fizh measured ~3x slower than bergamot on x86_64. Eight work orders of analysis
went into the kernels — the dot instruction, register blocking, relaxed SIMD —
before anyone asked why the shipped build is `ReleaseSafe` while bergamot ships
Emscripten `-O3`.

Measured, native, `es-en`, with `tools/profile.zig`:

| source tokens | ReleaseSafe | ReleaseFast | ratio |
|---|---|---|---|
| 24 | 12.97 ms | 4.90 ms | **2.65x** |
| 164 | 109.28 ms | 48.16 ms | **2.27x** |

**The comparison was a bounds-checked build against an optimized one.** That is
essentially the entire gap, and it is not a kernel property.

## Why not simply ship ReleaseFast

Because SPEC §3 chose `ReleaseSafe` for a real reason: fizh parses artifacts it
did not produce, and SPEC §11 promises that *any* byte sequence yields a status
rather than a trap. Turning off safety program-wide would put the load path —
the one part that touches untrusted input — on the honour system.

## Decision: unchecked interiors, checked boundaries

`@setRuntimeSafety(false)` in exactly **two** functions: `qgemmTile` and
`dotI8`. These are loop interiors whose every bound is asserted by the entry
points that call them.

| | 164 source tokens |
|---|---|
| ReleaseSafe, as shipped before | 109.28 ms |
| ReleaseSafe, two interiors unchecked | **54.18 ms** |
| whole-program ReleaseFast | 48.16 ms |

**2.02x**, and within 12% of an optimized build, while every other function and
the entire load path stay checked.

`qgemv` deliberately keeps its safety on even though it is the decoder's hot
path. It is an *entry point*: its asserts are the contract that makes `dotI8`'s
unchecked indexing sound. Keeping them costs 1.8% (54.18 ms against 53.21 with
`qgemv` unchecked too) and buys the property that the contract is enforced
rather than assumed.

That is the general shape and it is worth stating as a rule: **check once at the
boundary, not once per element.** It is the same principle SPEC §11 already
applies to `format.zig` — validation happens at one audited edge, and the
interior trusts it — extended from artifact bytes to array indices.

## What this does not change

- **Bit-exactness.** Removing a bounds check does not change arithmetic. T1
  passes unchanged, `ref/` still agrees with `vector/`, and the golden
  translations are identical.
- **The load path.** `format.zig`, `repack.zig`, the tokenizer and the arena
  carving are all still fully checked. A corrupt artifact still gets a status.
- **The wasm budgets.** 33,180 bytes gzipped against 204,800; still 0 imports.

## The gate this cost us

`standardOptimizeOption` with a `preferred_optimize_mode` silently swallows
`--release=fast`. The first ReleaseSafe/ReleaseFast comparison therefore
returned *identical* numbers for both, which is a plausible result — "safety is
free" — and was very nearly believed.

That is the **fourth** measurement gate in this project to fail silently, each
producing a plausible number rather than an error:

| | ADR |
|---|---|
| a synthetic artifact with the wrong decoder architecture | 0008 |
| a decode loop that terminated at step 0, so §14 timed nothing | 0016 |
| a reference engine running with no shortlist | 0018 |
| `--release=fast` swallowed, so ReleaseSafe reported itself as fast | this one |

All four were detectable from inside the harness. So the rule, now in SPEC §13:
**a benchmark asserts its own configuration, or it is not a measurement.**
`zig build bench` and `zig build profile` read the optimize mode, backend, lane
count and safety state *from the runtime module* and print them; `profile`
takes `--expect-mode` and refuses to run on a mismatch.

It earned itself immediately. Its first run reported "runtime safety on" for a
build whose interiors were unchecked, because the detection asked
`std.debug.runtime_safety` — which answers a different question, the build
mode's. Zig cannot introspect `@setRuntimeSafety`, so the flag is now
*declared* beside the calls that consume it and the two cannot drift. A gate
that catches its own author's error on day one is the right kind of gate.

It also surfaced that this machine has **64 integer lanes**, i.e. AVX-512, not
the 32 an earlier probe reported — which silently changes what every native
number on it means.

## The honest read on the 3x

With this and the row blocking of the previous commit, fizh's x86 position is
roughly **1.3–1.4x** bergamot rather than 3x — recomputed from the desktop
three-engine run's own fit, not measured directly, so it needs a re-run to
confirm.

The remaining lever measured but not taken is `i32x4.dot_i16x8_s` via a C shim,
worth 1.32x on the inner loop (ADR 0025).

And the number that decides invariant I1 is still not taken: **bergamot on ARM**.
x86 is intgemm's home turf, with AVX2 and VNNI paths that do not exist on a
phone. A 3x deficit there was never what I1 predicted, and a 1.3x one is not
what it predicted either — the prediction was about ARM, and the benchmark page
can now measure it.
