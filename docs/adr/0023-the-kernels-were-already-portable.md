# ADR 0023 — The kernels were already portable, and the name said otherwise

Status: decided
Date: 2026-07-25
Milestone: M11 (targets)
Amends: SPEC §5 layout, SPEC §8, invariant I9

## The audit

Before adding a target, count what is actually target-specific:

| file | code lines | wasm-specific |
|---|---|---|
| `vector/kernels.zig` (was `simd128/`) | 245 | **0** |
| `ref/kernels.zig` | 287 | **0** |
| `math.zig` | 257 | **0** |
| `relaxed/kernels.zig` | 46 | **0** |
| `backend.zig` | 35 | **3** |
| **total** | **870** | **3** |

**867 of 870 lines are target-neutral.** No wasm builtins, no `@wasmMemorySize`,
no inline assembly, no target branches. The three lines are feature detection:

```zig
const is_wasm = builtin.cpu.arch.isWasm();
const has_relaxed = is_wasm and std.Target.wasm.featureSetHas(builtin.cpu.features, .relaxed_simd);
const has_simd128 = is_wasm and std.Target.wasm.featureSetHas(builtin.cpu.features, .simd128);
```

So `simd128/` is renamed **`vector/`**. The old name encoded a constraint that
was never real, and every future reader would have believed it.

## The relaxed path is not what it looks like

The obvious plan for a native port is per-target siblings of the relaxed dot
product: `sdot` on ARMv8.2+, `vpdpbusd` on x86 with VNNI, plus runtime
detection for both.

**None of that is needed, because fizh does not use the relaxed dot product.**
ADR 0006 rejected `i16x8.relaxed_dot_i8x16_i7x16_s` deliberately: on x86 it
lowers to `pmaddubsw`, which saturates its `i16` intermediate, so the relaxed
build would disagree with the baseline build *depending on the host CPU*. The
integer path is `vector/`'s on every backend and every target.

What `relaxed/` actually owns is two functions, `dot` and `axpy`, both built on
**`@mulAdd`** — which is already portable. It becomes `f32x4.relaxed_madd` under
`+relaxed_simd`, an FMA on aarch64 and on x86 with FMA, and a multiply plus an
add anywhere else. There are no per-target siblings to write.

That collapses the native port from "per-target kernels" to CI wiring.

## Lane width comes from the target, except where it cannot

`std.simd.suggestVectorLength(T)` is the mechanism, and it reports:

| target | i8 | i32 | f32 |
|---|---|---|---|
| x86_64 + AVX2 | 32 | 8 | 8 |
| aarch64 | 16 | 4 | 4 |
| x86_64 baseline | 16 | 4 | 4 |
| **wasm32** | **null** | null | null |

**Integer lanes take the suggestion**, with `orelse 16` for wasm. This is free
of consequence: SPEC §7 accumulates in `i32`, integer addition is associative,
so any lane count reduces to the identical sum. `T1 qgemm is bit-exact across
backends` covers it, and it passes at 32 lanes on this machine and 16 on
aarch64.

**Float lanes stay fixed at 4.** Float addition is not associative, so lane
count fixes reduction order and reduction order fixes the last bits. Taking the
suggestion would make a native build round differently from a wasm one — the
same input, two answers.

The measurement is what settles it: the suggestion is **4 on aarch64 and
unavailable on wasm**. Only an AVX2 desktop would ever widen the float path, and
no phone would. Trading cross-target reproducibility for a faster development
machine is a bad trade at any price, and this one is free to decline.

## I9, scoped

I9 promised bit-exact determinism "for a given artifact and build", written when
there was one target. It now reads:

- **Integer paths are bit-exact across every target.** Guaranteed by
  associativity, checked by T1, and not contingent on lane width.
- **Float paths are bit-exact across every target too**, because the float lane
  count is fixed rather than suggested. This is a choice, and the cost is that
  `vector/` will not use AVX2's 256-bit float registers.

If a measurement ever shows wider float vectors mattering on a *deployment*
target, the decision changes and per-target float goldens come with it. Nothing
has measured that, and the deployment targets are ARM and wasm, where the
suggestion is 4 anyway.

## What was verified, and what was not

| target | built | suite run |
|---|---|---|
| `wasm32-freestanding` (baseline, relaxed) | yes | yes, on the desktop and on Android |
| `aarch64-linux` | yes | **yes — 113/113 under qemu** |
| `x86_64-linux` | yes | yes, 113/113 |
| `x86_64-windows` | yes | yes, 113/113 |
| `aarch64-macos` | **no** | no — needs the macOS SDK |
| `x86_64-macos` | **no** | no — same |

The aarch64 run is the one that matters, because it is the case where the
hardware differs most from what the kernels were developed against, and it
includes T1: `ref/` agrees with `vector/` and `relaxed/` on NEON. **qemu is not
silicon**, and this ADR does not claim it is — it emulates NEON faithfully
enough to prove the lowering compiles and agrees, not that it is fast.

macOS is not "supported"; it does not build here for want of an SDK, and
claiming a target on the strength of a cross-compile that never ran is the
thing this audit exists to avoid.
