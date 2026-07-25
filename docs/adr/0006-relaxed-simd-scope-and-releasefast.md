# ADR 0006 — What relaxed SIMD is for, and the ReleaseFast number

Status: accepted
Date: 2026-07-24
Milestone: M7

## Part 1 — the relaxed backend deliberately avoids the instruction it looks like it wants

`i16x8.relaxed_dot_i8x16_i7x16_s` is the obvious candidate for `qgemm_i8`. It
is also a trap.

On x86 it lowers to `pmaddubsw`, which **saturates** its `i16` intermediate. The
instruction is only exact when the second operand fits `i7` — that is,
`[-64, 63]`. SPEC §7 and I5 put weights in `[-127, 127]`. So on real weights the
relaxed dot product would produce different numbers on different host CPUs, and
`fizh.relaxed.wasm` would disagree with `fizh.baseline.wasm` in a way that
depends on the phone. I9 says bit-exact determinism for a given artifact and
build; a per-CPU-dependent kernel is the opposite of that.

**Decision:** every integer path in `kernel/relaxed/` is `kernel/simd128/`,
unchanged. Relaxed SIMD is used only where its rounding freedom is harmless: the
`f32` accumulation in `dot` and `axpy`, via `@mulAdd`.

## Part 2 — `@mulAdd` does not currently emit a relaxed instruction

Measured, not assumed. Both artifacts were disassembled and scanned for opcodes
in the relaxed range (`0xFD 0x100`–`0x115`):

    fizh.baseline.wasm   86611 bytes   relaxed opcodes: none
    fizh.relaxed.wasm    86992 bytes   relaxed opcodes: none

Zig's default float mode is strict, so `@mulAdd` becomes `llvm.fma`, and LLVM's
WebAssembly backend will not select `f32x4.relaxed_madd` for a strict FMA —
`relaxed_madd` is permitted but not required to fuse, which is exactly what a
strict FMA forbids. It expands to a multiply and an add instead. (Usefully, it
does *not* fall back to an `fmaf` libcall, so the import count stays at zero.)

**Consequence:** today `fizh.relaxed.wasm` is numerically identical to
`fizh.baseline.wasm`, and T1 confirms it. The variant is retained because SPEC
§3 specifies it and because it is the seam through which relaxed instructions
arrive — either when Zig grows a way to request contraction locally, or when a
future kernel uses `relaxed_laneselect` or the relaxed min/max, neither of which
has the `pmaddubsw` problem.

This also changed how the probe is built. `src/probe.zig` was compiled with
`+relaxed_simd` and `@mulAdd`, and came out containing no relaxed opcode — a
probe that validates on every browser, which would route every device to the
relaxed build. `tools/make_probe.zig` writes the 92-byte module out by hand
instead, and `zig build check` verifies the opcode is present. A compiler is the
wrong tool for a program whose entire purpose is to contain one instruction.

## Part 3 — the ReleaseFast delta

SPEC §3: "Ship build is `ReleaseSafe` — assertions stay on. Measure the
`ReleaseFast` delta at M7; the number goes in an ADR either way."

Measured with `zig build bench` and `zig build bench-fast` against a
`--selftest --big` artifact at SPEC §4.3 scale (`d_model=256`, `ffn=1536`,
`n_enc=6`, `n_dec=2`, `vocab=32000`, 19.66 MB). Three runs each, x86-64 desktop,
`simd128` backend. **Not the reference device** — SPEC §14 pins a 2022-class
mid-tier Android and says CI numbers are for trend detection only.

| Metric | ReleaseSafe | ReleaseFast | Delta | §14 budget |
|---|---|---|---|---|
| Warm p50, 12-token | 13.5–13.8 ms | 5.3–5.5 ms | **2.5× faster** | 80 ms |
| Warm p50, 120-token | 150–152 ms | 70.4–70.9 ms | **2.1× faster** | 800 ms |
| Cold start | 5.6–6.3 ms | 16.3–28.0 ms | 2.7–4.6× *slower* | 300 ms |

**Decision: keep shipping ReleaseSafe.** The 2.5× is real and it is affordable:
ReleaseSafe meets the warm p50 budget with 5.8× of headroom and the p99 budget
with 5.3×. The assertions are not overhead the project is tolerating, they are
the mechanism SPEC §11 relies on — an invariant violation has to kill the worker
rather than produce a quietly wrong translation, and `unreachable` in
`ReleaseFast` does neither.

The cold-start inversion is not explained. It is most likely an artifact of the
harness — `std.process.Init.gpa` selects a different allocator by optimization
mode, and cold start is dominated by first-touch page faults over ~28 MB — not
of the runtime. It is inside budget by a factor of fifty either way, and it
should be re-measured on the reference device before anyone reasons about it
further.

## Revisit when

- Zig gains a scoped way to permit contraction, so `@mulAdd` can reach
  `relaxed_madd` without `@setFloatMode(.optimized)`, which SPEC §3 forbids in
  `kernel/` and `graph/`.
- The reference device is pinned and T5 runs on it. A 2.5× gap matters much more
  at 80 ms of budget on an A55 than at 13 ms on a desktop, and if ReleaseSafe
  misses the budget there, this ADR is the thing to reopen.
