//! kernel/relaxed/kernels.zig — the `+relaxed_simd` variant.
//!
//! Relaxed SIMD's value here is `f32x4.relaxed_madd`: a fused multiply-add with
//! implementation-defined rounding. It costs nothing to ask for and shortens the
//! `f32` accumulation chains that dominate layer norm and attention scoring.
//!
//! What it deliberately does **not** touch:
//!
//! `i16x8.relaxed_dot_i8x16_i7x16_s` looks like the obvious win for `qgemm`, and
//! it is a trap. On x86 it lowers to `pmaddubsw`, which saturates its `i16`
//! intermediate; the instruction is only exact when the second operand fits
//! `i7`, and SPEC §7 / I5 puts weights in `[-127, 127]`. Using it would make the
//! relaxed build disagree with the baseline build on real inputs, in a way that
//! depends on the host CPU. So the integer path here is the SIMD128 one,
//! unchanged and still bit-exact against `ref/`.
//!
//! See `docs/adr/0006-relaxed-simd-scope.md`.

const std = @import("std");
const assert = std.debug.assert;

const ref = @import("../ref/kernels.zig");
const simd = @import("../vector/kernels.zig");

pub const name = "relaxed";
pub const Act = ref.Act;

// `dot` and `axpy` used `@mulAdd` here, on the premise that it lowers to
// `f32x4.relaxed_madd` under `+relaxed_simd`. **It does not, and the premise
// was never checked against the emitted bytes.**
//
// Scanning both shipped artifacts for the relaxed opcode range (0x100..0x115)
// finds *zero* occurrences in either, with `+relaxed_simd` enabled. What LLVM
// emits instead is an expansion — the relaxed binary carries 76 more of one
// f32 opcode and 21 more of another than the baseline — because `@mulAdd` is a
// *correctly rounded* fused multiply-add, and without a hardware instruction to
// lower to it must be emulated rather than relaxed into a multiply and an add.
//
// So the relaxed build was paying for a software FMA to avoid a multiply and an
// add. On the desktop three-engine run it was 5% slower on a short message and
// 24% slower on a long one, scaling with per-token work exactly as a kernel
// regression should. It also meant the artifact-selection logic was choosing
// the slower binary wherever relaxed SIMD is supported.
//
// Until the opcode is actually emitted, every path here is the vector one.
// ADR 0024.
pub const dot = simd.dot;
pub const axpy = simd.axpy;

// The rest is the vector backend's, including every integer path.
pub const qgemm = simd.qgemm;
pub const qgemv = simd.qgemv;
pub const quantizeRows = simd.quantizeRows;
pub const residualAdd = simd.residualAdd;
pub const layerNorm = simd.layerNorm;
pub const activation = simd.activation;
pub const ssruGate = simd.ssruGate;
pub const quantizeRowsWith = simd.quantizeRowsWith;
pub const softmax = simd.softmax;
pub const argmax = simd.argmax;
pub const embedGather = simd.embedGather;
pub const gatherRows = simd.gatherRows;
