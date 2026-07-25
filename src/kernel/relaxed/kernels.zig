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
const simd = @import("../simd128/kernels.zig");

pub const name = "relaxed";
pub const Act = ref.Act;

const flanes: usize = 4;
const F32x = @Vector(flanes, f32);

/// `@mulAdd` on a `@Vector` lowers to `f32x4.relaxed_madd` under
/// `+relaxed_simd`. On a target without it, LLVM expands to a multiply and an
/// add rather than calling `fmaf`, so no module import appears — `zig build
/// check` is what keeps that claim true.
pub fn dot(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);
    assert(a.len > 0);

    var acc: F32x = @splat(0);
    var i: usize = 0;
    while (i + flanes <= a.len) : (i += flanes) {
        const av: F32x = a[i..][0..flanes].*;
        const bv: F32x = b[i..][0..flanes].*;
        acc = @mulAdd(F32x, av, bv, acc);
    }
    var tail: f32 = 0;
    while (i < a.len) : (i += 1) tail += a[i] * b[i];
    return (acc[0] + acc[1]) + (acc[2] + acc[3]) + tail;
}

pub fn axpy(dst: []f32, src: []const f32, s: f32) void {
    assert(dst.len == src.len);
    assert(std.math.isFinite(s));

    const vs: F32x = @splat(s);
    var i: usize = 0;
    while (i + flanes <= dst.len) : (i += flanes) {
        const d: F32x = dst[i..][0..flanes].*;
        const v: F32x = src[i..][0..flanes].*;
        dst[i..][0..flanes].* = @mulAdd(F32x, v, vs, d);
    }
    while (i < dst.len) : (i += 1) dst[i] += src[i] * s;
}

// The rest is SIMD128's, including every integer path.
pub const qgemm = simd.qgemm;
pub const qgemv = simd.qgemv;
pub const quantizeRows = simd.quantizeRows;
pub const residualAdd = simd.residualAdd;
pub const layerNorm = simd.layerNorm;
pub const activation = simd.activation;
pub const ssruGate = simd.ssruGate;
pub const softmax = simd.softmax;
pub const argmax = simd.argmax;
pub const embedGather = simd.embedGather;
pub const gatherRows = simd.gatherRows;
