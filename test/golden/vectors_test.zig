//! SPEC §13 T0 — per-op golden vectors from `tools/reference.py`.
//!
//! The oracle computed these in float64. The kernels compute them in float32
//! with hand-written transcendentals (ADR 0003), so the tolerances below are
//! the honest gap between those two things — except where they are zero, and
//! those are the interesting cases: quantization and the integer matmul agree
//! with the oracle exactly, and are asserted to.

const std = @import("std");
const testing = std.testing;

const backend = @import("../../src/kernel/backend.zig");
const format = @import("../../src/model/format.zig");
const math = @import("../../src/kernel/math.zig");
const v = @import("vectors.zig");

fn worstRel(got: []const f32, want: []const f32) f32 {
    std.debug.assert(got.len == want.len);
    var scale: f32 = 0;
    for (want) |x| scale = @max(scale, @abs(x));
    if (scale == 0) scale = 1;

    var worst: f32 = 0;
    for (got, want) |a, b| worst = @max(worst, @abs(a - b));
    return worst / scale;
}

fn check(op: []const u8, got: []const f32, want: []const f32, limit: f32) !void {
    const rel = worstRel(got, want);
    if (rel > limit) {
        std.debug.print("T0 {s}: {e:.3} over limit {e:.3}\n", .{ op, rel, limit });
        return error.GoldenDrift;
    }
}

test "T0 layer_norm" {
    inline for (backend.all) |B| {
        var out: [v.layer_norm_in.len]f32 = undefined;
        B.layerNorm(&out, &v.layer_norm_in, &v.layer_norm_gain, &v.layer_norm_bias, 1, v.layer_norm_in.len, v.layer_norm_in.len, 1e-9);
        try check("layer_norm " ++ B.name, &out, &v.layer_norm_out, 1e-6);
    }
}

test "T0 softmax" {
    inline for (backend.all) |B| {
        var out: [v.softmax_in.len]f32 = undefined;
        @memcpy(&out, &v.softmax_in);
        B.softmax(&out);
        try check("softmax " ++ B.name, &out, &v.softmax_out, 1e-6);

        var total: f32 = 0;
        for (out) |x| total += x;
        try testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);
    }
}

test "T0 activations" {
    inline for (backend.all) |B| {
        var out: [v.act_in.len]f32 = undefined;

        @memcpy(&out, &v.act_in);
        B.activation(&out, .relu);
        // relu is exact: it is a comparison, not an approximation.
        try testing.expectEqualSlices(f32, &v.act_relu, &out);

        @memcpy(&out, &v.act_in);
        B.activation(&out, .gelu);
        try check("gelu " ++ B.name, &out, &v.act_gelu, 1e-6);

        @memcpy(&out, &v.act_in);
        B.activation(&out, .swish);
        try check("swish " ++ B.name, &out, &v.act_swish, 1e-6);
    }
}

test "T0 quantize matches the oracle exactly" {
    // SPEC §7 is an ABI, not an approximation: if fizh and the converter round
    // differently, every weight in every artifact is off by a step.
    inline for (backend.all) |B| {
        var out: [v.quantize_in.len]i8 = undefined;
        var scale: [1]f32 = undefined;
        B.quantizeRows(&out, v.quantize_in.len, &scale, &v.quantize_in, v.quantize_in.len, 1, v.quantize_in.len);

        try testing.expectEqualSlices(i8, &v.quantize_out, &out);
        try testing.expectApproxEqRel(v.quantize_scale[0], scale[0], 1e-7);
        for (out) |q| try testing.expect(q != -128); // I5
    }
}

test "T0 qgemv matches the oracle to float32 rounding" {
    // The accumulation is integer and therefore exact; the only difference from
    // the oracle is that the final dequantization happens in f32 rather than
    // f64.
    inline for (backend.all) |B| {
        var out: [v.qgemv_n]f32 = undefined;
        B.qgemv(&out, &v.qgemv_a, v.qgemv_a_scale, &v.qgemv_w, v.qgemv_k, &v.qgemv_w_scales, null, v.qgemv_k, v.qgemv_n);
        try check("qgemv " ++ B.name, &out, &v.qgemv_out, 1e-7);
    }
}

test "T0 positional encodings" {
    var out: [v.pos_enc_16x4.len]f32 = undefined;
    format.fillPositional(&out, v.pos_enc_d, v.pos_enc_steps);
    try check("pos_enc", &out, &v.pos_enc_16x4, 1e-7);
}

test "T0 exp" {
    var out: [v.exp_in.len]f32 = undefined;
    for (&out, v.exp_in) |*o, x| o.* = math.exp(x);

    // Relative per element, not relative to the largest: `exp` spans eighteen
    // orders of magnitude here and a whole-vector norm would hide everything
    // below the top one.
    for (out, v.exp_out) |got, want| {
        try testing.expectApproxEqRel(want, got, 2e-6);
    }
}
