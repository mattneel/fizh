//! SPEC §13 T1 — `ref/` vs `vector/`, max-abs-error per op.
//!
//! This is why I2 exists. It runs in one process with no FFI, so a jump in
//! error at one op localizes a transposed stride in minutes rather than days.
//!
//! Two classes of claim:
//!
//!   *bit-exact*   — the integer paths. SPEC §7 accumulates in `i32`, and
//!                   integer addition is associative, so a vector reduction is
//!                   the same number as a scalar one. Anything but zero error
//!                   here is a bug, not a rounding difference.
//!   *bounded*     — the `f32` paths, where a four-lane association differs
//!                   from a scalar one. The bound is what T1 tracks over time.

const std = @import("std");
const testing = std.testing;

const backend = @import("../src/kernel/backend.zig");
const ref = backend.ref;

const Rng = std.Random.DefaultPrng;

/// Bergamot-ish shapes, shrunk so the suite stays fast.
const d_model: u32 = 256;
const ffn_dim: u32 = 384;
const rows: u32 = 5;

fn randomFloats(gpa: std.mem.Allocator, n: usize, r: std.Random, spread: f32) ![]f32 {
    const buf = try gpa.alloc(f32, n);
    for (buf) |*x| x.* = (r.float(f32) - 0.5) * 2.0 * spread;
    return buf;
}

fn randomWeights(gpa: std.mem.Allocator, n: usize, r: std.Random) ![]i8 {
    const buf = try gpa.alloc(i8, n);
    for (buf) |*x| x.* = @intCast(r.intRangeAtMost(i32, -127, 127));
    return buf;
}

fn randomScales(gpa: std.mem.Allocator, n: usize, r: std.Random) ![]f32 {
    const buf = try gpa.alloc(f32, n);
    for (buf) |*x| x.* = 0.001 + r.float(f32) * 0.02;
    return buf;
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var worst: f32 = 0;
    for (a, b) |x, y| worst = @max(worst, @abs(x - y));
    return worst;
}

fn magnitude(a: []const f32) f32 {
    var m: f32 = 0;
    for (a) |x| m = @max(m, @abs(x));
    return m;
}

/// Reports rather than only asserting: the number is the point of T1.
fn report(op: []const u8, name: []const u8, err: f32, scale: f32, limit: f32) !void {
    const rel = if (scale > 0) err / scale else err;
    if (rel > limit) {
        std.debug.print(
            "T1 FAIL {s}: {s} vs ref, max-abs {e:.3} over magnitude {e:.3} = {e:.3} (limit {e:.3})\n",
            .{ op, name, err, scale, rel, limit },
        );
        return error.DifferentialDrift;
    }
}

test "T1 qgemm is bit-exact across backends" {
    const gpa = testing.allocator;
    var rng = Rng.init(0xa11ce);
    const r = rng.random();

    const k = d_model;
    const n = ffn_dim;
    const a = try randomWeights(gpa, rows * k, r);
    defer gpa.free(a);
    const a_scales = try randomScales(gpa, rows, r);
    defer gpa.free(a_scales);
    const w = try randomWeights(gpa, n * k, r);
    defer gpa.free(w);
    const w_scales = try randomScales(gpa, n, r);
    defer gpa.free(w_scales);
    const bias = try randomFloats(gpa, n, r, 0.5);
    defer gpa.free(bias);

    const want = try gpa.alloc(f32, rows * n);
    defer gpa.free(want);
    ref.qgemm(want, n, a, k, a_scales, w, k, w_scales, bias, rows, k, n);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(f32, rows * n);
        defer gpa.free(got);
        B.qgemm(got, n, a, k, a_scales, w, k, w_scales, bias, rows, k, n);
        // Exactly equal. SPEC §7's `i32` accumulation is what buys this.
        try testing.expectEqualSlices(f32, want, got);
    }
}

test "T1 qgemv is bit-exact across backends" {
    const gpa = testing.allocator;
    var rng = Rng.init(0xbeef);
    const r = rng.random();

    const k = d_model;
    const n = d_model;
    const a = try randomWeights(gpa, k, r);
    defer gpa.free(a);
    const w = try randomWeights(gpa, n * k, r);
    defer gpa.free(w);
    const w_scales = try randomScales(gpa, n, r);
    defer gpa.free(w_scales);

    const want = try gpa.alloc(f32, n);
    defer gpa.free(want);
    ref.qgemv(want, a, 0.01, w, k, w_scales, null, k, n);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(f32, n);
        defer gpa.free(got);
        B.qgemv(got, a, 0.01, w, k, w_scales, null, k, n);
        try testing.expectEqualSlices(f32, want, got);
    }
}

test "T1 qgemv handles a k that is not a multiple of the vector width" {
    const gpa = testing.allocator;
    var rng = Rng.init(0xc0ffee);
    const r = rng.random();

    // 19 and 37 are prime to 16 and to 4: every tail path gets exercised.
    for ([_]u32{ 1, 3, 15, 16, 17, 19, 37, 63, 65 }) |k| {
        const n: u32 = 7;
        const a = try randomWeights(gpa, k, r);
        defer gpa.free(a);
        const w = try randomWeights(gpa, n * k, r);
        defer gpa.free(w);
        const w_scales = try randomScales(gpa, n, r);
        defer gpa.free(w_scales);

        const want = try gpa.alloc(f32, n);
        defer gpa.free(want);
        ref.qgemv(want, a, 0.01, w, k, w_scales, null, k, n);

        inline for (backend.all) |B| {
            const got = try gpa.alloc(f32, n);
            defer gpa.free(got);
            B.qgemv(got, a, 0.01, w, k, w_scales, null, k, n);
            try testing.expectEqualSlices(f32, want, got);
        }
    }
}

test "T1 quantizeRows is bit-exact across backends" {
    const gpa = testing.allocator;
    var rng = Rng.init(0xd00d);
    const r = rng.random();

    const cols = ffn_dim + 3; // deliberately off the lane boundary
    const src = try randomFloats(gpa, rows * cols, r, 4.0);
    defer gpa.free(src);

    const want = try gpa.alloc(i8, rows * cols);
    defer gpa.free(want);
    const want_scales = try gpa.alloc(f32, rows);
    defer gpa.free(want_scales);
    ref.quantizeRows(want, cols, want_scales, src, cols, rows, cols);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(i8, rows * cols);
        defer gpa.free(got);
        const got_scales = try gpa.alloc(f32, rows);
        defer gpa.free(got_scales);
        B.quantizeRows(got, cols, got_scales, src, cols, rows, cols);
        try testing.expectEqualSlices(i8, want, got);
        try testing.expectEqualSlices(f32, want_scales, got_scales);
    }

    // SPEC §7 / I5, checked on the output and not merely intended.
    for (want) |q| try testing.expect(q != -128);
}

test "T1 layerNorm stays within tolerance" {
    const gpa = testing.allocator;
    var rng = Rng.init(0xfeed);
    const r = rng.random();

    const cols = d_model;
    const src = try randomFloats(gpa, rows * cols, r, 3.0);
    defer gpa.free(src);
    const gain = try randomFloats(gpa, cols, r, 1.0);
    defer gpa.free(gain);
    const bias = try randomFloats(gpa, cols, r, 0.2);
    defer gpa.free(bias);

    const want = try gpa.alloc(f32, rows * cols);
    defer gpa.free(want);
    ref.layerNorm(want, src, gain, bias, rows, cols, cols, 1e-9);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(f32, rows * cols);
        defer gpa.free(got);
        B.layerNorm(got, src, gain, bias, rows, cols, cols, 1e-9);
        try report("layerNorm", B.name, maxAbsDiff(want, got), magnitude(want), 1e-5);
    }
}

test "T1 dot and axpy stay within tolerance" {
    const gpa = testing.allocator;
    var rng = Rng.init(0x1337);
    const r = rng.random();

    const n = d_model;
    const a = try randomFloats(gpa, n, r, 2.0);
    defer gpa.free(a);
    const b = try randomFloats(gpa, n, r, 2.0);
    defer gpa.free(b);

    const want_dot = ref.dot(a, b);
    inline for (backend.all) |B| {
        const got = B.dot(a, b);
        try report("dot", B.name, @abs(got - want_dot), @abs(want_dot), 1e-5);
    }

    const want = try gpa.alloc(f32, n);
    defer gpa.free(want);
    @memcpy(want, a);
    ref.axpy(want, b, 0.37);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(f32, n);
        defer gpa.free(got);
        @memcpy(got, a);
        B.axpy(got, b, 0.37);
        try report("axpy", B.name, maxAbsDiff(want, got), magnitude(want), 1e-6);
    }
}

test "T1 elementwise ops agree exactly" {
    const gpa = testing.allocator;
    var rng = Rng.init(0x2468);
    const r = rng.random();

    const n = ffn_dim + 1;
    const src = try randomFloats(gpa, n, r, 5.0);
    defer gpa.free(src);
    const other = try randomFloats(gpa, n, r, 5.0);
    defer gpa.free(other);

    inline for ([_]ref.Act{ .relu, .gelu, .swish }) |kind| {
        const want = try gpa.alloc(f32, n);
        defer gpa.free(want);
        @memcpy(want, src);
        ref.activation(want, kind);

        inline for (backend.all) |B| {
            const got = try gpa.alloc(f32, n);
            defer gpa.free(got);
            @memcpy(got, src);
            B.activation(got, kind);
            try testing.expectEqualSlices(f32, want, got);
        }
    }

    const want_add = try gpa.alloc(f32, n);
    defer gpa.free(want_add);
    @memcpy(want_add, src);
    ref.residualAdd(want_add, other);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(f32, n);
        defer gpa.free(got);
        @memcpy(got, src);
        B.residualAdd(got, other);
        try testing.expectEqualSlices(f32, want_add, got);
    }
}

test "T1 softmax and argmax agree exactly" {
    const gpa = testing.allocator;
    var rng = Rng.init(0x9753);
    const r = rng.random();

    const n = 2048;
    const src = try randomFloats(gpa, n, r, 12.0);
    defer gpa.free(src);

    const want = try gpa.alloc(f32, n);
    defer gpa.free(want);
    @memcpy(want, src);
    ref.softmax(want);

    var total: f32 = 0;
    for (want) |v| total += v;
    try testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-4);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(f32, n);
        defer gpa.free(got);
        @memcpy(got, src);
        B.softmax(got);
        try testing.expectEqualSlices(f32, want, got);
        try testing.expectEqual(ref.argmax(want), B.argmax(got));
    }
}

test "T1 gathers agree exactly" {
    const gpa = testing.allocator;
    var rng = Rng.init(0x8642);
    const r = rng.random();

    const vocab: u32 = 64;
    const d = d_model;
    const emb = try randomWeights(gpa, vocab * d, r);
    defer gpa.free(emb);
    const emb_scales = try randomScales(gpa, vocab, r);
    defer gpa.free(emb_scales);
    const pos = try randomFloats(gpa, rows * d, r, 1.0);
    defer gpa.free(pos);

    var ids: [rows]u32 = undefined;
    for (&ids) |*id| id.* = r.intRangeLessThan(u32, 0, vocab);

    const want = try gpa.alloc(f32, rows * d);
    defer gpa.free(want);
    ref.embedGather(want, &ids, emb, d, emb_scales, pos, d, 16.0);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(f32, rows * d);
        defer gpa.free(got);
        B.embedGather(got, &ids, emb, d, emb_scales, pos, d, 16.0);
        try testing.expectEqualSlices(f32, want, got);
    }

    const want_rows = try gpa.alloc(i8, rows * d);
    defer gpa.free(want_rows);
    const want_row_scales = try gpa.alloc(f32, rows);
    defer gpa.free(want_row_scales);
    ref.gatherRows(want_rows, d, want_row_scales, &ids, emb, d, emb_scales, d);

    inline for (backend.all) |B| {
        const got = try gpa.alloc(i8, rows * d);
        defer gpa.free(got);
        const got_scales = try gpa.alloc(f32, rows);
        defer gpa.free(got_scales);
        B.gatherRows(got, d, got_scales, &ids, emb, d, emb_scales, d);
        try testing.expectEqualSlices(i8, want_rows, got);
        try testing.expectEqualSlices(f32, want_row_scales, got_scales);
    }
}

test "T1 quantize round-trip keeps the error where SPEC §7 says it should be" {
    // Not a backend comparison: a sanity bound on the quantization contract
    // itself, so a change to the rounding rule shows up as a number.
    const gpa = testing.allocator;
    var rng = Rng.init(0x5555);
    const r = rng.random();

    const cols = d_model;
    const src = try randomFloats(gpa, cols, r, 3.0);
    defer gpa.free(src);

    const q = try gpa.alloc(i8, cols);
    defer gpa.free(q);
    var scale: [1]f32 = undefined;
    ref.quantizeRows(q, cols, &scale, src, cols, 1, cols);

    var worst: f32 = 0;
    for (src, q) |x, v| {
        worst = @max(worst, @abs(x - @as(f32, @floatFromInt(v)) * scale[0]));
    }
    // Symmetric absmax quantization: at most half a step.
    try testing.expect(worst <= scale[0] * 0.5 + 1e-6);
}
