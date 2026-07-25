//! model/repack.zig — canonical `[N][K]` to the slot's padded layout.
//!
//! SPEC §6: the artifact is canonical row-major with `K` contiguous; the
//! backend repacks at load. SPEC §7 is checked here, once, on the way through:
//! after this file has run, every kernel may assert those invariants instead of
//! testing them.

const std = @import("std");
const assert = std.debug.assert;

const layout = @import("layout.zig");

pub const Error = error{
    /// SPEC §7 / I5: `-128` has no positive counterpart, so a weight that
    /// reaches it would break a future WGSL backend that assumes symmetry.
    WeightOutOfRange,
    /// A scale that is NaN, infinite, or non-positive poisons every product it
    /// touches. SPEC §12.6.
    ScaleNotFinite,
};

/// Copies an `int8` matrix into `slot`, padding each row out to the stride and
/// zeroing the pad so a kernel reading a whole vector never sees stale bytes.
pub fn quantMatrix(slot: []u8, m: layout.QuantMat, data: []const u8, scales: []const u8) Error!void {
    assert(data.len == @as(usize, m.n) * m.k);
    assert(scales.len == @as(usize, m.n) * 4);
    assert(m.data % layout.alignment == 0);

    if (hasNeg128(data)) return error.WeightOutOfRange;

    const stride = m.stride();
    const pad = stride - m.k;
    for (0..m.n) |row| {
        const dst = m.data + @as(u32, @intCast(row)) * stride;
        @memcpy(slot[dst..][0..m.k], data[row * m.k ..][0..m.k]);
        if (pad != 0) @memset(slot[dst + m.k ..][0..pad], 0);
    }

    @memcpy(slot[m.scales..][0 .. m.n * 4], scales);
    try checkScales(alignedFloats(slot, m.scales, m.n));
}

/// `f32` vectors — biases, layer-norm gains, unigram scores.
pub fn floats(slot: []u8, off: u32, src: []const u8, count: u32) Error!void {
    assert(src.len == @as(usize, count) * 4);
    assert(off % layout.alignment == 0);

    @memcpy(slot[off..][0 .. count * 4], src);
    for (alignedFloats(slot, off, count)) |x| {
        if (!std.math.isFinite(x)) return error.ScaleNotFinite;
    }
}

/// Raw bytes — piece text, flags. No interpretation, so nothing to check here;
/// `tok/trie.zig` validates the structure they form.
pub fn bytes(slot: []u8, off: u32, src: []const u8) void {
    assert(off % layout.alignment == 0);
    assert(off + src.len <= slot.len);
    @memcpy(slot[off..][0..src.len], src);
}

/// Slot offsets are 64-byte aligned by construction (`layout.zig`), so this
/// cast is sound; the assertion is what keeps it that way.
fn alignedFloats(slot: []const u8, off: u32, count: u32) []const f32 {
    assert(off % layout.alignment == 0);
    assert(off + count * 4 <= slot.len);
    const p: [*]const f32 = @ptrCast(@alignCast(slot.ptr + off));
    return p[0..count];
}

fn checkScales(view: []const f32) Error!void {
    assert(view.len > 0);
    for (view) |s| {
        if (!std.math.isFinite(s)) return error.ScaleNotFinite;
        if (s <= 0) return error.ScaleNotFinite;
    }
}

/// Sixteen bytes at a time, because this runs over every weight in the model
/// and SPEC §14 gives cold start 300 ms for everything.
fn hasNeg128(data: []const u8) bool {
    const V = @Vector(16, u8);
    const target: V = @splat(0x80);

    var i: usize = 0;
    while (i + 16 <= data.len) : (i += 16) {
        const v: V = data[i..][0..16].*;
        if (@reduce(.Or, v == target)) return true;
    }
    while (i < data.len) : (i += 1) {
        if (data[i] == 0x80) return true;
    }
    return false;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

test "rows are padded and the pad is zeroed" {
    const Slot = struct {
        var buf: [4096]u8 align(layout.alignment) = undefined;
    };
    @memset(&Slot.buf, 0xaa);

    const m: layout.QuantMat = .{ .data = 0, .scales = 1024, .n = 3, .k = 20 };
    try testing.expectEqual(@as(u32, 64), m.stride());

    var data: [60]u8 = undefined;
    for (&data, 0..) |*b, i| b.* = @intCast(i % 100);
    const scales = [_]f32{ 0.5, 0.25, 0.125 };

    try quantMatrix(&Slot.buf, m, &data, std.mem.sliceAsBytes(&scales));

    for (0..3) |row| {
        const base = row * 64;
        try testing.expectEqualSlices(u8, data[row * 20 ..][0..20], Slot.buf[base..][0..20]);
        for (Slot.buf[base + 20 ..][0..44]) |b| try testing.expectEqual(@as(u8, 0), b);
    }
    const got = std.mem.bytesAsSlice(f32, Slot.buf[1024..][0..12]);
    try testing.expectEqual(@as(f32, 0.25), got[1]);
}

test "a -128 weight is rejected wherever it hides" {
    const Slot = struct {
        var buf: [4096]u8 align(layout.alignment) = undefined;
    };
    const m: layout.QuantMat = .{ .data = 0, .scales = 1024, .n = 2, .k = 64 };
    const scales = [_]f32{ 1.0, 1.0 };

    var data: [128]u8 = @splat(1);
    try quantMatrix(&Slot.buf, m, &data, std.mem.sliceAsBytes(&scales));

    // Once in the vector body, once in the scalar tail, once at the very end.
    for ([_]usize{ 0, 7, 63, 64, 120, 127 }) |at| {
        data = @splat(1);
        data[at] = 0x80;
        try testing.expectError(
            error.WeightOutOfRange,
            quantMatrix(&Slot.buf, m, &data, std.mem.sliceAsBytes(&scales)),
        );
    }
}

test "non-finite and non-positive scales are rejected" {
    const Slot = struct {
        var buf: [4096]u8 align(layout.alignment) = undefined;
    };
    const m: layout.QuantMat = .{ .data = 0, .scales = 1024, .n = 2, .k = 64 };
    const data: [128]u8 = @splat(1);

    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -1.0, 0.0 }) |bad| {
        const scales = [_]f32{ 1.0, bad };
        try testing.expectError(
            error.ScaleNotFinite,
            quantMatrix(&Slot.buf, m, &data, std.mem.sliceAsBytes(&scales)),
        );
    }
}

test "hasNeg128 sees every byte" {
    var buf: [37]u8 = @splat(0);
    try testing.expect(!hasNeg128(&buf));
    for (0..buf.len) |i| {
        buf = @splat(0);
        buf[i] = 0x80;
        try testing.expect(hasNeg128(&buf));
    }
}
