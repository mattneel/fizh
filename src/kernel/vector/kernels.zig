//! kernel/vector/kernels.zig — the vector backend. Invariant I1.
//!
//! Hand-written and architecture-neutral: `@Vector`, `@shuffle`, `@select` and
//! `@reduce` with comptime-known indices only (SPEC §3), which is exactly the
//! subset that lowers to v128 on wasm, NEON on aarch64 and SSE/AVX on x86_64.
//! Nothing here is x86-tuned, because the devices are ARM and that is the
//! reason fizh exists.
//!
//! This was called `simd128/` until an audit found it contains no wasm and no
//! target branches at all — 245 lines of portable `@Vector`. The old name
//! encoded a constraint that was never real (ADR 0023).
//!
//! **The integer paths are bit-exact against `ref/`.** Integer addition is
//! associative, so accumulating into `i32` lanes and reducing at the end
//! produces the identical sum to a scalar loop. That is why SPEC §7 insists on
//! `i32` accumulation: it makes the fast path and the oracle the same function,
//! not merely close.
//!
//! The `f32` paths are not bit-exact — a four-lane sum has a different
//! association than a scalar one — which is what T1 measures.

const std = @import("std");
const assert = std.debug.assert;

const ref = @import("../ref/kernels.zig");
const math = @import("../math.zig");

pub const name = "vector";
pub const Act = ref.Act;

/// Integer lane count, from the target rather than from a guess.
///
/// `suggestVectorLength` returns null on wasm (no CPU model to ask), 16 on
/// aarch64, and 32 on an x86_64 with AVX2. Widening the integer path is free of
/// consequence: SPEC §7 accumulates in `i32`, integer addition is associative,
/// so N lanes and a final reduce give bit-identical sums for every N. That is
/// what keeps this backend bit-exact against `ref/` on every target.
const lanes: usize = std.simd.suggestVectorLength(i8) orelse 16;
const I8x = @Vector(lanes, i8);
const I16x = @Vector(lanes, i16);
const I32x = @Vector(lanes, i32);

/// Float lane count, deliberately **fixed** where the integer one is not.
///
/// Float addition is not associative, so a lane count decides reduction order
/// and reduction order decides the last bits. Taking the target's suggestion
/// here would make a native build round differently from the wasm one — the
/// same input, two answers — which is exactly the divergence I9 exists to
/// forbid. It would also buy nothing where it matters: the suggestion is 4 on
/// aarch64 and unavailable on wasm, so only an AVX2 desktop would ever widen,
/// and no phone would. ADR 0023.
const flanes: usize = 4;
const F32x = @Vector(flanes, f32);

comptime {
    assert(lanes >= flanes and lanes % flanes == 0);
}

// -- ops that are worth vectorizing -----------------------------------------

pub fn qgemm(
    out: []f32,
    out_stride: u32,
    a: []const i8,
    a_stride: u32,
    a_scales: []const f32,
    w: []const i8,
    w_stride: u32,
    w_scales: []const f32,
    bias: ?[]const f32,
    m: u32,
    k: u32,
    n: u32,
) void {
    assert(m > 0 and k > 0 and n > 0);
    assert(a_stride >= k and w_stride >= k and out_stride >= n);
    assert(a_scales.len >= m and w_scales.len >= n);

    for (0..m) |row| {
        qgemv(
            out[row * out_stride ..][0..n],
            a[row * a_stride ..][0..k],
            a_scales[row],
            w,
            w_stride,
            w_scales,
            bias,
            k,
            n,
        );
    }
}

pub fn qgemv(
    out: []f32,
    a: []const i8,
    a_scale: f32,
    w: []const i8,
    w_stride: u32,
    w_scales: []const f32,
    bias: ?[]const f32,
    k: u32,
    n: u32,
) void {
    assert(a.len >= k and out.len >= n);
    assert(std.math.isFinite(a_scale) and a_scale > 0);
    assert(w.len >= @as(usize, n - 1) * w_stride + k);
    assert(w_scales.len >= n);

    const a_row = a[0..k];
    for (0..n) |col| {
        const acc = dotI8(a_row, w[col * w_stride ..][0..k]);
        const w_scale = w_scales[col];
        assert(std.math.isFinite(w_scale) and w_scale > 0);
        var v = @as(f32, @floatFromInt(acc)) * (a_scale * w_scale);
        if (bias) |b| v += b[col];
        out[col] = v;
    }
}

/// Sixteen `int8` products per step. Each product is at most `127 * 127 =
/// 16129`, so the `i16` multiply cannot overflow; the widening to `i32` before
/// accumulation is what keeps the running sum exact past 132 terms.
fn dotI8(a: []const i8, b: []const i8) i32 {
    assert(a.len == b.len);

    var acc: I32x = @splat(0);
    var i: usize = 0;
    while (i + lanes <= a.len) : (i += lanes) {
        const av: I8x = a[i..][0..lanes].*;
        const bv: I8x = b[i..][0..lanes].*;
        const p: I16x = @as(I16x, av) * @as(I16x, bv);
        acc += @as(I32x, p);
    }

    var tail: i32 = 0;
    while (i < a.len) : (i += 1) tail += @as(i32, a[i]) * @as(i32, b[i]);
    return @reduce(.Add, acc) + tail;
}

/// SPEC §7. The absmax is a `@reduce(.Max, ...)`, which is order-independent
/// for finite inputs, so the scale — and therefore every quantized value — is
/// bit-identical to `ref/`.
pub fn quantizeRows(
    dst: []i8,
    dst_stride: u32,
    scales: []f32,
    src: []const f32,
    src_stride: u32,
    rows: u32,
    cols: u32,
) void {
    assert(rows > 0 and cols > 0);
    assert(dst_stride >= cols and src_stride >= cols);
    assert(scales.len >= rows);

    for (0..rows) |r| {
        const in = src[r * src_stride ..][0..cols];
        var absmax: f32 = 0;
        var i: usize = 0;
        var vmax: F32x = @splat(0);
        while (i + flanes <= cols) : (i += flanes) {
            const v: F32x = in[i..][0..flanes].*;
            vmax = @max(vmax, @abs(v));
        }
        absmax = @reduce(.Max, vmax);
        while (i < cols) : (i += 1) absmax = @max(absmax, @abs(in[i]));

        const scale: f32 = if (absmax == 0) 1.0 else absmax / 127.0;
        scales[r] = scale;
        quantizeRow(dst[r * dst_stride ..][0..cols], in, 1.0 / scale);
    }
}

fn quantizeRow(out: []i8, in: []const f32, inv: f32) void {
    assert(out.len == in.len);
    const magic: f32 = 8388608.0; // 2^23; round to nearest even without a libcall
    const vmagic: F32x = @splat(magic);
    const vinv: F32x = @splat(inv);
    const lo: F32x = @splat(-127.0);
    const hi: F32x = @splat(127.0);

    var i: usize = 0;
    while (i + flanes <= in.len) : (i += flanes) {
        const v: F32x = in[i..][0..flanes].*;
        const s = v * vinv;
        const rounded = @select(f32, s >= @as(F32x, @splat(0.0)), (s + vmagic) - vmagic, (s - vmagic) + vmagic);
        const clamped = @min(@max(rounded, lo), hi);
        const q: @Vector(flanes, i32) = @intFromFloat(clamped);
        const narrowed: @Vector(flanes, i8) = @intCast(q);
        out[i..][0..flanes].* = narrowed;
    }
    while (i < in.len) : (i += 1) {
        const s = in[i] * inv;
        const rounded = if (s >= 0) (s + magic) - magic else (s - magic) + magic;
        out[i] = @intFromFloat(std.math.clamp(rounded, -127.0, 127.0));
    }
}

pub fn residualAdd(dst: []f32, src: []const f32) void {
    assert(dst.len == src.len);
    assert(dst.len > 0);

    var i: usize = 0;
    while (i + flanes <= dst.len) : (i += flanes) {
        const d: F32x = dst[i..][0..flanes].*;
        const s: F32x = src[i..][0..flanes].*;
        dst[i..][0..flanes].* = d + s;
    }
    while (i < dst.len) : (i += 1) dst[i] += src[i];
}

pub fn axpy(dst: []f32, src: []const f32, s: f32) void {
    assert(dst.len == src.len);
    assert(std.math.isFinite(s));

    const vs: F32x = @splat(s);
    var i: usize = 0;
    while (i + flanes <= dst.len) : (i += flanes) {
        const d: F32x = dst[i..][0..flanes].*;
        const v: F32x = src[i..][0..flanes].*;
        dst[i..][0..flanes].* = d + v * vs;
    }
    while (i < dst.len) : (i += 1) dst[i] += src[i] * s;
}

/// Four partial sums, combined in a fixed order. Not the scalar association, so
/// not bit-exact against `ref/` — T1 is where that difference gets a number.
pub fn dot(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);
    assert(a.len > 0);

    var acc: F32x = @splat(0);
    var i: usize = 0;
    while (i + flanes <= a.len) : (i += flanes) {
        const av: F32x = a[i..][0..flanes].*;
        const bv: F32x = b[i..][0..flanes].*;
        acc += av * bv;
    }
    var tail: f32 = 0;
    while (i < a.len) : (i += 1) tail += a[i] * b[i];
    return horizontal(acc) + tail;
}

/// Fixed pairwise order: (0+1) + (2+3). Written out rather than `@reduce` so
/// the association is in the source, not in the backend's mood.
fn horizontal(v: F32x) f32 {
    return (v[0] + v[1]) + (v[2] + v[3]);
}

pub fn layerNorm(
    out: []f32,
    in: []const f32,
    gain: []const f32,
    bias: []const f32,
    rows: u32,
    cols: u32,
    stride: u32,
    eps: f32,
) void {
    assert(rows > 0 and cols > 0 and stride >= cols);
    assert(gain.len >= cols and bias.len >= cols);

    const inv_n = 1.0 / @as(f32, @floatFromInt(cols));
    for (0..rows) |r| {
        const x = in[r * stride ..][0..cols];
        const y = out[r * stride ..][0..cols];

        const mean = sum(x) * inv_n;
        const vmean: F32x = @splat(mean);

        var sq: F32x = @splat(0);
        var i: usize = 0;
        while (i + flanes <= cols) : (i += flanes) {
            const v: F32x = x[i..][0..flanes].*;
            const d = v - vmean;
            sq += d * d;
        }
        var sq_tail: f32 = 0;
        while (i < cols) : (i += 1) {
            const d = x[i] - mean;
            sq_tail += d * d;
        }

        const inv_std = 1.0 / @sqrt((horizontal(sq) + sq_tail) * inv_n + eps);
        const vinv: F32x = @splat(inv_std);

        i = 0;
        while (i + flanes <= cols) : (i += flanes) {
            const v: F32x = x[i..][0..flanes].*;
            const g: F32x = gain[i..][0..flanes].*;
            const b: F32x = bias[i..][0..flanes].*;
            y[i..][0..flanes].* = (v - vmean) * vinv * g + b;
        }
        while (i < cols) : (i += 1) {
            y[i] = (x[i] - mean) * inv_std * gain[i] + bias[i];
        }
    }
}

fn sum(x: []const f32) f32 {
    var acc: F32x = @splat(0);
    var i: usize = 0;
    while (i + flanes <= x.len) : (i += flanes) {
        const v: F32x = x[i..][0..flanes].*;
        acc += v;
    }
    var tail: f32 = 0;
    while (i < x.len) : (i += 1) tail += x[i];
    return horizontal(acc) + tail;
}

pub fn activation(x: []f32, kind: Act) void {
    assert(x.len > 0);
    // `relu` is the only one worth a vector path; `gelu` and `swish` are
    // dominated by `exp`, which is scalar (ADR 0003) until M7 says otherwise.
    if (kind != .relu) return ref.activation(x, kind);

    const zero: F32x = @splat(0);
    var i: usize = 0;
    while (i + flanes <= x.len) : (i += flanes) {
        const v: F32x = x[i..][0..flanes].*;
        x[i..][0..flanes].* = @max(v, zero);
    }
    while (i < x.len) : (i += 1) x[i] = @max(x[i], 0);
}

// -- ops where the scalar version is already the right one -------------------
//
// Re-exported rather than duplicated: a second implementation that is identical
// is a second place for a bug to hide. M7 revisits with T5 numbers in hand.

/// The gate is dominated by `sigmoid`, which is dominated by `exp`, which is a
/// polynomial rather than an instruction (ADR 0003). Vectorizing the surrounding
/// arithmetic without vectorizing `exp` buys nothing, so this is `ref`'s.
pub const ssruGate = ref.ssruGate;
pub const quantizeRowsWith = ref.quantizeRowsWith;

pub const softmax = ref.softmax;
pub const argmax = ref.argmax;
pub const embedGather = ref.embedGather;
pub const gatherRows = ref.gatherRows;
