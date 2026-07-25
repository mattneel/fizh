//! kernel/ref/kernels.zig — the scalar `f32` oracle. Invariant I2.
//!
//! Slow, obvious, and permanent. Every loop runs in index order and every
//! reduction accumulates in a single scalar, because SPEC §3 forbids
//! `@setFloatMode(.optimized)` here and I9 needs a fixed reduction order. When
//! `simd128/` disagrees with this file, this file is right.
//!
//! Nothing here allocates, and every buffer is the caller's. SPEC §12.5: shapes,
//! strides, alignment and quantized ranges are asserted at every entry.

const std = @import("std");
const assert = std.debug.assert;

const math = @import("../math.zig");

pub const name = "ref";

pub const Act = enum { relu, gelu, swish };

// -- quantization -----------------------------------------------------------

/// SPEC §7: dynamic per-row absmax to `i8` in `[-127, 127]`, with an `f32`
/// scale. A row of all zeros gets scale 1, not 0 — a zero scale would make
/// every downstream product `0 * inf`.
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
    assert(dst.len >= @as(usize, rows - 1) * dst_stride + cols);
    assert(src.len >= @as(usize, rows - 1) * src_stride + cols);
    assert(scales.len >= rows);

    for (0..rows) |r| {
        const in = src[r * src_stride ..][0..cols];
        var absmax: f32 = 0;
        for (in) |x| {
            assert(std.math.isFinite(x));
            absmax = @max(absmax, @abs(x));
        }

        const scale: f32 = if (absmax == 0) 1.0 else absmax / 127.0;
        const inv = 1.0 / scale;
        scales[r] = scale;

        const out = dst[r * dst_stride ..][0..cols];
        for (in, out) |x, *q| {
            const v = roundTiesEven(x * inv);
            q.* = @intFromFloat(std.math.clamp(v, -127.0, 127.0));
            assert(q.* != -128);
        }
    }
}

/// SPEC §7's alternative: quantize against a *fixed* scale rather than one
/// derived from the row. Bergamot ships these per matmul (ADR 0012), so this is
/// the path that reproduces its arithmetic. Values outside the scale's range
/// clamp, which is the whole risk of a static scale and why the dynamic path
/// stays.
pub fn quantizeRowsWith(
    dst: []i8,
    dst_stride: u32,
    src: []const f32,
    src_stride: u32,
    rows: u32,
    cols: u32,
    scale: f32,
) void {
    assert(rows > 0 and cols > 0);
    assert(dst_stride >= cols and src_stride >= cols);
    assert(std.math.isFinite(scale) and scale > 0);

    const inv = 1.0 / scale;
    for (0..rows) |r| {
        const in = src[r * src_stride ..][0..cols];
        const out = dst[r * dst_stride ..][0..cols];
        for (in, out) |x, *q| {
            assert(std.math.isFinite(x));
            const v = roundTiesEven(x * inv);
            q.* = @intFromFloat(std.math.clamp(v, -127.0, 127.0));
            assert(q.* != -128);
        }
    }
}

/// Round half to even without `@round`, which is a libcall on wasm (ADR 0003).
fn roundTiesEven(x: f32) f32 {
    const magic: f32 = 8388608.0; // 2^23
    if (@abs(x) >= magic) return x;
    const shifted = if (x >= 0) (x + magic) - magic else (x - magic) + magic;
    return shifted;
}

// -- matrix multiply --------------------------------------------------------

/// `out[m][n] = dequant(sum_k a[m][k] * w[n][k]) + bias[n]`.
///
/// SPEC §7: the inner product accumulates in `i32`; the dequantization is one
/// multiply at the end. Never `f32` accumulation of integer products.
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
    assert(out.len >= @as(usize, m - 1) * out_stride + n);

    for (0..m) |row| {
        const a_row = a[row * a_stride ..][0..k];
        const a_scale = a_scales[row];
        assert(std.math.isFinite(a_scale) and a_scale > 0);
        qgemvInto(out[row * out_stride ..][0..n], a_row, a_scale, w, w_stride, w_scales, bias, k, n);
    }
}

/// `out[n] = dequant(sum_k a[k] * w[n][k]) + bias[n]`. The decoder workhorse.
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
    assert(a.len >= k);
    assert(out.len >= n);
    qgemvInto(out[0..n], a[0..k], a_scale, w, w_stride, w_scales, bias, k, n);
}

fn qgemvInto(
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
    assert(out.len == n);
    assert(a.len == k);
    assert(std.math.isFinite(a_scale) and a_scale > 0);
    assert(w.len >= @as(usize, n - 1) * w_stride + k);

    for (0..n) |col| {
        const w_row = w[col * w_stride ..][0..k];
        var acc: i32 = 0;
        for (a, w_row) |x, y| {
            assert(x != -128 and y != -128);
            acc += @as(i32, x) * @as(i32, y);
        }
        const w_scale = w_scales[col];
        assert(std.math.isFinite(w_scale) and w_scale > 0);
        var v = @as(f32, @floatFromInt(acc)) * (a_scale * w_scale);
        if (bias) |b| v += b[col];
        out[col] = v;
    }
}

// -- normalization ----------------------------------------------------------

/// Row-wise layer norm with a fixed reduction order. Mean and variance are two
/// separate passes rather than the one-pass `E[x^2] - E[x]^2` trick, which
/// loses catastrophically when the mean is large relative to the spread.
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
    assert(rows > 0 and cols > 0);
    assert(stride >= cols);
    assert(gain.len >= cols and bias.len >= cols);
    assert(std.math.isFinite(eps) and eps > 0);

    const inv_n = 1.0 / @as(f32, @floatFromInt(cols));
    for (0..rows) |r| {
        const x = in[r * stride ..][0..cols];
        const y = out[r * stride ..][0..cols];

        var sum: f32 = 0;
        for (x) |v| sum += v;
        const mean = sum * inv_n;

        var sq: f32 = 0;
        for (x) |v| {
            const d = v - mean;
            sq += d * d;
        }
        const inv_std = 1.0 / @sqrt(sq * inv_n + eps);

        for (x, y, gain[0..cols], bias[0..cols]) |v, *o, g, b| {
            o.* = (v - mean) * inv_std * g + b;
        }
    }
}

/// Row-wise softmax in place: subtract the max, exponentiate, normalize. Fixed
/// order throughout.
pub fn softmax(row: []f32) void {
    assert(row.len > 0);

    var max: f32 = row[0];
    for (row) |v| max = @max(max, v);
    assert(std.math.isFinite(max));

    var sum: f32 = 0;
    for (row) |*v| {
        const e = math.exp(v.* - max);
        v.* = e;
        sum += e;
    }
    assert(sum > 0);

    const inv = 1.0 / sum;
    for (row) |*v| v.* *= inv;
}

// -- elementwise ------------------------------------------------------------

pub fn activation(x: []f32, kind: Act) void {
    assert(x.len > 0);
    switch (kind) {
        .relu => for (x) |*v| {
            v.* = math.relu(v.*);
        },
        .gelu => for (x) |*v| {
            v.* = math.gelu(v.*);
        },
        .swish => for (x) |*v| {
            v.* = math.swish(v.*);
        },
    }
}

pub fn residualAdd(dst: []f32, src: []const f32) void {
    assert(dst.len == src.len);
    assert(dst.len > 0);
    for (dst, src) |*d, s| d.* += s;
}

/// SPEC §8. Ties go to the lowest index, which is what makes greedy decode
/// reproducible (I9).
pub fn argmax(x: []const f32) u32 {
    assert(x.len > 0);
    assert(x.len <= std.math.maxInt(u32));

    var best: u32 = 0;
    var best_v: f32 = x[0];
    for (x[1..], 1..) |v, i| {
        if (v > best_v) {
            best_v = v;
            best = @intCast(i);
        }
    }
    assert(best < x.len);
    return best;
}

// -- gathers ----------------------------------------------------------------

/// SPEC §8 `embed_gather`: random access, dequantize on gather, scale by
/// `emb_scale`, add the positional row.
pub fn embedGather(
    out: []f32,
    ids: []const u32,
    emb: []const i8,
    emb_stride: u32,
    emb_scales: []const f32,
    pos: []const f32,
    d: u32,
    emb_scale: f32,
) void {
    assert(ids.len > 0);
    assert(out.len >= ids.len * d);
    assert(pos.len >= ids.len * d);
    assert(std.math.isFinite(emb_scale) and emb_scale > 0);

    for (ids, 0..) |id, t| {
        assert(id < emb_scales.len);
        const row = emb[id * emb_stride ..][0..d];
        const scale = emb_scales[id] * emb_scale;
        assert(std.math.isFinite(scale));

        const dst = out[t * d ..][0..d];
        const p = pos[t * d ..][0..d];
        for (row, dst, p) |q, *o, pv| {
            assert(q != -128);
            o.* = @as(f32, @floatFromInt(q)) * scale + pv;
        }
    }
}

/// SPEC §8 `logits_project` reads a gathered slice of the output projection.
/// This collects those rows and their scales so the projection is a plain
/// `qgemv` over contiguous memory.
pub fn gatherRows(
    dst: []i8,
    dst_stride: u32,
    dst_scales: []f32,
    ids: []const u32,
    src: []const i8,
    src_stride: u32,
    src_scales: []const f32,
    k: u32,
) void {
    assert(ids.len > 0);
    assert(dst_stride >= k);
    assert(dst.len >= @as(usize, ids.len - 1) * dst_stride + k);
    assert(dst_scales.len >= ids.len);

    for (ids, 0..) |id, i| {
        assert(id < src_scales.len);
        @memcpy(dst[i * dst_stride ..][0..k], src[id * src_stride ..][0..k]);
        dst_scales[i] = src_scales[id];
    }
}

/// `dst[i] += src[i] * s`, used to accumulate attention-weighted values.
pub fn axpy(dst: []f32, src: []const f32, s: f32) void {
    assert(dst.len == src.len);
    assert(std.math.isFinite(s));
    for (dst, src) |*d, v| d.* += v * s;
}

/// SSRU (ADR 0008): `c = sigmoid(f)*c + (1 - sigmoid(f))*x`, then `relu(c)`.
///
/// Marian spells this `highway(cellState, x, f)` followed by `relu`, where
/// `highway(y, x, t) = sigmoid(t)*y + (1 - sigmoid(t))*x`. The cell is updated
/// in place — it is the only thing the decoder carries between steps.
pub fn ssruGate(cell: []f32, x: []const f32, f: []const f32, out: []f32) void {
    assert(cell.len == x.len);
    assert(cell.len == f.len);
    assert(cell.len == out.len);
    assert(cell.len > 0);

    for (cell, x, f, out) |*c, xv, fv, *o| {
        const g = math.sigmoid(fv);
        const next = g * c.* + (1.0 - g) * xv;
        c.* = next;
        o.* = math.relu(next);
    }
}

/// Plain `f32` dot product, for attention scores. Not quantized: SPEC §4.2
/// keeps the KV cache in `f32` (int8 KV is deferred to v2, SPEC §16).
pub fn dot(a: []const f32, b: []const f32) f32 {
    assert(a.len == b.len);
    assert(a.len > 0);
    var acc: f32 = 0;
    for (a, b) |x, y| acc += x * y;
    return acc;
}
