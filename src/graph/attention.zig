//! graph/attention.zig — SPEC §8's three attention regimes.
//!
//! `prefill` is batched and bidirectional: the encoder sees the whole source at
//! once and nothing is masked. `decodeStep` is batch 1 over a growing prefix,
//! and is the same function whether the cache it reads is the decoder's own
//! (self-attention, appended to each step) or the encoder's (cross-attention,
//! computed once and then read-only). SPEC §7 keeps both caches in `f32`;
//! int8 KV is deferred to v2 (SPEC §16).

const std = @import("std");
const assert = std.debug.assert;

const backend = @import("../kernel/backend.zig");
const kernel = backend.active;
const pass = @import("pass.zig");

/// `1 / sqrt(head_dim)`. `@sqrt` is a wasm instruction, not a libcall.
fn headScale(head_dim: u32) f32 {
    assert(head_dim > 0);
    return 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
}

/// SPEC §8 `attn_prefill`: batched, bidirectional. `q`, `k`, `v` and `out` are
/// all `[len][d_model]`.
pub fn prefill(ctx: *const pass.Ctx, q: []const f32, k: []const f32, v: []const f32, out: []f32, len: u32) void {
    const d = ctx.hp.d_model;
    const heads = ctx.hp.n_heads;
    const hd = ctx.hp.head_dim;
    assert(len > 0 and len <= ctx.max_src_tokens);
    assert(q.len >= @as(usize, len) * d);
    assert(out.len >= @as(usize, len) * d);
    assert(@as(u32, heads) * hd == d);

    const inv = headScale(hd);
    for (0..heads) |raw_h| {
        const h: u32 = @intCast(raw_h);
        const off = h * hd;
        const row = ctx.scores(h)[0..len];

        for (0..len) |i| {
            const qi = q[i * d + off ..][0..hd];
            for (0..len) |j| {
                row[j] = kernel.dot(qi, k[j * d + off ..][0..hd]) * inv;
            }
            kernel.softmax(row);

            const o = out[i * d + off ..][0..hd];
            @memset(o, 0);
            for (0..len) |j| {
                kernel.axpy(o, v[j * d + off ..][0..hd], row[j]);
            }
        }
    }
}

/// SPEC §8 `attn_decode` and `xattn_decode`. Batch 1, attending over the first
/// `prefix` rows of a `[*][d_model]` cache. Causality is structural: the
/// decoder appends its own key and value before calling, so the prefix it
/// attends over is exactly what already exists.
pub fn decodeStep(
    ctx: *const pass.Ctx,
    q: []const f32,
    k_cache: []const f32,
    v_cache: []const f32,
    out: []f32,
    prefix: u32,
) void {
    const d = ctx.hp.d_model;
    const heads = ctx.hp.n_heads;
    const hd = ctx.hp.head_dim;
    assert(prefix > 0);
    assert(q.len >= d and out.len >= d);
    assert(k_cache.len >= @as(usize, prefix) * d);
    assert(v_cache.len >= @as(usize, prefix) * d);

    const inv = headScale(hd);
    @memset(out[0..d], 0);

    for (0..heads) |raw_h| {
        const h: u32 = @intCast(raw_h);
        const off = h * hd;
        const row = ctx.scores(h)[0..prefix];
        const qh = q[off..][0..hd];

        for (0..prefix) |j| {
            row[j] = kernel.dot(qh, k_cache[j * d + off ..][0..hd]) * inv;
        }
        kernel.softmax(row);

        const o = out[off..][0..hd];
        for (0..prefix) |j| {
            kernel.axpy(o, v_cache[j * d + off ..][0..hd], row[j]);
        }
    }
}
