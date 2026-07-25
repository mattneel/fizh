//! graph/decoder.zig — SPEC §8's `qgemv_i8` regime: 1×K×N, one token at a time.
//!
//! Greedy, beam width 1 (I7). One residual vector, three cached-attention
//! reads, one shortlisted projection, one argmax. Nothing here allocates and
//! nothing here grows: every buffer was carved at init and is reused every
//! step, which is what SPEC §12.10 asks the step test to prove.

const std = @import("std");
const assert = std.debug.assert;

const backend = @import("../kernel/backend.zig");
const format = @import("../model/format.zig");
const kernel = backend.active;
const layout = @import("../model/layout.zig");

const attention = @import("attention.zig");
const pass = @import("pass.zig");
const shortlist = @import("shortlist.zig");

/// Scratch vector slots, named so a reader does not have to count.
const slot_x: u32 = 0;
const slot_normed: u32 = 1;
const slot_q: u32 = 2;
const slot_k: u32 = 3;
const slot_v: u32 = 4;
const slot_ctx: u32 = 5;
const slot_branch: u32 = 6;
const slot_hidden: u32 = 7;
const slot_gate: u32 = 8;

/// Returns the number of target ids written to `ctx.tgt_ids`, including the
/// terminating end-of-sequence when there is one.
pub fn run(ctx: *pass.Ctx) u32 {
    assert(ctx.src_len > 0);
    shortlist.build(ctx);

    const limit = stepLimit(ctx);
    assert(limit > 0);

    const d = ctx.hp.d_model;
    // Marian shifts the decoder's target embeddings right by one and zero-fills
    // (`shift(embeddings, {1, 0, 0})`), so step 0 has no previous token at all.
    var prev: ?u32 = null;
    var t: u32 = 0;

    // SPEC §12.3: bounded, and bounded by a `for` rather than a promise.
    for (0..limit) |_| {
        const x = ctx.slot4(slot_x)[0..d];
        if (prev) |id| embedStep(ctx, id, t, x) else embedStart(ctx, x);
        for (0..ctx.hp.n_dec_layers) |l| decLayer(ctx, @intCast(l), t);

        if (ctx.hp.prenorm == 1) {
            normInto(ctx, x, x, ctx.sl.dec_ln_gain, ctx.sl.dec_ln_bias);
        }

        const next = project(ctx, x);
        ctx.tgt_ids[t] = next;
        t += 1;
        if (next == ctx.hp.eos_id) break;
        prev = next;
    }

    // SPEC §12.3 asserts *both* bounds, not whichever one happened to bind.
    assert(t <= ctx.max_tgt_tokens);
    assert(t <= limit);
    return t;
}

/// SPEC §12.3: generation is bounded by `max_tgt_tokens` and by
/// `max_length_factor · src_len`, whichever is smaller.
fn stepLimit(ctx: *const pass.Ctx) u32 {
    const factor = ctx.hp.max_length_factor;
    assert(std.math.isFinite(factor) and factor >= 1.0);

    const scaled = @ceil(factor * @as(f32, @floatFromInt(ctx.src_len)));
    const by_factor: u32 = if (scaled >= @as(f32, @floatFromInt(ctx.max_tgt_tokens)))
        ctx.max_tgt_tokens
    else
        @as(u32, @intFromFloat(scaled)) + 2;

    const limit = @min(ctx.max_tgt_tokens, by_factor);
    assert(limit >= 1);
    return limit;
}

/// The decoder input for step 0. Marian's decoder embeds the target sequence
/// shifted right by one with a *zero* fill, so the first step sees the
/// positional encoding alone and no token embedding whatsoever. Seeding with
/// `emb(</s>)` instead — which is what `bos_id` would suggest — gets about half
/// of all first tokens wrong while every later step is unaffected. ADR 0015.
fn embedStart(ctx: *pass.Ctx, out: []f32) void {
    const d = ctx.hp.d_model;
    assert(out.len == d);
    assert(ctx.pos_enc.len >= d);

    @memcpy(out, ctx.pos_enc[0..d]);
}

fn embedStep(ctx: *pass.Ctx, id: u32, t: u32, out: []f32) void {
    const d = ctx.hp.d_model;
    assert(id < ctx.hp.vocab_size);
    assert(t < ctx.max_tgt_tokens);

    const emb = ctx.sl.emb;
    const ids = [_]u32{id};
    kernel.embedGather(
        out,
        &ids,
        format.i8View(ctx.slot, emb.data, emb.n * emb.stride()),
        emb.stride(),
        format.f32View(ctx.slot, emb.scales, emb.n),
        ctx.pos_enc[t * d ..],
        d,
        ctx.hp.emb_scale,
    );
}

fn decLayer(ctx: *pass.Ctx, l: u32, t: u32) void {
    const dl = ctx.sl.decLayer(l);
    ssruStep(ctx, dl.ssru, l);
    crossAttention(ctx, dl.cross_attn, l);
    feedForward(ctx, dl.ffn);
    _ = t;
}

/// SPEC §8's `attn_decode` slot, filled by what Bergamot actually ships: a
/// Simpler Simple Recurrent Unit (ADR 0008).
///
///     f = Wf·x + bf
///     c = sigmoid(f)·c + (1 - sigmoid(f))·(W·x)
///     h = relu(c)
///
/// Two `qgemv`s and a pointwise gate. Constant work per step, where attention
/// over a growing prefix is linear in it — and the state is one vector, not a
/// cache.
fn ssruStep(ctx: *pass.Ctx, a: layout.Ssru, l: u32) void {
    const d = ctx.hp.d_model;
    const x = ctx.slot4(slot_x)[0..d];
    const src = branchInput(ctx, x, a.ln_gain, a.ln_bias);

    const wx = ctx.slot4(slot_q)[0..d];
    const f = ctx.slot4(slot_gate)[0..d];
    // `rnn_W` has no bias; only the forget gate does. Marian: `dot(x, W)` and
    // `affine(x, Wf, bf)`. Two matmuls, so two alphas under ADR 0012 — and
    // therefore two quantizations of the same input.
    gemv(ctx, a.w, null, wx, quantVec(ctx, src, a.w), d, d);
    gemv(ctx, a.wf, a.bf, f, quantVec(ctx, src, a.wf), d, d);

    const out = ctx.slot4(slot_branch)[0..d];
    kernel.ssruGate(ctx.ssruCell(l), wx, f, out);
    merge(ctx, x, out, a.ln_gain, a.ln_bias);
}

fn crossAttention(ctx: *pass.Ctx, a: layout.Attn, l: u32) void {
    const d = ctx.hp.d_model;
    const x = ctx.slot4(slot_x)[0..d];
    const src = branchInput(ctx, x, a.ln_gain, a.ln_bias);

    const q = ctx.slot4(slot_q)[0..d];
    const q_scale = quantVec(ctx, src, a.q);
    gemv(ctx, a.q, a.q_bias, q, q_scale, d, d);

    const c = ctx.slot4(slot_ctx)[0..d];
    attention.decodeStep(ctx, q, ctx.xattn(l, .k), ctx.xattn(l, .v), c, ctx.src_len);

    const out = ctx.slot4(slot_branch)[0..d];
    const ctx_scale = quantVec(ctx, c, a.o);
    gemv(ctx, a.o, a.o_bias, out, ctx_scale, d, d);
    merge(ctx, x, out, a.ln_gain, a.ln_bias);
}

fn feedForward(ctx: *pass.Ctx, f: layout.Ffn) void {
    const d = ctx.hp.d_model;
    const ffn = ctx.hp.ffn_dim;
    const x = ctx.slot4(slot_x)[0..d];
    const src = branchInput(ctx, x, f.ln_gain, f.ln_bias);

    const hidden = ctx.slot4(slot_hidden)[0..ffn];
    const in_scale = quantVec(ctx, src, f.w1);
    gemv(ctx, f.w1, f.bias1, hidden, in_scale, d, ffn);
    kernel.activation(hidden, act(ctx));

    const out = ctx.slot4(slot_branch)[0..d];
    const hidden_scale = quantVec(ctx, hidden, f.w2);
    gemv(ctx, f.w2, f.bias2, out, hidden_scale, ffn, d);
    merge(ctx, x, out, f.ln_gain, f.ln_bias);
}

/// Pre-norm normalizes the branch input; post-norm feeds the residual straight
/// in and normalizes the sum afterwards.
fn branchInput(ctx: *pass.Ctx, x: []f32, ln_gain: u32, ln_bias: u32) []const f32 {
    if (ctx.hp.prenorm != 1) return x;
    const normed = ctx.slot4(slot_normed)[0..ctx.hp.d_model];
    normInto(ctx, normed, x, ln_gain, ln_bias);
    return normed;
}

fn merge(ctx: *pass.Ctx, x: []f32, branch: []f32, ln_gain: u32, ln_bias: u32) void {
    assert(x.len == branch.len);
    if (ctx.hp.prenorm == 1) {
        kernel.residualAdd(x, branch);
        return;
    }
    kernel.residualAdd(branch, x);
    normInto(ctx, x, branch, ln_gain, ln_bias);
}

fn normInto(ctx: *pass.Ctx, out: []f32, in: []const f32, ln_gain: u32, ln_bias: u32) void {
    const d = ctx.hp.d_model;
    kernel.layerNorm(
        out,
        in,
        format.f32View(ctx.slot, ln_gain, d),
        format.f32View(ctx.slot, ln_bias, d),
        1,
        d,
        d,
        ctx.hp.norm_eps,
    );
}

/// SPEC §7: dynamic per-row absmax, for a single row, into `qslot(0)`.
/// Every `gemv` below reads that slot, so the two always appear as adjacent
/// statements — never as nested expressions.
fn quantVec(ctx: *pass.Ctx, src: []const f32, m: layout.QuantMat) f32 {
    const n: u32 = @intCast(src.len);
    assert(ctx.qvec.len >= n);

    if (ctx.staticScale(m)) |scale| {
        kernel.quantizeRowsWith(ctx.qslot(0)[0..n], n, src, n, 1, n, scale);
        return scale;
    }
    var scale: [1]f32 = undefined;
    kernel.quantizeRows(ctx.qslot(0)[0..n], n, &scale, src, n, 1, n);
    assert(std.math.isFinite(scale[0]) and scale[0] > 0);
    return scale[0];
}

fn gemv(ctx: *pass.Ctx, m: layout.QuantMat, bias_off: ?u32, out: []f32, scale: f32, k: u32, n: u32) void {
    assert(m.k == k and m.n == n);
    kernel.qgemv(
        out,
        ctx.qslot(0)[0..k],
        scale,
        format.i8View(ctx.slot, m.data, m.n * m.stride()),
        m.stride(),
        format.f32View(ctx.slot, m.scales, m.n),
        if (bias_off) |off| format.f32View(ctx.slot, off, n) else null,
        k,
        n,
    );
}

/// SPEC §8 `logits_project` then `argmax`: 1×K×S over the shortlist, never the
/// vocabulary (I6).
fn project(ctx: *pass.Ctx, x: []const f32) u32 {
    const d = ctx.hp.d_model;
    const n = ctx.shortlist_len;
    assert(n >= 2 and n <= ctx.max_shortlist);

    // `quantVec` fills `qslot(0)`, which the projection then reads. Sequenced
    // explicitly rather than left to argument evaluation order: the aliasing is
    // the whole point, and a reader should not have to know Zig's rules to see
    // that it is deliberate.
    const scale = quantVec(ctx, x, ctx.sl.emb);
    const logits = ctx.logits[0..n];
    kernel.qgemv(
        logits,
        ctx.qslot(0)[0..d],
        scale,
        ctx.shortlist_rows,
        d,
        ctx.shortlist_scales,
        null,
        d,
        n,
    );

    const bias = format.f32View(ctx.slot, ctx.sl.emb_bias, ctx.hp.vocab_size);
    for (logits, ctx.shortlist_ids[0..n]) |*v, id| v.* += bias[id];

    const best = kernel.argmax(logits);
    assert(best < n);
    return ctx.shortlist_ids[best];
}

fn act(ctx: *const pass.Ctx) backend.ref.Act {
    return switch (ctx.hp.act()) {
        .relu => .relu,
        .gelu => .gelu,
        .swish => .swish,
        _ => .relu,
    };
}
