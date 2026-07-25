//! graph/encoder.zig — SPEC §8's `qgemm_i8` regime: M×K×N with M = src_len.
//!
//! Runs once per pass. When it returns, `enc_states` holds the encoded source
//! and `xattn_kv` holds every decoder layer's cross-attention keys and values,
//! so the decode loop never touches an encoder weight again.

const std = @import("std");
const assert = std.debug.assert;

const backend = @import("../kernel/backend.zig");
const format = @import("../model/format.zig");
const kernel = backend.active;
const layout = @import("../model/layout.zig");

const attention = @import("attention.zig");
const pass = @import("pass.zig");

/// SPEC §13 T2 wants an error profile *per layer*, and a whole-encoder number
/// cannot give one: it says the answer is wrong without saying where. This is
/// how `tools/trace.zig` gets the intermediate states. Null in every build that
/// is not being traced, so the cost is one null check per layer.
///
/// Stage 0 is the embedding; stage `l + 1` is the output of encoder layer `l`.
pub var layer_sink: ?*const fn (stage: u32, states: []const f32) void = null;

/// SPEC §12.10's threshold, overridable for synthetic fixtures. See
/// `assertNotCollapsed`.
pub var collapse_limit: f32 = 0.95;

fn observe(ctx: *const pass.Ctx, stage: u32) void {
    const sink = layer_sink orelse return;
    sink(stage, ctx.enc_states[0 .. ctx.src_len * ctx.hp.d_model]);
}

pub fn run(ctx: *pass.Ctx) void {
    assert(ctx.src_len > 0);
    assert(ctx.src_len <= ctx.max_src_tokens);

    embed(ctx);
    observe(ctx, 0);
    for (0..ctx.hp.n_enc_layers) |l| {
        layer(ctx, @intCast(l));
        observe(ctx, @intCast(l + 1));
    }

    assertNotCollapsed(ctx);

    if (ctx.hp.prenorm == 1) {
        const s = ctx.src_len;
        const d = ctx.hp.d_model;
        // In place: `layerNorm` reads each element before writing the same
        // index, and the row statistics are computed in a separate pass first.
        kernel.layerNorm(
            ctx.enc_states[0 .. s * d],
            ctx.enc_states[0 .. s * d],
            gain(ctx, ctx.sl.enc_ln_gain),
            gain(ctx, ctx.sl.enc_ln_bias),
            s,
            d,
            d,
            ctx.hp.norm_eps,
        );
    }
    crossKv(ctx);
}

/// SPEC §12.11: assert on the *shape* of the representation, not only the
/// numbers in it.
///
/// Every other assertion in §12 checks magnitude — finite scales, values in
/// range, alignment — and layer norm makes all of them pass under total
/// representational collapse. When the weight layout was wrong, mean pairwise
/// cosine between source positions reached 0.9996 by layer six: every position
/// the same vector, every existing assertion satisfied, and the only visible
/// symptom was a mistranslation.
///
/// Cost is `O(src_len² · d_model)` worst case, so it is sampled: at most
/// `max_pairs` position pairs, which makes it `O(d_model)` and unmeasurable
/// against the six encoder layers that precede it.
///
/// The statistic is the **mean** pairwise cosine, not the max. Measured on
/// FLORES devtest: the max routinely reaches 0.93–0.99 in a perfectly healthy
/// encoder — adjacent positions in a sentence really are similar — so a
/// threshold on the max is either useless or a false alarm. The mean separates
/// cleanly:
///
///     es-en healthy   median 0.66   worst observed 0.76
///     en-de healthy   median 0.14   worst observed 0.25
///     collapsed       1.0000
///
/// 0.95 sits four times the worst healthy observation below the failure.
///
/// It is a statement about *trained* weights. `test/artifact.zig` builds
/// artifacts from uniform random `int8`, which mixes far harder than anything
/// trained and collapses legitimately, so the fixtures raise this. Nothing on
/// a real-model path touches it.
fn assertNotCollapsed(ctx: *const pass.Ctx) void {
    const s = ctx.src_len;
    const d = ctx.hp.d_model;
    if (s < 2) return;

    const max_pairs: u32 = 16;
    var total: f32 = 0;
    var pairs: u32 = 0;
    // Stride the pairs rather than taking a prefix: neighbours are the most
    // similar even in a healthy encoder, so the first few would be the one
    // sample guaranteed to look worst.
    const step = @max(1, s / 8);

    var i: u32 = 0;
    while (i < s and pairs < max_pairs) : (i += step) {
        var j: u32 = i + step;
        while (j < s and pairs < max_pairs) : (j += step) {
            const a = ctx.enc_states[i * d ..][0..d];
            const b = ctx.enc_states[j * d ..][0..d];
            const na = @sqrt(kernel.dot(a, a));
            const nb = @sqrt(kernel.dot(b, b));
            if (na > 0 and nb > 0) {
                total += kernel.dot(a, b) / (na * nb);
                pairs += 1;
            }
        }
    }
    if (pairs == 0) return;

    const mean = total / @as(f32, @floatFromInt(pairs));
    // `assert(mean <= limit)` rather than `if (mean > limit)`: NaN fails the
    // former and passes the latter, and a NaN here would mean the encoder
    // produced non-finite states, which is exactly as bad as collapse.
    assert(mean <= collapse_limit);
}

fn embed(ctx: *pass.Ctx) void {
    const d = ctx.hp.d_model;
    const emb = ctx.sl.emb;
    kernel.embedGather(
        ctx.enc_states,
        ctx.src_ids[0..ctx.src_len],
        format.i8View(ctx.slot, emb.data, emb.n * emb.stride()),
        emb.stride(),
        format.f32View(ctx.slot, emb.scales, emb.n),
        ctx.pos_enc,
        d,
        ctx.hp.emb_scale,
    );
}

fn layer(ctx: *pass.Ctx, l: u32) void {
    const el = ctx.sl.encLayer(l);
    selfAttention(ctx, el.attn);
    feedForward(ctx, el.ffn);
}

fn selfAttention(ctx: *pass.Ctx, a: layout.Attn) void {
    const s = ctx.src_len;
    const d = ctx.hp.d_model;
    const x = ctx.enc_states[0 .. s * d];

    // Pre-norm normalizes the branch input and leaves the residual alone;
    // post-norm (Marian's default) normalizes the sum afterwards.
    const src = if (ctx.hp.prenorm == 1) blk: {
        const normed = ctx.act_a[0 .. s * d];
        kernel.layerNorm(normed, x, gain(ctx, a.ln_gain), gain(ctx, a.ln_bias), s, d, d, ctx.hp.norm_eps);
        break :blk normed;
    } else x;

    quantize(ctx, src, s, d, a.q);
    project(ctx, a.q, a.q_bias, ctx.work(0), s, d, d);
    project(ctx, a.k, a.k_bias, ctx.work(1), s, d, d);
    project(ctx, a.v, a.v_bias, ctx.work(2), s, d, d);

    attention.prefill(ctx, ctx.work(0), ctx.work(1), ctx.work(2), ctx.work(3), s);

    quantize(ctx, ctx.work(3)[0 .. s * d], s, d, a.o);
    const out = ctx.act_b[0 .. s * d];
    project(ctx, a.o, a.o_bias, out, s, d, d);

    merge(ctx, x, out, a.ln_gain, a.ln_bias, s, d);
}

fn feedForward(ctx: *pass.Ctx, f: layout.Ffn) void {
    const s = ctx.src_len;
    const d = ctx.hp.d_model;
    const ffn = ctx.hp.ffn_dim;
    const x = ctx.enc_states[0 .. s * d];

    const src = if (ctx.hp.prenorm == 1) blk: {
        const normed = ctx.act_a[0 .. s * d];
        kernel.layerNorm(normed, x, gain(ctx, f.ln_gain), gain(ctx, f.ln_bias), s, d, d, ctx.hp.norm_eps);
        break :blk normed;
    } else x;

    quantize(ctx, src, s, d, f.w1);
    const hidden = ctx.act_a[0 .. s * ffn];
    project(ctx, f.w1, f.bias1, hidden, s, d, ffn);
    kernel.activation(hidden, act(ctx));

    quantize(ctx, hidden, s, ffn, f.w2);
    const out = ctx.act_b[0 .. s * d];
    project(ctx, f.w2, f.bias2, out, s, ffn, d);

    merge(ctx, x, out, f.ln_gain, f.ln_bias, s, d);
}

/// The residual join. Post-norm folds the normalization into it; pre-norm does
/// not, and normalizes at the top of the next branch instead.
fn merge(ctx: *pass.Ctx, x: []f32, branch: []f32, ln_gain: u32, ln_bias: u32, s: u32, d: u32) void {
    assert(x.len == branch.len);
    assertNotCollapsed(ctx);

    if (ctx.hp.prenorm == 1) {
        kernel.residualAdd(x, branch);
        return;
    }
    kernel.residualAdd(branch, x);
    kernel.layerNorm(x, branch, gain(ctx, ln_gain), gain(ctx, ln_bias), s, d, d, ctx.hp.norm_eps);
}

/// SPEC §7: dynamic per-row absmax by default; the artifact's static alpha when
/// it asked for one (ADR 0012). `m` names the matmul this input feeds, because
/// Bergamot's alphas are per matmul.
fn quantize(ctx: *pass.Ctx, src: []const f32, rows: u32, cols: u32, m: layout.QuantMat) void {
    assert(src.len >= @as(usize, rows) * cols);
    assert(ctx.qact.len >= @as(usize, rows) * cols);

    if (ctx.staticScale(m)) |scale| {
        kernel.quantizeRowsWith(ctx.qact, cols, src, cols, rows, cols, scale);
        for (ctx.qact_scales[0..rows]) |*s| s.* = scale;
        return;
    }
    kernel.quantizeRows(ctx.qact, cols, ctx.qact_scales, src, cols, rows, cols);
}

fn project(ctx: *pass.Ctx, m: layout.QuantMat, bias_off: u32, out: []f32, rows: u32, k: u32, n: u32) void {
    assert(m.k == k and m.n == n);
    kernel.qgemm(
        out,
        n,
        ctx.qact,
        k,
        ctx.qact_scales,
        format.i8View(ctx.slot, m.data, m.n * m.stride()),
        m.stride(),
        format.f32View(ctx.slot, m.scales, m.n),
        format.f32View(ctx.slot, bias_off, n),
        rows,
        k,
        n,
    );
}

/// Cross-attention keys and values, once per decoder layer. SPEC §4.2 sizes
/// `xattn_kv` for exactly this, and it is why the decode loop is `qgemv`-only.
fn crossKv(ctx: *pass.Ctx) void {
    const s = ctx.src_len;
    const d = ctx.hp.d_model;
    for (0..ctx.hp.n_dec_layers) |raw| {
        const l: u32 = @intCast(raw);
        const xa = ctx.sl.decLayer(l).cross_attn;
        // Re-quantized per layer: with static alphas the scale differs by
        // matmul, so one shared quantization would be wrong for all but one.
        quantize(ctx, ctx.enc_states[0 .. s * d], s, d, xa.k);
        project(ctx, xa.k, xa.k_bias, ctx.xattn(l, .k), s, d, d);
        quantize(ctx, ctx.enc_states[0 .. s * d], s, d, xa.v);
        project(ctx, xa.v, xa.v_bias, ctx.xattn(l, .v), s, d, d);
    }
}

fn gain(ctx: *const pass.Ctx, off: u32) []const f32 {
    return format.f32View(ctx.slot, off, ctx.hp.d_model);
}

fn act(ctx: *const pass.Ctx) backend.ref.Act {
    return switch (ctx.hp.act()) {
        .relu => .relu,
        .gelu => .gelu,
        .swish => .swish,
        _ => .relu,
    };
}
