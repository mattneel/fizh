//! graph/pass.zig — one model, source bytes in, target bytes out.
//!
//! Tokenize, encode, greedy decode, detokenize. SPEC §9: the host never sees a
//! tensor, so this is where text stops being text and starts being a tensor,
//! and where it stops again on the way out.
//!
//! `Ctx` is the pass's whole world: the model, its slot, and every arena region
//! it may touch, resolved once so that nothing below this file does pointer
//! arithmetic on the arena.

const std = @import("std");
const assert = std.debug.assert;

const format = @import("../model/format.zig");
const layout = @import("../model/layout.zig");
const charsmap_mod = @import("../tok/charsmap.zig");
const profile = @import("profile.zig");
const ssplit = @import("../tok/ssplit.zig");
const trie = @import("../tok/trie.zig");
const unigram = @import("../tok/unigram.zig");

const decoder = @import("decoder.zig");
const encoder = @import("encoder.zig");

pub const Error = error{
    /// The source needs more tokens than the arena was carved for.
    SrcTooLong,
    /// The translation does not fit the caller's buffer.
    OutTooSmall,
};

/// Everything a pass may read or write. Built once by `runtime.zig`.
pub const Ctx = struct {
    model: *const format.Model,
    slot: []const u8,
    hp: format.HParams,
    sl: layout.SlotLayout,

    /// Live token counts, filled as the pass proceeds.
    src_len: u32 = 0,
    tgt_len: u32 = 0,
    /// Candidate count from `graph/shortlist.zig`, fixed for the whole decode.
    shortlist_len: u32 = 0,

    max_src_tokens: u32,
    max_tgt_tokens: u32,
    max_shortlist: u32,

    pos_enc: []const f32,
    xattn_kv: []f32,
    ssru_state: []f32,
    enc_states: []f32,
    act_a: []f32,
    act_b: []f32,
    qact: []i8,
    qact_scales: []f32,
    attn_work: []f32,
    attn_scores: []f32,
    vec: []f32,
    qvec: []i8,
    shortlist_rows: []i8,
    shortlist_ids: []u32,
    shortlist_scales: []f32,
    logits: []f32,
    shortlist_seen: []u8,
    src_ids: []u32,
    tgt_ids: []u32,
    tok_raw: []u8,
    tok_norm: []u8,
    sent_spans: []ssplit.Span,
    tok_lattice: []unigram.LatticeNode,

    pub fn vocab(self: *const Ctx) trie.Vocab {
        return self.model.vocab(self.slot);
    }

    /// Non-breaking prefixes for this direction's source language, shipped in
    /// the artifact. Empty until the converter emits them.
    /// The static activation multiplier for a matmul, or null when this
    /// artifact wants SPEC §7's dynamic scales (ADR 0012).
    pub fn staticScale(self: *const Ctx, m: layout.QuantMat) ?f32 {
        if (self.hp.act_quant != 1) return null;
        assert(m.alpha != 0);
        const alpha = format.f32View(self.slot, m.alpha, 1)[0];
        assert(alpha > 0);
        return 1.0 / alpha;
    }

    /// The `nmt_nfkc` rewrite table for this direction, empty when the
    /// artifact carries none. ADR 0017.
    pub fn charsmap(self: *const Ctx) charsmap_mod.Charsmap {
        return self.model.charsmap(self.slot);
    }

    pub fn prefixes(self: *const Ctx) ssplit.Prefixes {
        return .{ .blob = self.model.prefixes(self.slot) };
    }

    /// The `i`-th `d_model`-wide scratch vector. Named slots, so a decode step
    /// can hold its residual, its query and its context live at once without
    /// anyone counting offsets.
    pub fn slot4(self: *const Ctx, i: u32) []f32 {
        const width = @max(self.hp.d_model, self.hp.ffn_dim);
        assert((i + 1) * width <= self.vec.len);
        return self.vec[i * width ..][0..width];
    }

    pub fn qslot(self: *const Ctx, i: u32) []i8 {
        const width = @max(self.hp.d_model, self.hp.ffn_dim);
        assert((i + 1) * width <= self.qvec.len);
        return self.qvec[i * width ..][0..width];
    }

    /// Encoder states as `[src_len][d_model]`.
    pub fn encRow(self: *const Ctx, t: u32) []f32 {
        const d = self.hp.d_model;
        assert(t < self.max_src_tokens);
        return self.enc_states[t * d ..][0..d];
    }

    /// Cross-attention K or V for decoder layer `l`, as `[src_len][d_model]`.
    /// SPEC §4.2: `2 · n_dec_layers · max_src_tokens · d_model`.
    pub fn xattn(self: *const Ctx, l: u32, which: enum { k, v }) []f32 {
        const d = self.hp.d_model;
        const per = self.max_src_tokens * d;
        const base = (2 * l + @intFromEnum(which)) * per;
        assert(base + per <= self.xattn_kv.len);
        return self.xattn_kv[base..][0..per];
    }

    /// The SSRU cell state for decoder layer `l`: one `d_model` vector, carried
    /// from one decode step to the next.
    pub fn ssruCell(self: *const Ctx, l: u32) []f32 {
        const d = self.hp.d_model;
        assert((l + 1) * d <= self.ssru_state.len);
        return self.ssru_state[l * d ..][0..d];
    }

    /// One of the four `[src_len][d_model]` attention buffers.
    pub fn work(self: *const Ctx, i: u32) []f32 {
        const per = self.max_src_tokens * self.hp.d_model;
        assert((i + 1) * per <= self.attn_work.len);
        return self.attn_work[i * per ..][0..per];
    }

    pub fn scores(self: *const Ctx, head: u32) []f32 {
        const per = @max(self.max_src_tokens, self.max_tgt_tokens);
        assert((head + 1) * per <= self.attn_scores.len);
        return self.attn_scores[head * per ..][0..per];
    }
};

/// SPEC §10: the pivot boundary is UTF-8 text, so this takes bytes and returns
/// bytes even when it is the first of two hops.
///
/// The input is split into sentences first. Bergamot's models are trained on
/// one sentence and emit `</s>` at the end of it, so handing them a paragraph
/// returns its first sentence and discards the rest — measured at −24.18 chrF++
/// on en→de before this existed. bergamot-translator splits for the same
/// reason. See ADR 0011.
pub fn run(ctx: *Ctx, in: []const u8, out: []u8) Error!u32 {
    assert(ctx.model.loaded);
    assert(in.len + 1 <= ctx.tok_norm.len);

    const spans = ctx.sent_spans;
    const count = ssplit.split(in, ctx.prefixes(), spans);
    if (count == 0) return 0;
    assert(count <= spans.len);

    var written: u32 = 0;
    var prev_end: u32 = 0;
    for (spans[0..count], 0..) |span, idx| {
        const sentence = in[span.start..span.end];
        if (sentence.len == 0) continue;

        // The separator between sentences is copied from the source verbatim,
        // not normalized to a space. bergamot-translator does the same, and a
        // seam that differs from its output costs chrF++ at every join —
        // word 2-grams straddle the boundary, so the penalty scales with the
        // sentence count rather than being a fixed cost per document.
        if (idx != 0) {
            const gap = in[prev_end..span.start];
            if (written + gap.len > out.len) return error.OutTooSmall;
            @memcpy(out[written..][0..gap.len], gap);
            written += @intCast(gap.len);
        }
        written += try sentence_(ctx, sentence, out[written..]);
        prev_end = span.end;
    }
    return written;
}

/// One sentence, hard-wrapped if it does not fit.
///
/// A single sentence longer than `max_src_tokens` used to fail the whole
/// translation with `src_too_long`, which made the behaviour of any long input
/// undefined in practice. bergamot-translator does not do that either: its
/// `max-length-break` chops an over-long sentence into pieces. So does this,
/// at word boundaries, and only when the sentence actually did not fit — a
/// sentence that fits is never touched.
fn sentence_(ctx: *Ctx, in: []const u8, out: []u8) Error!u32 {
    if (one(ctx, in, out)) |n| {
        return n;
    } else |e| switch (e) {
        error.OutTooSmall => return error.OutTooSmall,
        error.SrcTooLong => {},
    }

    var written: u32 = 0;
    var at: usize = 0;
    // Bounded: every iteration consumes at least one byte of `in`.
    for (0..in.len) |_| {
        if (at >= in.len) break;
        const chunk = wrapAt(in[at..], ctx.max_src_tokens);
        assert(chunk > 0);
        if (written != 0) {
            if (written + 1 > out.len) return error.OutTooSmall;
            out[written] = ' ';
            written += 1;
        }
        written += try one(ctx, in[at..][0..chunk], out[written..]);
        at += chunk;
        while (at < in.len and in[at] == ' ') at += 1;
    }
    return written;
}

/// Longest prefix of `text` that cannot exceed `max_tokens` pieces, ending on a
/// word boundary where one exists. One byte is at minimum one token, so a
/// prefix of `max_tokens` bytes is always safe.
fn wrapAt(text: []const u8, max_tokens: u32) usize {
    if (text.len <= max_tokens) return text.len;
    var end: usize = max_tokens;
    while (end > 1 and text[end] != ' ') end -= 1;
    return if (end <= 1) max_tokens else end;
}

/// One sentence: tokenize, encode, greedy decode, detokenize.
fn one(ctx: *Ctx, in: []const u8, out: []u8) Error!u32 {
    profile.calls += 1;
    var t = profile.start();
    const v = ctx.vocab();
    // SentencePiece normalizes before it segments: `nmt_nfkc` first, then the
    // whitespace rules and the word-boundary marker. Doing it the other way
    // round would run the rewrite over text that already carries markers.
    const raw = ctx.charsmap().normalize(in, ctx.tok_raw) catch return error.SrcTooLong;
    if (raw + 1 > ctx.tok_norm.len) return error.SrcTooLong;
    const n = unigram.normalize(ctx.tok_raw[0..raw], ctx.tok_norm);
    profile.stop(.normalize, t);
    t = profile.start();
    const params: unigram.Params = .{
        .unk_id = ctx.hp.unk_id,
        .eos_id = ctx.hp.eos_id,
    };

    ctx.src_len = unigram.encode(v, params, ctx.tok_norm[0..n], ctx.tok_lattice, ctx.src_ids) catch |e| switch (e) {
        error.TooManyTokens => return error.SrcTooLong,
        error.OutTooSmall => return error.OutTooSmall,
    };
    assert(ctx.src_len > 0);
    assert(ctx.src_len <= ctx.max_src_tokens);
    profile.stop(.tokenize, t);

    t = profile.start();
    encoder.run(ctx);
    profile.stop(.encoder, t);
    // The recurrence starts from zero for every sentence; a leftover cell would
    // make this sentence depend on the last one.
    @memset(ctx.ssru_state, 0);
    t = profile.start();
    ctx.tgt_len = decoder.run(ctx);
    profile.stop(.decoder, t);
    assert(ctx.tgt_len <= ctx.max_tgt_tokens);

    t = profile.start();
    defer profile.stop(.detokenize, t);
    return unigram.decode(v, ctx.tgt_ids[0..ctx.tgt_len], out) catch |e| switch (e) {
        error.OutTooSmall => error.OutTooSmall,
        error.TooManyTokens => error.OutTooSmall,
    };
}
