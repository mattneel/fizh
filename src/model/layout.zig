//! model/layout.zig — where every weight sits inside a slot.
//!
//! SPEC §6 says the artifact's canonical matrix layout is row-major `[N][K]`
//! with `K` contiguous, and that the backend repacks at load. This file is the
//! destination of that repack: a deterministic function from hyper-parameters
//! to byte offsets, so a loaded model needs no per-tensor pointer table — just
//! its `HParams` and its slot base.
//!
//! Every offset is 64-byte aligned, including every matrix *row*, so the
//! kernels can assume aligned loads without a prologue.

const std = @import("std");
const assert = std.debug.assert;

const format = @import("format.zig");

pub const alignment: u32 = 64;

/// Rows are padded so each one starts aligned. `d_model` and `ffn_dim` are
/// multiples of 16 (validated), so the padding is at most 48 bytes per row.
pub fn rowStride(k: u32) u32 {
    assert(k > 0);
    return std.mem.alignForward(u32, k, alignment);
}

/// An `int8` matrix and its per-output-channel scales. SPEC §7.
pub const QuantMat = struct {
    data: u32,
    scales: u32,
    /// Byte offset of this matmul's static activation multiplier — Bergamot's
    /// "alpha" (ADR 0012). Zero when the artifact ships none, in which case the
    /// activation scale is computed dynamically per SPEC §7.
    alpha: u32,
    n: u32,
    k: u32,

    pub fn stride(self: QuantMat) u32 {
        return rowStride(self.k);
    }

    pub fn dataBytes(self: QuantMat) u32 {
        return self.n * self.stride();
    }

    pub fn shift(self: QuantMat, by: u32) QuantMat {
        return .{
            .data = self.data + by,
            .scales = self.scales + by,
            .alpha = if (self.alpha == 0) 0 else self.alpha + by,
            .n = self.n,
            .k = self.k,
        };
    }
};

pub const Attn = struct {
    q: QuantMat,
    q_bias: u32,
    k: QuantMat,
    k_bias: u32,
    v: QuantMat,
    v_bias: u32,
    o: QuantMat,
    o_bias: u32,
    ln_gain: u32,
    ln_bias: u32,

    fn shift(self: Attn, by: u32) Attn {
        return .{
            .q = self.q.shift(by),
            .q_bias = self.q_bias + by,
            .k = self.k.shift(by),
            .k_bias = self.k_bias + by,
            .v = self.v.shift(by),
            .v_bias = self.v_bias + by,
            .o = self.o.shift(by),
            .o_bias = self.o_bias + by,
            .ln_gain = self.ln_gain + by,
            .ln_bias = self.ln_bias + by,
        };
    }
};

pub const Ffn = struct {
    w1: QuantMat,
    bias1: u32,
    w2: QuantMat,
    bias2: u32,
    ln_gain: u32,
    ln_bias: u32,

    fn shift(self: Ffn, by: u32) Ffn {
        return .{
            .w1 = self.w1.shift(by),
            .bias1 = self.bias1 + by,
            .w2 = self.w2.shift(by),
            .bias2 = self.bias2 + by,
            .ln_gain = self.ln_gain + by,
            .ln_bias = self.ln_bias + by,
        };
    }
};

pub const EncLayer = struct {
    attn: Attn,
    ffn: Ffn,
};

/// Bergamot's decoder is an SSRU, not self-attention (ADR 0008). Two weight
/// matrices, one bias, one layer norm — and a state of exactly `d_model`
/// floats per layer, rather than a KV cache that grows with the output.
pub const Ssru = struct {
    w: QuantMat,
    wf: QuantMat,
    bf: u32,
    ln_gain: u32,
    ln_bias: u32,

    fn shift(self: Ssru, by: u32) Ssru {
        return .{
            .w = self.w.shift(by),
            .wf = self.wf.shift(by),
            .bf = self.bf + by,
            .ln_gain = self.ln_gain + by,
            .ln_bias = self.ln_bias + by,
        };
    }
};

pub const DecLayer = struct {
    ssru: Ssru,
    cross_attn: Attn,
    ffn: Ffn,
};

/// Sizes that hyper-parameters do not determine: they come from the artifact's
/// tensor table and are validated before they get here.
pub const Sizes = struct {
    piece_bytes: u32,
    prefix_bytes: u32 = 0,
    charsmap_bytes: u32 = 0,
    shortlist_nnz: u32,
    shortlist_frequent: u32,
};

pub const SlotLayout = struct {
    emb: QuantMat,
    emb_bias: u32,
    enc_ln_gain: u32,
    enc_ln_bias: u32,
    dec_ln_gain: u32,
    dec_ln_bias: u32,

    enc0: EncLayer,
    enc_stride: u32,
    dec0: DecLayer,
    dec_stride: u32,

    tok_pieces: u32,
    tok_pieces_len: u32,
    tok_offsets: u32,
    tok_scores: u32,
    tok_order: u32,
    tok_flags: u32,
    tok_prefixes: u32,
    tok_prefixes_len: u32,
    tok_charsmap: u32,
    tok_charsmap_len: u32,

    sl_offsets: u32,
    sl_targets: u32,
    sl_targets_len: u32,
    sl_frequent: u32,
    sl_frequent_len: u32,

    total: u32,

    pub fn encLayer(self: SlotLayout, i: u32) EncLayer {
        const by = i * self.enc_stride;
        return .{ .attn = self.enc0.attn.shift(by), .ffn = self.enc0.ffn.shift(by) };
    }

    pub fn decLayer(self: SlotLayout, i: u32) DecLayer {
        const by = i * self.dec_stride;
        return .{
            .ssru = self.dec0.ssru.shift(by),
            .cross_attn = self.dec0.cross_attn.shift(by),
            .ffn = self.dec0.ffn.shift(by),
        };
    }

    /// Null when the layout does not fit a `u32`, which is a `model_too_large`
    /// at the call site.
    pub fn compute(hp: format.HParams, sizes: Sizes) ?SlotLayout {
        const d: u32 = hp.d_model;
        const f: u32 = hp.ffn_dim;
        const v: u32 = hp.vocab_size;

        var c: Cursor = .{};
        var out: SlotLayout = undefined;

        out.emb = c.quant(v, d);
        out.emb_bias = c.take(@as(u64, v) * 4);
        out.enc_ln_gain = c.take(@as(u64, d) * 4);
        out.enc_ln_bias = c.take(@as(u64, d) * 4);
        out.dec_ln_gain = c.take(@as(u64, d) * 4);
        out.dec_ln_bias = c.take(@as(u64, d) * 4);

        const enc_base = c.cursor;
        out.enc0 = .{ .attn = c.attn(d), .ffn = c.ffn(d, f) };
        out.enc_stride = @intCast(c.cursor - enc_base);
        c.skip(@as(u64, out.enc_stride) * (hp.n_enc_layers - 1));

        const dec_base = c.cursor;
        out.dec0 = .{ .ssru = c.ssru(d), .cross_attn = c.attn(d), .ffn = c.ffn(d, f) };
        out.dec_stride = @intCast(c.cursor - dec_base);
        c.skip(@as(u64, out.dec_stride) * (hp.n_dec_layers - 1));

        out.tok_pieces = c.take(sizes.piece_bytes);
        out.tok_pieces_len = sizes.piece_bytes;
        out.tok_offsets = c.take((@as(u64, v) + 1) * 4);
        out.tok_scores = c.take(@as(u64, v) * 4);
        out.tok_order = c.take(@as(u64, v) * 4);
        out.tok_flags = c.take(v);
        out.tok_prefixes = c.take(sizes.prefix_bytes);
        out.tok_prefixes_len = sizes.prefix_bytes;
        out.tok_charsmap = c.take(sizes.charsmap_bytes);
        out.tok_charsmap_len = sizes.charsmap_bytes;

        out.sl_offsets = c.take((@as(u64, v) + 1) * 4);
        out.sl_targets = c.take(@as(u64, sizes.shortlist_nnz) * 2);
        out.sl_targets_len = sizes.shortlist_nnz;
        out.sl_frequent = c.take(@as(u64, sizes.shortlist_frequent) * 4);
        out.sl_frequent_len = sizes.shortlist_frequent;

        if (c.overflowed or c.cursor > std.math.maxInt(u32)) return null;
        out.total = @intCast(c.cursor);
        assert(out.total % alignment == 0);
        assert(out.enc_stride % alignment == 0);
        return out;
    }
};

const Cursor = struct {
    cursor: u64 = 0,
    overflowed: bool = false,

    fn take(self: *Cursor, len: u64) u32 {
        assert(self.cursor % alignment == 0);
        const off = self.cursor;
        self.cursor = std.mem.alignForward(u64, off + len, alignment);
        if (self.cursor > std.math.maxInt(u32)) {
            self.overflowed = true;
            return 0;
        }
        return @intCast(off);
    }

    fn skip(self: *Cursor, len: u64) void {
        assert(len % alignment == 0);
        self.cursor += len;
        if (self.cursor > std.math.maxInt(u32)) self.overflowed = true;
    }

    fn quant(self: *Cursor, n: u32, k: u32) QuantMat {
        const data = self.take(@as(u64, n) * rowStride(k));
        const scales = self.take(@as(u64, n) * 4);
        const alpha = self.take(4);
        return .{ .data = data, .scales = scales, .alpha = alpha, .n = n, .k = k };
    }

    fn ssru(self: *Cursor, d: u32) Ssru {
        return .{
            .w = self.quant(d, d),
            .wf = self.quant(d, d),
            .bf = self.take(@as(u64, d) * 4),
            .ln_gain = self.take(@as(u64, d) * 4),
            .ln_bias = self.take(@as(u64, d) * 4),
        };
    }

    fn attn(self: *Cursor, d: u32) Attn {
        return .{
            .q = self.quant(d, d),
            .q_bias = self.take(@as(u64, d) * 4),
            .k = self.quant(d, d),
            .k_bias = self.take(@as(u64, d) * 4),
            .v = self.quant(d, d),
            .v_bias = self.take(@as(u64, d) * 4),
            .o = self.quant(d, d),
            .o_bias = self.take(@as(u64, d) * 4),
            .ln_gain = self.take(@as(u64, d) * 4),
            .ln_bias = self.take(@as(u64, d) * 4),
        };
    }

    fn ffn(self: *Cursor, d: u32, f: u32) Ffn {
        return .{
            .w1 = self.quant(f, d),
            .bias1 = self.take(@as(u64, f) * 4),
            .w2 = self.quant(d, f),
            .bias2 = self.take(@as(u64, d) * 4),
            .ln_gain = self.take(@as(u64, d) * 4),
            .ln_bias = self.take(@as(u64, d) * 4),
        };
    }
};

// -- tests ------------------------------------------------------------------

const testing = std.testing;

fn bergamotHParams() format.HParams {
    return .{
        .d_model = 256,
        .ffn_dim = 1536,
        .n_enc_layers = 6,
        .n_dec_layers = 2,
        .n_heads = 8,
        .head_dim = 32,
        .vocab_size = 32000,
        .max_pos = 512,
        .shortlist_width = 2048,
        .eos_id = 0,
        .bos_id = 0,
        .unk_id = 1,
        .pad_id = 0,
        .ffn_act = 0,
        .prenorm = 0,
        .tied_embeddings = 1,
        .act_quant = 0,
        .emb_scale = 16.0,
        .norm_eps = 1e-9,
        .max_length_factor = 3.0,
    };
}

test "every offset and every row is 64-byte aligned" {
    const hp = bergamotHParams();
    const l = SlotLayout.compute(hp, .{
        .piece_bytes = 300_000,
        .shortlist_nnz = 2_000_000,
        .shortlist_frequent = 1024,
    }).?;

    try testing.expectEqual(@as(u32, 0), l.emb.data % alignment);
    try testing.expectEqual(@as(u32, 0), l.emb.stride() % alignment);
    try testing.expectEqual(@as(u32, 0), l.enc_stride % alignment);
    try testing.expectEqual(@as(u32, 0), l.dec_stride % alignment);

    for (0..hp.n_enc_layers) |i| {
        const e = l.encLayer(@intCast(i));
        try testing.expectEqual(@as(u32, 0), e.attn.q.data % alignment);
        try testing.expectEqual(@as(u32, 0), e.ffn.w2.data % alignment);
        try testing.expect(e.ffn.w2.data + e.ffn.w2.dataBytes() <= l.total);
    }
    for (0..hp.n_dec_layers) |i| {
        const dl = l.decLayer(@intCast(i));
        try testing.expectEqual(@as(u32, 0), dl.ssru.w.data % alignment);
        try testing.expectEqual(@as(u32, 0), dl.cross_attn.v.data % alignment);
        try testing.expect(dl.ffn.ln_bias < l.total);
    }
}

test "layers do not overlap" {
    const hp = bergamotHParams();
    const l = SlotLayout.compute(hp, .{
        .piece_bytes = 1024,
        .shortlist_nnz = 16,
        .shortlist_frequent = 4,
    }).?;

    var prev: u32 = 0;
    for (0..hp.n_enc_layers) |i| {
        const e = l.encLayer(@intCast(i));
        try testing.expect(e.attn.q.data > prev);
        prev = e.ffn.ln_bias;
    }
    for (0..hp.n_dec_layers) |i| {
        const dl = l.decLayer(@intCast(i));
        try testing.expect(dl.ssru.w.data > prev);
        prev = dl.ffn.ln_bias;
    }
    try testing.expect(l.tok_pieces > prev);
}

test "a Bergamot-shaped model lands inside the SPEC §14 per-direction budget" {
    // SPEC §4.3: ~17 M parameters, so ~17 MB of int8 weights. SPEC §14 budgets
    // 20 MB per direction, and SPEC §6 puts the vocabulary and the shortlist in
    // the same file — so those three megabytes are the whole margin, and the
    // shortlist is the only term the converter can tune. At 32k source pieces
    // that is roughly 18 candidates each.
    const l = SlotLayout.compute(bergamotHParams(), .{
        .piece_bytes = 300_000,
        .shortlist_nnz = 600_000,
        .shortlist_frequent = 1024,
    }).?;
    try testing.expect(l.total < 20 << 20);
    try testing.expect(l.total > 16 << 20);
}

test "the shortlist is what pushes a direction over budget" {
    // Recorded so the tradeoff is visible: `tools/convert.py --shortlist-best`
    // buys coverage with slot bytes and nothing else does.
    const weights_only = SlotLayout.compute(bergamotHParams(), .{
        .piece_bytes = 300_000,
        .shortlist_nnz = 0,
        .shortlist_frequent = 0,
    }).?;
    const generous = SlotLayout.compute(bergamotHParams(), .{
        .piece_bytes = 300_000,
        .shortlist_nnz = 1_600_000, // Bergamot's own "best 50 per source"
        .shortlist_frequent = 1024,
    }).?;
    try testing.expect(weights_only.total < 18 << 20);
    try testing.expect(generous.total > 20 << 20);
}
