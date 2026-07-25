//! model/format.zig — the `.fzm` artifact parser. SPEC §6, §11.
//!
//! THE validation boundary. Everything in this file runs on bytes the runtime
//! did not produce, so everything in this file returns a status instead of
//! asserting. Once `load` has returned `.ok`, downstream code asserts freely,
//! because the invariants were established here, once, by code that cannot
//! trap.
//!
//! Layout, little-endian throughout (SPEC §6):
//!
//!     0    magic         [4]u8 "FIZH"
//!     4    version       u32
//!     8    src_lang      u32
//!     12   tgt_lang      u32
//!     16   hparams       [48]u8
//!     64   tensor_count  u32
//!     68   reserved      [60]u8, zero
//!     128  tensors       [tensor_count]TensorDesc, 56 bytes each
//!     ..   payload       64-byte-aligned tensor bytes
//!
//! The header is 128 bytes rather than 64 because language codes went from two
//! packed ASCII bytes to four (`abi.Lang`), which did not fit. The 60 reserved
//! bytes exist so the next field does not cost another format version.

const std = @import("std");
const assert = std.debug.assert;

const abi = @import("../abi.zig");
const layout = @import("layout.zig");
const names = @import("names.zig");
const repack = @import("repack.zig");
const math = @import("../kernel/math.zig");
const trie = @import("../tok/trie.zig");
const tok_charsmap = @import("../tok/charsmap.zig");

pub const magic = [4]u8{ 'F', 'I', 'Z', 'H' };
/// 3 widens `src_lang`/`tgt_lang` to `u32` and the header to 128 bytes.
/// 2 added: `act_quant` in the header, per-matmul `*.alpha` tensors,
/// `tok.nonbreaking`, `sl.targets` narrowed to u16, and `tok.charsmap`.
pub const version: u32 = 3;

/// SPEC §6: all offsets 64-byte aligned.
pub const payload_alignment: u64 = 64;
pub const header_bytes: u64 = 128;
/// A ceiling so a corrupt `tensor_count` cannot make the table walk forever.
pub const max_tensors: u32 = 4096;

pub const DType = enum(u8) {
    f32 = 0,
    i8 = 1,
    u32 = 2,
    u8 = 3,
    /// Shortlist target ids. A vocabulary of 32k needs 15 bits, and at
    /// Bergamot's own "best 50 per source" the difference between `u16` and
    /// `u32` is 1.5 MB — the difference between fitting SPEC §14's 20 MB and
    /// not. See ADR 0010.
    u16 = 4,
    _,

    pub fn size(self: DType) ?u32 {
        return switch (self) {
            .f32, .u32 => 4,
            .i8, .u8 => 1,
            .u16 => 2,
            _ => null,
        };
    }
};

/// SPEC §6.
pub const TensorDesc = extern struct {
    name_hash: u64,
    dtype: u8,
    rank: u8,
    _pad: u16,
    dims: [4]u32,
    /// The hole a C compiler would insert before the first `u64`, made
    /// explicit. Same bytes on the wire; one fewer way to leak stack garbage
    /// into an artifact.
    _pad2: u32,
    offset: u64,
    nbytes: u64,
    /// Byte offset of the per-output-channel `f32` scale vector, or
    /// `no_scales` for tensors that carry none.
    scale_offset: u64,

    pub const no_scales: u64 = std.math.maxInt(u64);

    /// SPEC §6 lists the fields; this is what they weigh under C layout.
    pub const wire_size: u64 = 56;

    comptime {
        assert(@sizeOf(TensorDesc) == wire_size);
        assert(@offsetOf(TensorDesc, "dims") == 12);
        assert(@offsetOf(TensorDesc, "offset") == 32);
    }

    pub fn elemCount(self: TensorDesc) u64 {
        var n: u64 = 1;
        for (self.dims[0..@min(self.rank, 4)]) |d| n *= d;
        return n;
    }
};

/// SPEC §6: `hparams [48]u8 packed`. No holes; the layout below is exactly 48
/// bytes with every field naturally aligned, so a `@bitCast` from the raw bytes
/// is well defined.
pub const HParams = extern struct {
    d_model: u16,
    ffn_dim: u16,
    n_enc_layers: u8,
    n_dec_layers: u8,
    n_heads: u8,
    head_dim: u8,
    vocab_size: u32,
    /// Positional encodings are generated into the arena up to this many
    /// steps; a source or target longer than this is a load-time rejection.
    max_pos: u16,
    /// Shortlist width the converter recommends; clamped to `max_shortlist`.
    shortlist_width: u16,
    eos_id: u32,
    bos_id: u32,
    unk_id: u32,
    pad_id: u32,
    ffn_act: u8,
    /// 0 = post-norm (Marian default), 1 = pre-norm.
    prenorm: u8,
    tied_embeddings: u8,
    /// 0 = dynamic per-row absmax (SPEC §7). 1 = the static per-matmul
    /// multipliers Bergamot ships as `*_QuantMultA` (ADR 0012).
    act_quant: u8,
    emb_scale: f32,
    norm_eps: f32,
    /// SPEC §12.3: generation is bounded by this times the source length as
    /// well as by `max_tgt_tokens`.
    max_length_factor: f32,

    comptime {
        assert(@sizeOf(HParams) == 48);
    }

    pub const Act = enum(u8) { relu = 0, gelu = 1, swish = 2, _ };

    pub fn act(self: HParams) Act {
        return @enumFromInt(self.ffn_act);
    }

    /// Untrusted. Every branch is a validation error.
    pub fn validate(self: HParams, cfg: abi.Config) ?abi.Status {
        if (self.d_model == 0 or self.d_model > cfg.max_d_model) return .model_too_large;
        if (self.ffn_dim == 0 or self.ffn_dim > cfg.max_ffn_dim) return .model_too_large;
        if (self.n_enc_layers == 0 or self.n_enc_layers > cfg.max_enc_layers) return .model_too_large;
        if (self.n_dec_layers == 0 or self.n_dec_layers > cfg.max_dec_layers) return .model_too_large;
        if (self.n_heads == 0 or self.n_heads > cfg.max_heads) return .model_too_large;
        if (self.vocab_size < 2 or self.vocab_size > cfg.max_vocab) return .model_too_large;
        if (self.max_pos < @max(cfg.max_src_tokens, cfg.max_tgt_tokens)) return .model_too_large;

        if (self.d_model % 16 != 0) return .bad_artifact;
        if (self.ffn_dim % 16 != 0) return .bad_artifact;
        if (@as(u32, self.n_heads) * self.head_dim != self.d_model) return .bad_artifact;
        // Positional encodings pair a sine with a cosine.
        if (self.d_model % 2 != 0) return .bad_artifact;

        if (self.eos_id >= self.vocab_size) return .bad_artifact;
        if (self.bos_id >= self.vocab_size) return .bad_artifact;
        if (self.unk_id >= self.vocab_size) return .bad_artifact;
        if (self.pad_id >= self.vocab_size) return .bad_artifact;

        if (self.ffn_act > 2) return .bad_artifact;
        if (self.prenorm > 1) return .bad_artifact;
        if (self.tied_embeddings > 1) return .bad_artifact;
        if (self.act_quant > 1) return .bad_artifact;

        // SPEC §12.6: negative-space assertions, here as validation.
        if (!std.math.isFinite(self.emb_scale) or self.emb_scale <= 0) return .bad_artifact;
        if (!std.math.isFinite(self.norm_eps) or self.norm_eps <= 0) return .bad_artifact;
        if (!std.math.isFinite(self.max_length_factor)) return .bad_artifact;
        if (self.max_length_factor < 1.0 or self.max_length_factor > 8.0) return .bad_artifact;
        return null;
    }
};

/// A loaded direction. Everything here is post-validation: downstream code
/// asserts against these values rather than checking them.
pub const Model = struct {
    loaded: bool,
    src_lang: abi.Lang,
    tgt_lang: abi.Lang,
    hp: HParams,
    /// Arena byte offset of this slot's repacked weights.
    slot_base: u32,
    /// Bytes actually used inside the slot.
    slot_used: u32,
    sl: layout.SlotLayout,
    max_piece_len: u32,
    min_score: f32,

    pub const empty: Model = .{
        .loaded = false,
        .src_lang = 0,
        .tgt_lang = 0,
        .hp = std.mem.zeroes(HParams),
        .slot_base = 0,
        .slot_used = 0,
        .sl = std.mem.zeroes(layout.SlotLayout),
        .max_piece_len = 0,
        .min_score = 0,
    };

    /// Non-breaking prefixes for the source language (`tok.nonbreaking`).
    /// Empty when the artifact predates them, which simply disables that rule
    /// in `tok/ssplit.zig`.
    /// The `nmt_nfkc` rewrite table (`tok.charsmap`), empty when the artifact
    /// carries none. ADR 0017.
    pub fn charsmap(self: *const Model, slot: []const u8) tok_charsmap.Charsmap {
        if (self.sl.tok_charsmap_len == 0) return tok_charsmap.none;
        return .{ .blob = slot[self.sl.tok_charsmap..][0..self.sl.tok_charsmap_len] };
    }

    /// This model's positional encodings, a view into its own slot. Never
    /// shared: see `SlotLayout.pos_enc` and SPEC §4.1.
    pub fn posEnc(self: *const Model, slot: []u8) []f32 {
        assert(self.sl.pos_enc % 4 == 0);
        const at = slot[self.sl.pos_enc..][0 .. @as(usize, self.sl.pos_enc_len) * 4];
        return @alignCast(std.mem.bytesAsSlice(f32, at));
    }

    /// Read-only view of the same region, for a pass.
    pub fn posEncConst(self: *const Model, slot: []const u8) []const f32 {
        assert(self.loaded);
        const at = slot[self.sl.pos_enc..][0 .. @as(usize, self.sl.pos_enc_len) * 4];
        return @alignCast(std.mem.bytesAsSlice(f32, at));
    }

    pub fn prefixes(self: *const Model, slot: []const u8) []const u8 {
        assert(self.loaded);
        if (self.sl.tok_prefixes_len == 0) return &.{};
        return slot[self.sl.tok_prefixes..][0..self.sl.tok_prefixes_len];
    }

    /// The tokenizer's view of this model's slot.
    pub fn vocab(self: *const Model, slot: []const u8) trie.Vocab {
        assert(self.loaded);
        const v = self.hp.vocab_size;
        return .{
            .pieces = slot[self.sl.tok_pieces..][0..self.sl.tok_pieces_len],
            .offsets = u32View(slot, self.sl.tok_offsets, v + 1),
            .scores = f32View(slot, self.sl.tok_scores, v),
            .order = u32View(slot, self.sl.tok_order, v),
            .flags = slot[self.sl.tok_flags..][0..v],
            .size = v,
            .max_piece_len = self.max_piece_len,
            .min_score = self.min_score,
        };
    }
};

pub fn u32View(slot: []const u8, off: u32, count: u32) []const u32 {
    assert(off % layout.alignment == 0);
    const p: [*]const u32 = @ptrCast(@alignCast(slot.ptr + off));
    return p[0..count];
}

pub fn u16View(slot: []const u8, off: u32, count: u32) []const u16 {
    assert(off % layout.alignment == 0);
    const p: [*]const u16 = @ptrCast(@alignCast(slot.ptr + off));
    return p[0..count];
}

pub fn i8View(slot: []const u8, off: u32, count: u32) []const i8 {
    assert(off % layout.alignment == 0);
    assert(off + count <= slot.len);
    const p: [*]const i8 = @ptrCast(slot.ptr + off);
    return p[0..count];
}

pub fn f32View(slot: []const u8, off: u32, count: u32) []const f32 {
    assert(off % layout.alignment == 0);
    const p: [*]const f32 = @ptrCast(@alignCast(slot.ptr + off));
    return p[0..count];
}

// -- loading ----------------------------------------------------------------

const LoadError = error{
    BadArtifact,
    BadVersion,
    BadLang,
    MissingTensor,
    ModelTooLarge,
};

fn statusOf(e: LoadError) abi.Status {
    return switch (e) {
        error.BadArtifact => .bad_artifact,
        error.BadVersion => .bad_version,
        error.BadLang => .bad_lang,
        error.MissingTensor => .missing_tensor,
        error.ModelTooLarge => .model_too_large,
    };
}

/// Parses, validates and repacks `blob` into `slot`, and fills `pos_enc` with
/// the positional table this model's `d_model` implies.
///
/// `blob` must not alias `slot` or `pos_enc`.
pub fn load(
    blob: []const u8,
    cfg: abi.Config,
    slot: []u8,
    out: *Model,
) abi.Status {
    assert(slot.len == cfg.max_model_bytes);
    assert(@intFromPtr(slot.ptr) % layout.alignment == 0);

    out.loaded = false;
    loadInner(blob, cfg, slot, out) catch |e| return statusOf(e);

    assert(out.loaded);
    assert(out.slot_used <= slot.len);
    return .ok;
}

fn loadInner(
    blob: []const u8,
    cfg: abi.Config,
    slot: []u8,
    out: *Model,
) LoadError!void {
    const head = try parseHeader(blob, cfg);
    try validateTable(blob, head.count);

    var sizes = try measure(blob, head.count, head.hp.vocab_size);
    // Per-model derived data is carved in the model's own slot (SPEC §4.1), so
    // the host's step ceiling has to reach the slot layout.
    sizes.pos_steps = @max(cfg.max_src_tokens, cfg.max_tgt_tokens);
    const sl = layout.SlotLayout.compute(head.hp, sizes) orelse return error.ModelTooLarge;
    if (sl.total > slot.len) return error.ModelTooLarge;

    var ctx: Ctx = .{ .blob = blob, .count = head.count, .slot = slot, .sl = sl, .hp = head.hp };
    try loadShared(&ctx);
    try loadEncoder(&ctx);
    try loadDecoder(&ctx);
    try loadTokenizer(&ctx);
    try loadShortlist(&ctx);

    out.hp = head.hp;
    out.src_lang = head.src_lang;
    out.tgt_lang = head.tgt_lang;
    out.sl = sl;
    out.slot_used = sl.total;
    out.max_piece_len = ctx.max_piece_len;
    out.min_score = ctx.min_score;
    out.loaded = true;

    if (!out.vocab(slot).validate()) {
        out.loaded = false;
        return error.BadArtifact;
    }
    fillPositional(out.posEnc(slot), head.hp.d_model, sizes.pos_steps);
}

const Header = struct {
    src_lang: abi.Lang,
    tgt_lang: abi.Lang,
    hp: HParams,
    count: u32,
};

/// The hyper-parameters, read without validating them against any config.
///
/// A host that wants to size its arena *from* an artifact rather than reject an
/// artifact that does not fit a guessed arena needs this: Firefox's registry
/// ships more than one student architecture, and `d_model=384, n_dec=4` is as
/// real as `256, 2`. Returns null only when the blob is too short or not a
/// `.fzm` — everything else is the caller's judgement.
pub fn peekHParams(blob: []const u8) ?HParams {
    if (blob.len < header_bytes) return null;
    if (!std.mem.eql(u8, blob[0..4], &magic)) return null;
    if (readU32(blob, 4) != version) return null;

    var raw: [48]u8 align(@alignOf(HParams)) = undefined;
    @memcpy(&raw, blob[16..64]);
    return @bitCast(raw);
}

fn parseHeader(blob: []const u8, cfg: abi.Config) LoadError!Header {
    if (blob.len < header_bytes) return error.BadArtifact;
    if (!std.mem.eql(u8, blob[0..4], &magic)) return error.BadArtifact;
    if (readU32(blob, 4) != version) return error.BadVersion;

    const src_lang = readU32(blob, 8);
    const tgt_lang = readU32(blob, 12);
    if (!abi.langValid(src_lang) or !abi.langValid(tgt_lang)) return error.BadLang;
    if (src_lang == tgt_lang) return error.BadLang;

    var raw: [48]u8 align(@alignOf(HParams)) = undefined;
    @memcpy(&raw, blob[16..64]);
    const hp: HParams = @bitCast(raw);
    if (hp.validate(cfg)) |bad| {
        return if (bad == .model_too_large) error.ModelTooLarge else error.BadArtifact;
    }

    const count = readU32(blob, 64);
    if (count == 0 or count > max_tensors) return error.BadArtifact;
    return .{ .src_lang = src_lang, .tgt_lang = tgt_lang, .hp = hp, .count = count };
}

/// Every descriptor must describe a region that exists, is aligned, and whose
/// element count agrees with its byte count. Checked for all of them before any
/// of them is used, so a later failure cannot leave a half-copied slot behind a
/// valid-looking header.
fn validateTable(blob: []const u8, count: u32) LoadError!void {
    const table_end = header_bytes + @as(u64, count) * TensorDesc.wire_size;
    if (table_end > blob.len) return error.BadArtifact;

    for (0..count) |i| {
        const d = readDesc(blob, @intCast(i));
        const elem = DType.size(@enumFromInt(d.dtype)) orelse return error.BadArtifact;
        if (d.rank == 0 or d.rank > 4) return error.BadArtifact;
        if (d._pad != 0 or d._pad2 != 0) return error.BadArtifact;
        if (d.elemCount() * elem != d.nbytes) return error.BadArtifact;
        if (d.nbytes == 0) return error.BadArtifact;
        if (d.offset % payload_alignment != 0) return error.BadArtifact;
        if (d.offset < table_end) return error.BadArtifact;
        if (d.offset + d.nbytes > blob.len) return error.BadArtifact;

        if (d.scale_offset != TensorDesc.no_scales) {
            if (d.scale_offset % payload_alignment != 0) return error.BadArtifact;
            const scale_bytes = @as(u64, d.dims[0]) * 4;
            if (d.scale_offset + scale_bytes > blob.len) return error.BadArtifact;
        }
    }
}

/// The three tensors whose size the hyper-parameters do not determine.
fn measure(blob: []const u8, count: u32, vocab_size: u32) LoadError!layout.Sizes {
    const pieces = find(blob, count, names.tok_pieces) orelse return error.MissingTensor;
    // Optional: artifacts written before sentence splitting existed have none,
    // and an empty list simply disables the prefix rule (ADR 0011).
    const prefixes = find(blob, count, names.tok_nonbreaking);
    // Optional for the same reason: without it the tokenizer sees the raw
    // bytes, which is what fizh did before ADR 0017.
    const charsmap = find(blob, count, names.tok_charsmap);
    const targets = find(blob, count, names.sl_targets) orelse return error.MissingTensor;
    const frequent = find(blob, count, names.sl_frequent) orelse return error.MissingTensor;

    if (pieces.nbytes > std.math.maxInt(u32)) return error.ModelTooLarge;
    if (targets.nbytes > std.math.maxInt(u32)) return error.ModelTooLarge;
    if (frequent.nbytes > std.math.maxInt(u32)) return error.ModelTooLarge;
    if (frequent.nbytes / 4 > vocab_size) return error.BadArtifact;
    // A vocabulary this format can address must fit the id width.
    if (vocab_size > std.math.maxInt(u16) + 1) return error.ModelTooLarge;

    if (prefixes) |p| {
        if (p.nbytes > 1 << 16) return error.BadArtifact;
    }
    if (charsmap) |c| {
        // A darts-clone blob is a u32 length followed by whole units.
        if (c.nbytes < 4 or c.nbytes > 1 << 22) return error.BadArtifact;
    }

    return .{
        .piece_bytes = @intCast(pieces.nbytes),
        .prefix_bytes = if (prefixes) |p| @intCast(p.nbytes) else 0,
        .charsmap_bytes = if (charsmap) |c| @intCast(c.nbytes) else 0,
        .shortlist_nnz = @intCast(targets.nbytes / 2),
        .shortlist_frequent = @intCast(frequent.nbytes / 4),
    };
}

const Ctx = struct {
    blob: []const u8,
    count: u32,
    slot: []u8,
    sl: layout.SlotLayout,
    hp: HParams,
    max_piece_len: u32 = 0,
    min_score: f32 = 0,
};

fn loadShared(c: *Ctx) LoadError!void {
    const d = c.hp.d_model;
    try quant(c, names.emb, names.alphaOf(names.emb), c.sl.emb);
    try vecF32(c, names.emb_bias, c.sl.emb_bias, c.hp.vocab_size);
    try vecF32(c, names.enc_ln_gain, c.sl.enc_ln_gain, d);
    try vecF32(c, names.enc_ln_bias, c.sl.enc_ln_bias, d);
    try vecF32(c, names.dec_ln_gain, c.sl.dec_ln_gain, d);
    try vecF32(c, names.dec_ln_bias, c.sl.dec_ln_bias, d);
}

fn loadEncoder(c: *Ctx) LoadError!void {
    for (0..c.hp.n_enc_layers) |raw| {
        const i: u32 = @intCast(raw);
        const l = c.sl.encLayer(i);
        try attn(c, .encoder, i, "att", l.attn);
        try ffn(c, .encoder, i, l.ffn);
    }
}

fn loadDecoder(c: *Ctx) LoadError!void {
    for (0..c.hp.n_dec_layers) |raw| {
        const i: u32 = @intCast(raw);
        const l = c.sl.decLayer(i);
        try ssru(c, i, l.ssru);
        try attn(c, .decoder, i, "xa", l.cross_attn);
        try ffn(c, .decoder, i, l.ffn);
    }
}

const Side = enum { encoder, decoder };

fn nameOf(side: Side, i: u32, comptime stem: []const u8) u64 {
    return switch (side) {
        .encoder => names.enc(i, stem),
        .decoder => names.dec(i, stem),
    };
}

/// SPEC §8's decoder self-attention slot, filled by Bergamot's SSRU (ADR 0008).
fn ssru(c: *Ctx, i: u32, a: layout.Ssru) LoadError!void {
    const d = c.hp.d_model;
    try quant(c, names.dec(i, "rnn.w"), names.dec(i, "rnn.w.alpha"), a.w);
    try quant(c, names.dec(i, "rnn.wf"), names.dec(i, "rnn.wf.alpha"), a.wf);
    try vecF32(c, names.dec(i, "rnn.bf"), a.bf, d);
    try vecF32(c, names.dec(i, "rnn.ln.gain"), a.ln_gain, d);
    try vecF32(c, names.dec(i, "rnn.ln.bias"), a.ln_bias, d);
}

fn attn(c: *Ctx, side: Side, i: u32, comptime tag: []const u8, a: layout.Attn) LoadError!void {
    const d = c.hp.d_model;
    try quant(c, nameOf(side, i, tag ++ ".q.w"), nameOf(side, i, tag ++ ".q.w.alpha"), a.q);
    try vecF32(c, nameOf(side, i, tag ++ ".q.bias"), a.q_bias, d);
    try quant(c, nameOf(side, i, tag ++ ".k.w"), nameOf(side, i, tag ++ ".k.w.alpha"), a.k);
    try vecF32(c, nameOf(side, i, tag ++ ".k.bias"), a.k_bias, d);
    try quant(c, nameOf(side, i, tag ++ ".v.w"), nameOf(side, i, tag ++ ".v.w.alpha"), a.v);
    try vecF32(c, nameOf(side, i, tag ++ ".v.bias"), a.v_bias, d);
    try quant(c, nameOf(side, i, tag ++ ".o.w"), nameOf(side, i, tag ++ ".o.w.alpha"), a.o);
    try vecF32(c, nameOf(side, i, tag ++ ".o.bias"), a.o_bias, d);
    try vecF32(c, nameOf(side, i, tag ++ ".ln.gain"), a.ln_gain, d);
    try vecF32(c, nameOf(side, i, tag ++ ".ln.bias"), a.ln_bias, d);
}

fn ffn(c: *Ctx, side: Side, i: u32, f: layout.Ffn) LoadError!void {
    const d = c.hp.d_model;
    try quant(c, nameOf(side, i, "ffn.w1"), nameOf(side, i, "ffn.w1.alpha"), f.w1);
    try vecF32(c, nameOf(side, i, "ffn.bias1"), f.bias1, c.hp.ffn_dim);
    try quant(c, nameOf(side, i, "ffn.w2"), nameOf(side, i, "ffn.w2.alpha"), f.w2);
    try vecF32(c, nameOf(side, i, "ffn.bias2"), f.bias2, d);
    try vecF32(c, nameOf(side, i, "ffn.ln.gain"), f.ln_gain, d);
    try vecF32(c, nameOf(side, i, "ffn.ln.bias"), f.ln_bias, d);
}

fn loadTokenizer(c: *Ctx) LoadError!void {
    const v = c.hp.vocab_size;
    try rawBytes(c, names.tok_pieces, c.sl.tok_pieces, .u8, c.sl.tok_pieces_len);
    try vecU32(c, names.tok_offsets, c.sl.tok_offsets, v + 1);
    try vecF32(c, names.tok_scores, c.sl.tok_scores, v);
    try vecU32(c, names.tok_order, c.sl.tok_order, v);
    try rawBytes(c, names.tok_flags, c.sl.tok_flags, .u8, v);
    if (c.sl.tok_prefixes_len != 0) {
        try rawBytes(c, names.tok_nonbreaking, c.sl.tok_prefixes, .u8, c.sl.tok_prefixes_len);
    }
    if (c.sl.tok_charsmap_len != 0) {
        try rawBytes(c, names.tok_charsmap, c.sl.tok_charsmap, .u8, c.sl.tok_charsmap_len);
    }

    const offs = u32View(c.slot, c.sl.tok_offsets, v + 1);
    var longest: u32 = 0;
    for (1..offs.len) |i| {
        if (offs[i] < offs[i - 1]) return error.BadArtifact;
        longest = @max(longest, offs[i] - offs[i - 1]);
    }
    c.max_piece_len = longest;

    const scores = f32View(c.slot, c.sl.tok_scores, v);
    var worst: f32 = std.math.inf(f32);
    for (scores) |s| worst = @min(worst, s);
    c.min_score = worst;
}

fn loadShortlist(c: *Ctx) LoadError!void {
    const v = c.hp.vocab_size;
    try vecU32(c, names.sl_offsets, c.sl.sl_offsets, v + 1);
    try vecU16(c, names.sl_targets, c.sl.sl_targets, c.sl.sl_targets_len);
    try vecU32(c, names.sl_frequent, c.sl.sl_frequent, c.sl.sl_frequent_len);

    const offs = u32View(c.slot, c.sl.sl_offsets, v + 1);
    if (offs[0] != 0) return error.BadArtifact;
    if (offs[v] != c.sl.sl_targets_len) return error.BadArtifact;
    for (1..offs.len) |i| {
        if (offs[i] < offs[i - 1]) return error.BadArtifact;
    }
    for (u16View(c.slot, c.sl.sl_targets, c.sl.sl_targets_len)) |t| {
        if (t >= v) return error.BadArtifact;
    }
    for (u32View(c.slot, c.sl.sl_frequent, c.sl.sl_frequent_len)) |t| {
        if (t >= v) return error.BadArtifact;
    }
}

// -- descriptor plumbing ----------------------------------------------------

fn quant(c: *Ctx, hash: u64, alpha_hash: u64, m: layout.QuantMat) LoadError!void {
    const d = find(c.blob, c.count, hash) orelse return error.MissingTensor;
    if (@as(DType, @enumFromInt(d.dtype)) != .i8) return error.BadArtifact;
    if (d.rank != 2 or d.dims[0] != m.n or d.dims[1] != m.k) return error.BadArtifact;
    if (d.scale_offset == TensorDesc.no_scales) return error.BadArtifact;

    const data = c.blob[@intCast(d.offset)..][0..@intCast(d.nbytes)];
    const scales = c.blob[@intCast(d.scale_offset)..][0 .. m.n * 4];
    repack.quantMatrix(c.slot, m, data, scales) catch return error.BadArtifact;

    // The static activation multiplier, present only when the converter was
    // asked for it. `act_quant == 1` without one is a malformed artifact, not
    // a silent fallback: the two paths produce different numbers.
    const a = find(c.blob, c.count, alpha_hash);
    if (c.hp.act_quant == 1) {
        const t = a orelse return error.MissingTensor;
        if (@as(DType, @enumFromInt(t.dtype)) != .f32 or t.rank != 1 or t.dims[0] != 1) {
            return error.BadArtifact;
        }
        repack.floats(c.slot, m.alpha, c.blob[@intCast(t.offset)..][0..4], 1) catch return error.BadArtifact;
        const v = f32View(c.slot, m.alpha, 1)[0];
        if (!(v > 0)) return error.BadArtifact;
    }
}

fn vecF32(c: *Ctx, hash: u64, off: u32, count: u32) LoadError!void {
    const d = try expectVec(c, hash, .f32, count);
    const src = c.blob[@intCast(d.offset)..][0 .. count * 4];
    repack.floats(c.slot, off, src, count) catch return error.BadArtifact;
}

fn vecU16(c: *Ctx, hash: u64, off: u32, count: u32) LoadError!void {
    const d = try expectVec(c, hash, .u16, count);
    repack.bytes(c.slot, off, c.blob[@intCast(d.offset)..][0 .. count * 2]);
}

fn vecU32(c: *Ctx, hash: u64, off: u32, count: u32) LoadError!void {
    const d = try expectVec(c, hash, .u32, count);
    repack.bytes(c.slot, off, c.blob[@intCast(d.offset)..][0 .. count * 4]);
}

fn rawBytes(c: *Ctx, hash: u64, off: u32, dt: DType, count: u32) LoadError!void {
    const d = try expectVec(c, hash, dt, count);
    repack.bytes(c.slot, off, c.blob[@intCast(d.offset)..][0..count]);
}

fn expectVec(c: *Ctx, hash: u64, dt: DType, count: u32) LoadError!TensorDesc {
    const d = find(c.blob, c.count, hash) orelse return error.MissingTensor;
    if (@as(DType, @enumFromInt(d.dtype)) != dt) return error.BadArtifact;
    if (d.rank != 1 or d.dims[0] != count) return error.BadArtifact;
    if (d.scale_offset != TensorDesc.no_scales) return error.BadArtifact;
    return d;
}

/// Linear scan. ~160 lookups over at most 4096 descriptors is a few hundred
/// thousand `u64` compares — microseconds against a 300 ms cold-start budget —
/// and it needs no scratch, no sort, and no static state. SPEC §6 asks for no
/// string compare at load, and there is none.
fn find(blob: []const u8, count: u32, hash: u64) ?TensorDesc {
    for (0..count) |i| {
        const at = header_bytes + @as(u64, i) * TensorDesc.wire_size;
        if (readU64(blob, at) != hash) continue;
        return readDesc(blob, @intCast(i));
    }
    return null;
}

fn readDesc(blob: []const u8, i: u32) TensorDesc {
    const at = header_bytes + @as(u64, i) * TensorDesc.wire_size;
    var raw: [@sizeOf(TensorDesc)]u8 align(@alignOf(TensorDesc)) = undefined;
    @memcpy(&raw, blob[@intCast(at)..][0..@sizeOf(TensorDesc)]);
    return @bitCast(raw);
}

fn readU16(blob: []const u8, at: u64) u16 {
    return std.mem.readInt(u16, blob[@intCast(at)..][0..2], .little);
}

fn readU32(blob: []const u8, at: u64) u32 {
    return std.mem.readInt(u32, blob[@intCast(at)..][0..4], .little);
}

fn readU64(blob: []const u8, at: u64) u64 {
    return std.mem.readInt(u64, blob[@intCast(at)..][0..8], .little);
}

// -- positional encodings ---------------------------------------------------

/// Marian's layout: sines in the first half of each row, cosines in the second,
/// not the interleaved form from the paper. Getting this wrong is a silent
/// quality loss, not a crash, which is why it is written down here.
///
/// The frequency is `10000^(-i / (d_model/2 - 1))`, from
/// `addPositionalEmbeddings` in Marian's `src/models/transformer.h`:
///
///     pow(1e-4, (i % numTimescales) / (numTimescales - 1.0))
///
/// Note the denominator: `numTimescales - 1` = 127 for `d_model = 256`, not the
/// `d_model` of the original paper's `10000^(2i/d)`. Off by one in an exponent
/// is a different set of frequencies at every position.
pub fn fillPositional(pos: []f32, d_model: u32, steps: u32) void {
    assert(pos.len >= @as(usize, steps) * d_model);
    assert(d_model % 2 == 0);
    assert(d_model >= 4);

    const half = d_model / 2;
    for (0..half) |raw_i| {
        const i: u32 = @intCast(raw_i);
        const e = @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(half - 1));
        const inv = 1.0 / math.powPositive(10000.0, e);
        for (0..steps) |p| {
            const sc = math.sinCos(@as(f64, @floatFromInt(p)) * inv);
            pos[p * d_model + i] = @floatCast(sc.sin);
            pos[p * d_model + i + half] = @floatCast(sc.cos);
        }
    }
}

// -- tests ------------------------------------------------------------------

test "descriptor and hparams have the sizes SPEC §6 promises" {
    try std.testing.expectEqual(TensorDesc.wire_size, @sizeOf(TensorDesc));
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(HParams));
    // §6 says the hparams block is 48 packed bytes; it must have no holes, or
    // `@bitCast` from the artifact would read undefined padding.
    try std.testing.expectEqual(@as(usize, 44), @offsetOf(HParams, "max_length_factor"));
    // The header ends exactly where the tensor table begins, and stays a
    // multiple of the payload alignment.
    try std.testing.expectEqual(@as(u64, 128), header_bytes);
    try std.testing.expectEqual(@as(u64, 0), header_bytes % payload_alignment);
}

test "positional encodings match the closed form" {
    const d: u32 = 16;
    const steps: u32 = 8;
    var pos: [d * steps]f32 = undefined;
    fillPositional(&pos, d, steps);

    for (0..steps) |p| {
        for (0..d / 2) |i| {
            const x = @as(f64, @floatFromInt(p)) /
                std.math.pow(f64, 10000.0, @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(d / 2 - 1)));
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(x))), pos[p * d + i], 1e-6);
            try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(x))), pos[p * d + i + d / 2], 1e-6);
        }
    }
}
