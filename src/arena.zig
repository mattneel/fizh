//! arena.zig — one arena, carved at init, never resized, no sub-allocators.
//!
//! SPEC §4. `Layout.compute` is a pure function of the config: the host can ask
//! for the byte count before it has grown memory, and get the identical answer
//! that `fizh_init` will carve against.
//!
//! Every region starts on a 64-byte boundary (SPEC §12.5). All arithmetic here
//! runs in `u64` and is proven to fit `u32` once, at the end; `abi.limits`
//! exists to make that proof cheap.

const std = @import("std");
const assert = std.debug.assert;

const abi = @import("abi.zig");
const ssplit = @import("tok/ssplit.zig");
const unigram = @import("tok/unigram.zig");

pub const alignment: u32 = 64;

/// A half-open byte range within the arena. Offsets, never pointers: a region
/// descriptor must survive being copied around without carrying provenance.
pub const Region = extern struct {
    off: u32,
    len: u32,

    pub fn end(self: Region) u64 {
        return @as(u64, self.off) + self.len;
    }
};

/// Named after SPEC §4.2. Regions past `io_dst` are implementation detail that
/// §4.2 does not enumerate; they are small, and they are listed in
/// `docs/adr/0002-arena-regions.md`.
pub const Layout = struct {
    total: u32,

    /// One per loadable direction. SPEC §4.1: the only additive region.
    weights: [abi.limits.models]Region,
    weights_count: u32,

    pos_enc: Region,
    xattn_kv: Region,
    /// SSRU carries one `d_model` cell vector per decoder layer, and that is
    /// the whole recurrent state (ADR 0008). The transformer KV cache this
    /// replaces was `2 · n_dec · max_tgt · d_model · 4` — three orders of
    /// magnitude larger, and it grew with the output.
    ssru_state: Region,
    enc_states: Region,
    act_a: Region,
    act_b: Region,
    qact: Region,
    /// Per-row activation scales for the current quantization. SPEC §7 makes
    /// these dynamic, so they are scratch, not weights.
    qact_scales: Region,
    /// Q, K, V and the attention context for one encoder layer, `[src][d]`
    /// each. Held live simultaneously, so they cannot share `act_*`.
    attn_work: Region,
    attn_scores: Region,
    vec: Region,
    qvec: Region,
    shortlist_rows: Region,
    shortlist_ids: Region,
    shortlist_scales: Region,
    logits: Region,
    shortlist_seen: Region,
    src_ids: Region,
    pivot_ids: Region,
    tgt_ids: Region,
    /// Normalized source bytes: whitespace collapsed, word boundaries marked,
    /// dummy prefix prepended. One byte longer than the raw source at most.
    tok_norm: Region,
    /// Sentence spans from `tok/ssplit.zig`. Bergamot's models are trained on
    /// single sentences, so a pass translates one at a time and rejoins.
    sent_spans: Region,
    tok_lattice: Region,
    io_src: Region,
    io_pivot: Region,
    io_dst: Region,

    /// Returns null when the config is valid but the layout does not fit in a
    /// `u32`; `usize` never crosses the ABI (SPEC §12.7), so neither does a
    /// size that cannot be named in one.
    pub fn compute(cfg: abi.Config) ?Layout {
        assert(cfg.validate() == null);

        const d: u64 = cfg.max_d_model;
        const f: u64 = cfg.max_ffn_dim;
        const s: u64 = cfg.max_src_tokens;
        const t: u64 = cfg.max_tgt_tokens;
        const dec: u64 = cfg.max_dec_layers;
        const heads: u64 = cfg.max_heads;
        const shortlist: u64 = cfg.max_shortlist;
        const src_bytes: u64 = cfg.max_src_bytes;
        const steps: u64 = @max(s, t);
        const widest: u64 = @max(d, f);

        var c: Carver = .{};

        var layout: Layout = undefined;
        layout.weights = @splat(.{ .off = 0, .len = 0 });
        layout.weights_count = cfg.max_models;
        for (0..cfg.max_models) |i| layout.weights[i] = c.take(cfg.max_model_bytes);

        layout.pos_enc = c.take(steps * d * 4);
        layout.xattn_kv = c.take(2 * dec * s * d * 4);
        layout.ssru_state = c.take(dec * d * 4);
        layout.enc_states = c.take(s * d * 4);
        layout.act_a = c.take(s * f * 4);
        layout.act_b = c.take(s * f * 4);
        layout.qact = c.take(s * f);
        layout.qact_scales = c.take(steps * 4);
        layout.attn_work = c.take(@as(u64, attn_work_slots) * s * d * 4);
        layout.attn_scores = c.take(heads * steps * 4);
        layout.vec = c.take(@as(u64, vec_slots) * widest * 4);
        layout.qvec = c.take(@as(u64, vec_slots) * widest);
        layout.shortlist_rows = c.take(shortlist * d);
        layout.shortlist_ids = c.take(shortlist * 4);
        layout.shortlist_scales = c.take(shortlist * 4);
        layout.logits = c.take(shortlist * 4);
        layout.shortlist_seen = c.take((@as(u64, cfg.max_vocab) + 7) / 8);
        layout.src_ids = c.take(s * 4);
        layout.pivot_ids = c.take(s * 4);
        layout.tgt_ids = c.take(t * 4);
        layout.tok_norm = c.take(src_bytes + norm_slack);
        // A sentence needs at least a terminator and a space, so the input
        // cannot contain more than half its bytes in sentences.
        layout.sent_spans = c.take((src_bytes / 2 + 2) * @sizeOf(ssplit.Span));
        layout.tok_lattice = c.take((src_bytes + norm_slack + 1) * @sizeOf(unigram.LatticeNode));
        layout.io_src = c.take(src_bytes);
        layout.io_pivot = c.take(src_bytes);
        layout.io_dst = c.take(src_bytes);

        if (c.cursor > std.math.maxInt(u32)) return null;
        layout.total = @intCast(c.cursor);

        assert(layout.total % alignment == 0);
        assert(layout.io_dst.end() <= layout.total);
        return layout;
    }
};

/// Named scratch vectors a single decode step may hold live at once. Sized for
/// the widest of `d_model` and `ffn_dim` so callers never have to think about
/// which one they are in.
pub const vec_slots: u32 = 16;

/// Q, K, V, context.
pub const attn_work_slots: u32 = 4;

/// Headroom `tok/unigram.zig` needs over the raw source: one byte for the dummy
/// prefix, and slack so the bound is obviously safe rather than exactly tight.
pub const norm_slack: u64 = 8;

const Carver = struct {
    cursor: u64 = 0,

    fn take(self: *Carver, len: u64) Region {
        assert(self.cursor % alignment == 0);
        const off = self.cursor;
        self.cursor = std.mem.alignForward(u64, off + len, alignment);
        // Callers below are all bounded by `abi.limits`; the product of any of
        // them cannot approach 2^63, so this cannot have wrapped.
        assert(self.cursor >= off);
        return .{
            .off = if (off <= std.math.maxInt(u32)) @intCast(off) else std.math.maxInt(u32),
            .len = if (len <= std.math.maxInt(u32)) @intCast(len) else std.math.maxInt(u32),
        };
    }
};

/// A carved arena bound to real memory. Handing out typed slices is the only
/// thing it does; there is no free, and no second phase.
pub const Arena = struct {
    base: [*]u8,
    layout: Layout,

    pub fn init(base: [*]u8, len: u32, layout: Layout) Arena {
        assert(len >= layout.total);
        assert(@intFromPtr(base) % alignment == 0);
        return .{ .base = base, .layout = layout };
    }

    /// Typed view of a region. Asserts the region is exactly divisible by the
    /// element size and that the resulting slice is naturally aligned — a
    /// silent stride error here would surface as garbage six layers later.
    pub fn view(self: Arena, comptime T: type, r: Region, count: usize) []T {
        assert(r.off % alignment == 0);
        assert(count * @sizeOf(T) <= r.len);
        assert(alignment % @alignOf(T) == 0);
        const p: [*]T = @ptrCast(@alignCast(self.base + r.off));
        return p[0..count];
    }

    pub fn bytes(self: Arena, r: Region) []u8 {
        assert(r.off % alignment == 0);
        assert(r.end() <= self.layout.total);
        return (self.base + r.off)[0..r.len];
    }

    /// SPEC §12.10: the decode loop allocates nothing. There is no allocator to
    /// prove that against, so the proof is that the arena is immutable after
    /// init — this is the value the step test watermarks.
    pub fn watermark(self: Arena) u64 {
        return @as(u64, self.layout.total) ^ @intFromPtr(self.base);
    }
};

test "layout is 64-byte aligned everywhere and fits the SPEC §4.3 example" {
    const cfg = abi.defaultTestConfig();
    const layout = Layout.compute(cfg).?;

    const regions = [_]Region{
        layout.pos_enc,     layout.xattn_kv,        layout.ssru_state, layout.enc_states,
        layout.act_a,       layout.act_b,           layout.qact,      layout.qact_scales,
        layout.attn_work,   layout.attn_scores,
        layout.vec,         layout.qvec,            layout.shortlist_rows, layout.shortlist_ids,
        layout.shortlist_scales, layout.logits,     layout.shortlist_seen, layout.src_ids,
        layout.pivot_ids,   layout.tgt_ids,         layout.tok_norm,  layout.sent_spans,
        layout.tok_lattice,
        layout.io_src,      layout.io_pivot,        layout.io_dst,
    };
    for (regions) |r| {
        try std.testing.expectEqual(@as(u32, 0), r.off % alignment);
        try std.testing.expect(r.end() <= layout.total);
    }

    // Regions must not overlap: each starts at or after the previous end.
    var prev: u64 = 0;
    for (regions) |r| {
        try std.testing.expect(r.off >= prev);
        prev = r.end();
    }

    // SPEC §14: shared scratch <= 16 MB. Subtract the additive weight slots.
    const scratch = layout.total - cfg.max_models * std.mem.alignForward(u32, cfg.max_model_bytes, alignment);
    try std.testing.expect(scratch <= 16 << 20);
}

test "scratch is sized by the max over models, not the sum" {
    // SPEC §4.1. Loading a second direction adds exactly one weight slot and
    // nothing else.
    var one = abi.defaultTestConfig();
    one.max_models = 1;
    var two = abi.defaultTestConfig();
    two.max_models = 2;

    const a = Layout.compute(one).?;
    const b = Layout.compute(two).?;
    const slot = std.mem.alignForward(u32, one.max_model_bytes, alignment);
    try std.testing.expectEqual(a.total + slot, b.total);
}

test "arena hands out aligned typed views" {
    // Static backing, not a heap allocation: SPEC §12.1 holds in tests too.
    const Backing = struct {
        var buf: [1 << 16]u8 align(alignment) = undefined;
    };

    var small = Layout.compute(abi.defaultTestConfig()).?;
    small.total = Backing.buf.len;
    small.io_src = .{ .off = 0, .len = 4096 };
    small.io_dst = .{ .off = 4096, .len = 4096 };

    const arena = Arena.init(&Backing.buf, small.total, small);
    const floats = arena.view(f32, small.io_src, 1024);
    try std.testing.expectEqual(@as(usize, 1024), floats.len);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(floats.ptr) % @alignOf(f32));
    try std.testing.expectEqual(@as(usize, 4096), arena.bytes(small.io_dst).len);
}
