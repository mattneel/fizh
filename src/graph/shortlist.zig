//! graph/shortlist.zig — SPEC §8 `shortlist_build`, and I6.
//!
//! Full-vocab projection is the single largest per-token cost, so fizh never
//! does one. The candidate set is the union of each source token's lexical
//! translations with a fixed list of frequent target pieces, and the output
//! projection reads only those rows.
//!
//! End-of-sequence goes in first, unconditionally. A shortlist without it is a
//! decoder that cannot stop, and `max_length_factor` would be the only thing
//! ending the loop — a bug that looks like a quality problem.

const std = @import("std");
const assert = std.debug.assert;

const backend = @import("../kernel/backend.zig");
const format = @import("../model/format.zig");
const kernel = backend.active;
const pass = @import("pass.zig");

pub fn build(ctx: *pass.Ctx) void {
    const v = ctx.hp.vocab_size;
    const bits = (v + 7) / 8;
    assert(ctx.shortlist_seen.len >= bits);
    assert(ctx.src_len > 0);
    @memset(ctx.shortlist_seen[0..bits], 0);

    var n: u32 = 0;
    n = add(ctx, n, ctx.hp.eos_id);
    n = add(ctx, n, ctx.hp.unk_id);

    const frequent = format.u32View(ctx.slot, ctx.sl.sl_frequent, ctx.sl.sl_frequent_len);
    for (frequent) |id| n = add(ctx, n, id);

    const offsets = format.u32View(ctx.slot, ctx.sl.sl_offsets, v + 1);
    const targets = format.u16View(ctx.slot, ctx.sl.sl_targets, ctx.sl.sl_targets_len);
    for (ctx.src_ids[0..ctx.src_len]) |src| {
        assert(src < v);
        const from = offsets[src];
        const to = offsets[src + 1];
        assert(to >= from and to <= targets.len);
        for (targets[from..to]) |id| n = add(ctx, n, id);
    }

    ctx.shortlist_len = n;
    assert(n >= 2);
    assert(n <= ctx.max_shortlist);

    gather(ctx);
}

/// Returns the new length. Duplicates and overflow are both no-ops: the
/// shortlist is a *candidate* set, so dropping the tail when it is full costs
/// coverage, never correctness.
fn add(ctx: *pass.Ctx, n: u32, id: u32) u32 {
    assert(id < ctx.hp.vocab_size);
    if (n >= ctx.max_shortlist) return n;

    const byte = id / 8;
    const bit = @as(u8, 1) << @intCast(id % 8);
    if (ctx.shortlist_seen[byte] & bit != 0) return n;

    ctx.shortlist_seen[byte] |= bit;
    ctx.shortlist_ids[n] = id;
    return n + 1;
}

/// Collects the chosen rows of the tied output projection so the per-token
/// projection is a `qgemv` over contiguous memory rather than a gather inside
/// the inner loop.
fn gather(ctx: *pass.Ctx) void {
    const d = ctx.hp.d_model;
    const n = ctx.shortlist_len;
    const emb = ctx.sl.emb;

    kernel.gatherRows(
        ctx.shortlist_rows,
        d,
        ctx.shortlist_scales,
        ctx.shortlist_ids[0..n],
        format.i8View(ctx.slot, emb.data, emb.n * emb.stride()),
        emb.stride(),
        format.f32View(ctx.slot, emb.scales, emb.n),
        d,
    );
    assert(ctx.shortlist_rows.len >= @as(usize, n) * d);
}
