//! runtime.zig — the instance. Owns the arena, the slot table, and the one
//! call that does the work.
//!
//! There is exactly one live instance per wasm instance (SPEC §10: fizh runs in
//! a dedicated Web Worker). The handle is a generation counter, so a stale
//! handle from before a re-init is a status, not a use-after-free.

const std = @import("std");
const assert = std.debug.assert;

/// Re-exported so `tools/` can drive the real runtime rather than a copy of it.
pub const abi = @import("abi.zig");
pub const format = @import("model/format.zig");
pub const kernel = @import("kernel/backend.zig");

pub const arena = @import("arena.zig");
const arena_mod = arena;
pub const encoder = @import("graph/encoder.zig");
pub const profile = @import("graph/profile.zig");

const pass = @import("graph/pass.zig");
const route = @import("route.zig");
const ssplit = @import("tok/ssplit.zig");
const unigram = @import("tok/unigram.zig");

pub const Instance = struct {
    cfg: abi.Config,
    arena: arena_mod.Arena,
    models: [abi.limits.models]format.Model,

    pub fn slots(self: *const Instance) []const format.Model {
        return self.models[0..self.cfg.max_models];
    }
};

var g_live: bool = false;
var g_handle: i32 = 0;
var g_instance: Instance = undefined;

/// SPEC §4 step 1. Zero means "config rejected"; the host learns the reason by
/// calling `fizh_init`, which returns a status.
pub fn arenaBytes(cfg_bytes: []const u8) u32 {
    const cfg = abi.Config.parse(cfg_bytes) orelse return 0;
    if (cfg.validate() != null) return 0;
    const layout = arena_mod.Layout.compute(cfg) orelse return 0;
    assert(layout.total % arena_mod.alignment == 0);
    assert(layout.total > 0);
    return layout.total;
}

/// SPEC §4 step 3. Returns a positive handle or a negative `abi.Status`.
pub fn init(base: [*]u8, len: u32, cfg_bytes: []const u8) i32 {
    const cfg = abi.Config.parse(cfg_bytes) orelse return abi.Status.bad_config.int();
    if (cfg.validate()) |bad| return bad.int();

    const layout = arena_mod.Layout.compute(cfg) orelse return abi.Status.bad_config.int();
    if (len < layout.total) return abi.Status.arena_too_small.int();
    // Host-supplied, therefore validated rather than asserted.
    if (@intFromPtr(base) % arena_mod.alignment != 0) return abi.Status.bad_arg.int();

    // SPEC §9: determinism starts from a known state, and a read-before-write
    // bug must reproduce rather than pick up whatever the host left behind.
    @memset(base[0..layout.total], 0);

    g_instance = .{
        .cfg = cfg,
        .arena = arena_mod.Arena.init(base, len, layout),
        .models = @splat(.empty),
    };
    for (0..cfg.max_models) |i| {
        g_instance.models[i].slot_base = layout.weights[i].off;
    }

    g_handle = if (g_handle == std.math.maxInt(i32)) 1 else g_handle + 1;
    g_live = true;

    assert(g_handle > 0);
    assert(g_instance.arena.layout.total == layout.total);
    return g_handle;
}

pub fn modelLoad(h: i32, slot: u32, blob: []const u8) i32 {
    const inst = resolve(h) orelse return abi.Status.bad_handle.int();
    if (slot >= inst.cfg.max_models) return abi.Status.bad_slot.int();
    assert(slot < abi.limits.models);

    const region = inst.arena.layout.weights[slot];
    assert(region.off % arena_mod.alignment == 0);
    const slot_bytes = inst.arena.bytes(region);

    const model = &inst.models[slot];
    const status = format.load(blob, inst.cfg, slot_bytes, model);
    if (status != .ok) {
        assert(!model.loaded);
        return status.int();
    }

    model.slot_base = region.off;
    assert(model.loaded);
    assert(model.slot_used <= region.len);
    return abi.Status.ok.int();
}

/// The bytes of a loaded slot. Callers assert `loaded` first.
pub fn slotBytes(inst: *const Instance, slot: u8) []const u8 {
    assert(slot < inst.cfg.max_models);
    assert(inst.models[slot].loaded);
    return inst.arena.bytes(inst.arena.layout.weights[slot]);
}

pub fn canTranslate(h: i32, src_lang: abi.Lang, tgt_lang: abi.Lang) i32 {
    const inst = resolve(h) orelse return abi.Status.bad_handle.int();
    const plan = route.resolve(inst.slots(), src_lang, tgt_lang);
    assert(plan.hops() <= 2);
    return @intFromEnum(plan.kind);
}

/// SPEC §9: one call does the work. Returns bytes written, or a status.
pub fn translate(
    h: i32,
    src: []const u8,
    src_lang: abi.Lang,
    tgt_lang: abi.Lang,
    out: []u8,
) i32 {
    const inst = resolve(h) orelse return abi.Status.bad_handle.int();
    if (src.len > inst.cfg.max_src_bytes) return abi.Status.src_too_long.int();
    if (!abi.langValid(src_lang) or !abi.langValid(tgt_lang)) return abi.Status.bad_lang.int();
    // The pivot boundary is UTF-8 text (SPEC §10), so the entry boundary is too.
    if (!std.unicode.utf8ValidateSlice(src)) return abi.Status.bad_utf8.int();

    const plan = route.resolve(inst.slots(), src_lang, tgt_lang);
    if (plan.kind == .none) return abi.Status.no_route.int();
    assert(plan.hops() <= 2);

    const io_src = inst.arena.bytes(inst.arena.layout.io_src);
    const io_pivot = inst.arena.bytes(inst.arena.layout.io_pivot);
    const io_dst = inst.arena.bytes(inst.arena.layout.io_dst);
    assert(io_src.len >= src.len);

    @memcpy(io_src[0..src.len], src);
    var live: []const u8 = io_src[0..src.len];

    // SPEC §12.6: pivot depth is a negative-space assertion, not a comment.
    var pivot_depth: u8 = 0;
    if (plan.kind == .pivot) {
        assert(pivot_depth == 0);
        const n = runPass(inst, plan.first, live, io_pivot) catch |e| return statusOf(e).int();
        live = io_pivot[0..n];
        pivot_depth += 1;
        assert(pivot_depth == 1);
    }

    if (plan.identity) {
        assert(plan.kind == .direct);
        @memcpy(io_dst[0..live.len], live);
        live = io_dst[0..live.len];
    } else {
        const slot = if (plan.kind == .pivot) plan.second else plan.first;
        const n = runPass(inst, slot, live, io_dst) catch |e| return statusOf(e).int();
        live = io_dst[0..n];
    }
    assert(pivot_depth <= 1);

    if (live.len > out.len) return abi.Status.out_too_small.int();
    @memcpy(out[0..live.len], live);
    assert(live.len <= std.math.maxInt(i32));
    return @intCast(live.len);
}

/// What the most recent pass computed. Written by `runPass` and read only by
/// `tools/trace.zig`, which feeds it to the SPEC §13 T2 oracle. Four words of
/// module state buys a differential test that can point at the layer.
pub const LastPass = struct {
    slot: u8 = 0,
    src_len: u32 = 0,
    tgt_len: u32 = 0,
    shortlist_len: u32 = 0,
};

var g_last: LastPass = .{};

pub fn lastPass() LastPass {
    return g_last;
}

const PassError = error{ NotLoaded, SrcTooLong, OutTooSmall };

/// One model, source bytes in, target bytes out.
fn runPass(inst: *Instance, slot: u8, in: []const u8, out: []u8) PassError!u32 {
    assert(slot < inst.cfg.max_models);
    assert(out.len >= in.len);
    if (!inst.models[slot].loaded) return error.NotLoaded;

    var ctx = buildCtx(inst, slot);
    const written = pass.run(&ctx, in, out) catch |e| switch (e) {
        error.SrcTooLong => return error.SrcTooLong,
        error.OutTooSmall => return error.OutTooSmall,
    };
    g_last = .{
        .slot = slot,
        .src_len = ctx.src_len,
        .tgt_len = ctx.tgt_len,
        .shortlist_len = ctx.shortlist_len,
    };
    return written;
}

/// Resolves every arena region the graph may touch, once. Nothing below this
/// function does pointer arithmetic on the arena.
fn buildCtx(inst: *Instance, slot: u8) pass.Ctx {
    const cfg = inst.cfg;
    const a = inst.arena;
    const l = a.layout;
    const m = &inst.models[slot];

    const s: u32 = cfg.max_src_tokens;
    const t: u32 = cfg.max_tgt_tokens;
    const d: u32 = cfg.max_d_model;
    const f: u32 = cfg.max_ffn_dim;
    const steps: u32 = @max(s, t);
    const widest: u32 = @max(d, f);

    assert(m.loaded);
    assert(m.hp.d_model <= d and m.hp.ffn_dim <= f);

    const slot_bytes = a.bytes(l.weights[slot]);
    // SPEC §4.1: slot-derived data is read through the slot it was derived
    // from, never through a shared region. The assertion is the category made
    // checkable — if this view ever came from anywhere else, it fires.
    const pos = m.posEncConst(slot_bytes);
    assert(pos.len == @as(usize, steps) * m.hp.d_model);
    assert(@intFromPtr(pos.ptr) >= @intFromPtr(slot_bytes.ptr));
    assert(@intFromPtr(pos.ptr) + pos.len * 4 <= @intFromPtr(slot_bytes.ptr) + slot_bytes.len);

    return .{
        .model = m,
        .slot = slot_bytes,
        .hp = m.hp,
        .sl = m.sl,
        .max_src_tokens = s,
        .max_tgt_tokens = t,
        .max_shortlist = cfg.max_shortlist,

        .pos_enc = pos,
        .xattn_kv = a.view(f32, l.xattn_kv, 2 * @as(usize, cfg.max_dec_layers) * s * d),
        .ssru_state = a.view(f32, l.ssru_state, @as(usize, cfg.max_dec_layers) * d),
        .enc_states = a.view(f32, l.enc_states, @as(usize, s) * d),
        .act_a = a.view(f32, l.act_a, @as(usize, s) * f),
        .act_b = a.view(f32, l.act_b, @as(usize, s) * f),
        .qact = a.view(i8, l.qact, @as(usize, s) * f),
        .qact_scales = a.view(f32, l.qact_scales, steps),
        .attn_work = a.view(f32, l.attn_work, arena_mod.attn_work_slots * @as(usize, s) * d),
        .attn_scores = a.view(f32, l.attn_scores, @as(usize, cfg.max_heads) * steps),
        .vec = a.view(f32, l.vec, arena_mod.vec_slots * widest),
        .qvec = a.view(i8, l.qvec, arena_mod.vec_slots * widest),
        .shortlist_rows = a.view(i8, l.shortlist_rows, @as(usize, cfg.max_shortlist) * d),
        .shortlist_ids = a.view(u32, l.shortlist_ids, cfg.max_shortlist),
        .shortlist_scales = a.view(f32, l.shortlist_scales, cfg.max_shortlist),
        .logits = a.view(f32, l.logits, cfg.max_shortlist),
        .shortlist_seen = a.bytes(l.shortlist_seen),
        .src_ids = a.view(u32, l.src_ids, s),
        .tgt_ids = a.view(u32, l.tgt_ids, t),
        .tok_raw = a.bytes(l.tok_raw),
        .tok_norm = a.bytes(l.tok_norm),
        .sent_spans = a.view(ssplit.Span, l.sent_spans, l.sent_spans.len / @sizeOf(ssplit.Span)),
        .tok_lattice = a.view(unigram.LatticeNode, l.tok_lattice, l.tok_lattice.len / @sizeOf(unigram.LatticeNode)),
    };
}

fn statusOf(e: PassError) abi.Status {
    return switch (e) {
        error.NotLoaded => .not_loaded,
        error.SrcTooLong => .src_too_long,
        error.OutTooSmall => .out_too_small,
    };
}

fn resolve(h: i32) ?*Instance {
    if (!g_live) return null;
    if (h != g_handle or h <= 0) return null;
    return &g_instance;
}

/// Tests only: forget the instance so a case can start from nothing.
/// Lives here rather than in `test/` because the instance is module state.
pub fn resetForTest() void {
    g_live = false;
}

/// Tests only: reach into the live instance to inspect what a load produced.
pub fn instanceForTest() ?*Instance {
    if (!g_live) return null;
    return &g_instance;
}

test "arena_bytes rejects junk configs with zero" {
    try std.testing.expectEqual(@as(u32, 0), arenaBytes(&[_]u8{}));
    try std.testing.expectEqual(@as(u32, 0), arenaBytes(&[_]u8{1} ** 64));
}
