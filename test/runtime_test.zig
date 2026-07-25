//! End-to-end exercises of the ABI as the host sees it.
//!
//! Lives outside `src/` because it allocates a backing arena, and SPEC §12.1
//! says `src/` never mentions an allocator — not even in a test.

const std = @import("std");
const testing = std.testing;

const abi = @import("../src/abi.zig");
const arena_mod = @import("../src/arena.zig");
const runtime = @import("../src/runtime.zig");

/// Small enough to allocate per test, large enough to exercise every region.
pub fn smallConfig() abi.Config {
    var cfg = abi.defaultTestConfig();
    cfg.max_models = 2;
    cfg.max_model_bytes = 4096;
    cfg.max_src_bytes = 256;
    cfg.max_src_tokens = 32;
    cfg.max_tgt_tokens = 32;
    cfg.max_shortlist = 64;
    cfg.max_d_model = 32;
    cfg.max_ffn_dim = 64;
    cfg.max_enc_layers = 2;
    cfg.max_dec_layers = 1;
    cfg.max_heads = 2;
    cfg.max_vocab = 256;
    return cfg;
}

const Fixture = struct {
    handle: i32,
    cfg: abi.Config,
    memory: []align(arena_mod.alignment) u8,

    fn init(cfg: abi.Config) !Fixture {
        runtime.resetForTest();
        const blob = cfg.bytes();
        const n = runtime.arenaBytes(&blob);
        try testing.expect(n > 0);

        const memory = try testing.allocator.alignedAlloc(u8, .fromByteUnits(arena_mod.alignment), n);
        errdefer testing.allocator.free(memory);

        const h = runtime.init(memory.ptr, n, &blob);
        try testing.expect(h > 0);
        return .{ .handle = h, .cfg = cfg, .memory = memory };
    }

    fn deinit(self: *Fixture) void {
        runtime.resetForTest();
        testing.allocator.free(self.memory);
    }
};

test "arena_bytes and init agree on the byte count" {
    const cfg = smallConfig();
    const blob = cfg.bytes();
    const n = runtime.arenaBytes(&blob);

    var fx = try Fixture.init(cfg);
    defer fx.deinit();

    // One byte short is a status, not a crash.
    runtime.resetForTest();
    const short = try testing.allocator.alignedAlloc(u8, .fromByteUnits(arena_mod.alignment), n);
    defer testing.allocator.free(short);
    try testing.expectEqual(
        abi.Status.arena_too_small.int(),
        runtime.init(short.ptr, n - 1, &blob),
    );
}

test "init rejects a misaligned base" {
    runtime.resetForTest();
    const cfg = smallConfig();
    const blob = cfg.bytes();
    const n = runtime.arenaBytes(&blob);

    const memory = try testing.allocator.alignedAlloc(u8, .fromByteUnits(arena_mod.alignment), n + 64);
    defer testing.allocator.free(memory);

    try testing.expectEqual(abi.Status.bad_arg.int(), runtime.init(memory.ptr + 1, n, &blob));
}

test "stale and forged handles are statuses, not crashes" {
    var fx = try Fixture.init(smallConfig());
    defer fx.deinit();

    for ([_]i32{ 0, -1, -7, std.math.maxInt(i32) }) |bad| {
        if (bad == fx.handle) continue;
        try testing.expectEqual(abi.Status.bad_handle.int(), runtime.canTranslate(bad, abi.lang_en, abi.lang_en));
    }
    try testing.expectEqual(abi.Status.bad_handle.int(), runtime.canTranslate(fx.handle + 1, abi.lang_en, abi.lang_en));
}

test "M0 passthrough: the identity route echoes the source" {
    var fx = try Fixture.init(smallConfig());
    defer fx.deinit();

    const es = abi.langFrom("es");
    try testing.expectEqual(@intFromEnum(abi.Route.direct), runtime.canTranslate(fx.handle, es, es));

    const msg = "hola, ¿qué tal? 🐟";
    var out: [64]u8 = undefined;
    const n = runtime.translate(fx.handle, msg, es, es, &out);
    try testing.expect(n > 0);
    try testing.expectEqualStrings(msg, out[0..@intCast(n)]);
}

test "translate rejects hostile input with statuses" {
    var fx = try Fixture.init(smallConfig());
    defer fx.deinit();

    const es = abi.langFrom("es");
    const de = abi.langFrom("de");
    var out: [64]u8 = undefined;

    const long = [_]u8{'a'} ** 512;
    try testing.expect(long.len > fx.cfg.max_src_bytes);
    try testing.expectEqual(abi.Status.src_too_long.int(), runtime.translate(fx.handle, &long, es, es, &out));
    try testing.expectEqual(abi.Status.bad_lang.int(), runtime.translate(fx.handle, "hi", 0, es, &out));
    try testing.expectEqual(abi.Status.bad_utf8.int(), runtime.translate(fx.handle, "\xff\xfe", es, es, &out));
    try testing.expectEqual(abi.Status.no_route.int(), runtime.translate(fx.handle, "hi", es, de, &out));

    var tiny: [2]u8 = undefined;
    try testing.expectEqual(abi.Status.out_too_small.int(), runtime.translate(fx.handle, "hello", es, es, &tiny));

    // Empty source is legal and echoes empty.
    try testing.expectEqual(@as(i32, 0), runtime.translate(fx.handle, "", es, es, &out));
}

test "model_load rejects out-of-range slots" {
    var fx = try Fixture.init(smallConfig());
    defer fx.deinit();

    try testing.expectEqual(abi.Status.bad_slot.int(), runtime.modelLoad(fx.handle, fx.cfg.max_models, &[_]u8{}));
    try testing.expectEqual(abi.Status.bad_slot.int(), runtime.modelLoad(fx.handle, 999, &[_]u8{}));
    try testing.expectEqual(abi.Status.bad_handle.int(), runtime.modelLoad(fx.handle + 1, 0, &[_]u8{}));
}

test "every region of the SPEC §4.3 example fits the §14 scratch budget" {
    const cfg = abi.defaultTestConfig();
    const blob = cfg.bytes();
    const total = runtime.arenaBytes(&blob);
    const slots = cfg.max_models * std.mem.alignForward(u32, cfg.max_model_bytes, arena_mod.alignment);
    try testing.expect(total > slots);

    const scratch = total - slots;
    // SPEC §14: shared scratch <= 16 MB. SPEC §4.3 predicts ~6.6 MB; the extra
    // is the positional table and the gathered shortlist rows.
    try testing.expect(scratch <= 16 << 20);
    try testing.expect(scratch >= 6 << 20);
}
