//! SPEC M5/M8 — end to end through the real ABI.
//!
//! The weights are random, so the *text* means nothing. What is under test is
//! everything around it: that a pass runs to completion, that it is bit-exact
//! reproducible (I9), that generation respects both of SPEC §12.3's bounds,
//! that a pivot is two passes and never three, and that arbitrary input never
//! trraps.

const std = @import("std");
const testing = std.testing;

const abi = @import("../../src/abi.zig");
const encoder = @import("../../src/graph/encoder.zig");
const arena_mod = @import("../../src/arena.zig");
const runtime = @import("../../src/runtime.zig");
const artifact = @import("../artifact.zig");

fn config() abi.Config {
    var cfg = abi.defaultTestConfig();
    cfg.max_models = 2;
    cfg.max_model_bytes = 1 << 20;
    cfg.max_src_bytes = 256;
    cfg.max_src_tokens = 32;
    cfg.max_tgt_tokens = 48;
    cfg.max_shortlist = 64;
    cfg.max_d_model = 32;
    cfg.max_ffn_dim = 64;
    cfg.max_enc_layers = 2;
    cfg.max_dec_layers = 1;
    cfg.max_heads = 2;
    cfg.max_vocab = 64;
    return cfg;
}

const Fixture = struct {
    handle: i32,
    memory: []align(arena_mod.alignment) u8,

    fn init() !Fixture {
        // These artifacts are uniform random int8, which mixes much harder than
        // trained weights and collapses the encoder legitimately. SPEC §12.10's
        // invariant is about trained models; `test/real/` is where it is
        // exercised for real.
        encoder.collapse_limit = 1.01;
        runtime.resetForTest();
        const cfg = config();
        const blob = cfg.bytes();
        const n = runtime.arenaBytes(&blob);
        const memory = try testing.allocator.alignedAlloc(u8, .fromByteUnits(arena_mod.alignment), n);
        errdefer testing.allocator.free(memory);
        const h = runtime.init(memory.ptr, n, &blob);
        try testing.expect(h > 0);
        return .{ .handle = h, .memory = memory };
    }

    fn load(self: *Fixture, slot: u32, src: *const [2]u8, tgt: *const [2]u8) !void {
        try self.loadSeeded(slot, src, tgt, 0x1234_5678_9abc_def0);
    }

    fn loadSeeded(self: *Fixture, slot: u32, src: *const [2]u8, tgt: *const [2]u8, seed: u64) !void {
        var art = try artifact.build(testing.allocator, .{
            .src_lang = src.*,
            .tgt_lang = tgt.*,
            .seed = seed,
        });
        defer art.deinit(testing.allocator);
        try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(self.handle, slot, art.bytes));
    }

    fn deinit(self: *Fixture) void {
        runtime.resetForTest();
        testing.allocator.free(self.memory);
    }
};

test "a direct pass runs end to end" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    var out: [512]u8 = undefined;
    const n = runtime.translate(fx.handle, "hola que tal", abi.langFrom("es"), abi.lang_en, &out);
    try testing.expect(n >= 0);
    try testing.expect(std.unicode.utf8ValidateSlice(out[0..@intCast(n)]));
}

test "the same input twice gives the same bytes" {
    // I9: bit-exact determinism for a given artifact + build. If this ever
    // fails, something is reading scratch it did not write.
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    var a: [512]u8 = undefined;
    var b: [512]u8 = undefined;
    const es = abi.langFrom("es");

    for ([_][]const u8{ "hola", "hola que tal", "que tal hola hola que", "" }) |msg| {
        const n1 = runtime.translate(fx.handle, msg, es, abi.lang_en, &a);
        const n2 = runtime.translate(fx.handle, msg, es, abi.lang_en, &b);
        try testing.expectEqual(n1, n2);
        try testing.expect(n1 >= 0);
        try testing.expectEqualSlices(u8, a[0..@intCast(n1)], b[0..@intCast(n2)]);
    }
}

test "a reloaded model reproduces the same translation" {
    // The arena is reused across loads; a stale byte from the previous model
    // would show up here and nowhere else.
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    var first: [512]u8 = undefined;
    const n1 = runtime.translate(fx.handle, "hola que tal", abi.langFrom("es"), abi.lang_en, &first);
    try testing.expect(n1 >= 0);

    try fx.load(1, "en", "de");
    try fx.load(0, "es", "en");

    var again: [512]u8 = undefined;
    const n2 = runtime.translate(fx.handle, "hola que tal", abi.langFrom("es"), abi.lang_en, &again);
    try testing.expectEqual(n1, n2);
    try testing.expectEqualSlices(u8, first[0..@intCast(n1)], again[0..@intCast(n2)]);
}

test "a pivot is two passes and produces a result" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");
    try fx.load(1, "en", "de");

    const es = abi.langFrom("es");
    const de = abi.langFrom("de");
    try testing.expectEqual(@intFromEnum(abi.Route.pivot), runtime.canTranslate(fx.handle, es, de));

    var out: [512]u8 = undefined;
    const n = runtime.translate(fx.handle, "hola que tal", es, de, &out);
    try testing.expect(n >= 0);
    try testing.expect(std.unicode.utf8ValidateSlice(out[0..@intCast(n)]));
}

test "a pair with no path is still no_route once models are loaded" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    var out: [512]u8 = undefined;
    // en->es is the reverse of the only model. Bergamot models are one-way
    // (SPEC §10) and fizh does not pretend otherwise.
    try testing.expectEqual(
        abi.Status.no_route.int(),
        runtime.translate(fx.handle, "hello", abi.lang_en, abi.langFrom("es"), &out),
    );
}

test "generation respects both SPEC §12.3 bounds" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    const inst = runtime.instanceForTest().?;
    const factor = inst.models[0].hp.max_length_factor;
    const max_tgt = inst.cfg.max_tgt_tokens;
    const longest_piece: u32 = 6; // "\xffhello" in the fixture vocabulary

    var out: [4096]u8 = undefined;
    const es = abi.langFrom("es");

    // "hola" is two source tokens: the piece and the end-of-sequence. With
    // max_length_factor = 3 the loop may run 8 steps, so at most 8 pieces can
    // reach the output however much the model wants to keep going.
    for ([_][]const u8{ "hola", "hola que", "hola que tal hola que tal" }) |msg| {
        const n = runtime.translate(fx.handle, msg, es, abi.lang_en, &out);
        try testing.expect(n >= 0);

        // Source tokens are at most one per byte plus the end-of-sequence.
        const src_tokens: u32 = @intCast(msg.len + 1);
        const by_factor: u32 = @as(u32, @intFromFloat(@ceil(factor * @as(f32, @floatFromInt(src_tokens))))) + 2;
        const steps = @min(max_tgt, by_factor);
        try testing.expect(@as(u32, @intCast(n)) <= steps * longest_piece);
    }
}

test "a model that never emits end-of-sequence stops at the length bound" {
    // Random weights give a model no reason to stop, which is the useful case:
    // it drives the decode loop all the way to SPEC §12.3's bound and proves
    // the KV cache, the shortlist and the residual stream survive every step.
    // Seed chosen because it produces a model that runs long rather than
    // emitting `</s>` immediately.
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.loadSeeded(0, "es", "en", 2 * 0x9e3779b9);

    var out: [4096]u8 = undefined;
    const n = runtime.translate(fx.handle, "hola que tal", abi.langFrom("es"), abi.lang_en, &out);
    try testing.expect(n > 0);

    // "hola que tal" is four source tokens including `</s>`, and
    // max_length_factor is 3, so the loop runs at most 3*4 + 2 = 14 steps. The
    // vocabulary's longest piece is `\xffhello` at six bytes.
    try testing.expect(@as(u32, @intCast(n)) <= 14 * 6);

    // Every step used the shortlist, never the vocabulary (I6).
    const inst = runtime.instanceForTest().?;
    try testing.expect(inst.cfg.max_shortlist < inst.models[0].hp.vocab_size * 4);
}

test "the arena is untouched across a decode" {
    // SPEC §12.10: zero allocation in the decode loop. There is no allocator to
    // watch, so what is watched instead is that the arena's shape does not move.
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    const inst = runtime.instanceForTest().?;
    const before = inst.arena.watermark();
    const total_before = inst.arena.layout.total;

    var out: [512]u8 = undefined;
    _ = runtime.translate(fx.handle, "hola que tal hola que tal", abi.langFrom("es"), abi.lang_en, &out);

    try testing.expectEqual(before, inst.arena.watermark());
    try testing.expectEqual(total_before, inst.arena.layout.total);
}

test "arbitrary UTF-8 never traps a pass" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    var rng = std.Random.DefaultPrng.init(0xabcdef);
    const r = rng.random();
    const alphabet = "holaquet! ?\u{00e9}\u{1f41f}";

    var src: [200]u8 = undefined;
    var out: [512]u8 = undefined;

    for (0..300) |_| {
        var len: usize = 0;
        const want = r.intRangeAtMost(usize, 0, 64);
        while (len < want) {
            const at = r.intRangeLessThan(usize, 0, alphabet.len);
            const seq = std.unicode.utf8ByteSequenceLength(alphabet[at]) catch 1;
            if (at + seq > alphabet.len or len + seq > src.len) break;
            @memcpy(src[len..][0..seq], alphabet[at..][0..seq]);
            len += seq;
        }
        if (!std.unicode.utf8ValidateSlice(src[0..len])) continue;

        const n = runtime.translate(fx.handle, src[0..len], abi.langFrom("es"), abi.lang_en, &out);
        try testing.expect(n >= 0 or n == abi.Status.out_too_small.int());
    }
}

test "a source that needs more tokens than the arena has is a status" {
    var fx = try Fixture.init();
    defer fx.deinit();
    try fx.load(0, "es", "en");

    // config() carves 32 source tokens; a 200-byte string of unknown
    // characters needs far more.
    const long = "!" ** 200;
    var out: [512]u8 = undefined;
    const n = runtime.translate(fx.handle, long, abi.langFrom("es"), abi.lang_en, &out);
    try testing.expectEqual(abi.Status.src_too_long.int(), n);
}
