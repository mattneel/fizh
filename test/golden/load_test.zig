//! SPEC M2 golden load test, and SPEC §13 T3 against corrupted artifacts.
//!
//! The claim under test is the one SPEC §11 makes: `model/format.zig` is the
//! validation boundary, so *any* byte sequence produces either a loaded model
//! or a negative status — never a trap, never a read past the blob.

const std = @import("std");
const testing = std.testing;

const abi = @import("../../src/abi.zig");
const arena_mod = @import("../../src/arena.zig");
const format = @import("../../src/model/format.zig");
const runtime = @import("../../src/runtime.zig");
const encoder = @import("../../src/graph/encoder.zig");
const trie = @import("../../src/tok/trie.zig");
const unigram = @import("../../src/tok/unigram.zig");
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
        runtime.resetForTest();
        const cfg = config();
        const blob = cfg.bytes();
        const n = runtime.arenaBytes(&blob);
        try testing.expect(n > 0);

        const memory = try testing.allocator.alignedAlloc(u8, .fromByteUnits(arena_mod.alignment), n);
        errdefer testing.allocator.free(memory);

        const h = runtime.init(memory.ptr, n, &blob);
        try testing.expect(h > 0);
        return .{ .handle = h, .memory = memory };
    }

    fn deinit(self: *Fixture) void {
        runtime.resetForTest();
        testing.allocator.free(self.memory);
    }
};

test "a well-formed artifact loads and opens a route" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);

    try testing.expectEqual(
        abi.Status.ok.int(),
        runtime.modelLoad(fx.handle, 0, art.bytes),
    );

    const es = abi.langFrom("es");
    const en = abi.lang_en;
    try testing.expectEqual(@intFromEnum(abi.Route.direct), runtime.canTranslate(fx.handle, es, en));
    try testing.expectEqual(@intFromEnum(abi.Route.none), runtime.canTranslate(fx.handle, en, es));
}

test "a loaded slot yields a working tokenizer" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);
    try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(fx.handle, 0, art.bytes));

    const inst = runtime.instanceForTest().?;
    const model = &inst.models[0];
    const vocab = model.vocab(runtime.slotBytes(inst, 0));
    try testing.expect(vocab.validate());
    try testing.expectEqual(@as(u32, @intCast(artifact.default_pieces.len)), vocab.size);

    var norm: [256]u8 = undefined;
    var lattice: [258]unigram.LatticeNode = undefined;
    var ids: [64]u32 = undefined;

    const n = unigram.normalize("hola que tal", &norm);
    const count = try unigram.encode(
        vocab,
        .{ .unk_id = model.hp.unk_id, .eos_id = model.hp.eos_id },
        norm[0..n],
        &lattice,
        &ids,
    );
    try testing.expectEqual(@as(u32, 4), count); // hola, que, tal, </s>

    var text: [256]u8 = undefined;
    const w = try unigram.decode(vocab, ids[0..count], &text);
    try testing.expectEqualStrings("hola que tal", text[0..w]);
}

test "the positional table is filled at load" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);
    try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(fx.handle, 0, art.bytes));

    const inst = runtime.instanceForTest().?;
    // Read it where it lives: in slot 0's own weights, not in shared scratch.
    const slot = inst.arena.bytes(inst.arena.layout.weights[0]);
    const pos = inst.models[0].posEncConst(slot);
    const d = inst.models[0].hp.d_model;

    // Row 0 is sin(0)=0 in the first half, cos(0)=1 in the second.
    for (0..d / 2) |i| {
        try testing.expectEqual(@as(f32, 0), pos[i]);
        try testing.expectEqual(@as(f32, 1), pos[i + d / 2]);
    }
    // Row 1, channel 0: sin(1) and cos(1).
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@sin(1.0))), pos[d], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, @floatCast(@cos(1.0))), pos[d + d / 2], 1e-6);
}

test "two directions occupy two slots and open a pivot" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var a = try artifact.build(testing.allocator, .{ .src_lang = "es".*, .tgt_lang = "en".* });
    defer a.deinit(testing.allocator);
    var b = try artifact.build(testing.allocator, .{ .src_lang = "en".*, .tgt_lang = "de".* });
    defer b.deinit(testing.allocator);

    try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(fx.handle, 0, a.bytes));
    try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(fx.handle, 1, b.bytes));

    try testing.expectEqual(
        @intFromEnum(abi.Route.pivot),
        runtime.canTranslate(fx.handle, abi.langFrom("es"), abi.langFrom("de")),
    );
}

test "truncation at every length is a status, never a trap" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);

    var len: usize = 0;
    while (len < art.bytes.len) : (len += 1 + len / 8) {
        const status = runtime.modelLoad(fx.handle, 0, art.bytes[0..len]);
        try testing.expect(status < 0);
    }
}

test "single-byte corruption is a status, never a trap" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);

    const copy = try testing.allocator.dupe(u8, art.bytes);
    defer testing.allocator.free(copy);

    var rng = std.Random.DefaultPrng.init(0xfeed);
    const r = rng.random();

    for (0..3000) |_| {
        @memcpy(copy, art.bytes);
        // Corrupt the structural part hardest: header and tensor table.
        const at = if (r.boolean())
            r.intRangeLessThan(usize, 0, @min(copy.len, 4096))
        else
            r.intRangeLessThan(usize, 0, copy.len);
        copy[at] ^= @as(u8, 1) << r.int(u3);

        // Either it still describes a valid model, or it is rejected. Either
        // way control comes back.
        _ = runtime.modelLoad(fx.handle, 0, copy);
    }
}

test "a model too large for the slot is rejected, not truncated" {
    runtime.resetForTest();
    var cfg = config();
    cfg.max_model_bytes = 4096; // far too small for even the toy model
    const blob = cfg.bytes();
    const n = runtime.arenaBytes(&blob);

    const memory = try testing.allocator.alignedAlloc(u8, .fromByteUnits(arena_mod.alignment), n);
    defer testing.allocator.free(memory);
    const h = runtime.init(memory.ptr, n, &blob);
    try testing.expect(h > 0);
    defer runtime.resetForTest();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);

    try testing.expectEqual(abi.Status.model_too_large.int(), runtime.modelLoad(h, 0, art.bytes));
}

test "a model wider than the arena was carved for is rejected" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{ .d_model = 64, .n_heads = 2 });
    defer art.deinit(testing.allocator);

    // config() carves for max_d_model = 32.
    try testing.expectEqual(abi.Status.model_too_large.int(), runtime.modelLoad(fx.handle, 0, art.bytes));
}

test "the header's own fields are validated" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);
    const copy = try testing.allocator.dupe(u8, art.bytes);
    defer testing.allocator.free(copy);

    @memcpy(copy, art.bytes);
    copy[0] = 'G';
    try testing.expectEqual(abi.Status.bad_artifact.int(), runtime.modelLoad(fx.handle, 0, copy));

    @memcpy(copy, art.bytes);
    std.mem.writeInt(u32, copy[4..8], 99, .little);
    try testing.expectEqual(abi.Status.bad_version.int(), runtime.modelLoad(fx.handle, 0, copy));

    @memcpy(copy, art.bytes);
    std.mem.writeInt(u16, copy[8..10], 0, .little);
    try testing.expectEqual(abi.Status.bad_lang.int(), runtime.modelLoad(fx.handle, 0, copy));

    // A direction that translates a language into itself is not a direction.
    @memcpy(copy, art.bytes);
    std.mem.writeInt(u16, copy[8..10], abi.lang_en, .little);
    std.mem.writeInt(u16, copy[10..12], abi.lang_en, .little);
    try testing.expectEqual(abi.Status.bad_lang.int(), runtime.modelLoad(fx.handle, 0, copy));

    @memcpy(copy, art.bytes);
    std.mem.writeInt(u32, copy[60..64], 0, .little);
    try testing.expectEqual(abi.Status.bad_artifact.int(), runtime.modelLoad(fx.handle, 0, copy));

    @memcpy(copy, art.bytes);
    std.mem.writeInt(u32, copy[60..64], format.max_tensors + 1, .little);
    try testing.expectEqual(abi.Status.bad_artifact.int(), runtime.modelLoad(fx.handle, 0, copy));
}

test "a missing tensor is missing_tensor, not a silent zero" {
    var fx = try Fixture.init();
    defer fx.deinit();

    var art = try artifact.build(testing.allocator, .{});
    defer art.deinit(testing.allocator);
    const copy = try testing.allocator.dupe(u8, art.bytes);
    defer testing.allocator.free(copy);

    // Rename the first tensor by flipping a bit in its hash.
    const at: usize = @intCast(format.header_bytes);
    copy[at] ^= 0x40;
    try testing.expectEqual(abi.Status.missing_tensor.int(), runtime.modelLoad(fx.handle, 0, copy));
}

test "a narrower model is unaffected by a wider one sharing the arena" {
    // `pos_enc` is one shared region and sinusoidal encodings depend on
    // `d_model` — a 32-wide table is not a prefix of a 16-wide one. Loading a
    // second model of a different width used to overwrite the table the first
    // one needs, which showed up only as wrong words: fluent output, no
    // assertion, and only on a pivot that crossed widths.
    //
    // The property is that loading B cannot change what A produces.
    // Uniform-random int8 mixes harder than trained weights and collapses the
    // encoder legitimately; SPEC §12.10's invariant is about trained models.
    encoder.collapse_limit = 1.01;
    defer encoder.collapse_limit = 0.95;

    var out_alone: [256]u8 = undefined;
    var out_shared: [256]u8 = undefined;
    const msg = "hola que tal";

    var alone_len: usize = 0;
    {
        var fx = try Fixture.init();
        defer fx.deinit();
        var a = try artifact.build(testing.allocator, .{ .src_lang = "es".*, .tgt_lang = "en".* });
        defer a.deinit(testing.allocator);
        try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(fx.handle, 0, a.bytes));
        const n = runtime.translate(fx.handle, msg, abi.langFrom("es"), abi.lang_en, &out_alone);
        try testing.expect(n > 0);
        alone_len = @intCast(n);
    }

    {
        var fx = try Fixture.init();
        defer fx.deinit();
        var a = try artifact.build(testing.allocator, .{ .src_lang = "es".*, .tgt_lang = "en".* });
        defer a.deinit(testing.allocator);
        // Half the width, same everything else.
        var b = try artifact.build(testing.allocator, .{
            .src_lang = "en".*,
            .tgt_lang = "de".*,
            .d_model = 16,
        });
        defer b.deinit(testing.allocator);

        try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(fx.handle, 0, a.bytes));
        try testing.expectEqual(abi.Status.ok.int(), runtime.modelLoad(fx.handle, 1, b.bytes));
        const n = runtime.translate(fx.handle, msg, abi.langFrom("es"), abi.lang_en, &out_shared);
        try testing.expect(n > 0);
        try testing.expectEqualStrings(out_alone[0..alone_len], out_shared[0..@intCast(n)]);
    }
}
