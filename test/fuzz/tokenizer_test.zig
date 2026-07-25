//! SPEC §13 T3 — the tokenizer against arbitrary bytes.
//!
//! Two claims, both of which the fuzzer gets to attack:
//!   1. It never traps. Any byte sequence at all.
//!   2. Every id it emits is a real vocabulary id, and it emits no more of them
//!      than there are bytes plus one.
//!
//!   zig build test -- --fuzz          (nightly; SPEC §13)

const std = @import("std");
const testing = std.testing;

const trie = @import("../../src/tok/trie.zig");
const unigram = @import("../../src/tok/unigram.zig");

/// Deliberately full of holes: multi-byte pieces with no single-byte cover, so
/// the unknown-edge path gets exercised constantly.
const Sparse = trie.Fixture(
    &.{ "</s>", "<unk>", "\xff", "\xffhola", "\xffque", "\xfftal", "a", "e", "h", "l", "o", "q", "t", "u", "!", "?" },
    &.{ 0, 1 },
);

const params: unigram.Params = .{ .unk_id = 1, .eos_id = 0 };

const limits = struct {
    const src_bytes = 4096;
};

var norm: [limits.src_bytes + 8]u8 = undefined;
var lattice: [limits.src_bytes + 9]unigram.LatticeNode = undefined;
var ids: [limits.src_bytes + 2]u32 = undefined;
var text: [limits.src_bytes * 2]u8 = undefined;

fn once(src: []const u8) !void {
    const v = Sparse.vocab;

    const n = unigram.normalize(src, &norm);
    try testing.expect(n <= src.len + 1);

    const count = try unigram.encode(v, params, norm[0..n], &lattice, &ids);
    try testing.expect(count <= n + 1);
    for (ids[0..count]) |id| try testing.expect(id < v.size);

    // Detokenizing what we just produced must also be total.
    const w = try unigram.decode(v, ids[0..count], &text);
    try testing.expect(w <= text.len);

    // The property the pivot boundary actually needs: iterating the cycle
    // converges. Not idempotence — the normalizer collapses whitespace runs
    // that detokenization can create, and dropped unknown pieces can leave a
    // bare space behind, so one extra step is sometimes real. What must never
    // happen is oscillation or growth: pass two's input has to be a settled
    // string, not a moving target.
    var prev_len = w;
    var a: [limits.src_bytes * 2]u8 = undefined;
    var b: [limits.src_bytes * 2]u8 = undefined;
    @memcpy(a[0..w], text[0..w]);
    var cur = a[0..w];

    for (0..4) |_| {
        const nn = unigram.normalize(cur, &norm);
        const c2 = try unigram.encode(v, params, norm[0..nn], &lattice, &ids);
        const w2 = try unigram.decode(v, ids[0..c2], &b);
        try testing.expect(w2 <= prev_len);
        if (w2 == cur.len and std.mem.eql(u8, b[0..w2], cur)) return;
        @memcpy(a[0..w2], b[0..w2]);
        cur = a[0..w2];
        prev_len = w2;
    }
    return error.CycleDidNotConverge;
}

test "fuzz: arbitrary bytes never trap the tokenizer" {
    try std.testing.fuzz({}, fuzzOne, .{});
}

fn fuzzOne(_: void, smith: *std.testing.Smith) !void {
    var buf: [limits.src_bytes]u8 = undefined;
    var len: usize = 0;
    while (!smith.eos() and len < buf.len) {
        buf[len] = smith.value(u8);
        len += 1;
    }
    try once(buf[0..len]);
}

test "seed corpus: the shapes a chat register actually produces" {
    const corpus = [_][]const u8{
        "",
        " ",
        "\x00",
        "\xff\xff\xff\xff",
        "hola",
        "hola que tal!",
        "HOLA QUE TAL",
        "??!!??",
        "\xed\xa0\x80", // lone surrogate, invalid UTF-8
        "\xf4\x90\x80\x80", // beyond U+10FFFF
        "\xc2", // truncated two-byte sequence
        "\xe2\x96\x81hola", // the three-byte U+2581 the converter rewrites
        "a" ** 1000,
        " " ** 1000,
        "\xff" ** 1000,
    };
    for (corpus) |c| try once(c);
}

test "a source at the configured ceiling still fits every buffer" {
    var big: [limits.src_bytes]u8 = undefined;
    for (&big, 0..) |*b, i| b.* = if (i % 7 == 0) ' ' else 'a';
    try once(&big);
}
