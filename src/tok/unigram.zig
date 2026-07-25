//! tok/unigram.zig — unigram-LM tokenization by Viterbi over a byte lattice.
//!
//! SPEC §12.2: no recursion, anywhere. The lattice is a flat array indexed by
//! byte offset and the backtrace is a bounded loop, by construction.
//! SPEC §13 T3: this file eats arbitrary bytes and must never trap.

const std = @import("std");
const assert = std.debug.assert;

const trie = @import("trie.zig");
const Vocab = trie.Vocab;
const Cursor = trie.Cursor;

pub const space_marker = trie.space_marker;

/// One byte position of the lattice. `arena.zig` sizes `tok_lattice` from this.
pub const LatticeNode = extern struct {
    /// Best cumulative log-probability of any segmentation ending here.
    score: f32,
    /// Vocabulary id of the piece that closes the best path at this position.
    piece_id: u32,
    /// Byte offset where that piece began. `unreachable_prev` when no path
    /// reaches this position.
    prev: u32,
    /// Token count along the best path, so the length bound can be asserted
    /// without a second walk.
    n_tokens: u32,

    pub const unreachable_prev: u32 = std.math.maxInt(u32);

    pub const unreachable_node: LatticeNode = .{
        .score = -std.math.inf(f32),
        .piece_id = 0,
        .prev = unreachable_prev,
        .n_tokens = 0,
    };

    comptime {
        assert(@sizeOf(LatticeNode) == 16);
    }
};

pub const Params = struct {
    unk_id: u32,
    eos_id: u32,
    /// Charged on top of the worst piece score for a character no piece covers.
    /// SentencePiece's default; large enough that an unknown edge is only ever
    /// taken when the lattice has no alternative.
    unk_penalty: f32 = 10.0,
    append_eos: bool = true,
};

pub const Error = error{
    /// The segmentation needs more token slots than the caller provided.
    TooManyTokens,
    /// Detokenized text does not fit the caller's buffer.
    OutTooSmall,
};

/// SentencePiece preprocessing, length-preserving except for the one-byte
/// dummy prefix: runs of ASCII whitespace collapse to a single `space_marker`,
/// leading and trailing whitespace disappear, and a marker is prepended.
///
/// Returns bytes written. `out` must hold `src.len + 1`.
pub fn normalize(src: []const u8, out: []u8) u32 {
    assert(out.len >= src.len + 1);
    assert(src.len <= std.math.maxInt(u32) - 1);

    out[0] = space_marker;
    var w: u32 = 1;
    var pending_space = false;

    for (src) |b| {
        if (isSpace(b)) {
            pending_space = true;
            continue;
        }
        if (pending_space and w > 1) {
            out[w] = space_marker;
            w += 1;
        }
        pending_space = false;
        out[w] = b;
        w += 1;
    }

    assert(w >= 1);
    assert(w <= src.len + 1);
    return w;
}

fn isSpace(b: u8) bool {
    return b == ' ' or b == '\t' or b == '\n' or b == '\r' or b == 0x0b or b == 0x0c;
}

/// Viterbi over the byte lattice of `text`, which must already be normalized.
/// Returns the number of ids written to `out`.
///
/// Every position is reachable because an unknown-character edge always exists,
/// so this cannot fail to find a path — only to fit one.
pub fn encode(v: Vocab, p: Params, text: []const u8, lattice: []LatticeNode, out: []u32) Error!u32 {
    assert(lattice.len >= text.len + 1);
    assert(p.unk_id < v.size and p.eos_id < v.size);

    const n: u32 = @intCast(text.len);
    forward(v, p, text, lattice[0 .. n + 1]);

    const final = lattice[n];
    assert(final.prev != LatticeNode.unreachable_prev or n == 0);

    const extra: u32 = if (p.append_eos) 1 else 0;
    if (final.n_tokens + extra > out.len) return error.TooManyTokens;

    backtrace(lattice[0 .. n + 1], out[0..final.n_tokens]);
    if (p.append_eos) out[final.n_tokens] = p.eos_id;

    const count = final.n_tokens + extra;
    assert(count <= n + 1);
    return count;
}

fn forward(v: Vocab, p: Params, text: []const u8, lattice: []LatticeNode) void {
    assert(lattice.len == text.len + 1);
    assert(v.max_piece_len >= 1);

    const unk_score = v.min_score - p.unk_penalty;
    @memset(lattice, LatticeNode.unreachable_node);
    lattice[0] = .{ .score = 0, .piece_id = 0, .prev = 0, .n_tokens = 0 };

    for (0..text.len) |i| {
        const here = lattice[i];
        if (i != 0 and here.prev == LatticeNode.unreachable_prev) continue;

        var cur = Cursor.root(v);
        var matched = false;
        const reach = @min(text.len - i, v.max_piece_len);
        for (0..reach) |k| {
            cur = cur.descend(v, text[i + k]) orelse break;
            if (cur.pieceHere(v)) |id| {
                matched = true;
                relax(lattice, @intCast(i), @intCast(i + k + 1), id, here.score + v.score(id), here.n_tokens + 1);
            }
        }

        // The edge that keeps the lattice connected, and only that. Offering it
        // alongside real pieces would let a cheap unknown outbid an expensive
        // but correct segmentation — which is how "¿qué" becomes "qu".
        if (!matched) {
            // One whole character, so a multi-byte sequence never splits into
            // mojibake.
            const clen = charLen(text[i..]);
            relax(lattice, @intCast(i), @intCast(i + clen), p.unk_id, here.score + unk_score, here.n_tokens + 1);
        }
    }
}

fn relax(lattice: []LatticeNode, from: u32, to: u32, id: u32, score: f32, n_tokens: u32) void {
    assert(to > from);
    assert(to < lattice.len);

    const target = &lattice[to];
    if (target.prev != LatticeNode.unreachable_prev and !(score > target.score)) return;
    target.* = .{ .score = score, .piece_id = id, .prev = from, .n_tokens = n_tokens };
}

fn backtrace(lattice: []const LatticeNode, out: []u32) void {
    assert(lattice.len >= 1);
    assert(out.len == lattice[lattice.len - 1].n_tokens);

    var at: u32 = @intCast(lattice.len - 1);
    var w: u32 = @intCast(out.len);
    // Bounded by the lattice: every hop strictly decreases `at`.
    for (0..lattice.len) |_| {
        if (at == 0) break;
        assert(w > 0);
        w -= 1;
        out[w] = lattice[at].piece_id;
        const prev = lattice[at].prev;
        assert(prev < at);
        at = prev;
    }
    assert(at == 0);
    assert(w == 0);
}

/// UTF-8 sequence length from the lead byte, clamped to what is left. A byte
/// that cannot start a sequence — including `space_marker` — advances by one,
/// which is what keeps this total over arbitrary input.
fn charLen(rest: []const u8) u32 {
    assert(rest.len >= 1);
    const lead = rest[0];
    const want: u32 = if (lead < 0x80)
        1
    else if (lead >= 0xc2 and lead <= 0xdf)
        2
    else if (lead >= 0xe0 and lead <= 0xef)
        3
    else if (lead >= 0xf0 and lead <= 0xf4)
        4
    else
        1;

    // Only take the whole sequence if the continuation bytes are really there.
    var have: u32 = 1;
    for (1..want) |k| {
        if (k >= rest.len or rest[k] & 0xc0 != 0x80) break;
        have += 1;
    }
    const len = @min(have, @as(u32, @intCast(rest.len)));
    assert(len >= 1 and len <= 4);
    return len;
}

/// Ids back to UTF-8. Control symbols are dropped, `space_marker` becomes a
/// space, and the dummy prefix is removed. Returns bytes written.
pub fn decode(v: Vocab, ids: []const u32, out: []u8) Error!u32 {
    assert(v.size >= 2);
    assert(out.len <= std.math.maxInt(u32));

    var w: u32 = 0;
    for (ids) |id| {
        assert(id < v.size);
        if (v.isSpecial(id)) continue;
        const piece = v.piece(id);
        if (@as(usize, w) + piece.len > out.len) return error.OutTooSmall;
        for (piece) |b| {
            out[w] = if (b == space_marker) ' ' else b;
            w += 1;
        }
    }

    if (w > 0 and out[0] == ' ') {
        std.mem.copyForwards(u8, out[0 .. w - 1], out[1..w]);
        w -= 1;
    }
    assert(w <= out.len);
    return w;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

/// A vocabulary covering every byte plus a few words, so `decode(encode(t))`
/// is exactly `normalize(t)` and the lattice is never forced onto an unknown
/// edge.
const full_words: [258][]const u8 = blk: {
    @setEvalBranchQuota(200_000);
    var words: [258][]const u8 = undefined;
    words[0] = "</s>";
    words[1] = "<unk>";
    for (0..256) |b| {
        const one = [_]u8{@intCast(b)};
        words[2 + b] = &one;
    }
    break :blk words;
};

const Full = trie.Fixture(&full_words, &.{ 0, 1 });

const Toy = trie.Fixture(
    &.{ "</s>", "<unk>", "\xff", "\xffhola", "\xffque", "\xfftal", "a", "h", "l", "o", "q", "t", "u", "!" },
    &.{ 0, 1 },
);

const params: Params = .{ .unk_id = 1, .eos_id = 0 };

fn run(comptime V: type, text: []const u8, ids: []u32) !u32 {
    const S = struct {
        var lattice: [1024]LatticeNode = undefined;
        var norm: [512]u8 = undefined;
    };
    const n = normalize(text, &S.norm);
    return encode(V.vocab, params, S.norm[0..n], &S.lattice, ids);
}

test "normalize collapses whitespace and prefixes a marker" {
    var out: [64]u8 = undefined;
    try testing.expectEqual(@as(u32, 9), normalize("  hola   tal ", &out));
    try testing.expectEqualSlices(u8, "\xffhola\xfftal", out[0..9]);
    try testing.expectEqual(@as(u32, 1), normalize("", &out));
    try testing.expectEqual(@as(u32, 1), normalize("   \t\n ", &out));
}

test "viterbi prefers whole words over their letters" {
    var ids: [64]u32 = undefined;
    const n = try run(Toy, "hola", &ids);
    try testing.expectEqual(@as(u32, 2), n); // "\xffhola" + eos
    try testing.expectEqualSlices(u8, "\xffhola", Toy.vocab.piece(ids[0]));
    try testing.expectEqual(params.eos_id, ids[1]);
}

test "unknown characters take one whole UTF-8 character each" {
    var ids: [64]u32 = undefined;
    // The toy vocabulary has no emoji and no accented letters.
    const n = try run(Toy, "hola 🐟", &ids);
    try testing.expect(n >= 3);
    // Exactly one unknown edge for the four-byte emoji, not four.
    var unks: u32 = 0;
    for (ids[0..n]) |id| {
        if (id == params.unk_id) unks += 1;
    }
    try testing.expectEqual(@as(u32, 1), unks);
}

test "a byte-complete vocabulary round-trips exactly" {
    const cases = [_][]const u8{
        "hola",
        "¿qué tal?",
        "hello world",
        "🐟🐠 fish",
        "a  b\tc\nd",
        "",
        "  leading and trailing  ",
        "Grüße aus Köln",
    };
    var ids: [1024]u32 = undefined;
    var text: [512]u8 = undefined;
    var norm: [512]u8 = undefined;
    var scratch: [512]u8 = undefined;

    for (cases) |c| {
        const n = try run(Full, c, &ids);
        const w = try decode(Full.vocab, ids[0..n], &text);
        const nn = normalize(c, &norm);
        // `decode` strips the dummy prefix, so compare against the normalized
        // form with its leading marker turned back into nothing.
        const expect = expected(norm[0..nn], &scratch);
        try testing.expectEqualStrings(expect, text[0..w]);
    }
}

/// The normalized form as `decode` would render it: markers to spaces, leading
/// space dropped.
fn expected(norm: []const u8, scratch: []u8) []const u8 {
    var w: usize = 0;
    for (norm) |b| {
        scratch[w] = if (b == space_marker) ' ' else b;
        w += 1;
    }
    if (w > 0 and scratch[0] == ' ') {
        std.mem.copyForwards(u8, scratch[0 .. w - 1], scratch[1..w]);
        w -= 1;
    }
    return scratch[0..w];
}

test "the full vocabulary round-trips" {
    // SPEC M1. The normalizer is deliberately lossy — it collapses whitespace —
    // so the property that actually holds is that tokenization loses nothing
    // the normalizer had not already discarded, for every piece in the
    // vocabulary including the whitespace ones.
    var ids: [1024]u32 = undefined;
    var text: [512]u8 = undefined;
    var got: [512]u8 = undefined;
    var norm: [512]u8 = undefined;
    var scratch: [512]u8 = undefined;

    for (0..Full.vocab.size) |i| {
        const id: u32 = @intCast(i);
        if (Full.vocab.isSpecial(id)) continue;

        const t = try decode(Full.vocab, &[_]u32{id}, &text);
        const n = try run(Full, text[0..t], &ids);
        const w = try decode(Full.vocab, ids[0..n], &got);

        const nn = normalize(text[0..t], &norm);
        try testing.expectEqualStrings(expected(norm[0..nn], &scratch), got[0..w]);
    }
}

test "every non-special piece is reachable from its own text" {
    // A piece the Viterbi can never select is dead weight in the artifact.
    var ids: [64]u32 = undefined;
    var text: [64]u8 = undefined;

    for (0..Toy.vocab.size) |i| {
        const id: u32 = @intCast(i);
        if (Toy.vocab.isSpecial(id)) continue;

        const t = try decode(Toy.vocab, &[_]u32{id}, &text);
        if (t == 0) continue; // a bare word-boundary marker renders to nothing
        const n = try run(Toy, text[0..t], &ids);
        try testing.expect(std.mem.indexOfScalar(u32, ids[0..n], id) != null);
    }
}

test "arbitrary bytes never trap and stay within bounds" {
    var ids: [2048]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x5eed);
    var buf: [256]u8 = undefined;

    for (0..2000) |_| {
        const len = rng.random().intRangeAtMost(usize, 0, buf.len);
        rng.random().bytes(buf[0..len]);
        const n = try run(Toy, buf[0..len], &ids);
        try testing.expect(n <= len + 2);
        for (ids[0..n]) |id| try testing.expect(id < Toy.vocab.size);
    }
}

test "encode reports rather than overflowing a short output" {
    var ids: [2]u32 = undefined;
    try testing.expectError(error.TooManyTokens, run(Toy, "abcdefghij", &ids));
}

test "decode reports rather than overflowing a short output" {
    var out: [2]u8 = undefined;
    const long = [_]u32{ 3, 3, 3, 3 }; // "\xffhola" four times
    try testing.expectError(error.OutTooSmall, decode(Toy.vocab, &long, &out));
}

test "lattice node is 16 bytes and starts unreachable" {
    try testing.expectEqual(@as(usize, 16), @sizeOf(LatticeNode));
    try testing.expect(LatticeNode.unreachable_node.prev == LatticeNode.unreachable_prev);
}
