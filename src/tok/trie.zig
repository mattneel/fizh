//! tok/trie.zig — the prefix structure the Viterbi walks.
//!
//! It is a trie by interface — `root`, `descend`, `pieceHere` — backed by the
//! vocabulary held in lexicographic order rather than by nodes and edges. A
//! cursor is the half-open rank range `[lo, hi)` of pieces sharing the prefix
//! walked so far; `descend` narrows it with two bounded binary searches.
//!
//! Why not nodes and edges: a node-and-edge trie over a 32k-piece vocabulary is
//! one to four megabytes that has to be either built at load or shipped in the
//! artifact, and SPEC §14 gives a direction 20 MB total against ~17 MB of
//! weights. This form costs one `u32` per piece and nothing at load. See
//! `docs/adr/0004-implicit-trie-and-tokenizer-conventions.md`.
//!
//! SPEC §12.2: no recursion. SPEC §12.3: every loop below has a comptime bound.

const std = @import("std");
const assert = std.debug.assert;

/// Word boundary. SPEC-adjacent artifact convention: `tools/convert.py`
/// rewrites SentencePiece's three-byte U+2581 to this single byte, which cannot
/// occur in well-formed UTF-8, so normalization stays length-preserving and the
/// lattice stays the size SPEC §4.2 says it is.
pub const space_marker: u8 = 0xff;

/// Ranks fit in `u32`, so 32 halvings always exhaust a search range. Written as
/// a comptime bound rather than a `while` so SPEC §12.3 holds literally.
const max_halvings: u32 = 32;

pub const max_piece_bytes: u32 = 512;

/// One direction's vocabulary, as slices into that direction's weight slot.
/// Everything here has been through `validate`.
pub const Vocab = struct {
    /// Piece bytes, concatenated in vocabulary-id order.
    pieces: []const u8,
    /// `size + 1` entries; `offsets[id]..offsets[id + 1]` is piece `id`.
    offsets: []const u32,
    /// Unigram log-probability per piece.
    scores: []const f32,
    /// Vocabulary ids sorted lexicographically by piece bytes.
    order: []const u32,
    /// Bit 0: control or unknown symbol. Never matched, never emitted.
    flags: []const u8,
    size: u32,
    max_piece_len: u32,
    /// The worst score in the vocabulary. `unigram.zig` prices its
    /// unknown-character edge below this, so an unknown edge is only ever taken
    /// when nothing else reaches.
    min_score: f32,

    pub const flag_special: u8 = 1;

    pub fn piece(self: Vocab, id: u32) []const u8 {
        assert(id < self.size);
        const a = self.offsets[id];
        const b = self.offsets[id + 1];
        assert(b >= a);
        return self.pieces[a..b];
    }

    pub fn score(self: Vocab, id: u32) f32 {
        assert(id < self.size);
        return self.scores[id];
    }

    pub fn isSpecial(self: Vocab, id: u32) bool {
        assert(id < self.size);
        return self.flags[id] & flag_special != 0;
    }

    /// Called once, by `model/format.zig`, on untrusted bytes. Returns false
    /// rather than asserting; every assertion elsewhere in `tok/` rests on this
    /// having returned true.
    pub fn validate(self: Vocab) bool {
        if (self.size < 2) return false;
        if (self.offsets.len != @as(usize, self.size) + 1) return false;
        if (self.scores.len != self.size) return false;
        if (self.order.len != self.size) return false;
        if (self.flags.len != self.size) return false;
        if (self.offsets[0] != 0) return false;
        if (self.offsets[self.size] != self.pieces.len) return false;

        var longest: u32 = 0;
        var worst: f32 = std.math.inf(f32);
        for (0..self.size) |i| {
            const a = self.offsets[i];
            const b = self.offsets[i + 1];
            if (b < a or b > self.pieces.len) return false;
            const len = b - a;
            if (len == 0 or len > max_piece_bytes) return false;
            if (len > longest) longest = len;
            if (!std.math.isFinite(self.scores[i])) return false;
            if (self.scores[i] < worst) worst = self.scores[i];
            if (self.flags[i] & ~flag_special != 0) return false;
        }
        if (longest != self.max_piece_len) return false;
        if (worst != self.min_score) return false;

        // Strictly increasing pieces prove `order` is a permutation: equal
        // indices would mean equal pieces, and `size` distinct values below
        // `size` cover the range.
        if (self.order[0] >= self.size) return false;
        for (1..self.size) |i| {
            if (self.order[i] >= self.size) return false;
            const prev = self.rawPiece(self.order[i - 1]);
            const cur = self.rawPiece(self.order[i]);
            if (std.mem.order(u8, prev, cur) != .lt) return false;
        }
        return true;
    }

    /// `piece` without the `id < size` assertion, for use inside `validate`
    /// before that invariant is known to hold.
    fn rawPiece(self: Vocab, id: u32) []const u8 {
        return self.pieces[self.offsets[id]..self.offsets[id + 1]];
    }
};

/// A position in the implicit trie: every piece in `[lo, hi)` begins with the
/// same `depth` bytes.
pub const Cursor = struct {
    lo: u32,
    hi: u32,
    depth: u32,

    pub fn root(v: Vocab) Cursor {
        assert(v.size >= 2);
        return .{ .lo = 0, .hi = v.size, .depth = 0 };
    }

    pub fn isEmpty(self: Cursor) bool {
        return self.lo >= self.hi;
    }

    /// Narrows to the pieces whose next byte is `b`, or null when none do.
    pub fn descend(self: Cursor, v: Vocab, b: u8) ?Cursor {
        assert(self.lo <= self.hi);
        assert(self.hi <= v.size);

        const start = bound(v, self.lo, self.hi, self.depth, b, .first_ge);
        if (start == self.hi) return null;
        const end = bound(v, start, self.hi, self.depth, b, .first_gt);
        if (start == end) return null;

        assert(end <= self.hi);
        return .{ .lo = start, .hi = end, .depth = self.depth + 1 };
    }

    /// The piece that ends exactly here, if the vocabulary has one and it is
    /// not a control symbol.
    pub fn pieceHere(self: Cursor, v: Vocab) ?u32 {
        if (self.isEmpty()) return null;
        assert(self.lo < v.size);
        const id = v.order[self.lo];
        const p = v.piece(id);
        if (p.len != self.depth) return null;
        if (v.isSpecial(id)) return null;
        return id;
    }
};

const Bound = enum { first_ge, first_gt };

/// Key of rank `r` at `depth`: the byte at that depth, or -1 for a piece that
/// has already ended, which sorts before every byte.
fn keyAt(v: Vocab, r: u32, depth: u32) i16 {
    assert(r < v.size);
    const p = v.piece(v.order[r]);
    if (p.len <= depth) return -1;
    return p[depth];
}

fn bound(v: Vocab, lo: u32, hi: u32, depth: u32, b: u8, kind: Bound) u32 {
    assert(lo <= hi);
    assert(hi <= v.size);

    var a = lo;
    var z = hi;
    for (0..max_halvings) |_| {
        if (a >= z) break;
        const mid = a + (z - a) / 2;
        const k = keyAt(v, mid, depth);
        const go_right = switch (kind) {
            .first_ge => k < b,
            .first_gt => k <= b,
        };
        if (go_right) a = mid + 1 else z = mid;
    }
    assert(a == z);
    return a;
}

// -- tests ------------------------------------------------------------------

/// A vocabulary literal, built at comptime so tests need no allocator.
pub fn Fixture(comptime words: []const []const u8, comptime specials: []const u32) type {
    return struct {
        const size = words.len;

        const blob = blk: {
            var out: []const u8 = "";
            for (words) |w| out = out ++ w;
            break :blk out;
        };
        const offs = blk: {
            var out: [size + 1]u32 = undefined;
            out[0] = 0;
            for (words, 0..) |w, i| out[i + 1] = out[i] + @as(u32, @intCast(w.len));
            break :blk out;
        };
        const scores = blk: {
            var out: [size]f32 = undefined;
            // Longer pieces score better, so the Viterbi prefers them.
            for (words, 0..) |w, i| out[i] = -20.0 + @as(f32, @floatFromInt(w.len));
            break :blk out;
        };
        const worst = blk: {
            var m: f32 = std.math.inf(f32);
            for (scores) |s| m = @min(m, s);
            break :blk m;
        };
        const flags = blk: {
            var out: [size]u8 = @splat(0);
            for (specials) |s| out[s] = Vocab.flag_special;
            break :blk out;
        };
        const order = blk: {
            @setEvalBranchQuota(20 * size * size + 10_000);
            var out: [size]u32 = undefined;
            for (0..size) |i| out[i] = i;
            // Insertion sort: comptime, tiny, and obviously correct.
            for (1..size) |i| {
                var j = i;
                while (j > 0 and std.mem.order(u8, words[out[j]], words[out[j - 1]]) == .lt) : (j -= 1) {
                    const t = out[j];
                    out[j] = out[j - 1];
                    out[j - 1] = t;
                }
            }
            break :blk out;
        };
        const longest = blk: {
            var m: u32 = 0;
            for (words) |w| m = @max(m, @as(u32, @intCast(w.len)));
            break :blk m;
        };

        pub const vocab: Vocab = .{
            .pieces = blob,
            .offsets = &offs,
            .scores = &scores,
            .order = &order,
            .flags = &flags,
            .size = size,
            .max_piece_len = longest,
            .min_score = worst,
        };
    };
}

const Toy = Fixture(&.{ "</s>", "<unk>", "\xffh", "\xffhola", "a", "l", "o", "\xffho" }, &.{ 0, 1 });

test "fixture vocabulary validates" {
    try std.testing.expect(Toy.vocab.validate());
}

test "descend narrows to the pieces sharing a prefix" {
    const v = Toy.vocab;
    const c0 = Cursor.root(v);
    try std.testing.expect(c0.pieceHere(v) == null);

    const c1 = c0.descend(v, trie_space).?;
    try std.testing.expectEqual(@as(u32, 1), c1.depth);
    try std.testing.expect(c1.pieceHere(v) == null);

    const c2 = c1.descend(v, 'h').?;
    try std.testing.expect(c2.pieceHere(v) != null); // "\xffh"

    const c3 = c2.descend(v, 'o').?;
    try std.testing.expect(c3.pieceHere(v) != null); // "\xffho"

    try std.testing.expect(c3.descend(v, 'z') == null);
    const c4 = c3.descend(v, 'l').?;
    try std.testing.expect(c4.pieceHere(v) == null); // "\xffhol" is not a piece
    try std.testing.expect(c4.descend(v, 'a').?.pieceHere(v) != null);
}

const trie_space = space_marker;

test "control symbols are never matched" {
    const v = Toy.vocab;
    var c = Cursor.root(v);
    for ("</s>") |b| c = c.descend(v, b).?;
    // The piece is there — it just refuses to be a match.
    try std.testing.expect(c.pieceHere(v) == null);
}

test "validate rejects a broken vocabulary" {
    var v = Toy.vocab;
    v.size = 0;
    try std.testing.expect(!v.validate());

    v = Toy.vocab;
    v.max_piece_len = 1;
    try std.testing.expect(!v.validate());

    v = Toy.vocab;
    const bad_order = [_]u32{ 7, 6, 5, 4, 3, 2, 1, 0 };
    v.order = &bad_order;
    try std.testing.expect(!v.validate());

    v = Toy.vocab;
    const short_offsets = [_]u32{ 0, 1 };
    v.offsets = &short_offsets;
    try std.testing.expect(!v.validate());
}

test "every byte value descends without trapping" {
    const v = Toy.vocab;
    const root = Cursor.root(v);
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        const c = root.descend(v, @intCast(b));
        if (c) |cur| try std.testing.expect(!cur.isEmpty());
    }
}
