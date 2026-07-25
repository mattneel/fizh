//! charsmap.zig — SentencePiece's `nmt_nfkc`, read from the artifact.
//!
//! The `.spm` vocabulary carries a `normalizer_spec` naming `nmt_nfkc` and a
//! `precompiled_charsmap`: a darts-clone double-array trie over UTF-8 byte
//! sequences followed by a pool of NUL-terminated replacements. Matching is
//! longest-prefix; a hit emits its replacement, a miss copies one UTF-8
//! character through.
//!
//! **This is not an implementation of NFKC.** It is an interpreter for the
//! table the model was trained with. Unicode normalization has a specification,
//! several revisions of it, and the model does not care about any of them — it
//! cares about the 237 KB of rewrite rules baked into its own vocabulary. ADR
//! 0017.
//!
//! Layout, little-endian throughout:
//!
//!     u32   trie_bytes
//!     [..]  trie_bytes of darts-clone units, u32 each
//!     [..]  replacement pool, NUL-terminated

const std = @import("std");
const assert = std.debug.assert;

pub const Error = error{OutTooSmall};

/// A matched rewrite rule: where its replacement lives in the pool, and how
/// many input bytes it consumes.
const Match = struct { at: u32, len: u32 };

/// A charsmap the artifact did not ship. `normalize` copies through.
pub const none: Charsmap = .{ .blob = &.{} };

pub const Charsmap = struct {
    blob: []const u8,

    pub fn present(self: Charsmap) bool {
        return self.blob.len >= 4;
    }

    fn trieLen(self: Charsmap) u32 {
        assert(self.present());
        return std.mem.readInt(u32, self.blob[0..4], .little) / 4;
    }

    fn unit(self: Charsmap, i: u32) u32 {
        const at = 4 + @as(usize, i) * 4;
        assert(at + 4 <= self.blob.len);
        return std.mem.readInt(u32, self.blob[at..][0..4], .little);
    }

    fn pool(self: Charsmap) []const u8 {
        const at = 4 + @as(usize, self.trieLen()) * 4;
        assert(at <= self.blob.len);
        return self.blob[at..];
    }

    /// Longest rewrite rule matching the head of `key`. Returns the pool offset
    /// and how many input bytes it consumes, or null when no rule applies.
    ///
    /// The bit layout is darts-clone's: bit 8 marks a leaf, bits 10.. hold the
    /// child offset scaled by bit 9, and the low byte is the edge label.
    fn longest(self: Charsmap, key: []const u8) ?Match {
        assert(self.present());
        const n = self.trieLen();
        if (n == 0) return null;

        var node: u32 = 0;
        var u = self.unit(node);
        node ^= offsetOf(u);

        var best: ?Match = null;
        for (key, 0..) |b, i| {
            node ^= b;
            if (node >= n) return best;
            u = self.unit(node);
            if (u & 0x800000FF != b) return best;
            node ^= offsetOf(u);
            if (node >= n) return best;
            if ((u >> 8) & 1 == 1) {
                best = .{ .at = self.unit(node) & 0x7FFFFFFF, .len = @intCast(i + 1) };
            }
        }
        return best;
    }

    fn offsetOf(u: u32) u32 {
        return (u >> 10) << @intCast((u & 0x200) >> 6);
    }

    fn replacement(self: Charsmap, at: u32) []const u8 {
        const p = self.pool();
        assert(at < p.len);
        const rest = p[at..];
        const end = std.mem.indexOfScalar(u8, rest, 0) orelse rest.len;
        return rest[0..end];
    }

    /// Rewrites `src` into `out`, returning the bytes written.
    ///
    /// `error.OutTooSmall` rather than truncation: the worst rule in the
    /// shipped tables expands 3 bytes to 33 (U+FDFA), so an adversarial input
    /// can grow 11x, and a half-normalized string is a wrong string rather than
    /// a short one. The caller turns this into `src_too_long`, which is what it
    /// is — after normalization, the text no longer fits.
    pub fn normalize(self: Charsmap, src: []const u8, out: []u8) Error!u32 {
        assert(out.len <= std.math.maxInt(u32));

        if (!self.present()) {
            if (src.len > out.len) return error.OutTooSmall;
            @memcpy(out[0..src.len], src);
            return @intCast(src.len);
        }

        var r: usize = 0;
        var w: u32 = 0;
        // SPEC §12.3: bounded. Every arm advances `r` by at least one byte.
        for (0..src.len) |_| {
            if (r >= src.len) break;
            if (self.longest(src[r..])) |hit| {
                const rep = self.replacement(hit.at);
                if (w + rep.len > out.len) return error.OutTooSmall;
                @memcpy(out[w..][0..rep.len], rep);
                w += @intCast(rep.len);
                r += hit.len;
                assert(hit.len > 0);
                continue;
            }
            const step = utf8Len(src[r]);
            const take = @min(step, src.len - r);
            if (w + take > out.len) return error.OutTooSmall;
            @memcpy(out[w..][0..take], src[r..][0..take]);
            w += @intCast(take);
            r += take;
        }

        assert(r == src.len);
        return w;
    }
};

fn utf8Len(b: u8) usize {
    if (b >= 0xF0) return 4;
    if (b >= 0xE0) return 3;
    if (b >= 0xC0) return 2;
    return 1;
}

test "an absent charsmap copies through" {
    var buf: [32]u8 = undefined;
    const n = try none.normalize("hola", &buf);
    try std.testing.expectEqualStrings("hola", buf[0..n]);
}

test "an absent charsmap still refuses to overflow" {
    var buf: [2]u8 = undefined;
    try std.testing.expectError(error.OutTooSmall, none.normalize("hola", &buf));
}

test "utf8Len covers every lead byte class" {
    try std.testing.expectEqual(@as(usize, 1), utf8Len('a'));
    try std.testing.expectEqual(@as(usize, 2), utf8Len(0xC3));
    try std.testing.expectEqual(@as(usize, 3), utf8Len(0xE2));
    try std.testing.expectEqual(@as(usize, 4), utf8Len(0xF0));
}
