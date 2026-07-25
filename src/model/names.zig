//! model/names.zig — tensor name hashes. SPEC §6.
//!
//! Names are FNV-1a-64 of a canonical dotted string. The stems are comptime, so
//! for a fixed name the whole hash folds away at compile time; only the layer
//! index is folded in at run time, and only ~160 times per load.
//!
//! `tools/convert.py` computes the identical hash from the identical string.
//! The canonical names are:
//!
//!     emb                    i8  [vocab][d_model]  + scales[vocab]
//!     emb.bias               f32 [vocab]
//!     enc.ln.gain/.bias      f32 [d_model]
//!     dec.ln.gain/.bias      f32 [d_model]
//!     enc.{i}.att.{q,k,v,o}.w      i8  [d][d] + scales[d]
//!     enc.{i}.att.{q,k,v,o}.bias   f32 [d]
//!     enc.{i}.att.ln.{gain,bias}   f32 [d]
//!     enc.{i}.ffn.w1               i8  [ffn][d] + scales[ffn]
//!     enc.{i}.ffn.bias1            f32 [ffn]
//!     enc.{i}.ffn.w2               i8  [d][ffn] + scales[d]
//!     enc.{i}.ffn.bias2            f32 [d]
//!     enc.{i}.ffn.ln.{gain,bias}   f32 [d]
//!     dec.{i}.rnn.w                i8  [d][d] + scales[d]   (SSRU, ADR 0008)
//!     dec.{i}.rnn.wf               i8  [d][d] + scales[d]
//!     dec.{i}.rnn.bf               f32 [d]
//!     dec.{i}.rnn.ln.{gain,bias}   f32 [d]
//!     dec.{i}.xa.*                 as enc.{i}.att.*
//!     dec.{i}.ffn.*                as enc.{i}.ffn.*
//!     tok.pieces   u8[]    tok.offsets u32[V+1]   tok.scores f32[V]
//!     tok.order    u32[V]  tok.flags   u8[V]
//!     tok.nonbreaking u8[]  (optional, NUL-separated lowercase prefixes)
//!     sl.offsets   u32[V+1]  sl.targets u32[nnz]  sl.frequent u32[F]

const std = @import("std");
const assert = std.debug.assert;

const fnv_offset: u64 = 0xcbf29ce484222325;
const fnv_prime: u64 = 0x00000100000001b3;

/// Hash of a fixed name. Always folds at compile time.
pub fn of(comptime name: []const u8) u64 {
    comptime assert(name.len > 0);
    return comptime fold(fnv_offset, name);
}

/// Hash of `<prefix><index><suffix>`, e.g. `enc.` + `3` + `.att.q.w`.
pub fn indexed(comptime prefix: []const u8, index: u32, comptime suffix: []const u8) u64 {
    assert(index < 1000);
    const seed = comptime fold(fnv_offset, prefix);
    return fold(foldDecimal(seed, index), suffix);
}

pub fn enc(index: u32, comptime suffix: []const u8) u64 {
    return indexed("enc.", index, "." ++ suffix);
}

pub fn dec(index: u32, comptime suffix: []const u8) u64 {
    return indexed("dec.", index, "." ++ suffix);
}

fn fold(seed: u64, bytes: []const u8) u64 {
    var h = seed;
    for (bytes) |b| {
        h ^= b;
        h *%= fnv_prime;
    }
    return h;
}

/// Decimal digits, no leading zeros, no allocation, no recursion.
fn foldDecimal(seed: u64, value: u32) u64 {
    assert(value < 1_000_000_000);
    if (value == 0) return fold(seed, "0");

    var digits: [10]u8 = undefined;
    var n: usize = 0;
    var v = value;
    // Bounded by the width of a `u32` in decimal.
    for (0..10) |_| {
        if (v == 0) break;
        digits[n] = '0' + @as(u8, @intCast(v % 10));
        n += 1;
        v /= 10;
    }
    assert(n > 0 and v == 0);

    var h = seed;
    var i = n;
    for (0..n) |_| {
        i -= 1;
        h ^= digits[i];
        h *%= fnv_prime;
    }
    return h;
}

// -- the fixed names --------------------------------------------------------

pub const emb = of("emb");
pub const emb_alpha = of("emb.alpha");

/// The static activation multiplier that goes with a weight (ADR 0012).
pub fn alphaOf(comptime _: u64) u64 {
    return emb_alpha;
}
pub const emb_bias = of("emb.bias");
pub const enc_ln_gain = of("enc.ln.gain");
pub const enc_ln_bias = of("enc.ln.bias");
pub const dec_ln_gain = of("dec.ln.gain");
pub const dec_ln_bias = of("dec.ln.bias");

pub const tok_pieces = of("tok.pieces");
pub const tok_offsets = of("tok.offsets");
pub const tok_scores = of("tok.scores");
pub const tok_order = of("tok.order");
pub const tok_flags = of("tok.flags");
/// Non-breaking prefixes for the source language, NUL-separated, lowercase.
/// Optional; see ADR 0011.
pub const tok_nonbreaking = of("tok.nonbreaking");

pub const sl_offsets = of("sl.offsets");
pub const sl_targets = of("sl.targets");
pub const sl_frequent = of("sl.frequent");

// -- tests ------------------------------------------------------------------

test "hashes match FNV-1a-64 of the canonical string" {
    // Reference values for the classic FNV-1a-64 test vectors, so a change to
    // the constants below shows up here and not in a mysterious missing_tensor.
    try std.testing.expectEqual(@as(u64, 0xaf63dc4c8601ec8c), of("a"));
    try std.testing.expectEqual(@as(u64, 0x85944171f73967e8), of("foobar"));
}

test "indexed names equal the flat name they spell" {
    try std.testing.expectEqual(of("enc.0.att.q.w"), enc(0, "att.q.w"));
    try std.testing.expectEqual(of("enc.7.ffn.w2"), enc(7, "ffn.w2"));
    try std.testing.expectEqual(of("dec.12.xa.ln.gain"), dec(12, "xa.ln.gain"));
    try std.testing.expectEqual(of("enc.31.att.o.bias"), enc(31, "att.o.bias"));
}

test "distinct names hash distinctly across the whole space" {
    // A collision would silently load one tensor's bytes into another's slot.
    var seen: [4096]u64 = undefined;
    var n: usize = 0;

    const fixed = [_]u64{
        emb,        emb_bias,   enc_ln_gain, enc_ln_bias, dec_ln_gain, dec_ln_bias,
        tok_pieces, tok_offsets, tok_scores, tok_order,   tok_flags,
        sl_offsets, sl_targets, sl_frequent,
    };
    for (fixed) |h| {
        seen[n] = h;
        n += 1;
    }

    const attn_parts = [_][]const u8{
        "att.q.w",  "att.q.bias", "att.k.w",  "att.k.bias", "att.v.w", "att.v.bias",
        "att.o.w",  "att.o.bias", "att.ln.gain", "att.ln.bias",
        "ffn.w1",   "ffn.bias1",  "ffn.w2",   "ffn.bias2",  "ffn.ln.gain", "ffn.ln.bias",
        "rnn.w",    "rnn.wf",     "rnn.bf",   "rnn.ln.gain", "rnn.ln.bias",
        "xa.q.w",   "xa.k.w",     "xa.v.w",   "xa.o.w",
    };
    for (0..32) |i| {
        inline for (attn_parts) |p| {
            seen[n] = enc(@intCast(i), p);
            n += 1;
            seen[n] = dec(@intCast(i), p);
            n += 1;
        }
    }

    std.mem.sort(u64, seen[0..n], {}, std.sort.asc(u64));
    for (1..n) |i| try std.testing.expect(seen[i] != seen[i - 1]);
}
