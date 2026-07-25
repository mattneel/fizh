//! tok/ssplit.zig — sentence splitting.
//!
//! Bergamot's models are trained on single sentences and emit `</s>` at the end
//! of one. Feed a paragraph in and you get its first sentence back: measured at
//! **−24.18 chrF++** on en→de against bergamot-translator, with fizh's output
//! 0.64× the reference length and short on 95 of 100 paragraphs.
//!
//! bergamot-translator solves this with ssplit-cpp, a Moses-derived splitter
//! with per-language non-breaking prefix lists. This is the same algorithm:
//! find a terminator, then look for a reason *not* to break there.
//!
//! SPEC §12: no allocator, no recursion, every loop bounded by the input.
//! SPEC §11: this runs on untrusted source text and must never trap — every
//! index below is bounded by `text.len` and the output count by `out.len`.

const std = @import("std");
const assert = std.debug.assert;

/// A half-open byte range of `text`.
pub const Span = extern struct {
    start: u32,
    end: u32,

    pub fn len(self: Span) u32 {
        assert(self.end >= self.start);
        return self.end - self.start;
    }
};

/// Non-breaking prefixes, NUL-separated and lowercase, as the artifact ships
/// them (`tok.nonbreaking`). Empty is legal and simply disables that rule.
pub const Prefixes = struct {
    blob: []const u8 = &.{},

    /// True when `word` is a prefix after which a period does not end a
    /// sentence — "Dr", "Nr", "z.B", "Sr".
    pub fn contains(self: Prefixes, word: []const u8) bool {
        if (word.len == 0 or word.len > max_prefix_len) return false;

        var lowered: [max_prefix_len]u8 = undefined;
        for (word, 0..) |c, i| lowered[i] = std.ascii.toLower(c);
        const needle = lowered[0..word.len];

        var at: usize = 0;
        // Bounded by the blob: every iteration consumes at least one byte.
        while (at < self.blob.len) {
            const end = std.mem.indexOfScalarPos(u8, self.blob, at, 0) orelse self.blob.len;
            if (std.mem.eql(u8, self.blob[at..end], needle)) return true;
            at = end + 1;
        }
        return false;
    }
};

pub const max_prefix_len: usize = 32;

/// Splits `text` into sentences, writing their spans to `out`. Returns the
/// count, which is at least one for non-empty input and never exceeds
/// `out.len`: when the caller's array fills, the remainder of the text becomes
/// the final span rather than being dropped.
pub fn split(text: []const u8, prefixes: Prefixes, out: []Span) u32 {
    assert(out.len >= 1);
    if (text.len == 0) return 0;

    var count: u32 = 0;
    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) : (i += 1) {
        if (!isTerminator(text[i])) continue;
        if (count + 1 >= out.len) break; // leave room for the tail

        const after = breakAfter(text, i, prefixes) orelse continue;
        const end = trimEnd(text, start, after);
        if (end > start) {
            out[count] = .{ .start = @intCast(start), .end = @intCast(end) };
            count += 1;
        }
        start = skipSpace(text, after);
        i = if (start > i) start - 1 else i;
    }

    const tail = trimEnd(text, start, text.len);
    if (tail > start) {
        out[count] = .{ .start = @intCast(start), .end = @intCast(tail) };
        count += 1;
    }

    assert(count <= out.len);
    assert(count > 0 or allSpace(text));
    return count;
}

/// Given a terminator at `i`, returns the offset just past the sentence, or
/// null when this is not a real boundary. This is where every "reason not to
/// break" lives.
fn breakAfter(text: []const u8, i: usize, prefixes: Prefixes) ?usize {
    var end = i + 1;

    // "..." is one terminator, not three sentences.
    while (end < text.len and isTerminator(text[end])) end += 1;

    // Trailing closers belong to the sentence that is ending: `He left."`
    while (end < text.len and isCloser(text[end])) end += 1;

    // A terminator that is not followed by space is inside a token —
    // "example.com", "3.14", "U.S.A".
    if (end >= text.len) return end;
    if (!isSpace(text[end])) return null;

    const next = skipSpace(text, end);
    if (next >= text.len) return end;
    if (!startsSentence(text, next)) return null;

    if (text[i] == '.') {
        // A single capital before the period is an initial: "J. R. Tolkien".
        const word = wordBefore(text, i);
        if (word.len == 1 and std.ascii.isUpper(text[i - 1])) return null;
        if (prefixes.contains(word)) return null;
    }
    return end;
}

fn wordBefore(text: []const u8, i: usize) []const u8 {
    var s = i;
    // Bounded by `i`; stops at the first byte that cannot be part of a word.
    while (s > 0 and isWordByte(text[s - 1])) s -= 1;
    return text[s..i];
}

fn isWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c >= 0x80;
}

fn isTerminator(c: u8) bool {
    return c == '.' or c == '!' or c == '?';
}

fn isCloser(c: u8) bool {
    return c == '"' or c == '\'' or c == ')' or c == ']' or c == '}';
}

fn isSpace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r';
}

/// What may begin a sentence: an ASCII capital, a digit, an opening quote or
/// bracket, or any non-ASCII lead byte. The last one is deliberately
/// permissive — `¿`, `«` and every accented capital live there, and case is not
/// knowable without a Unicode table this project does not carry.
fn startsSentence(text: []const u8, at: usize) bool {
    const c = text[at];
    if (std.ascii.isUpper(c) or std.ascii.isDigit(c)) return true;
    if (c == '"' or c == '\'' or c == '(' or c == '[' or c == '{') return true;
    return c >= 0xc0;
}

fn skipSpace(text: []const u8, from: usize) usize {
    var i = from;
    while (i < text.len and isSpace(text[i])) i += 1;
    return i;
}

fn trimEnd(text: []const u8, start: usize, end: usize) usize {
    var e = end;
    while (e > start and isSpace(text[e - 1])) e -= 1;
    return e;
}

fn allSpace(text: []const u8) bool {
    for (text) |c| {
        if (!isSpace(c)) return false;
    }
    return true;
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

fn parts(text: []const u8, prefixes: Prefixes, buf: []Span) ![]const []const u8 {
    const S = struct {
        var out: [32][]const u8 = undefined;
    };
    const n = split(text, prefixes, buf);
    for (0..n) |i| S.out[i] = text[buf[i].start..buf[i].end];
    return S.out[0..n];
}

fn expectSplit(text: []const u8, want: []const []const u8) !void {
    var buf: [32]Span = undefined;
    const got = try parts(text, .{}, &buf);
    try testing.expectEqual(want.len, got.len);
    for (want, got) |w, g| try testing.expectEqualStrings(w, g);
}

test "plain sentences" {
    try expectSplit("One. Two. Three.", &.{ "One.", "Two.", "Three." });
    try expectSplit("Hello!  How are you?  Fine.", &.{ "Hello!", "How are you?", "Fine." });
    try expectSplit("Only one", &.{"Only one"});
    try expectSplit("Trailing space. ", &.{"Trailing space."});
}

test "a terminator inside a token is not a boundary" {
    try expectSplit("Visit example.com now.", &.{"Visit example.com now."});
    try expectSplit("Pi is 3.14 exactly.", &.{"Pi is 3.14 exactly."});
    try expectSplit("It cost $1.50 total.", &.{"It cost $1.50 total."});
}

test "lowercase after a period does not start a sentence" {
    try expectSplit("a. b. c.", &.{"a. b. c."});
    try expectSplit("Version 2. beta was skipped.", &.{"Version 2. beta was skipped."});
}

test "a digit may start a sentence, which is why prefixes carry numbers" {
    // Structurally this is a boundary, and Moses agrees unless "fig" is in the
    // list as a numeric-only prefix. Without a list, splitting is honest.
    try expectSplit("See fig. 3 for details.", &.{ "See fig.", "3 for details." });

    var buf: [8]Span = undefined;
    const got = try parts("See fig. 3 for details.", .{ .blob = "fig\x00" }, &buf);
    try testing.expectEqual(@as(usize, 1), got.len);
}

test "ellipsis is one terminator" {
    try expectSplit("Wait... What?", &.{ "Wait...", "What?" });
}

test "closing punctuation stays with its sentence" {
    try expectSplit("\"Stop.\" He left.", &.{ "\"Stop.\"", "He left." });
    try expectSplit("(Done.) Next.", &.{ "(Done.)", "Next." });
}

test "initials do not end a sentence" {
    try expectSplit("J. R. R. Tolkien wrote it.", &.{"J. R. R. Tolkien wrote it."});
}

test "non-breaking prefixes" {
    var buf: [32]Span = undefined;
    const prefixes: Prefixes = .{ .blob = "dr\x00mr\x00nr\x00" };
    const got = try parts("Dr. Ehud Ur spoke. He left.", prefixes, &buf);
    try testing.expectEqual(@as(usize, 2), got.len);
    try testing.expectEqualStrings("Dr. Ehud Ur spoke.", got[0]);

    // Without the list, the same text splits at the title.
    const naive = try parts("Dr. Ehud Ur spoke. He left.", .{}, &buf);
    try testing.expectEqual(@as(usize, 3), naive.len);
}

test "non-ASCII may begin a sentence" {
    try expectSplit("Hola. ¿Qué tal?", &.{ "Hola.", "¿Qué tal?" });
    try expectSplit("Ende. Über alles.", &.{ "Ende.", "Über alles." });
}

test "arbitrary bytes never trap and always cover the input" {
    var buf: [64]Span = undefined;
    var rng = std.Random.DefaultPrng.init(0x5b17);
    var text: [256]u8 = undefined;

    for (0..3000) |_| {
        const n = rng.random().intRangeAtMost(usize, 0, text.len);
        rng.random().bytes(text[0..n]);
        const count = split(text[0..n], .{}, &buf);
        try testing.expect(count <= buf.len);
        var prev: u32 = 0;
        for (buf[0..count]) |s| {
            try testing.expect(s.start >= prev);
            try testing.expect(s.end <= n);
            try testing.expect(s.end > s.start);
            prev = s.end;
        }
    }
}

test "more sentences than the output array keeps the remainder as the tail" {
    const text = "One. Two. Three. Four. Five.";
    var buf: [2]Span = undefined;
    const n = split(text, .{}, &buf);
    try testing.expectEqual(@as(u32, 2), n);
    try testing.expectEqualStrings("One.", text[buf[0].start..buf[0].end]);
    // Nothing is dropped: the last span runs to the end of the input.
    try testing.expectEqual(@as(u32, text.len), buf[1].end);
}
