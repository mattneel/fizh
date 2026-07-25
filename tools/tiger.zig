//! tiger.zig — the mechanically checkable half of SPEC §12, run over `src/`.
//!
//! This is a lint, not a proof. It blanks comments and string literals first so
//! that brace matching and keyword searches see code and only code. What it
//! cannot see, it does not claim: mutual recursion and "two assertions per
//! function" are left to review.
//!
//!   zig build tiger

const std = @import("std");
const Io = std.Io;

const max_function_lines: u32 = 70; // SPEC §12.8

const Rule = struct {
    id: []const u8,
    needle: []const u8,
    why: []const u8,
    /// Only applies to paths containing this fragment; empty means everywhere.
    scope: []const u8 = "",
};

const line_rules = [_]Rule{
    .{ .id = "12.1", .needle = "std.mem.Allocator", .why = "no allocator in src/" },
    .{ .id = "12.1", .needle = "std.heap.", .why = "no allocator in src/" },
    .{ .id = "12.1", .needle = "testing.allocator", .why = "no allocator in src/, not even in tests" },
    .{ .id = "12.3", .needle = "while (true)", .why = "all loops bounded" },
    .{ .id = "12.3", .needle = "while(true)", .why = "all loops bounded" },
    .{ .id = "3", .needle = "@setFloatMode(.optimized)", .why = "fixed reduction order is what makes I9 hold", .scope = "kernel" },
    .{ .id = "3", .needle = "@setFloatMode(.optimized)", .why = "fixed reduction order is what makes I9 hold", .scope = "graph" },
    .{ .id = "3", .needle = "@setFloatMode(.optimized)", .why = "fixed reduction order is what makes I9 hold" },
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();

    var err_buf: [4096]u8 = undefined;
    var err_file: Io.File.Writer = .init(.stderr(), io, &err_buf);
    const err = &err_file.interface;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        try err.print("usage: tiger <src-dir>\n", .{});
        return error.BadUsage;
    }

    var dir = try Io.Dir.cwd().openDir(io, args[1], .{ .iterate = true });
    defer dir.close(io);

    var walker = try dir.walk(gpa);
    defer walker.deinit();

    var violations: u32 = 0;
    var files: u32 = 0;

    while (try walker.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.basename, ".zig")) continue;

        const raw = try dir.readFileAlloc(io, entry.path, gpa, .limited(8 << 20));
        defer gpa.free(raw);
        const clean = try gpa.dupe(u8, raw);
        defer gpa.free(clean);
        blank(clean);

        files += 1;
        violations += try checkLines(err, entry.path, clean);
        violations += try checkFunctions(err, entry.path, clean);
    }

    if (violations == 0) {
        try err.print("tiger: {d} files, clean\n", .{files});
        try err.flush();
        return;
    }
    try err.print("tiger: {d} violation(s) across {d} files\n", .{ violations, files });
    try err.flush();
    return error.TigerStyleViolation;
}

fn checkLines(err: *Io.Writer, path: []const u8, clean: []const u8) !u32 {
    var violations: u32 = 0;
    var line_no: u32 = 1;
    var it = std.mem.splitScalar(u8, clean, '\n');
    while (it.next()) |line| : (line_no += 1) {
        for (line_rules) |rule| {
            if (rule.scope.len != 0 and std.mem.indexOf(u8, path, rule.scope) == null) continue;
            if (std.mem.indexOf(u8, line, rule.needle) == null) continue;
            try err.print("{s}:{d}: SPEC §{s}: '{s}' — {s}\n", .{ path, line_no, rule.id, rule.needle, rule.why });
            violations += 1;
        }
        // SPEC §12.7: usize never crosses the ABI.
        if (std.mem.indexOf(u8, line, "export fn") != null and std.mem.indexOf(u8, line, "usize") != null) {
            try err.print("{s}:{d}: SPEC §12.7: usize in an exported signature\n", .{ path, line_no });
            violations += 1;
        }
    }
    return violations;
}

/// Finds every `fn name(` and measures its body by brace matching. Reports
/// bodies over `max_function_lines` (SPEC §12.8) and direct self-calls
/// (SPEC §12.2).
fn checkFunctions(err: *Io.Writer, path: []const u8, clean: []const u8) !u32 {
    var violations: u32 = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, clean, i, "fn ")) |kw| {
        i = kw + 3;
        if (!isTokenStart(clean, kw)) continue;

        const name_start = skipSpace(clean, kw + 3);
        const name_end = identEnd(clean, name_start);
        if (name_end == name_start) continue;
        const name = clean[name_start..name_end];

        const open = std.mem.indexOfScalarPos(u8, clean, name_end, '{') orelse continue;
        const close = matchBrace(clean, open) orelse continue;

        const body = clean[open..close];
        const lines = 1 + std.mem.count(u8, body, "\n");
        if (lines > max_function_lines) {
            try err.print("{s}:{d}: SPEC §12.8: fn {s} is {d} lines (max {d})\n", .{
                path, lineOf(clean, kw), name, lines, max_function_lines,
            });
            violations += 1;
        }
        if (callsItself(body, name)) {
            try err.print("{s}:{d}: SPEC §12.2: fn {s} recurses\n", .{ path, lineOf(clean, kw), name });
            violations += 1;
        }
        i = open + 1;
    }
    return violations;
}

fn callsItself(body: []const u8, name: []const u8) bool {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, body, i, name)) |at| {
        i = at + name.len;
        if (!isTokenStart(body, at)) continue;
        // `Arena.init(...)` inside `fn init` is not recursion.
        if (at > 0 and body[at - 1] == '.') continue;
        if (i < body.len and body[i] == '(') return true;
    }
    return false;
}

/// Replaces comment and string-literal bytes with spaces, preserving length and
/// newlines so that offsets still map to source lines.
fn blank(buf: []u8) void {
    var i: usize = 0;
    while (i < buf.len) {
        if (buf[i] == '/' and i + 1 < buf.len and buf[i + 1] == '/') {
            while (i < buf.len and buf[i] != '\n') : (i += 1) buf[i] = ' ';
            continue;
        }
        if (buf[i] == '\\' and i + 1 < buf.len and buf[i + 1] == '\\') {
            while (i < buf.len and buf[i] != '\n') : (i += 1) buf[i] = ' ';
            continue;
        }
        if (buf[i] == '"' or buf[i] == '\'') {
            const quote = buf[i];
            buf[i] = ' ';
            i += 1;
            while (i < buf.len and buf[i] != quote and buf[i] != '\n') {
                const escaped = buf[i] == '\\';
                buf[i] = ' ';
                i += 1;
                if (escaped and i < buf.len) {
                    buf[i] = ' ';
                    i += 1;
                }
            }
            if (i < buf.len and buf[i] == quote) {
                buf[i] = ' ';
                i += 1;
            }
            continue;
        }
        i += 1;
    }
}

fn matchBrace(s: []const u8, open: usize) ?usize {
    var depth: u32 = 0;
    var i = open;
    while (i < s.len) : (i += 1) {
        switch (s[i]) {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if (depth == 0) return i;
            },
            else => {},
        }
    }
    return null;
}

fn isTokenStart(s: []const u8, at: usize) bool {
    if (at == 0) return true;
    return !isIdent(s[at - 1]);
}

fn isIdent(c: u8) bool {
    return c == '_' or std.ascii.isAlphanumeric(c);
}

fn skipSpace(s: []const u8, from: usize) usize {
    var i = from;
    while (i < s.len and (s[i] == ' ' or s[i] == '\t')) : (i += 1) {}
    return i;
}

fn identEnd(s: []const u8, from: usize) usize {
    var i = from;
    while (i < s.len and isIdent(s[i])) : (i += 1) {}
    return i;
}

fn lineOf(s: []const u8, at: usize) u32 {
    return 1 + @as(u32, @intCast(std.mem.count(u8, s[0..at], "\n")));
}

test "blank erases strings and comments but keeps offsets" {
    var buf = "const a = \"while (true)\"; // while (true)\nconst b = 1;".*;
    blank(&buf);
    try std.testing.expect(std.mem.indexOf(u8, &buf, "while (true)") == null);
    try std.testing.expect(std.mem.indexOf(u8, &buf, "const b = 1;") != null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, &buf, "\n"));
}

test "self-call detection needs a real call" {
    try std.testing.expect(callsItself("  return foo(1);", "foo"));
    try std.testing.expect(!callsItself("  return foobar(1);", "foo"));
    try std.testing.expect(!callsItself("  const foo = 1;", "foo"));
    try std.testing.expect(!callsItself("  return Arena.init(a, b);", "init"));
}
