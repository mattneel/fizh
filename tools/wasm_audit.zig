//! wasm_audit.zig — SPEC §3 budgets, enforced against the real artifact.
//!
//! Parses the module's section table (no external toolchain), counts imports
//! and exports, gzips the bytes, and fails the build if any budget is blown.
//!
//!   zig build check

const std = @import("std");
const Io = std.Io;

const section_import: u8 = 2;
const section_export: u8 = 7;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();

    var out_buf: [8192]u8 = undefined;
    var out_file: Io.File.Writer = .init(.stderr(), io, &out_buf);
    const out = &out_file.interface;

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        try out.print("usage: wasm-audit <module.wasm> [--max-gzip=N] [--max-exports=N] [--max-imports=N]\n", .{});
        try out.flush();
        return error.BadUsage;
    }

    var max_gzip: u32 = std.math.maxInt(u32);
    var max_exports: u32 = std.math.maxInt(u32);
    var max_imports: u32 = std.math.maxInt(u32);
    var require_relaxed = false;
    for (args[2..]) |a| {
        if (try flag(a, "--max-gzip=")) |v| max_gzip = v;
        if (try flag(a, "--max-exports=")) |v| max_exports = v;
        if (try flag(a, "--max-imports=")) |v| max_imports = v;
        if (std.mem.eql(u8, a, "--require-relaxed")) require_relaxed = true;
    }

    const path = args[1];
    const bytes = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(bytes);

    const report = try audit(gpa, bytes, out);
    const gz = try gzippedLen(gpa, bytes);

    try out.print(
        \\{s}
        \\  raw          {d} bytes
        \\  gzipped      {d} bytes   (budget {d})
        \\  imports      {d}         (budget {d})
        \\  exports      {d}         (budget {d})
        \\
    , .{ path, bytes.len, gz, max_gzip, report.imports, max_imports, report.exports, max_exports });

    var failed = false;
    if (gz > max_gzip) {
        try out.print("  FAIL SPEC §3: gzipped size over budget\n", .{});
        failed = true;
    }
    if (report.imports > max_imports) {
        try out.print("  FAIL SPEC §3: module imports over budget (adding one needs an ADR)\n", .{});
        failed = true;
    }
    if (report.exports > max_exports) {
        try out.print("  FAIL SPEC §3: exported symbols over budget\n", .{});
        failed = true;
    }
    if (require_relaxed) {
        const found = hasRelaxedMadd(report.code);
        try out.print("  relaxed_madd {s}\n", .{if (found) "present" else "ABSENT"});
        if (!found) {
            try out.print(
                "  FAIL: a probe module without a relaxed instruction validates " ++
                    "everywhere, and would route every device to the relaxed build\n",
                .{},
            );
            failed = true;
        }
    }
    try out.flush();
    if (failed) return error.BudgetExceeded;
}

const section_code: u8 = 10;

const Report = struct {
    imports: u32 = 0,
    exports: u32 = 0,
    code: []const u8 = &.{},
};

/// `f32x4.relaxed_madd` is the two-byte prefixed opcode `0xFD 0x105`, which
/// LEB128-encodes as `FD 85 02`. Searching the code section for that byte
/// sequence is a heuristic — an immediate could spell it by accident — but it
/// is a *positive* check: a false negative would fail the build, and a false
/// positive needs the bytes to appear in a module that was compiled for
/// relaxed SIMD in the first place.
fn hasRelaxedMadd(code: []const u8) bool {
    return std.mem.indexOf(u8, code, &[_]u8{ 0xFD, 0x85, 0x02 }) != null;
}

fn audit(gpa: std.mem.Allocator, bytes: []const u8, out: *Io.Writer) !Report {
    _ = gpa;
    var r: Report = .{};
    var c: Cursor = .{ .bytes = bytes };

    if (!std.mem.eql(u8, try c.take(4), "\x00asm")) return error.NotWasm;
    _ = try c.take(4); // version

    while (c.pos < bytes.len) {
        const id = try c.byte();
        const size = try c.uleb();
        const payload = try c.take(size);
        switch (id) {
            section_import => r.imports = try countAndListImports(payload, out),
            section_export => r.exports = try countAndListExports(payload, out),
            section_code => r.code = payload,
            else => {},
        }
    }
    return r;
}

fn countAndListImports(payload: []const u8, out: *Io.Writer) !u32 {
    var c: Cursor = .{ .bytes = payload };
    const n = try c.uleb();
    for (0..n) |_| {
        const module = try c.name();
        const field = try c.name();
        try out.print("  import       {s}.{s}\n", .{ module, field });
        _ = try c.byte();
        _ = try c.uleb();
    }
    return n;
}

fn countAndListExports(payload: []const u8, out: *Io.Writer) !u32 {
    var c: Cursor = .{ .bytes = payload };
    const n = try c.uleb();
    for (0..n) |_| {
        const name = try c.name();
        const kind = try c.byte();
        _ = try c.uleb();
        try out.print("  export       {s} ({s})\n", .{ name, kindName(kind) });
    }
    return n;
}

fn kindName(kind: u8) []const u8 {
    return switch (kind) {
        0 => "func",
        1 => "table",
        2 => "memory",
        3 => "global",
        else => "?",
    };
}

const Cursor = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn byte(self: *Cursor) !u8 {
        if (self.pos >= self.bytes.len) return error.Truncated;
        defer self.pos += 1;
        return self.bytes[self.pos];
    }

    fn take(self: *Cursor, n: usize) ![]const u8 {
        if (self.pos + n > self.bytes.len) return error.Truncated;
        defer self.pos += n;
        return self.bytes[self.pos..][0..n];
    }

    fn uleb(self: *Cursor) !u32 {
        var result: u32 = 0;
        var shift: u5 = 0;
        for (0..5) |_| {
            const b = try self.byte();
            result |= @as(u32, b & 0x7f) << shift;
            if (b & 0x80 == 0) return result;
            shift += 7;
        }
        return error.BadLeb;
    }

    fn name(self: *Cursor) ![]const u8 {
        const n = try self.uleb();
        return self.take(n);
    }
};

fn gzippedLen(gpa: std.mem.Allocator, bytes: []const u8) !usize {
    const flate = std.compress.flate;
    const sink = try gpa.alloc(u8, bytes.len + (bytes.len / 2) + 4096);
    defer gpa.free(sink);
    const window = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(window);

    var w = Io.Writer.fixed(sink);
    var c = try flate.Compress.init(&w, window, .gzip, .level_9);
    try c.writer.writeAll(bytes);
    try c.finish();
    return w.end;
}

fn flag(arg: []const u8, prefix: []const u8) !?u32 {
    if (!std.mem.startsWith(u8, arg, prefix)) return null;
    return try std.fmt.parseInt(u32, arg[prefix.len..], 10);
}

test "cursor rejects truncated input instead of reading past the end" {
    var c: Cursor = .{ .bytes = &[_]u8{ 0x80, 0x80 } };
    try std.testing.expectError(error.Truncated, c.uleb());
}

test "uleb decodes multi-byte values" {
    var c: Cursor = .{ .bytes = &[_]u8{ 0xE5, 0x8E, 0x26 } };
    try std.testing.expectEqual(@as(u32, 624485), try c.uleb());
}
