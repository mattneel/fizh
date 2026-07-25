//! bench.zig — SPEC §13 T5 and the SPEC §14 budgets.
//!
//!   zig build bench                 (builds a §4.3-scale synthetic artifact)
//!   zig build bench -- model.fzm --enforce
//!
//! §14 is explicit that the reference device is a 2022-class mid-tier Android
//! and that "CI numbers are for trend detection only". So this prints the
//! budget next to the measurement and does not fail unless asked: a desktop
//! passing every budget proves nothing, and a desktop failing one is worth
//! looking at immediately.
//!
//! p50 and p99 are reported separately and never averaged (§14).

const std = @import("std");
const Io = std.Io;

const fizh = @import("fizh");
const abi = fizh.abi;

/// SPEC §14, which is a **mobile** budget measured on the pinned Android.
///
/// These were briefly retightened to ~1.5x *desktop* measurements, which was a
/// mistake with a cost: the paragraph row then failed on the phone at 287.7 ms
/// against 100, a false failure produced by the budget rather than the runtime.
/// A desktop is not a basis for a mobile budget.
///
/// So this binary compares desktop measurements against mobile budgets and will
/// pass every one by a wide margin. That is expected and it is not a result —
/// see the banner it prints. The number this step is for is the *delta between
/// commits*, not the distance to the budget.
const Budget = struct {
    cold_start_ms: f64 = 300,
    warm_p50_direct_ms: f64 = 80,
    warm_p50_pivot_ms: f64 = 160,
    warm_p99_long_ms: f64 = 900,
    /// Eight sentences, fixed. Without a declared sentence count the row could
    /// be neither passed nor failed (work order 8 P1).
    warm_p50_paragraph_ms: f64 = 400,
    scratch_bytes: u64 = 12 << 20,
    weights_bytes: u64 = 64 << 20,
};

const iterations: u32 = 60;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();

    var buf: [8192]u8 = undefined;
    var file: Io.File.Writer = .init(.stderr(), io, &buf);
    const out = &file.interface;
    defer out.flush() catch {};

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) {
        try out.print("usage: bench <model.fzm> [--enforce]\n", .{});
        return error.BadUsage;
    }
    var enforce = false;
    for (args[2..]) |a| {
        if (std.mem.eql(u8, a, "--enforce")) enforce = true;
    }

    const blob = try Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .limited(128 << 20));
    defer gpa.free(blob);

    const cfg = bergamotConfig();
    const cfg_bytes = cfg.bytes();
    const arena_bytes = fizh.arenaBytes(&cfg_bytes);
    if (arena_bytes == 0) return error.BadConfig;

    const memory = try gpa.alignedAlloc(u8, .fromByteUnits(64), arena_bytes);
    defer gpa.free(memory);

    // -- cold start: instantiate + load + repack, one direction -------------
    const cold_start = Io.Timestamp.now(io, .awake);
    const handle = fizh.init(memory.ptr, arena_bytes, &cfg_bytes);
    if (handle <= 0) {
        try out.print("init failed: {s}\n", .{abi.statusStr(handle)});
        return error.InitFailed;
    }
    const status = fizh.modelLoad(handle, 0, blob);
    const cold_ns = elapsed(cold_start, Io.Timestamp.now(io, .awake));
    if (status != 0) {
        try out.print("load failed: {s}\n", .{abi.statusStr(status)});
        return error.LoadFailed;
    }

    const inst = fizh.instanceForTest().?;
    const weights = inst.models[0].slot_used;
    const scratch = arena_bytes - cfg.max_models * std.mem.alignForward(u32, cfg.max_model_bytes, 64);

    // -- warm timings -------------------------------------------------------
    const short = try message(arena, 12);
    const long = try message(arena, 120);
    const para = try paragraph(arena, 8);
    const out_buf = try gpa.alloc(u8, cfg.max_src_bytes * fizh.arena.io_expansion);
    defer gpa.free(out_buf);

    const short_ns = try timeMany(gpa, io, handle, short, out_buf);
    defer gpa.free(short_ns);
    const long_ns = try timeMany(gpa, io, handle, long, out_buf);
    defer gpa.free(long_ns);
    const para_ns = try timeMany(gpa, io, handle, para, out_buf);
    defer gpa.free(para_ns);

    const b: Budget = .{};
    var failed: u32 = 0;

    try out.print("{s}\n  {s} backend, {s} build\n\n", .{
        args[1],
        fizh.kernel.active.name,
        @tagName(@import("builtin").mode),
    });
    try out.print("  SPEC §14 budgets are MOBILE, measured on the pinned Android.\n", .{});
    try out.print("  These are DESKTOP measurements against them, so passing is\n", .{});
    try out.print("  expected and means nothing. Use the delta between commits.\n", .{});
    try out.print("  The real numbers come from the Pages harness (ADR 0021).\n\n", .{});
    try out.print("  {s:<34} {s:>12} {s:>12}\n", .{ "metric", "measured", "budget" });
    try out.print("  {s:-<34} {s:->12} {s:->12}\n", .{ "", "", "" });

    failed += try row(out, "cold start (init+load+repack)", ms(cold_ns), b.cold_start_ms, "ms");
    failed += try row(out, "warm p50, 12-token, direct", ms(pct(short_ns, 50)), b.warm_p50_direct_ms, "ms");
    failed += try row(out, "warm p99, 12-token, direct", ms(pct(short_ns, 99)), b.warm_p99_long_ms, "ms");
    failed += try row(out, "warm p50, 120-token, direct", ms(pct(long_ns, 50)), b.warm_p99_long_ms, "ms");
    failed += try row(out, "warm p99, 120-token, direct", ms(pct(long_ns, 99)), b.warm_p99_long_ms, "ms");
    failed += try row(out, "warm p50, 8-sentence paragraph", ms(pct(para_ns, 50)), b.warm_p50_paragraph_ms, "ms");
    failed += try row(out, "shared scratch", mb(scratch), mb(b.scratch_bytes), "MB");
    failed += try row(out, "weights, per direction", mb(weights), mb(b.weights_bytes), "MB");

    try out.print(
        \\
        \\  A pivot is two sequential passes (SPEC §10), so its budget is met
        \\  when each hop meets half of {d:.0} ms. Measure it directly with two
        \\  real directions loaded.
        \\
    , .{b.warm_p50_pivot_ms});

    if (failed != 0) {
        try out.print("\n  {d} budget(s) exceeded on this machine.\n", .{failed});
        try out.print("  SPEC §14: the reference device is a pinned 2022-class mid-tier\n", .{});
        try out.print("  Android; these numbers are for trend detection only.\n", .{});
        if (enforce) return error.BudgetExceeded;
    }
}

fn row(out: *Io.Writer, name: []const u8, got: f64, budget: f64, unit: []const u8) !u32 {
    const over = got > budget;
    try out.print("  {s:<34} {d:>9.2} {s} {d:>9.0} {s}{s}\n", .{
        name, got, unit, budget, unit, if (over) "  OVER" else "",
    });
    return if (over) 1 else 0;
}

fn timeMany(gpa: std.mem.Allocator, io: Io, handle: i32, src: []const u8, out_buf: []u8) ![]u64 {
    const es = abi.langFrom("es");
    const en = abi.lang_en;

    // One untimed pass so the first measurement is not the one that faults in
    // twenty megabytes of weights.
    _ = fizh.translate(handle, src, es, en, out_buf);

    const samples = try gpa.alloc(u64, iterations);
    for (samples) |*s| {
        const t0 = Io.Timestamp.now(io, .awake);
        const n = fizh.translate(handle, src, es, en, out_buf);
        const t1 = Io.Timestamp.now(io, .awake);
        if (n < 0) {
            std.debug.print("bench: translate returned {s}\n", .{abi.statusStr(n)});
            return error.TranslateFailed;
        }
        s.* = elapsed(t0, t1);
    }
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return samples;
}

/// Percentiles, never a mean. SPEC §14.
fn pct(sorted: []const u64, p: u32) u64 {
    std.debug.assert(sorted.len > 0);
    const at = (sorted.len * p) / 100;
    return sorted[@min(at, sorted.len - 1)];
}

fn elapsed(a: Io.Timestamp, b: Io.Timestamp) u64 {
    const d = b.nanoseconds - a.nanoseconds;
    return if (d < 0) 0 else @intCast(d);
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn mb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

/// Real Spanish, because a message of invented words tokenizes into one piece
/// per letter and would time a source four times longer than it claims.
/// SPEC §14 sizes its budgets in tokens, so these are grown by clause.
const clauses = [_][]const u8{
    "el gato negro duerme en la mesa de la cocina",
    "no puedo ir contigo porque tengo que trabajar",
    "me gusta mucho la comida que preparaste ayer",
    "vamos a la playa si hace buen tiempo el domingo",
    "ella dijo que llegaria tarde a la reunion de hoy",
};

/// Multi-sentence input, which the pre-segmentation budgets never measured.
fn paragraph(arena: std.mem.Allocator, sentences: u32) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    for (0..sentences) |i| {
        if (i != 0) try out.append(arena, ' ');
        try out.appendSlice(arena, clauses[i % clauses.len]);
        try out.append(arena, '.');
    }
    return out.toOwnedSlice(arena);
}

fn message(arena: std.mem.Allocator, words: u32) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var written: u32 = 0;
    var i: usize = 0;
    while (written < words) : (i += 1) {
        const c = clauses[i % clauses.len];
        if (written != 0) try out.appendSlice(arena, ", ");
        try out.appendSlice(arena, c);
        written += @intCast(std.mem.count(u8, c, " ") + 1);
    }
    return out.toOwnedSlice(arena);
}

/// SPEC §4.3's worked example.
fn bergamotConfig() abi.Config {
    return .{
        .abi_version = abi.abi_version,
        .max_models = 1,
        .max_model_bytes = 24 << 20,
        .max_src_bytes = 4096,
        .max_src_tokens = 256,
        .max_tgt_tokens = 768,
        .max_shortlist = 2048,
        .max_d_model = 256,
        .max_ffn_dim = 1536,
        .max_enc_layers = 6,
        .max_dec_layers = 2,
        .max_heads = 8,
        .max_vocab = 32768,
        .reserved = .{ 0, 0, 0 },
    };
}
