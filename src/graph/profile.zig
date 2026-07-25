//! profile.zig — per-phase timing for one translation pass.
//!
//! A benchmark that reports one number per call cannot tell a kernel problem
//! from a setup problem. The desktop run showed fizh at 38.1 ms of fixed cost
//! per call against bergamot's 0.27 — a 140x gap that a linear fit separates
//! cleanly from the 3x per-token gap, and which no amount of kernel work would
//! have touched.
//!
//! Off by default and compiled out entirely: `builtin.mode` aside, the whole
//! module folds to nothing when `enabled` is false, so the shipped wasm carries
//! neither the timers nor a clock import it could not satisfy anyway.
//!
//!     zig build profile -Dprofile
//!
//! Native only. `wasm32-freestanding` has no monotonic clock and I2 forbids
//! importing one.

const std = @import("std");
const builtin = @import("builtin");

pub const enabled = @import("build_options").profile and !builtin.cpu.arch.isWasm();

pub const Phase = enum {
    normalize,
    tokenize,
    shortlist,
    encoder,
    decoder,
    detokenize,

    pub const count = @typeInfo(Phase).@"enum".fields.len;
};

/// Nanoseconds per phase, accumulated across calls, plus a call counter. Module
/// state rather than a parameter so the timers do not change any signature —
/// an instrumented build and a shipped build must run the same code path.
pub var total: [Phase.count]u64 = @splat(0);
pub var calls: u64 = 0;

/// Monotonic nanoseconds, supplied by the host.
///
/// The graph has no `io` handle and is not getting one: I2 keeps the runtime
/// free of ambient IO, and a clock is ambient IO. So the *tool* injects one and
/// the runtime stays exactly as dependency-free as it was. Null means the
/// timers are compiled in but nobody asked for numbers, which costs a
/// predictable branch and nothing else.
pub var clock: ?*const fn () u64 = null;

pub inline fn start() u64 {
    if (!enabled) return 0;
    const c = clock orelse return 0;
    return c();
}

pub inline fn stop(phase: Phase, since: u64) void {
    if (!enabled) return;
    const c = clock orelse return;
    total[@intFromEnum(phase)] += c() -| since;
}

pub fn reset() void {
    if (!enabled) return;
    total = @splat(0);
    calls = 0;
}

pub fn report(out: anytype) !void {
    if (!enabled) {
        try out.print("  profiling not compiled in: build with -Dprofile\n", .{});
        return;
    }
    var sum: u64 = 0;
    for (total) |v| sum += v;
    if (calls == 0 or sum == 0) {
        try out.print("  no calls recorded\n", .{});
        return;
    }
    const per = @as(f64, @floatFromInt(calls));
    try out.print("  {s:<12} {s:>10} {s:>8}\n", .{ "phase", "ms/call", "share" });
    try out.print("  {s:-<12} {s:->10} {s:->8}\n", .{ "", "", "" });
    inline for (@typeInfo(Phase).@"enum".fields) |f| {
        const v = total[f.value];
        try out.print("  {s:<12} {d:>10.3} {d:>7.1}%\n", .{
            f.name,
            @as(f64, @floatFromInt(v)) / per / 1_000_000.0,
            100.0 * @as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(sum)),
        });
    }
    try out.print("  {s:<12} {d:>10.3}\n", .{
        "total", @as(f64, @floatFromInt(sum)) / per / 1_000_000.0,
    });
}
