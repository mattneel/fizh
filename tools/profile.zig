//! profile.zig — where a single translation call spends its time.
//!
//!   zig build profile -Dprofile -- zig-out/esen.fzm
//!
//! The desktop three-engine run put fizh at 38.1 ms of fixed cost per call
//! against bergamot's 0.27 ms, separated from the per-token cost by a linear
//! fit across the 12-word and 120-word cases. 140x on a constant is not a
//! kernel problem, and no amount of kernel work would have found it.
//!
//! This reports the constant. It runs the same two message lengths the fit
//! used, so the per-phase split can be read the same way: whatever does not
//! grow between them is the fixed cost.

const std = @import("std");
const Io = std.Io;

const fizh = @import("fizh");
const abi = fizh.abi;
const profile = fizh.profile;

const clauses = [_][]const u8{
    "el gato negro duerme en la mesa de la cocina",
    "no puedo ir contigo porque tengo que trabajar",
    "me gusta mucho la comida que preparaste ayer",
    "vamos a la playa si hace buen tiempo el domingo",
    "ella dijo que llegaria tarde a la reunion de hoy",
};

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

/// The clock fizh's graph does not own. Module state because a Zig function
/// pointer cannot capture, and the graph takes a plain `fn () u64`.
var g_io: Io = undefined;
fn nanos() u64 {
    const t = Io.Timestamp.now(g_io, .awake);
    return @intCast(@max(0, t.nanoseconds));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    g_io = io;
    profile.clock = &nanos;
    const arena = init.arena.allocator();

    var buf: [8192]u8 = undefined;
    var file: Io.File.Writer = .init(.stderr(), io, &buf);
    const out = &file.interface;
    defer out.flush() catch {};

    if (!profile.enabled) {
        try out.print("  built without -Dprofile; nothing to report\n", .{});
        return error.NotInstrumented;
    }

    const args = try init.minimal.args.toSlice(arena);
    const path = if (args.len > 1) args[1] else "zig-out/esen.fzm";
    const blob = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(128 << 20));
    defer gpa.free(blob);

    const hp = fizh.format.peekHParams(blob) orelse return error.BadArtifact;
    const slot = std.mem.alignForward(u32, @intCast(blob.len), 64);
    const cfg: abi.Config = .{
        .abi_version = abi.abi_version,
        .max_models = 1,
        .max_model_bytes = slot +| (slot / 4),
        .max_src_bytes = 4096,
        .max_src_tokens = @min(256, hp.max_pos),
        .max_tgt_tokens = @min(768, hp.max_pos),
        .max_shortlist = 2048,
        .max_d_model = hp.d_model,
        .max_ffn_dim = hp.ffn_dim,
        .max_enc_layers = hp.n_enc_layers,
        .max_dec_layers = hp.n_dec_layers,
        .max_heads = hp.n_heads,
        .max_vocab = hp.vocab_size,
        .reserved = .{ 0, 0, 0 },
    };
    const cfg_bytes = cfg.bytes();
    const arena_bytes = fizh.arenaBytes(&cfg_bytes);
    if (arena_bytes == 0) return error.BadConfig;

    const memory = try gpa.alignedAlloc(u8, .fromByteUnits(64), arena_bytes);
    defer gpa.free(memory);
    const handle = fizh.init(memory.ptr, arena_bytes, &cfg_bytes);
    if (handle <= 0) return error.InitFailed;
    if (fizh.modelLoad(handle, 0, blob) != 0) return error.LoadFailed;

    const es = abi.langFrom("es");
    const en = abi.lang_en;
    const out_buf = try gpa.alloc(u8, 8192);
    defer gpa.free(out_buf);

    try out.print("{s}\n  arena {d:.1} MiB, {s} backend\n\n", .{
        path, @as(f64, @floatFromInt(arena_bytes)) / (1 << 20), fizh.kernel.active.name,
    });

    for ([_]u32{ 6, 12, 24, 48, 96, 120 }) |words| {
        const src = try message(arena, words);
        // Warm first: the fixed cost being measured is the steady-state one,
        // not first-touch page faults.
        for (0..5) |_| _ = fizh.translate(handle, src, es, en, out_buf);

        profile.reset();
        const runs: u32 = if (words <= 24) 120 else 20;
        for (0..runs) |_| {
            const n = fizh.translate(handle, src, es, en, out_buf);
            if (n < 0) return error.TranslateFailed;
        }
        var sum: u64 = 0;
        for (profile.total) |v| sum += v;
        const per_call = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(runs)) / 1e6;
        const enc = @as(f64, @floatFromInt(profile.total[@intFromEnum(profile.Phase.encoder)])) /
            @as(f64, @floatFromInt(runs)) / 1e6;
        const dec = @as(f64, @floatFromInt(profile.total[@intFromEnum(profile.Phase.decoder)])) /
            @as(f64, @floatFromInt(runs)) / 1e6;
        const setup = per_call - enc - dec;
        try out.print("  {d:>4} words, {d:>4} src tokens: total {d:>8.3}  enc {d:>8.3}  dec {d:>8.3}  setup {d:>6.3} ms\n",
            .{ words, fizh.lastPass().src_len, per_call, enc, dec, setup });
    }
}
