//! translate.zig — one line in, one line out, through the real runtime.
//!
//!   translate --model es-en.fzm [--model en-de.fzm] --src es --tgt de < in > out
//!
//! Exists so `tools/eval/run.py` can score fizh without linking against it, and
//! so a human can look at a translation without opening a browser. Load order
//! is load order: pass two `--model` flags and SPEC §10's pivot resolution finds
//! the path by itself.

const std = @import("std");
const Io = std.Io;

const fizh = @import("fizh");
const abi = fizh.abi;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();

    var errbuf: [4096]u8 = undefined;
    var errfile: Io.File.Writer = .init(.stderr(), io, &errbuf);
    const err = &errfile.interface;
    defer err.flush() catch {};

    var models: [abi.limits.models][]const u8 = undefined;
    var model_count: u32 = 0;
    var src_lang: []const u8 = "es";
    var tgt_lang: []const u8 = "en";

    const args = try init.minimal.args.toSlice(arena);
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--model")) {
            if (model_count == models.len) return error.TooManyModels;
            models[model_count] = args[i + 1];
            model_count += 1;
        } else if (std.mem.eql(u8, args[i], "--src")) {
            src_lang = args[i + 1];
        } else if (std.mem.eql(u8, args[i], "--tgt")) {
            tgt_lang = args[i + 1];
        }
    }
    if (model_count == 0) {
        try err.print("usage: translate --model a.fzm [--model b.fzm] --src es --tgt en\n", .{});
        return error.BadUsage;
    }
    if (src_lang.len != 2 or tgt_lang.len != 2) return error.BadLang;

    var cfg = defaultConfig();
    cfg.max_models = model_count;
    const cfg_bytes = cfg.bytes();
    const arena_bytes = fizh.arenaBytes(&cfg_bytes);
    if (arena_bytes == 0) return error.BadConfig;

    const memory = try gpa.alignedAlloc(u8, .fromByteUnits(64), arena_bytes);
    defer gpa.free(memory);

    const handle = fizh.init(memory.ptr, arena_bytes, &cfg_bytes);
    if (handle <= 0) {
        try err.print("init: {s}\n", .{abi.statusStr(handle)});
        return error.InitFailed;
    }

    for (models[0..model_count], 0..) |path, slot| {
        const blob = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(128 << 20));
        defer gpa.free(blob);
        const status = fizh.modelLoad(handle, @intCast(slot), blob);
        if (status != 0) {
            try err.print("{s}: {s}\n", .{ path, abi.statusStr(status) });
            return error.LoadFailed;
        }
    }

    const src = abi.langFrom(src_lang[0..2]);
    const tgt = abi.langFrom(tgt_lang[0..2]);
    const route = fizh.canTranslate(handle, src, tgt);
    try err.print("{s}->{s}: route {d} (0 none, 1 direct, 2 pivot)\n", .{ src_lang, tgt_lang, route });
    if (route <= 0) return error.NoRoute;

    var inbuf: [8192]u8 = undefined;
    var outbuf: [8192]u8 = undefined;
    var stdin_file: Io.File.Reader = .init(.stdin(), io, &inbuf);
    var stdout_file: Io.File.Writer = .init(.stdout(), io, &outbuf);
    const stdin = &stdin_file.interface;
    const stdout = &stdout_file.interface;

    const result = try gpa.alloc(u8, cfg.max_src_bytes * 4);
    defer gpa.free(result);

    // `takeDelimiter` consumes the newline and returns null at end of stream,
    // which is the only one of the four `takeDelimiter*` variants that does
    // both. The exclusive form leaves the delimiter in the buffer and will
    // hand back the same line forever.
    while (try stdin.takeDelimiter('\n')) |line| {
        const n = fizh.translate(handle, line, src, tgt, result);
        if (n < 0) {
            try stdout.print("\n", .{});
            try err.print("line rejected: {s}\n", .{abi.statusStr(n)});
            continue;
        }
        try stdout.print("{s}\n", .{result[0..@intCast(n)]});
    }
    try stdout.flush();
}

fn defaultConfig() abi.Config {
    return .{
        .abi_version = abi.abi_version,
        .max_models = 1,
        .max_model_bytes = 22 << 20,
        .max_src_bytes = 4096,
        .max_src_tokens = 256,
        .max_tgt_tokens = 384,
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
