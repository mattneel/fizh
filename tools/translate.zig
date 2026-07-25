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

    // Read every artifact first: the arena is sized from what they actually
    // are, not from a guess that a 384-wide model then fails to fit.
    var blobs: [8][]u8 = undefined;
    var loaded: u32 = 0;
    defer for (blobs[0..loaded]) |b| gpa.free(b);
    for (models[0..model_count]) |path| {
        blobs[loaded] = try Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(128 << 20));
        loaded += 1;
    }

    var cfg = configFor(blobs[0]) orelse {
        try err.print("{s}: not a version {d} .fzm\n", .{ models[0], fizh.format.version });
        return error.BadArtifact;
    };
    for (blobs[1..loaded]) |b| {
        const other = configFor(b) orelse return error.BadArtifact;
        cfg.max_model_bytes = @max(cfg.max_model_bytes, other.max_model_bytes);
        cfg.max_d_model = @max(cfg.max_d_model, other.max_d_model);
        cfg.max_ffn_dim = @max(cfg.max_ffn_dim, other.max_ffn_dim);
        cfg.max_enc_layers = @max(cfg.max_enc_layers, other.max_enc_layers);
        cfg.max_dec_layers = @max(cfg.max_dec_layers, other.max_dec_layers);
        cfg.max_heads = @max(cfg.max_heads, other.max_heads);
        cfg.max_vocab = @max(cfg.max_vocab, other.max_vocab);
    }
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

    for (blobs[0..loaded], 0..) |blob, slot| {
        const status = fizh.modelLoad(handle, @intCast(slot), blob);
        if (status != 0) {
            try err.print("{s}: {s}\n", .{ models[slot], abi.statusStr(status) });
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

/// Sized from the artifact rather than guessed at.
///
/// Firefox's registry ships more than one student architecture: most pairs are
/// SPEC §4.3's `d_model=256, n_dec=2`, but ar, eu and gl are `384, 4`. A
/// hardcoded config rejected those with `model_too_large` even though every
/// kernel handles them — the limit was the tool's, not the runtime's. A host
/// embedding fizh picks its own ceilings; a command-line tool pointed at one
/// file should fit that file.
fn configFor(blob: []const u8) ?abi.Config {
    const hp = fizh.format.peekHParams(blob) orelse return null;
    const slot = std.mem.alignForward(u32, @intCast(blob.len), 64);
    return .{
        .abi_version = abi.abi_version,
        .max_models = 1,
        // Repack can grow a slot past the file; a quarter is ample headroom.
        .max_model_bytes = slot +| (slot / 4),
        .max_src_bytes = 4096,
        // Positional encodings only exist up to the artifact's `max_pos`, and
        // the loader rejects a config that asks for more. Clamping here rather
        // than failing means an artifact converted before SPEC §4.3 raised
        // `max_tgt_tokens` still loads, with its own shorter bound.
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
}
