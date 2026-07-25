//! trace.zig — dumps what a real pass computed, for the SPEC §13 T2 oracle.
//!
//!   zig build oracle
//!   ./zig-out/bin/trace model.fzm "hola que tal" trace.bin
//!   python3 tools/reference.py --compare trace.bin --model model.fzm
//!
//! The file format is deliberately dull — a fixed header, then three arrays —
//! because the only thing on the other side is `reference.py`, and a format
//! that needs a parser is a format that can disagree with itself.
//!
//!     "FZTR" u32 version
//!     u32 src_len, tgt_len, d_model, shortlist_len, text_len
//!     text                        (text_len bytes, padded to 4)
//!     u32 src_ids[src_len]        the tokenizer's output
//!     u32 tgt_ids[tgt_len]        greedy decode's output
//!     f32 enc_states[src_len * d_model]
//!     u32 n_stages
//!     f32 stage[n_stages][src_len * d_model]     stage 0 = embedding

const std = @import("std");
const Io = std.Io;

const fizh = @import("fizh");
const abi = fizh.abi;

const magic = "FZTR";
const version: u32 = 1;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();

    var errbuf: [4096]u8 = undefined;
    var errfile: Io.File.Writer = .init(.stderr(), io, &errbuf);
    const err = &errfile.interface;
    defer err.flush() catch {};

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 4) {
        try err.print("usage: trace <model.fzm> <text> <out.bin>\n", .{});
        return error.BadUsage;
    }
    const text = args[2];

    const blob = try Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .limited(128 << 20));
    defer gpa.free(blob);

    const cfg = benchConfig();
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
    const status = fizh.modelLoad(handle, 0, blob);
    if (status != 0) {
        try err.print("load: {s}\n", .{abi.statusStr(status)});
        return error.LoadFailed;
    }

    const inst = fizh.instanceForTest().?;
    const out_buf = try gpa.alloc(u8, 1 << 16);
    defer gpa.free(out_buf);

    // SPEC §13 T2: capture every encoder stage, not just the last one.
    Stages.gpa = gpa;
    fizh.encoder.layer_sink = Stages.observe;
    defer {
        fizh.encoder.layer_sink = null;
        for (Stages.saved[0..Stages.count]) |stage| gpa.free(stage);
    }

    const n = fizh.translate(handle, text, inst.models[0].src_lang, inst.models[0].tgt_lang, out_buf);
    if (n < 0) {
        try err.print("translate: {s}\n", .{abi.statusStr(n)});
        return error.TranslateFailed;
    }

    const last = fizh.lastPass();
    const d: u32 = inst.models[0].hp.d_model;
    const l = inst.arena.layout;
    const src_ids = inst.arena.view(u32, l.src_ids, last.src_len);
    const tgt_ids = inst.arena.view(u32, l.tgt_ids, last.tgt_len);
    const enc = inst.arena.view(f32, l.enc_states, @as(usize, last.src_len) * d);

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(gpa);
    try body.appendSlice(gpa, magic);
    try appendU32(gpa, &body, version);
    for ([_]u32{ last.src_len, last.tgt_len, d, last.shortlist_len, @intCast(text.len) }) |x| {
        try appendU32(gpa, &body, x);
    }
    try body.appendSlice(gpa, text);
    while (body.items.len % 4 != 0) try body.append(gpa, 0);
    try body.appendSlice(gpa, std.mem.sliceAsBytes(src_ids));
    try body.appendSlice(gpa, std.mem.sliceAsBytes(tgt_ids));
    try body.appendSlice(gpa, std.mem.sliceAsBytes(enc));
    try appendU32(gpa, &body, Stages.count);
    for (Stages.saved[0..Stages.count]) |stage| {
        try body.appendSlice(gpa, std.mem.sliceAsBytes(stage));
    }

    var filebuf: [4096]u8 = undefined;
    var file = try Io.Dir.cwd().createFile(io, args[3], .{});
    defer file.close(io);
    var w = file.writer(io, &filebuf);
    try w.interface.writeAll(body.items);
    try w.interface.flush();

    try err.print("{s}: {d} src tokens, {d} tgt tokens, {d} shortlist, {d} stages, {d} bytes\n", .{
        args[3], last.src_len, last.tgt_len, last.shortlist_len, Stages.count, body.items.len,
    });
}

/// The encoder hands its intermediate states out through a function pointer, so
/// the sink has to be a namespace rather than a closure.
const Stages = struct {
    var gpa: std.mem.Allocator = undefined;
    var saved: [64][]f32 = undefined;
    var count: u32 = 0;

    fn observe(stage: u32, states: []const f32) void {
        if (count >= saved.len) return;
        const copy = gpa.alloc(f32, states.len) catch return;
        @memcpy(copy, states);
        saved[count] = copy;
        count += 1;
        std.debug.assert(count == stage + 1);
    }
};

fn appendU32(gpa: std.mem.Allocator, list: *std.ArrayList(u8), x: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, x, .little);
    try list.appendSlice(gpa, &buf);
}

fn benchConfig() abi.Config {
    return .{
        .abi_version = abi.abi_version,
        .max_models = 1,
        .max_model_bytes = 24 << 20,
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
