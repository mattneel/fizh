//! fzm_load.zig — load a `.fzm` with the real loader and say what happened.
//!
//! This is what makes `tools/convert.py` trustworthy: the converter and
//! `src/model/format.zig` are two independent implementations of SPEC §6, and
//! the only way to know they agree is to run one against the other.
//!
//!   zig build convert-selftest
//!   zig build fzm-load -- path/to/model.fzm --d-model 256 --ffn 1536 ...

const std = @import("std");
const Io = std.Io;

const fizh = @import("fizh");
const abi = fizh.abi;

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
        try out.print("usage: fzm-load <model.fzm> [--d-model N] [--ffn N] [--vocab N] ...\n", .{});
        return error.BadUsage;
    }

    const blob = try Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .limited(128 << 20));
    defer gpa.free(blob);

    // Ceilings default to whatever the artifact declares, so `fzm-load x.fzm`
    // answers "does this load" rather than "does this fit the shape I guessed".
    // `zig build convert-selftest` passes every dimension explicitly, which is
    // the opposite question and still the one it asks.
    const hp = fizh.format.peekHParams(blob);
    const slot = std.mem.alignForward(u32, @intCast(blob.len), 64);
    var cfg = abi.Config{
        .abi_version = abi.abi_version,
        .max_models = 1,
        .max_model_bytes = slot +| (slot / 4),
        .max_src_bytes = 4096,
        .max_src_tokens = if (hp) |h| @min(256, h.max_pos) else 256,
        .max_tgt_tokens = if (hp) |h| @min(768, h.max_pos) else 768,
        .max_shortlist = 2048,
        .max_d_model = if (hp) |h| h.d_model else 256,
        .max_ffn_dim = if (hp) |h| h.ffn_dim else 1536,
        .max_enc_layers = if (hp) |h| h.n_enc_layers else 6,
        .max_dec_layers = if (hp) |h| h.n_dec_layers else 2,
        .max_heads = if (hp) |h| h.n_heads else 8,
        .max_vocab = if (hp) |h| h.vocab_size else 32768,
        .reserved = .{ 0, 0, 0 },
    };

    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        const v = try std.fmt.parseInt(u32, args[i + 1], 10);
        const name = args[i];
        if (std.mem.eql(u8, name, "--d-model")) cfg.max_d_model = v;
        if (std.mem.eql(u8, name, "--ffn")) cfg.max_ffn_dim = v;
        if (std.mem.eql(u8, name, "--vocab")) cfg.max_vocab = v;
        if (std.mem.eql(u8, name, "--enc")) cfg.max_enc_layers = v;
        if (std.mem.eql(u8, name, "--dec")) cfg.max_dec_layers = v;
        if (std.mem.eql(u8, name, "--heads")) cfg.max_heads = v;
        if (std.mem.eql(u8, name, "--slot-bytes")) cfg.max_model_bytes = v;
        if (std.mem.eql(u8, name, "--src-tokens")) cfg.max_src_tokens = v;
        if (std.mem.eql(u8, name, "--tgt-tokens")) cfg.max_tgt_tokens = v;
    }

    if (cfg.validate()) |bad| {
        try out.print("config rejected: {s}\n", .{abi.statusStr(bad.int())});
        return error.BadConfig;
    }

    const cfg_bytes = cfg.bytes();
    const n = fizh.arenaBytes(&cfg_bytes);
    if (n == 0) return error.BadConfig;

    const memory = try gpa.alignedAlloc(u8, .fromByteUnits(64), n);
    defer gpa.free(memory);

    const h = fizh.init(memory.ptr, n, &cfg_bytes);
    if (h <= 0) {
        try out.print("init failed: {s}\n", .{abi.statusStr(h)});
        return error.InitFailed;
    }

    const status = fizh.modelLoad(h, 0, blob);
    if (status != 0) {
        try out.print("{s}: REJECTED — {s}\n", .{ args[1], abi.statusStr(status) });
        return error.LoadFailed;
    }

    const inst = fizh.instanceForTest().?;
    const m = &inst.models[0];
    try out.print(
        \\{s}: loaded
        \\  {s} -> {s}
        \\  d_model={d} ffn={d} enc={d} dec={d} heads={d} vocab={d}
        \\  arena {d} bytes, slot {d} bytes of {d}
        \\
    , .{
        args[1],
        &abi.langBytes(m.src_lang),
        &abi.langBytes(m.tgt_lang),
        m.hp.d_model,
        m.hp.ffn_dim,
        m.hp.n_enc_layers,
        m.hp.n_dec_layers,
        m.hp.n_heads,
        m.hp.vocab_size,
        n,
        m.slot_used,
        cfg.max_model_bytes,
    });

    // The tokenizer is the cheapest end-to-end proof that the vocabulary
    // tensors survived the trip.
    const vocab = m.vocab(fizh.slotBytes(inst, 0));
    if (!vocab.validate()) {
        try out.print("  vocabulary FAILED validation\n", .{});
        return error.BadVocab;
    }
    try out.print("  vocabulary ok: {d} pieces, longest {d} bytes\n", .{ vocab.size, vocab.max_piece_len });
}
