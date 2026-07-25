//! A `.fzm` writer, for tests.
//!
//! It is the second implementation of SPEC §6 in this repository, which is the
//! point: `tools/convert.py` is the first, `src/model/format.zig` is the
//! reader, and a disagreement between any two of the three shows up as a
//! failing test rather than as a mistranslation.
//!
//! Deterministic throughout — the weights come from a fixed LCG — so a golden
//! blob is reproducible byte for byte.

const std = @import("std");
const Allocator = std.mem.Allocator;

const names = @import("../src/model/names.zig");
const format = @import("../src/model/format.zig");
const trie = @import("../src/tok/trie.zig");

pub const Spec = struct {
    src_lang: [2]u8 = "es".*,
    tgt_lang: [2]u8 = "en".*,
    d_model: u16 = 32,
    ffn_dim: u16 = 64,
    n_enc_layers: u8 = 2,
    n_dec_layers: u8 = 1,
    n_heads: u8 = 2,
    max_pos: u16 = 64,
    prenorm: u8 = 0,
    ffn_act: u8 = 0,
    /// Piece texts in vocabulary-id order. Ids 0 and 1 are the control symbols.
    pieces: []const []const u8 = &default_pieces,
    /// Candidates per source id in the shortlist.
    shortlist_per_source: u32 = 3,
    shortlist_frequent: u32 = 4,
    seed: u64 = 0x1234_5678_9abc_def0,
};

pub const default_pieces = [_][]const u8{
    "</s>", "<unk>", "\xff",   "\xffhola", "\xffque", "\xfftal",
    "\xffhello", "\xffhow",   "\xffare",  "\xffyou",  "a",     "e",
    "h",    "l",    "o",      "q",        "t",        "u",     "!",
    "?",
};

/// A built artifact plus the pieces it was built from, so a test can check what
/// came back out.
pub const Artifact = struct {
    bytes: []u8,
    spec: Spec,

    pub fn deinit(self: *Artifact, gpa: Allocator) void {
        gpa.free(self.bytes);
    }
};

const Tensor = struct {
    hash: u64,
    dtype: format.DType,
    rank: u8,
    dims: [4]u32,
    data: []const u8,
    scales: ?[]const u8,
};

pub fn build(gpa: Allocator, spec: Spec) !Artifact {
    var b: Builder = .{ .gpa = gpa, .spec = spec, .rng = std.Random.DefaultPrng.init(spec.seed) };
    defer b.deinit();

    try b.shared();
    try b.layers();
    try b.tokenizer();
    try b.shortlist();

    return .{ .bytes = try b.emit(), .spec = spec };
}

const Builder = struct {
    gpa: Allocator,
    spec: Spec,
    rng: std.Random.DefaultPrng,
    tensors: std.ArrayList(Tensor) = .empty,
    owned: std.ArrayList([]u8) = .empty,

    fn deinit(self: *Builder) void {
        for (self.owned.items) |p| self.gpa.free(p);
        self.owned.deinit(self.gpa);
        self.tensors.deinit(self.gpa);
    }

    fn keep(self: *Builder, buf: []u8) ![]u8 {
        try self.owned.append(self.gpa, buf);
        return buf;
    }

    fn vocabSize(self: *const Builder) u32 {
        return @intCast(self.spec.pieces.len);
    }

    /// int8 in [-127, 127]: SPEC §7 / I5 forbids -128 and the loader checks.
    fn quant(self: *Builder, hash: u64, n: u32, k: u32) !void {
        const data = try self.keep(try self.gpa.alloc(u8, @as(usize, n) * k));
        const r = self.rng.random();
        for (data) |*x| x.* = @bitCast(@as(i8, @intCast(r.intRangeAtMost(i32, -127, 127))));

        const scales = try self.keep(try self.gpa.alloc(u8, @as(usize, n) * 4));
        const view = std.mem.bytesAsSlice(f32, scales);
        for (view) |*s| s.* = 0.001 + r.float(f32) * 0.01;

        try self.tensors.append(self.gpa, .{
            .hash = hash,
            .dtype = .i8,
            .rank = 2,
            .dims = .{ n, k, 0, 0 },
            .data = data,
            .scales = scales,
        });
    }

    fn floats(self: *Builder, hash: u64, count: u32, fill: enum { random, ones, zeros }) !void {
        const data = try self.keep(try self.gpa.alloc(u8, @as(usize, count) * 4));
        const view = std.mem.bytesAsSlice(f32, data);
        const r = self.rng.random();
        for (view) |*x| x.* = switch (fill) {
            .random => (r.float(f32) - 0.5) * 0.2,
            .ones => 1.0,
            .zeros => 0.0,
        };
        try self.vector(hash, .f32, count, data);
    }

    fn u32s(self: *Builder, hash: u64, values: []const u32) !void {
        const data = try self.keep(try self.gpa.alloc(u8, values.len * 4));
        @memcpy(data, std.mem.sliceAsBytes(values));
        try self.vector(hash, .u32, @intCast(values.len), data);
    }

    fn u16s(self: *Builder, hash: u64, values: []const u32) !void {
        const data = try self.keep(try self.gpa.alloc(u8, values.len * 2));
        const view = std.mem.bytesAsSlice(u16, data);
        for (values, view) |v, *o| o.* = @intCast(v);
        try self.vector(hash, .u16, @intCast(values.len), data);
    }

    fn u8s(self: *Builder, hash: u64, values: []const u8) !void {
        const data = try self.keep(try self.gpa.dupe(u8, values));
        try self.vector(hash, .u8, @intCast(values.len), data);
    }

    fn vector(self: *Builder, hash: u64, dtype: format.DType, count: u32, data: []const u8) !void {
        try self.tensors.append(self.gpa, .{
            .hash = hash,
            .dtype = dtype,
            .rank = 1,
            .dims = .{ count, 0, 0, 0 },
            .data = data,
            .scales = null,
        });
    }

    fn shared(self: *Builder) !void {
        const d: u32 = self.spec.d_model;
        try self.quant(names.emb, self.vocabSize(), d);
        try self.floats(names.emb_bias, self.vocabSize(), .zeros);
        try self.floats(names.enc_ln_gain, d, .ones);
        try self.floats(names.enc_ln_bias, d, .zeros);
        try self.floats(names.dec_ln_gain, d, .ones);
        try self.floats(names.dec_ln_bias, d, .zeros);
    }

    fn layers(self: *Builder) !void {
        for (0..self.spec.n_enc_layers) |raw| {
            const i: u32 = @intCast(raw);
            try self.attn(.enc, i, "att");
            try self.ffn(.enc, i);
        }
        for (0..self.spec.n_dec_layers) |raw| {
            const i: u32 = @intCast(raw);
            try self.ssru(i);
            try self.attn(.dec, i, "xa");
            try self.ffn(.dec, i);
        }
    }

    const Side = enum { enc, dec };

    fn nameOf(side: Side, i: u32, comptime stem: []const u8) u64 {
        return switch (side) {
            .enc => names.enc(i, stem),
            .dec => names.dec(i, stem),
        };
    }

    /// Bergamot's decoder cell (ADR 0008).
    fn ssru(self: *Builder, i: u32) !void {
        const d: u32 = self.spec.d_model;
        try self.quant(names.dec(i, "rnn.w"), d, d);
        try self.quant(names.dec(i, "rnn.wf"), d, d);
        try self.floats(names.dec(i, "rnn.bf"), d, .zeros);
        try self.floats(names.dec(i, "rnn.ln.gain"), d, .ones);
        try self.floats(names.dec(i, "rnn.ln.bias"), d, .zeros);
    }

    fn attn(self: *Builder, side: Side, i: u32, comptime tag: []const u8) !void {
        const d: u32 = self.spec.d_model;
        inline for (.{ "q", "k", "v", "o" }) |part| {
            try self.quant(nameOf(side, i, tag ++ "." ++ part ++ ".w"), d, d);
            try self.floats(nameOf(side, i, tag ++ "." ++ part ++ ".bias"), d, .zeros);
        }
        try self.floats(nameOf(side, i, tag ++ ".ln.gain"), d, .ones);
        try self.floats(nameOf(side, i, tag ++ ".ln.bias"), d, .zeros);
    }

    fn ffn(self: *Builder, side: Side, i: u32) !void {
        const d: u32 = self.spec.d_model;
        const f: u32 = self.spec.ffn_dim;
        try self.quant(nameOf(side, i, "ffn.w1"), f, d);
        try self.floats(nameOf(side, i, "ffn.bias1"), f, .zeros);
        try self.quant(nameOf(side, i, "ffn.w2"), d, f);
        try self.floats(nameOf(side, i, "ffn.bias2"), d, .zeros);
        try self.floats(nameOf(side, i, "ffn.ln.gain"), d, .ones);
        try self.floats(nameOf(side, i, "ffn.ln.bias"), d, .zeros);
    }

    fn tokenizer(self: *Builder) !void {
        const v = self.vocabSize();
        const pieces = self.spec.pieces;

        var blob: std.ArrayList(u8) = .empty;
        defer blob.deinit(self.gpa);
        const offsets = try self.gpa.alloc(u32, v + 1);
        defer self.gpa.free(offsets);

        offsets[0] = 0;
        for (pieces, 0..) |p, i| {
            try blob.appendSlice(self.gpa, p);
            offsets[i + 1] = @intCast(blob.items.len);
        }

        const order = try self.gpa.alloc(u32, v);
        defer self.gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, pieces, lexLess);

        const scores = try self.keep(try self.gpa.alloc(u8, @as(usize, v) * 4));
        const sview = std.mem.bytesAsSlice(f32, scores);
        for (pieces, 0..) |p, i| sview[i] = -20.0 + @as(f32, @floatFromInt(p.len));

        const flags = try self.gpa.alloc(u8, v);
        defer self.gpa.free(flags);
        @memset(flags, 0);
        flags[0] = trie.Vocab.flag_special;
        flags[1] = trie.Vocab.flag_special;

        try self.u8s(names.tok_pieces, blob.items);
        try self.u32s(names.tok_offsets, offsets);
        try self.vector(names.tok_scores, .f32, v, scores);
        try self.u32s(names.tok_order, order);
        try self.u8s(names.tok_flags, flags);
    }

    fn shortlist(self: *Builder) !void {
        const v = self.vocabSize();
        const per = self.spec.shortlist_per_source;

        const offsets = try self.gpa.alloc(u32, v + 1);
        defer self.gpa.free(offsets);
        const targets = try self.gpa.alloc(u32, @as(usize, v) * per);
        defer self.gpa.free(targets);

        for (0..v) |src| {
            offsets[src] = @intCast(src * per);
            for (0..per) |j| {
                targets[src * per + j] = @intCast((src + j * 7 + 1) % v);
            }
        }
        offsets[v] = @intCast(targets.len);

        const frequent = try self.gpa.alloc(u32, self.spec.shortlist_frequent);
        defer self.gpa.free(frequent);
        for (frequent, 0..) |*f, i| f.* = @intCast(i % v);

        try self.u32s(names.sl_offsets, offsets);
        try self.u16s(names.sl_targets, targets);
        try self.u32s(names.sl_frequent, frequent);
    }

    fn emit(self: *Builder) ![]u8 {
        const count: u32 = @intCast(self.tensors.items.len);
        const table_end = format.header_bytes + @as(u64, count) * format.TensorDesc.wire_size;

        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(self.gpa);
        try out.appendNTimes(self.gpa, 0, @intCast(std.mem.alignForward(u64, table_end, 64)));

        // Payload first, recording where everything landed.
        var placed = try self.gpa.alloc(format.TensorDesc, count);
        defer self.gpa.free(placed);

        for (self.tensors.items, 0..) |t, i| {
            const at = try self.append(&out, t.data);
            var scale_off: u64 = format.TensorDesc.no_scales;
            if (t.scales) |s| scale_off = try self.append(&out, s);
            placed[i] = .{
                .name_hash = t.hash,
                .dtype = @intFromEnum(t.dtype),
                .rank = t.rank,
                ._pad = 0,
                .dims = t.dims,
                ._pad2 = 0,
                .offset = at,
                .nbytes = t.data.len,
                .scale_offset = scale_off,
            };
        }

        self.writeHeader(out.items, count);
        for (placed, 0..) |d, i| {
            const at: usize = @intCast(format.header_bytes + @as(u64, i) * format.TensorDesc.wire_size);
            const raw: [@sizeOf(format.TensorDesc)]u8 = @bitCast(d);
            @memcpy(out.items[at..][0..raw.len], &raw);
        }
        return out.toOwnedSlice(self.gpa);
    }

    fn append(self: *Builder, out: *std.ArrayList(u8), data: []const u8) !u64 {
        const at = std.mem.alignForward(usize, out.items.len, 64);
        try out.appendNTimes(self.gpa, 0, at - out.items.len);
        try out.appendSlice(self.gpa, data);
        return at;
    }

    fn writeHeader(self: *const Builder, out: []u8, count: u32) void {
        @memcpy(out[0..4], &format.magic);
        std.mem.writeInt(u32, out[4..8], format.version, .little);
        std.mem.writeInt(u16, out[8..10], packLang(self.spec.src_lang), .little);
        std.mem.writeInt(u16, out[10..12], packLang(self.spec.tgt_lang), .little);

        const hp: format.HParams = .{
            .d_model = self.spec.d_model,
            .ffn_dim = self.spec.ffn_dim,
            .n_enc_layers = self.spec.n_enc_layers,
            .n_dec_layers = self.spec.n_dec_layers,
            .n_heads = self.spec.n_heads,
            .head_dim = @intCast(self.spec.d_model / self.spec.n_heads),
            .vocab_size = self.vocabSize(),
            .max_pos = self.spec.max_pos,
            .shortlist_width = 64,
            .eos_id = 0,
            .bos_id = 0,
            .unk_id = 1,
            .pad_id = 0,
            .ffn_act = self.spec.ffn_act,
            .prenorm = self.spec.prenorm,
            .tied_embeddings = 1,
            .act_quant = 0,
            .emb_scale = @sqrt(@as(f32, @floatFromInt(self.spec.d_model))),
            .norm_eps = 1e-9,
            .max_length_factor = 3.0,
        };
        const raw: [48]u8 = @bitCast(hp);
        @memcpy(out[12..60], &raw);
        std.mem.writeInt(u32, out[60..64], count, .little);
    }
};

fn lexLess(pieces: []const []const u8, a: u32, b: u32) bool {
    return std.mem.order(u8, pieces[a], pieces[b]) == .lt;
}

fn packLang(s: [2]u8) u16 {
    return (@as(u16, s[0]) << 8) | s[1];
}

test "the builder produces something the reader accepts" {
    var art = try build(std.testing.allocator, .{});
    defer art.deinit(std.testing.allocator);
    try std.testing.expect(art.bytes.len > 1024);
    try std.testing.expectEqualSlices(u8, "FIZH", art.bytes[0..4]);
}
