//! abi.zig — status codes and C-layout structs that cross the boundary.
//!
//! Nothing in this file allocates, traps, or touches untrusted memory. It is
//! pure description: the vocabulary that `root.zig` speaks to the host.
//!
//! SPEC §9 (public ABI), §11 (error model), §12.7 (explicitly sized types).

const std = @import("std");
const assert = std.debug.assert;

/// Bumped whenever any struct below, any status code meaning, or any exported
/// signature changes. The host compares this against its own constant before
/// doing anything else.
pub const abi_version: u32 = 1;

// -- status codes -----------------------------------------------------------
//
// SPEC §11: validation errors are negative status codes and must never trap.
// Zero and positive values are success (`fizh_translate` returns bytes written,
// `fizh_can_translate` returns a route kind).

pub const Status = enum(i32) {
    ok = 0,

    /// A pointer was null, a length was zero where it may not be, or two
    /// arguments disagreed.
    bad_arg = -1,
    /// The config blob is the wrong length, has the wrong ABI version, or
    /// carries a limit outside the supported range.
    bad_config = -2,
    /// The handle is not the live instance.
    bad_handle = -3,
    /// `fizh_init` called with fewer bytes than `fizh_arena_bytes` asked for.
    arena_too_small = -4,
    /// Artifact bytes are truncated, misaligned, self-inconsistent, or carry a
    /// magic that is not "FIZH".
    bad_artifact = -5,
    /// Artifact version is not one this build understands.
    bad_version = -6,
    /// Artifact parsed, but a tensor the architecture requires is absent.
    missing_tensor = -7,
    /// Artifact is larger than `max_model_bytes`, or its hyper-parameters
    /// exceed a `max_*` the arena was carved for.
    model_too_large = -8,
    /// Slot index >= `max_models`.
    bad_slot = -9,
    /// No direct model and no English pivot pair for this language pair.
    no_route = -10,
    /// Source text is longer than `max_src_bytes`.
    src_too_long = -11,
    /// Source text is not well-formed UTF-8.
    bad_utf8 = -12,
    /// The output buffer is too small for the translation.
    out_too_small = -13,
    /// Called before `fizh_init`, or a slot referenced before it was loaded.
    not_loaded = -14,
    /// A language code is not two printable ASCII letters.
    bad_lang = -15,

    pub fn int(self: Status) i32 {
        return @intFromEnum(self);
    }
};

/// NUL-terminated, static, safe to call with any i32 including junk.
pub fn statusStr(code: i32) [*:0]const u8 {
    if (code >= 0) return "ok";
    return switch (code) {
        Status.bad_arg.int() => "bad_arg: null pointer or inconsistent length",
        Status.bad_config.int() => "bad_config: config blob rejected",
        Status.bad_handle.int() => "bad_handle: not the live instance",
        Status.arena_too_small.int() => "arena_too_small: fewer bytes than fizh_arena_bytes asked for",
        Status.bad_artifact.int() => "bad_artifact: truncated, misaligned or self-inconsistent",
        Status.bad_version.int() => "bad_version: artifact version unsupported",
        Status.missing_tensor.int() => "missing_tensor: required tensor absent",
        Status.model_too_large.int() => "model_too_large: exceeds a configured max",
        Status.bad_slot.int() => "bad_slot: slot index out of range",
        Status.no_route.int() => "no_route: no direct model and no English pivot",
        Status.src_too_long.int() => "src_too_long: exceeds max_src_bytes",
        Status.bad_utf8.int() => "bad_utf8: source is not well-formed UTF-8",
        Status.out_too_small.int() => "out_too_small: output buffer too small",
        Status.not_loaded.int() => "not_loaded: no instance or no model in slot",
        Status.bad_lang.int() => "bad_lang: language code is not two ASCII letters",
        else => "unknown status",
    };
}

/// Return of `fizh_can_translate`. SPEC §9.
pub const Route = enum(i32) {
    none = 0,
    direct = 1,
    pivot = 2,
};

// -- language codes ---------------------------------------------------------

/// One to four lowercase ASCII bytes packed big-endian and left-padded with
/// zeros: `"es"` is `0x0000_6573`, `"zhs"` is `0x007a_6873`.
///
/// This was `u16` — exactly two bytes — which is a convention this project
/// invented and then found the registry does not honour. ISO 639-3 codes are
/// three letters, and Firefox ships script-qualified pairs (`zh-Hans`,
/// `zh-Hant`) that two bytes cannot express at all: `zh-Hans-en` was refused
/// for that reason alone. Four bytes covers 639-1, 639-3, and fizh's
/// short forms for the script-qualified cases (see `langShort` in
/// `tools/bergamot.py`).
pub const Lang = u32;

pub const max_lang_len = 4;

pub const lang_en: Lang = langFrom("en");

/// Comptime-friendly: takes any 1..4 byte string.
pub fn langFrom(s: []const u8) Lang {
    assert(s.len >= 1 and s.len <= max_lang_len);
    var out: Lang = 0;
    for (s) |c| out = (out << 8) | c;
    return out;
}

pub fn langValid(l: Lang) bool {
    if (l == 0) return false;
    var seen_end = false;
    // Big-endian with zero padding: once a zero byte appears, every byte above
    // it must be zero too, and every byte below must be a lowercase letter.
    var i: u5 = 0;
    while (i < max_lang_len) : (i += 1) {
        const b: u8 = @truncate(l >> (@as(u5, 8) * (max_lang_len - 1 - i)));
        if (b == 0) {
            seen_end = i == 0 or !seen_end;
            continue;
        }
        if (b < 'a' or b > 'z') return false;
        // A zero between two letters is not padding, it is corruption.
        if (i > 0) {
            const prev: u8 = @truncate(l >> (@as(u5, 8) * (max_lang_len - i)));
            if (prev == 0 and i > 1) {
                const above: u8 = @truncate(l >> (@as(u5, 8) * (max_lang_len - i + 1)));
                if (above != 0) return false;
            }
        }
    }
    return true;
}

/// The code as text, into `buf`. Returns the written slice.
pub fn langBytes(l: Lang, buf: *[max_lang_len]u8) []const u8 {
    var n: usize = 0;
    var i: u5 = 0;
    while (i < max_lang_len) : (i += 1) {
        const b: u8 = @truncate(l >> (@as(u5, 8) * (max_lang_len - 1 - i)));
        if (b == 0 and n == 0) continue;
        buf[n] = b;
        n += 1;
    }
    return buf[0..n];
}

// -- configuration ----------------------------------------------------------

/// Hard ceilings on every `max_*` a host may ask for. These exist so that
/// `Layout.compute` can do all of its arithmetic in `u64` and prove the result
/// fits in `u32` without a single overflow check in the hot path.
pub const limits = struct {
    pub const models: u32 = 8;
    pub const model_bytes: u32 = 64 << 20;
    pub const src_bytes: u32 = 1 << 20;
    pub const src_tokens: u32 = 4096;
    pub const tgt_tokens: u32 = 8192;
    pub const shortlist: u32 = 1 << 16;
    pub const d_model: u32 = 4096;
    pub const ffn_dim: u32 = 16384;
    pub const enc_layers: u32 = 32;
    pub const dec_layers: u32 = 32;
    pub const heads: u32 = 64;
    pub const vocab: u32 = 1 << 20;
};

/// The config blob the host passes to `fizh_arena_bytes` and `fizh_init`.
/// `extern` layout, little-endian, 64 bytes, no padding holes.
pub const Config = extern struct {
    abi_version: u32,
    /// Weight slots to carve. SPEC §4.1: only these are additive.
    max_models: u32,
    /// Per-slot ceiling on a `.fzm` payload, post-repack.
    max_model_bytes: u32,
    max_src_bytes: u32,
    max_src_tokens: u32,
    max_tgt_tokens: u32,
    max_shortlist: u32,
    max_d_model: u32,
    max_ffn_dim: u32,
    max_enc_layers: u32,
    max_dec_layers: u32,
    max_heads: u32,
    /// Largest vocabulary across the models that will be loaded. Sizes the
    /// shortlist dedupe bitmap.
    max_vocab: u32,
    reserved: [3]u32,

    comptime {
        assert(@sizeOf(Config) == 64);
        assert(@alignOf(Config) == 4);
    }

    /// Every rejection here is a *validation* error: these bytes came from the
    /// host and are untrusted. SPEC §11.
    pub fn validate(self: Config) ?Status {
        if (self.abi_version != abi_version) return .bad_config;

        if (self.max_models == 0 or self.max_models > limits.models) return .bad_config;
        if (self.max_model_bytes == 0 or self.max_model_bytes > limits.model_bytes) return .bad_config;
        if (self.max_src_bytes < 16 or self.max_src_bytes > limits.src_bytes) return .bad_config;
        if (self.max_src_tokens == 0 or self.max_src_tokens > limits.src_tokens) return .bad_config;
        if (self.max_tgt_tokens == 0 or self.max_tgt_tokens > limits.tgt_tokens) return .bad_config;
        if (self.max_shortlist == 0 or self.max_shortlist > limits.shortlist) return .bad_config;
        if (self.max_d_model < 16 or self.max_d_model > limits.d_model) return .bad_config;
        if (self.max_ffn_dim < 16 or self.max_ffn_dim > limits.ffn_dim) return .bad_config;
        if (self.max_enc_layers == 0 or self.max_enc_layers > limits.enc_layers) return .bad_config;
        if (self.max_dec_layers == 0 or self.max_dec_layers > limits.dec_layers) return .bad_config;
        if (self.max_heads == 0 or self.max_heads > limits.heads) return .bad_config;
        if (self.max_vocab < 2 or self.max_vocab > limits.vocab) return .bad_config;

        // Lane-width and head-split assumptions the kernels rely on.
        if (self.max_d_model % 16 != 0) return .bad_config;
        if (self.max_ffn_dim % 16 != 0) return .bad_config;
        if (self.max_d_model % self.max_heads != 0) return .bad_config;

        for (self.reserved) |r| if (r != 0) return .bad_config;
        return null;
    }

    /// Reads a config out of untrusted host bytes. Never traps, never reads
    /// past `len`, and does not require the pointer to be aligned.
    pub fn parse(blob: []const u8) ?Config {
        if (blob.len != @sizeOf(Config)) return null;
        var raw: [@sizeOf(Config)]u8 align(@alignOf(Config)) = undefined;
        @memcpy(&raw, blob);
        // wasm is little-endian and so is every host we support; SPEC §6 says
        // no byte swapping exists anywhere in this project.
        return @bitCast(raw);
    }

    pub fn bytes(self: *const Config) [@sizeOf(Config)]u8 {
        return @bitCast(self.*);
    }
};

test "status strings are total" {
    // Any i32, including values that are not codes, must produce a string.
    const probes = [_]i32{ 0, 1, -1, -15, -16, -1000, std.math.maxInt(i32), std.math.minInt(i32) };
    for (probes) |p| {
        const s = statusStr(p);
        try std.testing.expect(std.mem.len(s) > 0);
    }
}

test "language codes round-trip" {
    try std.testing.expectEqual(@as(Lang, 'e' << 8 | 's'), langFrom("es"));
    try std.testing.expect(langValid(langFrom("de")));
    try std.testing.expect(langValid(langFrom("zhs")));
    try std.testing.expect(langValid(langFrom("nqo")));
    try std.testing.expect(!langValid(0));
    try std.testing.expect(!langValid(langFrom("ES")));
    // A zero between letters is corruption, not padding.
    try std.testing.expect(!langValid(('a' << 24) | ('b' << 8) | 'c'));
    var buf: [max_lang_len]u8 = undefined;
    try std.testing.expectEqualSlices(u8, "en", langBytes(lang_en, &buf));
    try std.testing.expectEqualSlices(u8, "zhs", langBytes(langFrom("zhs"), &buf));
}

test "config rejects junk without trapping" {
    var cfg = std.mem.zeroes(Config);
    try std.testing.expectEqual(Status.bad_config, Config.validate(cfg).?);

    cfg = defaultTestConfig();
    try std.testing.expectEqual(@as(?Status, null), cfg.validate());

    // A short blob is a validation error, not a crash.
    try std.testing.expect(Config.parse(&[_]u8{ 1, 2, 3 }) == null);

    const round = Config.parse(&cfg.bytes()).?;
    try std.testing.expectEqual(cfg.max_ffn_dim, round.max_ffn_dim);

    cfg.reserved[2] = 7;
    try std.testing.expectEqual(Status.bad_config, cfg.validate().?);
}

/// The SPEC §4.3 worked example, used by tests across the tree.
pub fn defaultTestConfig() Config {
    return .{
        .abi_version = abi_version,
        .max_models = 2,
        .max_model_bytes = 20 << 20,
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
