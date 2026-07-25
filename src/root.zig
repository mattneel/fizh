//! root.zig — the ABI exports. SPEC §9.
//!
//! Nothing else lives here. Every export is a thin, total function: it
//! validates its arguments, hands them to `runtime.zig`, and turns the result
//! into an `i32`. No export may trap on host input (SPEC §11).

const builtin = @import("builtin");
const std = @import("std");

const abi = @import("abi.zig");
const runtime = @import("runtime.zig");

/// SPEC §10/§11: an invariant violation must kill the worker, not narrate. On
/// freestanding wasm the panic handler is a bare `unreachable`, which keeps the
/// module import count at zero and the artifact small.
pub const panic = if (builtin.os.tag == .freestanding)
    std.debug.no_panic
else
    std.debug.FullPanic(std.debug.defaultPanic);

export fn fizh_abi_version() u32 {
    return abi.abi_version;
}

/// Bytes the arena needs for this config. Zero means the config was rejected;
/// the host can call `fizh_init` to learn why.
export fn fizh_arena_bytes(cfg: [*]const u8, cfg_len: u32) u32 {
    const blob = slice(cfg, cfg_len) orelse return 0;
    return runtime.arenaBytes(blob);
}

/// Returns a positive handle, or a negative `abi.Status`.
export fn fizh_init(base: [*]u8, len: u32, cfg: [*]const u8, cfg_len: u32) i32 {
    if (@intFromPtr(base) == 0) return abi.Status.bad_arg.int();
    const blob = slice(cfg, cfg_len) orelse return abi.Status.bad_arg.int();
    return runtime.init(base, len, blob);
}

export fn fizh_model_load(h: i32, slot: u32, blob: [*]const u8, len: u32) i32 {
    const bytes = slice(blob, len) orelse return abi.Status.bad_arg.int();
    return runtime.modelLoad(h, slot, bytes);
}

/// Bytes written into `out`, or a negative `abi.Status`. One call does the
/// work: tokenization, pivoting and detokenization are internal.
export fn fizh_translate(
    h: i32,
    src: [*]const u8,
    src_len: u32,
    src_lang: u16,
    tgt_lang: u16,
    out: [*]u8,
    out_cap: u32,
) i32 {
    const in = slice(src, src_len) orelse return abi.Status.bad_arg.int();
    if (@intFromPtr(out) == 0) return abi.Status.bad_arg.int();
    return runtime.translate(h, in, src_lang, tgt_lang, out[0..out_cap]);
}

/// 0 = no, 1 = direct, 2 = pivot. Lets the host grey out a language before the
/// user picks it.
export fn fizh_can_translate(h: i32, src_lang: u16, tgt_lang: u16) i32 {
    return runtime.canTranslate(h, src_lang, tgt_lang);
}

export fn fizh_status_str(code: i32) [*:0]const u8 {
    return abi.statusStr(code);
}

/// A zero-length blob at a null pointer is the one shape the host is allowed to
/// get wrong for free; anything else with a null pointer is `bad_arg`.
fn slice(p: [*]const u8, len: u32) ?[]const u8 {
    if (@intFromPtr(p) == 0) return null;
    return p[0..len];
}
