//! make_probe.zig — emits `fizh.probe.wasm`, the feature-detection module.
//!
//! SPEC §3: "Host picks via probe-module feature detection." The host does
//!
//!     const relaxed = WebAssembly.validate(probeBytes);
//!     const url = relaxed ? "fizh.relaxed.wasm" : "fizh.baseline.wasm";
//!
//! and a browser without relaxed SIMD rejects the module at validation, before
//! instantiation and before anything runs.
//!
//! The bytes are written out by hand rather than compiled, because a compiler
//! is the wrong tool for this job: the probe's entire purpose is to contain one
//! specific instruction, and LLVM decides for itself which instructions to
//! emit. `src/probe.zig` was compiled with `+relaxed_simd` and `@mulAdd` and
//! came out containing no relaxed opcode at all — see ADR 0006. A probe that
//! validates everywhere would route every device to the relaxed build, which is
//! the exact failure this module exists to prevent.

const std = @import("std");
const Io = std.Io;

/// `f32x4.relaxed_madd` is `0xFD` followed by LEB128(0x105).
const relaxed_madd = [_]u8{ 0xFD, 0x85, 0x02 };
/// `v128.const` is `0xFD 12`, followed by sixteen immediate bytes.
const v128_const = [_]u8{ 0xFD, 0x0C };
/// `f32x4.extract_lane` is `0xFD 31`, followed by the lane index.
const extract_lane_0 = [_]u8{ 0xFD, 0x1F, 0x00 };

/// (module (func (export "p") (result f32)
///   (f32x4.extract_lane 0 (f32x4.relaxed_madd (v128.const …) ×3))))
pub const bytes = blk: {
    const ones = [_]u8{ 0x00, 0x00, 0x80, 0x3F } ** 4; // f32x4(1,1,1,1)

    const body =
        [_]u8{0x00} ++ // no locals
        v128_const ++ ones ++
        v128_const ++ ones ++
        v128_const ++ ones ++
        relaxed_madd ++
        extract_lane_0 ++
        [_]u8{0x0B}; // end

    const code_payload = [_]u8{ 0x01, body.len } ++ body;

    break :blk [_]u8{ 0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00 } ++
        // type: one functype () -> f32
        [_]u8{ 0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7D } ++
        // function: one function, type 0
        [_]u8{ 0x03, 0x02, 0x01, 0x00 } ++
        // export: "p" -> func 0
        [_]u8{ 0x07, 0x05, 0x01, 0x01, 'p', 0x00, 0x00 } ++
        // code
        [_]u8{ 0x0A, code_payload.len } ++ code_payload;
};

comptime {
    // Every length byte above is a single-byte LEB128, which only holds while
    // the module stays under 128 bytes.
    std.debug.assert(bytes.len < 128);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();

    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) return error.BadUsage;

    var buf: [256]u8 = undefined;
    var file = try Io.Dir.cwd().createFile(io, args[1], .{});
    defer file.close(io);
    var w = file.writer(io, &buf);
    try w.interface.writeAll(&bytes);
    try w.interface.flush();
}

test "the probe is a well-formed module carrying the opcode it advertises" {
    try std.testing.expectEqualSlices(u8, "\x00asm", bytes[0..4]);
    try std.testing.expect(std.mem.indexOf(u8, &bytes, &relaxed_madd) != null);

    // Walk the section table the way a validator would, and check the declared
    // sizes actually land on the end of the module.
    var i: usize = 8;
    var seen: u32 = 0;
    while (i < bytes.len) {
        const id = bytes[i];
        const size = bytes[i + 1];
        try std.testing.expect(id == 1 or id == 3 or id == 7 or id == 10);
        i += 2 + size;
        seen += 1;
    }
    try std.testing.expectEqual(bytes.len, i);
    try std.testing.expectEqual(@as(u32, 4), seen);
}
