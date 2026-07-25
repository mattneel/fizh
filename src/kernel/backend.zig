//! kernel/backend.zig — which set of kernels this build uses.
//!
//! All three backends compile on every target. That is what makes T1 possible:
//! the differential harness runs `ref`, `simd128` and `relaxed` in one process,
//! with no FFI and no second toolchain (I2).
//!
//! Selection is by target feature, not by a build option, so `fizh.relaxed.wasm`
//! picks the relaxed path because it *is* the relaxed target — there is no way
//! to build an artifact whose name and contents disagree.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

pub const ref = @import("ref/kernels.zig");
pub const simd128 = @import("simd128/kernels.zig");
pub const relaxed = @import("relaxed/kernels.zig");

const is_wasm = builtin.cpu.arch.isWasm();
const has_relaxed = is_wasm and std.Target.wasm.featureSetHas(builtin.cpu.features, .relaxed_simd);
const has_simd128 = is_wasm and std.Target.wasm.featureSetHas(builtin.cpu.features, .simd128);

/// Native builds take the SIMD128 path: `@Vector` is portable, so the same
/// source that becomes v128 on wasm becomes NEON or AVX here, and the test
/// suite exercises the code that ships rather than a scalar stand-in.
pub const active = if (has_relaxed)
    relaxed
else if (has_simd128 or !is_wasm)
    simd128
else
    ref;

pub const all = [_]type{ ref, simd128, relaxed };

// A backend that is missing an op would otherwise fail at the call site of
// whichever graph function happened to use it first. This fails at the backend
// instead.
comptime {
    for (all) |B| {
        for ([_][]const u8{
            "qgemm",       "qgemv",     "quantizeRows", "layerNorm",
            "softmax",     "activation", "residualAdd", "argmax",
            "embedGather", "gatherRows", "axpy",        "dot",
            "ssruGate",
        }) |op| {
            if (!@hasDecl(B, op)) @compileError("backend " ++ B.name ++ " is missing " ++ op);
        }
    }
}

test "the active backend is the one the target asked for" {
    if (is_wasm) {
        try std.testing.expect(active != ref or !has_simd128);
    } else {
        try std.testing.expectEqualStrings("simd128", active.name);
    }
}
