//! kernel/backend.zig — which set of kernels this build uses.
//!
//! Both backends compile on every target. That is what makes T1 possible: the
//! differential harness runs `ref` and `vector` in one process, with no FFI and
//! no second toolchain (I2).
//!
//! There was a third, `relaxed/`. It is gone: `f32x4.relaxed_madd` was never
//! emitted, `i32x4.relaxed_dot_i8x16_i7x16_add_s` is unreachable from Zig
//! source at all, and enabling `+relaxed_simd` made float code *slower* by
//! letting LLVM contract `a*b+c` into a correctly-rounded `llvm.fma` it then had
//! to emulate. ADR 0025.

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;

pub const ref = @import("ref/kernels.zig");
pub const vector = @import("vector/kernels.zig");

const is_wasm = builtin.cpu.arch.isWasm();
const has_simd128 = is_wasm and std.Target.wasm.featureSetHas(builtin.cpu.features, .simd128);

/// Native builds take the vector path: it is portable `@Vector` code, so the
/// same source that becomes v128 on wasm becomes NEON on aarch64 and SSE/AVX on
/// x86_64, and the test suite exercises the code that ships rather than a
/// scalar stand-in. `has_simd128` names a *wasm feature*; it is not a claim
/// about the backend, which has no wasm in it (ADR 0023).
pub const active = if (has_simd128 or !is_wasm) vector else ref;

pub const all = [_]type{ ref, vector };

// A backend that is missing an op would otherwise fail at the call site of
// whichever graph function happened to use it first. This fails at the backend
// instead.
comptime {
    for (all) |B| {
        for ([_][]const u8{
            "qgemm",       "qgemv",     "quantizeRows", "layerNorm",
            "softmax",     "activation", "residualAdd", "argmax",
            "embedGather", "gatherRows", "axpy",        "dot",
            "ssruGate",     "quantizeRowsWith",
        }) |op| {
            if (!@hasDecl(B, op)) @compileError("backend " ++ B.name ++ " is missing " ++ op);
        }
    }
}

test "the active backend is the one the target asked for" {
    if (is_wasm) {
        try std.testing.expect(active != ref or !has_simd128);
    } else {
        try std.testing.expectEqualStrings("vector", active.name);
    }
}
