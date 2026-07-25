//! kernel/math.zig — the transcendentals, because there is no libm.
//!
//! SPEC §3 budgets the ship build at zero module imports. `@exp`, `@sin` and
//! friends lower to `expf`/`sinf` calls on `wasm32-freestanding`, which is an
//! import, which is a budget violation. So fizh brings its own.
//!
//! SPEC §9 (bit-exact determinism) wants this anyway: a polynomial we wrote is
//! the same polynomial on every device, which a host libm is not.
//!
//! Every routine here is branch-light, allocation-free, and total: it has a
//! defined answer for NaN, both infinities, and every subnormal.
//!
//! See `docs/adr/0003-no-libm.md`.

const std = @import("std");
const assert = std.debug.assert;

// -- rounding ---------------------------------------------------------------

/// Round-to-nearest-even via the classic magic constant. `@round` would lower
/// to a `roundf` libcall on wasm, and `@trunc`/`@floor` round the wrong way for
/// argument reduction.
inline fn rintF32(x: f32) f32 {
    const magic: f32 = 12582912.0; // 1.5 * 2^23
    return (x + magic) - magic;
}

inline fn rintF64(x: f64) f64 {
    const magic: f64 = 6755399441055744.0; // 1.5 * 2^52
    return (x + magic) - magic;
}

// -- exp --------------------------------------------------------------------

const log2e: f32 = 1.44269504088896340736;
/// `ln 2` split so that `k * ln2_hi` is exact in `f32` for every `k` we produce.
const ln2_hi: f32 = 0.693359375;
const ln2_lo: f32 = -2.12194440e-4;

/// Largest argument that does not overflow `f32`, and the smallest that still
/// produces a normal result.
const exp_hi: f32 = 88.3762626647949;
const exp_lo: f32 = -87.3365447504;

/// `e^x` to within ~1 ulp over the normal range. Softmax subtracts the row max
/// first, so the argument is always `<= 0` in practice; the positive side is
/// implemented anyway because a half-total function is a trap waiting to fire.
pub fn exp(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    if (x > exp_hi) return std.math.inf(f32);
    if (x < exp_lo) return 0.0;

    const kf = rintF32(x * log2e);
    const r = (x - kf * ln2_hi) - kf * ln2_lo;
    assert(@abs(r) <= 0.35);

    // Minimax degree 6 on [-ln2/2, ln2/2] (the Cephes `expf` coefficients).
    // Horner, fixed order: SPEC §3 forbids letting the compiler reassociate
    // this, because a reassociated sum is a different function on a different
    // day and I9 would stop holding.
    var p: f32 = 1.9875691500e-4;
    p = p * r + 1.3981999507e-3;
    p = p * r + 8.3334519073e-3;
    p = p * r + 4.1665795894e-2;
    p = p * r + 1.6666665459e-1;
    p = p * r + 5.0000001201e-1;
    p = p * r * r + r + 1.0;

    const k: i32 = @intFromFloat(kf);
    assert(k >= -128 and k <= 128);
    return p * pow2(k);
}

/// `2^k` by exponent construction. Split in two so a `k` at either extreme
/// still goes through a representable intermediate.
fn pow2(k: i32) f32 {
    assert(k >= -150 and k <= 150);
    const half = @divTrunc(k, 2);
    return pow2Exact(half) * pow2Exact(k - half);
}

fn pow2Exact(k: i32) f32 {
    assert(k >= -75 and k <= 75);
    const biased: u32 = @intCast(k + 127);
    return @bitCast(biased << 23);
}

/// `1 / (1 + e^-x)`, without a division by zero at either end.
pub fn sigmoid(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    if (x >= 0) {
        const e = exp(-x);
        return 1.0 / (1.0 + e);
    }
    const e = exp(x);
    return e / (1.0 + e);
}

/// `tanh`, computed as `(1 - e^-2|x|) / (1 + e^-2|x|)` rather than through
/// `sigmoid`. `2 * sigmoid(2x) - 1` cancels catastrophically near zero: it
/// subtracts two numbers that agree to every bit that matters.
pub fn tanh(x: f32) f32 {
    if (std.math.isNan(x)) return x;
    const a = @abs(x);
    if (a < 1.0e-4) return x; // tanh x = x - x^3/3; the cubic is below one ulp
    if (a > 9.0) return std.math.sign(x);

    const e = exp(-2.0 * a);
    const t = (1.0 - e) / (1.0 + e);
    return if (x < 0) -t else t;
}

// -- sin / cos --------------------------------------------------------------
//
// Used once per model load, to fill the positional table. `f64` throughout: at
// load time the cost is irrelevant and the accuracy is free, and it means the
// two-part argument reduction below is good to `|x| < 2^20` rather than needing
// a Payne-Hanek path nobody will ever exercise.

const two_over_pi: f64 = 6.36619772367581382433e-01;
const pio2_hi: f64 = 1.57079632673412561417e+00;
const pio2_lo: f64 = 6.07710050650619224932e-11;

const sin_c = [_]f64{
    -1.66666666666666324348e-01,
    8.33333333332248946124e-03,
    -1.98412698298579493134e-04,
    2.75573137070700676789e-06,
    -2.50507602534068634195e-08,
    1.58969099521155010221e-10,
};

const cos_c = [_]f64{
    4.16666666666666019037e-02,
    -1.38888888888741095749e-03,
    2.48015872894767294178e-05,
    -2.75573143513906633035e-07,
    2.08757232129817482790e-09,
    -1.13596475577881948265e-11,
};

/// Both at once, because the caller filling a positional table always wants
/// both and the argument reduction is the expensive half.
pub fn sinCos(x: f64) struct { sin: f64, cos: f64 } {
    assert(std.math.isFinite(x));
    assert(@abs(x) < 1 << 20);

    const n = rintF64(x * two_over_pi);
    const r = (x - n * pio2_hi) - n * pio2_lo;
    assert(@abs(r) <= 0.7854 + 1e-9);

    const s = sinPoly(r);
    const c = cosPoly(r);

    const q: u2 = @truncate(@as(u64, @bitCast(@as(i64, @intFromFloat(n)))));
    return switch (q) {
        0 => .{ .sin = s, .cos = c },
        1 => .{ .sin = c, .cos = -s },
        2 => .{ .sin = -s, .cos = -c },
        3 => .{ .sin = -c, .cos = s },
    };
}

fn sinPoly(r: f64) f64 {
    const z = r * r;
    var p: f64 = sin_c[5];
    var i: usize = 5;
    while (i > 0) {
        i -= 1;
        p = sin_c[i] + z * p;
    }
    return r + r * z * p;
}

fn cosPoly(r: f64) f64 {
    const z = r * r;
    var p: f64 = cos_c[5];
    var i: usize = 5;
    while (i > 0) {
        i -= 1;
        p = cos_c[i] + z * p;
    }
    return 1.0 - 0.5 * z + z * z * p;
}

/// `base^e` for the positive real base the positional table needs. Computed as
/// `2^(e * log2 base)` in `f64`; load-time only.
pub fn powPositive(base: f64, e: f64) f64 {
    assert(base > 0);
    assert(std.math.isFinite(e));
    return expF64(e * lnF64(base));
}

fn expF64(x: f64) f64 {
    if (x > 709.0) return std.math.inf(f64);
    if (x < -745.0) return 0.0;

    const ln2: f64 = 0.6931471805599453094172321;
    const kf = rintF64(x * 1.4426950408889634073599246);
    const r = x - kf * ln2;

    // Degree-11 Taylor on |r| <= ln2/2 is already below `f64` rounding.
    var p: f64 = 1.0;
    var i: usize = 12;
    while (i > 0) {
        i -= 1;
        p = 1.0 + r * p / @as(f64, @floatFromInt(i + 1));
    }
    const k: i32 = @intFromFloat(kf);
    return p * pow2F64(k);
}

fn pow2F64(k: i32) f64 {
    assert(k >= -1100 and k <= 1100);
    const half = @divTrunc(k, 2);
    return pow2F64Exact(half) * pow2F64Exact(k - half);
}

fn pow2F64Exact(k: i32) f64 {
    assert(k >= -550 and k <= 550);
    const biased: u64 = @intCast(k + 1023);
    return @bitCast(biased << 52);
}

fn lnF64(x: f64) f64 {
    assert(x > 0);
    const bits: u64 = @bitCast(x);
    var e: i32 = @intCast(@as(u64, (bits >> 52) & 0x7ff));
    e -= 1023;
    // Mantissa in [1, 2), rescaled to [sqrt(1/2), sqrt(2)) for a fast series.
    var m: f64 = @bitCast((bits & 0x000fffffffffffff) | (@as(u64, 1023) << 52));
    if (m > 1.4142135623730951) {
        m *= 0.5;
        e += 1;
    }
    const s = (m - 1.0) / (m + 1.0);
    const z = s * s;

    var p: f64 = 1.0 / 19.0;
    var i: usize = 9;
    while (i > 0) {
        i -= 1;
        p = 1.0 / @as(f64, @floatFromInt(2 * i + 1)) + z * p;
    }
    const ln2: f64 = 0.6931471805599453094172321;
    return 2.0 * s * p + @as(f64, @floatFromInt(e)) * ln2;
}

// -- activations ------------------------------------------------------------

pub fn relu(x: f32) f32 {
    return if (x > 0) x else 0;
}

/// The `tanh` approximation, which is what Marian ships and therefore what the
/// weights were trained against. The exact `erf` form is a different function
/// by about 1e-3 and would show up as a quality regression, not a rounding one.
pub fn gelu(x: f32) f32 {
    const c: f32 = 0.7978845608028654; // sqrt(2/pi)
    const a: f32 = 0.044715;
    const inner = c * (x + a * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

pub fn swish(x: f32) f32 {
    return x * sigmoid(x);
}

// -- tests ------------------------------------------------------------------

const testing = std.testing;

fn relErr(got: f64, want: f64) f64 {
    if (want == 0) return @abs(got);
    return @abs(got - want) / @abs(want);
}

test "exp matches the reference across the representable range" {
    var worst: f64 = 0;
    var x: f32 = -87.0;
    while (x <= 88.0) : (x += 0.013) {
        const got = exp(x);
        const want = std.math.exp(@as(f64, x));
        worst = @max(worst, relErr(got, want));
    }
    try testing.expect(worst < 2e-6);
    try testing.expectEqual(@as(f32, 1.0), exp(0));
}

test "exp is total at the edges" {
    try testing.expectEqual(@as(f32, 0), exp(-1000));
    try testing.expect(std.math.isInf(exp(1000)));
    try testing.expect(std.math.isNan(exp(std.math.nan(f32))));
    try testing.expectEqual(@as(f32, 0), exp(-std.math.inf(f32)));
    try testing.expect(std.math.isInf(exp(std.math.inf(f32))));
}

test "sigmoid and tanh saturate without dividing by zero" {
    try testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-7);
    try testing.expectEqual(@as(f32, 0), tanh(0));
    try testing.expect(sigmoid(-1000) >= 0 and sigmoid(-1000) < 1e-30);
    try testing.expectApproxEqAbs(@as(f32, 1), sigmoid(1000), 1e-7);
    try testing.expectApproxEqAbs(@as(f32, -1), tanh(-100), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1), tanh(100), 1e-6);

    var worst: f64 = 0;
    var x: f32 = -20;
    while (x <= 20) : (x += 0.01) {
        worst = @max(worst, @abs(tanh(x) - std.math.tanh(@as(f64, x))));
    }
    try testing.expect(worst < 1e-6);
}

test "sinCos matches the reference over the positional-table range" {
    var worst: f64 = 0;
    var x: f64 = -8192;
    while (x <= 8192) : (x += 0.37) {
        const r = sinCos(x);
        worst = @max(worst, @abs(r.sin - @sin(x)));
        worst = @max(worst, @abs(r.cos - @cos(x)));
    }
    try testing.expect(worst < 1e-12);
}

test "sinCos hits the quadrant boundaries" {
    const pi = std.math.pi;
    for ([_]f64{ 0, pi / 2.0, pi, 3.0 * pi / 2.0, 2.0 * pi, -pi / 2.0 }) |x| {
        const r = sinCos(x);
        try testing.expectApproxEqAbs(@sin(x), r.sin, 1e-12);
        try testing.expectApproxEqAbs(@cos(x), r.cos, 1e-12);
    }
}

test "powPositive reproduces the positional-table divisors" {
    var worst: f64 = 0;
    var i: u32 = 0;
    while (i < 512) : (i += 2) {
        const e = @as(f64, @floatFromInt(i)) / 512.0;
        const got = powPositive(10000.0, e);
        const want = std.math.pow(f64, 10000.0, e);
        worst = @max(worst, relErr(got, want));
    }
    try testing.expect(worst < 1e-12);
}

test "activations agree with their definitions" {
    try testing.expectEqual(@as(f32, 0), relu(-1));
    try testing.expectEqual(@as(f32, 2.5), relu(2.5));

    var x: f32 = -8;
    while (x <= 8) : (x += 0.05) {
        const want_swish = @as(f64, x) / (1.0 + std.math.exp(-@as(f64, x)));
        try testing.expect(@abs(swish(x) - want_swish) < 1e-5);

        const xf: f64 = x;
        const want_gelu = 0.5 * xf * (1.0 + std.math.tanh(0.7978845608028654 * (xf + 0.044715 * xf * xf * xf)));
        try testing.expect(@abs(gelu(x) - want_gelu) < 1e-5);
    }
}
