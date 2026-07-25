# ADR 0003 — fizh brings its own transcendentals

Status: decided
Date: 2026-07-24
Milestone: M2/M3

## Context

SPEC §3 budgets the ship build at **zero module imports** and says adding one
needs an ADR. On `wasm32-freestanding` there is no libm, so `@exp`, `@sin`,
`@cos`, `@log` and `@pow` lower to calls to `expf`, `sinf`, … which LLD resolves
as wasm imports. Three places need them:

- `softmax` needs `exp`, once per shortlist entry per generated token.
- The GELU and SWISH activations need `tanh` and `sigmoid`.
- Sinusoidal positional encodings need `sin`, `cos` and `pow`.

## Decision

`src/kernel/math.zig` implements `exp`, `sigmoid`, `tanh`, `sinCos` and
`powPositive` from scratch. No imports are added.

- `exp` is Cody-Waite reduction plus the degree-6 Cephes minimax polynomial,
  under 2e-6 relative error across the whole `f32` normal range.
- `tanh` is `(1 - e^-2|x|)/(1 + e^-2|x|)`, not `2·sigmoid(2x) - 1`, because the
  latter cancels catastrophically near zero.
- `sinCos` and `powPositive` run in `f64` and are used only at model load, to
  fill the positional table. At load time the accuracy is free, and `f64` means
  two-part argument reduction covers `|x| < 2^20` without a Payne-Hanek path
  nobody would ever exercise.
- Rounding uses the magic-constant trick rather than `@round`, which is itself a
  `roundf` libcall on wasm.

Every routine is total: NaN in, NaN out; infinities and subnormals have defined
answers rather than assertions.

## Consequences

- Import count stays at zero. `zig build check` enforces it against the real
  artifact by parsing the module's import section.
- I9 gets stronger, not weaker. A host libm is a different function on a
  different browser; a polynomial in the repository is the same function
  everywhere, which is what "bit-exact for a given artifact + build" needs.
- `src/kernel/math.zig` is tested against `std.math` in `f64` at ~1e-6 for the
  `f32` routines and ~1e-12 for the `f64` ones. Those tests are the contract.
- Cost: `exp` is roughly 20 arithmetic operations. Softmax over a 2048-entry
  shortlist is ~40k operations per token, which is under 2% of the projection
  it follows.
