# ADR 0025 — Relaxed SIMD is unreachable from Zig, and the gap is real

Status: decided
Date: 2026-07-25
Milestone: M11
Amends: ADR 0006, ADR 0024

Four measurements, each closing a question that had been answered by reasoning.

## 1. `+relaxed_simd` *pessimized* float code

ADR 0024 found no relaxed opcode in either artifact and made `relaxed/` alias
`vector/`. The mechanism is worth recording on its own, because it is a trap
anyone enabling a CPU feature can walk into:

LLVM sees `relaxed_madd` in the feature set and marks FMA **legal**. Legal FMA
lets it contract `a*b + c` into an `llvm.fma` node. It then cannot emit
`relaxed_madd` for a *correctly rounded* `llvm.fma` — the relaxed instruction is
explicitly allowed to differ — so it expands to a software FMA sequence.

**Enabling the feature made float code slower that nobody asked it to change.**
Not the code using `@mulAdd`; any `a*b + c` LLVM chose to contract. That is why
the regression scaled with per-token work rather than appearing as a constant.

## 2. There is no path from Zig 0.16 to the dot instruction

ADR 0006's entire justification for a second artifact was
`i32x4.relaxed_dot_i8x16_i7x16_add_s`, and SPEC §7's `[-127, 127]` clamp was
designed around its `i7` operand. So: can Zig emit it?

Zig exposes no dot-product builtin and no arbitrary LLVM intrinsics, which
leaves pattern matching. Written idiomatically — widen `i8`→`i16`, multiply,
pairwise-add to `i32`, add an accumulator — with `+simd128+relaxed_simd`:

| probe | emitted |
|---|---|
| relaxed dot shape | `i16x8.extmul_low/high_i8x16_s`, `i16x8.mul` — **no relaxed opcode** |
| plain `i32x4.dot_i16x8_s` shape | `i32x4.extend_low/high_i16x8_s`, `i16x8.mul` — **not even 0xba** |

Neither the relaxed dot nor the *non-relaxed* dot is selectable. So
**"the instruction may land" is false**: the compiler has no way to choose it
from this source, and waiting changes nothing.

### Consequence

The relaxed artifact, the probe module, `zig build check --require-relaxed`, and
the host-side feature detection are maintenance surface buying **nothing**, with
no path to buying anything from Zig source. The two artifacts are byte-identical
today.

This ADR does not delete them, because doing so touches SPEC §3, ADR 0006, the
build, the probe, the host and the benchmark page — a spec-level decision rather
than a cleanup. **The recommendation is to remove them** and to keep the `[-127,
127]` clamp, which earns its keep for a different reason: it makes `ref/` and
`vector/` bit-exact, and it is what lets a widened integer lane count stay
bit-exact across targets (ADR 0023).

## 3. bergamot's wasm is single-threaded

The 3x per-token gap only means something if both engines get one core. Read
from the artifact rather than assumed:

- `env.memory` is imported with flags `0x01` — **`shared` bit not set**. A
  threaded Emscripten build requires shared memory.
- initial 256 pages (16 MiB), maximum 32768 pages (2 GiB).
- **Zero** occurrences of `pthread`, `PThread`, `SharedArrayBuffer` or
  `Atomics.` in the 82 KB of glue.

**Single-threaded.** So the 3x is real per-core arithmetic, not 8 cores against
1, and there is genuine kernel work behind it. intgemm is hand-tuned
register-blocked x86 with prepared-B layouts; a portable `@Vector` kernel a few
times off that is a fair standing, and closing it is optimization rather than
correction.

## 4. The cost is quadratic, and a linear fit misleads at chat length

ADR 0024 corrected a two-point fit on a words axis. Re-fitting the six-point
token-axis data without an intercept (zero tokens is zero work) shows the
residual was curvature:

| | fit | max error |
|---|---|---|
| encoder | **0.4307·n + 0.000591·n²** | **0.88%** |
| total | 0.5804·n + 0.000891·n² | 11% |
| total, linear with intercept | 0.7386·n − 4.06 | 2.45% |

That is attention: `O(src_len²·d)` in the encoder, and cross-attention makes the
decoder grow superlinearly too. The n² term is 18% of encoder cost at 164
tokens.

The linear-with-intercept fits the *measured range* better and extrapolates
badly, which is the whole problem:

| tokens | quadratic | linear fit | error |
|---|---|---|---|
| 12 | 7.09 ms | 4.81 ms | **−32%** |
| 24 | 14.44 ms | 13.67 ms | −5% |
| 40 | 24.64 ms | 25.49 ms | +3% |

**A budget derived from the 164-token case carries curvature that does not
exist at chat lengths.** §14's short-message row is measured at 12 tokens
directly and is therefore fine; nothing else should be extrapolated across that
range.

## 5. The memory metric, defined

ADR 0024's table claimed fizh wins peak heap ~6x. The metric now has a
definition, because it needed one: **`WebAssembly.Memory.buffer.byteLength` at
its maximum during the run**, measured identically for both engines.

It is *linear memory demanded*, **not** resident set size. Growth commits pages
but the OS need not keep them resident. Neither engine reserves up front —
bergamot starts at 16 MiB against a 2 GiB maximum — so a large figure is memory
actually grown into, not a reservation policy artifact. Read it as "how much
address space each engine asked for".
