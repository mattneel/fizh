# ADR 0001 — The arena owns the weight slots

Status: decided
Date: 2026-07-24
Milestone: M0

## Context

SPEC §4 lists `weights[slot]` as an arena region sized "from artifact header",
and §4.1 says the weight slots are the only additive region. But §4 step 1 has
the host call `fizh_arena_bytes(cfg)` *before* any artifact exists, and §9 hands
`fizh_model_load` a `[*]const u8` blob. Read literally, the arena cannot be
sized for weights it has not seen, and the loader cannot repack into a blob it
promised not to write.

Something has to give.

## Decision

`fizh_arena_bytes` sizes and `fizh_init` carves `max_models` weight slots of
`max_model_bytes` each. `Config` grows two fields to say so:

    max_models        how many directions may be resident
    max_model_bytes   per-slot ceiling on the repacked weights

`fizh_model_load` copies and repacks the host's blob **into** its slot. The blob
is borrowed for the duration of the call and never after, so the host is free to
reuse or drop that memory the moment `fizh_model_load` returns.

SPEC §4 step 2 — "memory.grow to cover n + all model blobs" — still reads
correctly: `n` is the arena including its slots, and the blobs are the staging
buffers the host downloads into.

## Consequences

- `fizh_arena_bytes` stays a pure function of the config, which is what makes
  two-phase init work at all.
- §4.1 still holds exactly: adding a direction adds one slot and nothing else.
  `test/runtime_test.zig` and `src/arena.zig` both assert it.
- Peak host memory during a load is `arena + one blob`, not `arena + all blobs`,
  as long as the host loads directions one at a time.
- A model larger than `max_model_bytes` is `model_too_large`, a validation
  error. The host picks the ceiling; a wrong guess is recoverable.
- The repack is a copy, so the canonical `[N][K]` layout in the artifact and the
  register-tiled layout in the slot can differ freely (SPEC §6).
