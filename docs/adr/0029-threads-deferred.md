# ADR 0029 — Threads: the design, the cost, and why not now

Status: deferred — design recorded so the decision is not rediscovered
Date: 2026-07-25
Milestone: M11
Relates: SPEC §4.1, invariant I3

Work order 9 P4 asked for the design and explicitly not the implementation.
This is that.

## What would be parallelised

The encoder, and only the encoder. It is `qgemm` at `M = src_len`: rows are
independent, the profiler puts it at **72%** of a call, and it is the one place
in fizh with real data parallelism.

The decoder is not a candidate. It is `qgemv` at `M = 1`, strictly sequential
across steps because each token's input is the previous token's output, and it
is 27% of the call. Amdahl caps any threading win at ~3.6× even with a free
encoder, and realistically well under 2×.

## What stops it today

**The arena assumes one thread.** `enc_states`, `act_a`, `act_b`, `qact`,
`attn_work`, `attn_scores` and the `vec`/`qvec` slots are shared, mutable, and
written by every layer of every pass. Two threads in `encoder.run` would race
on all of them.

**I3 forbids the allocator a thread pool wants.** No `std.mem.Allocator` in
`src/`, so a pool cannot allocate per-thread scratch on demand; it has to be
carved at init like everything else.

## The design

Per-thread scratch regions, carved at init, arena sized by thread count.

`abi.Config` gains `max_threads`. `arena.Layout.compute` multiplies the
per-pass scratch regions by it, and `pass.Ctx` takes a thread index that
selects a stripe. Weights, `pos_enc`, `xattn_kv` and the `io_*` buffers stay
shared — they are read-only during a pass or per-model.

`encoder.run` splits its row range across threads and joins per layer, because
a layer's output is the next layer's input. That is `n_enc_layers` joins per
pass — six for the Bergamot students.

## The cost

**Memory.** Per-pass scratch is ~6.4 MB at the §4.3 config, and nearly all of it
is per-pass rather than per-model. Four threads is ~19 MB more, on a budget
where §4.3's two-direction pivot case is already ~122 MB resident. On the device
this is for, that is the expensive part.

**The join.** Six barriers per pass against an encoder that costs 15–75 ms
total. At 164 tokens the per-layer slice is ~12 ms and a barrier is cheap
against it; at 12 tokens the encoder is ~3 ms total, so six barriers plausibly
cost more than they save. Threading would need a length threshold, which is a
policy knob in a library that currently has none.

**The invariant.** I3 survives — regions are still carved at init — but SPEC
§4.1's "scratch is shared across a pivot" acquires an exception, and ADR 0020's
rule (per-model derived data lives in the slot) acquires a sibling: per-*thread*
scratch lives in a stripe. Two rules about who owns a region instead of one.

**Not available where it matters.** On the web, threads need
`SharedArrayBuffer`, which needs COOP/COEP headers, which a library cannot
impose on its host. bergamot's own wasm build is single-threaded for exactly
this reason (ADR 0025). So this buys nothing for the browser target, which is
the target.

## Why not now

The measurement removed the motive. ADR 0028 has fizh **1.44–1.71× faster than
bergamot on ARM** and at parity on x86, single-threaded on both sides. Threading
would win a race fizh is already winning, on the one platform where it cannot
run, at the cost of ~19 MB on the platform where memory is scarcest.

The honest trigger to revisit: a native embedding where memory is cheap, latency
matters more than it does in a messaging app, and a measurement shows the
encoder dominating a workload longer than a chat message. None of those is true
today.
