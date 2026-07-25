# ADR 0002 — Regions and files beyond the SPEC tables

Status: decided
Date: 2026-07-24
Milestone: M0

## Context

SPEC §4.2 enumerates the arena regions and §5 the repository layout. Both are
right about the parts that matter and silent about a handful of small things a
working implementation needs. Recording them here beats letting them accumulate
as folklore.

## Decision — extra arena regions

Everything in §4.2 exists under its §4.2 name. These are added:

| Region | Size | Why |
|---|---|---|
| `pos_enc` | `max(src,tgt) tokens · d_model · 4` | Sinusoidal positional encodings, generated at model load. See ADR 0003: there is no libm to call at inference time, and precomputing keeps the transcendentals off the hot path entirely. |
| `attn_scores` | `heads · max(src,tgt) tokens · 4` | One score row per head. Prefill streams row by row rather than materialising an `S×S` matrix per head. |
| `vec`, `qvec` | `16 · max(d_model, ffn_dim) · {4,1}` | Named scratch vectors for a single decode step: residual, q, k, v, context, normed, and their quantized forms. |
| `shortlist_scales` | `max_shortlist · 4` | The per-output-channel weight scales gathered alongside `shortlist_rows`. §7 requires one scale per output channel; gathering rows without their scales would be useless. |
| `shortlist_seen` | `max_vocab / 8` | Dedupe bitmap for the union in `shortlist_build`. Sizing it is why `Config` carries `max_vocab`. |
| `src_ids` | `max_src_tokens · 4` | Source token ids. §4.2 sizes the byte-level `io_*` buffers and the lattice but not the ids they produce. |
| `tgt_ids` | `max_tgt_tokens · 4` | Target token ids. Sized by *target* length, deliberately: see the audit note below. |

**`pivot_ids` was removed.** It was carved at `max_src_tokens · 4` and never read. SPEC §10 makes the pivot boundary UTF-8 text rather than tokens — the two models have different vocabularies, so pass one detokenizes fully and pass two tokenizes fresh — which leaves nothing for a shared id buffer to hold. Had it ever been used it would also have been mis-sized: it would have carried pass one's *output*, bounded by `max_tgt_tokens`, in a region sized by `max_src_tokens`.

Measured for the §4.3 example these add ≈0.95 MB on top of the §4.3 estimate,
for ≈7.6 MB of shared scratch against the §14 budget of 16 MB.

## Decision — extra files

| Path | Why |
|---|---|
| `src/runtime.zig` | The instance: arena, slot table, and the body of `fizh_translate`. §5 says `root.zig` is "ABI exports, nothing else", which leaves the thing behind the exports unnamed. |
| `src/model/names.zig` | The comptime perfect hash over tensor names that §6 requires. |
| `src/model/layout.zig` | Slot-internal weight offsets derived from hparams, so a loaded model needs no per-tensor pointer table. |
| `src/kernel/math.zig` | See ADR 0003. |
| `test.zig` | The native test entry point. It sits at the repository root because a Zig module may only import files below its root file's directory, and the suite has to reach both `src/` and `test/`. |
| `tools/tiger.zig`, `tools/wasm_audit.zig` | §12 "enforced in CI where mechanically checkable" and the §3 budgets, as build steps. |

## Consequences

- `Config` is 64 bytes with three reserved words left.
- The §4.2 table remains the specification of the *large* regions; this ADR
  covers the small ones. If a region here grows past a megabyte it belongs in
  the SPEC, not in an ADR.
