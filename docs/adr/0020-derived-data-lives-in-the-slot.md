# ADR 0020 — Per-model derived data lives in the slot

Status: decided
Date: 2026-07-25
Milestone: M9 (coverage)
Amends: SPEC §4.1, SPEC §4.2

## Context

ADR 0019 records the defect: `pos_enc` was one shared arena region, filled by
every `fizh_model_load` at that model's `d_model`. Sinusoidal encodings depend
on `d_model` — a 256-wide table is not a prefix of a 384-wide one — so loading
a 384 model after a 256 model left the narrower one reading a table laid out
for the wider. Fluent output, wrong words, no assertion.

The instance fix was a refill in `runPass` when the active width differed. That
worked and it was the wrong shape of fix, because nothing stops the next
derived table from being carved in shared scratch the same way.

## The defect is in the spec, not the code

SPEC §4.1 said: *scratch is sized by the max over loaded models, not the sum.*
That is correct about **sizing** and silent about **content**. It never
distinguished two kinds of region:

| kind | lifetime | may be shared |
|---|---|---|
| scratch | one pass; overwritten by the next | yes |
| per-model derived data | one model; computed at load from *its* parameters | no |

`pos_enc` is the second kind living in a region built for the first, so load
order decided correctness. Every other region in §4.2 happens to be the first
kind, which is why the rule had never been needed and why nothing enforced it.

## Decision

**Per-model derived data is carved in that model's slot.**

`model/layout.SlotLayout` gains `pos_enc`, sized `max(max_src_tokens,
max_tgt_tokens) · d_model · 4` from the host's config, and `format.loadInner`
fills it through `Model.posEnc(slot)`. `arena.Layout` loses its `pos_enc`
region entirely.

This makes the category **unrepresentable rather than auditable**: there is no
shared positional table left to write the wrong thing into. `runtime.buildCtx`
additionally asserts that the view it hands a pass lies inside that pass's own
slot, so a future refactor that reintroduces sharing fails at the boundary
rather than in the output.

The `pos_d` refill and its cache are gone.

## What was checked alongside it

`tok.nonbreaking` is the other obvious candidate — it is per-*source-language*
and a pivot has two source languages, which is the same shape. It was already
safe: `Model.prefixes(slot)` indexes into the model's own weight slot, and
`buildCtx` passes the active slot. It is a per-model region that was never in
shared scratch.

Auditing `loadInner` settles the rest: the only thing it wrote outside the slot
was `fillPositional`. `pos_enc` was the category's sole member.

## Cost

None, on net. The table moves from scratch into each slot:

| | before | after |
|---|---|---|
| shared scratch | 7.16 MB | **6.41 MB** |
| weights, per `d=256` direction | 19.23 MB | **19.98 MB** |

The same bytes, now owned. Two resident directions pay for two tables, which is
correct — they need two.

## How live was it

Only pivots crossing the `d=256`/`d=384` boundary were affected, which sounds
narrow and is not: of the 99 usable pairs, 76 are `d=256` and 23 are `d=384`.
A non-English pivot picks two of those at whatever widths the two languages
happen to ship, so a large share of real pivots crossed it. Measured on
`cs→en→ru`:

    before   Вскоре, без ранее, без первого, без «четвертых, доставленных»
    after    «Теперь у нас есть четырёхмесячные мыши без диабета, которые
             раньше имели его», — добавил он.

It was live, and nothing in the suite could see it: the pivot path had a
`can_translate` test and no output test until ADR 0019.

## The general form

*If two loaded models would write different bytes into a region, it is
derived, and it belongs in a slot.* Scratch does not care which model wrote
last, because the next pass overwrites it before reading. That is the whole
test, and it is cheap enough to apply to every new region.
