# ADR 0030 — mmap is viable, because repack turns out to be a copy

Status: investigated — viable, not implemented
Date: 2026-07-25
Milestone: M11
Relates: SPEC §4.3 transient peak, ADR 0026

Work order 9 P3 asked whether `mmap`-ing the `.fzm` can replace taking a blob
pointer, and said to check first whether repacking permits it. It does, and the
reason is more useful than the answer.

## What repack actually does

The name suggests a layout transform — a transpose, a tiling, an interleave.
`model/repack.zig` does none of that:

```zig
const pad = stride - m.k;
for (0..m.n) |row| {
    @memcpy(slot[dst..][0..m.k], data[row * m.k ..][0..m.k]);
    if (pad != 0) @memset(slot[dst + m.k ..][0..pad], 0);
}
```

**A row-wise copy that pads each row out to a 64-byte stride.** The bytes are
not reordered. ADR 0009 already established Marian's on-disk int8 is neither
register-tiled nor needing transposition; this is the consequence nobody
followed up.

## And the pad is zero on every model that ships

`stride` rounds `k` up to 64. Across the registry, `k` is either `d_model` or
`ffn_dim`:

| | `d_model` | `ffn_dim` |
|---|---|---|
| tiny | 256 (pad 0) | 1536 (pad 0) |
| base-memory | 384 (pad 0) | 1536 (pad 0) |
| base | 512 (pad 0) | 2048 (pad 0) |

Every one is a multiple of 64, because transformer widths are. **So for every
artifact fizh will ever load, `quantMatrix` is a pure `@memcpy` and the
canonical on-disk layout is byte-identical to the slot layout.**

## Which makes the answer yes, with one change

Tensors are still *relocated*: the file places them at payload offsets the
converter chose, while `SlotLayout.compute` assigns different ones. Mapping the
file and pointing the slot at it needs those to agree.

They can. The converter already emits every tensor 64-byte aligned; having it
emit in slot order at slot offsets makes the payload *be* the slot image. Load
then becomes: validate the header and table, `mmap` the payload `MAP_PRIVATE`,
point the slot at the mapping. Zero copy, and the validation that matters —
scales finite, no `-128` weights, ids in range — stays, because it is read-only
checking rather than rewriting.

`MAP_PRIVATE` is the right mode and needs no writes: nothing mutates weights
after load. The only writer would be the pad, and the pad is empty.

## What it is worth

ADR 0026 and SPEC §4.3 measured the transient: `fizh_model_load` repacks, so
the blob and its copy are both resident and the peak is `arena + largest blob`.
On the Android run that was 86 MiB peak against a 53.5 MiB arena.

With mmap on native, the weight slot *is* the file: no host copy, no repack
destination, and the pages are file-backed so the OS can evict them under
pressure rather than counting them against an anonymous allocation. For the
§4.3 pivot case that projects **~122 MB resident and ~178 MB transient down to
roughly the arena's non-weight remainder plus whatever the OS keeps resident**.

## Why it is not implemented here

**It is native-only and the target is the browser.** wasm has no `mmap`; a
browser host must fetch bytes and hand them over, so the wasm path keeps the
staging buffer of ADR 0026 unchanged. That was the instruction — *native-only,
behind a build flag, the wasm path does not change* — and it means the work
buys nothing for the platform the product ships on.

**It is a format change.** Emitting tensors at slot offsets means the converter
must run `SlotLayout.compute`, which currently lives in Zig and would need a
Python twin or an exported entry point. `zig build convert-selftest` exists to
stop those two drifting and would have to cover this too. That is a version
bump and a real chunk of work for a native-only win.

**Nothing measured needs it.** The transient peak is documented as a hard
requirement, hosts are told to stage through one buffer and load one model at a
time, and no run has yet hit an OOM.

Recorded rather than built, so the next person does not re-derive that repack is
a copy — which is the whole finding, and took ten minutes to establish once
somebody read the function instead of its name.
