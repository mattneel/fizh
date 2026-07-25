# ADR 0017 — Ship the model's own normalizer, don't reimplement NFKC

Status: decided
Date: 2026-07-25
Milestone: M8 (correctness)
Amends: SPEC §1 scope, SPEC §4.2, SPEC §6

## Context

fizh's tokenizer normalization was whitespace only: collapse runs of ASCII
whitespace, prepend the word-boundary marker. SentencePiece models declare a
normalizer, and both fetched Bergamot artifacts declare `nmt_nfkc`:

```
field 3 (237552 bytes)
    sub 1: nmt_nfkc
    sub 2: <237538 bytes>
```

That second field is a `precompiled_charsmap`: a darts-clone double-array trie
over UTF-8 byte sequences followed by a pool of NUL-terminated replacements,
matched longest-prefix. It is the exact rewrite table the model was trained
with — 44,288 trie units and 14,791 replacement strings.

## Decision

**Read the table out of the artifact and interpret it. Do not implement
Unicode normalization.**

`tools/charsmap.py` parses the `.spm` and is the reference implementation.
`tools/bergamot.py` copies the blob into the artifact as `tok.charsmap`, and
refuses by name any normalizer that is not `nmt_nfkc` rather than silently
applying the wrong rules. `src/tok/charsmap.zig` walks the trie at runtime.

The alternative was implementing NFKC in Zig. That is the wrong shape of
problem: Unicode normalization has a specification, several revisions of it,
and a large table either way — and the model does not care about any of them.
It cares about the 237 KB of rules baked into its own vocabulary. An NFKC
implementation that is *correct* and disagrees with the model's table is worse
than a table interpreter that is bug-compatible with it, because the model's
segmentation was trained on the table.

This is SPEC §6's M2 discipline applied to the tokenizer, and the same lesson
as ADR 0008 and ADR 0015: **the artifact is the specification.**

## What it changes

Order matters: the charsmap runs on raw source bytes, then whitespace handling
and the boundary marker. Running it the other way would apply rewrite rules to
text that already carries markers. That needs two buffers — the rewrite can
grow the text and the whitespace pass prepends a byte, so neither is safe in
place — so `arena.zig` gains `tok_raw` alongside `tok_norm`.

Measured on the corpora, `nmt_nfkc` fires rarely:

| corpus | lines it changes |
|---|---|
| FLORES es, 500 | 2 (0.4%) |
| FLORES en, 500 | 1 (0.2%) |
| chat register, 40 | 0 (0.0%) |

The three real cases are `…` → `...` and `¾` → `3⁄4` (twice). On FLORES chrF++
this is worth +0.01 on es→en and 0.00 on en→de — nothing, and **that is the
expected result, not a disappointment.** It was never a candidate for the
residual; ADR 0015 was. It was implemented because user input is not FLORES:
ligatures, fullwidth Latin, non-breaking spaces and fraction characters are
what people paste into a messaging app, and every one of them tokenizes wrong
without this.

Byte-identity with `bergamot-translator` moved 222 → 223 of 500 on es→en:
every line the table touched now matches the reference exactly. Three probe
cases — fractions, ellipsis, fullwidth — are recorded in
`golden_translations.tsv` and were verified byte-identical against
`bergamot-translator` when recorded.

## Bounds, because there is no allocator

The worst rule in the shipped table expands three bytes to thirty-three:
U+FDFA, the Arabic ligature *sallallahou alayhe wasallam*. An adversarial input
can therefore grow 11x, which is not a size the arena can pre-commit to for
every message.

`normalize` returns `error.OutTooSmall` rather than truncating, and the pass
turns that into `src_too_long` — which is what it is. After normalization the
text no longer fits, and a half-normalized string is a wrong string rather than
a short one. In practice `max_src_bytes` is 4096 against messages of a few
hundred bytes, so the bound is never approached; when it is, the host raises
`max_src_bytes`.

## Cost

237,538 bytes per direction. Weights go from 19.00 MB to 19.23 MB against the
SPEC §14 budget of 20 MB, which is the tightest any budget now sits. Shared
scratch goes from 6.76 MB to 6.77 MB for `tok_raw`. Cold start, p50 and p99 are
unchanged within noise — the trie walk is a handful of array reads per
character and does not touch the decode loop at all.

If a future artifact needs the headroom, the table is the first thing to drop:
it is optional in the format, and an artifact without `tok.charsmap` falls back
to the whitespace-only path fizh had before. That fallback is exercised by the
synthetic artifacts, which carry no charsmap.
