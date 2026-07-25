# ADR 0010 — Special ids come from the vocabulary; shortlist ids are `u16`

Status: accepted
Date: 2026-07-25
Milestone: M2/M8

Two small decisions forced by real artifacts. Both were "obviously fine"
assumptions until a second language pair existed.

## 1. Special-token ids are not constants

The converter took `--eos 0 --unk 1` as defaults, because that is what the
`es-en` vocabulary uses. The `en-de` vocabulary does not:

| pair | id 0 | id 1 | id 2 | id 3 |
|---|---|---|---|---|
| `esen` | `</s>` | `<unk>` | `▁de` | `,` |
| `ende` | `<unk>` | `<s>` | `</s>` | `<blank>` |

With `eos_id = 0` against `en-de`, the decoder waits for a token that means
`<unk>`, never sees it, and runs to `max_length_factor · src_len` every time.
The output looked like this:

    Schlafen Sie die schwarze Katze schläft auf dem Tisch.en. . schläft die
    schwarze Katze auf dem Tisch. Die schwarze Katze schl

Recognisable German, then repetition — the signature of a decoder that cannot
stop. Notably it is *not* the signature of a wrong `eos_id`, which is what made
it worth writing down: the failure looks like a quality problem, and it is a
configuration problem.

**Decision.** `tools/bergamot.py` looks `</s>`, `<unk>` and `<s>` up by piece
text and warns if one is missing. `--eos`/`--unk` remain as fallbacks only.
Marian starts the decoder with end-of-sequence when there is no explicit
beginning-of-sequence, so `bos_id = eos_id`.

## 2. Shortlist target ids are `u16`

Bergamot's `lex.50.50.esen.s2t.bin` holds 901,208 candidate ids — "best 50 per
source piece". At `u32` that is 3.44 MB, and the slot came to **20.28 MB**
against SPEC §14's 20 MB budget for a direction.

The obvious move was to trim to `--shortlist-best 40`. It fits, and it costs
quality: chrF++ on the chat corpus was **33.05** trimmed versus **41.72** with
Bergamot's full list. Nearly nine points of chrF++ to save 280 KB is a bad
trade, and it would have been invisible without T4.

A vocabulary of 32,000 needs 15 bits. `u16` halves the shortlist to 1.72 MB and
brings the whole direction to **19.01 MB** — under budget, with the full list.

**Decision.** `sl.targets` is `DType.u16`. `format.zig` rejects any artifact
whose `vocab_size` exceeds `maxInt(u16) + 1`, so the narrowing can never
silently truncate an id.

This is the only place fizh varies a width, and it does not touch I4: I4 is
about *weights and arithmetic* being int8 with no mixed precision. A shortlist
id is an index, not a number the kernels multiply.

## Consequence

Both decisions exist because a second language pair was tried. One model is not
a test of a converter; it is a test of one model.
