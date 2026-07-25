# ADR 0011 — fizh splits sentences

Status: decided
Date: 2026-07-25
Milestone: M5 (rework)
Amends: SPEC §1 scope, SPEC §5 layout, SPEC §9

## Context

Bergamot's models are trained on single sentences and emit `</s>` at the end of
one. `bergamot-translator` therefore splits its input with ssplit-cpp — a
Moses-derived splitter with per-language non-breaking prefix lists — translates
each sentence independently, and rejoins.

fizh handed the whole input to the model. SPEC never mentions segmentation, so
neither did the implementation.

## What it cost, measured

FLORES devtest is one sentence per line, so the calibration in P0 could not see
this at all: it came back at parity and said nothing. Against 100 paragraphs of
four FLORES sentences each, scored against `bergamot-translator` on the same
input:

| direction | fizh, no splitter | fizh, splitting | bergamot |
|---|---|---|---|
| en→de | **40.48** | 62.37 | 64.66 |
| es→en | **51.64** | 56.46 | 57.06 |

Twenty-four chrF++ on en→de. The mechanism is visible in the output lengths:
fizh's were 0.642× bergamot's and more than 20% short on **95 of 100**
paragraphs. It was translating the first sentence and stopping, exactly as a
sentence-trained model does.

After splitting, the length ratio is 0.988 and the residual gap matches the
gap already present on single sentences — that is, it is not segmentation.

## Decision

`src/tok/ssplit.zig` implements the Moses algorithm: find a terminator, then
look for a reason *not* to break there.

- `...` is one terminator, not three.
- Closing quotes and brackets belong to the sentence that is ending.
- A terminator not followed by whitespace is inside a token — `example.com`,
  `3.14`.
- The next sentence must *start* like one: an ASCII capital, a digit, an
  opening quote, or any non-ASCII lead byte. That last case is deliberately
  permissive — `¿`, `«` and every accented capital live there, and case is not
  knowable without a Unicode table this project does not carry.
- A single capital before the period is an initial: `J. R. R. Tolkien`.
- A non-breaking prefix suppresses the break: `Dr.`, `Sr.`, `z.B.`

`graph/pass.zig` splits, translates each sentence, and joins with a single
space. The SSRU cell is zeroed per *sentence*, not per pass — a recurrent state
carried across a sentence boundary would make each sentence depend on the last.

**Language knowledge lives in the converter, not the runtime.**
`tools/nonbreaking.py` holds the per-language lists and the converter ships them
as `tok.nonbreaking`, a NUL-separated lowercase blob (484 bytes for `es`).
`ssplit.zig` takes whatever list the artifact carries and has no opinion about
which language it is looking at. The tensor is optional: an artifact without one
simply loses that rule.

## Behaviour over `max_src_tokens`, which was undefined

A single sentence longer than `max_src_tokens` used to fail the entire
translation with `src_too_long`. Since the splitter cannot help with a sentence
that has no terminators, that made the behaviour of long input undefined in
practice — a 1133-byte unpunctuated paragraph returned nothing at all.

`graph/pass.zig` now hard-wraps such a sentence at word boundaries, which is
what bergamot-translator's `max-length-break` does. The wrap only happens after
a sentence has actually failed to fit, so a sentence that fits is never touched.
One byte is at minimum one token, so a prefix of `max_src_tokens` bytes is
always safe without a second tokenization pass.

## Consequences

- SPEC §1's "In:" list gains sentence segmentation. It was an omission, not a
  deliberate exclusion.
- SPEC §5 gains `tok/ssplit.zig`; SPEC §4.2 gains a `sent_spans` region, sized
  at `max_src_bytes / 2` spans because a sentence needs at least a terminator
  and a separator.
- SPEC §9's "one call does the work" gets stronger, not weaker: the host still
  passes a blob and still gets a blob.
- **A corpus of single sentences cannot measure this.** Any future quality work
  needs a multi-sentence corpus alongside FLORES, reported separately — the two
  numbers move independently, and averaging them would have hidden a
  twenty-four-point defect behind a parity result.
