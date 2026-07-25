# ADR 0019 — Coverage is not two pairs

Status: decided
Date: 2026-07-25
Milestone: M9 (coverage)

## Context

fizh was developed against `es-en` and `en-de` and declared to translate real
Bergamot models. Firefox's registry has **105 pairs**, and ADR 0015 had already
shown artifacts are not interchangeable: a token convention that differed
between those two directions cost 1.5 chrF++ and survived three rounds of
review.

`tools/eval/sweep.py` walks the registry. Per pair: fetch, convert, load
through the real loader, translate a fixed 100-segment FLORES slice, and score
against `bergamot-translator` on the same slice with `--per-line`. A converter
failure is a hard stop for that pair and is reported, never skipped.

## What it found

Five defects, none of which either development pair could have exposed.

### 1. The fetcher selected pre-release models — 21 of 105 pairs

`min(models, key=size)` was meant to prefer the tiny student architecture over
the base one beside it. It also prefers an *alpha* over the release it
preceded, because alphas are often smaller: `cs-en` v1.0a1 is 295 bytes under
v1.0.

The cost is not cosmetic. Through fizh, `cs-en` v1.0a1 renders a FLORES segment
as

> Dr. Ehud Ur, profesor medicín on Dalhousie's universia in Halifax in Novém
> Skotsek, and the equal foot of the klina and science divas of the
> Canean-diabetic asocic asocic has pronated that the drudsts are d'etat

and v1.0 as

> Dr. Ehud Ur, a professor of medicine at Dalhousie University in Halifax, New
> Scotland, and chairman of the clinical and scientific division of the
> Canadian Diabetic Association, pointed out that the research is only in its
> early years.

−30.94 chrF++ becomes +0.28. Same code, same converter, same tokenizer.

**Finding it meant ruling out everything structural first:** tensor sets match
(195 each, identical names), the embedded `model.yml` matches, header
hyper-parameters match, dtypes match (`0x4101`/`0x404`/`0x101`), quantization
scales are the same order, the tokenizer round-trips, the encoder's mean
pairwise cosine is a healthy 0.75, and every weight dequantizes to within 1e-7
of Marian's own.

Teacher forcing localized it, as it did in ADR 0015 — but this time by the
*shape* of the result rather than its location: 0.20 next-token agreement
**uniformly across position**, where `es-en` is 0.97. A defect at one point
looks like ADR 0015's table. A model that disagrees everywhere means the
weights are wrong, and the weights were a faithful copy of the file. So the
file was wrong.

Both `fetch-model.sh` and the sweep now discard versions containing a letter
before taking the smallest.

### 2. Hardcoded configs rejected a whole architecture

Most pairs are SPEC §4.3's `d_model=256, n_dec=2`. Arabic, Basque and Galician
are `384, 4`. `tools/translate.zig` and `tools/fzm_load.zig` each carried a
config for the former and returned `model_too_large` for the latter — even
though `abi.limits` allows `d_model` 4096 and 32 decoder layers, and every
kernel is generic over both. **The limit was the tool's, not the runtime's.**

`format.peekHParams` now reads the header without validating it against a
config, so a host can size an arena *from* an artifact instead of rejecting one
that does not fit a guess. `en-gl` translates "Hello there, how are you?" as
"Ola aí, como estás?".

### 3. Two shortlist files are shorter than the vocabulary

`en-es` ships `wordToOffset` at exactly `vocab_size`, `en-fa` at
`vocab_size - 2`, against the usual `vocab_size + 1`. The trailing source words
simply have no candidate list, which is usable — they contribute nothing to the
union. The converter pads the sentinel and now rejects only a shortlist
*longer* than the vocabulary.

The old check's message was `wordToOffsetSize 32000 but the vocabulary has
32000 pieces`, which is a self-contradicting way to report an off-by-one.

### 4. The normalizer was gated on its name

ADR 0017 refused anything not called `nmt_nfkc`. `en-bs` and `en-sr` declare
`user_defined`, which turns out to be the same thing wearing a different label:
the same kind of compiled darts-clone table and the same structural flags. The
gate now reads `add_dummy_prefix`, `remove_extra_whitespaces` and
`escape_whitespaces` and refuses on those, because those are what
`src/tok/unigram.zig` assumes. The name records how a table was built, not what
it does.

### 5. `fzm-load` answered the wrong question

Bare `fzm-load x.fzm` used its own hardcoded ceilings, so it reported
`LoadFailed` for every 384-wide pair that `translate` handled. It answered "does
this fit the shape I guessed" where the sweep was asking "does this load".

## The result

See the README for the matrix. The distribution is what matters: most pairs sit
within a chrF++ point of `bergamot-translator`, and the pairs that do not are
named rather than averaged away.

## The lesson

**Every one of these five is a limit that lived in the harness or the tooling
rather than in the runtime, and every one presented as a property of a
language.** "fizh is bad at Czech" was a wrong sentence about a real
measurement. The sweep's value is not that it found bugs; it is that it made
each one a line in a table next to a hundred that worked, which is what turned
"bad at Czech" into "selected an alpha for 21 pairs".

Two pairs cannot do that. A pair that fails loudly is a fine outcome; a pair
that translates badly and quietly is the one this exists to catch, and four of
the five above were quiet.
