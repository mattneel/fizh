# ADR 0018 — The shortlist is not free, and the reference was not using one

Status: decided
Date: 2026-07-25
Milestone: M8 (calibration)

## What was asked

Ablate the `lex.50.50` shortlist at n=500: full-vocabulary projection against
fizh's candidate set, same decoder, same inputs. Gap closes → the candidate-set
construction differs from bergamot's. Gap unchanged → ruled out permanently.

Run with `tools/eval/ablate_shortlist.py`, which exists so nobody re-runs this
by hand. It takes about fifteen minutes per direction.

## Result: the gap does not close, and it is not flat either

    es->en   n=500  shortlist 54.45  full-vocab 54.47  delta +0.02  identical 456/500 (91.2%)
    en->de   n=500  shortlist 62.48  full-vocab 62.76  delta +0.27  identical 380/500 (76.0%)

es→en is ruled out: two hundredths of a chrF++ point, nine segments in ten
byte-identical. en→de is not. **+0.27 against a remaining gap to bergamot of
−0.33** — on this corpus the shortlist is most of what is left.

## What it is not

**Not the `max_shortlist` cap.** It binds on 8 of 500 en→de segments (1.6%);
mean set size is 1205 against a cap of 2048. Removing the cap entirely moves
the mean by two.

**Not the frequent-word set.** fizh includes the first `firstNum`=50 vocabulary
ids, as Marian does. Growing that to 8000 — a seven-fold larger shortlist, mean
size 8518 — only moves the miss rate from 0.95% to 0.31%:

| frequent set | en→de reference tokens missing | segments affected |
|---|---|---|
| 50 | 0.95% | 112/500 |
| 200 | 0.87% | 101/500 |
| 1000 | 0.66% | 84/500 |
| 4000 | 0.47% | 64/500 |
| 8000 | 0.31% | 46/500 |

The missing tokens are spread through the vocabulary (median id 4136), not
concentrated at the frequent end.

## What it is

The per-source-token candidate lists. Measured directly: 1.10% of
`bergamot-translator`'s own en→de output tokens are absent from the shortlist
fizh builds for the same source, across 114 of 500 segments — which matches the
24% of segments whose output the shortlist changes. fizh cannot reproduce a
reference token it never scores.

The absentees are mostly subword continuation pieces — `m`, `st`, `uf`, `lös`,
`od` — the fragments needed to spell a word the candidate set does not carry
whole. German needs more of them, which is why es→en barely notices.

## Resolved: it was the harness, not the candidate lists

**The key was active.** Diffed rather than reasoned about: the same
configuration run twice differs on 0 of 500 lines, and with-key against
without-key differs on 6. Non-empty, so `shortlist:` is honoured.

**fizh's candidate lists are exact.** The artifact's `sl.offsets` and per-source
list lengths are byte-identical to `lex.50.50` — same 921,897 entries, same mean
28.8, same max 50. Nothing is dropped in conversion.

**Marian builds one shortlist per mini-batch, not per sentence.** The candidate
set is the union over every source token in the batch, so batch size changes
what the decoder can emit. The harness fed all 500 FLORES segments as a single
batch, giving the reference a candidate set unioned over 16,000 source tokens
where fizh builds one per sentence — because fizh translates one message at a
time and has no batch to union over.

Re-running the reference with `--per-line`, one sentence per batch, which is
the only configuration comparable to fizh:

| en→de reference | chrF++ | vs fizh | byte-identical | its tokens outside fizh's shortlist |
|---|---|---|---|---|
| batch = 500 sentences | 62.79 | −0.33 | 231/500 | 1.10% |
| batch = 1 sentence | 62.62 | **−0.16** | **274/500** | **0.28%** |

es→en moves 0.28% → 0.04% missing and 223 → 237 identical, on a delta already
inside the noise.

So most of the "+0.27 shortlist gap" was the harness comparing a per-sentence
shortlist against a 500-sentence batch union. The ablation that produced it
(`ablate_shortlist.py`, fizh-oracle against itself) was sound and is unchanged:
fizh's own shortlist does cost it 0.27 against its own full-vocabulary
projection. What was wrong was calling that a difference *from bergamot*.
Bergamot pays the same tax at the same batch size.

**The residual is 0.28% of tokens on en→de.** Not zero, and not worth chasing:
the direction is −0.16 with an interval of [−0.47, +0.14].

## The thing found on the way, which is worse

**`tools/eval/reference_engine.mjs` never enabled bergamot's shortlist.**
`bergamot-translator` builds a shortlist generator only when
`options->hasAndNotEmpty("shortlist")`; the `AlignedMemory` handed to
`TranslationModel` is otherwise ignored. The config had no such key, so every
number reported against this reference — in this ADR, in ADR 0015, in the
README — compares fizh *with* a shortlist against bergamot at *full-vocabulary
projection*.

That makes the reported deltas conservative rather than wrong: fizh is being
compared against a stronger configuration than Firefox ships. But it is not the
comparison that was claimed, and it is exactly the class of error ADR 0015
records — a configuration assumed rather than read.

Adding the key changed 6 of 500 en→de lines and 2 of 500 es→en, and moved no
score (62.79 → 62.79, 54.49 → 54.48). **That is not yet a settled result.**
Either bergamot's shortlist is near-lossless where fizh's costs 0.27 — which
would confirm a construction difference — or the argument form passed
(`- dummy` / `- false`, standing in for a path already supplied as memory) is
not being honoured the way Firefox's is. Marian parses `shortlist` as a vector
whose later elements can override `firstNum` and `bestNum`, so a wrong form can
silently produce a different set rather than an error.

That is now resolved — see above. The key is active, and the remaining
difference was batch size rather than construction.

## Decision

**Nothing changes in the runtime.** fizh's shortlist construction is exact
against `lex.50.50`, and its per-sentence scope is correct for what fizh is: a
library that translates one message at a time. A batch union is not available
to it and would not be desirable — it would make one message's output depend on
what else was in the buffer.

`reference_engine.mjs` gains `--per-line`, and comparisons against the
reference use it. Anything else measures the batch.

## Negative results, recorded so nobody re-runs them

- The `max_shortlist` cap is not the cause. Binds on 8 of 500 segments.
- The frequent-word set is not the cause. 50 → 8000 costs seven times the
  projection and recovers two thirds of one percent.
- Adding the source token ids to the candidate set — the obvious guess for a
  shared vocabulary, since the model can copy names and numbers through —
  changes nothing. 0.95% before, 0.95% after, on both directions.
- The converter is not dropping entries. Offsets and list lengths match the
  lex file exactly.

## The general lesson

Both defects in this ADR are the same one: **a measurement harness is a
configuration too, and an unaudited one is a source of findings that are about
the harness.** The missing `shortlist:` key and the 500-sentence batch each
produced a plausible, specific, wrong conclusion about fizh. The diff caught
the first; only asking "what does the reference do that fizh structurally
cannot" caught the second.
