# ADR 0018 — The shortlist is not free, and the reference was not using one

Status: open — cause identified as the candidate lists, mechanism not yet found
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

Resolving that is the next step and it is a prerequisite for the rest: until
the reference is known to be running Firefox's configuration, "fizh's shortlist
is lossier than bergamot's" is a hypothesis with one measurement behind it.

## Decision

Nothing changes in the runtime yet. The candidate-set construction in
`tools/bergamot.py` and `src/graph/shortlist.zig` is the suspect, and this
records the evidence so the next pass starts from measurements rather than
from the top.

The cheap mitigation, if it comes to that, is not a bigger shortlist: 8000
frequent words cost seven times the projection and recovered two thirds of one
percent. It is finding what bergamot puts in the set that fizh does not.

## Negative results, recorded so nobody re-runs them

- The `max_shortlist` cap is not the cause. Binds 1.6%.
- The frequent-word set is not the cause. 7x recovers a third.
- es→en shortlist loss is negligible (+0.02) and needs no further work.
