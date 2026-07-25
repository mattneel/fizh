# ADR 0022 — The GCS registry is authoritative, and bigger models win

Status: decided
Date: 2026-07-25
Milestone: M10 (registry)
Amends: SPEC §14 weights budget, SPEC §4.3

## Context

`mozilla/firefox-translations-models` was archived read-only on 2025-12-15.
fizh never read it: `tools/fetch-model.sh` enumerated **Firefox's
remote-settings CDN**, which is Firefox's *shipped* set — every version it has
ever shipped, including legacy ones, with no field saying which is current.

That is the root of ADR 0019's worst finding. With no status field, selection
had to be inferred from size, and `min(size)` picked pre-release artifacts for
21 of 105 pairs.

Models now live in a public GCS bucket with a generated manifest:
`.../moz-fx-translations-data--303e-prod-translations-data/db/models.json`.

## What the two sources actually are

| | remote-settings | GCS `models.json` |
|---|---|---|
| pairs | 105 | **109** |
| records | every shipped version | 137, one curated set |
| status field | none | `releaseStatus` |
| architecture | inferred from size | `architecture` |
| integrity | attachment size only | **`uncompressedHash`, sha256, on all 137** |
| quality | none | per-model chrF++ on flores200-plus |

Nine pairs exist only in GCS (`en-no`, `no-en`, `hbs-en`, `mr-en`, `ur-en`,
plus the `zh`/`zh_hant` spellings). One exists only in remote-settings:
**`mt-en`**.

## Decision 1: GCS is authoritative

It carries the two things selection needs and remote-settings does not — an
explicit release status and a content hash. The hash also gives P3's
immutability story something to stand on: an artifact is identified by its
content, not by a URL that may be rewritten.

**The trade is reproducibility.** An archived repository is frozen; a live
bucket is not. `models.json` carries a `generated` timestamp, and fizh records
the upstream path, size and sha256 of every artifact it converts, so a `.fzm`
can be traced to exact upstream bytes even after the bucket moves on. That is
the story fizh is buying: *not* "this will resolve identically forever", but
"this is exactly what it was built from, and you can check".

`mt-en` is the cost. It is in the shipped set and not in the bucket, so it
leaves the supported list until upstream republishes it.

## Decision 2: select the best measured chrF++ among released records

Not "smallest" and not "biggest". The registry publishes chrF++ per model, so
selection is a measurement rather than a heuristic — which is SPEC §12.11
applied to the fetcher.

Bigger is *usually* better, and the registry says by how much:

| comparison | pairs | mean Δ chrF++ | better on | size |
|---|---|---|---|---|
| base-memory vs tiny | 17 | **+1.53** | 17/17 | +14.4 MiB |
| base vs tiny | 5 | **+2.18** | 5/5 | +30.4 MiB |
| base vs base-memory | 11 | +0.55 | **9/11** | +12.1 MiB |

The last row is why the rule is not "prefer the largest architecture":
`de-en` is **−1.36** on `base` and `en-ko` is **−1.04**. A size-ordered rule
would have shipped a worse model for both, silently, and it would have looked
like a language property — the exact failure §12.11 exists to prevent.

Resulting selection over 109 pairs: 55 `base-memory`, 50 `tiny` (where nothing
better is released), 4 `base`.

## Decision 3: the weights budget follows the models

§14 budgeted weights at 20 MB, then 35 MB. Under this rule the largest artifact
is **56.7 MiB** and a pivot holds two, so the budget becomes **64 MB per
direction** and the §4.3 worked case becomes ~122 MB resident.

That is a large number for a phone and it is deliberately not hedged: the
quality is worth measuring against, and whether a device tolerates it is a
question for T5 on real hardware, not for a budget written on a desktop. If it
turns out a phone cannot hold two `base-memory` directions, the answer is a
per-device selection policy — fetch `tiny` where memory is tight — which the
registry now has the metadata to express.

## What this does not fix

The **five split-vocabulary pairs** carry `srcVocab`/`trgVocab` upstream too —
`en-ja`, `en-ko`, `en-zh`, `en-zh_hant`, `zh_hant-en`, exactly the set fizh
refuses. A newer source does not make them loadable; separate source and target
vocabularies is a runtime change.

`zh-Hans-en` was the sixth refusal and is already recovered: it failed on the
two-byte language code, which format 3 widened.
