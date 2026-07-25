# fizh — coverage matrix

Every language pair in Firefox's `translations-models` registry, fetched,
converted, loaded through the real loader, and translated against a fixed
100-segment FLORES devtest slice. `delta` is chrF++ against
`bergamot-translator` on the same slice, run `--per-line` — one sentence per
batch, which is the only configuration comparable to a library that translates
one message at a time (ADR 0018).

Regenerate with:

```sh
python3 tools/eval/sweep.py --flores <flores200_dataset/devtest>
python3 tools/eval/matrix.py --markdown
```

## Summary

| | |
|---|---|
| pairs in the registry | 105 |
| usable | **99** |
| within ±1.0 chrF++ of the reference | **87 / 99** |
| median delta | **+0.17** |
| mean byte-identical segments | 40.6 / 100 |
| architectures | `d=256, 6+2` (76 pairs), `d=384, 6+4` (23 pairs) |

`identical` counts segments byte-identical to the reference. It is *not* a
quality measure — two int8 implementations rounding differently diverge at a
per-token hazard of ~0.025 and compound from there — but a collapse in it is a
good early warning, which is why it is here.

## The six that are not usable

| pair(s) | why |
|---|---|
| `en-ja`, `en-ko`, `en-zh-Hans`, `en-zh-Hant`, `zh-Hant-en` | separate source and target vocabularies. fizh has one table end to end — the tokenizer, the shortlist's source index and the tied output projection all read it — so either choice mistranslates. |
| `zh-Hans-en` | SPEC §9 packs a language into two ASCII letters; a script-qualified code has nowhere to go. |

Both are refusals with a stated reason, not crashes.

## The four that are usable but bad

`bg-en` −10.87, `en-bg` −20.93, `en-fr` −21.70, `fr-en` −19.97.

These are **not fizh defects**. Their v1.0 "tiny" artifact — the one the
fetcher selects, because SPEC §14 budgets weights at 20 MB — is a poor one.
Their v2.0 translates correctly through fizh:

| | v1.0 tiny (17 MB) | v2.0 (33 MB) |
|---|---|---|
| en→fr | −21.70 | `Le chat noir dort sur la table.` |
| en→bg | −20.93 | `Черната котка спи на масата.` |

v2.0 is over the §14 weights budget, so the selection policy stays and these
four are named here rather than averaged into the median. Point
`tools/bergamot.py` at the newer model directly if you want it.

## Full matrix

| pair | arch | fizh | bergamot | delta | identical | MiB |
|---|---|---|---|---|---|---|
| ar-en | 384/6+4 | 61.47 | 61.84 | -0.37 | 48/100 | 33.4 |
| az-en | 256/6+2 | 47.4 | 46.47 | 0.93 | 21/100 | 19.7 |
| be-en | 256/6+2 | 46.03 | 46.14 | -0.11 | 30/100 | 19.7 |
| bg-en | 256/6+2 | 51.56 | 62.43 | -10.87 | 2/100 | 19.4 |
| bn-en | 256/6+2 | 53.25 | 54.68 | -1.43 | 29/100 | 19.8 |
| bs-en | 384/6+4 | 62.3 | 62.35 | -0.05 | 41/100 | 33.6 |
| ca-en | 256/6+2 | 59.69 | 58.82 | 0.87 | 43/100 | 19.9 |
| cs-en | 256/6+2 | 60.37 | 60.09 | 0.28 | 35/100 | 19.7 |
| da-en | 256/6+2 | 68.11 | 67.98 | 0.13 | 36/100 | 19.5 |
| de-en | 256/6+2 | 63.11 | 63.05 | 0.06 | 50/100 | 19.8 |
| el-en | 256/6+2 | 57.5 | 56.35 | 1.15 | 32/100 | 19.7 |
| en-ar | 384/6+4 | 57.56 | 56.9 | 0.66 | 42/100 | 32.7 |
| en-az | 256/6+2 | 40.61 | 40.18 | 0.43 | 50/100 | 19.0 |
| en-bg | 256/6+2 | 44.93 | 65.86 | -20.93 | 0/100 | 19.0 |
| en-bn | 256/6+2 | 47.66 | 47.66 | 0.0 | 40/100 | 19.1 |
| en-bs | 384/6+4 | 58.15 | 58.16 | -0.01 | 43/100 | 32.7 |
| en-ca | 256/6+2 | 65.55 | 65.37 | 0.18 | 51/100 | 19.1 |
| en-cs | 256/6+2 | 55.68 | 54.33 | 1.35 | 32/100 | 19.1 |
| en-da | 256/6+2 | 64.94 | 64.54 | 0.4 | 54/100 | 19.2 |
| en-de | 256/6+2 | 62.88 | 62.33 | 0.55 | 48/100 | 19.2 |
| en-el | 256/6+2 | 53.05 | 52.73 | 0.32 | 36/100 | 18.8 |
| en-es | 256/6+2 | 51.62 | 51.75 | -0.13 | 37/100 | 19.0 |
| en-et | 256/6+2 | 52.55 | 52.32 | 0.23 | 28/100 | 18.7 |
| en-eu | 384/6+4 | 51.93 | 51.88 | 0.05 | 52/100 | 32.7 |
| en-fa | 256/6+2 | 52.86 | 52.47 | 0.39 | 50/100 | 19.1 |
| en-fi | 256/6+2 | 51.89 | 51.52 | 0.37 | 22/100 | 19.0 |
| en-fr | 256/6+2 | 46.09 | 67.79 | -21.7 | 0/100 | 19.5 |
| en-gl | 384/6+4 | 58.44 | 58.34 | 0.1 | 58/100 | 32.6 |
| en-gu | 256/6+2 | 49.36 | 50.1 | -0.74 | 48/100 | 19.0 |
| en-he | 256/6+2 | 56.78 | 56.57 | 0.21 | 47/100 | 18.9 |
| en-hi | 256/6+2 | 64.17 | 63.49 | 0.68 | 50/100 | 19.2 |
| en-hr | 256/6+2 | 53.71 | 53.35 | 0.36 | 39/100 | 19.0 |
| en-hu | 384/6+4 | 56.92 | 55.83 | 1.09 | 57/100 | 32.8 |
| en-id | 256/6+2 | 66.52 | 66.63 | -0.11 | 62/100 | 19.0 |
| en-is | 384/6+4 | 52.87 | 52.74 | 0.13 | 31/100 | 33.0 |
| en-it | 256/6+2 | 56.26 | 55.69 | 0.57 | 51/100 | 19.5 |
| en-ja | — | — | — | **convert** | — | — |
| en-kn | 256/6+2 | 52.4 | 52.65 | -0.25 | 48/100 | 19.0 |
| en-ko | — | — | — | **convert** | — | — |
| en-lt | 384/6+4 | 55.25 | 54.53 | 0.72 | 36/100 | 32.8 |
| en-lv | 384/6+4 | 56.99 | 56.47 | 0.52 | 58/100 | 32.8 |
| en-ml | 256/6+2 | 54.31 | 53.07 | 1.24 | 34/100 | 18.9 |
| en-ms | 256/6+2 | 66.75 | 66.27 | 0.48 | 65/100 | 19.2 |
| en-nb | 256/6+2 | 57.92 | 57.61 | 0.31 | 52/100 | 19.4 |
| en-nl | 256/6+2 | 54.81 | 54.97 | -0.16 | 47/100 | 19.5 |
| en-pl | 256/6+2 | 49.14 | 48.65 | 0.49 | 31/100 | 19.1 |
| en-pt | 256/6+2 | 70.45 | 69.61 | 0.84 | 46/100 | 19.5 |
| en-ro | 256/6+2 | 65.51 | 65.04 | 0.47 | 42/100 | 19.2 |
| en-ru | 384/6+4 | 55.54 | 55.19 | 0.35 | 45/100 | 32.6 |
| en-sk | 384/6+4 | 59.04 | 58.81 | 0.23 | 49/100 | 32.8 |
| en-sl | 256/6+2 | 51.84 | 51.85 | -0.01 | 28/100 | 19.0 |
| en-sq | 256/6+2 | 56.76 | 56.59 | 0.17 | 61/100 | 19.0 |
| en-sr | 384/6+4 | 56.44 | 56.23 | 0.21 | 50/100 | 32.5 |
| en-sv | 256/6+2 | 67.31 | 66.65 | 0.66 | 63/100 | 19.3 |
| en-ta | 256/6+2 | 52.7 | 53.25 | -0.55 | 48/100 | 19.0 |
| en-te | 256/6+2 | 57.76 | 58.2 | -0.44 | 47/100 | 19.0 |
| en-th | 384/6+4 | 47.98 | 47.47 | 0.51 | 54/100 | 32.8 |
| en-tr | 256/6+2 | 55.78 | 55.36 | 0.42 | 38/100 | 18.9 |
| en-uk | 384/6+4 | 57.1 | 56.75 | 0.35 | 52/100 | 32.7 |
| en-vi | 384/6+4 | 61.57 | 61.43 | 0.14 | 60/100 | 33.3 |
| en-zh-Hans | — | — | — | **convert** | — | — |
| en-zh-Hant | — | — | — | **convert** | — | — |
| es-en | 256/6+2 | 53.85 | 53.62 | 0.23 | 48/100 | 19.2 |
| et-en | 256/6+2 | 56.52 | 56.67 | -0.15 | 30/100 | 19.3 |
| eu-en | 384/6+4 | 53.58 | 53.58 | 0.0 | 52/100 | 33.0 |
| fa-en | 256/6+2 | 55.22 | 55.19 | 0.03 | 49/100 | 19.3 |
| fi-en | 256/6+2 | 54.09 | 54.16 | -0.07 | 39/100 | 19.8 |
| fr-en | 256/6+2 | 43.89 | 63.86 | -19.97 | 1/100 | 19.8 |
| gl-en | 384/6+4 | 60.75 | 60.75 | 0.0 | 57/100 | 33.3 |
| gu-en | 256/6+2 | 56.72 | 56.79 | -0.07 | 45/100 | 19.5 |
| he-en | 256/6+2 | 60.58 | 59.74 | 0.84 | 44/100 | 19.6 |
| hi-en | 256/6+2 | 62.61 | 63.12 | -0.51 | 35/100 | 19.7 |
| hr-en | 256/6+2 | 57.79 | 56.74 | 1.05 | 21/100 | 19.2 |
| hu-en | 256/6+2 | 55.72 | 54.89 | 0.83 | 33/100 | 19.9 |
| id-en | 256/6+2 | 63.79 | 62.71 | 1.08 | 36/100 | 19.3 |
| is-en | 384/6+4 | 55.14 | 55.12 | 0.02 | 57/100 | 33.1 |
| it-en | 256/6+2 | 57.51 | 57.28 | 0.23 | 41/100 | 19.8 |
| ja-en | 384/6+4 | 53.54 | 53.78 | -0.24 | 38/100 | 48.3 |
| kn-en | 256/6+2 | 52.84 | 53.68 | -0.84 | 32/100 | 19.8 |
| ko-en | 384/6+4 | 51.12 | 51.13 | -0.01 | 39/100 | 47.9 |
| lt-en | 256/6+2 | 54.13 | 53.35 | 0.78 | 29/100 | 19.6 |
| lv-en | 256/6+2 | 55.81 | 55.48 | 0.33 | 22/100 | 19.4 |
| ml-en | 256/6+2 | 56.61 | 56.93 | -0.32 | 44/100 | 20.1 |
| ms-en | 256/6+2 | 63.3 | 63.73 | -0.43 | 44/100 | 19.5 |
| mt-en | 256/6+2 | 69.18 | 69.02 | 0.16 | 51/100 | 18.8 |
| nb-en | 256/6+2 | 62.84 | 63.14 | -0.3 | 65/100 | 19.5 |
| nl-en | 256/6+2 | 55.2 | 55.19 | 0.01 | 42/100 | 19.7 |
| nn-en | 256/6+2 | 63.67 | 63.11 | 0.56 | 39/100 | 13.7 |
| pl-en | 256/6+2 | 52.6 | 52.78 | -0.18 | 25/100 | 19.7 |
| pt-en | 256/6+2 | 69.06 | 69.12 | -0.06 | 36/100 | 19.7 |
| ro-en | 256/6+2 | 66.08 | 66.37 | -0.29 | 40/100 | 19.6 |
| ru-en | 256/6+2 | 56.32 | 56.21 | 0.11 | 25/100 | 19.9 |
| sk-en | 256/6+2 | 58.43 | 57.88 | 0.55 | 21/100 | 19.7 |
| sl-en | 256/6+2 | 54.34 | 53.71 | 0.63 | 34/100 | 19.3 |
| sq-en | 256/6+2 | 61.28 | 61.51 | -0.23 | 35/100 | 19.4 |
| sr-en | 256/6+2 | 58.12 | 57.54 | 0.58 | 37/100 | 19.6 |
| sv-en | 256/6+2 | 66.09 | 65.99 | 0.1 | 43/100 | 19.6 |
| ta-en | 384/6+4 | 52.31 | 52.98 | -0.67 | 44/100 | 33.8 |
| te-en | 256/6+2 | 57.34 | 57.09 | 0.25 | 43/100 | 19.9 |
| th-en | 384/6+4 | 54.29 | 54.55 | -0.26 | 34/100 | 33.5 |
| tr-en | 256/6+2 | 56.94 | 56.7 | 0.24 | 51/100 | 19.6 |
| uk-en | 256/6+2 | 58.24 | 58.13 | 0.11 | 31/100 | 19.3 |
| vi-en | 256/6+2 | 59.26 | 57.93 | 1.33 | 34/100 | 19.2 |
| zh-Hans-en | — | — | — | **convert** | — | — |
| zh-Hant-en | — | — | — | **convert** | — | — |
