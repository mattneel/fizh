# Corpora

SPEC §13 T4 asks for two, **reported separately, never averaged**.

## `chat.{src}-{tgt}.tsv` — the chat register

Ships here. Short messages, emoji, code-switching, typos, missing punctuation,
lowercase — the register a direct messaging app actually carries. Nobody
publishes a standard one, so this is hand-written and small.

Format: `source<TAB>reference`, one segment per line, UTF-8, `#` comments.

It is small on purpose. Forty segments will not resolve a half-point of chrF++,
and it is not meant to: it is meant to catch the failure where a model gets
better at Wikipedia prose and worse at "q tal". Grow it when a real regression
slips past it, and keep the additions in register.

## `flores.{src}-{tgt}.tsv` — comparability

**Not in this repository.** FLORES-200 is CC-BY-SA-4.0 and large; vendoring a
subset would make this repository's licence a conversation nobody wants to have,
and a *subset someone chose* is worse for comparability than the real thing.

To produce one:

```sh
curl -LO https://tinyurl.com/flores200dataset   # or the current NLLB mirror
tar xf flores200_dataset.tar.gz
paste flores200_dataset/dev/spa_Latn.dev \
      flores200_dataset/dev/eng_Latn.dev \
  | head -200 > tools/eval/corpora/flores.es-en.tsv
```

`run.py` picks it up automatically when the filename matches the language pair,
and says so when it is missing rather than quietly scoring one corpus.

## Pivot pairs

Evaluated end to end, never per hop:

```sh
python3 tools/eval/run.py --model es-en.fzm --model en-de.fzm --src es --tgt de
```

The user never sees the English in the middle, so neither does the score.
