# ADR 0004 — Implicit trie, and the tokenizer's artifact conventions

Status: decided
Date: 2026-07-24
Milestone: M1

## Context

SPEC §5 names `tok/trie.zig` and §6 says the vocabulary ships as tensors. It
does not say what shape either takes. Three things had to be decided before the
Viterbi could be written.

## Decision 1 — the trie is implicit

`tok/trie.zig` exposes a trie interface (`root`, `descend`, `pieceHere`) over
the vocabulary held in lexicographic order. A cursor is the half-open rank range
`[lo, hi)` of pieces sharing the prefix walked so far; `descend` narrows it with
two binary searches bounded at 32 halvings.

The alternative — real nodes and edges in CSR form — costs one node per distinct
prefix. For a 32k-piece SentencePiece vocabulary that is 100k–250k nodes, so
1.5–4 MB, either built at load or shipped in the artifact. SPEC §14 budgets 20 MB
per direction against ~17 MB of weights. There is no room, and buying it back
would mean trading translation quality for a data structure.

Cost of the implicit form: one `u32` per piece (128 KB at 32k), zero load-time
work, and `O(log V)` on the first descend collapsing to near-`O(1)` after, since
the range shrinks geometrically. A 4096-byte source needs on the order of 10⁵
byte comparisons.

## Decision 2 — the word-boundary marker is one byte

`tools/convert.py` rewrites SentencePiece's `▁` (U+2581, three bytes) to the
single byte `0xFF` in every piece it emits. `0xFF` cannot occur in well-formed
UTF-8, and SPEC §11 has the runtime reject non-UTF-8 source at the boundary, so
the byte is free.

This keeps normalization length-preserving. Without it, escaping spaces would
expand the source by up to 3×, and SPEC §4.2 sizes `tok_lattice` at
`max_src_bytes` nodes — the lattice would have to triple. `decode` maps `0xFF`
back to a space.

The converter asserts no piece contains a literal `0xFF` before rewriting.

## Decision 3 — the unknown edge only appears when nothing matched

The lattice needs an escape edge or a byte no piece covers disconnects it. But
offering that edge *alongside* real pieces lets it outbid them: with
`unk_score = min_score - 10`, a two-byte character covered by two single-byte
pieces scores `2 · min_score`, which is worse. "¿qué" tokenizes to "qu".

So the unknown edge is added at position `i` only when no piece matched at `i`
at all. It then spans one whole UTF-8 character, never a fragment of one. This
matches SentencePiece, which adds its unknown node only in the absence of a
single-character node. The lattice stays connected because every reachable
position keeps at least one outgoing edge.

## Known gap — normalization

SentencePiece's `nmt_nfkc` normalizer is a character-rewriting table shipped
inside the `.spm` model. `tok/unigram.zig` implements the structural half of the
preprocessing — collapse whitespace runs, drop leading and trailing whitespace,
prepend the dummy prefix — and none of the Unicode rewriting.

For chat register this is mostly harmless; for FLORES it costs a little on text
with unusual punctuation or full-width forms. If T4 shows the gap is real, the
fix is a `tok.normalizer` tensor holding the table `convert.py` reads out of the
`.spm`, applied in `normalize`. Recorded as a gap rather than fixed blind,
because SPEC §13 T4 is what should decide it.

## Consequences

- The vocabulary ships as five tensors: `tok.pieces`, `tok.offsets`,
  `tok.scores`, `tok.order`, `tok.flags`.
- `Vocab.validate` proves `order` is a permutation by proving the pieces it
  indexes strictly increase — `size` distinct values below `size` cover the
  range — so no scratch bitmap is needed at load.
- `arena.zig` grows a `tok_norm` region of `max_src_bytes + 8`, and the lattice
  is sized to match.
