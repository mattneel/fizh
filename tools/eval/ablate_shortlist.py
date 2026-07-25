#!/usr/bin/env python3
"""ablate_shortlist.py — does the 50x50 lexical shortlist cost quality?

    python3 tools/eval/ablate_shortlist.py --model m.fzm --src s.txt --gold g.txt

fizh projects the decoder's output over a per-sentence candidate set built from
`lex.50.50` (SPEC §7) rather than over all 32k rows. That is a ~16x saving on
the single largest matmul in the decode loop, and it is only sound if the
argmax never lands outside the candidate set.

This ablates it: the same decoder, the same inputs, the same arithmetic, run
once with the shortlist and once projecting the full vocabulary. Anything the
shortlist costs shows up as a difference between those two columns and nothing
else — which is why this compares oracle against oracle rather than against
bergamot, whose shortlist construction is a second variable.

Slow on purpose. The full-vocabulary column is the whole cost of the thing the
shortlist exists to avoid, so a run at n=500 is the point rather than an
accident of impatience.

Reading it:

  identical 100%, delta 0.00  -> the shortlist never excludes the argmax on
                                 this corpus. Ruled out as a quality factor.
  delta > 0                   -> the candidate set is dropping tokens the model
                                 wanted. Look at `lex.50.50` construction:
                                 candidate semantics, dedup, forced inclusions.
  delta < 0                   -> the shortlist is *helping*, which means it is
                                 masking tokens the model would otherwise get
                                 wrong. Real, but not a reason to keep it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import reference as R  # noqa: E402
from chrf import corpus_score  # noqa: E402


def detokenize(m: R.Model, ids) -> str:
    offsets = m.u32("tok.offsets")
    blob = m.u8("tok.pieces").tobytes()
    raw = b"".join(blob[offsets[i]:offsets[i + 1]] for i in ids)
    return raw.replace(b"\xff", b" ").decode("utf-8", "replace").strip()


def decode(m: R.Model, text: str, full_vocab: bool) -> str:
    """Greedy decode. Mirrors src/graph/decoder.zig, including the ADR 0015
    zero-fill at step 0."""
    src_ids = R.tokenize(m, text)
    if not src_ids:
        return ""
    enc = R.encode(m, src_ids)
    sl = np.arange(m.vocab_size) if full_vocab else R.build_shortlist(m, src_ids, 2048)

    eq, es = m.quant("emb")
    rows_q, rows_s = eq[sl], es[sl]
    rows_b = m.f32("emb.bias")[sl]

    cell = [np.zeros((1, m.d_model)) for _ in range(m.n_dec)]
    limit = min(384, int(np.ceil(m.max_length_factor * len(src_ids))) + 2)

    prev, produced = None, []
    for t in range(limit):
        x = R.embed(m, [prev], pos_offset=t) if prev is not None else R.f32(
            R.positional(m.d_model, 1))
        for l in range(m.n_dec):
            p = f"dec.{l}.rnn"
            h = R.branch_input(m, x, p)
            ww, sw = m.quant(f"{p}.w")
            wf, sf = m.quant(f"{p}.wf")
            wx = R.qmatmul(h, ww, sw)
            g = 1.0 / (1.0 + np.exp(-R.qmatmul(h, wf, sf, m.f32(f"{p}.bf"))))
            cell[l] = R.f32(g * cell[l] + (1.0 - g) * wx)
            x = R.merge(m, x, R.f32(np.maximum(cell[l], 0.0)), p)

            p = f"dec.{l}.xa"
            h = R.branch_input(m, x, p)
            wq, sq = m.quant(f"{p}.q.w")
            wk, sk = m.quant(f"{p}.k.w")
            wv, sv = m.quant(f"{p}.v.w")
            wo, so = m.quant(f"{p}.o.w")
            q = R.qmatmul(h, wq, sq, m.f32(f"{p}.q.bias"))
            xk = R.qmatmul(enc, wk, sk, m.f32(f"{p}.k.bias"))
            xv = R.qmatmul(enc, wv, sv, m.f32(f"{p}.v.bias"))
            c = R.attn_over(q, xk, xv, m.n_heads, m.head_dim)
            x = R.merge(m, x, R.qmatmul(c, wo, so, m.f32(f"{p}.o.bias")), p)

            p = f"dec.{l}.ffn"
            x = R.merge(m, x, R.ffn_block(m, R.branch_input(m, x, p), p), p)

        nxt = int(sl[int(np.argmax(R.qmatmul(x, rows_q, rows_s)[0] + rows_b))])
        if nxt == m.eos_id:
            break
        produced.append(nxt)
        prev = nxt
    return detokenize(m, produced)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--label", default="")
    args = p.parse_args(argv)

    m = R.Model(args.model)
    src = args.src.read_text(encoding="utf-8").splitlines()[:args.limit]
    gold = args.gold.read_text(encoding="utf-8").splitlines()[:args.limit]
    n = min(len(src), len(gold))
    src, gold = src[:n], gold[:n]

    short = [decode(m, s, full_vocab=False) for s in src]
    full = [decode(m, s, full_vocab=True) for s in src]

    a, b = corpus_score(short, gold), corpus_score(full, gold)
    same = sum(1 for x, y in zip(short, full) if x == y)
    print(f"  {args.label or args.model.name:<8} n={n}  shortlist {a:.2f}  "
          f"full-vocab {b:.2f}  delta {b - a:+.2f}  "
          f"identical {same}/{n} ({100 * same / n:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
