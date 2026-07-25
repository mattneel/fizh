#!/usr/bin/env python3
"""token_ratio.py — how many target tokens does a source token buy?

    python3 tools/eval/token_ratio.py --models zig-out/sweep --flores <devtest>

`max_tgt_tokens` is 384 against `max_src_tokens` 256 — a ratio of 1.5 that
nobody justified. The decode loop is also bounded by `max_length_factor ·
src_len` (SPEC §12.3), so between them they decide when a long sentence gets
truncated. Both should come from data.

The measurement is per language pair, over FLORES devtest, tokenized with the
model's own vocabulary: the source side and the target side of the same
multi-parallel corpus, so the ratio is what a faithful translation of that
sentence actually costs. Reported as a distribution, because the maximum is
what a bound has to survive and the median is what it is usually paying for.

A ratio above the configured one does not mean output is wrong — the decoder
stops at `max_tgt_tokens` and returns what it has. It means the tail of the
sentence is silently missing, which is the failure this exists to size against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import reference as R  # noqa: E402
from sweep import FLORES  # noqa: E402


def tokenizer(m: R.Model):
    """R.tokenize rebuilds its piece table on every call; a sweep over a
    hundred models cannot afford that. Same lattice, hoisted tables."""
    offsets = m.u32("tok.offsets")
    blob = m.u8("tok.pieces").tobytes()
    scores = m.f32("tok.scores")
    flags = m.u8("tok.flags")
    pieces = [blob[offsets[i]:offsets[i + 1]] for i in range(m.vocab_size)]
    by_bytes = {p: i for i, p in enumerate(pieces) if not flags[i]}
    longest = max(len(p) for p in pieces)
    unk = float(scores.min()) - 10.0

    def count(text: str) -> int:
        s = R.normalize(text)
        n = len(s)
        best = [(-np.inf, 0)] * (n + 1)
        best[0] = (0.0, 0)
        for i in range(n):
            if best[i][0] == -np.inf:
                continue
            hit = False
            for ln in range(1, min(longest, n - i) + 1):
                j = by_bytes.get(s[i:i + ln])
                if j is None:
                    continue
                hit = True
                cand = best[i][0] + float(scores[j])
                if cand > best[i + ln][0]:
                    best[i + ln] = (cand, best[i][1] + 1)
            if not hit:
                step = R.char_len(s[i:])
                cand = best[i][0] + unk
                if cand > best[i + step][0]:
                    best[i + step] = (cand, best[i][1] + 1)
        return best[n][1]

    return count


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", type=Path, default=Path("zig-out/sweep"))
    p.add_argument("--flores", type=Path, required=True)
    p.add_argument("--segments", type=int, default=100)
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    rows = []
    for fzm in sorted(args.models.glob("*.fzm")):
        stem = fzm.stem
        pair = next(((a, b) for a in FLORES for b in FLORES
                     if a.replace("-", "") + b.replace("-", "") == stem), None)
        if pair is None:
            continue
        src, tgt = pair
        sf = args.flores / f"{FLORES[src]}.devtest"
        tf = args.flores / f"{FLORES[tgt]}.devtest"
        if not sf.exists() or not tf.exists():
            continue
        try:
            m = R.Model(fzm)
            count = tokenizer(m)
        except Exception as e:
            print(f"  {src}-{tgt:<8} unreadable: {type(e).__name__}", flush=True)
            continue
        ss = sf.read_text(encoding="utf-8").splitlines()[:args.segments]
        ts = tf.read_text(encoding="utf-8").splitlines()[:args.segments]
        ratios, srcs, tgts = [], [], []
        for a, b in zip(ss, ts):
            na, nb = count(a), count(b)
            if na:
                ratios.append(nb / na)
                srcs.append(na)
                tgts.append(nb)
        r = np.array(ratios)
        rows.append({"pair": f"{src}-{tgt}", "n": len(r),
                     "median": round(float(np.median(r)), 3),
                     "p99": round(float(np.percentile(r, 99)), 3),
                     "max": round(float(r.max()), 3),
                     "max_src_tokens": int(max(srcs)),
                     "max_tgt_tokens": int(max(tgts))})
        print(f"  {rows[-1]['pair']:<10} median {rows[-1]['median']:.2f}  "
              f"p99 {rows[-1]['p99']:.2f}  max {rows[-1]['max']:.2f}  "
              f"longest src {rows[-1]['max_src_tokens']}  tgt {rows[-1]['max_tgt_tokens']}",
              flush=True)

    if not rows:
        print("  no models found")
        return 1
    allmax = max(r["max"] for r in rows)
    worst = max(rows, key=lambda r: r["max"])
    print(f"\n  pairs {len(rows)}")
    print(f"  worst per-segment ratio anywhere: {allmax:.3f}  ({worst['pair']})")
    print(f"  median of per-pair medians:       "
          f"{np.median([r['median'] for r in rows]):.3f}")
    print(f"  longest target seen:              "
          f"{max(r['max_tgt_tokens'] for r in rows)} tokens")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
