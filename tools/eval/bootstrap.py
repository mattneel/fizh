#!/usr/bin/env python3
"""bootstrap.py — is the delta real? SPEC §13 T4.

    python3 tools/eval/bootstrap.py --hyp a.txt --ref b.txt --gold g.txt

Paired bootstrap resampling (Koehn 2004), the standard MT significance test.
Resample the segment set with replacement, rescore both systems on the *same*
resample, and look at the distribution of the difference. Pairing matters: the
two systems see identical segments in every draw, so segment difficulty cancels
and what is left is the systematic difference between them.

A chrF++ delta without an interval is not a measurement. −0.42 on 500 segments
may be a real regression or may be the corpus; only the interval says which,
and hunting a cause for a delta that straddles zero is hunting noise.

Resampling sums per-segment n-gram statistics rather than re-tokenizing, so a
thousand draws costs about as much as one scoring pass.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chrf import score, sentence_stats  # noqa: E402


def read(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines


def aggregate(stats, idx):
    """Sum per-segment (overlap, hyp_total, ref_total) triples over `idx`."""
    total = None
    for i in idx:
        s = stats[i]
        if total is None:
            total = [list(t) for t in s]
        else:
            for acc, t in zip(total, s):
                for k in range(3):
                    acc[k] += t[k]
    return score(total)


def paired_bootstrap(hyp, ref, gold, draws: int, seed: int):
    n = len(gold)
    hs = [sentence_stats(hyp[i], gold[i]) for i in range(n)]
    rs = [sentence_stats(ref[i], gold[i]) for i in range(n)]

    observed = aggregate(hs, range(n)) - aggregate(rs, range(n))

    rng = random.Random(seed)
    deltas = []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(aggregate(hs, idx) - aggregate(rs, idx))
    deltas.sort()

    lo = deltas[int(0.025 * draws)]
    hi = deltas[int(0.975 * draws) - 1]
    # Fraction of draws in which the hypothesis system is not worse. A value
    # near 0.5 means the two are indistinguishable on this corpus.
    not_worse = sum(1 for d in deltas if d >= 0) / draws
    return observed, lo, hi, not_worse


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hyp", type=Path, required=True, help="the system under test")
    p.add_argument("--ref", type=Path, required=True, help="the reference system")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260725)
    p.add_argument("--label", default="")
    args = p.parse_args(argv)

    hyp, ref, gold = read(args.hyp), read(args.ref), read(args.gold)
    n = min(len(hyp), len(ref), len(gold), args.limit or 10**9)
    hyp, ref, gold = hyp[:n], ref[:n], gold[:n]

    obs, lo, hi, not_worse = paired_bootstrap(hyp, ref, gold, args.draws, args.seed)
    covers_zero = lo <= 0 <= hi
    verdict = "INDISTINGUISHABLE" if covers_zero else "significant"
    print(f"  {args.label or args.hyp.name:<10} n={n} draws={args.draws}  "
          f"delta {obs:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
          f"P(not worse)={not_worse:.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
