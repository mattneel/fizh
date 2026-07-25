#!/usr/bin/env python3
"""divergence.py — where and how often fizh and the reference part company.

    python3 tools/eval/divergence.py --model m.fzm --hyp a.txt --ref b.txt

Sentence-level byte-identity is not comparable across language pairs: a longer
target has more chances to diverge, so the same per-token agreement produces a
lower sentence rate. Comparing 21.6% (es→en) with 21.2% (en→de) without
normalizing for length compares two different things.

So this reports the per-token hazard instead — the probability that greedy
decode makes a different choice at a step, given it has agreed up to that step —
plus the distribution of *where* the first divergence lands.

The shape of that distribution is the diagnostic:

  uniform, spread evenly    -> the two systems round differently and diverge at
                               random, compounding through greedy decode. Not a
                               defect; nothing to fix.
  clustered early           -> something is wrong at the start of decode: the
                               start token, the first cross-attention read, or
                               the shortlist seeding.
  clustered at a position   -> a bound, a cache boundary, or an off-by-one.

Tokenization is the model's own, applied to detokenized output. For a unigram
tokenizer that round-trips (which `zig build test` asserts), it recovers the
decoder's sequence for all but pathological strings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import reference as R  # noqa: E402


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--hyp", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--label", default="")
    p.add_argument("--buckets", type=int, default=5)
    args = p.parse_args(argv)

    m = R.Model(args.model)
    hyp = args.hyp.read_text(encoding="utf-8").splitlines()
    ref = args.ref.read_text(encoding="utf-8").splitlines()
    n = min(len(hyp), len(ref))

    lengths, firsts, compared, diverged, identical = [], [], 0, 0, 0
    for i in range(n):
        a = R.tokenize(m, hyp[i])
        b = R.tokenize(m, ref[i])
        lengths.append(len(b))
        d = first_divergence(a, b)
        if d is None:
            identical += 1
            compared += len(b)
        else:
            diverged += 1
            firsts.append((d, len(b)))
            compared += d + 1

    hazard = diverged / compared if compared else 0.0
    print(f"\n  === {args.label or args.model.name} ===")
    print(f"  segments                     {n}")
    print(f"  sentence-level identical     {identical}/{n} ({100 * identical / n:.1f}%)")
    print(f"  target length, tokens        mean {statistics.mean(lengths):.1f}  "
          f"median {statistics.median(lengths):.0f}")
    print(f"  decisions compared           {compared}")
    print(f"  per-token divergence hazard  {hazard:.4f}  "
          f"(per-token agreement {1 - hazard:.4f})")

    if not firsts:
        return 0
    abs_pos = [d for d, _ in firsts]
    rel_pos = [d / L for d, L in firsts if L]
    print(f"  first divergence, absolute   mean {statistics.mean(abs_pos):.1f}  "
          f"median {statistics.median(abs_pos):.0f}")
    print(f"  first divergence, relative   mean {statistics.mean(rel_pos):.3f}  "
          f"median {statistics.median(rel_pos):.3f}")

    # If divergence is memoryless the first-divergence position is geometric,
    # so the raw histogram skews early even with no defect. What distinguishes a
    # defect is the *hazard* — the chance of diverging in a bucket given you
    # reached it — being flat across position rather than concentrated.
    print("  hazard by relative position (flat => no positional defect):")
    for k in range(args.buckets):
        lo = k / args.buckets
        hi = (k + 1) / args.buckets
        # At risk in this bucket: everything that had not diverged before `lo`.
        at_risk = identical + sum(1 for d, L in firsts if L and d / L >= lo)
        events = sum(1 for d, L in firsts if L and lo <= d / L < hi)
        rate = events / at_risk if at_risk else 0.0
        print(f"    [{lo:.1f},{hi:.1f})  events {events:4d}  at-risk {at_risk:4d}  "
              f"rate {rate:.3f} {'#' * int(rate * 60)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
