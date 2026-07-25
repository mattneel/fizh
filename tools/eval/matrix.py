#!/usr/bin/env python3
"""matrix.py — render the coverage sweep as a table.

    python3 tools/eval/matrix.py [--json zig-out/sweep.json] [--markdown]

Reads what `sweep.py` recorded and prints it in the order that matters: the
failures first, because a pair that does not convert is a harder fact than a
pair that scores half a point low, and the whole reason the sweep exists is
that two directions out of a hundred and five is not coverage.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def arch_of(row: dict) -> str:
    """`d=256 ffn=1536 enc=6 dec=2` reduced to something comparable."""
    a = row.get("arch", "")
    bits = {}
    for token in a.split():
        if "=" in token:
            k, v = token.split("=", 1)
            bits[k] = v
    if not bits:
        return "?"
    return f"{bits.get('d','?')}/{bits.get('enc','?')}+{bits.get('dec','?')}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", type=Path, default=Path("zig-out/sweep.json"))
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args(argv)

    rows = json.loads(args.json.read_text())
    rows.sort(key=lambda r: r["pair"])
    good = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    scored = [r for r in good if "delta" in r]

    if bad:
        print(f"\n  {len(bad)} pair(s) not usable\n")
        for r in bad:
            print(f"    {r['pair']:<12} {r.get('stage','?'):<10} "
                  f"{(r.get('error') or r.get('load_error') or '')[:96]}")

    if scored:
        ds = sorted(r["delta"] for r in scored)
        within = sum(1 for x in ds if abs(x) <= 1.0)
        print(f"\n  {len(good)} pair(s) usable, {len(scored)} scored against "
              f"bergamot-translator on 100 FLORES segments\n")
        print(f"    delta median {statistics.median(ds):+.2f}   "
              f"mean {statistics.mean(ds):+.2f}   "
              f"range [{ds[0]:+.2f}, {ds[-1]:+.2f}]")
        print(f"    within +/-1.0 chrF++: {within}/{len(ds)}")
        print(f"    worse by more than 2: "
              f"{[(r['pair'], r['delta']) for r in scored if r['delta'] < -2] or 'none'}")

    archs = {}
    for r in good:
        archs.setdefault(arch_of(r), []).append(r["pair"])
    print("\n  architectures seen (d_model/enc+dec):")
    for a, ps in sorted(archs.items(), key=lambda kv: -len(kv[1])):
        print(f"    {a:<12} {len(ps):3d} pairs" +
              (f"   {', '.join(sorted(ps))}" if len(ps) <= 8 else ""))

    if args.markdown:
        print("\n| pair | arch | fizh | bergamot | delta | identical | MiB |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            if not r.get("ok"):
                print(f"| {r['pair']} | — | — | — | **{r.get('stage')}** | — | — |")
                continue
            mib = r.get("fzm_bytes", 0) / 2**20
            print(f"| {r['pair']} | {arch_of(r)} | {r.get('chrf_gold','—')} | "
                  f"{r.get('chrf_ref','—')} | {r.get('delta','—')} | "
                  f"{r.get('identical','—')}/100 | {mib:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
