#!/usr/bin/env python3
"""calibrate.py — anchor fizh's chrF++ against bergamot-translator. SPEC §13 T4.

    python3 tools/eval/calibrate.py --flores <flores200_dataset> [--limit N]

A chrF++ number on its own says nothing. 42.51 could be a correct
implementation of a 42-point model, or a subtly broken implementation of a
45-point one — the 127-vs-128 positional bug moved it 0.79 and was invisible
until measured. So every number here is reported next to the *same* corpus run
through the actual Bergamot engine, in-process, over the same model files.

The gate: **more than 2 chrF++ below the reference on any direction means there
is another bug.** Encoder collapse passed every assertion in the tree.

Pivots are reported per-hop and end to end, because those fail differently:
a pivot that is fine per-hop and bad end to end is a composition bug, and one
that is bad at hop one is just hop one.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chrf import corpus_score  # noqa: E402

HERE = Path(__file__).parent
FLORES_CODE = {"es": "spa_Latn", "en": "eng_Latn", "de": "deu_Latn"}


def flores(root: Path, lang: str, split: str, limit: int | None):
    path = root / split / f"{FLORES_CODE[lang]}.{split}"
    if not path.exists():
        raise SystemExit(f"{path} not found — see tools/eval/corpora/README.md")
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[:limit] if limit else lines


def run(cmd, lines, label):
    proc = subprocess.run(cmd, input="\n".join(lines) + "\n",
                          capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        raise SystemExit(f"{label} exited {proc.returncode}")
    out = proc.stdout.split("\n")
    while out and out[-1] == "":
        out.pop()
    if len(out) != len(lines):
        raise SystemExit(f"{label}: {len(out)} outputs for {len(lines)} inputs")
    return out


def fizh(models, src, tgt, lines, binary):
    cmd = [str(binary)]
    for m in models:
        cmd += ["--model", str(m)]
    cmd += ["--src", src, "--tgt", tgt]
    return run(cmd, lines, "fizh")


def reference(bundle, lines):
    return run(["node", str(HERE / "reference_engine.mjs"), str(bundle)],
               lines, "reference")


def report(rows):
    w = max(len(r[0]) for r in rows)
    print(f"\n  {'direction':<{w}}  {'fizh':>7}  {'bergamot':>9}  {'delta':>7}  verdict")
    print(f"  {'-' * w}  {'-' * 7}  {'-' * 9}  {'-' * 7}  -------")
    worst = 0.0
    for name, a, b in rows:
        d = a - b
        worst = min(worst, d)
        verdict = "ok" if d >= -2.0 else "INVESTIGATE"
        print(f"  {name:<{w}}  {a:7.2f}  {b:9.2f}  {d:+7.2f}  {verdict}")
    return worst


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flores", type=Path, required=True)
    p.add_argument("--split", default="devtest")
    p.add_argument("--limit", type=int)
    p.add_argument("--models", type=Path, default=Path("zig-out"))
    p.add_argument("--bundles", type=Path, default=Path("zig-out/bergamot"))
    p.add_argument("--binary", type=Path, default=Path("zig-out/bin/translate"))
    args = p.parse_args(argv)

    es = flores(args.flores, "es", args.split, args.limit)
    en = flores(args.flores, "en", args.split, args.limit)
    de = flores(args.flores, "de", args.split, args.limit)
    n = len(es)
    print(f"FLORES-200 {args.split}, {n} segments, beam 1 both sides")

    esen = args.models / "esen.fzm"
    ende = args.models / "ende.fzm"
    rows = []

    # --- direct ------------------------------------------------------------
    print("  es->en ...", flush=True)
    f_esen = fizh([esen], "es", "en", es, args.binary)
    r_esen = reference(args.bundles / "esen", es)
    rows.append(("es->en", corpus_score(f_esen, en), corpus_score(r_esen, en)))

    print("  en->de ...", flush=True)
    f_ende = fizh([ende], "en", "de", en, args.binary)
    r_ende = reference(args.bundles / "ende", en)
    rows.append(("en->de", corpus_score(f_ende, de), corpus_score(r_ende, de)))

    # --- pivot, per hop and end to end -------------------------------------
    # Hop two is scored on *gold* English so it measures only the en->de model;
    # end to end feeds hop one's own output, which is what a user gets.
    print("  es->de (end to end) ...", flush=True)
    f_esde = fizh([esen, ende], "es", "de", es, args.binary)
    r_esde = reference(args.bundles / "ende", r_esen)
    rows.append(("es->de hop1 (es->en)", rows[0][1], rows[0][2]))
    rows.append(("es->de hop2 (en->de, gold in)", rows[1][1], rows[1][2]))
    rows.append(("es->de end to end", corpus_score(f_esde, de), corpus_score(r_esde, de)))

    worst = report(rows)

    # How often does fizh agree with the reference exactly? A high rate means
    # the remaining delta is a handful of segments, not a systematic drift.
    same = sum(1 for a, b in zip(f_esen, r_esen) if a.strip() == b.strip())
    print(f"\n  es->en exact-match with bergamot: {same}/{n} ({100 * same / n:.1f}%)")

    print()
    if worst < -2.0:
        print(f"  GATE FAILED: {worst:.2f} chrF++ below reference. There is another bug.")
        return 1
    print(f"  GATE PASSED: worst direction is {worst:+.2f} chrF++ against the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
