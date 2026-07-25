#!/usr/bin/env python3
"""regress.py — lock in what the real model actually produces.

    zig build real                      check
    python3 tools/regress.py --update   re-record, deliberately

Skips with a clear message when the model is absent, because the model is 19 MB
of CC-BY-SA data that this repository does not vendor. It is one command away:
`tools/fetch-model.sh es en`.

This is the test the project did not have. Ninety-eight tests passed against
synthetic artifacts while the decoder was the wrong architecture; nothing in
that suite could have known. A handful of real sentences would have.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GOLDEN = Path("test/real/golden_translations.tsv")
DIVERGENT = Path("test/real/known_divergences.tsv")


def load(path: Path, columns: int = 2):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < columns:
            raise SystemExit(f"{path}: expected {columns} tab-separated columns")
        rows.append(parts)
    return rows


def translate(binary: Path, model: Path, lines) -> list[str]:
    proc = subprocess.run(
        [str(binary), "--model", str(model), "--src", "es", "--tgt", "en"],
        input="\n".join(lines) + "\n", capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{binary} exited {proc.returncode}")
    out = proc.stdout.split("\n")
    while out and out[-1] == "":
        out.pop()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, default=Path("zig-out/esen.fzm"))
    p.add_argument("--binary", type=Path, default=Path("zig-out/bin/translate"))
    p.add_argument("--update", action="store_true")
    args = p.parse_args(argv)

    if not args.model.exists():
        print(f"regress: {args.model} absent — skipping.", file=sys.stderr)
        print("         tools/fetch-model.sh es en", file=sys.stderr)
        return 0
    if not args.binary.exists():
        raise SystemExit(f"{args.binary} not found — run `zig build` first")

    pairs = load(GOLDEN, 2)
    got = translate(args.binary, args.model, [r[0] for r in pairs])

    if args.update:
        header = [l for l in GOLDEN.read_text(encoding="utf-8").splitlines()
                  if l.startswith("#") or not l.strip()]
        body = [f"{r[0]}\t{g}" for r, g in zip(pairs, got)]
        GOLDEN.write_text("\n".join(header + body) + "\n", encoding="utf-8")
        print(f"regress: re-recorded {len(body)} translations", file=sys.stderr)
        return 0

    bad = 0
    for (src, want), g in zip(((r[0], r[1]) for r in pairs), got):
        if g != want:
            bad += 1
            print(f"  CHANGED  {src}\n    was {want!r}\n    now {g!r}", file=sys.stderr)

    # Known divergences from bergamot-translator. Recorded with a diagnosis
    # rather than as golden, so that fixing one reads as a fix and not as a
    # regression.
    resolved = 0
    if DIVERGENT.exists():
        rows = load(DIVERGENT, 4)
        outs = translate(args.binary, args.model, [r[0] for r in rows])
        for (src, mine, theirs, why), g in zip(rows, outs):
            if g == theirs:
                resolved += 1
                print(f"  RESOLVED {src}\n    now matches bergamot: {g!r}\n"
                      f"    re-record it into the golden file.", file=sys.stderr)
            elif g != mine:
                bad += 1
                print(f"  CHANGED  {src}\n    was {mine!r}\n    now {g!r}\n"
                      f"    (bergamot: {theirs!r})\n    diagnosis: {why}", file=sys.stderr)

    if bad:
        print(f"regress: {bad} translation(s) changed unexpectedly.", file=sys.stderr)
        print("         If that was intended: python3 tools/regress.py --update",
              file=sys.stderr)
        return 1
    note = f", {resolved} divergence(s) resolved" if resolved else ""
    print(f"regress: {len(pairs)} golden unchanged, "
          f"{len(load(DIVERGENT, 4)) if DIVERGENT.exists() else 0} known divergence(s) "
          f"still diverging{note}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
