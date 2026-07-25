#!/usr/bin/env python3
"""run.py — end-to-end quality. SPEC §13 T4.

    python3 tools/eval/run.py --model es-en.fzm --src es --tgt en

Two corpora, **reported separately, never averaged** (SPEC §13):

  chat     Short messages, emoji, code-switching, typos, missing punctuation.
           Ships in this repository, because it is the register the product is
           for and nobody else publishes one.
  flores   A FLORES-200 subset, for comparability with published numbers. Not
           in this repository — it is CC-BY-SA and large. See corpora/README.md.

"A model that gains on FLORES and loses on chat register is a regression."
That sentence is the reason this script refuses to print a combined number.

Pivot pairs are evaluated end to end: pass both directions with `--model` twice
and ask for the far language. Scoring each hop separately would measure
something the user never sees.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from chrf import corpus_score  # noqa: E402

CORPORA = Path(__file__).parent / "corpora"


def load(path: Path):
    src, ref = [], []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise SystemExit(f"{path}:{lineno}: expected 'source<TAB>reference'")
        src.append(parts[0])
        ref.append(parts[1])
    return src, ref


def translate(binary: Path, models, src_lang: str, tgt_lang: str, lines):
    cmd = [str(binary)]
    for m in models:
        cmd += ["--model", str(m)]
    cmd += ["--src", src_lang, "--tgt", tgt_lang]

    proc = subprocess.run(
        cmd,
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{binary} exited {proc.returncode}")
    out = proc.stdout.split("\n")
    while out and out[-1] == "":
        out.pop()
    if len(out) != len(lines):
        raise SystemExit(f"{binary} returned {len(out)} lines for {len(lines)} inputs")
    return out, proc.stderr.strip()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", type=Path, action="append", required=True,
                   help="a .fzm; pass twice for a pivot pair")
    p.add_argument("--src", default="es")
    p.add_argument("--tgt", default="en")
    p.add_argument("--binary", type=Path, default=Path("zig-out/bin/translate"))
    p.add_argument("--corpus", action="append",
                   help="path to a source<TAB>reference TSV; repeatable")
    p.add_argument("--show", type=int, default=0, help="print this many examples")
    args = p.parse_args(argv)

    if not args.binary.exists():
        raise SystemExit(f"{args.binary} not found — run `zig build` first")

    corpora = [Path(c) for c in (args.corpus or [])]
    if not corpora:
        default = CORPORA / f"chat.{args.src}-{args.tgt}.tsv"
        if default.exists():
            corpora.append(default)
        flores = CORPORA / f"flores.{args.src}-{args.tgt}.tsv"
        if flores.exists():
            corpora.append(flores)
        else:
            print(f"note: {flores.name} absent; see {CORPORA / 'README.md'}",
                  file=sys.stderr)
    if not corpora:
        raise SystemExit("no corpora to score")

    print(f"{args.src} -> {args.tgt}, {len(args.model)} model(s)")
    results = []
    for path in corpora:
        src, ref = load(path)
        hyp, note = translate(args.binary, args.model, args.src, args.tgt, src)
        s = corpus_score(hyp, ref)
        results.append((path.stem, len(src), s))
        if note:
            print(f"  ({note})")
        for i in range(min(args.show, len(src))):
            print(f"    src {src[i]}\n    hyp {hyp[i]}\n    ref {ref[i]}\n")

    width = max(len(name) for name, _, _ in results)
    print()
    for name, n, s in results:
        print(f"  {name:<{width}}  {n:4d} segments   chrF++ {s:6.2f}")
    print("\n  Reported separately. A gain on one and a loss on the other is a")
    print("  regression, and an average would hide it (SPEC §13 T4).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
