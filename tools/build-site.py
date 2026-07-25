#!/usr/bin/env python3
"""build-site.py — assemble the GitHub Pages benchmark site.

    python3 tools/build-site.py --out _site

Expects `zig build wasm` to have run and `tools/fetch-model.sh` to have
produced a `.fzm` for every pair in `pins.BENCH`. Copies the runtime, the
models, and `web/`, then writes `manifest.json`.

The manifest is what makes a screenshot attributable: the git SHA, the Zig
version, the build time, and a SHA-256 of every artifact. A benchmark result
that cannot be traced to an exact build is an anecdote.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pins import BENCH  # noqa: E402


def sh(*cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hparams(p: Path) -> dict:
    b = p.read_bytes()[:32]
    if b[:4] != b"FIZH":
        raise SystemExit(f"{p}: not a .fzm")
    u16 = lambda at: int.from_bytes(b[at:at + 2], "little")  # noqa: E731
    return {"d_model": u16(12), "ffn_dim": u16(14), "n_enc": b[16], "n_dec": b[17]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("_site"))
    ap.add_argument("--wasm", type=Path, default=Path("zig-out/wasm"))
    ap.add_argument("--models", type=Path, default=Path("zig-out"))
    ap.add_argument("--repo", default="")
    ap.add_argument("--built", default="", help="ISO timestamp; the workflow supplies it")
    args = ap.parse_args(argv)

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)

    for name in ("index.html", "style.css", "bench.js", "worker.js", "fizh.js"):
        shutil.copy(Path("web") / name, out / name)
    for name in ("fizh.baseline.wasm", "fizh.relaxed.wasm", "fizh.probe.wasm"):
        src = args.wasm / name
        if not src.exists():
            raise SystemExit(f"{src} missing — run `zig build wasm` first")
        shutil.copy(src, out / name)

    models = {}
    for src, tgt in BENCH:
        fzm = args.models / f"{src}{tgt}.fzm"
        if not fzm.exists():
            raise SystemExit(f"{fzm} missing — run tools/fetch-model.sh {src} {tgt}")
        shutil.copy(fzm, out / "models" / fzm.name)
        models[(src, tgt)] = {
            "pair": f"{src}-{tgt}",
            "file": fzm.name,
            "bytes": fzm.stat().st_size,
            "sha256": digest(fzm),
            **hparams(fzm),
        }

    # One set per shape SPEC §14 budgets separately. The pivot set is the
    # expensive one and the reason the page has to survive an OOM.
    sets = []
    if ("es", "en") in models:
        sets.append({
            "id": "esen", "label": "es→en, tiny (d=256)",
            "from": "es", "to": "en", "models": [models[("es", "en")]],
        })
    if ("en", "ar") in models:
        sets.append({
            "id": "enar", "label": "en→ar, wide (d=384)",
            "from": "en", "to": "ar", "models": [models[("en", "ar")]],
        })
    if ("es", "en") in models and ("en", "de") in models:
        sets.append({
            "id": "pivot", "label": "es→en + en→de, two slots resident (pivot)",
            "from": "es", "to": "en", "pivot": "de",
            "models": [models[("es", "en")], models[("en", "de")]],
        })

    manifest = {
        "build": {
            "sha": sh("git", "rev-parse", "HEAD") or "unknown",
            "short": sh("git", "rev-parse", "--short", "HEAD") or "unknown",
            "zig": sh("zig", "version") or "unknown",
            "built": args.built or sh("date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
            "repo": args.repo,
        },
        "runtime": {
            name: {"bytes": (out / name).stat().st_size, "sha256": digest(out / name)}
            for name in ("fizh.baseline.wasm", "fizh.relaxed.wasm", "fizh.probe.wasm")
        },
        "sets": sets,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    # Pages serves .wasm with the right type already, but .nojekyll keeps it
    # from swallowing anything that starts with an underscore.
    (out / ".nojekyll").write_text("", encoding="utf-8")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"  {out}: {len(sets)} sets, {total / 2**20:.1f} MiB total", file=sys.stderr)
    for s in sets:
        print(f"    {s['id']:<8} {s['label']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
