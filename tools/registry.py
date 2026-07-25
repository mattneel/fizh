#!/usr/bin/env python3
"""registry.py — the upstream model registry, and which record fizh uses.

    python3 tools/registry.py                 # list the selection for every pair
    python3 tools/registry.py es en           # one pair, in detail

Mozilla's models live in a public GCS bucket with a generated manifest. It
carries the two things selection needs and Firefox's remote-settings CDN does
not: an explicit `releaseStatus`, and a sha256 of every model file. ADR 0022.

Selection is **the best measured chrF++ among released records**, not the
smallest and not the largest. The registry publishes chrF++ per model, so this
is a measurement rather than a heuristic (SPEC §12.11). Largest would be wrong:
`base` beats `base-memory` on only 9 of 11 pairs, and ships a *worse* model for
`de-en` (−1.36) and `en-ko` (−1.04).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

MANIFEST = ("https://storage.googleapis.com/moz-fx-translations-data--303e-prod-"
            "translations-data/db/models.json")

# Registry language codes that are not what fizh calls them. The registry uses
# `zh`/`zh_hant`; fizh's four-byte code (SPEC §9) uses the same short forms the
# converter does.
ALIAS = {"zh": "zh-Hans", "zh_hant": "zh-Hant"}


def fetch(url: str, timeout: int = 600) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def load(cache: Path | None = None) -> dict:
    """The upstream manifest, cached on disk so a sweep does not refetch it."""
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    raw = fetch(MANIFEST)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    return json.loads(raw)


def records(manifest: dict) -> list[dict]:
    out = []
    for v in manifest["models"].values():
        out.extend(v if isinstance(v, list) else [v])
    return out


def chrfpp(r: dict) -> float:
    return ((r.get("metrics") or {}).get("flores200-plus") or {}).get("chrfpp", -1.0)


def released(v: list[dict]) -> list[dict]:
    """Records upstream calls released. `Release`, `Release Desktop` and
    `Release Android` all count; `Nightly` and null do not — that distinction is
    the whole reason this registry replaced size-guessing (ADR 0019)."""
    rel = [r for r in v if "Release" in str(r.get("releaseStatus") or "")]
    return rel or v


def choose(v: list[dict]) -> dict:
    """Best measured chrF++ among released records; smaller wins a tie."""
    return max(released(v), key=lambda r: (chrfpp(r), -size(r)))


def size(r: dict) -> int:
    return r["files"]["model"].get("uncompressedSize", 0)


def by_pair(manifest: dict) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = {}
    for r in records(manifest):
        key = (ALIAS.get(r["sourceLanguage"], r["sourceLanguage"]),
               ALIAS.get(r["targetLanguage"], r["targetLanguage"]))
        out.setdefault(key, []).append(r)
    return out


def supported(r: dict) -> str | None:
    """Why fizh cannot use this record, or None. Checked before downloading
    tens of megabytes, and it is the same reason the converter would give."""
    if "srcVocab" in r["files"] or "trgVocab" in r["files"]:
        return ("separate source and target vocabularies; fizh has one "
                "vocabulary end to end")
    return None


def download(manifest: dict, r: dict, into: Path) -> dict[str, Path]:
    """Fetch and decompress a record's files into `into`, verifying the model's
    sha256 against the manifest. Everything upstream is gzipped; the hash and
    the size describe the *decompressed* bytes.

    Files already present with the right hash are not refetched — a sweep over
    a hundred pairs must be resumable."""
    into.mkdir(parents=True, exist_ok=True)
    base = manifest["baseUrl"].rstrip("/")
    got = {}
    for kind, spec in r["files"].items():
        name = Path(spec["path"]).name
        assert name.endswith(".gz"), f"{name}: expected a gzipped upstream file"
        dst = into / name[:-3]
        want = spec.get("uncompressedHash")
        if dst.exists() and (want is None or digest(dst) == want):
            got[kind] = dst
            continue
        blob = gzip.decompress(fetch(f"{base}/{spec['path']}"))
        dst.write_bytes(blob)
        if want is not None:
            have = hashlib.sha256(blob).hexdigest()
            if have != want:
                raise SystemExit(f"{name}: sha256 {have}, manifest says {want}")
        got[kind] = dst
    return got


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    m = load()
    pairs = by_pair(m)
    if len(argv) == 2:
        v = pairs.get((argv[0], argv[1]))
        if not v:
            print(f"  {argv[0]}-{argv[1]}: not in the registry")
            return 1
        for r in v:
            mark = "  <- selected" if r is choose(v) else ""
            print(f"  {r['architecture']:<12} {str(r.get('releaseStatus')):<16} "
                  f"{size(r) / 2**20:6.1f} MiB  chrF++ {chrfpp(r):6.2f}{mark}")
        why = supported(choose(v))
        if why:
            print(f"  unsupported: {why}")
        return 0

    print(f"  generated {m['generated']}   {len(pairs)} pairs")
    for p, v in sorted(pairs.items()):
        r = choose(v)
        why = supported(r)
        print(f"  {p[0] + '-' + p[1]:<14} {r['architecture']:<12} "
              f"{size(r) / 2**20:6.1f} MiB  chrF++ {chrfpp(r):6.2f}"
              f"{'  REFUSED: ' + why if why else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
