#!/usr/bin/env python3
"""pins.py — which registry version of a model fizh uses.

Two rules, and one table of exceptions.

**Default: the smallest stable release.** "Stable" excludes any version string
containing a letter — `1.0a1` sits next to the `1.0` it preceded and is often a
few hundred bytes *smaller*, so a plain smallest-first rule selected
pre-releases for 21 of 105 pairs and made cs→en render fluent nonsense (ADR
0019). "Smallest" then picks the tiny student architecture over the base model
beside it, which is what SPEC §4.3 specifies and what the §14 weights budget is
sized for.

**Exception: `PINS`.** Some pairs' tiny artifact is simply a bad one. bg and fr
score 10–22 chrF++ below `bergamot-translator` on v1.0 and are correct on v2.0,
which is 33 MB — affordable since §14's weights budget went to 35 MB. Those
four are pinned by version rather than chosen by size.

A benchmark whose inputs drift is not a benchmark, so `BENCH` names the exact
set the Pages harness builds, and every entry is a pin.
"""

from __future__ import annotations

import re

# pair -> registry version string. See the module docstring.
PINS: dict[tuple[str, str], str] = {
    ("bg", "en"): "2.0",
    ("en", "bg"): "2.0",
    ("fr", "en"): "2.0",
    ("en", "fr"): "2.0",
}

# What the GitHub Pages benchmark builds. One d=256 direct pair, one d=384
# pair, and both halves of a pivot — the shapes SPEC §14 budgets separately.
BENCH: list[tuple[str, str]] = [
    ("es", "en"),   # d=256, the §4.3 worked example
    ("en", "de"),   # d=256, and the second half of the es->de pivot
    ("en", "ar"),   # d=384, the wider student architecture
]


def stable(record) -> bool:
    return not re.search(r"[a-zA-Z]", str(record.get("version", "")))


def choose(pair: tuple[str, str], models: list) -> dict | None:
    """The model record fizh uses for `pair`, or None if there is none."""
    if not models:
        return None
    want = PINS.get(pair)
    if want is not None:
        exact = [r for r in models if str(r.get("version")) == want]
        if exact:
            return min(exact, key=lambda r: r["attachment"]["size"])
    released = [r for r in models if stable(r)] or models
    return min(released, key=lambda r: r["attachment"]["size"])
