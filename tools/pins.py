#!/usr/bin/env python3
"""pins.py — what the GitHub Pages benchmark builds.

Model *selection* moved to `tools/registry.py` when the GCS manifest became
authoritative (ADR 0022): it carries an explicit `releaseStatus` and per-model
chrF++, so the version-guessing this file used to do is gone.

What remains is the benchmark set. A benchmark whose inputs drift is not a
benchmark, so this names the pairs and the page records the sha256 of whatever
each one resolved to.
"""

from __future__ import annotations

# One pair per shape SPEC §14 budgets separately: a direct pair, a wide one,
# and both halves of a pivot.
BENCH: list[tuple[str, str]] = [
    ("es", "en"),   # the §4.3 worked example, and the first half of es->de
    ("en", "de"),   # the second half of that pivot
    ("en", "ar"),   # a second architecture, for the baseline/relaxed comparison
]
