#!/usr/bin/env python3
"""chrf.py — chrF++ (Popović 2017). SPEC §13 T4.

Character n-grams of order 1..6 plus word n-grams of order 1..2, F-score with
beta=2 so recall counts double. Whitespace is stripped before the character
n-grams are taken, which is what sacrebleu's default does; matching that default
matters more than any argument about whether it is the right one, because the
number has to be comparable to published Bergamot figures.

Self-contained and dependency-free, so `zig build eval` needs nothing but a
Python. `--selftest` checks it against cases whose answers are arithmetic.
"""

from __future__ import annotations

import sys
from collections import Counter

CHAR_ORDER = 6
WORD_ORDER = 2
BETA = 2.0
EPSILON = 1e-16


def char_ngrams(text: str, n: int) -> Counter:
    s = "".join(text.split())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def word_ngrams(text: str, n: int) -> Counter:
    w = text.split()
    return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def _match(hyp: Counter, ref: Counter):
    """Clipped overlap: a hypothesis cannot earn credit for repeating an n-gram
    more often than the reference uses it."""
    overlap = sum((hyp & ref).values())
    return overlap, sum(hyp.values()), sum(ref.values())


def sentence_stats(hyp: str, ref: str):
    """Per-order (overlap, hyp_total, ref_total), so corpus-level aggregation is
    a sum rather than an average of averages."""
    stats = []
    for n in range(1, CHAR_ORDER + 1):
        stats.append(_match(char_ngrams(hyp, n), char_ngrams(ref, n)))
    for n in range(1, WORD_ORDER + 1):
        stats.append(_match(word_ngrams(hyp, n), word_ngrams(ref, n)))
    return stats


def score(stats) -> float:
    """chrF++ from accumulated per-order statistics, as a percentage."""
    precisions, recalls = [], []
    for overlap, hyp_total, ref_total in stats:
        # An order with no n-grams on either side is skipped rather than scored
        # zero: a two-word sentence has no word trigrams, and pretending it
        # failed at them would punish short messages, which are the whole point
        # of the chat-register corpus.
        if hyp_total == 0 and ref_total == 0:
            continue
        precisions.append(overlap / hyp_total if hyp_total else 0.0)
        recalls.append(overlap / ref_total if ref_total else 0.0)

    if not precisions:
        return 0.0
    avg_p = sum(precisions) / len(precisions)
    avg_r = sum(recalls) / len(recalls)
    if avg_p + avg_r < EPSILON:
        return 0.0
    b2 = BETA * BETA
    return 100.0 * (1 + b2) * avg_p * avg_r / (b2 * avg_p + avg_r)


def corpus_score(hyps, refs) -> float:
    assert len(hyps) == len(refs), "hypothesis and reference counts differ"
    total = None
    for h, r in zip(hyps, refs):
        stats = sentence_stats(h, r)
        if total is None:
            total = [list(s) for s in stats]
        else:
            for acc, s in zip(total, stats):
                for i in range(3):
                    acc[i] += s[i]
    return score(total) if total else 0.0


def selftest() -> int:
    cases = [
        ("identical", "hola que tal", "hola que tal", 100.0),
        ("empty hypothesis", "", "hola que tal", 0.0),
        ("nothing in common", "xxxxxxx", "hola que tal", 0.0),
    ]
    failures = 0
    for name, hyp, ref, want in cases:
        got = corpus_score([hyp], [ref])
        ok = abs(got - want) < 1e-9
        print(f"  {name:<22} {got:6.2f}  expected {want:6.2f}  {'ok' if ok else 'FAIL'}")
        failures += 0 if ok else 1

    # Recall-weighted: dropping half the reference must hurt more than adding
    # the same amount of junk, because beta = 2.
    ref = "hola que tal amigo mio"
    short = corpus_score(["hola que"], [ref])
    padded = corpus_score([ref + " xxxx xxxx xxxx"], [ref])
    ok = padded > short
    print(f"  {'beta=2 favours recall':<22} {padded:6.2f} > {short:6.2f}  {'ok' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    # Monotone: a closer hypothesis must score higher.
    a = corpus_score(["hola que tal"], ["hola que tal amigo"])
    b = corpus_score(["hola que"], ["hola que tal amigo"])
    ok = a > b
    print(f"  {'monotone in overlap':<22} {a:6.2f} > {b:6.2f}  {'ok' if ok else 'FAIL'}")
    failures += 0 if ok else 1

    print("chrf:", "ok" if failures == 0 else f"{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(selftest())
