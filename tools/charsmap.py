#!/usr/bin/env python3
"""charsmap.py — SentencePiece's `nmt_nfkc` normalizer, read from the artifact.

The `.spm` vocabulary carries a `normalizer_spec` naming `nmt_nfkc` and a
`precompiled_charsmap`: a darts-clone double-array trie over UTF-8 byte
sequences, followed by a pool of NUL-terminated replacement strings. Matching
is longest-prefix; a hit appends its replacement, a miss copies one UTF-8
character through.

This is the reference implementation, used by the converter to build the
runtime's table and by the tests to check `src/tok/charsmap.zig` against it.
Nothing here is a reimplementation of NFKC: the mapping is whatever the model
was trained with, read out of the model.

Layout:

    u32le trie_size
    trie_size bytes   darts-clone units, u32le each
    remainder         replacement pool, NUL-terminated strings
"""

from __future__ import annotations

import struct
from pathlib import Path


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    v = shift = 0
    while True:
        b = buf[i]
        i += 1
        v |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return v, i


def read_spec(spm_path: Path) -> tuple[str, bytes]:
    """Returns (normalizer name, precompiled_charsmap) from a .spm file."""
    raw = spm_path.read_bytes()
    i = 0
    while i < len(raw):
        tag, i = _varint(raw, i)
        ln, i = _varint(raw, i)
        if tag >> 3 == 3:  # normalizer_spec
            sub, j, name, charsmap = raw[i:i + ln], 0, "", b""
            while j < len(sub):
                t, j = _varint(sub, j)
                sl, j = _varint(sub, j)
                if t >> 3 == 1:
                    name = sub[j:j + sl].decode()
                elif t >> 3 == 2:
                    charsmap = sub[j:j + sl]
                j += sl
            return name, charsmap
        i += ln
    return "", b""


class Charsmap:
    """Longest-prefix rewriter over a darts-clone trie."""

    def __init__(self, blob: bytes):
        (trie_size,) = struct.unpack_from("<I", blob, 0)
        assert trie_size % 4 == 0, "trie blob is not a whole number of units"
        self.units = list(struct.unpack_from(f"<{trie_size // 4}I", blob, 4))
        self.pool = blob[4 + trie_size:]

    # darts-clone unit accessors, from `darts.h`.
    @staticmethod
    def _has_leaf(u: int) -> bool:
        return bool((u >> 8) & 1)

    @staticmethod
    def _value(u: int) -> int:
        return u & 0x7FFFFFFF

    @staticmethod
    def _label(u: int) -> int:
        return u & 0x800000FF

    @staticmethod
    def _offset(u: int) -> int:
        return (u >> 10) << ((u & 0x200) >> 6)

    def longest(self, key: bytes) -> tuple[int, int]:
        """Longest match at the head of `key`. Returns (pool offset, bytes
        consumed), or (-1, 0) when nothing matches."""
        node = 0
        unit = self.units[node]
        node ^= self._offset(unit)
        best_value, best_len = -1, 0
        for i, b in enumerate(key):
            node ^= b
            unit = self.units[node]
            if self._label(unit) != b:
                break
            node ^= self._offset(unit)
            if self._has_leaf(unit):
                best_value, best_len = self._value(self.units[node]), i + 1
        return best_value, best_len

    def replacement(self, offset: int) -> bytes:
        end = self.pool.index(b"\0", offset)
        return self.pool[offset:end]

    def normalize(self, text: bytes) -> bytes:
        out, i, n = bytearray(), 0, len(text)
        while i < n:
            value, length = self.longest(text[i:])
            if length > 0:
                out += self.replacement(value)
                i += length
                continue
            # No rule: copy one whole UTF-8 character through unchanged.
            step = 1
            b = text[i]
            if b >= 0xF0:
                step = 4
            elif b >= 0xE0:
                step = 3
            elif b >= 0xC0:
                step = 2
            out += text[i:i + step]
            i += step
        return bytes(out)


if __name__ == "__main__":
    import sys

    name, blob = read_spec(Path(sys.argv[1]))
    cm = Charsmap(blob)
    print(f"  normalizer {name}, charsmap {len(blob)} bytes, "
          f"{len(cm.units)} trie units, {len(cm.pool)} bytes of replacements")
    for line in sys.stdin:
        raw = line.rstrip("\n").encode("utf-8")
        got = cm.normalize(raw)
        flag = "" if got == raw else "  CHANGED"
        print(f"    {got.decode('utf-8', 'replace')}{flag}")
