#!/usr/bin/env python3
"""marian.py — readers for the three files Bergamot ships. SPEC §6.

    model.{pair}.intgemm.alphas.bin   Marian binary, int8 weights
    vocab.{pair}.spm                  SentencePiece (read by convert.py)
    lex.50.50.{pair}.s2t.bin          Marian binary lexical shortlist

Every offset and every size below was verified against the real Mozilla
`esen` tiny model: both readers consume their file to the last byte.

The intgemm question, answered
------------------------------
SPEC §6 set an M2 task: "confirm whether Marian's on-disk int8 layout is already
register-tiled for intgemm and unshuffle to canonical if so."

**It is not tiled.** Every weight tensor in the shipped model carries type
`0x4101`, which in browsermt/marian-dev `src/common/types.h` is

    intgemm8 = signed_type + 1u + intgemm_type   // 0x0100 + 1 + 0x4000
               ///< Int8 quantized (not packed) matrices for intgemm

with `intgemm_type` documented as "intgemm quantized architecture agnostic
models". The register-tiled variants are distinct types — `intgemm8ssse3`,
`intgemm8avx2`, `intgemm8avx512` — and the shipped artifact uses none of them.
bergamot-translator calls intgemm's `PrepareB` at load, on the target machine,
precisely so the file can stay architecture-neutral.

So the answer is: read it row-major and transpose. No unshuffle, no width to
guess. ADR 0005's refusal to touch this path was over-cautious.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# browsermt/marian-dev src/common/types.h
TYPE_FLOAT32 = 0x0404
TYPE_INTGEMM8 = 0x4101

# Marian appends the quantization multiplier as one float after an int8
# tensor's data, then rounds the whole block up to a 256-byte boundary:
#
#     dataLength == roundUp(elems + 4, 256)
#
# Verified on the real esen model for elems = 1, 65536 and 8192000.
QUANT_MULT_BYTES = 4
DATA_ALIGN = 256

SHORTLIST_MAGIC = 0xF11A48D5013417F5


def _round_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


@dataclass
class Tensor:
    name: str
    dtype: int
    shape: tuple
    data: np.ndarray
    """For int8 tensors: `w = q / quant_mult`. None for float32."""
    quant_mult: float | None = None

    def dequantized(self) -> np.ndarray:
        if self.quant_mult is None:
            return self.data.astype(np.float32)
        return self.data.astype(np.float32) / self.quant_mult

    @property
    def scale(self) -> float:
        """fizh stores `w = q * scale`; Marian stores `w = q / quant_mult`."""
        assert self.quant_mult is not None
        return 1.0 / self.quant_mult


def read_model(path: Path) -> dict[str, Tensor]:
    """Marian binary format:

        u64 binaryFileVersion
        u64 numHeaders
        numHeaders x { u64 nameLength, u64 type, u64 shapeLength, u64 dataLength }
        names    (NUL-terminated, nameLength includes the NUL)
        shapes   (int32[shapeLength] each)
        <align to 256>
        data     (dataLength bytes each, back to back)
    """
    b = path.read_bytes()
    version, count = struct.unpack_from("<QQ", b, 0)
    if version != 1:
        raise SystemExit(f"{path}: binaryFileVersion {version}, expected 1")

    at = 16
    headers = []
    for _ in range(count):
        headers.append(struct.unpack_from("<QQQQ", b, at))
        at += 32

    names = []
    for name_len, _, _, _ in headers:
        names.append(b[at:at + name_len - 1].decode("utf-8"))
        at += name_len

    shapes = []
    for _, _, shape_len, _ in headers:
        shapes.append(struct.unpack_from("<" + "i" * shape_len, b, at))
        at += 4 * shape_len

    at = _round_up(at, DATA_ALIGN)

    out: dict[str, Tensor] = {}
    for name, (_, dtype, _, data_len), shape in zip(names, headers, shapes):
        elems = int(np.prod(shape))
        if dtype == TYPE_INTGEMM8:
            want = _round_up(elems + QUANT_MULT_BYTES, DATA_ALIGN)
            if data_len != want:
                raise SystemExit(f"{name}: dataLength {data_len}, expected {want} for {elems} elems")
            q = np.frombuffer(b, np.int8, elems, at).reshape(shape)
            # Unaligned for odd `elems`, so `struct` rather than `frombuffer`.
            mult = struct.unpack_from("<f", b, at + elems)[0]
            if not np.isfinite(mult) or mult <= 0:
                raise SystemExit(f"{name}: quantization multiplier {mult}")
            out[name] = Tensor(name, dtype, tuple(shape), q, mult)
        elif dtype == TYPE_FLOAT32:
            v = np.frombuffer(b, np.float32, elems, at).reshape(shape)
            out[name] = Tensor(name, dtype, tuple(shape), v)
        else:
            # `special:model.yml` and friends: keep the bytes, interpret later.
            out[name] = Tensor(name, dtype, tuple(shape), np.frombuffer(b, np.uint8, data_len, at))
        at += data_len

    if at != len(b):
        raise SystemExit(f"{path}: consumed {at} of {len(b)} bytes — format mismatch")
    return out


def model_config(tensors: dict[str, Tensor]) -> str:
    """The YAML Marian embeds in the artifact. The authority on topology."""
    t = tensors.get("special:model.yml")
    if t is None:
        return ""
    return bytes(t.data).rstrip(b"\0").decode("utf-8", "replace")


def read_shortlist(path: Path, vocab_size: int):
    """Marian's `BinaryShortlistGenerator` layout (src/data/shortlist.cpp):

        u64 magic, checksum, firstNum, bestNum, wordToOffsetSize, shortListsSize
        u64 wordToOffset[wordToOffsetSize]
        u32 shortLists[shortListsSize]

    Returns (offsets u32[vocab+1], targets u32[nnz], first_num, best_num).
    """
    b = path.read_bytes()
    if len(b) < 48:
        raise SystemExit(f"{path}: too small for a shortlist header")
    magic, _checksum, first_num, best_num, n_offsets, n_entries = struct.unpack_from("<6Q", b, 0)
    if magic != SHORTLIST_MAGIC:
        raise SystemExit(f"{path}: magic {magic:#x}, expected {SHORTLIST_MAGIC:#x}")
    # `wordToOffset` is normally vocab_size + 1: one start per source word plus
    # a terminating sentinel. Artifacts in Firefox's registry ship short ones —
    # en-es at exactly vocab_size, en-fa at vocab_size - 2 — where the trailing
    # source words have no candidate list at all. That is usable: those words
    # contribute nothing to the union, which is what the file says about them.
    # A shortlist *longer* than the vocabulary is a real disagreement.
    if n_offsets > vocab_size + 1:
        raise SystemExit(
            f"{path}: wordToOffsetSize {n_offsets} exceeds {vocab_size + 1} for a "
            f"vocabulary of {vocab_size} pieces; the shortlist and the "
            f"vocabulary disagree"
        )

    want = 48 + n_offsets * 8 + n_entries * 4
    if want != len(b):
        raise SystemExit(f"{path}: header implies {want} bytes, file is {len(b)}")

    offsets = np.frombuffer(b, np.uint64, n_offsets, 48)
    targets = np.frombuffer(b, np.uint32, n_entries, 48 + n_offsets * 8)

    if offsets[-1] != n_entries:
        raise SystemExit(f"{path}: final offset {offsets[-1]} != {n_entries} entries")

    if n_offsets < vocab_size + 1:
        # Pad so callers can always read offsets[id + 1]. Every word past the
        # end of the file gets an empty candidate list.
        pad = np.full(vocab_size + 1 - n_offsets, np.uint64(n_entries))
        offsets = np.concatenate([offsets, pad])
    if int(targets.max(initial=0)) >= vocab_size:
        raise SystemExit(f"{path}: target id {targets.max()} outside the vocabulary")

    return offsets.astype(np.uint32), targets, first_num, best_num


def trim_shortlist(offsets: np.ndarray, targets: np.ndarray, best: int):
    """Keeps only the first `best` candidates per source piece.

    Marian orders each source word's list by descending probability, so the
    head is the part worth keeping. SPEC §14 gives a direction 20 MB and the
    weights already take 17; `best=50` costs 3.6 MB and does not fit. This is
    the lever, and it is the only one.
    """
    if best <= 0:
        raise SystemExit("--shortlist-best must be positive")
    kept_offsets = np.zeros(len(offsets), dtype=np.uint32)
    kept = []
    for i in range(len(offsets) - 1):
        kept_offsets[i] = len(kept)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        kept.extend(targets[lo:min(hi, lo + best)])
    kept_offsets[-1] = len(kept)
    return kept_offsets, np.array(kept, dtype=np.uint32)


if __name__ == "__main__":
    import sys

    tensors = read_model(Path(sys.argv[1]))
    ints = [t for t in tensors.values() if t.dtype == TYPE_INTGEMM8]
    print(f"{len(tensors)} tensors, {len(ints)} int8")
    worst = min((int(t.data.min()) for t in ints), default=0)
    print(f"minimum int8 weight across the model: {worst}  (I5 needs > -128)")
    print("\n--- special:model.yml ---")
    print(model_config(tensors))
