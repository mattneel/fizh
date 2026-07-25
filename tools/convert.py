#!/usr/bin/env python3
"""convert.py — Marian/Bergamot to `.fzm`. SPEC §6.

Three inputs, one output:

    model      Marian weights. `--npz model.npz` is the verified path: Marian
               writes it, numpy reads it, and this script does the int8
               quantization itself under the SPEC §7 contract. `--intgemm
               model.intgemm.alphas.bin` is the *unverified* path — see
               "The intgemm question" below.
    vocab      `--spm vocab.spm`, the SentencePiece model. Parsed here with a
               60-line protobuf reader rather than a dependency.
    shortlist  `--lex lex.s2t.bin`, Marian's binary lexical shortlist.
               Optional; without it every source token contributes nothing and
               only `--shortlist-frequent` most-frequent targets are candidates.

Self-test:

    python3 tools/convert.py --selftest out.fzm

builds a small synthetic artifact with no Marian input at all. `zig build
convert-selftest` feeds it to the real loader, which is what keeps this file
and `src/model/format.zig` honest about SPEC §6.

The intgemm question — answered, and the answer was no
------------------------------------------------------
This file used to claim the on-disk int8 was register-tiled and refuse to read
it. That was wrong; see ADR 0009. `intgemm8` (0x4101) is Marian's
architecture-agnostic type, tiling happens at load on the target CPU, and the
payload is plain int8 in `[out][in]` order. **Use `tools/bergamot.py`**, which
reads a real Bergamot bundle directly. This file's `--npz` path remains for
Marian checkpoints that were never quantized.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import nonbreaking

MAGIC = b"FIZH"
VERSION = 3
ALIGN = 64
HEADER_BYTES = 128
DESC_BYTES = 56

DT_F32, DT_I8, DT_U32, DT_U8, DT_U16 = 0, 1, 2, 3, 4

# tok/trie.zig: the one-byte word-boundary marker that replaces U+2581.
SPACE_MARKER = 0xFF
SPM_SPACE = "\u2581"
SPM_SPACE_BYTES = SPM_SPACE.encode("utf-8")  # E2 96 81

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
MASK64 = (1 << 64) - 1


def fnv1a(name: str) -> int:
    """Must agree with src/model/names.zig, byte for byte."""
    h = FNV_OFFSET
    for b in name.encode("utf-8"):
        h ^= b
        h = (h * FNV_PRIME) & MASK64
    return h


# --------------------------------------------------------------------------
# artifact writer
# --------------------------------------------------------------------------


@dataclass
class Tensor:
    name: str
    dtype: int
    dims: tuple
    data: bytes
    scales: bytes | None = None


@dataclass
class HParams:
    d_model: int
    ffn_dim: int
    n_enc_layers: int
    n_dec_layers: int
    n_heads: int
    vocab_size: int
    max_pos: int
    shortlist_width: int
    eos_id: int
    bos_id: int
    unk_id: int
    pad_id: int
    ffn_act: int
    prenorm: int
    tied_embeddings: int
    emb_scale: float
    norm_eps: float
    max_length_factor: float
    act_quant: int = 0

    def pack(self) -> bytes:
        head_dim = self.d_model // self.n_heads
        assert head_dim * self.n_heads == self.d_model
        blob = struct.pack(
            "<HHBBBBIHHIIIIBBBBfff",
            self.d_model,
            self.ffn_dim,
            self.n_enc_layers,
            self.n_dec_layers,
            self.n_heads,
            head_dim,
            self.vocab_size,
            self.max_pos,
            self.shortlist_width,
            self.eos_id,
            self.bos_id,
            self.unk_id,
            self.pad_id,
            self.ffn_act,
            self.prenorm,
            self.tied_embeddings,
            self.act_quant,
            self.emb_scale,
            self.norm_eps,
            self.max_length_factor,
        )
        assert len(blob) == 48, len(blob)
        return blob


class Artifact:
    def __init__(self, src_lang: str, tgt_lang: str, hp: HParams):
        # Validated by short_lang, which is where the mapping lives.
        short_lang(src_lang), short_lang(tgt_lang)
        assert src_lang != tgt_lang
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.hp = hp
        self.tensors: list[Tensor] = []

    def add_quant(self, name: str, mat: np.ndarray) -> None:
        """Symmetric per-output-channel int8. SPEC §7.

        `mat` is canonical `[N][K]`: N output channels, K contiguous.
        """
        assert mat.ndim == 2, name
        mat = np.ascontiguousarray(mat, dtype=np.float32)
        n, k = mat.shape

        absmax = np.abs(mat).max(axis=1)
        # A dead output channel would give a zero scale, and format.zig rejects
        # non-positive scales. Give it the smallest scale that keeps the row at
        # zero rather than poisoning the load.
        absmax[absmax == 0] = 1.0
        scales = (absmax / 127.0).astype(np.float32)

        q = np.rint(mat / scales[:, None])
        # I5: clamp to [-127, 127]. One line in the converter, exactly as SPEC
        # §7 says, and format.zig asserts it on the way in.
        q = np.clip(q, -127, 127).astype(np.int8)
        assert q.min() >= -127

        self.tensors.append(
            Tensor(name, DT_I8, (n, k, 0, 0), q.tobytes(), scales.tobytes())
        )

    def add_f32(self, name: str, vec: np.ndarray) -> None:
        v = np.ascontiguousarray(vec, dtype=np.float32).ravel()
        assert np.isfinite(v).all(), f"{name} has a non-finite value"
        self.tensors.append(Tensor(name, DT_F32, (len(v), 0, 0, 0), v.tobytes()))

    def add_u32(self, name: str, vec) -> None:
        v = np.ascontiguousarray(vec, dtype=np.uint32).ravel()
        self.tensors.append(Tensor(name, DT_U32, (len(v), 0, 0, 0), v.tobytes()))

    def add_u16(self, name: str, vec) -> None:
        v = np.ascontiguousarray(vec, dtype=np.uint16).ravel()
        self.tensors.append(Tensor(name, DT_U16, (len(v), 0, 0, 0), v.tobytes()))

    def add_u8(self, name: str, data: bytes) -> None:
        self.tensors.append(Tensor(name, DT_U8, (len(data), 0, 0, 0), bytes(data)))

    def write(self, path: Path) -> int:
        count = len(self.tensors)
        table_end = HEADER_BYTES + count * DESC_BYTES
        body = bytearray(align_up(table_end, ALIGN))

        descs = []
        for t in self.tensors:
            at = _append(body, t.data)
            scale_at = (1 << 64) - 1
            if t.scales is not None:
                scale_at = _append(body, t.scales)
            descs.append(
                struct.pack(
                    "<QBBHIIIIIQQQ",
                    fnv1a(t.name),
                    t.dtype,
                    _rank(t.dims),
                    0,
                    t.dims[0],
                    t.dims[1],
                    t.dims[2],
                    t.dims[3],
                    0,
                    at,
                    len(t.data),
                    scale_at,
                )
            )
            assert len(descs[-1]) == DESC_BYTES

        body[0:4] = MAGIC
        struct.pack_into("<I", body, 4, VERSION)
        struct.pack_into("<I", body, 8, pack_lang(self.src_lang))
        struct.pack_into("<I", body, 12, pack_lang(self.tgt_lang))
        body[16:64] = self.hp.pack()
        struct.pack_into("<I", body, 64, count)
        for i, d in enumerate(descs):
            at = HEADER_BYTES + i * DESC_BYTES
            body[at : at + DESC_BYTES] = d

        # A clean checkout has no zig-out/. `zig build` creates it for its own
        # outputs, but the converter runs as a build *step* whose working
        # directory may predate that, so it makes its own parent. CI on a fresh
        # runner is what found this; a developer tree always already has one.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(body))
        return len(body)


def _rank(dims: tuple) -> int:
    return 2 if dims[1] else 1


def _append(body: bytearray, data: bytes) -> int:
    at = align_up(len(body), ALIGN)
    body.extend(b"\0" * (at - len(body)))
    body.extend(data)
    return at


def align_up(x: int, a: int) -> int:
    return (x + a - 1) // a * a


# fizh's short form for a registry language code. Up to four lowercase ASCII
# bytes, big-endian, zero-padded on the left (`abi.Lang`).
#
# The registry is not all ISO 639-1: it ships `zh-Hans` and `zh-Hant`, which no
# two-letter code can express and which four bytes still cannot hold verbatim.
# fizh maps those to three-letter forms rather than inventing a longer field —
# the code is an identifier for routing, not a BCP-47 tag, and it only has to
# be unique and stable.
LANG_SHORT = {"zh-hans": "zhs", "zh-hant": "zht"}


def short_lang(s: str) -> str:
    s = s.strip()
    out = LANG_SHORT.get(s.lower(), s.lower())
    if not (1 <= len(out) <= 4) or not out.isascii() or not out.isalpha():
        raise SystemExit(
            f"language {s!r} has no fizh short form: codes are one to four "
            f"lowercase ASCII letters; add it to convert.LANG_SHORT")
    return out


def pack_lang(s: str) -> int:
    out = 0
    for c in short_lang(s):
        out = (out << 8) | ord(c)
    return out


# --------------------------------------------------------------------------
# sentencepiece
# --------------------------------------------------------------------------


@dataclass
class Vocab:
    pieces: list[bytes] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    special: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.pieces)


def read_spm(path: Path) -> Vocab:
    """Minimal protobuf reader for SentencePiece's ModelProto.

    ModelProto.pieces is field 1, repeated SentencePiece:
        1 piece  (string)
        2 score  (float)
        3 type   (enum; 1=NORMAL 2=UNKNOWN 3=CONTROL 4=USER_DEFINED
                        5=BYTE 6=UNUSED)
    Everything else in the file is skipped by wire type.
    """
    blob = path.read_bytes()
    v = Vocab()
    for field_no, payload in _pb_fields(blob):
        if field_no != 1:
            continue
        piece, score, kind = b"", 0.0, 1
        for f, p in _pb_fields(payload):
            if f == 1:
                piece = p
            elif f == 2:
                score = struct.unpack("<f", p)[0]
            elif f == 3:
                kind = p
        # Substitute on the raw UTF-8 bytes. Decoding to `str` and re-encoding
        # as latin-1 silently turns every accented character into ONE byte
        # instead of its two UTF-8 bytes, and then no Spanish word in the
        # vocabulary can ever match the source text.
        v.pieces.append(piece.replace(SPM_SPACE_BYTES, bytes([SPACE_MARKER])))
        v.scores.append(score)
        v.special.append(kind in (2, 3, 6))
    if not v.pieces:
        raise SystemExit(f"{path}: no pieces found; is this a SentencePiece model?")
    return v


def _pb_fields(blob: bytes):
    """Yields (field_number, payload). Payload is bytes for wire type 2, an int
    for varints, and raw bytes for fixed32/64."""
    i = 0
    while i < len(blob):
        tag, i = _varint(blob, i)
        field_no, wire = tag >> 3, tag & 7
        if wire == 0:
            val, i = _varint(blob, i)
            yield field_no, val
        elif wire == 1:
            yield field_no, blob[i : i + 8]
            i += 8
        elif wire == 2:
            n, i = _varint(blob, i)
            yield field_no, blob[i : i + n]
            i += n
        elif wire == 5:
            yield field_no, blob[i : i + 4]
            i += 4
        else:
            raise SystemExit(f"unsupported protobuf wire type {wire}")


def _varint(blob: bytes, i: int):
    out, shift = 0, 0
    while True:
        b = blob[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def emit_vocab(art: Artifact, v: Vocab) -> None:
    for p in v.pieces:
        if not p:
            raise SystemExit("empty vocabulary piece; format.zig rejects those")
    # tok/trie.zig walks the pieces in lexicographic byte order.
    order = sorted(range(len(v)), key=lambda i: v.pieces[i])
    for a, b in zip(order, order[1:]):
        if v.pieces[a] == v.pieces[b]:
            raise SystemExit(f"duplicate vocabulary piece {v.pieces[a]!r}")

    # The pieces are matched against UTF-8 source text. Writing them in any
    # other encoding means no accented word can ever match, and the symptom is
    # a fluent-looking mistranslation rather than an error. One check, once.
    for i, p in enumerate(v.pieces):
        probe = p.replace(bytes([SPACE_MARKER]), "\u2581".encode("utf-8"))
        try:
            probe.decode("utf-8")
        except UnicodeDecodeError as e:
            raise SystemExit(
                f"vocabulary piece {i} is not valid UTF-8 at byte offset "
                f"{e.start} of {p!r}: {e.reason}"
            )

    blob = b"".join(v.pieces)
    offsets = np.zeros(len(v) + 1, dtype=np.uint32)
    np.cumsum([len(p) for p in v.pieces], out=offsets[1:], dtype=np.uint32)

    art.add_u8("tok.pieces", blob)
    art.add_u32("tok.offsets", offsets)
    art.add_f32("tok.scores", np.array(v.scores, dtype=np.float32))
    art.add_u32("tok.order", np.array(order, dtype=np.uint32))
    art.add_u8("tok.flags", bytes(1 if s else 0 for s in v.special))


# --------------------------------------------------------------------------
# shortlist
# --------------------------------------------------------------------------


def read_lex(path: Path, vocab_size: int, best: int):
    """Marian's binary lexical shortlist.

        uint64 lemmaSize, uint64 wordToOffsetSize, uint64 ... (version dependent)

    The format has changed across Marian releases, so this refuses to guess:
    it checks the sizes it reads against `vocab_size` and bails out loudly if
    they disagree, rather than emitting a shortlist that silently degrades
    every translation.
    """
    raw = path.read_bytes()
    if len(raw) < 24:
        raise SystemExit(f"{path}: too small to be a lexical shortlist")
    n_lemma, n_offset, n_data = struct.unpack_from("<QQQ", raw, 0)
    if not (0 < n_lemma <= vocab_size + 1 and 0 < n_offset <= vocab_size + 1):
        raise SystemExit(
            f"{path}: header says lemma={n_lemma} offset={n_offset}, which does "
            f"not fit a {vocab_size}-piece vocabulary. This build of convert.py "
            f"has not been verified against your Marian version — pass "
            f"--no-shortlist to proceed without one, and see SPEC §6."
        )
    del n_data
    raise SystemExit(
        f"{path}: the Marian lexical-shortlist reader is unverified. "
        f"Pass --no-shortlist for now; see the module docstring."
    )


def frequent_shortlist(vocab_size: int, count: int):
    """The fallback: the `count` lowest ids, which in a SentencePiece vocabulary
    are the most frequent pieces. Empty per-source lists.

    This is a real shortlist, just a bad one — SPEC §8 `shortlist_build` unions
    per-source candidates with the top-N frequent, and this supplies only the
    second half. Quality will be poor until `--lex` works.
    """
    offsets = np.zeros(vocab_size + 1, dtype=np.uint32)
    targets = np.zeros(0, dtype=np.uint32)
    frequent = np.arange(min(count, vocab_size), dtype=np.uint32)
    return offsets, targets, frequent


# --------------------------------------------------------------------------
# Marian weights
# --------------------------------------------------------------------------

# Marian stores `x @ W`, so its matrices are [in][out]; the canonical `.fzm`
# layout is [out][in] (SPEC §6). Every matrix below is transposed on the way in.
ATTN_PARTS = (("q", "Wq", "bq"), ("k", "Wk", "bk"), ("v", "Wv", "bv"), ("o", "Wo", "bo"))


def convert_npz(args) -> int:
    z = np.load(args.npz)
    names = set(z.files)
    if args.dump_names:
        for n in sorted(names):
            print(f"{n:56s} {z[n].shape}")
        return 0

    def need(name: str) -> np.ndarray:
        if name not in names:
            raise SystemExit(f"{args.npz}: missing Marian parameter {name!r}")
        return np.array(z[name], dtype=np.float32)

    emb = need("Wemb")
    vocab_size, d_model = emb.shape
    n_enc = count_layers(names, "encoder_l")
    n_dec = count_layers(names, "decoder_l")
    ffn_dim = need("encoder_l1_ffn_W1").shape[1]

    v = read_spm(args.spm)
    if len(v) != vocab_size:
        raise SystemExit(
            f"vocabulary has {len(v)} pieces but Wemb has {vocab_size} rows"
        )

    hp = HParams(
        d_model=d_model,
        ffn_dim=ffn_dim,
        n_enc_layers=n_enc,
        n_dec_layers=n_dec,
        n_heads=args.heads,
        vocab_size=vocab_size,
        max_pos=args.max_pos,
        shortlist_width=args.shortlist_width,
        eos_id=args.eos,
        bos_id=args.bos,
        unk_id=args.unk,
        pad_id=args.pad,
        ffn_act={"relu": 0, "gelu": 1, "swish": 2}[args.activation],
        prenorm=1 if args.prenorm else 0,
        tied_embeddings=1,
        emb_scale=float(np.sqrt(d_model)),
        norm_eps=1e-9,
        max_length_factor=args.length_factor,
    )
    art = Artifact(args.src, args.tgt, hp)

    art.add_quant("emb", emb)
    art.add_f32(
        "emb.bias",
        need("decoder_ff_logit_out_b").ravel()
        if "decoder_ff_logit_out_b" in names
        else np.zeros(vocab_size),
    )
    for side in ("enc", "dec"):
        art.add_f32(f"{side}.ln.gain", np.ones(d_model))
        art.add_f32(f"{side}.ln.bias", np.zeros(d_model))

    for i in range(n_enc):
        m = f"encoder_l{i + 1}"
        emit_attn(art, f"enc.{i}.att", need, m + "_self", d_model)
        emit_ffn(art, f"enc.{i}.ffn", need, m + "_ffn", d_model, ffn_dim)
    for i in range(n_dec):
        m = f"decoder_l{i + 1}"
        # Bergamot's decoder is an SSRU, not self-attention (ADR 0008).
        art.add_quant(f"dec.{i}.rnn.w", need(f"{m}_rnn_W").T)
        art.add_quant(f"dec.{i}.rnn.wf", need(f"{m}_rnn_Wf").T)
        art.add_f32(f"dec.{i}.rnn.bf", need(f"{m}_rnn_bf"))
        art.add_f32(f"dec.{i}.rnn.ln.gain", need(f"{m}_rnn_ffn_ln_scale"))
        art.add_f32(f"dec.{i}.rnn.ln.bias", need(f"{m}_rnn_ffn_ln_bias"))
        emit_attn(art, f"dec.{i}.xa", need, m + "_context", d_model)
        emit_ffn(art, f"dec.{i}.ffn", need, m + "_ffn", d_model, ffn_dim)

    emit_vocab(art, v)
    offsets, targets, frequent = frequent_shortlist(vocab_size, args.shortlist_frequent)
    art.add_u32("sl.offsets", offsets)
    art.add_u16("sl.targets", targets)
    art.add_u32("sl.frequent", frequent)

    size = art.write(args.out)
    report(args.out, size, hp, len(targets))
    return 0


def count_layers(names: set, prefix: str) -> int:
    n = 0
    while any(x.startswith(f"{prefix}{n + 1}_") for x in names):
        n += 1
    if n == 0:
        raise SystemExit(f"no layers found with prefix {prefix!r}")
    return n


def emit_attn(art: Artifact, out: str, need, marian: str, d: int) -> None:
    for tag, w, b in ATTN_PARTS:
        art.add_quant(f"{out}.{tag}.w", need(f"{marian}_{w}").T)
        art.add_f32(f"{out}.{tag}.bias", need(f"{marian}_{b}"))
    art.add_f32(f"{out}.ln.gain", need(f"{marian}_Wo_ln_scale"))
    art.add_f32(f"{out}.ln.bias", need(f"{marian}_Wo_ln_bias"))


def emit_ffn(art: Artifact, out: str, need, marian: str, d: int, f: int) -> None:
    art.add_quant(f"{out}.w1", need(f"{marian}_W1").T)
    art.add_f32(f"{out}.bias1", need(f"{marian}_b1"))
    art.add_quant(f"{out}.w2", need(f"{marian}_W2").T)
    art.add_f32(f"{out}.bias2", need(f"{marian}_b2"))
    art.add_f32(f"{out}.ln.gain", need(f"{marian}_ffn_ln_scale"))
    art.add_f32(f"{out}.ln.bias", need(f"{marian}_ffn_ln_bias"))


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------

SELFTEST_PIECES = [
    "</s>", "<unk>", "\xff", "\xffhola", "\xffque", "\xfftal",
    "\xffhello", "\xffhow", "\xffare", "\xffyou", "a", "e",
    "h", "l", "o", "q", "t", "u", "!", "?",
]


def selftest(out: Path, big: bool = False) -> int:
    """Builds an artifact with no Marian input, matching test/artifact.zig's
    shape. `zig build convert-selftest` loads it with the real loader, which is
    the only thing that proves this file and `src/model/format.zig` agree.

    `--big` builds it at SPEC §4.3 scale instead — the same shape as a Bergamot
    student, with random weights — so `zig build bench` has something
    representative to measure. The translations are gibberish; the *timings*
    are not."""
    rng = np.random.default_rng(20260724)
    if big:
        d, ffn, n_enc, n_dec, heads = 256, 1536, 6, 2, 8
        pieces = synthetic_pieces(32000)
    else:
        d, ffn, n_enc, n_dec, heads = 32, 64, 2, 1, 2
        pieces = SELFTEST_PIECES
    vocab_size = len(pieces)

    hp = HParams(
        d_model=d, ffn_dim=ffn, n_enc_layers=n_enc, n_dec_layers=n_dec,
        n_heads=heads, vocab_size=vocab_size, max_pos=64, shortlist_width=64,
        eos_id=0, bos_id=0, unk_id=1, pad_id=0, ffn_act=0, prenorm=0,
        tied_embeddings=1, emb_scale=float(np.sqrt(d)), norm_eps=1e-9,
        max_length_factor=3.0,
    )
    hp.max_pos = 512 if big else 64
    art = Artifact("es", "en", hp)

    def rnd(*shape):
        return rng.normal(0.0, 0.05, size=shape).astype(np.float32)

    art.add_quant("emb", rnd(vocab_size, d))
    art.add_f32("emb.bias", np.zeros(vocab_size))
    for side in ("enc", "dec"):
        art.add_f32(f"{side}.ln.gain", np.ones(d))
        art.add_f32(f"{side}.ln.bias", np.zeros(d))

    def attn(prefix):
        for tag in ("q", "k", "v", "o"):
            art.add_quant(f"{prefix}.{tag}.w", rnd(d, d))
            art.add_f32(f"{prefix}.{tag}.bias", np.zeros(d))
        art.add_f32(f"{prefix}.ln.gain", np.ones(d))
        art.add_f32(f"{prefix}.ln.bias", np.zeros(d))

    def ffn_block(prefix):
        art.add_quant(f"{prefix}.w1", rnd(ffn, d))
        art.add_f32(f"{prefix}.bias1", np.zeros(ffn))
        art.add_quant(f"{prefix}.w2", rnd(d, ffn))
        art.add_f32(f"{prefix}.bias2", np.zeros(d))
        art.add_f32(f"{prefix}.ln.gain", np.ones(d))
        art.add_f32(f"{prefix}.ln.bias", np.zeros(d))

    for i in range(n_enc):
        attn(f"enc.{i}.att")
        ffn_block(f"enc.{i}.ffn")
    for i in range(n_dec):
        # SSRU, matching what Bergamot ships (ADR 0008).
        art.add_quant(f"dec.{i}.rnn.w", rnd(d, d))
        art.add_quant(f"dec.{i}.rnn.wf", rnd(d, d))
        art.add_f32(f"dec.{i}.rnn.bf", np.zeros(d))
        art.add_f32(f"dec.{i}.rnn.ln.gain", np.ones(d))
        art.add_f32(f"dec.{i}.rnn.ln.bias", np.zeros(d))
        attn(f"dec.{i}.xa")
        ffn_block(f"dec.{i}.ffn")

    v = Vocab(
        pieces=[p.encode("latin-1") for p in pieces],
        scores=[-20.0 + len(p) for p in pieces],
        special=[True, True] + [False] * (vocab_size - 2),
    )
    emit_vocab(art, v)
    art.add_u8("tok.nonbreaking", nonbreaking.blob("es"))

    per = 18 if big else 3
    offsets = np.arange(vocab_size + 1, dtype=np.uint32) * per
    targets = np.array(
        [(s + j * 7 + 1) % vocab_size for s in range(vocab_size) for j in range(per)],
        dtype=np.uint32,
    )
    art.add_u32("sl.offsets", offsets)
    art.add_u16("sl.targets", targets)
    art.add_u32("sl.frequent", np.arange(1024 if big else 4, dtype=np.uint32))

    size = art.write(out)
    report(out, size, hp, len(targets))
    return 0


def synthetic_pieces(n: int) -> list[str]:
    """A vocabulary shaped like a SentencePiece one: a word-boundary marker, the
    single bytes, then increasingly long pieces sharing prefixes — which is what
    exercises `tok/trie.zig`'s descend path realistically."""
    out = ["</s>", "<unk>", "\xff"]
    alpha = "abcdefghijklmnopqrstuvwxyz"
    for c in alpha:
        out.append(c)
        out.append("\xff" + c)
    i = 0
    while len(out) < n:
        a = alpha[i % 26]
        b = alpha[(i // 26) % 26]
        c = alpha[(i // 676) % 26]
        d = alpha[(i // 17576) % 26]
        out.append("\xff" + a + b + c + d)
        i += 1
    return out[:n]


def report(path: Path, size: int, hp: HParams, nnz: int) -> None:
    print(f"{path}: {size / (1 << 20):.2f} MB", file=sys.stderr)
    print(
        f"  d_model={hp.d_model} ffn={hp.ffn_dim} enc={hp.n_enc_layers} "
        f"dec={hp.n_dec_layers} heads={hp.n_heads} vocab={hp.vocab_size}",
        file=sys.stderr,
    )
    print(f"  shortlist entries: {nnz}", file=sys.stderr)
    if size > 20 << 20:
        print(
            "  WARNING: over the SPEC §14 per-direction budget of 20 MB. "
            "Lower --shortlist-best.",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path, help="output .fzm")
    p.add_argument("--npz", type=Path, help="Marian model in numpy format")
    p.add_argument("--intgemm", type=Path, help="Marian .intgemm.alphas.bin (unverified)")
    p.add_argument("--intgemm-tile", choices=("none", "ssse3", "avx2"))
    p.add_argument("--spm", type=Path, help="SentencePiece vocabulary")
    p.add_argument("--lex", type=Path, help="Marian binary lexical shortlist")
    p.add_argument("--no-shortlist", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--big", action="store_true",
                   help="with --selftest, build at SPEC §4.3 scale for benching")
    p.add_argument("--dump-names", action="store_true")
    p.add_argument("--src", default="es", help="source language code")
    p.add_argument("--tgt", default="en", help="target language code")
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--max-pos", type=int, default=512)
    p.add_argument("--activation", choices=("relu", "gelu", "swish"), default="relu")
    p.add_argument("--prenorm", action="store_true")
    p.add_argument("--length-factor", type=float, default=3.0)
    p.add_argument("--shortlist-width", type=int, default=2048)
    p.add_argument("--shortlist-best", type=int, default=18,
                   help="candidates kept per source piece; SPEC §14 is why this "
                        "is not 50")
    p.add_argument("--shortlist-frequent", type=int, default=1024)
    p.add_argument("--eos", type=int, default=0)
    p.add_argument("--bos", type=int, default=0)
    p.add_argument("--unk", type=int, default=1)
    p.add_argument("--pad", type=int, default=0)
    args = p.parse_args(argv)

    if args.selftest:
        return selftest(args.out, big=args.big)
    if args.intgemm:
        raise SystemExit(
            "use tools/bergamot.py for .intgemm.alphas.bin — it reads the real "
            "Marian binary format directly (ADR 0009). This script's --npz path "
            "is for unquantized Marian checkpoints."
        )
    if not args.npz or not args.spm:
        p.error("--npz and --spm are required (or --selftest)")
    if args.lex and not args.no_shortlist:
        read_lex(args.lex, 0, args.shortlist_best)
    return convert_npz(args)


if __name__ == "__main__":
    sys.exit(main())
