#!/usr/bin/env python3
"""reference.py — the oracle. SPEC §13 T0 and T2.

Two jobs:

    --golden test/golden/vectors.zig
        Per-op golden vectors, emitted as a Zig source file so the diffs are
        readable and the test needs no binary parser. SPEC §13 T0, gated on
        every commit.

    --compare trace.bin --model model.fzm
        Full-model differential. `tools/trace.zig` runs the real runtime and
        dumps what it computed; this reimplements the same forward pass in
        float64 and reports the error profile per stage. SPEC §13 T2.

The forward pass below emulates fizh's *quantization*, not just its arithmetic:
the same per-row absmax, the same clamp to [-127, 127], the same i32
accumulation. An oracle that computed in pure float would report the
quantization error — a number nobody can act on — instead of the implementation
error, which is the one that localizes a transposed stride.

What a passing T2 looks like
----------------------------
Not "every number agrees to 1e-6". Dynamic int8 quantization is a *step*
function: a one-ulp difference in an activation can flip a code by one, which
moves that channel by `absmax / 127` — about 1%. Six encoder layers of that
compounds to a few percent, from arithmetic that is entirely correct.

`--selfcheck` measures exactly that noise floor by diverging the oracle from
itself, and it comes out at the same magnitude as the fizh-vs-oracle number.
So the verdict below is structural rather than a single epsilon:

  * the tokenizer's ids must match **exactly**
  * the embedding must match to `f32` rounding — no quantization there
  * the **first** encoder layer must match to `f32` rounding, before chaos has
    anywhere to accumulate. A transposed stride or a swapped weight shows up
    here at ~1e0, five orders of magnitude clear of the noise
  * later layers are reported and bounded generously
  * the decoder's ids must match **exactly**. Seventeen consecutive argmaxes
    over a thousand-entry shortlist agreeing is a far stronger statement about
    the graph than any tolerance on a float

See ADR 0007.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

from convert import DESC_BYTES, HEADER_BYTES, MAGIC, fnv1a

SPACE_MARKER = 0xFF


# --------------------------------------------------------------------------
# artifact reader
# --------------------------------------------------------------------------


class Model:
    """The canonical (pre-repack) view of a `.fzm`. SPEC §6."""

    def __init__(self, path: Path):
        self.raw = path.read_bytes()
        assert self.raw[:4] == MAGIC, "not a .fzm"
        (self.version,) = struct.unpack_from("<I", self.raw, 4)
        (self.src_lang, self.tgt_lang) = struct.unpack_from("<HH", self.raw, 8)
        hp = struct.unpack_from("<HHBBBBIHHIIIIBBBBfff", self.raw, 12)
        (
            self.d_model, self.ffn_dim, self.n_enc, self.n_dec, self.n_heads,
            self.head_dim, self.vocab_size, self.max_pos, self.shortlist_width,
            self.eos_id, self.bos_id, self.unk_id, self.pad_id, self.ffn_act,
            self.prenorm, self.tied, _r, self.emb_scale, self.norm_eps,
            self.max_length_factor,
        ) = hp
        (self.count,) = struct.unpack_from("<I", self.raw, 60)

        self.index = {}
        for i in range(self.count):
            at = HEADER_BYTES + i * DESC_BYTES
            h, dtype, rank, _pad, d0, d1, d2, d3, _pad2, off, nbytes, scale_off = (
                struct.unpack_from("<QBBHIIIIIQQQ", self.raw, at)
            )
            self.index[h] = (dtype, rank, (d0, d1, d2, d3), off, nbytes, scale_off)

    def has(self, name: str) -> bool:
        return fnv1a(name) in self.index

    def raw_tensor(self, name: str):
        key = fnv1a(name)
        if key not in self.index:
            raise SystemExit(f"missing tensor {name!r}")
        return self.index[key]

    def f32(self, name: str) -> np.ndarray:
        dtype, _rank, dims, off, nbytes, _s = self.raw_tensor(name)
        assert dtype == 0, name
        return np.frombuffer(self.raw, np.float32, nbytes // 4, off).astype(np.float64)

    def u32(self, name: str) -> np.ndarray:
        dtype, _rank, dims, off, nbytes, _s = self.raw_tensor(name)
        assert dtype == 2, name
        return np.frombuffer(self.raw, np.uint32, nbytes // 4, off)

    def u16(self, name: str) -> np.ndarray:
        """Shortlist target ids are `u16` (ADR 0010)."""
        dtype, _rank, dims, off, nbytes, _s = self.raw_tensor(name)
        assert dtype == 4, name
        return np.frombuffer(self.raw, np.uint16, nbytes // 2, off)

    def u8(self, name: str) -> np.ndarray:
        dtype, _rank, dims, off, nbytes, _s = self.raw_tensor(name)
        assert dtype == 3, name
        return np.frombuffer(self.raw, np.uint8, nbytes, off)

    def quant(self, name: str):
        """Returns (int8 [N][K], scales [N]) exactly as the artifact stores it."""
        dtype, _rank, dims, off, nbytes, scale_off = self.raw_tensor(name)
        assert dtype == 1, name
        n, k = dims[0], dims[1]
        q = np.frombuffer(self.raw, np.int8, n * k, off).reshape(n, k)
        s = np.frombuffer(self.raw, np.float32, n, scale_off).astype(np.float64)
        assert q.min() >= -127, f"{name}: I5 violated in the artifact"
        return q, s


# --------------------------------------------------------------------------
# the ops, mirroring src/kernel/ref/kernels.zig
# --------------------------------------------------------------------------


def f32(x):
    """Every intermediate is rounded to float32.

    The runtime computes in float32; the oracle used to compute in float64 and
    then report the difference, which was ~3e-2 relative after six encoder
    layers. That number was not a bug — quantization is a step function, so a
    difference of one ulp in an activation flips a code by one, which is 1/127
    of that channel, and six layers of that compounds. An oracle whose noise
    floor is larger than the bugs it is looking for is not an oracle. So: same
    precision, and the only remaining difference is the order of operations."""
    return np.asarray(x, dtype=np.float64).astype(np.float32).astype(np.float64)


def quantize_rows(x: np.ndarray):
    """SPEC §7: dynamic per-row absmax to int8, f32 scale."""
    x = f32(np.atleast_2d(x))
    absmax = np.abs(x).max(axis=1)
    scale = np.where(absmax == 0, 1.0, absmax / 127.0)
    q = np.rint(f32(x / f32(scale)[:, None]))
    # np.rint is round-half-to-even, which is what `roundTiesEven` does.
    q = np.clip(q, -127, 127).astype(np.int32)
    return q, scale


def qmatmul(x: np.ndarray, w_q: np.ndarray, w_s: np.ndarray, bias=None):
    """`out[m][n] = dequant(sum_k a[m][k] * w[n][k]) + bias[n]`, i32 accumulation."""
    q, a_scale = quantize_rows(x)
    acc = q.astype(np.int64) @ w_q.astype(np.int64).T
    out = f32(acc.astype(np.float64) * f32(f32(a_scale)[:, None] * w_s[None, :]))
    if bias is not None:
        out = f32(out + bias[None, :])
    return out


def layer_norm(x: np.ndarray, gain: np.ndarray, bias: np.ndarray, eps: float):
    x = np.atleast_2d(x)
    x = f32(x)
    mean = f32(f32(x.sum(axis=1, keepdims=True)) / x.shape[1])
    d = f32(x - mean)
    var = f32(f32((d * d).sum(axis=1, keepdims=True)) / x.shape[1])
    inv = f32(1.0 / np.sqrt(f32(var + eps)))
    return f32(f32(f32(d * inv) * gain[None, :]) + bias[None, :])


def softmax(x: np.ndarray) -> np.ndarray:
    x = f32(x)
    m = x.max(axis=-1, keepdims=True)
    e = f32(np.exp(f32(x - m)))
    return f32(e / f32(e.sum(axis=-1, keepdims=True)))


def activation(x: np.ndarray, kind: int) -> np.ndarray:
    x = f32(x)
    if kind == 0:
        return np.maximum(x, 0.0)
    if kind == 1:
        c = np.sqrt(2.0 / np.pi)
        return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x**3)))
    return x / (1.0 + np.exp(-x))


def positional(d_model: int, steps: int) -> np.ndarray:
    """Marian's layout: sines in the first half of a row, cosines in the second."""
    half = d_model // 2
    pos = np.zeros((steps, d_model))
    i = np.arange(half)
    # Marian: pow(1e-4, i / (numTimescales - 1)), numTimescales = d_model/2.
    inv = 1.0 / np.power(10000.0, i / (half - 1.0))
    for p in range(steps):
        pos[p, :half] = np.sin(p * inv)
        pos[p, half:] = np.cos(p * inv)
    return pos


# --------------------------------------------------------------------------
# tokenizer, mirroring src/tok/
# --------------------------------------------------------------------------


def normalize(text: str) -> bytes:
    out = bytearray([SPACE_MARKER])
    pending = False
    for ch in text.encode("utf-8"):
        if ch in (0x20, 0x09, 0x0A, 0x0D, 0x0B, 0x0C):
            pending = True
            continue
        if pending and len(out) > 1:
            out.append(SPACE_MARKER)
        pending = False
        out.append(ch)
    return bytes(out)


def char_len(rest: bytes) -> int:
    lead = rest[0]
    if lead < 0x80:
        want = 1
    elif 0xC2 <= lead <= 0xDF:
        want = 2
    elif 0xE0 <= lead <= 0xEF:
        want = 3
    elif 0xF0 <= lead <= 0xF4:
        want = 4
    else:
        want = 1
    have = 1
    for k in range(1, want):
        if k >= len(rest) or rest[k] & 0xC0 != 0x80:
            break
        have += 1
    return min(have, len(rest))


def tokenize(m: Model, text: str):
    """Viterbi over the byte lattice. Mirrors src/tok/unigram.zig, including the
    rule that the unknown edge exists only when nothing else matched."""
    offsets = m.u32("tok.offsets")
    blob = m.u8("tok.pieces").tobytes()
    scores = m.f32("tok.scores")
    flags = m.u8("tok.flags")
    pieces = [blob[offsets[i]:offsets[i + 1]] for i in range(m.vocab_size)]
    by_bytes = {}
    for i, p in enumerate(pieces):
        if not flags[i]:
            by_bytes[p] = i
    longest = max(len(p) for p in pieces)
    unk_score = float(scores.min()) - 10.0

    s = normalize(text)
    n = len(s)
    best = [(-np.inf, 0, -1, 0)] * (n + 1)
    best[0] = (0.0, 0, 0, 0)

    for i in range(n):
        if best[i][2] < 0:
            continue
        score_i, _, _, ntok = best[i]
        matched = False
        for k in range(1, min(longest, n - i) + 1):
            pid = by_bytes.get(s[i:i + k])
            if pid is None:
                continue
            matched = True
            cand = score_i + float(scores[pid])
            if best[i + k][2] < 0 or cand > best[i + k][0]:
                best[i + k] = (cand, pid, i, ntok + 1)
        if not matched:
            k = char_len(s[i:])
            cand = score_i + unk_score
            if best[i + k][2] < 0 or cand > best[i + k][0]:
                best[i + k] = (cand, m.unk_id, i, ntok + 1)

    ids = []
    at = n
    while at > 0:
        ids.append(best[at][1])
        at = best[at][2]
    ids.reverse()
    ids.append(m.eos_id)
    return ids


# --------------------------------------------------------------------------
# forward pass, mirroring src/graph/
# --------------------------------------------------------------------------


def attn(q, k, v, heads, head_dim):
    """Bidirectional when `q` and `k` have the same length; causal masking is
    structural in the decoder, so there is none here."""
    s = q.shape[0]
    out = np.zeros_like(q)
    inv = 1.0 / np.sqrt(head_dim)
    for h in range(heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        scores = f32(f32(q[:, sl] @ k[:, sl].T) * inv)
        out[:, sl] = f32(softmax(scores) @ v[:, sl])
    assert out.shape == (s, q.shape[1])
    return out


def attn_block(m: Model, x, prefix, kv_source=None):
    wq, sq = m.quant(f"{prefix}.q.w")
    wk, sk = m.quant(f"{prefix}.k.w")
    wv, sv = m.quant(f"{prefix}.v.w")
    wo, so = m.quant(f"{prefix}.o.w")
    src = x if kv_source is None else kv_source

    q = qmatmul(x, wq, sq, m.f32(f"{prefix}.q.bias"))
    k = qmatmul(src, wk, sk, m.f32(f"{prefix}.k.bias"))
    v = qmatmul(src, wv, sv, m.f32(f"{prefix}.v.bias"))

    c = attn(q, k, v, m.n_heads, m.head_dim)
    return qmatmul(c, wo, so, m.f32(f"{prefix}.o.bias"))


def ffn_block(m: Model, x, prefix):
    w1, s1 = m.quant(f"{prefix}.w1")
    w2, s2 = m.quant(f"{prefix}.w2")
    h = qmatmul(x, w1, s1, m.f32(f"{prefix}.bias1"))
    h = activation(h, m.ffn_act)
    return qmatmul(h, w2, s2, m.f32(f"{prefix}.bias2"))


def merge(m: Model, x, branch, prefix):
    if m.prenorm == 1:
        return f32(x + branch)
    return layer_norm(f32(x + branch), m.f32(f"{prefix}.ln.gain"), m.f32(f"{prefix}.ln.bias"), m.norm_eps)


def branch_input(m: Model, x, prefix):
    if m.prenorm != 1:
        return x
    return layer_norm(x, m.f32(f"{prefix}.ln.gain"), m.f32(f"{prefix}.ln.bias"), m.norm_eps)


def embed(m: Model, ids, pos_offset=0):
    eq, es = m.quant("emb")
    scale = f32(f32(es[ids]) * m.emb_scale)
    rows = f32(eq[ids].astype(np.float64) * scale[:, None])
    pos = f32(positional(m.d_model, pos_offset + len(ids))[pos_offset:])
    return f32(rows + pos)


def encode(m: Model, ids, trace=False):
    x = embed(m, ids)
    stages = [x.copy()]
    for l in range(m.n_enc):
        p = f"enc.{l}.att"
        x = merge(m, x, attn_block(m, branch_input(m, x, p), p), p)
        p = f"enc.{l}.ffn"
        x = merge(m, x, ffn_block(m, branch_input(m, x, p), p), p)
        stages.append(x.copy())
    if m.prenorm == 1:
        x = layer_norm(x, m.f32("enc.ln.gain"), m.f32("enc.ln.bias"), m.norm_eps)
    return (x, stages) if trace else x


def build_shortlist(m: Model, src_ids, cap: int):
    seen, out = set(), []

    def add(i):
        i = int(i)
        if len(out) >= cap or i in seen:
            return
        seen.add(i)
        out.append(i)

    add(m.eos_id)
    add(m.unk_id)
    for i in m.u32("sl.frequent"):
        add(i)
    offsets = m.u32("sl.offsets")
    targets = m.u16("sl.targets")
    for s in src_ids:
        for i in targets[offsets[s]:offsets[s + 1]]:
            add(i)
    return out


def decode(m: Model, enc, src_ids, max_tgt: int, shortlist_cap: int):
    shortlist = build_shortlist(m, src_ids, shortlist_cap)
    eq, es = m.quant("emb")
    emb_bias = m.f32("emb.bias")
    rows_q = eq[shortlist]
    rows_s = es[shortlist]

    limit = min(max_tgt, int(np.ceil(m.max_length_factor * len(src_ids))) + 2)
    cell = [np.zeros((1, m.d_model)) for _ in range(m.n_dec)]

    prev, produced = m.bos_id, []
    for t in range(limit):
        x = embed(m, [prev], pos_offset=t)
        for l in range(m.n_dec):
            # SSRU (ADR 0008): c = sigmoid(f)*c + (1-sigmoid(f))*(W x); h = relu(c)
            p = f"dec.{l}.rnn"
            h = branch_input(m, x, p)
            ww, sw = m.quant(f"{p}.w")
            wf, sf = m.quant(f"{p}.wf")
            wx = qmatmul(h, ww, sw)
            fg = qmatmul(h, wf, sf, m.f32(f"{p}.bf"))
            g = f32(1.0 / (1.0 + np.exp(-fg)))
            cell[l] = f32(g * cell[l] + (1.0 - g) * wx)
            x = merge(m, x, f32(np.maximum(cell[l], 0.0)), p)

            p = f"dec.{l}.xa"
            h = branch_input(m, x, p)
            wq, sq = m.quant(f"{p}.q.w")
            wk, sk = m.quant(f"{p}.k.w")
            wv, sv = m.quant(f"{p}.v.w")
            wo, so = m.quant(f"{p}.o.w")
            q = qmatmul(h, wq, sq, m.f32(f"{p}.q.bias"))
            xk = qmatmul(enc, wk, sk, m.f32(f"{p}.k.bias"))
            xv = qmatmul(enc, wv, sv, m.f32(f"{p}.v.bias"))
            c = attn_over(q, xk, xv, m.n_heads, m.head_dim)
            x = merge(m, x, qmatmul(c, wo, so, m.f32(f"{p}.o.bias")), p)

            p = f"dec.{l}.ffn"
            x = merge(m, x, ffn_block(m, branch_input(m, x, p), p), p)

        if m.prenorm == 1:
            x = layer_norm(x, m.f32("dec.ln.gain"), m.f32("dec.ln.bias"), m.norm_eps)

        logits = qmatmul(x, rows_q, rows_s)[0] + emb_bias[shortlist]
        nxt = shortlist[int(np.argmax(logits))]
        produced.append(nxt)
        if nxt == m.eos_id:
            break
        prev = nxt
    return produced, shortlist


def attn_over(q, k, v, heads, head_dim):
    out = np.zeros_like(q)
    inv = 1.0 / np.sqrt(head_dim)
    for h in range(heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        scores = f32(f32(q[:, sl] @ k[:, sl].T) * inv)
        out[:, sl] = f32(softmax(scores) @ v[:, sl])
    return out


# --------------------------------------------------------------------------
# T2: compare against a trace from the real runtime
# --------------------------------------------------------------------------

TRACE_MAGIC = b"FZTR"


# The gap between two correct float32 implementations, before quantization
# amplifies it.
EPS_ROUNDING = 1e-5
# The chaos ceiling. Measured self-divergence is ~3e-2; a transposed stride is
# ~1e0. This sits between them, closer to the noise.
EPS_CHAOS = 2.5e-1


def compare(model_path: Path, trace_path: Path, tolerance: float) -> int:
    m = Model(model_path)
    raw = trace_path.read_bytes()
    if raw[:4] != TRACE_MAGIC:
        raise SystemExit(f"{trace_path}: not a fizh trace")

    src_len, tgt_len, d_model, sl_len, text_len = struct.unpack_from("<IIIII", raw, 8)
    at = 28
    text = raw[at:at + text_len].decode("utf-8"); at += text_len
    at = (at + 3) & ~3
    src_ids = np.frombuffer(raw, np.uint32, src_len, at); at += src_len * 4
    tgt_ids = np.frombuffer(raw, np.uint32, tgt_len, at); at += tgt_len * 4
    enc = np.frombuffer(raw, np.float32, src_len * d_model, at).reshape(src_len, d_model)
    at += src_len * d_model * 4
    (n_stages,) = struct.unpack_from("<I", raw, at); at += 4
    stages = []
    for _ in range(n_stages):
        stages.append(np.frombuffer(raw, np.float32, src_len * d_model, at).reshape(src_len, d_model))
        at += src_len * d_model * 4

    print(f"{trace_path}: {text!r}")
    failures = 0

    want_ids = tokenize(m, text)
    if list(src_ids) != want_ids:
        print(f"  tokenizer  MISMATCH\n    fizh   {list(src_ids)}\n    oracle {want_ids}")
        failures += 1
    else:
        print(f"  tokenizer  ok  ({src_len} tokens)")
        want_enc, want_stages = encode(m, want_ids, trace=True)
        failures += stage_profile(want_stages, stages)
        del tolerance

        want_tgt, _ = decode(m, want_enc, want_ids, 384, 2048)
        if list(tgt_ids) != want_tgt:
            print(f"  decoder    MISMATCH\n    fizh   {list(tgt_ids)}\n    oracle {want_tgt}")
            failures += 1
        else:
            print(f"  decoder    ok  ({tgt_len} tokens)")

    print("T2:", "ok" if failures == 0 else f"{failures} stage(s) diverged")
    return 1 if failures else 0


def stage_profile(want_stages, got_stages) -> int:
    """SPEC §13 T1/T2: the number that matters is *where* the error appears."""
    if len(want_stages) != len(got_stages):
        print(f"  stages     MISMATCH {len(got_stages)} traced, {len(want_stages)} expected")
        return 1

    print("  stage      max-abs     magnitude   relative   limit")
    bad = 0
    for i, (w, g) in enumerate(zip(want_stages, got_stages)):
        e = float(np.abs(w - g).max())
        mag = float(np.abs(w).max())
        rel = e / max(mag, 1e-30)
        # Stages 0 and 1 are the diagnostic ones: no quantization chaos has had
        # room to accumulate, so they hold to plain float32 rounding.
        limit = EPS_ROUNDING if i <= 1 else EPS_CHAOS
        label = "embed" if i == 0 else f"layer {i - 1}"
        mark = "" if rel <= limit else "  <-- FAIL"
        print(f"    {label:<8} {e:.3e}   {mag:.3e}   {rel:.3e}   {limit:.0e}{mark}")
        if rel > limit:
            bad += 1
    return bad


def profile(want, got):
    """SPEC §13: error profile per row. A jump at one position localizes a
    stride; a uniform error is a scale."""
    err = np.abs(want - got).max(axis=1)
    print("    per-token max-abs:")
    for i, e in enumerate(err):
        print(f"      [{i:3d}] {e:.3e}")


# --------------------------------------------------------------------------
# T0: golden vectors
# --------------------------------------------------------------------------


def selfcheck(model_path: Path) -> int:
    """Runs the encoder twice — float32 intermediates, then float64 — and reports
    how far apart two *identical* implementations land. That is T2's noise floor,
    and it is why `stage_profile` bounds the later layers at 2.5e-1 rather than
    at something that looks more impressive."""
    global f32
    m = Model(model_path)
    ids = tokenize(m, "hola aaaa baaa caaa")

    _, tight = encode(m, ids, trace=True)
    original = f32
    f32 = lambda x: np.asarray(x, dtype=np.float64)  # noqa: E731
    try:
        _, loose = encode(m, ids, trace=True)
    finally:
        f32 = original

    print(f"{model_path}: T2 noise floor (oracle vs itself, one ulp apart)")
    worst = 0.0
    for i, (a, b) in enumerate(zip(tight, loose)):
        e = float(np.abs(a - b).max())
        rel = e / max(float(np.abs(a).max()), 1e-30)
        worst = max(worst, rel)
        label = "embed" if i == 0 else f"layer {i - 1}"
        print(f"  {label:<8} max-abs {e:.3e}   relative {rel:.3e}")
    print(f"  worst {worst:.3e}; stage_profile bounds later layers at {EPS_CHAOS:.0e}")
    if worst > EPS_CHAOS:
        print("  the noise floor now exceeds the bound; EPS_CHAOS needs revisiting")
        return 1
    return 0


def zig_f32(values) -> str:
    return ", ".join(f"{float(v):.9e}" for v in np.ravel(values))


def zig_i8(values) -> str:
    return ", ".join(str(int(v)) for v in np.ravel(values))


def golden(out: Path) -> int:
    rng = np.random.default_rng(20260724)
    lines = [
        "//! GENERATED by tools/reference.py --golden. Do not edit.",
        "//!",
        "//! SPEC §13 T0: per-op golden vectors, checked in, gated on every commit.",
        "//! The oracle is numpy in float64; these are what it computed. If a",
        "//! kernel changes and these stop matching, the kernel is wrong until",
        "//! someone regenerates this file *and says why* in the commit message.",
        "",
    ]

    def emit(name: str, values, kind="f32"):
        body = zig_f32(values) if kind == "f32" else zig_i8(values)
        lines.append(f"pub const {name} = [_]{kind}{{ {body} }};")

    # layer_norm
    x = rng.normal(0, 2.0, 24)
    gain = rng.normal(1.0, 0.1, 24)
    bias = rng.normal(0, 0.1, 24)
    emit("layer_norm_in", x)
    emit("layer_norm_gain", gain)
    emit("layer_norm_bias", bias)
    emit("layer_norm_out", layer_norm(x, gain, bias, 1e-9)[0])

    # softmax
    s = rng.normal(0, 6.0, 32)
    emit("softmax_in", s)
    emit("softmax_out", softmax(s))

    # activations
    a = np.linspace(-6.0, 6.0, 25)
    emit("act_in", a)
    emit("act_relu", activation(a, 0))
    emit("act_gelu", activation(a, 1))
    emit("act_swish", activation(a, 2))

    # quantize: the scale and the codes must both match exactly
    qx = rng.normal(0, 3.0, 32)
    q, sc = quantize_rows(qx)
    emit("quantize_in", qx)
    emit("quantize_out", q[0], kind="i8")
    emit("quantize_scale", sc)

    # qgemv: integer accumulation, so this one is exact
    k, n = 32, 5
    aq = rng.integers(-127, 128, k).astype(np.int8)
    wq = rng.integers(-127, 128, (n, k)).astype(np.int8)
    ws = rng.uniform(0.001, 0.02, n)
    a_scale = 0.0137
    acc = aq.astype(np.int64) @ wq.astype(np.int64).T
    emit("qgemv_a", aq, kind="i8")
    emit("qgemv_w", wq, kind="i8")
    emit("qgemv_w_scales", ws)
    emit("qgemv_out", acc.astype(np.float64) * (a_scale * ws))
    lines.append(f"pub const qgemv_a_scale: f32 = {a_scale:.9e};")
    lines.append(f"pub const qgemv_k: u32 = {k};")
    lines.append(f"pub const qgemv_n: u32 = {n};")

    # positional encodings
    emit("pos_enc_16x4", positional(16, 4))
    lines.append("pub const pos_enc_d: u32 = 16;")
    lines.append("pub const pos_enc_steps: u32 = 4;")

    # transcendentals
    e = np.linspace(-20.0, 20.0, 21)
    emit("exp_in", e)
    emit("exp_out", np.exp(e))

    out.write_text("\n".join(lines) + "\n")
    print(f"{out}: {len(lines)} lines", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--golden", type=Path, help="write per-op golden vectors (T0)")
    p.add_argument("--compare", type=Path, help="a trace from tools/trace.zig (T2)")
    p.add_argument("--model", type=Path, help="the .fzm the trace was made with")
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--selfcheck", action="store_true",
                   help="measure T2's noise floor by diverging the oracle from itself")
    args = p.parse_args(argv)

    if args.selfcheck:
        if not args.model:
            p.error("--selfcheck needs --model")
        return selfcheck(args.model)
    if args.golden:
        return golden(args.golden)
    if args.compare:
        if not args.model:
            p.error("--compare needs --model")
        return compare(args.model, args.compare, args.tolerance)
    p.error("nothing to do: pass --golden or --compare")


if __name__ == "__main__":
    sys.exit(main())
