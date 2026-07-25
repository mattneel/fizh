#!/usr/bin/env python3
"""bergamot.py — a real Bergamot bundle to a real `.fzm`. SPEC §6, M2.

    python3 tools/bergamot.py out.fzm \
        --model  model.esen.intgemm.alphas.bin \
        --vocab  vocab.esen.spm \
        --lex    lex.50.50.esen.s2t.bin \
        --src es --tgt en

Weights pass through as int8, untouched. Marian's on-disk `intgemm8` is
canonical row-major (see tools/marian.py), so the only transform is a transpose
from Marian's `[in][out]` to SPEC §6's `[out][in]`, and a reciprocal on the
scale: Marian stores `w = q / quantMult`, fizh stores `w = q * scale`.

No requantization happens anywhere. The int8 values that Mozilla trained and
shipped are the int8 values fizh multiplies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import marian
import charsmap as charsmap_mod
import nonbreaking
from convert import Artifact, HParams, Vocab, emit_vocab, read_spm, short_lang

# Marian writes `x @ W`, so its matrices are [in][out]. SPEC §6 is [out][in].
ATTN = (("q", "Wq", "bq"), ("k", "Wk", "bk"), ("v", "Wv", "bv"), ("o", "Wo", "bo"))


def alpha_from(art: Artifact, name: str, tensors, marian_name: str) -> None:
    """Bergamot's static activation multiplier for the matmul using this weight.

    `fetchAlphaFromModelNodeOp` in marian-dev looks these up as
    `<weight>_QuantMultA`, so the key is the weight, not the layer. ADR 0012.
    """
    t = tensors.get(f"{marian_name}_QuantMultA")
    if t is None:
        raise SystemExit(f"{marian_name}: no _QuantMultA; the model is not an .alphas. build")
    v = float(np.asarray(t.dequantized()).ravel()[0])
    if not (v > 0):
        raise SystemExit(f"{marian_name}_QuantMultA = {v}, which cannot be a scale")
    art.add_f32(f"{name}.alpha", np.array([v], dtype=np.float32))


def quant_from(art: Artifact, name: str, t: marian.Tensor, gemm: bool) -> None:
    """Passes Marian's int8 through unchanged, broadcasting its single scale.

    `gemm=True` for a matrix Marian multiplies as `x @ W`. Its header shape is
    the *logical* `[in][out]`, but the bytes are laid out `[out][in]` row-major
    — intgemm consumes B transposed, and `intgemm8` keeps that layout even
    though it is otherwise unpacked. SPEC §6's canonical form is `[N=out][K=in]`
    row-major, so the correct read is a reshape, not a transpose. Transposing
    as well as reshaping is what made the encoder collapse to one vector by
    layer six. See ADR 0009.

    `gemm=False` for `Wemb`, which Marian stores `[vocab][dim]` and transposes
    at the point of use. That one really is what its header says.

    SPEC §7 stores one scale per output channel; Marian ships one per tensor.
    Broadcasting is exact — every channel simply gets the same number.
    """
    assert t.data.dtype == np.int8, name
    if gemm:
        rows, cols = t.shape          # logical [in][out]
        q = t.data.reshape(cols, rows)  # actual memory: [out][in]
    else:
        q = t.data
    n, k = q.shape

    if int(q.min()) <= -128:
        raise SystemExit(f"{name}: contains -128, which invariant I5 forbids")

    scales = np.full(n, t.scale, dtype=np.float32)
    art.tensors.append(marian_tensor(name, q, scales))


def marian_tensor(name, q, scales):
    from convert import DT_I8, Tensor

    n, k = q.shape
    return Tensor(name, DT_I8, (n, k, 0, 0), q.tobytes(), scales.tobytes())


def f32_from(art: Artifact, name: str, t: marian.Tensor) -> None:
    art.add_f32(name, np.asarray(t.data, dtype=np.float32).ravel())


def convert(args) -> int:
    tensors = marian.read_model(args.model)
    cfg = marian.model_config(tensors)

    def need(marian_name: str) -> marian.Tensor:
        t = tensors.get(marian_name)
        if t is None:
            raise SystemExit(f"{args.model}: missing {marian_name!r}")
        return t

    def yaml_get(key: str, default=None):
        for line in cfg.splitlines():
            if line.startswith(key + ":"):
                # Marian writes an empty setting as `key: ""`; the quotes are
                # not the value, and treating them as one silently flips the
                # model from post-norm to pre-norm.
                return line.split(":", 1)[1].strip().strip('"').strip("'")
        return default

    d_model = int(yaml_get("dim-emb"))
    ffn_dim = int(yaml_get("transformer-dim-ffn"))
    n_enc = int(yaml_get("enc-depth"))
    n_dec = int(yaml_get("dec-depth"))
    heads = int(yaml_get("transformer-heads"))
    act = {"relu": 0, "gelu": 1, "swish": 2}[yaml_get("transformer-ffn-activation", "relu")]
    dec_cell = yaml_get("dec-cell", "ssru")
    prenorm = 1 if yaml_get("transformer-preprocess", "") else 0

    if dec_cell != "ssru":
        raise SystemExit(
            f"dec-cell is {dec_cell!r}; fizh implements Bergamot's SSRU decoder "
            f"(ADR 0008). A self-attention decoder needs the KV-cache path back."
        )

    vocab = read_spm(args.vocab)

    # Special-token ids are NOT a constant across Bergamot's language pairs.
    # `esen` is `</s>`=0, `<unk>`=1; `ende` is `<unk>`=0, `<s>`=1, `</s>`=2.
    # Hardcoding them makes the decoder wait for an end-of-sequence token that
    # means something else, so it never terminates and loops until the length
    # bound. Look them up.
    def special(piece: str, fallback: int) -> int:
        want = piece.encode("utf-8")
        for i, p in enumerate(vocab.pieces):
            if p == want:
                return i
        print(f"  warning: {piece} absent from the vocabulary, using {fallback}",
              file=sys.stderr)
        return fallback

    eos = special("</s>", args.eos)
    unk = special("<unk>", args.unk)
    bos = special("<s>", eos)  # Marian starts the decoder with EOS when there is no BOS
    if "Wemb" not in tensors and "encoder_Wemb" in tensors:
        # en-ja, en-ko and en-zh-* ship `tied-embeddings-all: false` with
        # separate `encoder_Wemb`/`decoder_Wemb` and two vocabulary files
        # (srcvocab/trgvocab). fizh assumes one vocabulary end to end: the
        # tokenizer, the shortlist's source index, and the tied output
        # projection all read the same table. Supporting untied embeddings is a
        # different runtime, not a converter flag, so refuse by name rather than
        # by the symptom. ADR 0019.
        raise SystemExit(
            f"{args.model}: untied embeddings (encoder_Wemb + decoder_Wemb, "
            f"tied-embeddings-all: false). fizh requires one shared vocabulary; "
            f"this pair needs separate source and target vocabularies.")
    emb = need("Wemb")
    vocab_size = emb.shape[0]
    if len(vocab) != vocab_size:
        raise SystemExit(f"vocab has {len(vocab)} pieces, Wemb has {vocab_size} rows")

    # SPEC §9 packs a language into up to four lowercase ASCII bytes. The
    # registry's script-qualified codes get a fizh short form rather than a
    # longer field -- see convert.LANG_SHORT. Raises with the reason if a code
    # has none.
    for code in (args.src, args.tgt):
        short_lang(code)

    print(f"  {args.src}->{args.tgt}  d={d_model} ffn={ffn_dim} enc={n_enc} "
          f"dec={n_dec} heads={heads} vocab={vocab_size} cell={dec_cell} "
          f"act={yaml_get('transformer-ffn-activation')} "
          f"{'pre' if prenorm else 'post'}-norm  eos={eos} unk={unk} bos={bos}",
          file=sys.stderr)

    hp = HParams(
        d_model=d_model, ffn_dim=ffn_dim, n_enc_layers=n_enc, n_dec_layers=n_dec,
        n_heads=heads, vocab_size=vocab_size, max_pos=args.max_pos,
        shortlist_width=args.shortlist_width,
        eos_id=eos, bos_id=eos, unk_id=unk, pad_id=args.pad,
        ffn_act=act, prenorm=prenorm, tied_embeddings=1,
        act_quant=1 if args.activation_quant == "static" else 0,
        emb_scale=float(np.sqrt(d_model)), norm_eps=1e-9,
        max_length_factor=args.length_factor,
    )
    art = Artifact(args.src, args.tgt, hp)

    # Wemb is [vocab][dim] already — the one matrix that needs no transpose.
    quant_from(art, "emb", emb, gemm=False)
    if args.activation_quant == "static":
        alpha_from(art, "emb", tensors, "Wemb")
    art.add_f32("emb.bias", np.asarray(need("decoder_ff_logit_out_b").data).ravel())

    # Post-norm models carry no final norm; an identity keeps the loader uniform.
    for side in ("enc", "dec"):
        art.add_f32(f"{side}.ln.gain", np.ones(d_model))
        art.add_f32(f"{side}.ln.bias", np.zeros(d_model))

    for i in range(n_enc):
        m = f"encoder_l{i + 1}"
        for tag, w, b in ATTN:
            quant_from(art, f"enc.{i}.att.{tag}.w", need(f"{m}_self_{w}"), gemm=True)
            if args.activation_quant == "static":
                alpha_from(art, f"enc.{i}.att.{tag}.w", tensors, f"{m}_self_{w}")
            f32_from(art, f"enc.{i}.att.{tag}.bias", need(f"{m}_self_{b}"))
        f32_from(art, f"enc.{i}.att.ln.gain", need(f"{m}_self_Wo_ln_scale"))
        f32_from(art, f"enc.{i}.att.ln.bias", need(f"{m}_self_Wo_ln_bias"))
        emit_ffn(art, f"enc.{i}.ffn", need, f"{m}_ffn", tensors, args.activation_quant == "static")

    for i in range(n_dec):
        m = f"decoder_l{i + 1}"
        # SSRU (ADR 0008): rnn_W has no bias; only the forget gate does.
        quant_from(art, f"dec.{i}.rnn.w", need(f"{m}_rnn_W"), gemm=True)
        quant_from(art, f"dec.{i}.rnn.wf", need(f"{m}_rnn_Wf"), gemm=True)
        if args.activation_quant == "static":
            alpha_from(art, f"dec.{i}.rnn.w", tensors, f"{m}_rnn_W")
            alpha_from(art, f"dec.{i}.rnn.wf", tensors, f"{m}_rnn_Wf")
        f32_from(art, f"dec.{i}.rnn.bf", need(f"{m}_rnn_bf"))
        f32_from(art, f"dec.{i}.rnn.ln.gain", need(f"{m}_rnn_ffn_ln_scale"))
        f32_from(art, f"dec.{i}.rnn.ln.bias", need(f"{m}_rnn_ffn_ln_bias"))

        for tag, w, b in ATTN:
            quant_from(art, f"dec.{i}.xa.{tag}.w", need(f"{m}_context_{w}"), gemm=True)
            if args.activation_quant == "static":
                alpha_from(art, f"dec.{i}.xa.{tag}.w", tensors, f"{m}_context_{w}")
            f32_from(art, f"dec.{i}.xa.{tag}.bias", need(f"{m}_context_{b}"))
        f32_from(art, f"dec.{i}.xa.ln.gain", need(f"{m}_context_Wo_ln_scale"))
        f32_from(art, f"dec.{i}.xa.ln.bias", need(f"{m}_context_Wo_ln_bias"))
        emit_ffn(art, f"dec.{i}.ffn", need, f"{m}_ffn", tensors, args.activation_quant == "static")

    emit_vocab(art, vocab)
    # ADR 0011: the splitter is language-agnostic; the list is not.
    art.add_u8("tok.nonbreaking", nonbreaking.blob(args.src))

    # ADR 0017: ship the model's own `nmt_nfkc` table rather than reimplementing
    # Unicode normalization. Refuse an unrecognised normalizer instead of
    # silently applying the wrong rules.
    norm_name, charsmap, norm_flags = charsmap_mod.read_spec(args.vocab)
    if charsmap:
        # Gate on behaviour, not on the label. `nmt_nfkc` and `user_defined`
        # ship the same kind of compiled table and the same structural flags;
        # refusing the latter by name blocked en-bs and en-sr for nothing. What
        # would genuinely break is a normalizer that skips the dummy prefix or
        # keeps runs of whitespace, because src/tok/unigram.zig does both
        # unconditionally.
        bad = [k for k, v in norm_flags.items() if not v]
        if bad:
            raise SystemExit(
                f"normalizer {norm_name!r} disables {', '.join(bad)}; "
                f"src/tok/unigram.zig assumes all three")
        art.add_u8("tok.charsmap", charsmap)
        print(f"  normalizer: {norm_name}, {len(charsmap)} bytes", file=sys.stderr)
    else:
        print(f"  normalizer: none in {args.vocab.name}", file=sys.stderr)

    if args.lex:
        offsets, targets, first_num, best_num = marian.read_shortlist(args.lex, vocab_size)
        print(f"  shortlist: firstNum={first_num} bestNum={best_num} "
              f"entries={len(targets)}", file=sys.stderr)
        if args.shortlist_best < best_num:
            offsets, targets = marian.trim_shortlist(offsets, targets, args.shortlist_best)
            print(f"  trimmed to best {args.shortlist_best}: {len(targets)} entries",
                  file=sys.stderr)
        frequent = np.arange(min(first_num, vocab_size), dtype=np.uint32)
    else:
        offsets = np.zeros(vocab_size + 1, dtype=np.uint32)
        targets = np.zeros(0, dtype=np.uint32)
        frequent = np.arange(min(args.shortlist_frequent, vocab_size), dtype=np.uint32)

    art.add_u32("sl.offsets", offsets)
    art.add_u16("sl.targets", targets)
    art.add_u32("sl.frequent", frequent)

    size = art.write(args.out)
    print(f"  {args.out}: {size / (1 << 20):.2f} MB "
          f"({'over' if size > 64 << 20 else 'within'} the SPEC §14 64 MB budget)",
          file=sys.stderr)
    return 0


def emit_ffn(art, prefix, need, m, tensors=None, static=False):
    quant_from(art, f"{prefix}.w1", need(f"{m}_W1"), gemm=True)
    if static:
        alpha_from(art, f"{prefix}.w1", tensors, f"{m}_W1")
    f32_from(art, f"{prefix}.bias1", need(f"{m}_b1"))
    quant_from(art, f"{prefix}.w2", need(f"{m}_W2"), gemm=True)
    if static:
        alpha_from(art, f"{prefix}.w2", tensors, f"{m}_W2")
    f32_from(art, f"{prefix}.bias2", need(f"{m}_b2"))
    f32_from(art, f"{prefix}.ln.gain", need(f"{m}_ffn_ln_scale"))
    f32_from(art, f"{prefix}.ln.bias", need(f"{m}_ffn_ln_bias"))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", type=Path)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--lex", type=Path)
    p.add_argument("--src", default="es")
    p.add_argument("--tgt", default="en")
    # Positional encodings are generated up to this many steps and the loader
    # rejects a config whose max_src/max_tgt exceeds it. SPEC §4.3 sets
    # max_tgt_tokens to 3x max_src_tokens = 768 from measured target/source
    # ratios, so 512 would reject every artifact; 1024 leaves room to raise
    # max_src_tokens without re-converting.
    p.add_argument("--max-pos", type=int, default=1024)
    p.add_argument("--length-factor", type=float, default=3.0)
    p.add_argument("--activation-quant", choices=("dynamic", "static"), default="dynamic",
                   help="dynamic = SPEC §7 per-row absmax; static = the "
                        "*_QuantMultA alphas the artifact ships (ADR 0012)")
    p.add_argument("--shortlist-width", type=int, default=2048)
    p.add_argument("--shortlist-best", type=int, default=50,
                   help="candidates kept per source piece; lower to fit SPEC §14")
    p.add_argument("--shortlist-frequent", type=int, default=1024)
    p.add_argument("--eos", type=int, default=0)
    p.add_argument("--bos", type=int, default=0)
    p.add_argument("--unk", type=int, default=1)
    p.add_argument("--pad", type=int, default=0)
    return convert(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
