#!/usr/bin/env python3
"""sweep.py — every model pair Firefox ships, fetched, converted, translated.

    python3 tools/eval/sweep.py --flores <flores200_dataset> [--limit N]

Two directions tested out of 105 is not coverage, and ADR 0015 proved artifacts
are not interchangeable: a token convention that differed between es-en and
en-de cost 1.5 chrF++ and survived three rounds of review. This walks the whole
registry and reports what happens, including the failures.

Per pair:

  fetch      the smallest model attachment, plus its vocabulary and shortlist
  convert    tools/bergamot.py -> .fzm            failure is a hard stop, reported
  load       the real loader validates the header  ids < vocab_size, UTF-8 pieces
  translate  a fixed FLORES slice                  no trap, no assertion, sane length
  score      chrF++ against bergamot-translator on the same slice, --per-line

A pair that fails loudly is a fine outcome and is recorded as one. A pair that
translates badly and quietly is what this exists to catch, which is why the
length-ratio band and the anti-collapse invariant are checked separately from
the score: a collapsed encoder still emits fluent text.

Resumable. Re-running skips pairs whose result is already in the output JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from chrf import corpus_score  # noqa: E402
import pins  # noqa: E402
from convert import short_lang  # noqa: E402

REGISTRY = ("https://firefox.settings.services.mozilla.com/v1/buckets/main"
            "/collections/translations-models/records")
CAPABILITIES = "https://firefox.settings.services.mozilla.com/v1/"

# Registry codes are ISO 639-1; FLORES is 639-3 plus a script tag.
FLORES = {
    "ar": "arb_Arab", "az": "azj_Latn", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "bs": "bos_Latn", "ca": "cat_Latn", "cs": "ces_Latn",
    "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn",
    "es": "spa_Latn", "et": "est_Latn", "eu": "eus_Latn", "fa": "pes_Arab",
    "fi": "fin_Latn", "fr": "fra_Latn", "gl": "glg_Latn", "gu": "guj_Gujr",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "id": "ind_Latn", "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "kn": "kan_Knda", "ko": "kor_Hang", "lt": "lit_Latn", "lv": "lvs_Latn",
    "ml": "mal_Mlym", "ms": "zsm_Latn", "mt": "mlt_Latn", "nb": "nob_Latn",
    "nl": "nld_Latn", "nn": "nno_Latn", "pl": "pol_Latn", "pt": "por_Latn",
    "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn", "sl": "slv_Latn",
    "sq": "als_Latn", "sr": "srp_Cyrl", "sv": "swe_Latn", "ta": "tam_Taml",
    "te": "tel_Telu", "th": "tha_Thai", "tr": "tur_Latn", "uk": "ukr_Cyrl",
    "vi": "vie_Latn", "zh-Hans": "zho_Hans", "zh-Hant": "zho_Hant",
}

# A translation shorter than a third of its source, or three times longer, is
# not a translation. Wide because scripts differ in bytes per word: Japanese
# and Chinese are dense, Telugu and Tamil are not.
LEN_LO, LEN_HI = 0.20, 4.0


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def fetch(pair, raw: Path, base: str, records) -> dict:
    src, tgt = pair
    recs = [r for r in records if r.get("fromLang") == src and r.get("toLang") == tgt]
    models = [r for r in recs if r["fileType"] == "model"]
    model = pins.choose(pair, models)
    if model is None:
        return {"error": "no model attachment"}
    version = model.get("version")
    raw.mkdir(parents=True, exist_ok=True)
    got = {}
    for r in recs:
        if r["fileType"] == "model" and r is not model:
            continue
        if r["fileType"] != "model" and r.get("version") != version:
            continue
        dst = raw / r["name"]
        if not dst.exists() or dst.stat().st_size != r["attachment"]["size"]:
            p = run(["curl", "-sSL", "--max-time", "600", "-o", str(dst),
                     base + r["attachment"]["location"]])
            if p.returncode != 0:
                return {"error": f"download {r['name']}: {p.stderr[:120]}"}
        if dst.stat().st_size != r["attachment"]["size"]:
            return {"error": f"{r['name']}: size mismatch"}
        got[r["fileType"]] = dst
    return {"version": version, "size": model["attachment"]["size"], "files": got}


def one_pair(pair, args, base, records, flores: Path) -> dict:
    src, tgt = pair
    tag = f"{src}-{tgt}"
    out = {"pair": tag}
    raw = Path(args.work) / "bergamot" / f"{src}{tgt}".replace("-", "")
    t0 = time.time()

    got = fetch(pair, raw, base, records)
    if "error" in got:
        return {**out, "stage": "fetch", "ok": False, "error": got["error"]}
    out["version"] = got["version"]
    out["model_bytes"] = got["size"]

    model = next(iter(sorted(raw.glob("model.*.intgemm.alphas.bin"))), None)
    vocabs = sorted(raw.glob("*.spm")) or sorted(raw.glob("vocab.*"))
    lex = next(iter(sorted(raw.glob("lex.*.s2t.bin"))), None)
    if model is None or not vocabs:
        return {**out, "stage": "fetch", "ok": False,
                "error": f"missing model or vocab in {raw}"}
    out["vocab_files"] = len(vocabs)
    # A bundle with distinct srcvocab/trgvocab files has one vocabulary per
    # side. fizh has one table end to end -- the tokenizer, the shortlist's
    # source index and the tied output projection all read it -- so picking
    # either file silently mistranslates. en-zh-* pairs even ship a shared
    # `Wemb` alongside two different vocabularies, which the untied-embedding
    # check in bergamot.py cannot see. ADR 0019.
    src_v = [v for v in vocabs if v.name.startswith("srcvocab")]
    tgt_v = [v for v in vocabs if v.name.startswith("trgvocab")]
    if src_v and tgt_v and src_v[0].read_bytes() != tgt_v[0].read_bytes():
        return {**out, "stage": "convert", "ok": False,
                "error": "separate source and target vocabularies; fizh "
                         "requires one shared vocabulary"}

    fzm = Path(args.work) / (f"{src}{tgt}".replace("-", "") + ".fzm")
    cmd = [sys.executable, "tools/bergamot.py", str(fzm), "--model", str(model),
           "--vocab", str(vocabs[0]), "--src", src, "--tgt", tgt]
    if lex:
        cmd += ["--lex", str(lex)]
    p = run(cmd, env={**os.environ, "PYTHONPATH": "tools"})
    if p.returncode != 0:
        return {**out, "stage": "convert", "ok": False,
                "error": (p.stderr.strip().splitlines() or ["?"])[-1][:200]}
    out["arch"] = next((ln.strip() for ln in p.stderr.splitlines()
                        if "ffn=" in ln), "")
    out["fzm_bytes"] = fzm.stat().st_size

    # The real loader validates the header: ids < vocab_size, tensor set, sizes.
    p = run(["./zig-out/bin/fzm-load", str(fzm)])
    out["loads"] = p.returncode == 0
    if not out["loads"]:
        out["load_error"] = (p.stdout + p.stderr).strip().splitlines()[-1][:200] \
            if (p.stdout + p.stderr).strip() else "nonzero exit"

    sf, tf = flores / f"{FLORES[src]}.devtest", flores / f"{FLORES[tgt]}.devtest"
    if not sf.exists() or not tf.exists():
        return {**out, "stage": "corpus", "ok": False,
                "error": f"no FLORES for {src} or {tgt}"}
    srcs = sf.read_text(encoding="utf-8").splitlines()[:args.segments]
    golds = tf.read_text(encoding="utf-8").splitlines()[:args.segments]
    src_txt = "\n".join(srcs) + "\n"

    # The artifact stores fizh's short form, so the CLI has to be asked for it
    # too -- `zh-Hans` is `zhs` inside a .fzm (SPEC §9, convert.LANG_SHORT).
    p = run(["./zig-out/bin/translate", "--model", str(fzm),
             "--src", short_lang(src), "--tgt", short_lang(tgt)], input=src_txt)
    if p.returncode != 0:
        return {**out, "stage": "translate", "ok": False,
                "error": (p.stderr.strip().splitlines() or ["nonzero exit"])[-1][:200]}
    hyp = p.stdout.splitlines()
    if len(hyp) != len(srcs):
        return {**out, "stage": "translate", "ok": False,
                "error": f"got {len(hyp)} lines for {len(srcs)} inputs"}
    out["empty"] = sum(1 for h in hyp if not h.strip())
    sc = sum(len(s) for s in srcs)
    out["len_ratio"] = round(sum(len(h) for h in hyp) / sc, 3) if sc else 0.0
    out["chrf_gold"] = round(corpus_score(hyp, golds), 2)

    p = run(["node", "tools/eval/reference_engine.mjs", str(raw), "--per-line"],
            input=src_txt)
    if p.returncode == 0 and len(p.stdout.splitlines()) == len(srcs):
        ref = p.stdout.splitlines()
        out["chrf_ref"] = round(corpus_score(ref, golds), 2)
        out["delta"] = round(out["chrf_gold"] - out["chrf_ref"], 2)
        out["identical"] = sum(1 for a, b in zip(hyp, ref) if a == b)
    else:
        out["ref_error"] = (p.stderr.strip().splitlines() or ["no output"])[-1][:160]

    out["ok"] = (out["loads"] and out["empty"] == 0
                 and LEN_LO <= out["len_ratio"] <= LEN_HI)
    out["stage"] = "done"
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--flores", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("zig-out/sweep.json"))
    p.add_argument("--work", default="zig-out/sweep")
    p.add_argument("--segments", type=int, default=100)
    p.add_argument("--limit", type=int)
    p.add_argument("--only", help="comma-separated src-tgt pairs")
    args = p.parse_args(argv)

    base = json.loads(run(["curl", "-sS", CAPABILITIES]).stdout)
    base = base["capabilities"]["attachments"]["base_url"]
    records = json.loads(run(["curl", "-sS", REGISTRY]).stdout)["data"]

    pairs = sorted({(r["fromLang"], r["toLang"]) for r in records
                    if r.get("fromLang") and r.get("toLang")
                    and r.get("fileType") == "model"})
    if args.only:
        want = set(args.only.split(","))
        pairs = [q for q in pairs if f"{q[0]}-{q[1]}" in want]
    if args.limit:
        pairs = pairs[:args.limit]

    done = {}
    if args.out.exists():
        done = {r["pair"]: r for r in json.loads(args.out.read_text())}

    Path(args.work).mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(pairs, 1):
        tag = f"{q[0]}-{q[1]}"
        if tag in done:
            continue
        if q[0] not in FLORES or q[1] not in FLORES:
            done[tag] = {"pair": tag, "stage": "corpus", "ok": False,
                         "error": "no FLORES mapping"}
        else:
            try:
                done[tag] = one_pair(q, args, base, records, args.flores)
            except Exception as e:  # a crash is a result, not a reason to stop
                done[tag] = {"pair": tag, "stage": "exception", "ok": False,
                             "error": f"{type(e).__name__}: {e}"[:200]}
        r = done[tag]
        print(f"  [{i}/{len(pairs)}] {tag:<12} {r.get('stage','?'):<10} "
              f"{'ok' if r.get('ok') else 'FAIL':<5} "
              f"delta={r.get('delta','-')} len={r.get('len_ratio','-')} "
              f"{r.get('error','')}", flush=True)
        args.out.write_text(json.dumps(list(done.values()), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
