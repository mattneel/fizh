#!/usr/bin/env bash
# fetch-model.sh — download a real Bergamot model and convert it to .fzm.
#
#   tools/fetch-model.sh es en            # -> zig-out/esen.fzm
#   tools/fetch-model.sh en de models/    # into a directory of your choosing
#
# The models are Mozilla's, MPL-2.0, and are NOT vendored here. Firefox
# downloads them from remote settings at runtime and so does this script; the
# GitHub mirror stores them in LFS and does not serve the objects anonymously.
set -euo pipefail

SRC="${1:-es}"
TGT="${2:-en}"
OUT="${3:-zig-out}"
PAIR="${SRC}${TGT}"
RAW="${OUT}/bergamot/${PAIR}"

mkdir -p "$RAW"

BASE=$(curl -sS https://firefox.settings.services.mozilla.com/v1/ \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["capabilities"]["attachments"]["base_url"])')

curl -sS "https://firefox.settings.services.mozilla.com/v1/buckets/main/collections/translations-models/records" \
  > "$RAW/records.json"

python3 - "$RAW" "$BASE" "$SRC" "$TGT" <<'PY'
import json, os, re, subprocess, sys
raw, base, src, tgt = sys.argv[1:5]
recs = [r for r in json.load(open(f"{raw}/records.json"))["data"]
        if r.get("fromLang") == src and r.get("toLang") == tgt]
if not recs:
    raise SystemExit(f"no {src}->{tgt} records; pick another pair")

# Several versions coexist, and they are not interchangeable. Two filters,
# in order:
#
#   1. Skip pre-releases. Version strings like "1.0a1" are alpha artifacts that
#      sit next to the "1.0" they preceded, and Firefox does not ship them.
#      They are often a few hundred bytes *smaller* than the release, so a
#      smallest-first rule selects them for 21 of the 105 pairs -- and cs-en
#      v1.0a1 translates to fluent nonsense through fizh where v1.0 is correct.
#   2. Then take the smallest, which is the "tiny" student architecture fizh
#      implements (SPEC §4.3) rather than the "base" model beside it.
sys.path.insert(0, "tools")
import pins

models = [r for r in recs if r["fileType"] == "model"]
model = pins.choose((src, tgt), models)
if model is None:
    raise SystemExit(f"no {src}->{tgt} model attachment")
version = model.get("version")
print(f"  {src}->{tgt} version {version}, model {model['attachment']['size']} bytes")

for r in recs:
    if r["fileType"] != "model" and r.get("version") != version:
        continue
    if r["fileType"] == "model" and r is not model:
        continue
    dst = os.path.join(raw, r["name"])
    subprocess.run(["curl", "-sSL", "-o", dst, base + r["attachment"]["location"]], check=True)
    got = os.path.getsize(dst)
    if got != r["attachment"]["size"]:
        raise SystemExit(f"{r['name']}: got {got}, expected {r['attachment']['size']}")
    print(f"  {r['name']}  {got} bytes")
PY

MODEL=$(ls "$RAW"/model.*.intgemm.alphas.bin)

# A bundle with distinct srcvocab/trgvocab files has one vocabulary per side.
# fizh has one table end to end, so either choice silently mistranslates.
if [ -f "$RAW"/srcvocab.*.spm ] && [ -f "$RAW"/trgvocab.*.spm ]; then
  if ! cmp -s "$RAW"/srcvocab.*.spm "$RAW"/trgvocab.*.spm; then
    echo "  ${SRC}->${TGT}: separate source and target vocabularies; fizh requires one shared vocabulary" >&2
    exit 1
  fi
fi

VOCAB=$(ls "$RAW"/vocab.*.spm 2>/dev/null || ls "$RAW"/srcvocab.*.spm)
LEX=$(ls "$RAW"/lex.*.s2t.bin 2>/dev/null || true)

PYTHONPATH=tools python3 tools/bergamot.py "${OUT}/${PAIR}.fzm" \
  --model "$MODEL" --vocab "$VOCAB" ${LEX:+--lex "$LEX"} --src "$SRC" --tgt "$TGT"

echo "  ${OUT}/${PAIR}.fzm ready"
echo
echo "  try it:  echo 'El gato está en la mesa.' | ./zig-out/bin/translate \\"
echo "             --model ${OUT}/${PAIR}.fzm --src ${SRC} --tgt ${TGT}"
