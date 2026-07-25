#!/usr/bin/env bash
# fetch-model.sh — download a real Bergamot model and convert it to .fzm.
#
#   tools/fetch-model.sh es en            # -> zig-out/esen.fzm
#   tools/fetch-model.sh en de models/    # into a directory of your choosing
#
# The models are Mozilla's, CC-BY-SA-4.0, and are NOT vendored here. Firefox
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
import json, os, subprocess, sys
raw, base, src, tgt = sys.argv[1:5]
recs = [r for r in json.load(open(f"{raw}/records.json"))["data"]
        if r.get("fromLang") == src and r.get("toLang") == tgt]
if not recs:
    raise SystemExit(f"no {src}->{tgt} records; pick another pair")

# Several versions coexist. Take the smallest model — that is the "tiny"
# student architecture fizh implements (SPEC §4.3).
models = [r for r in recs if r["fileType"] == "model"]
model = min(models, key=lambda r: r["attachment"]["size"])
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
VOCAB=$(ls "$RAW"/vocab.*.spm)
LEX=$(ls "$RAW"/lex.*.s2t.bin 2>/dev/null || true)

PYTHONPATH=tools python3 tools/bergamot.py "${OUT}/${PAIR}.fzm" \
  --model "$MODEL" --vocab "$VOCAB" ${LEX:+--lex "$LEX"} --src "$SRC" --tgt "$TGT"

echo "  ${OUT}/${PAIR}.fzm ready"
echo
echo "  try it:  echo 'El gato está en la mesa.' | ./zig-out/bin/translate \\"
echo "             --model ${OUT}/${PAIR}.fzm --src ${SRC} --tgt ${TGT}"
