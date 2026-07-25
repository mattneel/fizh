#!/usr/bin/env bash
# fetch-model.sh — download a real Mozilla model and convert it to .fzm.
#
#   tools/fetch-model.sh es en            # -> zig-out/esen.fzm
#   tools/fetch-model.sh en de models/    # into a directory of your choosing
#
# The models are Mozilla's, MPL-2.0, and are NOT vendored here. They come from
# the public GCS bucket that `mozilla/translations` publishes a manifest for;
# `tools/registry.py` decides which record and verifies its sha256. ADR 0022
# covers why that bucket rather than Firefox's remote-settings CDN, and why
# selection is by measured chrF++ rather than by size.
set -euo pipefail

SRC="${1:-es}"
TGT="${2:-en}"
OUT="${3:-zig-out}"
PAIR="${SRC}${TGT}"
RAW="${OUT}/bergamot/${PAIR}"

mkdir -p "$RAW"

# Selection, download, decompression and hash verification all live in
# registry.py so that this script and the coverage sweep cannot disagree about
# which artifact a pair means.
PYTHONPATH=tools python3 - "$RAW" "$SRC" "$TGT" "${OUT}/models.json" <<'PY'
import sys
from pathlib import Path
import registry

raw, src, tgt, cache = sys.argv[1:5]
manifest = registry.load(Path(cache))
v = registry.by_pair(manifest).get((src, tgt))
if not v:
    raise SystemExit(f"no {src}->{tgt} in the registry; see tools/registry.py")

r = registry.choose(v)
why = registry.supported(r)
if why:
    raise SystemExit(f"{src}->{tgt}: {why}")

spec = r["files"]["model"]
print(f"  {src}->{tgt} {r['architecture']} ({r.get('releaseStatus')}), "
      f"{spec['uncompressedSize'] / 2**20:.1f} MiB, "
      f"upstream chrF++ {registry.chrfpp(r):.2f}", file=sys.stderr)
for kind, path in registry.download(manifest, r, Path(raw)).items():
    print(f"  {kind:<18} {path.name}  {path.stat().st_size} bytes", file=sys.stderr)
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
