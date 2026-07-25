// reference_engine.mjs — ground truth from bergamot-translator itself.
//
//   node tools/eval/reference_engine.mjs <bundle-dir> < input.txt > output.txt
//
// SPEC §13 T4 is only meaningful against a reference, and a published BLEU from
// somebody's README is not one: it is a number for a model version and an eval
// harness you cannot inspect. This runs the *actual* Bergamot engine over the
// *actual* model files fizh was given, in this process, right now.
//
// The engine is `bergamot-translator-worker.{js,wasm}` from browsermt's
// `latest` release. `tools/eval/fetch-reference.sh` downloads it.
//
// Same stdin/stdout contract as `zig-out/bin/translate`, so `run.py` can drive
// either one and compare like for like.

import { readFileSync, existsSync } from "node:fs";
import { createRequire } from "node:module";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ENGINE = join(HERE, "reference", "engine.cjs");

// Firefox's own decoder settings (toolkit/components/translations). Beam size 1
// matters most: fizh is greedy by invariant I7, and comparing against a beam
// search would measure the beam, not the implementation.
const CONFIG = [
  "beam-size: 1",
  "normalize: 1.0",
  "word-penalty: 0",
  "max-length-break: 128",
  "mini-batch-words: 1024",
  "workspace: 128",
  "max-length-factor: 2.0",
  "skip-cost: true",
  "cpu-threads: 0",
  "quiet: true",
  "quiet-translation: true",
  "gemm-precision: int8shiftAlphaAll",
  "alignment: soft",
  // bergamot-translator only builds a shortlist generator when this key is
  // present and non-empty -- the AlignedMemory passed to TranslationModel is
  // ignored without it. Omitting it silently ran the reference at full-vocab
  // projection, which is not what Firefox ships. ADR 0018.
  "shortlist:\n    - dummy\n    - false",
].join("\n");

function findOne(dir, pattern) {
  const { readdirSync } = createRequire(import.meta.url)("fs");
  const hit = readdirSync(dir).find((f) => pattern.test(f));
  if (!hit) throw new Error(`${dir}: no file matching ${pattern}`);
  return join(dir, hit);
}

async function main() {
  const bundle = process.argv[2];
  if (!bundle) {
    console.error("usage: reference_engine.mjs <bundle-dir> < input > output");
    process.exit(2);
  }
  if (!existsSync(ENGINE)) {
    console.error(`reference engine absent: ${ENGINE}`);
    console.error("run: tools/eval/fetch-reference.sh");
    process.exit(3);
  }

  // The glue's gemm-fallback notice goes to console.log, i.e. *stdout*, which
  // is the channel translations come out on. One stray line desynchronises the
  // whole corpus against its references.
  const realLog = console.log;
  console.log = (...a) => console.error(...a);
  const require = createRequire(import.meta.url);
  const M = require(ENGINE);
  await new Promise((r) => { M.onRuntimeInitialized = r; });
  console.log = realLog;

  const aligned = (path, alignment) => {
    const buf = readFileSync(path);
    const view = new Int8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    const mem = new M.AlignedMemory(view.byteLength, alignment);
    mem.getByteArrayView().set(view);
    return mem;
  };

  const vocabs = new M.AlignedMemoryList();
  vocabs.push_back(aligned(findOne(bundle, /^vocab\..*\.spm$/), 64));

  const model = new M.TranslationModel(
    CONFIG,
    aligned(findOne(bundle, /\.intgemm\.alphas\.bin$/), 256),
    aligned(findOne(bundle, /^lex\..*\.s2t\.bin$/), 64),
    vocabs,
    null,
  );
  const service = new M.BlockingService({ cacheSize: 0 });

  const lines = readFileSync(0, "utf-8").split("\n");
  while (lines.length && lines[lines.length - 1] === "") lines.pop();

  // Marian builds one lexical shortlist per *batch*, unioned over every source
  // token in it, so batch size changes the candidate set and therefore the
  // output. `--per-line` translates each line as its own batch, which is the
  // only configuration comparable to fizh's per-sentence shortlist. ADR 0018.
  const perLine = process.argv.includes("--per-line");
  const out = [];
  for (const group of perLine ? lines.map((l) => [l]) : [lines]) {
    const input = new M.VectorString();
    const options = new M.VectorResponseOptions();
    for (const line of group) {
      input.push_back(line);
      options.push_back({ qualityScores: false, alignment: false, html: false });
    }
    const responses = service.translate(model, input, options);
    for (let i = 0; i < responses.size(); i++) {
      out.push(responses.get(i).getTranslatedText().replace(/\n/g, " "));
    }
  }
  process.stdout.write(out.join("\n") + "\n");
}

main().catch((e) => {
  console.error(`reference_engine: ${e.message}`);
  process.exit(1);
});
