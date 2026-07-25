# fizh — public surface and stability

What you may depend on, what may change under you, and what a version bump
means for artifacts you have already downloaded.

## The public surface

Three things, and nothing else:

| | |
|---|---|
| `fizh_translate` | text in, translated text out |
| `fizh_can_translate` | is this pair routable, directly or via English |
| `.fzm` format version | which artifacts this build will load |

Everything reachable from the wasm module is listed by `zig build check`, which
fails the build if the export count moves. As of format version 2 the exports
are `fizh_abi_version`, `fizh_arena_bytes`, `fizh_init`, `fizh_model_load`,
`fizh_translate`, `fizh_can_translate`, `fizh_status_str`, and `memory`. The
first four are setup and are stable in the same sense as the other three; they
are separated here because they are what a *host* calls once, while
`fizh_translate` is what an application calls constantly.

`src/` is not a public surface. Region names, kernel signatures, the arena
layout, the tensor names inside a `.fzm` — all of it moves without notice.
`tools/` is a development harness, not an API.

## `fizh_abi_version`

Returns the ABI version as a `u32`. A host must call it before anything else
and refuse to proceed on a mismatch. It changes when the *shape* of a call
changes: a parameter added, a status code repurposed, a struct field moved.

Status codes are negative and additive. New ones may appear; existing ones do
not change meaning or value. A host must therefore treat any unrecognised
negative return as an error rather than switching exhaustively — and should
print `fizh_status_str` rather than the number, because that string is where
the detail lives.

## `.fzm` format version

**This is the one that affects data you have already downloaded.** The version
is a `u32` at byte 4 of the artifact. `fizh_model_load` compares it for
equality and returns `bad_version` on any mismatch — older *or* newer.

There is no forward or backward compatibility, deliberately. An artifact is
tens of megabytes of quantized weights whose interpretation depends on tensor
layout, quantization scheme and token conventions; a loader that guesses at a
version it does not know is a loader that produces fluent, wrong translations.
ADR 0008 and ADR 0015 are both cases where a plausible misreading of a real
artifact cost quality and looked fine from the outside. `bad_version` is the
only safe answer.

**What a bump means for you:** re-run `tools/fetch-model.sh` for every pair you
ship. Conversion is deterministic and takes seconds; the download is the slow
part. Nothing in a `.fzm` is user data, so there is nothing to migrate — cache
eviction is the whole upgrade path.

Version history:

| version | what changed |
|---|---|
| 1 | initial format |
| 2 | `act_quant` in the header, per-matmul `*.alpha` tensors, `tok.nonbreaking`, `sl.targets` narrowed to `u16`, `tok.charsmap` (optional) |

A version bump is a converter change and a loader change in the same commit.
`zig build convert-selftest` runs `tools/convert.py` against the real loader so
the two cannot drift.

## Arena and configuration

`fizh_arena_bytes(config)` is a pure function of the config: it returns the
exact byte count `fizh_init` will carve, before you have grown memory. The
config's meaning is stable; its *ceilings* are yours to choose.

Sizing an arena for an architecture you guessed at is the mistake
`tools/translate.zig` made — Firefox's registry ships `d_model=384, n_dec=4`
alongside the `256, 2` most pairs use, and a hardcoded config rejected the
former with `model_too_large` even though every kernel handles it. Use
`format.peekHParams` to size from the artifact, or set ceilings that cover
every pair you intend to ship.

fizh never allocates. The host owns the memory, passes it once, and fizh dies
inside it or not at all.

## What is not promised

- **Byte-identical output across versions.** fizh is greedy and deterministic
  for a given artifact and build, but a kernel change that alters rounding will
  move some translations. Measured against `bergamot-translator` the per-token
  divergence hazard is ~0.025; a fizh-to-fizh change is smaller than that and
  still not zero. If you need stability, pin the build.
- **Byte-identical output against `bergamot-translator`.** 44–46% of segments
  match exactly. The rest are inside the confidence interval, not outside it —
  see the README's table.
- **Timing.** SPEC §14's budgets are desktop proxies until T5 runs on the
  pinned reference device.
