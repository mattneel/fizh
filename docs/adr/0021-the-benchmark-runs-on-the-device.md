# ADR 0021 — The benchmark runs on the device

Status: decided — harness shipped, T5 not yet run
Date: 2026-07-25
Milestone: M9

## Context

SPEC §14 pins the reference device as a 2022-class mid-tier Android. Every
number the project has published is from a desktop, carried with the caveat
"trend detection only". That caveat has survived nine milestones.

It is not a small gap. Invariant **I1** is the premise of the whole project:
*Mozilla's engine is intgemm-based and x86-tuned, while phones are ARM.* If
that is true, fizh's baseline-vs-relaxed-SIMD split is worth its complexity. If
it is false, the project's reason for existing is wrong. Nobody has measured
it on ARM.

A desktop cannot settle it, and neither can CI: GitHub's runners are x86.

## Decision

**Publish the harness to GitHub Pages and run it on the phone.**

`.github/workflows/pages.yml` builds both wasm artifacts, fetches and converts
the pinned models, assembles `_site/`, and deploys with
`upload-pages-artifact` / `deploy-pages`. `web/` is the page.

### What makes a result trustworthy

- **Pinned inputs.** `tools/pins.py` names the exact registry version of every
  model. A benchmark whose inputs drift is not a benchmark, and ADR 0019 is
  what happens when selection is implicit.
- **Attributable output.** `manifest.json` carries the git SHA, the Zig
  version, the build timestamp and a SHA-256 of every `.wasm` and every
  `.fzm`; the page stamps them into the copyable JSON. A screenshot with no
  build behind it is an anecdote.
- **The real corpus.** The page runs the same strings `tools/bench.zig`
  measures — 12-token, 120-token, 8-sentence paragraph, and the pivot — so a
  phone number sits beside the desktop proxy and means the same thing.
- **Warm and cold never blended.** Cold start is init + load + repack, timed
  once. Warm is p50 and p99 over 30 runs after 5 warmups, reported separately.
  Percentiles, never a mean (§14).

### What makes a failure a result

The page is built so the interesting outcomes survive:

- The runtime lives in a **dedicated Web Worker** (I10). A trap arrives as
  `WebAssembly.RuntimeError`, kills that worker, and is reported with
  `fatal: true` — it does not white-screen the page. `worker.onerror` rejects
  every outstanding call, so a worker that dies without replying cannot leave
  the page silently stalled.
- **OOM is caught and reported.** `memory.grow` refusing is expected on a
  phone with two 33 MB artifacts resident — SPEC §4.3 puts that case at ~76 MB
  before the host's own allocations. The result JSON records the failure, the
  set, and the device rather than losing the run.
- **Both builds on one device.** Relaxed SIMD is detected by instantiating the
  92-byte probe (ADR 0006), and a toggle forces the baseline build. That one
  comparison is the ARM thesis, and it needs both numbers from the same
  silicon.

### Size

The models are fetched **on demand**, per set, not at page load: 71 MiB total
across three sets and the wide pair alone is 33 MiB. Nobody downloads a
benchmark they did not ask to run.

### Licence

`mozilla/firefox-translations-models` is **MPL-2.0** — not CC-BY-SA-4.0, which
is what this repository claimed in two places and has been corrected.
Redistribution of the converted artifacts is permitted with notice, so the page
footer carries attribution and the licence. (FLORES *is* CC-BY-SA-4.0; that
claim was about the corpus and was right.)

## Status

The harness is verified as far as it can be without the device: the browser
host wrapper boots two slots, translates direct and pivot correctly, and
returns negative statuses as data — 94.1 MiB of wasm heap for the pivot pair on
this desktop, which is the number a phone has to beat.

**T5 itself has not run.** It needs someone to open the page on the pinned
Android. Until that happens §14's desktop-proxy caveat stands, and this ADR is
not the measurement — it is the instrument.
