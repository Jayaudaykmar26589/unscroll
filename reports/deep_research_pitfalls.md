# Deep-research: performance pitfalls + challenge-winning plan (2026-10-28)

Source: `deep-research` workflow (5 researchers + critic + gap follow-up +
synthesis; 8 agents). Status **partial**: ink/data/agents/geometry evidence
inline; tracer-perf, rubric-gap, critic, synthesis submissions spilled, so the
tracer + rubric sections below are parent-inspected instead (marked [P]).
Every item cites an inspected file:line. Unresolved: no flatten→ink inference
script exists (resample-vs-retrain cost unmeasured); no profiling run
(wall-times/peaks unmeasured); Tier A/B eval-window sizes unpinned.

## A. Ranked pitfalls

### P0 — RAM correctness (can OOM or contradict the no-full-volume claim)
1. `src/ink/dataset.py:24` — `lru_cache(132)` holds whole frag1+frag3 stacks
   (~11.9 GB uint16 warmed; frag1 slice 103 MB, frag3 80 MB, 65 slices each).
   Docstring at `:4-5` claims no full-volume RAM. Fix: byte-budgeted cache
   or memmap + crop-before-decode.
2. `src/dataio/preprocess.py:32-34,76` — normalize upcasts to float64
   (~3x transient) + duplicate `_norm.npy`; `:24-27` full-volume load, no
   streaming. Fix: float32 streaming stats, mmap.
3. `src/eval/metrics.py:19` — cdist temp 2048×8192×3 float64 (~400 MB) per
   inner iter, O(N·M) brute force, float64 copies at `:11-12`. Fix: cKDTree
   or float32 + smaller inner chunk.
4. `src/flatten/surface.py:32` + `scripts/flatten_sheet.py:37-41` — list,
   stack, save, mean, percentile ≈ 3x volume peak RAM. Fix: preallocated
   memmap writer.
5. Per-slice full-res decode + float32 transient (~207 MB frag1) before
   further copies: `src/trace/ridge_tracker.py:126`,
   `src/trace/sam_tracker.py:53`, `scripts/render_frames.py:38`,
   `src/flatten/surface.py:24-31`. Fix: decode once, crop early, reuse.

### P1 — throughput (GPU starved, quadratic loops, serial I/O)
6. `src/ink/train.py:51-54` `num_workers=0` + sync tifffile decode starves
   GPU (no pin_memory/prefetch); `:72` `float(loss)` syncs CUDA per batch;
   `:23-36,80` per-batch val argsort over ~131k + GPU sync per batch.
7. `src/ink/dataset.py:57-59` — 16 full-slice decodes per 128px patch
   (~38k decodes/epoch cold for 2400 patches) + repeated full-crop
   normalize copies.
8. `src/eval/metrics.py:41-45` — g1 per-slice loop re-scans full `d`
   (O(S·N)) + recomputed `unique`; `:82-90,108` pure-Python `_lev` per
   glyph for CER.
9. `src/trace/sam3_tracker.py:94-116` [P] — ridge snap re-decodes full-res
   slice per call, `.astype(float)` = float64, full-frame median/std +
   `np.roll` copies, then a Python per-point search loop.
10. `src/flatten/anchor.py:46-55` — ~42 fancy-index copies + norms per
    slice; greedy chain from slice 0 drifts; `:29` clip-clamped duplicate
    endpoints; `:71` float64 stack + save duplicate.
11. `src/eval/meshref.py:8-35` — `.obj` Python line loop, per-face Python
    resample, full reload + resample on every `window_points` call.
12. `src/eval/runlog.py:37-44` — open/append/close per z; `read()` loads
    whole JSONL into a list.
13. Fetch layer: `src/dataio/download.py:114` serial urlretrieve, no
    resume/skip; `:115` second full read for sha256; `scripts/fetch_*.py`
    HEAD-then-GET, unbounded reads, full REDO on mismatch. `data/recipe.lock`
    has null bytes/sha256 with index-page URLs (no verifiable pin).

### P2 — API cost/latency multiplier
14. `src/agents/spark.py:32,26,20,43` — new client per call (TLS each),
    `.env` re-read + stat per call, whole-PNG base64 per `read_sheet`, no
    retry/max-tokens, 120 s worst-case hang per call.
15. `src/agents/translate.py:144` — serial one-call-per-reading loop
    (N×120 s worst case), no batch/memo.
16. `src/agents/harness.py:161,176,82,141` — `improve()` costs 1+3·budget
    live probes (4/7/10), re-probes current each round, prompts grow per
    round, `lessons.jsonl` unbounded (load caps 20, file never truncated).
17. `scripts/translate_demo.py:22,48` — whole file load + serial improve
    per reading, no cap/streaming flags.
18. `scripts/freeze_golden.py:19` — forces offline, so golden freezes
    `[*-STUB]` de/fr; no `--live` flag for a live-quality golden.

### P3 — demo/viewer papercuts
19. `scripts/render_frames.py:33-38` — full-res decode + float32 + full
    percentile per frame; globs all before `[:max]`; stem-index OOB risk.
20. `demo/viewer.html` — GOLDEN O(langs·modes·readings) inline, scrub
    handler swaps src with no debounce/preload, row `innerHTML` unescaped.
21. `scripts/build_viewer.py:17-20` — brittle literal replace; silent
    wrong N on template drift; `max="-1"` on empty dir.
22. `scripts/demo.py:44` — whole-file `read_bytes` SHA verify per file.

### P4 — quality-vs-speed traps (ML "performance")
23. `reports/ink20/history.json` + `src/ink/train.py:26,88` — 20 epochs
    cost ~3.3x the 6-epoch default for +0.007 F1 over ep0 (best ep5
    0.5446; 14 later epochs wasted); val AUC climbs 0.61→0.69 while F1
    flat → fixed 0.5 threshold miscalibrated; per-batch F1 mean is noisy.
    Cheapest win: threshold sweep + larger val, not more epochs.
24. `src/ink/model.py:22-55` — no-skip UNet-lite head (2x upsample, no
    skips) loses 128px detail; structural F1~0.5 ceiling. Deepening the
    2D head beats more 3D (the 6x Conv3d stem at `:38-40` dominates
    FLOPs; `:41` AdaptiveAvgPool3d slower than a fixed mean).
25. `src/flatten/surface.py:17,27` vs `src/ink/dataset.py:20` — surface
    emits depth-9, model expects depth-16: resample (interp cost +
    shift) or retrain `half_width=8` (~2x surface+stem). No inference
    script exists yet — decide + build it once.
26. `src/flatten/surface.py:24-31` — `.astype(int)` nearest-neighbor
    truncation (−0.5 px bias/jitter); use bilinear.
27. `scripts/flatten_sheet.py:34-35` — sheet row r paired to `tifs[r]`
    while the shifts log carries true z: missing-slice mispairing hazard.
28. `src/agents/translate.py:31,110-115` — 10-word offline glossary; all
    non-en offline output is `[STUB]`-prefixed English. Fine for API-free
    demo if disclosed; misleading otherwise.
29. Test gaps: `tests/test_metrics.py` toy-only (never crosses the 2048
    chunk boundary); `tests/test_translate.py:6` /
    `tests/test_harness.py:4` set `UNROLL_OFFLINE=1` at import (global
    side effect) and never cover live/timeout/parse/multiplier paths.

## B. Challenge-winning plan (Impact 30 / Tooling 25 / Demo 25 / Depth 20) [P]

Lead with what already wins: Tier B 50/50 tracking PASS on real scroll
data with honest-limits framing (Impact + Depth), SAM-3.1-as-video-tracker
+ self-improving translation harness as the Tooling story, golden
API-free replay as the Demo guarantee. Sequenced by leverage:

1. **Hours, near-zero risk:** threshold sweep for ink F1 (P4-23);
   `num_workers=2` + `pin_memory` + batched `float(loss)` (P1-6); cached
   key + reused client + `max_tokens` (P2-14); truncate lessons file on
   load (P2-16); streaming SHA verify (P3-22); viewer debounce + escape
   (P3-20). Each is a one-spot edit with a test.
2. **This week:** patch-decode diet (memmap/tile reads, crop before
   normalize, byte-budgeted cache) (P0-1, P1-7); cKDTree/float32 g1 +
   vectorized per-slice + C CER (P0-3, P1-8); batched+memoized translate
   + harness early-exit/parallel probes/fixed templates (P2-15/16);
   depth-16 decision + `src/ink/infer.py` (P4-25, closes unresolved #1);
   run Tier A/B/C eval reports = the G4 honest-limits curve.
3. **Frontier narrative for the video:** click → 50-slice Tier B trace →
   flattened sheet → F1-thresholded letters → Spark read → 6-language
   gloss, all reproducible from `golden/` with network disabled; then
   show the harness loop improving a weak translation live (or cached
   trace if quota is pinned). Suspect-region overlay in the viewer is
   the highest-visible remaining Demo item (B6/M3).
4. **Do not do:** more training epochs before the threshold sweep; live
   `--live` golden before batching exists (cost); battery mode (cut).

## C. Omitted scope
Workflow synthesis + 3 submissions spilled (contents retained, not
inspected); parent covered tracer + rubric inline instead. No profiling
run, so byte/time figures are structural (from probed geometry: frag1
6.7 GB / frag3 5.2 GB uint16 stacks, scroll window 12.8 GB) rather than
measured. Re-run `pytest` + `demo.py --offline` after any P0/P1 fix.
