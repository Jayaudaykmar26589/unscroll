# Unscroll: Click-to-Unroll for Rolled-Sheet CT Volumes

*   **Author:** Jayachandiran U
*   **Status:** Approved 2026-10-27 (re-audit fixes applied; residuals: data IDs, VRAM actuals, credit math close via Day-0/5 probes)
*   **Date:** 2026-10-27
*   **Canonical:** this file. `../design_draft.md` is superseded (bannered).

--------------------------------------------------------------------------------

## Overview

Unscroll traces a single rolled layer through a CT slice stack from one
click, flattens it, and recovers the markings on it. This plan is
scroll-only: Herculaneum scrolls (recover Greek ink). Battery cells are
explicitly cut to future work (see Non-Goals).

Built for the Meta Global AI Developer Hackathon (Indie track, Push the
Frontier theme; submissions close Nov 4 â€” verify exact cutoff and the
reported $150 API credits in the portal). Meta models throughout: SAM 3.1
tracking, Spark reading, Wonder orchestration.

All training and evaluation uses publicly available data â€” no synthetic
generator. Scroll side: Vesuvius Challenge open data (CC BY-NC):
labeled fragments and released segments with ink labels for training,
plus small streamed Scroll 1 crops (1cm crop, OME-Zarr partial reads)
for tracking eval. Volumes are never redistributed; the
repo ships code plus a pinned, reproducible data recipe.

The core bet is treating the slice stack as video: SAM 3.1 tracks the
clicked layer slice-by-slice, geometric checks (winding, spacing,
smoothness) catch layer jumps, and stacked traced curves parameterize the
flattened surface. A Day-2 go/no-go decides SAM vs a named classical
fallback, so the demo never depends on an unproven tracker.

The laptop constraint shapes everything: one RTX 4060 (assume 8GB VRAM
until confirmed), stream-don't-download volumes, patch-based training
and inference, mixed precision, local data footprint under 100GB.

## The Problem

Reading a crushed scroll today needs dense manual segmentation slice by
slice. Unscroll tests whether one click plus video tracking can
replace dense manual tracing, with public ink labels and released
segment surfaces as ground truth making the claim checkable instead of
vibes-based â€” on hardware a solo builder actually owns.

## Goals

Difficulty tiers (public data, fixed Day 0): Tier A = fragment with
separated layers; Tier B = fragment with touching layers; Tier C =
streamed scroll crop, crushed.

*   G1: One click per layer traces >=90% of sampled surface points
    within 2 voxels 3D Euclidean distance of the released segment
    surface, on Tiers A-B. Sampling: 1 point per voxel along each
    slice curve, every slice. Voxel size recorded at probe. Also
    reported: per-slice pass rate (>=80% of slices with >=80%
    in-tolerance) and p95 distance (tail metric).
*   G2: Jump rate <=2 layer jumps per 100 slices on Tiers A-B
    (absolute threshold, after-only; jump = excursion onto a different
    segment-surface ID for >=3 consecutive slices).
    Tier C: manual breakage log (visible jumps per 100 slices,
    disclosed, no ID oracle claimed).
*   G3: Letter recovery >=70% exact-glyph accuracy vs public ink labels
    on Tier A val sheets, + character error rate (CER) alongside.
    Charset: Greek/Coptic Unicode block as labeled. Segmentation:
    connected components on the thresholded letter mask. Exclusions:
    illegible/damaged glyphs marked in labels (counted separately).
    n = all Tier A val glyphs (count recorded in report).
*   G4: Honest-limits curve: G1-G3 reported across Tiers A-C with
    per-slice breakage logs, showing where the method breaks.
*   G5: Reproducible public-data recipe released by Day 7: pinned
    dataset URLs/versions, download + preprocess scripts, local
    footprint under 100GB, environment pin. No volume redistribution
    (respects CC BY-NC).
*   G6: Golden demo runs API-free: precomputed trace + cached readings,
    so the video survives quota/outage.

## Non-Goals

*   Synthetic data generation of any kind (deleted from the plan).
*   Full-volume downloads: no single artifact over 10GB local; stream
    or crop everything bigger (OME-Zarr partial reads).
*   Redistribution of Vesuvius volumes or derived full-volume copies.
*   Reading unpublished scroll text (demo re-reads already-published
    regions only).
*   Battery mode in the 8-day plan: cut to future work (CUJ 2 deleted;
    one-line mention only). Revisit post-hackathon with its own N,
    thresholds, and defect metric.
*   Multi-script text: Greek only; no charset abstraction this sprint.
*   Production hardening: no CI-gated release, no multi-user viewer.
*   Beating winner tools head-to-head on real data (out of scope for
    8 days; winner predictions used as reference only).

## Open Questions

*   **Q1 (resources):** Solo or team? RTX 4060 VRAM size? Meta Model API
    access + quota confirmed? Assumes worst case (solo, 8GB, unconfirmed)
    until answered.
*   **Q2 (data picks):** Which exact public artifacts (fragment IDs,
    segment IDs + ink-label set, scroll crop coordinates)? Selection
    criteria (Day 0 probe): each artifact <10GB, total <100GB,
    Tier A fragment has a released segment surface, Tier B fragment
    shows touching layers, ink labels exist for train. IDs recorded in
    `data/recipe.lock` Day 0; doc updated with the picks. Region =
    fragment ID + bounding box; split 80/20 by region, seed 0; Scroll 1
    crop must not spatially overlap train regions (coordinate check in
    preprocess). Weights-from-NC-data posture: research-only release,
    disclose, no relicense (see NFR).
*   **Q3 (SAM go/no-go):** Decided: SAM must hold a Tier B touching
    layer for 50 consecutive slices (fixed z-window recorded in probe)
    with >=45/50 slices within 2 voxels of reference, by end of Day 2;
    else the classical fallback (spec'd stub) gets built Days 3-4 and
    becomes primary.
*   **Q4 (clicks baseline):** What "dense manual" number does clicks/cm^2
    beat? Proposal: time the author tracing one layer manually in an
    open tool on Day 0 (n=1, disclosed as such).

## Critical User Journeys (CUJs)

### CUJ 1: Click a scroll layer, get Greek letters

*   **Given**: Scroll crop cached locally (Tier A/B), viewer open,
    tracing engine warm.
*   **When**: User clicks one pixel on a layer in a slice.
*   **Then**: Layer traced through the stack (suspect regions flagged),
    flat sheet rendered with letter suggestions + confidences;
    per-slice log + metrics recorded under a run ID. (3D spiral is
    pre-rendered eye candy, not a live view.)

### CUJ 2: deleted (battery cut to future work; see Non-Goals).

## Technical Plan (Architecture)

Five components, one data recipe. Everything runs on the laptop except
explicitly capped API calls.

### C1. Data layer (Days 0-1) â€” stream, pin, cache

*   **Sources (pinned Day 0, Q2):** Vesuvius fragments + released
    segments with ink labels (train/val, split 80/20 by region, seed 0;
    region = fragment ID + bounding box); 1cm Scroll 1 crop for tracking
    eval (coordinate-checked to not overlap train regions).
*   **Access:** `vesuvius` Python lib + OME-Zarr partial reads; only
    crops/tiles touch local disk. No artifact over 10GB, total under
    100GB. All URLs/versions/hashes/byte-counts pinned in
    `data/recipe.lock` (schema: JSON list of
    {artifact, url, version, sha256, bytes, role}).
*   **Preprocess:** 16-bit to float via (x - mean) / std with per-volume
    stats saved to `data/norm_stats.json`; tile index; train/val split
    file. CPU-only, cached once. Cache invalidated when recipe.lock
    hash changes (re-download + re-preprocess).

### C2. Tracing engine (Days 2-3) â€” video tracking + jump guards

*   **Track:** slice stack as video frames. Click = exactly one
    positive point prompt + previous-frame mask (no negative points,
    no multi-click; correction = re-prompt loop). Weights: SAM 3.1
    open-weights video checkpoint, latest stable release at Day 0,
    exact URL + sha256 pinned in recipe.lock; named fallback if
    unavailable: SAM 2.1 video checkpoint (same pin procedure). SAM
    mask per frame â†’ Zhang-Suen
    skeletonize + spur prune (<3 voxels, error bound <1 voxel, keeps
    the 2-voxel G1 tolerance valid). Curve format: JSONL per slice
    {z, points [[x,y]...], origin, shift}. API SAM for live demo only
    (capped: <=200 frames/run, recorded).
*   **Stream in z-chunks:** 32 slices, fp16, batch 1, memory bank last 8
    frames; 8-slice overlap between chunks with last-8-frame state
    handoff; winding computed on the stitched curve (no seams).
    Actuals probed and recorded Day 0.
*   **Jump guards (`src/trace/guards.yaml`, all logged per slice):**
    cumulative winding non-decreasing with per-slice reversal <5
    degrees; gap within 30% of trailing-10-slice median; kink =
    adjacent-segment angle change >60 degrees. Tier C: guards
    advisory-only (folds break winding). Violation â†’ auto re-prompt
    (max 3 per region; region = 10-slice window) â†’ else flag suspect
    into `reports/suspect_shortlist.json` for manual review (no auto
    Spark API loop).
*   **Fallback:** click-seeded classical ridge tracker, specified now
    (same curve JSONL interface), built ONLY if the Day-2 go/no-go
    fails (Days 3-4). Golden demo trace precomputed either way.

### C3. Flatten + ink (Day 4) â€” anchored parameterization

*   **De-drift:** resample each slice curve to 512 arc-length points;
    global origin alignment (minimize total inter-slice shift, not
    sequential template-match); log per-slice origin/shift; shear
    metric = |median glyph aspect - 1| on Tier A val (fail if >0.25).
    Anchored parameterization is the only flattener (mesh fallback
    deleted).
*   **Surface volume:** sample Â±4 voxels along the curve normal.
*   **Ink CNN (timeboxed, fixed):** 4-level 3D U-Net tiny (channels
    16-32-64-128, 3^3 kernels, ~0.5M params, recorded actual), 64^3
    voxel patches, Dice + BCE loss, AdamW lr 3e-4, effective batch 8
    (micro 2 x 4 accum), fp16, 50 epochs max, early stop on val Dice
    patience 5, 1 retrain max. Checkpoints to `checkpoints/`. Tiled
    inference (64^3 tile, 16 overlap). Baseline reference: compare
    against released winner ink-model predictions on the same Tier A
    sheets where available (reference only, not a dependency).

### C4. Agent layer (Day 5) â€” capped, audited, optional-each

*   **Spark (only agent in plan):** letter suggestions + confidences
    on flat sheet; trace quality check. Cut to future work: Wonder
    orchestration, Voice, Image (kept as one-line mentions for the
    Tooling narrative, not built).
*   API-off degraded mode via `UNROLL_OFFLINE=1` + `src/agents/stubs.py`
    fixtures; golden demo locked to cached outputs at freeze (Day 6
    AM, after first Tier A/B eval â€” never an unevaluated freeze).

### C5. Eval + viewer (Days 6-7) â€” logs first, pixels second

*   **Eval harness first:** run IDs, per-slice pass/fail logs, tier
    reports for G1-G4. Started Days 0-2 alongside C1.
*   **Viewer:** 2 live views only (slice + trace overlay; flat sheet +
    letters), static HTML + canvas 2D reading run JSON. 3D spiral +
    metrics panels pre-rendered PNGs from the golden run via
    `scripts/render_frames.py` (no live 3D).

```text
[Public sources] --stream/crop--> [Local cache <100GB, recipe.lock]
        | preprocess (CPU, cached once; lock-hash change = invalidate)
        v
[Click] --> [C2 Tracing: SAM/video + jump guards + stub fallback]
        |  ^-- re-prompt loop (max 3 / 10-slice region)
        |  +-- suspect flags --> reports/suspect_shortlist.json (manual)
        | per-slice curves + run ID
        v
[C3 Flatten: anchored curves --> flat sheet + surface volume]
        v
[C3 Ink CNN (patch, fp16)] --> letter mask
        v
[C4 Spark read (capped) / stubs when UNROLL_OFFLINE=1]
        v
[C5 Eval logs + viewer (2 live 2D views + pre-rendered PNGs)]
```

### VRAM budget (8GB card, ~6.5GB usable after OS/viewer reserve)

Bytes math (fp16 = 2 bytes): total = weight_bytes +
frames * H * W * C * 2 + activations. Probe Day 0 measures with
nvidia-smi and records actuals; table below must close against 6.5GB.

| Load | Cap |
|---|---|
| SAM weights + 8-frame bank, fp16, batch 1 | <= 4GB (else shrink bank to 4, then window 32â†’16) |
| Ink CNN train (micro 2 x 4 accum, 64^3) | <= 4GB, never resident with SAM |
| Viewer | CPU-only canvas 2D (zero VRAM); target >=10 fps |
| Swap protocol | one model resident at a time; unload before load; swap cost logged |
| Day 4-6 time math | track + train + infer + eval sized in probe to fit one GPU; overflow cuts slices only pre-freeze, disclosed |
| Quality floor | min 16-slice window / 32^3 tile; below = run FAILED |

## Alternatives Considered

### Alternative A: Synthetic-first (original brief: own generator + ground truth)

*   **Pros**: Perfect labels; no license or download constraints; owned
    benchmark to release.
*   **Cons**: Ink realism unprovable â€” a CNN acing synthetic faint ink
    proves nothing about carbon ink; generator eats Days 0-1; killed by
    the author's pivot to public-data training.
*   **Reason for Rejection**: Transfer gap is fatal to the "did it really
    work?" story, and public labeled data exists at laptop scale.

### Alternative B: Winner-style segmentation as the primary tracer

*   **Pros**: Purpose-built for thin touching papyrus layers; proven on
    real scrolls; no SAM domain-mismatch risk.
*   **Cons**: Re-implements solved work; weak Tooling story for a
    Meta-models hackathon; still needs jump handling for the click UX.
*   **Reason for Rejection**: Kept as the Day-2 fallback, not the lead â€”
    the lead bet is SAM-as-video, which is the project's differentiator.

### Alternative C: API-everything (no local weights)

*   **Pros**: No VRAM pressure; always latest SAM/Spark behavior.
*   **Cons**: Per-slice API cost/latency over hundreds of slices Ã— tiers;
    rate limits or outage kills both dev loop and live demo.
*   **Reason for Rejection**: Local bulk + capped API checks; golden demo
    locked API-free.

### Alternative D: Keep battery as stretch goal (REJECTED in resolve)

*   **Pros**: Energy-and-materials theme fit; nearly free if the tracer
    is domain-agnostic.
*   **Cons**: Retuning + second geometry + defect metric don't fit 8
    solo days; "nearly free" is false (contrast, thickness, morphology
    all differ).
*   **Reason for Rejection**: Cut to future work. Scroll-only wins.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Security & privacy:** API keys via env vars only, never committed
    (`.gitignore` literals + pre-freeze grep, see Verification). No
    Vesuvius volume bytes in the repo or release (NC compliance).
    Weights trained on NC data: research-only release, disclosed, no
    relicense.
*   **Performance budgets:** 6.5GB usable VRAM (see budget table); local
    data under 100GB; eval streams (no full-volume RAM loads); Day-6
    benchmark sized to finish within 10 hours on one GPU or cut to
    fewer slices pre-freeze (recorded in tier report, disclosed).
*   **Rollout & rollback:** Day-6 AM freeze (after first Tier A/B eval)
    locks golden demo inputs (trace + overlays + cached readings +
    SHA256SUMS manifest). Post-freeze, tier reports may gain Tier C +
    final numbers as v2 (v1 = pre-freeze), but golden inputs never
    change; post-freeze breakage rolls back to golden, never
    forward-fixed live. Post-freeze slice-count reduction in v2 is
    banned unless disclosed as a separate reduced run.
*   **Demo disclosure (live vs precomputed):** click + trace playback =
    live local; API SAM frames + Spark readings = live if quota holds,
    else disclosed cached replay; spiral + metrics = pre-rendered PNG.
    $150 credit mapping: record actual per-call cost Day 5, then
    calls-per-run x cost must fit remaining credits or PIN to cached.

## Glossary

CT = computed tomography (X-ray slice stack). VRAM = GPU memory.
OME-Zarr = chunked cloud array format enabling partial reads (no full
download). CC BY-NC = Creative Commons non-commercial license (no
redistribution, no commercial use). CNN = convolutional neural net.
fp16 = 16-bit floats (halves VRAM). OOM = out of memory. Tile = a
spatial sub-block processed independently. z-chunk = consecutive slice
range. Memory bank = recent frames SAM conditions on. Region = fragment
ID + bounding box (split unit). Run ID = unique eval-run handle linking
numbers to logs. Artifact = one pinned downloadable (lock entry).

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Single ordered list, grouped by day. All paths relative to `unscroll/`.

### Day 0 â€” probe + data recipe (C1)

*   `scripts/day0_probe.py` (create): measures usable VRAM (nvidia-smi),
    RAM, disk, download speed, candidate artifact bytes; asserts
    VRAM_usable >= 6.5GB (else record reduced caps) and each artifact
    <10GB; writes `reports/probe_report.md`. Why: recorded actuals.
*   `reports/probe_report.md` (generated, committed): hardware +
    data-size facts + fixed go/no-go z-window. Why: evidence base.
*   `data/recipe.lock` (create): JSON list of {artifact, url, version,
    sha256, bytes, role} for fragments, segments, ink labels, scroll
    crop. Generated by download.py --init from criteria (<10GB each,
    <100GB total, Tier A has released surface, Tier B touches, ink
    labels exist), then frozen. Why: reproducibility + G5.
*   `src/dataio/download.py` (create): --init writes lock from criteria;
    fetch verifies sha256 + bytes; violation of 10GB rule = hard error
    naming the artifact (no skip/truncate). Includes OME-Zarr
    partial-read helpers + tile index (stream.py merged in). Why: one
    choke point for acquisition + streaming.
*   `src/dataio/preprocess.py` (create): (x-mean)/std to
    `data/norm_stats.json`; 80/20 region split (seed 0) with Scroll 1
    overlap coordinate check; writes `src/dataio/splits.json`. Why:
    leakage-free splits.
*   `src/dataio/splits.json` (generated, committed): {region_id, bbox,
    split} records. Why: auditability of G3.
*   `requirements.txt` (create): pinned deps. Why: G5 environment pin.
*   `.gitignore` (create) literals: `data/cache/`, `checkpoints/`,
    `.env`, `reports/*.json`, `*.zarr/`, `*.tif`, `*.tiff`.
    Why: secrets + NC bytes out of git.

### Day 1 â€” download + preprocess + eval schema (C1 done, C5 started)

*   Run download + preprocess; verify splits; write `src/eval/runlog.py`
    (run-ID + per-slice JSONL schema {run_id, z, status, curve_ref,
    checks, reprompts}) and `src/eval/metrics.py` stubs (G1/G2/G3
    function signatures per Goals). Why: schema before first trace;
    fills the missing day; de-risks Day 5.

### Days 2-3 â€” tracing engine (C2)

*   `src/trace/sam_tracker.py` (create): point prompt + prev-mask,
    32-slice z-chunks (8 overlap, 8-frame handoff), Zhang-Suen +
    spur-prune to curve JSONL {z, points, origin, shift}. CLI:
    `track --click X,Y,Z --window 32 --out curves/`. Why: CUJ 1 path.
*   `src/trace/guards.py` + `src/trace/guards.yaml` (create): checks per
    C2 values (reversal <5deg, spacing 30%/10-slice median, kink >60deg;
    Tier C advisory). Why: G2 enforcement.
*   `src/trace/reprompt.py` (create): auto re-prompt (max 3 per
    10-slice region); writes `reports/suspect_shortlist.json`
    [{z, reason, check_values}] for manual review. Why: bounded recovery.
*   `src/trace/classical_fallback.py` (SPEC ONLY until go/no-go fails):
    click-seeded ridge tracker interface doc; build Days 3-4 only on
    fail. Why: one project at a time.
*   `scripts/day2_gono.py` (create): Tier B 50-slice hold on the probe's
    fixed z-window; pass = >=45/50 within 2 voxels; writes
    `reports/gono_verdict.md`. Why: the forcing function.
*   `reports/gono_verdict.md` (generated, committed): verdict + numbers.
    Why: record.

### Day 4 â€” flatten + ink (C3)

*   `src/flatten/anchor.py` (create): 512-pt resample, global origin
    alignment, per-slice shift log, shear = |median aspect - 1| (fail
    >0.25). Why: de-drift, single flattener.
*   `src/flatten/surface.py` (create): Â±4-voxel normal sampling,
    streamed writer to surface volumes. Why: ink-model input.
*   `src/ink/dataset.py` (create): 64^3 patches, stride 32, balanced
    ink/background sampling, binary label encoding. Why: 8GB training.
*   `src/ink/model.py` (create): 3D U-Net tiny 16-32-64-128, 3^3
    kernels, ~0.5M params, in 1x64^3, out 1x64^3 logits. Why: G3.
*   `src/ink/train.py` (create): Dice+BCE, AdamW 3e-4, eff. batch 8
    (2x4), fp16, 50 epochs, early-stop val Dice patience 5, 1 retrain
    max, checkpoints to `checkpoints/`. Why: reproducible training.
*   `src/ink/infer.py` (create): tiled inference (64^3, overlap 16) to
    letter masks. Why: no OOM.

### Day 1-2 + Day 5 â€” eval harness first, Spark + stubs only (C4, C5 core)

*Eval files start Day 1 (schemas) and complete Day 2. Agent scope cut:
spark_read + stubs only; wonder/cli deleted from plan (future work).*

*   `src/agents/spark_read.py` (create): letter suggestions + confidences
    + trace quality check; capped calls/run (recorded). Why: CUJ 1 payoff.
*   `src/agents/stubs.py` (create): fixtures in `src/agents/fixtures/`;
    `UNROLL_OFFLINE=1` forces stub path. Why: G6 golden demo.
*   `src/eval/runlog.py` (create, Day 1): run IDs, per-slice JSONL
    {run_id, z, status, curve_ref, checks, reprompts}. Why: traceability.
*   `src/eval/metrics.py` (create, Day 1-2): `g1(points, ref)` =
    fraction within 2 voxels + per-slice pass rate + p95; `g2(ids)` =
    jumps/100 slices (ID change >=3 slices); `g3(pred, labels)` =
    exact-glyph accuracy + CER with exclusions. Why: no drift.
*   `src/eval/report.py` (create): `report --tier A|B|C` (one run per
    tier); tier markdown embeds run ID + log paths. Why: G4.
*   `scripts/freeze_golden.py` (create, run Day 6 AM): copies trace +
    overlays + cached readings to `golden/` + SHA256SUMS manifest.
    Why: G6, rollback target.

### Days 6-7 â€” benchmark + viewer + release (C5 rest)

*   `viewer/index.html`, `viewer/app.js` (create): static HTML + canvas
    2D; reads `golden/run.json` (+ live run JSON same schema); click =
    (x, y, z) display; 2 views. Why: demo surface.
*   `scripts/render_frames.py` (create): renders spiral + metrics PNGs
    (1280px) from golden run to `viewer/frames/`. Why: pre-rendered eye
    candy, no live 3D.
*   `scripts/demo.py` (create): `--offline` installs a socket-blocking
    guard (any network call raises) then replays `golden/` for the
    API-free test + video capture. Why: G6 executable proof.
*   `README.md` (create): setup, data recipe, exact run commands (probe,
    track, train, report A/B/C, freeze, golden demo), NC license notes,
    known limits. Why: judges + reproducibility.
*   `tests/test_metrics.py` (create): G1 90%-in-2voxels on a 10x10x5 toy
    (expect pass), p95 on skewed distances (expect 4.0), G2 jump count
    on crafted ID flips (expect 2/100). Why: guard definitions. (Toy
    unit arrays only â€” not product synthetic data.)
*   `tests/test_guards.py` (create): winding reversal 3deg (pass) vs
    10deg (fail); spacing 20% (pass) vs 40% (fail); kink 45deg (pass)
    vs 75deg (fail). Why: guard G2 logic.
*   `docs/designs/unscroll.md` (this file): finalized design, SITEMAP
    entry kept current. Why: EGM record.

### Verification Plan

*   **Probe:** `python scripts/day0_probe.py` â†’ exit 0 asserts
    VRAM_usable >= 6.5GB (or records reduced caps) + each artifact
    <10GB; writes `reports/probe_report.md`.
*   **Go/no-go:** `python scripts/day2_gono.py` â†’ exit 0 + PASS line
    (>=45/50 Tier B slices) or exit 2 + FAIL line; writes
    `reports/gono_verdict.md`.
*   **Unit:** `pytest tests/ -q` â†’ all pass (cases above).
*   **Eval:** `python -m src.eval.report --tier A` (then B, then C) â†’
    `reports/tier_{A,B,C}.md`, each embedding run ID + log paths.
*   **Golden:** `python scripts/freeze_golden.py` â†’ `golden/` +
    SHA256SUMS; then `python scripts/demo.py --offline` (socket guard on, OS
    network also disabled; any socket call = fail) â†’ exit 0 + expected PNGs byte
    identical to manifest.
*   **Hygiene:** `pip freeze | diff - requirements.txt` â†’ empty;
    `grep -rEi "(api[_-]?key|secret|token)\s*=" --include="*.py" .` â†’
    no hits; `git status --porcelain data/` â†’ empty (no volume bytes).

--------------------------------------------------------------------------------

*Drafted with EGM*
