# Unscroll: Click-to-Unroll for Rolled-Sheet CT Volumes

> SUPERSEDED 2026-10-27 by [docs/designs/unscroll.md](docs/designs/unscroll.md)
> (canonical, post-resolve). Kept as a historical artifact — do not edit.

*   **Author:** Jayachandiran U
*   **Status:** Superseded
*   **Date:** 2026-10-27

--------------------------------------------------------------------------------

## Overview

Unscroll traces a single rolled layer through a CT slice stack from one
click, flattens it, and recovers the markings on it. The same pipeline
targets two domains: Herculaneum scrolls (recover Greek ink) and
cylindrical battery cells (locate electrode defects).

Built for the Meta Global AI Developer Hackathon (Indie track, Push the
Frontier theme; submissions close Nov 4 — verify exact cutoff and the
reported $150 API credits in the portal). Meta models throughout: SAM 3.1
tracking, Spark reading, Wonder orchestration.

All training and evaluation uses publicly available data — no synthetic
generator. Scroll side: Vesuvius Challenge open data (CC BY-NC):
labeled fragments and released segments with ink labels for training,
plus small streamed Scroll 1 crops (1cm crop, OME-Zarr partial reads)
for tracking eval. Battery side (stretch): one cylindrical scan from the
public 1000+ battery CT dataset. Volumes are never redistributed; the
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
slice; battery jellyroll inspection needs separate dedicated tooling.
Both are the same geometry problem (a rolled sheet imaged in 3D) solved
twice, slowly. Unscroll tests whether one click plus video tracking can
replace dense manual tracing, with public ink labels and released
segment surfaces as ground truth making the claim checkable instead of
vibes-based — on hardware a solo builder actually owns.

## Goals

Difficulty tiers (public data, fixed Day 0): Tier A = fragment with
separated layers; Tier B = fragment with touching layers; Tier C =
streamed scroll crop, crushed.

*   G1: One click per layer traces >=90% of layer area (within 2 voxels
    of released segment surface, averaged over surface points) on
    Tiers A-B.
*   G2: Jump checks cut layer jumps (excursion onto a different
    segment-surface ID for >=3 consecutive slices) to <=2 per 100
    slices on Tiers A-B, measured before/after.
*   G3: Letter recovery >=70% exact-glyph match vs public ink labels on
    Tier A flat sheets (edit distance reported alongside).
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
*   Battery mode as a second product: stretch goal, one public
    cylindrical scan through the same tracer, cut first if behind.
*   Multi-script text: Greek only; no charset abstraction this sprint.
*   Production hardening: no CI-gated release, no multi-user viewer.
*   Beating winner tools head-to-head on real data (out of scope for
    8 days; synthetic head-to-head harness only if time permits).

## Open Questions

*   **Q1 (resources):** Solo or team? RTX 4060 VRAM size? Meta Model API
    access + quota confirmed? Assumes worst case (solo, 8GB, unconfirmed)
    until answered.
*   **Q2 (data picks):** Which exact public artifacts (fragment IDs,
    segment IDs + ink-label set, scroll crop coordinates, battery scan
    ID)? Decide Day 0 by probing sizes and download speed. Also confirm
    the trained-model-release posture under CC BY-NC (code + weights
    from NC data: disclose, don't relicense).
*   **Q3 (SAM go/no-go):** Proposal: SAM must hold a touch-free layer for
    50 consecutive slices by end of Day 2, else the classical fallback
    becomes primary. Confirm threshold.
*   **Q4 (clicks baseline):** What "dense manual" number does clicks/cm^2
    beat? Proposal: time the author tracing one layer manually in an
    open tool on Day 0 (n=1, disclosed as such).

## Critical User Journeys (CUJs)

### CUJ 1: Click a scroll layer, get Greek letters

*   **Given**: Scroll crop cached locally (Tier A/B), viewer open,
    tracing engine warm.
*   **When**: User clicks one pixel on a layer in a slice.
*   **Then**: Layer traced through the stack (suspect regions flagged),
    spiral shown, flat sheet rendered with letter suggestions +
    confidences; per-slice log + metrics recorded under a run ID.

### CUJ 2: Trace a battery electrode, flag a defect

*   **Given**: Public cylindrical battery scan cached (stretch goal).
*   **When**: User clicks the electrode layer.
*   **Then**: Same tracer unrolls it; folds/tears/gaps highlighted with
    localization numbers; identical run-ID logging as CUJ 1.

## Technical Plan (Architecture)

Five components, one data recipe. Everything runs on the laptop except
explicitly capped API calls.

### C1. Data layer (Day 0) — stream, pin, cache

*   **Sources (pinned Day 0, Q2):** Vesuvius fragments + released
    segments with ink labels (train/val, split by physical region —
    never by slice, to avoid leakage); 1cm Scroll 1 crop for tracking
    eval; one cylindrical battery scan (stretch).
*   **Access:** `vesuvius` Python lib + OME-Zarr partial reads; only
    crops/tiles touch local disk. No artifact over 10GB, total under
    100GB. All URLs/versions/commit hashes pinned in `data/recipe.lock`.
*   **Preprocess:** 16-bit to float normalize per-volume (record stats),
    tile index, train/val split file. CPU-only, cached once.

### C2. Tracing engine (Days 2-3) — video tracking + jump guards

*   **Track:** slice stack as video frames. Click prompt → SAM 3.1 mask
    per frame → thin to 1-voxel curve (skeletonize). Local open weights
    for bulk; API SAM for live demo + capped hard-case checks.
*   **Stream in z-chunks** (proposed: 32 slices, fp16, batch 1, memory
    bank capped to last 8 frames — actuals probed and recorded Day 0).
*   **Jump guards (thresholds tuned Day 3, all logged per slice):**
    winding angle monotonic within tolerance; gap to neighbor layers
    within 30% of local median; curvature spike = kink. Violation →
    auto re-prompt (max 3 per region) → else flag suspect for Spark
    vision review (capped count per run).
*   **Fallback (primary if Day-2 go/no-go fails):** click-seeded
    classical ridge tracker. Golden demo trace precomputed either way.

### C3. Flatten + ink (Day 4) — anchored parameterization

*   **De-drift:** resample each slice curve to fixed arc-length points
    with cross-slice origin alignment (template-match anchor); log
    per-slice origin/shift; report shear metric (letter-aspect drift).
    If drift is unfixable, fall back to mesh parameterization.
*   **Surface volume:** sample ±N voxels along the curve normal.
*   **Ink CNN:** small 3D patch model (proposed 64^3 patches, fp16,
    grad accumulation), trained on fragment ink labels only. Tiled
    inference, nothing full-volume in VRAM. Baseline reference: compare
    against released winner ink-model predictions on the same Tier A
    sheets where available (reference only, not a dependency).

### C4. Agent layer (Day 5) — capped, audited, optional-each

*   **Spark:** letter suggestions + confidences on flat sheet; trace
    quality check. **Wonder:** next-click picker + report writer, with
    iteration cap and reasoning audit log. **Voice:** text-command
    equivalent always available. **Image:** clearly labeled illustration.
*   Every agent has an API-off degraded mode; golden demo locked to
    cached outputs at freeze.

### C5. Eval + viewer (Days 6-7) — logs first, pixels second

*   **Eval harness first:** run IDs, per-slice pass/fail logs, tier
    reports for G1-G4. Built alongside C1, not on Day 6.
*   **Viewer:** 2 live views (slice + trace overlay; flat sheet +
    letters). 3D spiral + metrics panel pre-rendered from golden run
    (fallback if the live build slips).

```text
[Public sources] --stream/crop--> [Local cache <100GB, recipe.lock]
        | preprocess (CPU, cached once)
        v
[Click] --> [C2 Tracing: SAM/video + jump guards + fallback]
        | per-slice curves + suspect flags (run ID)
        v
[C3 Flatten: anchored curves --> flat sheet + surface volume]
        v
[C3 Ink CNN (patch, fp16)] --> letter mask
        v
[C4 Agents: Spark read / Wonder orchestrate (capped, logged)]
        v
[C5 Eval logs + viewer (2 live views + pre-rendered fallback)]
```

### VRAM budget (8GB assumed; probe and record actuals Day 0)

| Load | Cap |
|---|---|
| SAM weights + 8-frame memory bank, fp16 | <= 5GB (else shrink bank/window) |
| Ink CNN train (64^3 patch, micro-batch 1-2 + accum) | <= 5GB, never with SAM resident |
| Inference | one model resident at a time, tiled |
| Overflow rule | any OOM → halve tile/window, record, continue |

## Alternatives Considered

### Alternative A: Synthetic-first (original brief: own generator + ground truth)

*   **Pros**: Perfect labels; no license or download constraints; owned
    benchmark to release.
*   **Cons**: Ink realism unprovable — a CNN acing synthetic faint ink
    proves nothing about carbon ink; generator eats Days 0-1; killed by
    the author's pivot to public-data training.
*   **Reason for Rejection**: Transfer gap is fatal to the "did it really
    work?" story, and public labeled data exists at laptop scale.

### Alternative B: Winner-style segmentation as the primary tracer

*   **Pros**: Purpose-built for thin touching papyrus layers; proven on
    real scrolls; no SAM domain-mismatch risk.
*   **Cons**: Re-implements solved work; weak Tooling story for a
    Meta-models hackathon; still needs jump handling for the click UX.
*   **Reason for Rejection**: Kept as the Day-2 fallback, not the lead —
    the lead bet is SAM-as-video, which is the project's differentiator.

### Alternative C: API-everything (no local weights)

*   **Pros**: No VRAM pressure; always latest SAM/Spark behavior.
*   **Cons**: Per-slice API cost/latency over hundreds of slices × tiers;
    rate limits or outage kills both dev loop and live demo.
*   **Reason for Rejection**: Local bulk + capped API checks; golden demo
    locked API-free.

### Alternative D: Scroll-only scope (cut battery entirely)

*   **Pros**: Less surface area; all 8 days on one domain.
*   **Cons**: Loses the energy-and-materials theme fit; battery is nearly
    free if the tracer is domain-agnostic (one extra scan through C2-C3).
*   **Reason for Rejection**: Not cut, demoted — battery stays a stretch
    goal, first to go if behind.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Security & privacy:** API keys via env vars only, never committed
    (`.gitignore` + pre-freeze grep check). No Vesuvius volume bytes in
    the repo or release artifacts (NC compliance).
*   **Performance budgets:** VRAM 8GB (see budget table); local data
    under 100GB; eval harness must stream (no full-volume RAM loads);
    Day-6 benchmark sized to finish overnight on one GPU or it gets cut
    to fewer slices (recorded, disclosed).
*   **Rollout & rollback:** Day-5 freeze locks golden demo (precomputed
    trace + cached agent outputs); any post-freeze breakage rolls back
    to golden artifacts, never forward-fixed live.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

Single ordered list, grouped by day. All paths relative to `unscroll/`.

### Day 0 — probe + data recipe (C1)

*   `scripts/day0_probe.py` (create): measures VRAM, RAM, disk, download
    speed, and candidate artifact sizes; writes `probe_report.md`.
    Why: replaces assumed budgets with recorded actuals.
*   `probe_report.md` (generated, committed): hardware + data-size facts.
    Why: evidence base for every later cap.
*   `data/recipe.lock` (create): pinned URLs, versions, hashes of chosen
    fragments, segments, ink labels, scroll crop, battery scan. Why:
    reproducibility + G5.
*   `src/dataio/download.py` (create): fetches per recipe.lock, verifies
    hashes/sizes, enforces 10GB-per-artifact rule. Why: one choke point
    for NC-compliant acquisition.
*   `src/dataio/stream.py` (create): OME-Zarr partial-read helpers + tile
    index. Why: stream-don't-download for everything large.
*   `src/dataio/preprocess.py` (create): 16-bit normalize (record stats),
    region-based train/val split writer. Why: leakage-free splits.
*   `src/dataio/splits.json` (generated, committed): the split record.
    Why: auditability of G3.
*   `requirements.txt` (create): pinned deps. Why: G5 environment pin.
*   `.gitignore` (create): data cache, checkpoints, `.env`, reports.
    Why: keep secrets + NC bytes out of git.

### Days 2-3 — tracing engine (C2)

*   `src/trace/sam_tracker.py` (create): click + z-chunk video tracking,
    mask-to-curve thinning, memory-bank cap. Why: core CUJ 1 path.
*   `src/trace/guards.py` (create): winding/spacing/smoothness checks +
    threshold config. Why: G2 jump detection.
*   `src/trace/reprompt.py` (create): auto re-prompt loop (max 3/region),
    suspect flags, Spark-review shortlist writer. Why: bounded recovery.
*   `src/trace/classical_fallback.py` (create): click-seeded ridge
    tracker, same curve-log interface as SAM path. Why: Day-2 fallback.
*   `scripts/day2_gono.py` (create): runs the 50-slice hold test, writes
    `gono_verdict.md` (SAM vs fallback). Why: the forcing function.
*   `gono_verdict.md` (generated, committed): the verdict. Why: record.

### Day 4 — flatten + ink (C3)

*   `src/flatten/anchor.py` (create): arc-length resample, cross-slice
    origin alignment, per-slice shift log, shear metric. Why: de-drift.
*   `src/flatten/surface.py` (create): normal sampling to surface
    volume (streamed writer). Why: ink-model input.
*   `src/ink/dataset.py` (create): patch dataset from ink labels + flat
    sheets. Why: patch-based training on 8GB.
*   `src/ink/model.py` (create): small 3D CNN architecture. Why: G3.
*   `src/ink/train.py` (create): fp16 + grad-accumulation loop,
    checkpoints, val split eval. Why: reproducible training.
*   `src/ink/infer.py` (create): tiled inference writer. Why: no OOM.

### Days 0-2 + Day 5 — eval harness first, agents at freeze (C4, C5 core)

*Eval files (`src/eval/*`) start Days 0-2 alongside C1 — log schema before
the first trace. Agent files land Day 5.*

*   `src/agents/spark_read.py` (create): letter suggestions + confidences
    + trace quality check. Why: CUJ 1 payoff.
*   `src/agents/wonder.py` (create): next-click picker + report writer,
    iteration cap + reasoning audit log. Why: multi-layer story.
*   `src/agents/cli.py` (create): text-command equivalents of Voice
    commands. Why: guaranteed-fallback interaction.
*   `src/agents/stubs.py` (create): cached-output fixtures + API-off
    switch. Why: G6 golden demo.
*   `src/eval/runlog.py` (create): run IDs, per-slice log schema/writer.
    Why: every claim traceable to slices.
*   `src/eval/metrics.py` (create): exact G1-G4 formulas. Why: honest
    numbers, no definition drift.
*   `src/eval/report.py` (create): tier report generator (A/B/C). Why: G4.
*   `scripts/freeze_golden.py` (create): locks golden artifacts
    (trace + overlays + cached readings). Why: G6, rollback target.

### Days 6-7 — benchmark + viewer + release (C5 rest)

*   `viewer/index.html`, `viewer/app.js` (create): 2 live views (slice +
    trace; sheet + letters). Why: demo surface.
*   `viewer/frames/` (generated): pre-rendered spiral + metrics panels
    from golden run. Why: viewer-slip fallback.
*   `README.md` (create): setup, data recipe, run commands, license
    notes (NC), known limits. Why: judges + reproducibility.
*   `tests/test_metrics.py` (create): metric formulas on toy arrays.
    Why: guard G1-G4 definitions. (Toy unit arrays only — not
    product synthetic data.)
*   `tests/test_guards.py` (create): jump checks on toy curves. Why:
    guard G2 logic.
*   `docs/designs/unscroll.md` (create at Step 6): finalized design.
    Why: EGM record + SITEMAP entry.

### Verification Plan

*   **Probe:** `python scripts/day0_probe.py` → `probe_report.md` shows
    VRAM/RAM/disk/speeds; all later caps cite it.
*   **Go/no-go:** `python scripts/day2_gono.py` → `gono_verdict.md`
    (50-slice hold: pass = SAM primary, fail = fallback primary).
*   **Unit:** `pytest tests/` — metric + guard formulas on toy inputs.
*   **Eval:** `python -m src.eval.report --tier A B C` → tier reports
    with per-slice logs; every G1-G4 number links to a run ID.
*   **Golden:** `python scripts/freeze_golden.py` → API-free demo runs
    with network disabled (the actual test).
*   **Hygiene:** `pip freeze` matches `requirements.txt`; grep finds no
    keys; repo contains zero volume bytes (size audit in README).

--------------------------------------------------------------------------------

*Drafted with EGM*
