# Consolidated Review Report — unscroll.md

* **Target:** `../docs/designs/unscroll.md` (370 lines)
* **Ranges:** Feature 1-262, Detailed 263-370 (split at line 263)
* **Date:** 2026-10-27 / **Mode:** consolidated (no checkpoints)
* **Raw stage reports:** `raw/stage5-zoom-out.txt`,
  `raw/stage6-comprehension.txt`, `raw/stage7a-design-critic.txt`,
  `raw/stage7b-codebase-details.txt`, `raw/stage9-readiness.txt`

## Stage verdicts

| Stage | Reviewer | Verdict |
|---|---|---|
| 5 | zoom-out | SCOPE FAIL: ~30 files, two tracers, two domains, custom CNN + agents cannot fit 8 solo days — cut list given |
| 6 | comprehension | FAIL: newcomer cannot build/evaluate — data IDs, model roles, metrics, schedule all referenced, never defined |
| 7a | design-critic | FAIL: unproven tracker + placeholder numbers + silent quality degradation under resource pressure |
| 7b | codebase-details | PASS on design claims; workspace metadata inconsistent (status + duplicated source of truth) |
| 9 | readiness | **NO-GO**: lists files/intents, leaves algorithms/schemas/thresholds/pass-fail to the implementer |

**Overall: NO-GO. Resolve before implementing.**

## Convergent blockers (merged, deduped)

*   **[CB1] Scope exceeds 8 solo days** (S5, S7a). Dual-tracer build,
    from-scratch 3D CNN training, Wonder/Voice/Image agents, battery
    CUJ 2, 3-tier before/after eval. Fix direction: SAM-primary with
    fallback as spec'd stub until go/no-go fails; inference-only ink
    (winner weights/2D baseline); Spark-read-only + stubs; cut CUJ 2 to
    future work; Tier C qualitative-only, after-only metrics.
*   **[CB2] Go/no-go tests the wrong difficulty** (S7a). Tier A 50-slice
    hold falsely greenlights Tier B/C work. Must be Tier B touching
    layers with a numeric pass threshold + slice-selection rule.
*   **[CB3] No chunk-continuity plan** (S7a). 32-slice chunks + 8-frame
    bank discard long-range spiral context; seams every 32 slices.
    Specify overlap, state handoff, cross-boundary winding.
*   **[CB4] VRAM budget is arithmetic-free** (S7a, S6). No params/bytes
    math, no OS/viewer reserve (~6-7GB usable, not 8), no swap schedule
    for "never with SAM resident." Show the bytes + time math.
*   **[CB5] Overflow rule has no quality floor** (S7a). Halve-on-OOM
    makes runs incomparable. Minimum window/tile or mark run failed.
*   **[CB6] Data identities + split rule undefined** (S6, S7a, S9).
    No fragment/segment/crop/battery IDs; "region" undefined. Pin Day-0
    defaults + IDs in the doc now; define region schema + Scroll 1
    overlap guard.
*   **[CB7] Metric formulas absent** (S6, S7a, S9). G1 (voxel size,
    sampling, tail metric), G2 (Tier C ID-free oracle or A/B-only),
    G3 (charset, segmentation, CER procedure, exclusions, n). Exact
    formulas + per-slice pass rate + p95.
*   **[CB8] Plan not decision-free** (S9). Missing: CNN arch/hparams,
    SAM variant/click encoding/thinning/curve format, guard formulas,
    shared curve-log interface schema, runlog/metrics schemas, CLI
    signatures, golden-test procedure, recipe.lock schema, freeze
    manifest. Every starred item is a decision disguised as a step.
*   **[CB9] Schedule broken** (S7a, S9). Day 1 unassigned; Day-5 freeze
    before Days 6-7 eval (frozen demo unevaluated); 4 agents in one
    solo day. Replan: freeze-after-eval (or define unevaluated-golden),
    assign Day 1, cut Day-5 load per CB1.
*   **[CB10] Duplicated source of truth + status clash** (S7b).
    `design_draft.md` == `unscroll.md` byte-identical; brief names the
    former, SITEMAP the latter; header says Draft, SITEMAP says
    Approved. Keep one canonical copy; align status.

## Unique findings (see raw files for full text)

*   S5 majors: guard+reprompt+Spark pipeline overbuilt (one guard +
    flag-suspect); Day-0 data layer 8 files (collapse to dataio.py +
    lock); viewer spiral serves no CUJ (delete); Non-Goal synthetic
    contradiction (L74 vs L85).
*   S6 majors/nits: flattening/sampling/agent-cap/VRAM/streaming/
    preprocess/license-posture definitions missing; acronym glossary
    needed (CT, VRAM, OME-Zarr, CC BY-NC, CNN, fp16, OOM...); artifact/
    overnight/grep-check ambiguities.
*   S7a majors/nits: skeletonize-vs-2voxel tolerance conflict; dataflow
    omits feedback paths (reprompt loop, Spark path, battery branch,
    cache invalidation); live-vs-canned demo disclosure + $150 mapping;
    rollback may demo fewer slices than claimed; per-artifact byte
    estimates; weights-release posture; viewer CPU/GPU row.
*   S9 majors/nits: output-dir ambiguity for root files; download/stream/
    reprompt/anchor/surface/dataset/infer/wonder/cli/stubs/report/
    freeze/viewer spec gaps (see raw); "Step 6" self-reference (L351);
    README command list; test cases; .gitignore literals.

## Resolve queue (proposed order)

1. CB10 (30s metadata fix) → 2. CB1 scope cuts (owner decisions) →
    3. CB6 data IDs → 4. CB7 metric formulas → 5. CB2/CB3/CB5 tracker
    specs → 6. CB4 VRAM math → 7. CB9 schedule → 8. CB8 remaining
    decision-free specs → 9. S6 glossary + S7a/S9 majors.
