# Goldfish Brainstorm Review — Unscroll

* **Date:** 2026-10-27 (Day 0)
* **Target:** `../concepts_brief.md` (author v0)
* **Lenses:** contrarian/YAGNI, cost/quota, UX/ergonomics,
  dependency/fallback, observability, tech-debt
* **Raw reports:** `raw/lens-*.txt` (verbatim, unedited)
* **Convergence rule:** a finding raised by 3+ lenses is load-bearing.

## Convergent findings (raised by 3+ lenses — fix first)

1. **SAM-on-touching-layers is unproven and has no fallback** (all 6 lenses).
   CT layers are thin, touching, near-identical grey curves — nothing like
   SAM's natural-video training. Winding/spacing/smoothness checks only
   *catch* jumps; they do not trace if SAM cannot lock at all. Every wow
   beat after the click depends on this, including battery mode.
   *Fix: name a non-SAM fallback now (classical curve tracker, click-seeded
   U-Net, or precomputed golden trace for the demo volume) and define the
   Day-2 go/no-go signal for SAM.*
2. **Compute budget does not close on one RTX 4060** (cost, fallback, debt).
   512^3 volumes + SAM video memory bank over hundreds of slices + 3D CNN
   + ground truth, with no tiling/chunking/VRAM-partition plan.
   *Fix: VRAM budget table (weights, volume tiles, memory bank, CNN) +
   downsampled demo volume + mixed precision, before Day 2.*
3. **Metrics are names without formulas** (observability, debt, contrarian).
   Trace-share (threshold? denominator? aggregation?), jump events (what
   counts?), letter match rule + confidence cutoff, clicks/cm^2 baseline,
   battery localization formula — all undefined. Two implementations could
   report 94% vs 60% on the same trace.
   *Fix: exact formulas + thresholds in the design doc; frame them as
   regression tests.*
4. **"Flattening for free" ignores parameterization drift** (contrarian,
   observability, debt). Stacked per-slice curves differ in length and
   origin; (arc-length, slice) shears letters unless origins are aligned.
   No distortion/shear metric, so sheared text is indistinguishable from
   ink-model failure.
   *Fix: per-slice origin/arc-length log + shear metric; honest
   mesh-parameterization fallback if drift is unfixable.*
5. **Live/API demo path has no offline fallback** (cost, UX, fallback, debt).
   API SAM for the live click, Spark reading, Wonder, Voice — one rate
   limit or outage kills the video. Same for the 1-day viewer build.
   *Fix: cached API outputs + pre-recorded golden run + API-free demo mode
   locked at Day-5 freeze; pre-rendered viewer frames from Day-4 outputs.*

## Blockers (unique, by lens)

- **[B1]** Contrarian: scroll half duplicates solved winner segmentation
  work with a worse-suited tracker; needs a head-to-head vs winner tools
  on the same volume, not just self-solved synthetic metrics.
- **[B2]** Contrarian: battery mode conflates a hard problem (crushed
  scroll) with an easy one (clean cylinders) to fit the track — scope as
  re-skin with shared abstraction, or descope to stretch goal.
- **[B3]** Cost: Day-6 single-day full benchmark (5 crush levels x hundreds
  of slices + battery + CNN + API calls) exceeds one day on one GPU —
  re-plan with buffers; start the harness early.
- **[B4]** Cost: Spark hard-case review cost peaks exactly where local SAM
  fails most — set per-slice caps and a max re-prompt budget.
- **[B5]** UX: "one click" contradicts per-layer reality (Wonder "no human
  clicks at all" vs click-per-layer demo) — decide and reword.
- **[B6]** UX: no failed-trace / suspect-region / confidence UX designed;
  Voice commands are demo-fragile with no confirmation or fallback.
- **[B7]** Fallback: Vesuvius late-yes invites Day-6 scope creep — hard
  gate: no reply by Day-5 freeze = no; late permission swaps into the
  video only, never the pipeline.
- **[B8]** Observability: Day-6 overnight batch has no run IDs, per-slice
  pass/fail log, checkpointing, or alerting — silent divergence poisons
  morning numbers with nothing to bisect.
- **[B9]** Debt: no versioned benchmark contract (seeds, splits, schema,
  dataset version) — first generator tweak silently invalidates metrics.
- **[B10]** Debt: local-weights vs API SAM split has no pinned model
  version or adapter interface — results drift on either side's update.

## Majors

- **[M1]** Ink CNN transfer gap: synthetic-trained CNN proves nothing about
  carbon ink; winners'-model fallback is a second unbuilt branch with no
  adapter or validation harness. (contrarian, debt)
- **[M2]** Agent layer (Spark/Wonder/Voice/Image) serves the rubric, not
  the pipeline — make each optional with a defined degraded mode. (contrarian)
- **[M3]** Jump checks lack numeric tolerances + per-slice verdict logs;
  re-prompt/Spark decisions leave no audit trail. (observability)
- **[M4]** Wonder autonomous clicks + uncapped iteration, no reasoning log —
  the report's author is the suspect with no record. (observability, cost)
- **[M5]** 512^3 + single-center spiral priors hardcoded, yet crush warps
  and battery folds violate them — expect rewrites. (debt)
- **[M6]** Greek-only pipeline baked through generator, CNN, and reader —
  no charset abstraction; battery carries dead OCR machinery. (debt)
- **[M7]** Flat-surface format unspecified but shared by CNN, reader, and
  1-day viewer — guarantees viewer/pipeline skew. (debt)
- **[M8]** No tests, CI, container, or dependency pins; nothing in Evidence
  framed as regression tests for the Prize path to June 2027. (debt)

## Nits

- Verify exact GFS family LICENSE on Day 0 (families ship mixed licenses);
  else swap to confirmed-OFL Greek coverage. (fallback)
- Image reconstruction placement must not sit beside real letters or it
  undermines the ground-truth story. (UX)
- State the scored metric set explicitly (5 + optional real-scroll). (obs.)
- Flat sheet needs axis labels/scale in the viewer. (UX)
- Overnight license-header checklist: OFL font vs CC BY benchmark vs NC
  real data must not mix in the release. (debt)
- Note ongoing API/GPU cost posture past the 8-day window. (cost)

## Elephant's 3 open questions — reviewer cross-check

1. Flattening consistency → confirmed as convergent finding 4.
2. Ink difficulty calibration → partially covered by M1; still needs an
   explicit calibration method (match synthetic ink SNR to published
   real-scan statistics).
3. Resources (team/API/VRAM) → unanswered; cost + fallback blockers assume
   worst case (solo, 8GB, unconfirmed quota) until confirmed.
