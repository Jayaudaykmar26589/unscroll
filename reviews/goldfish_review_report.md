# Goldfish Resolve Report — unscroll.md

* **Date:** 2026-10-27 / **Mode:** resolve-all (consolidated report)
* **Rule:** user constraints win — public-data *training* stays,
  8-day solo laptop scope wins over completeness.

## Classification

| ID | Decision | Fix applied |
|----|----------|-------------|
| CB1 scope | PROPOSE FIX (partial) + CONTEST (1 item) | Cut: CUJ 2 to future work; Wonder/Voice/Image/CLI out of 8-day plan (Spark-read-only + stubs); mesh fallback deleted; Tier C qualitative-only, after-only; fallback = spec'd stub, built only if go/no-go fails. CONTEST: keep custom CNN training (user explicitly requires training on public data) — timeboxed: fixed arch, 50 epochs max, 1 retrain max. |
| CB2 go/no-go | PROPOSE FIX | Tier B touching layers; pass = >=45/50 slices within 2 voxels; fixed z-window recorded in probe_report. |
| CB3 continuity | PROPOSE FIX | 8-slice overlap between 32-slice chunks; last-8-frame state handoff; winding on stitched curve. |
| CB4 VRAM math | PROPOSE FIX | Formula bytes = params + frames*H*W*C*2; usable 6.5GB reserve; swap protocol + Day 4-6 time math; viewer CPU-only row. |
| CB5 floor | PROPOSE FIX | Min 16-slice window / 32^3 tile; below = run FAILED, not degraded. |
| CB6 data IDs | PROPOSE FIX (partial) | Criteria + region schema + split rule pinned now; exact IDs recorded in recipe.lock Day 0 (cannot pre-date the probe). Region = fragment ID + bbox; 80/20 split, seed 0; Scroll 1 crop must not overlap train fragments (coordinate check in preprocess). |
| CB7 metrics | PROPOSE FIX | Exact G1/G2/G3 formulas: sampling, distance, tail metrics, Tier C breakage log, glyph protocol (charset, connected-component segmentation, exact accuracy + CER, exclusions, n). |
| CB8 decision-free | PROPOSE FIX | CNN arch + hparams; SAM pin procedure; click encoding; thinning + spur bound; curve JSONL schema; guards.yaml; runlog/metrics schemas; recipe.lock schema; CLI syntax; golden procedure; freeze manifest; stub switch; wonder cap; viewer spec; render command; probe assertions; hygiene commands. |
| CB9 schedule | PROPOSE FIX | Day 1 = download + preprocess + splits + eval schema; freeze moves to post-first-eval (Day 6 AM); Day 5 = spark + stubs only. |
| CB10 canonical | PROPOSE FIX | unscroll.md canonical; design_draft.md bannered superseded; Status = In Review; SITEMAP aligned. |
| S5 guard-pipeline | PROPOSE FIX (partial) | Keep 3 guards (cheap); Spark auto-review → manual shortlist file (no auto API). |
| S5 dataio collapse | CONTEST (partial) | Merge stream.py into download.py only; keep split-writer/lock (judge-readable, G5 evidence). |
| S5 spiral | PROPOSE FIX | Spiral + metrics pre-rendered only; 2 live 2D views. |
| S5 Non-Goal clash | PROPOSE FIX | Removed harness clause; toy unit tests stay. |
| S6 glossary+defs | PROPOSE FIX | Glossary added; N = ±4 voxels; artifact/hours/grep defined. |
| S7a skeletonize | CONTEST | Keep 2-voxel tolerance + Zhang-Suen spur-prune bound (<1 voxel error) recorded; widen only if probe shows otherwise. |
| S7a dataflow | PROPOSE FIX | Diagram gains reprompt loop, Spark path, battery cut note, invalidation note. |
| S7a live-vs-canned | PROPOSE FIX | Demo disclosure table (live vs precomputed per beat) + $150 mapping procedure (record actual rates). |
| S7a rollback | PROPOSE FIX | Post-freeze slice reduction banned without re-issued tier reports. |
| S7a weights | PROPOSE FIX | Weights-from-NC-data: research-only release, disclose, no relicense. |
| S9 dirs/CLI/hygiene | PROPOSE FIX | Generated reports → reports/; eval CLI = 3 runs; exact hygiene commands; Step-6 self-ref fixed; test cases + gitignore literals added. |

Contests logged with reasons; all else applied to `docs/designs/unscroll.md`.
