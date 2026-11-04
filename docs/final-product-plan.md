# Unscroll — final product plan (agreed 2026-10-30)

Deadline: Wed Nov 4, 9 PM ET (Thu Nov 5, 7:30 AM IST). Submit Tue Nov 3.
All GPU work runs on Kaggle (T4). Meta API calls run locally (key stays in `.env`).
Every new capability is measured and reported, including where it fails.

## A. Meta models, each with a real job

| # | Feature | Meta model | Measured by | Day |
|---|---|---|---|---|
| A1 | **Patch-growing agent**: reads trace status, re-seeds where SAM lost the sheet, proposes next clicks along the sheet, submits Kaggle batches, stops on guard failures | Muse Wonder (tool calling) | slices traced per seed before/after (59/100 today); on-layer share where meshes exist | Sat |
| A2 | **Reviewer v3**: pairwise "which trace follows one sheet?" and CT-without-overlay variant, tested on held-out seeds only | Muse Spark 1.3 | accept precision, review load vs v1/v2 | Sat |
| A3 | **QA report writer**: per-seed plain-language report from metrics + overlays for scholars | Muse Spark 1.3 | spot-checked against numbers (no invented figures) | Sun |
| A4 | **Voice control in the workbench**: "next slice", "mark error", "jump to worst" | Muse Voice Transcribe | command accuracy on a short scripted test | Sun |
| A5 | **Reading on unrolled surfaces**: letter suggestions on ink predictions, always with confidence and human acceptance | Muse Spark 1.3 | only shown if the ink model finds anything; otherwise reported as none | Mon |
| — | Muse Image, Muse Glimmer, Spark Open | not used | — | — |

Muse Image is left out on purpose (generated scroll imagery would mislead
scholars). Glimmer (30B) and Spark Open (816B) do not fit a 16 GB T4 at useful
precision; the hosted API covers the same need.

## B. Ink modelling (Kaggle GPU)

| # | Item | Day |
|---|---|---|
| B1 | Train an ink model on the 2023 Kaggle ink-detection fragments (train 1+2, validate on 3). Target: beat the current F1 0.54 | Sat |
| B2 | Apply it to the unrolled scroll surface (65-layer surface volume, ±32 voxels); report what it finds, including nothing | Sun |
| B3 | JEPA-style self-supervised pretraining on unlabelled scroll + fragment patches, then fine-tune; compare with B1 on fragment 3 | Sun–Mon (stretch) |

**User action needed for B:** accept the rules at
https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/rules
so notebooks can attach that data.

## C. Product recommendations

| # | Item | Day |
|---|---|---|
| C1 | Held-out check (running), added to results, README, page, video | Fri |
| C2 | Diagnose the 100-slice stop (motion verified; SAM side) — crop runs now keep the tracker run log | Fri |
| C3 | Second unrolled surface (seed 20231106155351) | Sat |
| C4 | Surface workbench shows the new walk trace, not the historical one | Sun |
| C5 | Fix the `node` test (skip with reason when Node is missing) | Fri |
| C6 | Repository tidy (scratch logs, debug scripts) — **only with user OK** | Mon |
| C7 | Clean launch test + feature freeze | Mon |
| C8 | Submission package: portal text, links, citations, licence and data-permission notes | Mon |
| C9 | Video recorded by the user from `docs/video-script.md` (updated Mon) | Tue |
| — | Live in-browser tracing | cut: needs a local GPU; replay + Kaggle batches instead |

## D. After the hackathon (prepared now, run later)

| # | Item |
|---|---|
| D1 | Vesuvius Progress Prize package: tracing tool, first-slice walk, attention fix, seeded benchmark, docs (`docs/vesuvius-progress-prize.md`) |
| D2 | Automatic patch growing at scale (A1 without a human in the loop) |
| D3 | JEPA ink pretraining at scale toward First Letters (June 2027) |

## Cut lines (if time runs short, drop from the bottom)

1. Must: C1, C2, C5, A1 (small), B1, C4, C7, C8, C9
2. Should: A2, C3, B2, A3
3. Could: A4, A5, B3, D1 docs
