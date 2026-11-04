# Unscroll — submission package

Meta Global AI Developer Hackathon · Indie track · Push the Frontier.
Deadline: Wed Nov 4, 9:00 PM ET (Thu Nov 5, 7:30 AM IST). Target: submit Tue Nov 3.
Portal: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit
(field names below are generic; paste into whichever fields the form shows).

## Name

Unscroll

## One line (≤140 characters)

Click one layer of a 2,000-year-old Herculaneum scroll in CT; SAM 3.1 traces it and Unscroll measures whether the AI stayed on it.

## Short description (~100 words)

The Herculaneum scrolls, carbonized by Vesuvius in 79 AD, can only be read from
CT scans by tracing each crushed papyrus sheet through thousands of slices.
Unscroll turns one click into a SAM 3.1 trace on real Scroll 1 data and measures,
against layers people traced by hand, whether it stayed on the right sheet. A
diagnosed first-slice fix lifts accuracy from 36% to 55% within 40 µm (49% on
held-out clicks). A Muse Wonder agent re-seeds traces by itself, Muse Spark
reviews traces and writes number-checked QA notes, and the 2023 Grand Prize ink
model reads letters through Unscroll's pipeline. Everything is measured,
including where it fails.

## Long description

**Problem.** Reading the Herculaneum scrolls means following one papyrus sheet
through a stack of CT slices where neighbouring sheets are crushed together. A
tracer that slips one layer produces the wrong surface, and nobody can tell
without slow manual review.

**What Unscroll does.**
1. Click a sheet on one CT slice; SAM 3.1 tracks it through the slices as video.
2. A CT-ridge step refines the trace; on the first slice a local walk keeps it on
   the clicked sheet instead of jumping to a brighter neighbour.
3. Every trace is measured against human-traced layers released for Scroll 1,
   with exact point-to-triangle distances.
4. A Muse Wonder agent (tool calling) inspects a trace, re-seeds where tracking
   stops, launches Kaggle GPU runs, scores and unrolls the result.
5. Muse Spark reviews traces from images and writes plain-language QA notes for
   scholars; every number in a note is checked automatically against the data.
6. Traced sheets are unrolled into flat images; the 2023 Grand Prize ink model
   (MIT) looks for ink; Muse Spark reads and translates a published passage
   with human correction and an audit trail.

**Results on real Scroll 1 CT** (15 clicks on human-traced layers, 20 slices):
- Within 5 voxels (40 µm) of the human layer, median: 36.7% → **55.1%**;
  seeds on the clicked layer 6 → **10** of 15.
- Held-out (15 new clicks, never used in development): 36.2% → **48.5%**; the
  fix trades some coverage for accuracy (234 vs 285 of 300 slices traced).
- 100-slice trace: 87.3% within 5 voxels where a reference exists. The Wonder
  agent extended it from 58 to 97 slices, then its own seam check (numbers plus
  a Muse Spark verdict) found the first re-seed had jumped sheets; it excluded
  that run and its two descendants and kept the verified 58 slices.
- Letters: on a sheet with known Greek text, Unscroll's data path reproduces the
  Grand Prize ink map from raw CT (correlation 0.83). Our own traces are not yet
  accurate enough for ink (7 voxels median with ten scholar clicks; ink needs
  ~2), so no new reading is claimed. Our 3D-normal unrolling reproduces the
  letters on the official surface (0.82): the gap is trace accuracy.
- Triage: reference-free risk ranks traces so reviewers check the riskiest
  first. Held-out: reviewing 73% of traces catches every wrong-layer trace
  (Muse Spark reviewer: 90%); AUC 0.66. An earlier rule failed on held-out and
  is reported.
- Engineering: found and fixed a 4.8 GB attention allocation that crashed 16 GB
  GPUs and stalled 8 GB laptops (exact chunked attention, bit-identical output).

**What it is not.** It has not read new text, and its traces are not perfect.
The Muse Spark reviewer is safe (never accepted a wrong-layer trace, 0 of 26) but
does not yet reduce review work (it flags 90%+ of traces).

## Built during the hackathon (paste into the form; rules require it)

All Unscroll code was written during the hackathon, from 27 October 2026; nothing
was carried over from an earlier project. Built here: the click-to-trace pipeline
around SAM 3.1 and its first-slice fix; four fixes to SAM 3.1's code
(including the 4.8 GB attention fix); the evaluation against human-traced layers,
the classic baselines and statistics; the Muse Wonder agent, Muse Spark reviewer,
QA notes and triage; unrolling, the ink drift experiment and the Kaggle runners;
the web app with voice review, Read & Translate, Predict and the interactive
Story page (crush a scroll, scrub 50 real slices, push a trace off the ink).
Used, not built: SAM 3.1 and the Muse models (Meta); the 2023 Grand Prize ink
model (MIT), whose output includes the full-page ink map; Scroll 1 CT, the
human-traced layers and the published transcriptions (Vesuvius Challenge,
CC BY-NC 4.0); the Perseus Greek corpus (CC BY-SA 4.0). Full list: README,
"What was built during the hackathon".

## Meta models used

| Model | Use |
|---|---|
| SAM 3.1 (open weights; Kaggle T4) | Tracks the clicked sheet through CT slices as video |
| Muse Wonder (Meta Model API, tool calling) | Patch-growing agent (`scripts/grow_agent.py`) |
| Muse Spark 1.3 (Meta Model API) | Trace reviewer, QA notes with number checks, seam verdicts for the agent, Greek reading and translation |
| Muse Spark 1.3, Predict page | Completes Greek sentences that are not in the attested corpus; answers are labelled as conjecture and checked to contain the fragment |
| Muse Voice Transcribe (Meta Model API) | Push-to-talk voice commands for hands-free review in the workbench |

Not used on purpose: Muse Image (generated scroll imagery would mislead
scholars); Muse Glimmer and Spark Open (do not fit a 16 GB T4 at useful precision).

## How to run

```powershell
.\.venv\Scripts\python.exe scripts/workbench.py --open
```
Opens `http://127.0.0.1:8765/home.html`; works offline. For a first look, open
`story.html`: a five-chapter interactive explainer written for a newcomer. GPU batch jobs run on
Kaggle via `scripts/kaggle_run.py` (see README "Compute").

## Evidence for every claim

| Claim | Where |
|---|---|
| 36.7% → 55.1%, 6 → 10 seeds | `reports/seeded_eval/score.json`, `score_walk.json`, `RESULTS.md` |
| Held-out 36.2% → 48.5% | `score_heldout.json`, `score_walk_heldout.json` |
| 100-slice trace, 87.3% | `reports/flatten/score_20231210121321_w100.json` |
| Wonder agent: extension, then self-correction to 58 verified slices | `reports/agent/*.jsonl`, `reports/flatten/excluded_20231210121321.json` |
| Triage: held-out 73% review catches all, AUC 0.66 (v1 rule 0.52) | `reports/triage/triage.json`, `scripts/triage.py` |
| Spark reviewer 0/26 let through, 90–93% review load | `reports/spark_review/summary*.json`, `RESULTS.md` |
| QA notes, 15/15 number checks | `reports/qa/*.json` |
| Letters via control, corr 0.83 | `reports/letters/summary.json`, `docs/ink-research.md` |
| 3D-normal unrolling, corr 0.82 (ours 0.21) | `reports/letters/assisted/summary.json` (`control_own_normals`, `ours3d`) |
| Attention fix bit-identical | `RESUME.md`, `external/sam3/sam3/model/decoder.py` (`_unscroll_sdpa`) |
| Full-page ink map from CT: whole segment 20230827161847, GP model, 8.4 min on T4; Spark line reading unscored | `reports/ink_gp_full/`, `scripts/build_fullpage.py`, `scripts/read_fullpage.py` |
| Read & Translate: 4 published Scroll 1 columns (33 lines); Spark line reading 79% character error, 0/33 exact | `data/reading/passages/`, `reports/reading/columns/summary.json` |
| Predict: 99.3% of 900 fragments, 890/890 attested answers correct; Spark alone 1/40 exact | `reports/predict/eval.json` |
| 74 new clicks: fix vs original +16 points (p = 0.0002), vs optical flow +4.5 (2.2–7.3) | `reports/seeded_eval/compare.json`, `scripts/compare_methods.py`, `scripts/baseline_flow.py` |
| Ink drift: steady offset tolerated, wobble destroys letters | `reports/letters/shift/summary.json` |
| Smoothing raises ink agreement on our trace 0.21 → 0.73 (plateau at σ 12–16; official 0.83; broad strokes, no letters) | `reports/letters/assisted_smooth/summary.json`, `reports/letters/assisted_smooth_s1216/summary.json` |
| Cost of reading, same sheet with 1 / 5 / 10 clicks: 19% / 29% / 32% of the trace on the sheet, covering 362 / 165 / 278 of 400 slices; best agreement 0.48 / 0.62 / 0.73 over what each covered (not a ranking; no letters) | `reports/letters/{unassisted_smooth_one,assisted80_smooth_a80,assisted_smooth_s1216}/summary.json`, Evaluation page "cost of reading" |
| Story page: 5-chapter interactive explainer on real data (scrub 50 slices, drift slider, 74-dot test) | `viewer/story.html` |
| 2.4 µm probe: 32/32 runs, fixed tracer median 27 µm from the 2026 layer | `reports/probe24/summary.json`, `scripts/probe24_kaggle.py` |
| tifxyz export for VC3D | `scripts/export_tifxyz.py`, `exports/tifxyz/` |
| Tests | `pytest tests`: 148 passed, 1 skipped (Node not installed) |

## Data, licences, permissions

- CT and released meshes: EduceLab-Scrolls, Scroll 1 (PHercParis4), volume
  20230205180739; Parsons et al. (2023), arXiv:2304.02084; CC BY-NC 4.0 per
  https://scrollprize.org/data. The Vesuvius Challenge team confirmed on Discord
  that this hackathon entry may use the data (keep a screenshot of the reply).
- SAM 3.1: SAM License (copy shipped with the weights dataset).
- Grand Prize ink model: MIT (scrollprize/ink_detection_pipeline).
- An early ink-training experiment on the 2023 Kaggle competition fragments is
  not part of this submission or the public repository (competition data is for
  competition use); its outputs are excluded in `.gitignore`.
- Perseus Greek text (Predict corpus): CC BY-SA 4.0, see `data/predict/README.md`.

## Checklist (user)

- [ ] Record the video from `docs/video-script.md` (≤3 minutes), upload, paste link
- [ ] Create a **public** GitHub repository (rules: repositories must be public)
      and push. `.gitignore` already keeps out `.env`, weights, CT data, caches,
      scraped pages, unused third-party folders and competition-trained outputs
      (dry run: about 8,000 files, 300 MB, largest file 12 MB; no API key or
      Kaggle token found in them). Add a LICENSE for Unscroll's own code first
- [ ] Optional: make the Kaggle notebooks public for judges
      (unscroll-seeded-trace, unscroll-letters, unscroll-ink-gp)
- [ ] Attach or link the Vesuvius team's permission reply
- [ ] Submit by Tue Nov 3; Wed Nov 4 is buffer only
- [ ] Rotate the Kaggle token (it was pasted in chat earlier) after submission
