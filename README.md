# Unscroll

**Click one layer of a 2,000-year-old scroll. Know whether the AI stayed on it.**

The Herculaneum scrolls were carbonized by Vesuvius in 79 AD and can only be
read from CT scans, by tracing each papyrus sheet through thousands of slices
where neighbouring sheets are crushed together. Unscroll turns one click into a
SAM 3.1 trace on real Scroll 1 CT, then measures whether that trace stayed on
the clicked sheet, against layers people traced by hand.

Built for the **Meta Global AI Developer Hackathon**, Indie / Push the Frontier.

## What was built during the hackathon

All Unscroll code in this repository was written during the hackathon, starting
27 October 2026. Nothing was carried over from an earlier project.

**Built here**
- Click-to-trace pipeline around SAM 3.1: prompt design, CT-ridge refinement and
  the first-slice walk fix (`src/trace/`).
- Four fixes to the SAM 3.1 code, shipped as `patches/sam3-unscroll.patch`:
  an exact chunked attention path that removed a 4.8 GB allocation
  (`decoder.py`), a CPU fallback where Triton is unavailable on Windows
  (`edt.py`), an init-argument filter and a boolean sort cast.
- The evaluation: clicks placed on human-traced layers, exact distances to those
  layers, development, held-out and 59-click test sets, two classic optical-flow
  baselines and the paired statistics (`scripts/seeded_eval.py`,
  `scripts/baseline_flow.py`, `scripts/compare_methods.py`, `src/eval/`).
- The Muse Wonder re-seeding agent and its seam checks (`scripts/grow_agent.py`),
  the Muse Spark reviewer, number-checked QA notes and review triage.
- Unrolling of traced sheets (`scripts/flatten_trace.py`, `scripts/unroll3d.py`),
  the ink drift experiment and every Kaggle runner (`scripts/*kaggle*.py`).
- The web app: surface workbench with voice review, Evaluation, Read & Translate
  and Predict pages, and the local server (`viewer/`, `scripts/workbench.py`).

**Used, not built here (credited)**
- SAM 3.1 model and code: Meta, SAM License. Not included in this repository;
  see "Set up SAM 3.1" below. Unscroll only adds the patch. Muse Spark, Muse Wonder and Muse Voice Transcribe: Meta
  Model API.
- The 2023 Vesuvius Challenge Grand Prize ink model (MIT). Ink maps on the
  official segment, including the full-page map in Read & Translate, are that
  model's output; Unscroll's part is running it and the measurements around it.
- Scroll 1 CT and the human-traced layers: EduceLab-Scrolls / Vesuvius
  Challenge, CC BY-NC 4.0, used with the Vesuvius Challenge team's permission.
- Published Scroll 1 transcriptions and column images: Vesuvius Challenge
  Grand Prize page, CC BY-NC 4.0. Greek corpus for Predict: Perseus Digital
  Library, CC BY-SA 4.0 (`data/predict/README.md`).
- The translation-prompt improvement loop (`src/agents/harness.py`) follows the
  method of google-research/harness2 (Apache 2.0); the code is Unscroll's own.

## Vesuvius Challenge open problems this addresses

The Vesuvius Challenge lists what still blocks reading the scrolls in its
[2026 open problems](https://scrollprize.org/2026_open_problems). Unscroll speaks
to three of them; it is a measuring tool for tracers, not a replacement for the
community's tools (VC3D, GrowPatch, Lasagna), which work on newer 2.4 µm scans.
Unscroll works on the 7.91 µm Scroll 1 volume.

| Open problem | What the Challenge says | What Unscroll contributes |
|---|---|---|
| [#2 Meshing and tracing](https://scrollprize.org/2026_open_problems) | Sheet switches onto neighbouring wraps and mergers are the main failure modes; one small error sends a mesh onto the wrong wrap. | Measures sheet switches directly against human-traced layers, on 89 clicks with paired baselines; the first-slice fix reduces them. |
| [#5 Label quality](https://scrollprize.org/2026_open_problems) | Human meshes drift and jump layers; active learning should ask people to fix only uncertain regions. | Reference-free triage ranks traces riskiest-first; on held-out clicks, reviewing 73% catches every wrong-layer trace. |
| [#6 Ink detection](https://scrollprize.org/2026_open_problems) | Better diagnostics are needed; "no ink" is hard to tell from "ink not yet recovered". | The drift test: a steady surface error under 5 voxels keeps the letters, slice-to-slice wobble of 2–5 voxels destroys them. |

The Challenge also notes that no standalone benchmark for tracing exists; the
paired evaluation here (`scripts/seeded_eval.py`, `scripts/compare_methods.py`)
is a small one for Scroll 1.

## Results on real data

**Headline (74 clicks never used while building the fix):** the share of each
trace within 40 µm (5 voxels) of the sheet people traced by hand rose from 25%
to 50% (median; better on 50 clicks, worse on 19, p = 0.0002), and SAM 3.1 with
the fix beats a classic optical-flow tracer from the same start by +4.5 points.
Wobble from slice to slice, not distance, is what destroys the letters, and
smoothing Unscroll's own trace raises ink agreement from 0.21 to 0.73 (0.83 is the
best possible here), in broad strokes, not readable letters.

**Development clicks.** 15 clicks, each placed exactly on a sheet released by the
Vesuvius Challenge community, traced through 20 slices of Scroll 1 (PHercParis4)
and scored with exact point-to-triangle distances. The reference never enters
the tracker. These clicks were used to build the fix.

| | Original first slice | First-slice ridge walk |
|---|---:|---:|
| Share of trace within 5 voxels (40 µm) of the human layer, median | 36.7% | **55.1%** |
| Within 2 voxels, median | 12.1% | **19.4%** |
| Seeds staying on the clicked layer (median distance ≤ 6 voxels) | 6 / 15 | **10 / 15** |
| **Held-out clicks** (15 new, never used in development): within 5 voxels, median | 36.2% | **48.5%** |

On held-out clicks the walk also traces fewer slices (234 vs 285 of 300): it
trades some coverage for accuracy.

- **Diagnosis:** tracking between slices was stable; the error was choosing the
  sheet on the first slice, where a global brightest path crossed to a brighter
  neighbour. A local walk from the click fixed 12 of 15 seeds; one got worse.
- **How far a trace can drift before the letters go:** on the released sheet
  with known Greek text, the official surface was moved off the papyrus on
  purpose and the Grand Prize ink model run again (agreement with the official
  ink map 0.83 with no error). A steady offset costs little: 0.81 at 4 voxels,
  0.76 at 7. Slice-to-slice wobble, as in a real trace, destroys the letters:
  0.63 at a median of 1 voxel, 0.50 at 2, 0.01 at 5. What decides whether a
  trace can be read is staying steady from slice to slice, more than its
  distance from the sheet (`scripts/letters_kaggle.py --shift-curve`,
  `reports/letters/shift/summary.json`).
- **Larger test and classic baselines (74 new clicks, never used in
  development; `reports/seeded_eval/compare.json`):** the fix lifts the share
  within 5 voxels from a median of 25% to 50%, better on 50 clicks and worse on
  19 (mean gain +16 points, 95% interval 11–21; sign test p = 0.0002). Against
  two classic optical-flow tracers started from the same first slice, SAM 3.1
  with the fix is better by +4.5 points (2.2–7.3, better on 44 clicks, worse on
  13) and +3.4 points (1.0–6.2). It traces fewer slices (1,070 of 1,480; it
  stops when it loses the sheet), while optical flow never stops.
- **Steadiness filter (`scripts/smooth_traces.py`):** smoothing each trace
  across slices cuts slice-to-slice wobble ninefold (median kink 0.97 → 0.11
  voxels) but barely changes accuracy (55.1% → 55.3% within 5 voxels): the
  remaining error is a steady offset or a wrong sheet, not jitter.
- **Smoothing brings back ink on our own trace (`scripts/letters_kaggle.py --sigmas 2 4 8`,
  `reports/letters/assisted_smooth/`; σ 12 and 16 in `assisted_smooth_s1216/`):** the same assisted trace on the
  known-text sheet, unrolled after smoothing across slices, agrees with the
  official ink map at 0.21 (σ = 2 slices, as before), 0.44 (σ = 4), 0.63
  (σ = 8), 0.72 (σ = 12) and 0.73 (σ = 16): a plateau about 0.1 below the
  official surface (0.83). The re-run at σ 12/16 reproduced the identical trace.
  Caveats: the maps show broad stroke shapes, not crisp letters, so part of the
  gain may be smoother maps matching large strokes, and the depth snap chose a
  different offset at each strength (0 at σ 12 and 16). The σ 8, 12 and 16 maps
  look alike to the eye: one broad stroke, no letters. No new text is claimed.
- **The cost of reading (`--tag one`, `reports/letters/unassisted_smooth_one/`,
  Evaluation page "cost of reading"):** the same sheet traced from one click
  with no help at all: median 125 µm from the official surface, 19% within
  40 µm (the 10-click assisted trace: 55 µm, 32%), over a narrower patch
  (417 columns vs 786). Its smoothed ink agreement is 0.45 / 0.48 / 0.40 / 0.42
  at σ = 2 / 4 / 8 / 16. That is not "one click beats ten": the score is
  averaged over a different, smaller area, and neither trace shows readable
  letters. With a click every 80 slices (5 clicks, `assisted80_smooth_a80/`) the
  trace covers only 165 of the 400 slices (the 10-click trace: 278), at 56 µm
  median and 29% within 40 µm; agreement 0.58 / 0.62 / 0.60 / 0.54 at σ = 2 / 4
  / 8 / 16 over that shorter piece, and its maps show no strokes. So none of the
  three scores is a ranking: each is averaged over what that trace covered.
  Only the 10-click trace covers most of the window at a steady distance.
- **Probe on the Challenge's current 2.4 µm scan (`scripts/probe24_kaggle.py`,
  `reports/probe24/`):** the unchanged tracer ran on Scroll 1's 2.4 µm volume
  (streamed from the open data bucket) for 8 clicks on the community's 2026
  trace of segment 20231210121321, before and after the fix. All 32 runs
  completed. At 2.4 µm the fixed tracer stays a median 27 µm from that layer
  (71% of the trace within 40 µm; original first slice: 34 µm, 64%); at 4.8 µm
  it does worse (50 µm, 38%). Caveats: 8 clicks on one sheet, short traces
  (about 0.5 mm), a reference on a 48 µm grid that crosses some slices twice,
  and densely packed sheets, so same-sheet tracking is not confirmed.
- **Two tracers as an error flag (negative result, `scripts/tracer_agreement.py`):**
  where SAM 3.1 and optical flow disagree does not predict wrong-sheet traces
  (AUC 0.50 on new clicks). Both start from the same first slice, so a wrong
  sheet chosen there is shared: the first slice is where errors are decided.
- **Is the reference right? (`scripts/reference_check.py`):** the Vesuvius
  Challenge's 2026 re-trace of the same sheet (segment
  `20260604223808-20231210121321_v8`), mapped onto this 7.91 µm volume, lies a
  median 8.8 voxels from the 2023 reference. On the CT it is a coarse 20-voxel
  grid that cuts across the papyrus bends, while the 2023 reference and
  Unscroll's trace follow the sheet (`reports/seeded_eval/reference_check_z6120.png`),
  so scores stay against the 2023 reference.
- **VC3D-compatible export (`scripts/export_tifxyz.py`):** traces export as
  tifxyz (x/y/z float32 grids plus `meta.json`, invalid points −1), the format
  VC3D and the Vesuvius tools read; example in
  `exports/tifxyz/unscroll_20231210121321_w100/`.
- **Muse Spark reviewer:** given only the CT and the trace, it never accepted a
  wrong-layer trace (0 of 26), but it flagged most good traces too, so it is a
  safe second opinion rather than a workload saving.
- **Triage (which traces to check first):** reference-free risk signals (does the
  trace pass through the click, does it sit on a bright sheet core, do the two
  first-slice methods agree) rank traces riskiest-first. On held-out clicks,
  reviewing 73% of traces catches every wrong-layer trace and 63% catches 80%
  (AUC 0.66; development 0.88; Spark reviewer: 90% review load). A first rule
  written before any results failed on held-out (AUC 0.52) and is reported.
  Per-slice signals do not predict error within a trace, so no per-slice
  confidence ribbon is shown (`scripts/triage.py`, `reports/triage/triage.json`).
- **Engineering:** found and fixed a 4.8 GB attention allocation that crashed
  16 GB GPUs and stalled 8 GB laptops (exact, chunked; bit-identical output).
- **Muse Wonder agent:** given tools to inspect, re-seed, run Kaggle traces,
  score and unroll, it extended a 100-slice trace from 58 to 97 slices. Given a
  seam check (continuity numbers plus a Muse Spark visual verdict), it then found
  that its first re-seed had jumped to another sheet (181 voxels apart, length
  ×4.4), excluded it and the two runs built on it, and kept the verified 58
  slices. Every decision is logged in `reports/agent/`.
- **Predict (complete a sentence from a fragment):** type any part of a Greek
  sentence (beginning, middle or end; accents optional; Latin letters accepted;
  edge words may be cut) and get the whole sentence with its reference from
  attested text (Diogenes Laertius incl. Epicurus, Perseus CC BY-SA, plus the
  PHerc. Paris. 4 passage). Measured on 900 masked fragments: 99.3% recovered,
  and 890/890 answers shown as "attested" were correct. When nothing is attested,
  Muse Spark proposes completions, labelled as conjecture; alone it reproduces
  only 1 of 40 unseen sentences exactly (16% of missing words), which is why the
  lookup runs first (`src/agents/complete.py`, `scripts/eval_complete.py`,
  `reports/predict/eval.json`).
- **Reading letters:** on a Scroll 1 sheet with known Greek text, the 2023 Grand
  Prize ink model reads letters from the official surface, and Unscroll's data
  path reproduces them from raw CT (correlation 0.83). Unscroll's own traces are
  not yet accurate enough for ink (7 voxels median with 10 scholar clicks; ink
  needs ~2), so no new reading is claimed. Unscroll's own 3D-normal unrolling
  (`scripts/unroll3d.py`) reproduces the letters on the official surface (0.82),
  so the remaining gap is trace accuracy, not unrolling.

Details: `reports/seeded_eval/RESULTS.md`, `reports/spark_review/RESULTS.md`,
`docs/ink-research.md`, `reports/agent/`.

## Set up SAM 3.1

The SAM 3.1 code is not redistributed here. Clone it at the commit Unscroll was
built on and apply Unscroll's patch:

```powershell
git clone https://github.com/facebookresearch/sam3.git external/sam3
git -C external/sam3 checkout f1647e8d0c1d280991f4d1b7d974c2704537fe66
git -C external/sam3 apply ../../patches/sam3-unscroll.patch
```

The SAM 3.1 checkpoint (`checkpoints/sam3.1_multiplex.pt`) comes from Meta under
the SAM License and is not included either.

## Meta models

| Model | Role |
|---|---|
| **SAM 3.1** (open weights, Kaggle T4 or local) | Tracks the clicked sheet through the CT slice stack as video |
| **Muse Wonder** (Meta Model API, tool calling) | Patch-growing agent: decides where to re-seed, launches Kaggle runs, scores and unrolls (`scripts/grow_agent.py`) |
| **Muse Voice Transcribe** (Meta Model API) | Hands-free review in the surface workbench: push-to-talk commands ("next", "slice 20", "worst", "hide mesh", "mark"; some Tamil) via the local server (`viewer/voice.js`, `src/agents/voice.py`) |
| **Muse Spark 1.3** (Meta Model API) | Trace reviewer (`scripts/spark_review.py`); QA notes for scholars with every number checked (`scripts/qa_report.py`); reads and translates a published Greek passage with human correction |

Also used: the 2023 Vesuvius Grand Prize ink model (TimeSformer, MIT) for ink
detection, run on Kaggle (`scripts/ink_gp_kaggle.py`, `scripts/letters_kaggle.py`).
Muse Image is deliberately not used: generated scroll imagery would mislead.

## Compute

Batch GPU work runs on Kaggle T4 notebooks through `scripts/kaggle_run.py`
(private datasets, per-job notebooks; `stage`, `push-data`, `push-kernel`,
`status`, `fetch`; jobs: seeded evaluation, `--crop`, `--ink-gp`, `--letters
[--assist N]`). CT for the letter demo is read on Kaggle from the Scroll 1
OME-Zarr in 128³ chunks; nothing large is downloaded locally.

## Open it

Double-click `Launch Unscroll.cmd`, or run:

```powershell
.\.venv\Scripts\python.exe scripts/workbench.py --open
```

It opens `http://127.0.0.1:8765/home.html` and works offline. Pages: **Home**,
**Story** (an interactive explainer: crush a scroll, follow one sheet through 50 real
slices, push a trace off the ink, run the 74-click test), **Evaluation** (results, held-out check, before/after animations,
every seed, Spark QA notes, reading letters, unrolled surface, Spark reviewer),
**Surface workbench** (slice-by-slice review of a historical trace),
**Read & Translate**.

## Reproduce

```powershell
# real-data evaluation (tracing on Kaggle: scripts/kaggle_run.py; see RESUME.md)
.\.venv\Scripts\python.exe scripts/seeded_eval.py score --first-frame walk
.\.venv\Scripts\python.exe scripts/spark_review.py score --variant v1
.\.venv\Scripts\python.exe scripts/before_after.py
.\.venv\Scripts\python.exe scripts/build_evaluation.py
.\.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider --basetemp=.test-temp-local
```

## Open the surface workbench

The server binds only to
`http://127.0.0.1:8765/` and serves the `viewer/` directory, not the repository.
Stop it with Ctrl+C. The generated `viewer/index.html` also opens directly
offline; review-marker persistence depends on browser support for local files.

- Scrub all 50 slices and toggle corrected, previous and released-mesh overlays.
- Click the error chart or jump directly to the largest measured discrepancy.
- Inspect the flattened CT intensity and an oblique view of predicted geometry.
- Place review markers, export the queue, validation JSON or predicted OBJ mesh.
- Exports preserve the volume coordinate frame. Markers are proposals; they do
  not silently alter geometry or recompute scores.

This is a precomputed, reproducible replay. It does not run SAM inference in the
browser, recover new text, or present fixture translations as scientific results.

## Read & Translate

Choose **Read & Translate** in the workbench, or open
`http://127.0.0.1:8765/read.html`. This separate example uses the official
published image and preliminary Greek transcription of PHerc. Paris. 4,
column −8, lines 10–14 (excerpt), from the February 2024 Grand Prize release.

**More passages.** The picker at the top opens every Scroll 1 column published
with a transcription: column −8 (lines 2–14, the whole column), −7, −2 and −1
(33 image-linked lines; CC BY-NC 4.0, credited on the page). Line rectangles are
fitted to each image (`scripts/line_boxes.py`) and checked by eye; their starting
English and Tamil translations are unreviewed Muse Spark drafts
(`scripts/build_passages.py`). Muse Spark reading only the line crops of these
columns had **79% character error, 0 of 33 lines exact**
(`scripts/read_columns.py`, `reports/reading/columns/summary.json`): reading is
not yet automatic. Other scrolls with published title readings (PHerc. 172,
139, 1667) are listed with links only, because their images are not ours to copy.

**Full page from CT.** `kaggle_run.py ... --ink-gp --full-page` streams all 26
surface layers of released segment 20230827161847 (9163 × 5048 px) and runs the
Grand Prize ink model over the whole sheet on a Kaggle T4 (8.4 minutes).
`scripts/build_fullpage.py` turns it the right way up (rotated, mirrored,
levelled) and `scripts/read_fullpage.py` has Muse Spark read its four readable
lines. That reading is **unscored**: no published transcription is aligned to
this sheet, and it is an official segment, not one Unscroll traced.

- Select one of five image regions to select its Greek line, or use the cards.
- Save human corrections while preserving the published reading and audit trail.
- Inspect English and Tamil drafts. A Greek edit marks both translations stale.
- Edit each translation independently, or request both from Muse Spark.
- **Load recorded Spark draft** replays a real saved API response offline when
  the Greek matches the published excerpt. **Translate corrected Greek with
  Spark** makes a new request; only Greek text is sent, with the key on the server.
- Export the complete reading, source image rectangles, original/corrected
  Greek, translation provenance and audit history as UTF-8 JSON.

Translation output is explicitly a draft, not expert-verified scholarship.
The example is not text recovered from the experimental 50-slice trace.
Image rectangles are line-level annotations on the published illustration,
not CT coordinates. Source underdots and other uncertainty marks are retained.

```powershell
# Once, if the published image/source snapshot is not already cached
.\.venv\Scripts\python.exe scripts/fetch_reading_source.py
# Offline bundle build and exact published-text/image-integrity checks
.\.venv\Scripts\python.exe scripts/build_reading.py
.\.venv\Scripts\python.exe scripts/validate_reading_assets.py
# Optional live API check; uses configured META_MODEL_API_KEY from .env
.\.venv\Scripts\python.exe scripts/validate_reading_live.py
```

The reading workspace requires no local GPU. Saved editorial and recorded
model drafts work without an API; arbitrary corrected-text translation needs
the configured API or a human translation. See `reports/reading/RESULTS.md`.

## Reproduce the measurements

```powershell
# Public metadata and bounded mesh catalog scan (network required once)
.\.venv\Scripts\python.exe scripts/probe_mesh_catalog.py
.\.venv\Scripts\python.exe scripts/catalog_meshes.py

# Evaluate the existing frozen 50-slice traces; no GPU or API calls
.\.venv\Scripts\python.exe scripts/audit_reference_frame.py
.\.venv\Scripts\python.exe scripts/validate_mesh_reference.py
.\.venv\Scripts\python.exe scripts/build_workbench.py

# Regression suite and the separate historical offline fixture replay
.\.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider --basetemp=.test-temp-local
.\.venv\Scripts\python.exe scripts/demo.py --offline
```

The original cached-mask tracing replay is `scripts/validate_departure_fix.py`.
Its finalized output is `reports/departure_fix/verified/`. Reference meshes are
never inputs to that extraction. `reports/mesh_reference/validation.json`
contains exact geometric agreement, the original proxy scores, per-slice data,
source hashes, unresolved catalog entries and measured CPU runtime/memory.

## What the metrics mean

| Measurement | What it establishes |
|---|---|
| Trace length within 2 voxels of fixed released mesh | Geometric agreement with that surface, subject to coverage and target identity |
| Exact mean / p95 triangle distance | Spatial discrepancy without sparse-vertex or Monte Carlo bias |
| Reference-to-trace coverage in fixed ROI | Slice-length coverage of that reference; not whole-mesh area recall |
| Ridge proximity | CT intensity support, a proxy that can favor the wrong layer |
| Continuity / kink guards | Local temporal and geometric consistency; not semantic correctness |

The original reference was chosen for z overlap alone. Its XY region differs
from the traced patch. OBJ XYZ already means TIFF column, row, physical slice;
no fitted offset is justified. The new evaluator rejects mismatched volume IDs,
wrong TIFF dimensions and missing physical z. It selects a candidate once using
the original click, then fixes that ID for both trace versions and all slices.
Partial annotations and an unconfirmed target are reported explicitly.

## Hardware budget

Test machine: RTX 4060 Laptop, 8 GB VRAM; 32 GB system RAM. Workbench replay and
mesh validation are CPU-only. The builder memory-maps one TIFF crop at a time;
the browser retains at most eight small crop images. Mesh queries use a bounded
NumPy BVH. Catalog scanning has per-file limits and records unresolved files.
SAM uses nearly all available VRAM: run tracking and ink training sequentially.
No whole-scroll model training or new multi-gigabyte volume download is needed
for this demonstration. The existing environment is reused; its legacy SciPy
and OpenCV pins conflict with NumPy, so the new evaluator deliberately depends
only on NumPy, not those libraries. A clean-environment lock remains future work.

## Scientific provenance and attribution

Data: **EduceLab-Scrolls**, Scroll 1 / PHercParis4, volume `20230205180739`,
7.91 µm voxels, physical slices 6100–6149. Cite Parsons et al. (2023),
[EduceLab-Scrolls](https://doi.org/10.48550/arXiv.2304.02084).
See the [official data and license guidance](https://scrollprize.org/data).
Public mesh URLs and SHA-256 hashes are recorded in the report. Large volumes,
model checkpoints and source meshes remain local cache inputs; do not bundle
them into the submission. The small CT crops and texture in the local viewer
are derived research data and retain dataset attribution.

Design and adversarial review notes: `docs/designs/surface-workbench.md`.
Judge walkthrough: `docs/hackathon-demo.md`.


### Image recognition in Read & Translate

The workspace now offers live Spark recognition and a recorded image-only baseline
for five manually located Greek line regions. Only the actual PNG crops and
anonymous IDs enter the recognition request; reference scoring happens afterwards.
Click a region, review its prediction, explicitly apply it, then correct and save.
Exports preserve predictions, image/crop hashes, regions, acceptance history and
translation stale/current state. Existing browser corrections survive loading a run.

The first real run took 23.3 seconds: **95.4% normalized character error, 90.2%
strict character error, 0/5 normalized exact lines**. This is not reliable automatic
transcription. Human review is necessary; the five public lines are not a held-out
benchmark. Replay works offline; live inference requires the configured Meta API.
CPU cropping requires no local GPU or new model download.

Rebuild offline: `.venv\Scripts\python.exe scripts/build_reading.py`.
Run a fresh, billed API validation: `.venv\Scripts\python.exe scripts/validate_recognition.py`.
See `reports/reading/RECOGNITION.md` for evaluation definitions and limitations.


## CPU OCR investigation update

Read & Translate now includes **Recognize locally (CPU)** and **Load recorded CPU OCR**.
The selected Tesseract ancient-Greek model uses grayscale denoising/contrast,
2× enlargement and raw-line mode. Two identical outputs measured 84.6% normalized
CER, 93.5% strict CER, 0/5 exact lines. It is faster (0.3–0.6 s) and has lower
normalized error than Spark, but worse strict error; neither is reliable OCR.
Measured OCR-process peak RAM: 56.45 MiB; no GPU. Existing Spark recordings remain.
Show both metrics and human review in the demo. No reference words enter inference.
Full experiment ledger and setup: `reports/reading/OCR-INVESTIGATION.md`.
