# Unscroll: click once, unroll anything rolled

* **Status:** Pivoted 2026-10-27 — public-data training replaces the synthetic generator (Part 1 dead). design_draft.md is now source of truth.
* **Date:** 2026-10-27
* **Track:** Push the Frontier (energy and materials)

> Saved verbatim from author's project brief. Target: from 2,000-year-old
> scrolls to battery cells.

## Concept

The adapted version keeps the wow factor and removes the three blockers:

| Blocker | Fix |
|---|---|
| **License conflict** (scroll data is CC BY-NC) | Build and prove everything on a **synthetic scroll generator you own**. Real Herculaneum data becomes an *optional* final demo, shown only if the Vesuvius team says yes. |
| **Terabyte-sized data** | Small synthetic volumes (about 512³ voxels) that fit your laptop and RTX 4060 |
| **"Did it really work?"** | Synthetic scrolls come with **perfect ground truth**: you know exactly where every layer and every letter is. |

On top of that, you widen the impact: **cylindrical lithium-ion battery cells are rolled sheets too**. The same algorithm can inspect electrode layers in battery CT scans, which puts the project squarely in "energy and materials" under Push the Frontier.

## The wow moment

1. A grey CT slice of a crushed, carbonized scroll. One click on a layer.
2. The layer lights up and **races through hundreds of slices**, peeling out as a glowing 3D spiral.
3. The spiral flattens into a sheet, and **Greek letters appear** where there was only grey before.
4. Muse Spark reads them out with confidence scores.
5. Cut to a battery-cell CT scan. Same click, same unrolling, and a **defect in the electrode layer** is highlighted.

## Architecture: four parts

### Part 1: Synthetic scroll and battery generator (your own data, ₹0)

Python and numpy only:

1. Render Greek text onto a virtual sheet using an open-licensed Greek font (the GFS fonts, licensed under the SIL OFL).
2. Roll the sheet into a 3D spiral in a 512³ volume.
3. **Crush it:** squash, add random warps and creases, and make neighbouring layers touch, which is exactly what makes real scrolls hard.
4. Add papyrus fibre texture and **ink as only a very slight density change**, since carbon ink barely shows up in real scans. Then add CT-style blur and noise.
5. **Battery mode:** alternating electrode and separator layers, with defects you plant on purpose: folds, tears, gaps.

The generator outputs the volume plus the **ground truth**: the true surface of each layer and the true letter positions.

It's also an open benchmark you can release under CC BY, so other teams and the Vesuvius community can use it.

### Part 2: Click-to-trace with SAM 3.1 (the core innovation)

- **Treating the scan as video:** the stack of CT slices is played as a video, and one click on a layer becomes a prompt to **SAM 3.1**, which tracks that layer slice by slice.
  - Run the open-weights SAM locally on the 4060 for bulk tracking.
  - Use the **Meta Model API's SAM 3.1** for the live demo and for checking tricky regions.
- **Catching layer jumps (the hard part):**
  - **Winding check:** the angle around the scroll's centre must keep increasing steadily along a layer.
  - **Spacing check:** the gap to the neighbouring layers must stay consistent.
  - **Smoothness check:** a sudden kink means SAM jumped to the next layer.
  - Suspect regions get re-prompted automatically, and **Muse Spark vision** reviews the remaining hard cases.

### Part 3: Flatten and detect ink

- **Flattening for free:** each slice gives a traced curve, and stacking the curves gives a surface that is *already* laid out flat. The position along the curve and the slice number become the flat image's coordinates, so no heavy flattening algorithm is needed.
- **Sampling the papyrus:** sample the volume a few voxels above and below the traced surface, which gives a thin "surface volume".
- **Ink detection:** a small 3D CNN trained on the synthetic ground truth, running in minutes on the 4060. On real scroll data, use the open-source ink models from past Vesuvius winners.

### Part 4: The agent layer (Meta Model API)

- **Muse Spark 1.3** reads the flattened sheet: Greek letter suggestions with confidence scores, plus a quality check of trace images.
- **Muse Wonder** is the orchestrator agent. It chooses where to click next, so later layers need no human clicks at all. It runs the pipeline and writes the report, e.g. "Layer 7: 94% traced, 2 jumps fixed, 31 letters recovered".
- **Muse Voice:** hands-free commands while inspecting, e.g. "trace the next layer outward", "show me the weakest region".
- **Muse Image:** a "what the scroll looked like" reconstruction for the video. Optional, and clearly labelled as an illustration.

**The web viewer** (three.js): the CT slice with live tracking, the 3D spiral, the flattened sheet with letters, and a metrics panel.

## Evidence (synthetic ground truth makes it bulletproof)

| Metric | How |
|---|---|
| **Share of layer area traced correctly**, measured as crushing gets worse | Compare against the true surface across 5 crushing levels |
| **Layer jumps per 100 slices**, before and after the checks | Shows the checks actually work |
| **Letter recovery rate** | Recovered letters against the true text |
| **Human clicks per cm²** | One click versus the dense manual work traditional tools need |
| **Battery mode** | Planted electrode defects found, and how precisely they're located |
| *(Optional)* real scroll | Rediscover letters in a **region already read and published** in Scroll 1, only with Vesuvius permission |

## How it scores

| Criterion | Score drivers |
|---|---|
| **Impact 30%** | Reading a lost ancient library, *plus* battery quality inspection: culture and energy with one method |
| **Tooling 25%** | SAM 3.1 video tracking applied to 3D volumes, an "unexpected capability" exactly as the rubric describes it. Plus Spark reading and checking, Wonder orchestrating, Voice, Image. |
| **Demo 25%** | A click, then a spiral peeling out, then letters appearing. Unforgettable. |
| **Depth 20%** | Your own CT simulator, geometric jump correction, surface flattening, a trained ink model, measured results |

## 8-day plan (Tue Oct 27 – Wed Nov 4)

| Day | Date | Work |
|---|---|---|
| 0 | Tue Oct 27 | Synthetic generator: rolled sheet with Greek text in a 512³ volume. **Post the permission question on the Vesuvius Discord.** |
| 1 | Wed Oct 28 | Crushing, touching layers, faint ink, noise; save the ground truth |
| 2 | Thu Oct 29 | SAM click-to-trace through the slice stack (local weights), first traced layer |
| 3 | Fri Oct 30 | Layer-jump checks and automatic re-prompting; first metrics |
| 4 | Sat Oct 31 | Flattening + surface volume + small ink CNN, **letters visible** |
| 5 | Sun Nov 1 | Muse Spark letter reading, Wonder orchestrator, battery mode. **Feature freeze** |
| 6 | Mon Nov 2 | Full benchmark across crushing levels; real-scroll demo *if* permitted |
| 7 | Tue Nov 3 | three.js viewer, README, open-source the repo and the benchmark |
| 8 | Wed Nov 4 | Video, submit by 6 PM ET (4:30 AM IST, Nov 5) |

## Risks

- **SAM jumps layers on heavily crushed areas:** that's the point of the crushing-level benchmark. Report where it breaks; honest limits score well on Depth.
- **Synthetic data isn't real data:** say so openly. The real-scroll demo (if permitted) and the open benchmark address it.
- **The Vesuvius team doesn't reply in time:** the project still stands on synthetic scrolls and battery mode. Nothing depends on their data.
- **Calling the API on every slice gets expensive:** do the bulk tracking with local open weights, and use the API for live, showcase and checking calls.

## After the hackathon

Submit the tracing tool and the synthetic benchmark to the **Vesuvius Progress Prizes** (monthly, up to $20k). Keep improving the tool toward the First Letters prizes ($50k per scroll) before June 2027.

## EGM Goldfish Review (2026-10-27)

Full synthesis: [reviews/goldfish_brainstorm_report.md](reviews/goldfish_brainstorm_report.md).
Raw per-lens reports: `reviews/raw/lens-*.txt` (verbatim).

Convergent findings (3+ lenses each): (1) SAM-on-touching-layers unproven,
no fallback; (2) VRAM/compute budget unclosed on RTX 4060; (3) metric
formulas undefined; (4) flattening parameterization drift unaddressed;
(5) no offline/API-free demo fallback. Plus 10 blockers, 8 majors, 6 nits.

Author decisions pending — see review report. My 3 pre-review questions
stand: flattening consistency (confirmed by reviewers), ink calibration
method (still open), resources team/API/VRAM (still open).
