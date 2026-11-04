# Reference-seeded evaluation on real Scroll 1 CT

**Update:** a local first-slice ridge walk raises the median share within 5
voxels from 36.7% to 55.1% on the same 15 seeds (12 improve, 1 regresses);
see "First-slice fix" below.

**Tracking between slices is stable; the error is choosing the layer at the
click.** 8 of 15 seeds stay on the clicked, human-traced layer for all 20 slices
(median distance 3–7 voxels). 7 of 15 are on a neighbouring layer from the very
first slice (median 10–39 voxels) and then follow that wrong layer steadily.

## Setup

- Data: EduceLab-Scrolls Scroll 1 (PHercParis4), volume `20230205180739`,
  7.91 µm voxels, physical slices 6100–6119. Parsons et al. (2023).
- Seeds: 15 of 20 compatible released segment meshes cross slices 6100–6149.
  Each click is the intersection-segment midpoint nearest the median of that
  mesh's intersection with slice 6100, so the intended layer is known.
  Meshes choose the click and score the result; the tracker never sees them.
- Tracer: unchanged `src.trace.sam3_tracker` (SAM 3.1 local weights, CT ridge
  refinement, guards), 2048-voxel crop, 20 slices, `--trim-memory`.
- Compute: Kaggle T4 16 GB, 2–4 minutes per seed (notebook
  `jaya260589/unscroll-seeded-trace`, version 4).
- Score: exact point-to-triangle distance from each trace (arc-length samples,
  ≤1 voxel spacing) to its own mesh. Other meshes passing within 2 voxels of
  the seed are treated as the same sheet.

## Results

282 of 300 slices produced a trace.

| Seed mesh | Median dist. slice 6100 | Median dist. slice 6119 | Within 5 vx | Verdict |
|---|---:|---:|---:|---|
| 20230606105130 | 3 | 3 | 64.4% | on layer |
| 20230613204956 | 4 | — (10 slices) | 82.3% | on layer |
| 20230929220926 | 4 | 4 | 72.7% | on layer |
| 20231106155351 | 3 | — (16 slices) | 95.3% | on layer |
| 20230702182347 | 6 | 5 | 41.3% | on layer, offset |
| 20230702185753 | 6 | 6 | 47.1% | on layer, offset |
| 20231031143852 | 5 | 7 | 36.7% | on layer, slow drift |
| 20231221180251 | 6 | 10 | 43.1% | on layer, slow drift |
| 20230518181521 | — | 10 | 26.9% | neighbouring layer |
| 20230629215956 | 15 | 18 | 19.2% | neighbouring layer |
| 20230701115953 | 18 | 19 | 14.5% | neighbouring layer |
| 20230701020044 | 21 | 25 | 15.5% | neighbouring layer |
| 20231210121321 | 23 | 23 | 22.2% | neighbouring layer |
| 20230627170800 | 34 | 37 | 2.2% | neighbouring layer |
| 20230619113941 | 39 | 30 | 1.2% | neighbouring layer |

Across seeds: within 2 voxels median 12.1% (mean 13.4%); within 5 voxels
median 36.7% (mean 39.0%); median of per-seed mean distance 12.0 voxels.
Per-slice medians: `per_slice_median.json`. Overlays at slice 6110:
`overlay_z6110.png`. Full per-seed scores: `score.json`.

## What this establishes

- **No drift or jumps along z for most seeds.** Per-slice median distance
  changes by at most a few voxels over 20 slices in 13 of 15 seeds.
- **Initial layer choice fails in about half the seeds.** The offset exists at
  slice 6100 itself, where the click lies on the mesh. The SAM mask and ridge
  refinement settle on an adjacent sheet in the dense, crushed region.
- **A ~3-voxel floor on good seeds** is consistent with the trace following the
  bright sheet centre while released meshes mark a sheet surface; within 5
  voxels is therefore the more informative tolerance here.
- "Closer to another annotated layer" is near zero mostly because neighbouring
  layers are usually not annotated, not because traces stay on target.

## Limits

15 seeds, one 20-slice block, development code, no held-out tuning split. Not a
certified accuracy estimate and not text recovery.

## First-slice fix: local ridge walk (`--first-frame walk`)

Diagnosis: on slice 6100 the curve is near the mesh within ~25 voxels of the
click and then crosses to a neighbouring sheet. The first-slice step solved a
global brightest-path DP over the whole (broad) SAM region, so a brighter
neighbour justified a short crossing of the dark gap. `src/trace/image_ridge.py`
`_walk` instead follows the clicked ridge outward, at most one voxel sideways
per step from its momentum-predicted position, stopping where the sheet fades.
Later slices are unchanged (motion + 1-voxel corridor). Opt-in; the original DP
remains the default. Unit test: `tests/test_ridge_walk.py`.

Same 15 seeds, same 20 slices, same Kaggle T4 setup (notebook version 5);
outputs in `curves_walk/`, `logs_walk/`, `score_walk.json`.

| Across 15 seeds | Global DP | Local walk |
|---|---:|---:|
| Within 5 voxels, median / mean | 36.7% / 39.0% | **55.1% / 55.2%** |
| Within 2 voxels, median / mean | 12.1% / 13.4% | **19.4% / 23.7%** |
| Median of per-seed mean distance | 12.0 vx | **7.5 vx** |
| Seeds with median distance <= 6 vx | 6 | **10** |
| Slices traced | 282 / 300 | 282 / 300 |

| Seed mesh | Within 5 vx, DP -> walk | Median dist., DP -> walk |
|---|---:|---:|
| 20231210121321 | 22.2% -> 90.3% | 23.0 -> 1.7 |
| 20230701020044 | 15.5% -> 59.7% | 22.1 -> 3.6 |
| 20230702182347 | 41.3% -> 80.5% | 5.9 -> 3.6 |
| 20231221180251 | 43.1% -> 79.0% | 8.2 -> 2.1 |
| 20230629215956 | 19.2% -> 49.4% | 16.0 -> 5.1 |
| 20230627170800 | 2.2% -> 31.2% | 36.1 -> 10.5 |
| 20230701115953 | 14.5% -> 31.7% | 18.9 -> 8.5 |
| 20230619113941 | 1.2% -> 15.5% | 33.4 -> 28.8 |
| 20230929220926 | 72.7% -> 83.6% | 3.6 -> 3.6 |
| 20231031143852 | 36.7% -> 45.6% | 6.2 -> 5.3 |
| 20230518181521 | 26.9% -> 29.8% | 10.8 -> 12.2 |
| 20231106155351 | 95.3% -> 96.0% | 3.2 -> 3.1 |
| 20230613204956 | 82.3% -> 78.9% | 3.0 -> 2.8 |
| 20230606105130 | 64.4% -> 55.1% | 3.3 -> 3.8 |
| **20230702185753** | **47.1% -> 2.2%** | **5.4 -> 13.7** |

12 of 15 seeds improve within 5 voxels; one regresses badly (`20230702185753`
walks onto a neighbouring sheet) and two dip slightly. Remaining failures:
`20230619113941`, `20230518181521`, `20230702185753`.

## Held-out check (new clicks, never used in development)

One new click per mesh on the same released layer, about 700 voxels from the
original click (`seeds_heldout.json`), same 20 slices, both modes, Kaggle T4
(notebook v6). The walk was not changed after the development run.

| Across 15 held-out seeds | Global DP | Local walk |
|---|---:|---:|
| Within 5 voxels, median / mean | 36.2% / 35.7% | **48.5% / 45.2%** |
| Within 2 voxels, median | 10.3% | **19.2%** |
| Median of per-seed mean distance | 14.7 vx | **8.1 vx** |
| Seeds with median distance <= 6 vx | 7 | **8** |
| Slices traced | **285 / 300** | 234 / 300 |

The gain holds on new clicks but is smaller than on the development seeds
(+12.3 vs +18.4 points within 5 voxels). 9 of 15 seeds improve and 3 get worse
by more than 5 points. The walk also stops more often (234 vs 285 slices
traced): it trades coverage for accuracy. Files: `score_heldout.json`,
`score_walk_heldout.json`, `curves_heldout/`, `curves_walk_heldout/`.

## 100-slice trace and unrolled surface (seed 20231210121321)

Walk mode, slices 6100–6199 (a 2560-voxel crop around the seed, Kaggle T4,
~10 minutes). The tracker kept the sheet for **59 of 100 slices** (to 6158),
then lost it. Over the 50 slices with released-mesh coverage (6100–6149): **87.3%
within 5 voxels, 56.6% within 2 voxels, median 1.7 voxels**, steady slice to
slice. Flattened with `scripts/flatten_trace.py` (1-voxel arc spacing, columns
anchored at the click): 320 × 60 voxels = 2.53 × 0.47 mm, showing crossed
papyrus fibres. No legible ink is claimed. Files: `reports/flatten/
curves_20231210121321_kaggle`, `sheet_20231210121321_w100`, `score_20231210121321_w100.json`.

**Caveat:** the walk was designed after diagnosing these same seeds. Its
parameters were not tuned on them (defaults as first written), but this is still
not a held-out result. The next step is a held-out block (different slices or
seeds) before calling the improvement general.
