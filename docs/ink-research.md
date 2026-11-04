# Ink detection: what the public code says (researched 2026-10-30)

## Sources read

- 2023 Grand Prize ink model (Nader, Farritor, Schilliger): TimeSformer, MIT,
  https://huggingface.co/scrollprize/timesformer_GP_scroll1 and pipeline
  https://huggingface.co/scrollprize/ink_detection_pipeline (model.safetensors, 152 MB);
  inference script https://github.com/younader/Vesuvius-Grandprize-Winner (archived;
  development continues in ScrollPrize/villa).
- Current 9 µm ink models, MIT, trained on labels from four scrolls including
  PHerc. Paris 4 (Scroll 1): https://huggingface.co/scrollprize/ink_9um
  (code: villa, branch merge-ink-pipelines, `koine_machines`).
- 2023 Kaggle ink-detection competition (fragments only): podium ~0.68 F0.5;
  8th-place code https://github.com/alexbar445/vesuvius_challenge_8th_place_solution.

## Grand Prize model, exact settings (from the inference script)

- Input: surface-volume layers **17–42** (26 of 65), read as **8-bit**, clipped to
  **0–200**, divided by 255; tiles **64×64**, stride 21; Gaussian-weighted blending.
- TimeSformer(dim 512, image 64, patch 16, frames 30, depth 8, heads 6, dim_head 64),
  4×4 output per tile upsampled ×16, sigmoid.
- Caveat: the HF pipeline file stitches correctly only with `scale_factor=1`.

## Result (Kaggle T4, 4.8 minutes, 2026-10-30)

`scripts/ink_gp_kaggle.py` via `kaggle_run.py --ink-gp`. All 137 TimeSformer tensors
loaded strictly (names and shapes verified from the safetensors header first).
- **Validation — letters reproduced.** Segment 20230827161847, rows 200–5600 ×
  cols 1000–4600, layers 17–42 streamed on Kaggle: clearly legible Greek letter
  columns (`reports/ink_gp/val_pred.png`; 23% of pixels > 0.5, p99 0.92). The
  preprocessing, weights and tiling are therefore correct.
- **Unscroll's own surface — no ink.** 59 × 320 voxels (0.47 × 2.5 mm) is smaller
  than one letter and shorter than one 64-voxel tile; prediction is padded,
  blurred bands (p99 0.50, 1% > 0.5). Not evidence of absence of ink.
- Next: trace a taller patch on a sheet known to carry text, then run the same
  model on Unscroll's own unrolled surface.

## Letter-scale demo (Kaggle, 2026-10-30/31)

`kaggle_run.py --letters [--assist 40]` (scripts/letters_kaggle.py). Window rows
3650–4650 × cols 3000–3400 of segment 20230827161847 (z ≈ 10656–11068), chosen by
ink contrast. CT from the Scroll 1 OME-Zarr (360 chunks), calibration r = 0.977.

| | Traced slices | Within 5 vx of official surface | Median vx | Letters? |
|---|---:|---:|---:|---|
| Control (official surface re-sampled from Zarr) | — | — | — | **yes** (ink corr 0.83 with official) |
| Unscroll, 1 click + 6 automatic re-seeds | 362 | 18% | 15.7 | no |
| Unscroll, 10 simulated scholar clicks (1 per 40 slices) | 278 | 32% | 7.0 | no |
| Control unrolled with Unscroll's own 3D normals (`scripts/unroll3d.py`) | — | — | — | **yes** (ink corr 0.82; normals agree 0.996) |
| Assisted trace, 3D-normal unroll + reference-free depth snap | 278 | 34% | — | no (ink corr 0.21) |

Why ours shows no letters: ink lies on the papyrus surface, so the model needs
~2-voxel alignment; ours is ~7 voxels off even with assistance. Unscroll also
unrolls slice by slice with in-plane normals, while this sheet is oblique to
the slices; the official surface uses 3D normals.

**3D-normal unrolling (2026-10-31).** `scripts/unroll3d.py` builds a smoothed
surface grid from traced curves, takes normals from its 3D tangents and samples
65 layers trilinearly; a reference-free "snap" picks the sheet side and depth
shift (±8 vx) that maximise ink-map contrast. On the official surface it
reproduces the letters with our own normals (corr 0.82 vs 0.83 with the
official normals), so the unrolling is correct. On our assisted trace it is
not enough (corr 0.21, no legible letters): the snap chose the edge of its
range (−8 vx), consistent with the trace sitting on or near a neighbouring
sheet. The limit is trace accuracy, not unrolling. Next step: surface-level
(not centre-line) alignment and a wider snap range.

## Conclusions for Unscroll

1. **Use scroll-trained public models for scroll ink, not fragment-trained ones.**
   The 2023 Kaggle data are detached fragments; the Grand Prize and the later
   9 µm models were trained on scroll segments, including Scroll 1. Our patch is
   Scroll 1 at 7.91 µm, close to the 9 µm models' training resolution.
2. **Validate on known ink first.** Run the Grand Prize model on a released
   Scroll 1 segment with known letters (e.g. 20230827161847, used in the official
   example) entirely on Kaggle (layers streamed from dl.ash2txt.org; nothing
   downloaded locally). If that does not show letters, our pipeline is wrong.
3. **Our traced patch is smaller than one letter.** 2.53 × 0.47 mm vs letters of
   roughly 2–4 mm, so even a perfect ink model can show at most part of a stroke.
   Letter-scale results need a trace several hundred slices tall; Kaggle can
   stream and crop those slices itself (no local transfer).
4. **Fragment training (scripts/ink_kaggle.py) is a benchmark, not the scroll
   model.** Keep it for a reproducible F0.5 on fragment 3 if GPU time allows.
