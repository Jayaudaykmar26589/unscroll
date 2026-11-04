# Muse Spark 1.3 as a trace reviewer

**Safe but not yet useful as a filter.** Across both image designs Spark never
accepted a wrong-layer trace (0 of 26), but it also flagged 13 of 16 and 14 of
16 good traces, so 90–93% of traces would still go to an expert.

## Setup

- 30 traces: 15 reference seeds x 2 first-slice modes (`curves/` DP,
  `curves_walk/` walk), from `reports/seeded_eval/`.
- Model: `muse-spark-1.3` (standard tier, not a contributor variant), via
  `src/agents/spark.py`. Raw replies, image and prompt hashes in
  `responses.jsonl` / `responses_tiles.jsonl` for offline replay.
- Input: CT crops with the red trace and a yellow click marker only. Released
  meshes are never shown to the model; they are used afterwards for scoring.
- Truth: on the clicked layer when the trace's median distance to that seed's
  mesh is <= 6 voxels. 16 traces on layer, 14 on another layer (v2: 28 traces,
  seed 20230518181521 has no slice-6100 trace in either mode).
- Accept rule (fixed before running): verdict `same_sheet` with confidence
  >= 0.7; v2 additionally requires no tile to report `switches_sheet`.

## Results

| | v1: overview panels | v2: six full-resolution tiles |
|---|---:|---:|
| Traces | 30 | 28 |
| Wrong-layer traces let through | **0 / 14** | **0 / 12** |
| Auto-accepted, all correct | 3 | 2 |
| Good traces sent to review | 13 / 16 | 14 / 16 |
| Sent to expert review | 90.0% | 92.9% |

v1 shows a close-up around the click plus the whole trace on slices 6100 and
6119 (downsampled). v2 was declared in advance as the single follow-up: six
native-resolution close-ups spaced along the slice-6100 trace, with per-tile
verdicts. Neither design separates good from bad traces; confidences cluster at
0.75–0.95 regardless of truth. No further prompt iteration was done on these
30 traces, to avoid fitting the test set.

## Interpretation

At this level of CT detail (sheets 10–30 voxels apart, crushed and touching),
the model treats nearly any red line in dense papyrus as crossing sheets. That
makes it a conservative second opinion, not an automatic accept/reject step.
Useful next tests, on new seeds only: giving the model the CT without the
overlay plus a separate trace mask, or a per-tile ranking task ("which of these
two traces follows one sheet?") instead of an absolute judgement.

Reproduce: `python scripts/spark_review.py render`, `render-tiles`,
`ask --variant v1|v2`, `score --variant v1|v2` (ask needs `META_MODEL_API_KEY`).
