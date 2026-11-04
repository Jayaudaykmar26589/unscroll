# Independent released-surface evaluation — 50 slices

**The tracing improved, but surface accuracy is not established.** The frozen
corrected trace agrees with the fixed released reference within 2 voxels over
only **6.95%** of its sampled arc length. The ridge proxy is **80.50%**; these
measure different things and must not be substituted for one another.

| Measurement | Previous continuity-snapped trace | Corrected CT-ridge trace |
|---|---:|---:|
| Mean exact distance to triangles | 20.899 voxels | 10.679 voxels |
| Median distance | 16.087 voxels | 11.216 voxels |
| 95th-percentile distance | 53.196 voxels | 18.294 voxels |
| Maximum sampled distance | 73.900 voxels | 23.514 voxels |
| Sampled trace arc within 2 voxels | 2.950% | 6.951% |
| Sampled trace arc within 5 voxels | 12.399% | 19.791% |
| Ridge proximity proxy | 66.054% | 80.502% |

The corrected run still has 49/50 continuity/kink passes, with the initial
reference unassessed, no >20-voxel along-curve steps and no >60-degree kinks.
Those are consistency checks, not independent surface labels. Mean surface
distance improves by 48.90%; this is not a 48.90-point accuracy gain.

## Root cause of the reference problem

The originally chosen mesh `20230518181521` is linked to the correct volume,
but lies in a different XY region. The original click is **2329.775 voxels**
from it. Z overlap was incorrectly treated as coverage of the clicked layer.
The prior approximate “330-voxel offset” diagnosis was not a valid coordinate
calibration. The documented transform is identity:

`OBJ (x, y, z) = TIFF (column, row, physical slice number)`.

The evaluator now requires a matching volume UUID and TIFF dimensions, refuses
local file index z as physical z, and rejects contradictory filename metadata.
It measures continuous triangle distance rather than nearest sparse vertices
or eight random points per face. The coordinate audit image shows the original
mesh on its own CT sheet, far from the clicked region. No fitted translation,
axis permutation or curve registration was introduced.

## Reference selection and limits

- Surveyed all **256** listed segment IDs with bounded downloads. **Four remain
  unresolved**: one missing OBJ (`20230514182829`) and three beyond the 64 MiB
  vertex-scan limit (`20231005123336`, `20231007101619`, `20231012184424`).
- Seventeen candidate mesh bounds intersect the fixed ROI. One belongs to a
  different volume (`20240101215220`, volume `20230206171837`) and is excluded.
- Among the sixteen compatible candidates, the nearest to the **original
  manual click** is `20231210121321`, at **12.107 voxels**. This ID is fixed for
  every slice and both trace versions. It is not selected again per frame.
- Target layer identity is **unconfirmed**. The numbers above are actual
  geometric agreement with that released reference, not a certified accuracy
  estimate for the intended layer. Partial or missing annotations are not
  proof that a traced layer is wrong, and nearby layers are not interchangeable.
- Corrected reference-to-trace coverage is **2.777% within 2 voxels** over the
  reference's slice intersections in the fixed ROI. This is slice-length
  coverage, not full surface-area recall; the reference can include more curve
  extent than the traced patch.
- Query samples are arc-length midpoints at spacing <=1 voxel, with arc-length
  weights. Distances to triangles are exact; trace-length percentages are
  numerical quadrature estimates. This tuned development patch is not held out.

## Reproducibility and hardware

`validation.json` contains per-slice scores, exact mesh URLs/hashes, metadata
hashes, frozen trace hashes, source proxy hash, catalog omissions and runtime.
No reference mesh was fed into extraction; no new SAM inference was run.

- CPU validation: **78.49 seconds**.
- OS peak resident memory: **492.93 MiB**; GPU use: **none**.
- Workbench bundle: **5.23 MB**, including 50 small CT crops.
- Full regression suite: **77 passed**. Offline fixture replay: **GOLDEN OK**.
- Browser verified: loading, physical slice 6149, previous-trace overlay, review
  markers, marker JSON export, largest-error navigation and OBJ export.
- Exported OBJ parsed successfully: **6,400 vertices, 12,446 triangles**, finite
  coordinates, physical z range **6100–6149**.

The strongest next experiment is a target-identified, reference-covered seed
and a held-out window, followed by measured review time. Do not claim the
project's original >=90% surface goal has been met.

See `coordinate_audit.json`, `coordinate_audit.png`, `workbench.png`, and the
implementation in `src/eval/meshref.py` and `scripts/validate_mesh_reference.py`.
