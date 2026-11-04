# Unscroll surface workbench — implementation brief

Target: Meta Global AI Developer Hackathon, confirmed by the owner on 2026-10-28.
Scientific context: Vesuvius / EduceLab Scroll 1. This increment follows the local
EGM artifact-first method: define the claim, challenge it with independent
reference geometry, then build the demonstration around the evidence.

## Product claim

Inspect a single clicked papyrus layer across 50 CT slices, understand where
tracking is supported by released geometry, and export reproducible evidence.
SAM 3.1 supplies the tracked region; CT ridge refinement produces the curve.
The demo must distinguish image support, temporal continuity and surface
agreement. None of these is an ink-recognition or translation result.

## Design and constraints

- CPU-only local workbench, no installation of a frontend dependency stack,
  no CDN, API key or network required for replay. Static assets work offline.
- Slice scrubber, CT/trace/mesh overlay, before/after comparison, error chart,
  flattened texture, 3D wireframe, correction markers and JSON/OBJ export.
- Source geometry uses XYZ = column,row,physical slice. Match volume UUID and
  TIFF dimensions; fail on missing physical z. Never fit a reference to a trace.
- Choose a reference once from the original click; keep its identity for the
  whole experiment. Label incomplete coverage and unconfirmed identity.
- Exact point-to-triangle distances with a NumPy BVH; one-voxel arc quadrature.
  Fixed-ROI slice coverage is separate from full-mesh area recall.
- Stream one TIFF at a time. Only small crops reach the browser. One RTX 4060
  8 GB: do not load SAM and ink training together. Replay uses zero CUDA memory.

## Review findings and resolutions

1. A smooth wrong-layer trace can pass guards. Surface agreement is a separate
   panel, and failed geometry cannot inherit a green guard verdict.
2. The original four reference meshes were selected for z overlap alone. Scan
   the released catalog, record XYZ bounds, preserve volume links and hashes.
3. Sampling eight random points per face confounds mesh density with accuracy.
   Use continuous triangle distance; arc-weight trace metrics.
4. Selecting a different nearest mesh every slice hides layer switches. Fix one
   ID once; do not claim target identity solely because a neighbor is close.
5. Reference data can leak into tracing. Freeze trace hashes before evaluation;
   keep the evaluator separate from extraction and preserve original proxies.
6. A polished viewer can overclaim. Label the development benchmark, partial
   annotations, proxy scores and absence of recovered text directly in the UI.

## Verification and demo acceptance

Regression cases cover physical z, row/column order, incompatible volume/crop,
sparse triangle interiors, long faces, degenerate faces, negative OBJ indices,
wrong parallel layers, slice intersections and density-invariant metrics.
Run the full test suite, frozen 50-slice surface evaluation, offline replay,
and browser interaction checks. Ship report hashes, command recipes and a
short judge walkthrough. Do not present a hackathon win or new reading as an
achieved result.
