# Read & Translate: evidence-linked reading review

Implemented 2026-10-28 for the Meta hackathon demo.

The reading example is PHerc. Paris. 4, column −8, lines 10–14, ending at
νομίζομεν. The official Vesuvius Grand Prize announcement publishes both the
column illustration and a preliminary transcription. The passage discusses
scarce versus plentiful food and pleasure. It is separate from the experimental
50-slice CT surface and is never described as newly recovered Unscroll text.

## Workflow

1. Select a line in the published image or its transcription card. A manually
   located rectangle and selected card point to the same line ID.
2. Preserve the original scholarly reading, including uncertainty underdots.
   Save corrections as a separate revision with before/after audit history.
3. Both English and Tamil drafts become stale when their source Greek changes.
   Human edits can update each language separately. “Current” means attached
   to this reading, not validated accuracy.
4. Replay a real recorded Spark response offline for the unchanged published
   text, or explicitly request a new English/Tamil draft for corrected Greek.
5. Export source identity, image hash/regions, published and corrected text,
   translations, provenance, stale state and audit history as UTF-8 JSON.

## Important boundaries

- Image coordinates belong to the published 936×1244 column illustration;
  they are not CT voxel coordinates or exact word polygons.
- The excerpt is a fragment of a longer sentence/discussion. No missing Greek
  is filled in. Translation renders its proposed sense and remains unreviewed.
- Editorial drafts, recorded model drafts, new model drafts and human edits
  have distinct provenance. Numerical model confidence is not displayed.
- API keys stay on the Python server. Only the Greek text goes to Meta after
  an explicit click. Same-origin JSON POST, size limits and one concurrent
  request protect this local demo. There is no backend retry loop.
- A response is attached only if its source hash and the current revision
  still match. A late result cannot overwrite subsequent human work.
- Pending unsaved edits disable export/translation. Local storage failure is
  surfaced; export remains available. Persisted drafts are namespaced by the
  content-addressed source bundle and validated before restoration.

## Validation

Unit/state tests cover uncertainty preservation, Unicode normalization, source
hashes, image bounds, malformed provider output, Tamil-script presence,
cross-origin calls, secret-safe errors, concurrency, stale translations, human
overrides, source immutability, persistence and corrupt local storage.
Browser QA covers line selection, correction/reload, stale indicators, model
response, bilingual rendering, source restoration and exported provenance.
Live response evidence is in reports/reading/live-validation.json. This is
functional and semantic sanity validation, not expert translation benchmarking.


## Image recognition extension

Live and recorded image-only recognition now use five pinned, manually located crops. The server accepts only an empty recognition request; clients cannot supply reference text, URLs or arbitrary filesystem paths. Predictions are keyed by validated line IDs and image coordinates. Receiving them never changes Greek corrections. Explicit per-line acceptance records the run ID and before/after text; subsequent edits retain the original prediction. Translation source checks continue to govern staleness. See reports/reading/RECOGNITION.md for the unsuccessful automatic-recognition baseline and exact evaluation protocol.
