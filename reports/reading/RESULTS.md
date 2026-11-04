# Read & Translate — validation

Implemented as `viewer/read.html`, linked from the surface workbench. The
existing local launch command serves both pages. No local GPU is required.

## Published evidence

Source: https://scrollprize.org/grandprize#what-does-the-scroll-say,
published 2024-02-05. PHerc. Paris. 4, column −8, lines 10–14, ending at
νομίζομεν. The five-line excerpt discusses scarcity, plentiful food and
pleasure. The column illustration is credited to Vesuvius Challenge, with
EduceLab-Scrolls and Institut de France attribution in the UI and exports.

- All **five lines match** the cached official publication exactly after
  canonical Unicode normalization, including uncertainty underdots.
- The source image is pinned by SHA-256 and 936×1244 dimensions. Replacing it
  causes the builder to fail until its line regions have been rechecked.
- Five manually located line rectangles were visually inspected against the
  column. They are not word polygons or CT voxel mappings.
- Published text is immutable. Corrections live in a separate versioned draft.

## Translation validation

`reports/reading/live-validation.json` records a successful Muse Spark 1.3
request for the complete excerpt: **13.73 seconds**, English and Tamil
responses, uncertainty notes and an input SHA-256. A recorded-response button
makes this real result available without network access. It is disabled when
the Greek differs from that request's input.

The browser also successfully translated corrected Greek through the new
same-origin endpoint. Its exported reading contains the new input binding
and model provenance. API keys remain on the server and are never exported.

Semantic sanity review: both languages retain the negative claim and the
comparison between scarce and plentiful food. Tamil uses real Tamil script,
including the negative கருதுவதில்லை. The model's rendering of ἤδη as
“already” is literal and less idiomatic than the editorial draft's
“immediately.” These are **unreviewed drafts**, not an expert-scored translation
benchmark. No numerical model confidence or recovered-text accuracy is claimed.

## Functional validation

- **92 tests passed** (15 additional pytest cases; the browser-state case
  exercises multiple JavaScript invariants).
- Preserved Greek uncertainty, canonical Unicode hashes and source immutability.
- Greek corrections invalidate both translation drafts; updating English alone
  does not mark Tamil reviewed or current.
- Late model responses cannot overwrite a newer reading or human translation.
- Malformed output and fake/non-Tamil placeholders are rejected.
- Same-origin API checks, payload limits and secret-safe error responses tested.
- Browser: image line selection updates the corresponding editor; correction
  saves and survives reload; stale indicators appear; live bilingual translation
  succeeds; Tamil font rendering was visually checked; no console errors seen.
- Browser JSON export parsed and validated: original text retained, correction
  history present, and translation current/stale flags match their Greek source.
- An isolated local draft successfully loaded the recorded Spark result without
  an API call, then saved an independent human Tamil edit while retaining the
  English model draft and separate provenance.

The exported browser test example has revision 3: one sigma-form correction
and two model-translation events. It is a workflow example, not a new scholarly
reading. The interface's separate published-example disclosure is visible
throughout the demo. The experimental 50-slice CT result is unchanged.

## Files and reproduction

`source-manifest.json` records the published image, regions, exact text and
bundle identity. `asset-validation.json` records publication fidelity and
browser-export checks. `browser-export.json` is the tested reading export.
`live-validation.json` contains the saved real API response.

Run `scripts/build_reading.py`, `scripts/validate_reading_assets.py` and the
full pytest suite. `scripts/validate_reading_live.py` is an optional network/API
probe; offline replay needs no new inference. The revised three-minute
walkthrough is in `docs/hackathon-demo.md`.
