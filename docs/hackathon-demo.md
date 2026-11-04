# Judge walkthrough: Unscroll

Positioning: a surface-review tool that makes model-assisted scroll tracing
inspectable and measurable on an 8 GB laptop GPU. Do not pitch a demonstrated
new reading, complete automatic unwrapping, or guaranteed prize outcome.

The official event describes the Indie track in terms of creativity, technical
ambition and whether it works, and includes scientific applications under Push
the Frontier: https://dev.meta.ai/events/global-hackathon (checked 2026-10-28).
The public page gives a November 4, 9 PM Eastern submission deadline; recheck
the participant portal for the definitive submission requirements and rubric.

## Three-minute demonstration

1. **0:00–0:25 — Problem.** A scroll CT contains many tightly packed sheets.
   A model can produce a smooth, plausible-looking trace on the wrong surface.
   Show the CT and explain the cost of manual surface review.
2. **0:25–1:05 — Model contribution.** SAM 3.1 tracks the original clicked
   region across the stack. Toggle previous and corrected geometry. Explain
   why broad-mask medial axes were unstable and how continuity-preserving
   CT refinement fixes the local extraction problem. Clearly label replay.
3. **1:05–1:35 — Evidence.** Show the actual surface-agreement panel alongside
   the ridge and guard proxies. Scrub slices and jump to the largest error.
   State the measured result from the JSON, including any missing-reference
   limitation. Do not substitute the 80.5% ridge score for surface accuracy.
4. **1:35–2:40 — Recognize, review, translate.** Open Read & Translate and
   identify the separate published Greek example. Click **Load recorded recognition**
   (a saved real image-only API response). Show its 95.4% normalized character error,
   0/5 exact lines: this is a failed automatic transcription baseline. Select image
   line 11; show its raw prediction alongside the published reference. Click **Use
   prediction for this line**; both translations become stale. Correct the Greek
   using the published reference and save, explicitly crediting that reference.
   Load the recorded translation to show English and Tamil. Export the reading:
   prediction, region, human changes and model provenance remain distinguishable.
   The live recognition button is optional; allow roughly 25 seconds and expect
   variable errors. Do not represent human-restored text as successful OCR.
5. **2:40–3:00 — Practical contribution.** The replay runs offline on CPU;
   model inference fits the existing laptop and runs separately. State the
   next experiment: reference-covered, held-out seeds and a timed comparison
   of review effort against manual tracing.

## Submission evidence to include

- Workbench recording with real CT/mesh overlays and readable measured scores.
- `reports/mesh_reference/validation.json` and coordinate audit.
- Regression-test result and reproducible commands from README.
- Explicit SAM 3.1 attribution and explanation of its role in the pipeline.
- Muse Spark 1.3 image-recognition and translation contributions, with recorded/live
  modes clearly identified. Include `reports/reading/recognition-validation.json`:
  published transcription, raw image prediction and human correction are separate.
- EduceLab data attribution, hardware, runtime, memory, and replay disclosure.

## Highest-value remaining work before submission

1. Establish target identity on reference-covered seeds; keep the original
   50-slice experiment in the report even if it performs poorly.
2. Measure held-out surface agreement and review time, including failures.
3. Turn exported correction proposals into re-tracing jobs with before/after
   provenance. Current workbench markers intentionally do not mutate results.
4. Record the demo. Add ink recovery only after a separately evaluated model
   produces a defensible result; fixture translation is not evidence.

Recognition implementation uses the official Meta image-input contract: https://dev.meta.ai/docs/image-understanding . No local OCR weights or GPU inference are required. Automatic line detection and recognition on unseen scrolls remain unvalidated.


## CPU OCR investigation update

Read & Translate now includes **Recognize locally (CPU)** and **Load recorded CPU OCR**.
The selected Tesseract ancient-Greek model uses grayscale denoising/contrast,
2× enlargement and raw-line mode. Two identical outputs measured 84.6% normalized
CER, 93.5% strict CER, 0/5 exact lines. It is faster (0.3–0.6 s) and has lower
normalized error than Spark, but worse strict error; neither is reliable OCR.
Measured OCR-process peak RAM: 56.45 MiB; no GPU. Existing Spark recordings remain.
Show both metrics and human review in the demo. No reference words enter inference.
Full experiment ledger and setup: `reports/reading/OCR-INVESTIGATION.md`.
