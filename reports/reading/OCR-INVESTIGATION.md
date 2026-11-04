# Greek recognition investigation

The CPU alternative is faster and modestly better on normalized CER, but neither recognizer produces a usable automatic transcription. Strict CER worsens. Do not pitch this as solved Greek OCR.

## Findings

- Visual inspection of the source and crop boundaries found the five intended rows present, without an obvious line offset. Line 14 is deliberately a partial-line excerpt. This is not an independent papyrological verification.
- The published source is a damaged, textured ink rendering. Dark strokes compete with papyrus fibers, gaps and stains; some strokes are missing. That is an image/model suitability problem that enhancement alone did not solve.
- Spark contact sheets worsened recognition. Enlarged individual enhanced crops were also worse than the earlier repeat. The proposed narrow-crop/downsampling cause is therefore not established; do not claim a provider-resizing bug.
- Tesseract SINGLE_LINE (PSM 7) rejected all raw and binary crops and four of five enhanced crops. RAW_LINE (PSM 13) bypassed enough line-processing heuristics to return text. Returned text is still largely wrong: avoiding rejection is not equivalent to recovering ink.
- Normalized errors remain high after removing accents, underdots, spacing and punctuation. Editorial marks are not the main explanation. Normalized CER retains letters from any script and question marks; it is not a confidence or percent-correct metric.

## Complete experiment ledger

| Experiment | Seconds | Strict CER | Normalized CER | Exact lines |
|---|---:|---:|---:|---:|
| recognition-validation.json | 23.281 | 90.22% | 95.38% | 0/5 |
| recognition-browser-validation.json | 26.127 | 88.04% | 87.69% | 0/5 |
| contact-1.json | 18.450 | 104.35% | 129.23% | 0/5 |
| gray_contact-1.json | 12.754 | 100.00% | 100.00% | 0/5 |
| gray_crops-1.json | 20.697 | 100.00% | 92.31% | 0/5 |
| tesseract_binary-1.json | 0.312 | 100.00% | 100.00% | 0/5 |
| tesseract_binary-rawline-1.json | 0.560 | 92.39% | 92.31% | 0/5 |
| tesseract_gray-1.json | 0.460 | 98.91% | 98.46% | 0/5 |
| tesseract_gray-rawline-1.json | 0.631 | 93.48% | 84.62% | 0/5 |
| tesseract_gray-rawline-2.json | 0.317 | 93.48% | 84.62% | 0/5 |
| tesseract_raw-1.json | 0.349 | 100.00% | 100.00% | 0/5 |
| tesseract_raw-rawline-1.json | 0.564 | 94.57% | 86.15% | 0/5 |
| tesseract_red-rawline-1.json | 0.315 | 91.30% | 92.31% | 0/5 |

Best normalized local result: grayscale, 3×3 median filter, 1% autocontrast, 2× Lanczos enlargement, 24-pixel white border, ancient-Greek tessdata_best LSTM, PSM 13. Both local runs return identical text. Error is 55 edits / 65 normalized reference letters (84.62%), versus 62/65 (95.38%) for the original Spark run and 57/65 (87.69%) for its earlier repeat. Strict error is 86/92 (93.48%), worse than Spark’s 83/92 and 81/92. All runs have zero exact lines.

The grayscale contact sheet and red-channel/binary alternatives were rejected. We keep all results and the original Spark recording. Selecting the CPU configuration on these five lines makes this an exploratory development comparison, not a held-out accuracy estimate. Repeating deterministic OCR verifies reproducibility, not generalization. No per-line best-answer assembly was used.

## No reference leakage

Only pixels and anonymous region IDs enter Spark requests; the prompt is unchanged. Tesseract receives only processed crop images and its general pretrained Greek model. There are no user-word lists, reference-derived dictionaries, post-correction prompts, fine-tuning, or generated missing ink. The source bundle supplies image coordinates, but its Greek text is not an inference input. Experiment scripts save raw predictions before scoring. Regression tests replace all reference lines with a sentinel and assert identical model-request messages/crop bytes. Same-origin endpoints reject client-supplied reference fields.

Published data can have been seen during pretraining, particularly by Spark. These tests prove absence of application-level reference input, not absence of pretraining exposure. The five crop rectangles were previously curated manually and are not blind layout detection.

## Hardware and deployment

The repeated CPU run measured 59,191,296 bytes (56.45 MiB) peak working set for the Python OCR process, including native Tesseract. Its inference/preprocessing time was 0.317 s; the first matching run took 0.631 s. This excludes browser RAM and interpreter startup. GPU inference is not used. The Greek model is 5,168,122 bytes, pinned by SHA-256 dfc9bda286cd9d8755b1832e5731a5425d1cf0803393a5fa6dee078466178cf1. The Windows tesserocr wheel is 23.1 MB. No CUDA/PyTorch dependency was changed.

Installed tesserocr 2.12.0 from a PyPI Windows wheel. The selected local model is an official general ancient-Greek OCR model, not a newly trained Vesuvius handwriting model. Model provenance: https://github.com/tesseract-ocr/tessdata_best . Language-model documentation: https://tesseract-ocr.github.io/tessdoc/Data-Files .

## Demo and reproduction

The workspace now has **Recognize locally (CPU)** and **Load recorded CPU OCR**, alongside the unchanged Spark options. Predictions use the same region validation, explicit acceptance, correction history and translation-staleness controls. Loading a new result does not replace a human correction.

Install optional CPU dependency: `.venv\Scripts\python.exe -m pip install -r requirements-ocr.txt`.
Fetch pinned weights: `.venv\Scripts\python.exe scripts/fetch_ocr_model.py`.
Rerun: `.venv\Scripts\python.exe scripts/recognition_experiment.py tesseract_gray --raw-line --run 3`.
Rebuild: `.venv\Scripts\python.exe scripts/build_workbench.py`.
Tests: 110 passed, including reference-input invariance, deterministic preprocessing, crop order/dimensions, model hash rejection, CPU endpoint routing and client-reference rejection. Existing correction/provenance tests remain passing.

Next meaningful accuracy step is a properly licensed model trained on comparable papyrus handwriting and evaluated on separate labeled lines. The present five-line set is too small for credible training or generalization claims.

Browser verification: the live CPU button returned in 0.6 seconds with 84.6% normalized / 93.5% strict CER, preserved the existing human reading, and showed no browser console errors. Screenshot: cpu-ocr-workspace.png. The active user draft was not reset.
