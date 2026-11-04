# Greek image recognition validation

Status: recognition workflow implemented; automatic transcription accuracy is insufficient.

## Real image-only runs

| Run | API wall time | Strict CER | Normalized CER | Exact normalized lines |
|---|---:|---:|---:|---:|
| Recorded baseline | 23.28 s | 90.22% | 95.38% | 0/5 |
| Browser live verification | 26.13 s | 88.04% | 87.69% | 0/5 |

The second run is a repeat with identical model, prompt and crops, not a tuned improvement. Both runs fail to produce a usable automatic reading. The demo ships the first run, not the better-looking repeat.

## Protocol and provenance

Five manually located crops of published PHerc. Paris. 4 column −8, lines 10–14 (last line excerpt). Original image SHA-256 is pinned in passage.json. Each PNG crop has its own digest. The request includes only crops, anonymous r1–r5 identifiers and a transcription instruction: no Greek reference, title, source URL, translation or publisher metadata. Reference text is loaded separately for post-response scoring. No OCR training, threshold tuning, or reference substitution is performed.

Official source: https://scrollprize.org/grandprize#what-does-the-scroll-say . Image-input interface: https://dev.meta.ai/docs/image-understanding .

CER is summed Levenshtein edit distance divided by summed reference length, computed per line to penalize line mapping errors. Strict scoring uses NFC text including spaces, punctuation, accents and uncertainty underdots. Normalized scoring removes combining marks, spaces and punctuation, lowercases, and folds lunate/final sigma to sigma; question marks remain errors. Scores can exceed 100% with insertions. Strict marks are editorial and not all visible in the image. Normalized scores do not measure recovery of uncertainty.

This is only 65 normalized reference letters across five lines from one public image. It is not a held-out benchmark; pretraining exposure cannot be excluded. Line boxes are manual, not model-detected. This is recognition on a published ink image, not ink detection from CT and not a new reading of our 50-slice trace.

## Hardware and execution

Local Pillow crops the fixed 936×1244 image (3.33 MiB of RGB pixel data); five crops are sent in one API request. No local model weights, torch import, CUDA allocation or GPU inference is needed by this feature. Total process RAM was not profiled. One shared server lock serializes recognition and translation; requests use a 45-second provider timeout with retries disabled and a 55-second browser timeout. Recorded recognition and manual corrections work offline. Live recognition needs internet and the configured server-side Meta key.

## Regression and browser validation

103 pytest tests passed, including the Node state suite. Recognition regressions cover crop dimensions, reference exclusion, shuffled ID mapping, duplicate/missing/malformed predictions, image/coordinate mismatch, Unicode scoring, unknown characters, insertion penalties, endpoint payload restrictions, explicit acceptance, immutable predictions, persistence, stale translations and provenance export.

Live browser request succeeded. Selecting image line 11 selected its prediction; applying it marked both translations stale. A human test correction using the published reference was saved and survived reload, while the original model text remained. Export was parsed and its distinct prediction, published text, correction and acceptance history were verified. This correction is a workflow test, not an independently validated papyrological reading.

## Reproduction

Build recorded assets: `.venv\Scripts\python.exe scripts/build_reading.py`.
Run another real API experiment: `.venv\Scripts\python.exe scripts/validate_recognition.py` (overwrites the recorded baseline; preserve prior reports first).
Tests: `.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider --basetemp=.review-tmp-recognition`.

Artifacts: recognition-validation.json, recognition-browser-validation.json, recognition-browser-export.json. Demo: docs/hackathon-demo.md.
