# Translation product slice — demo report (2026-10-28)

## What shipped
Product translation layer on top of the existing Spark agent:

- `src/agents/translate.py` — `translate()` / `translate_readings()`,
  6 targets (en/de/fr/es/hi/ta), fluent + gloss modes, per-item conf,
  offline glossary stubs (exact-phrase + word fallback, `?` for unknown).
- `src/agents/harness.py` — harness2-inspired Propose-Probe-Compose loop
  (invariant + domain prompt edits, no-ground-truth behavioral scoring,
  persistent lessons in `src/agents/lessons.jsonl`). 4 of harness2's 8
  components: system prompt, guardrails, memory, tool guidance.
  Source: https://github.com/google-research/harness2 (Apache 2.0, pattern only).
- `scripts/translate_demo.py` — CLI (`--text/--file --to --mode --improve`).
- `scripts/freeze_golden.py` + `scripts/demo.py --offline` — golden freeze
  (run.json + SHA256SUMS) and socket-guarded API-free replay.
- `demo/viewer.html` — reading + translation panel (lang/mode selectors,
  golden fixtures inline, file:// safe).
- `src/agents/fixtures/translation_golden.json` — frozen fixtures.

## Verified this session
- `pytest tests/ -q` → **20 passed** (11 pre-existing + 9 new).
- `freeze_golden.py` → `golden/` (8 frames + run.json + SHA256SUMS).
- `demo.py --offline` → GOLDEN OK (API-free), socket guard on.
- `translate_demo.py --to de --mode gloss` + `--improve` → correct output.
- Live Spark path not re-tested (quota); prior live check still stands.

## Demo script (60s, API-free)
1. `python scripts/demo.py --offline` — golden replay + translations.
2. Open `demo/viewer.html` — scrub slices, switch EN/DE/FR + fluent/gloss.
3. `UNROLL_OFFLINE=1 python scripts/translate_demo.py --text "καὶ ὁ μῦθος" --improve`
   — refinement trace + scores.
