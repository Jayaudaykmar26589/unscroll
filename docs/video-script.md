# Unscroll — 3-minute demo video (final)

Record at 1920×1080 with OBS (free). Start the app with `Launch Unscroll.cmd`
(opens `http://127.0.0.1:8765/home.html`). Browser zoom 110%, dark theme, other
tabs closed. Speak slowly. Every number below is in `reports/`.

## Shot list and voice-over

| Time | Screen | Voice-over |
|---|---|---|
| 0:00–0:15 | Home page hero | "In 79 AD Vesuvius buried a library. Its scrolls are carbonized; touch one and it crumbles. The only way to read them is to trace each papyrus sheet through thousands of CT slices." |
| 0:15–0:30 | Surface workbench: click 🎙 Voice, say "next", then "worst" | "This is real CT of Scroll 1, and I'm reviewing it by voice with Muse Voice. Every bright band is a sheet, crushed against its neighbours. A tracer that slips one layer gives you the wrong text, and nobody can tell." |
| 0:30–0:45 | Home: "How it works" strip | "Unscroll turns one click into a trace. SAM 3.1 follows the clicked sheet through the slices like an object in a video, and then we measure it against layers people traced by hand." |
| 0:45–1:15 | Evaluation: top tiles, then the before/after animation for 20231210121321 | "Fifteen clicks on human-traced layers. At first the trace stayed within forty microns only 37% of the time. Watch the red line jump to the brighter neighbour. We fixed how the first slice chooses its sheet: 55%, and ten of fifteen seeds on the right layer instead of six." |
| 1:15–1:30 | Evaluation: held-out tiles | "On fifteen new clicks we never developed on, it still rises, from 36 to 49%, at the cost of stopping a little more often. We show that too." |
| 1:30–1:55 | Agent log (`reports/agent/…jsonl` in an editor, scrolled) then the Home stat | "When tracking stops, a Muse Wonder agent takes over: it re-seeds from the last good slice and launches GPU runs. Then it checks its own work: at each seam it measures continuity and asks Muse Spark to look. It found its first re-seed had jumped to another sheet, threw out that run and the two built on it, and explained why." |
| 1:55–2:20 | Evaluation: "Reading letters" panels 1–4 | "On a sheet with known Greek text, the 2023 Grand Prize model reads the letters, and our pipeline reproduces them from the raw CT. Our own traces aren't close enough yet: ink sits on the surface and needs two-voxel accuracy; we're at seven with a scholar's help. That gap is now measured: our own 3D unrolling reads the letters on the right surface, so the trace is what's left to fix." |
| 2:20–2:35 | Evaluation: "Which traces to check first" chart, hover at 73% | "Reviewers can't check everything. Unscroll ranks traces by risk without any reference. On clicks we never developed on, checking the riskiest 73% catches every wrong layer; Muse Spark alone needed 90%. Our first rule failed that test, and we show it." |
| 2:35–2:50 | Read & Translate: "Load recorded Spark draft", edit one word, show the stale translation flag | "For a published passage, Muse Spark drafts readings and English and Tamil translations, and a person accepts or corrects every line." |
| 2:50–3:00 | Home page hero | "Unscroll. Click a layer. Know whether the AI stayed on it." |

## Before recording

- [ ] Rebuild data: `python scripts/build_evaluation.py`
- [ ] Start the app; all four pages load (the server sends no-cache headers)
- [ ] Read & Translate: "Load recorded Spark draft" works offline
- [ ] Voice: on the Surface workbench, 🎙 Voice → "next" → click again; allow the microphone once
- [ ] Open the agent log `reports/agent/20231210121321_20261030T113950.jsonl` (first
      two re-seeds and the "curve length jumped" check; the final run is in
      `…T120843.jsonl`) in an editor with word wrap
- [ ] One full rehearsal with a timer; cut words rather than speeding up

## Hackathon rule: show only what was built here

The demo may only highlight what was built during the hackathon, and judges
must be able to tell what is new. So:
- Say once, early: "Everything you see was built during the hackathon, except
  SAM 3.1, the Muse models, the 2023 Grand Prize ink model and the Vesuvius data,
  which I'll name when they appear."
- Whenever ink maps appear, say they come from the Grand Prize model; what
  Unscroll adds is running it and the measurements around it.
- Do not feature the full-page ink map: it is the Grand Prize model's output on
  the official segment, not Unscroll's work.

## Do not say

- That Unscroll read new text or recovered letters itself.
- That 55% is a general accuracy (quote the held-out 49% alongside it).
- That Muse Spark reduces expert workload (it does not yet).
- That the agent's extra slices are verified (beyond slice 6149 they are not).
