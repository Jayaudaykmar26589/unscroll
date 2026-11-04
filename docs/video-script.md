# Unscroll — 3-minute demo video

Four beats: the stakes, a trace slipping and the fix, the agent checking its own
work, and what stands between a trace and readable letters. It says early and
plainly that no new text has been read.

**Built video:** `scripts/make_demo_video.py` (run with Anaconda's Python, which has
`websocket-client`) speaks this narration with the Windows voice Mark, records the hosted pages in
headless Chrome in time with it, and burns in captions and shot labels:
`demo/video/unscroll-demo.mp4` and `unscroll-demo.srt`. The table below is generated from it.

**To record by hand instead:** 1920×1080 with OBS (free), browser zoom 110%, one window, other tabs
closed. Most shots use the hosted pages, so no local server is needed:

- Story: https://jayaudaykmar26589.github.io/unscroll/viewer/story.html
- Evaluation: https://jayaudaykmar26589.github.io/unscroll/viewer/evaluate.html

Narration is about 462 words; the built video runs 2:55. Every number
below is in `reports/`. Rehearse once with a timer and cut words rather than
speeding up.

## Shot list and voice-over

| Time | Screen | Voice-over |
|---|---|---|
| 0:00–0:16 | Story ch. 1: hero, then the Crush slider | "In 79 AD, Vesuvius carbonized a library. The scrolls can't be opened, only scanned. To read one, you follow a single papyrus sheet through thousands of CT slices, where every sheet is crushed against its neighbours." |
| 0:16–0:35 | Story ch. 1: real CT slice, plus a credits card (built here / not ours) | "Everything in this video was built during the hackathon, except SAM 3.1, the Muse models, the 2023 Grand Prize ink model and the Vesuvius data, which I'll name when they appear. And to be clear: Unscroll has not read any new text. It measures how close an AI trace gets to being readable." |
| 0:35–0:56 | Story ch. 2: Play, with the earlier-attempt line switched on; distance meter outlined | "I click one sheet, and SAM 3.1 follows it through the slices like an object in a video. Pink is the sheet a person traced by hand. These are two early traces from day two: orange slides further and further off, and teal is closer but still off. The meter turns teal only within 40 microns, about half a hair." |
| 0:56–1:12 | Evaluation: the 20231210121321 before/after animation, full screen | "The cause was the first slice: it took the brightest path through SAM's mask, which can jump to the neighbouring sheet. Unscroll's fix walks outward from the click instead. On this click: 23 voxels off before, under 2 after." |
| 1:12–1:36 | Story ch. 4: Run the test, then Compare with a classic tracer | "We tested it on 74 clicks we never used while building the fix. The share of the trace within 40 microns of the hand-traced sheet doubles, from 25 to 50 percent: better on 50 clicks, worse on 19. Against a classic optical-flow tracer, the gain is smaller, four and a half points, and on our 15 development clicks it's a tie. We report both." |
| 1:36–1:59 | Agent log page (reports/agent/…T180912.jsonl) with the seam image; first seam outlined | "When a trace stops, a Muse Wonder agent takes over. It calls tools to re-seed from the last good slice, launch GPU runs on Kaggle and score the result. Then it checks its own seams: continuity numbers, plus Muse Spark looking at the image. Here it caught its first re-seed jumping to another sheet, threw out that run and the two built on it, and wrote down why." |
| 1:59–2:22 | Story ch. 3: drift slider to 5 voxels (Steady), then Wobbly | "Why does staying on the sheet matter? Ink here is invisible to the eye; the Grand Prize model finds it from the surface. If the surface is off by a steady five voxels, the ink survives: agreement 0.79. If it wobbles by five voxels from slice to slice, it's gone: 0.01. So the enemy is wobble, not distance." |
| 2:22–2:45 | Evaluation: smoothing panel, then the cost-of-reading chart and table | "So we smooth our own trace across slices. On a sheet with known text, agreement with the official ink map rises from 0.21 to 0.73, close to the 0.83 the official surface gets. But the maps show broad strokes, not letters, and fewer scholar clicks cover less of the sheet. That's the measured gap that's left." |
| 2:45–2:55 | Story ch. 5: flip three cards, then the end card with links | "Unscroll: click a sheet, see whether the AI stayed on it, and see how far it is from reading. The code, the data and every number are public." |

## Before recording

- [ ] Open both hosted pages once; hard-refresh (Ctrl+Shift+R) so the latest version shows
- [ ] Story chapter 2: Play runs; chapter 3 slider and Steady/Wobbly switch work; chapter 4 buttons colour the dots
- [ ] Evaluation: scroll to the smoothing panel and the cost-of-reading table in advance
- [ ] Open `reports/agent/20231210121321_20261031T180912.jsonl` in an editor with word wrap,
      and `reports/agent/seam_20231210121321_reseed_6159.png` in a second tab
- [ ] One full rehearsal with a timer
- [ ] Upload (YouTube unlisted or similar), check it plays signed out, paste the link into the form

## Hackathon rule: show only what was built here

The demo may only highlight what was built during the hackathon, and judges
must be able to tell what is new. So:
- The second shot names everything that is not ours, with a credits card on screen. Keep it.
- Whenever ink maps appear, they come from the 2023 Grand Prize model; what
  Unscroll adds is running it on Unscroll's own traces and the measurements around it.
- Do not feature the full-page ink map: it is the Grand Prize model's output on
  the official segment, not Unscroll's work.
- Pink reference layers are human traces released by the Vesuvius Challenge community.

## Do not say

- That Unscroll read new text or recovered letters itself.
- That the fix beats optical flow everywhere (it ties on the development clicks).
- That 0.73 means readable letters (the maps show broad strokes only).
- That fewer clicks score better: the cost-of-reading scores are not a ranking.
- That Muse Spark reduces expert workload (it does not yet).
- That the agent's extra slices are verified (beyond slice 6149 they are not).
