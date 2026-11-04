# >>> RESUME HERE (written 2026-11-02) — "resume" means: do the NEXT steps below, in order <<<

Deadline: submissions due Wed Nov 4, 9:00 PM ET (Thu Nov 5, 7:30 AM IST); target submit Tue Nov 3.
Judging rubric (participant guide): Impact 30%, Use of Meta AI tooling 25%, Technical depth 20%,
Demo 25%. Required: PUBLIC repo + 3-minute video showing only what was built here.
Current self-rating: ~7.3/10 now, ~7.8 with a strong video. Biggest gap: "Unscroll doesn't read new text".

## UPDATE 2026-11-04 (latest): steps 2 and most of 4 DONE
- Public repo: https://github.com/Jayaudaykmar26589/unscroll (branch main). Commits carry NO
  Claude Co-Authored-By line (user asked to remove Claude as contributor; history rewritten once).
- GitHub Pages on (main, root, .nojekyll): https://jayaudaykmar26589.github.io/unscroll/viewer/story.html
  and .../viewer/home.html. Linked at the top of README and docs/submission.md "Links".
- Step 2 messaging done: README + submission open with "has not read new text"; dev-set tie with
  optical flow stated; triage claim gives set size (30 traces, 15 wrong); 4.8 GB labelled an
  estimate; tests/test_attention_patch.py proves chunked attention bit-identical on CPU fp32.
  .gitignore no longer drops reports/*.json. Suite: 149 passed, 1 skipped.
- LEFT: video (step 3, script rewrite), Vesuvius permission screenshot into docs/ (user has it),
  submission form, then rotate tokens; ask about _archive/.

## Standing rules (from the user; keep)
- Heavy compute on Kaggle GPU ("Work on the Kaggle. Avoid downloading"); token in ~/.kaggle/access_token.
- Never use -contributor Meta model variants. No Meta logos/branding. No credentials via browser.
- Only copy content whose licence allows it (Scroll 1 data CC BY-NC with credit; others link only).
- Commit/push only when the user asks. Never add a Claude Co-Authored-By line.
- Moving scratch files into _archive/ still needs the user's approval.
- Be honest: no claim of new text; report negative results.

## State at this point (all verified; 144 tests pass: `.venv\Scripts\python.exe -m pytest -q tests`)
- 74 new clicks: fix 25% -> 50% within 40 um (p=0.0002); +4.5 pts vs optical flow (2.2–7.3).
  reports/seeded_eval/compare.json (scripts/compare_methods.py, baseline_flow.py).
- Drift: steady 40 um offset 0.80 vs 40 um wobble 0.01 (reports/letters/shift/).
- Smoothing our assisted known-text trace (click every 40 slices): ink agreement 0.21 (σ2) ->
  0.44 (σ4) -> 0.63 (σ8); official surface 0.83 (reports/letters/assisted_smooth/).
- 2.4 um probe: 32/32 runs, fixed tracer median 27 um, 71% within 40 um (reports/probe24/).
- Reference check vs 2026 re-trace (scripts/reference_check.py); tifxyz export (scripts/export_tifxyz.py).
- Home + Evaluation rewritten for judges (4 headline results, plain words, "More detail" fold).
- MIT LICENSE; SAM fixes as patches/sam3-unscroll.patch; .gitignore ready (dry run ~300 MB, no secrets).
- Local server: `.venv\Scripts\python.exe scripts\workbench.py --port 8765` -> http://127.0.0.1:8765/home.html

## NEXT (do in order)

### 1. "Cost of reading" experiment on Kaggle (answers "doesn't read new text")
PROGRESS 2026-11-02: 1a DONE (naming fixed, `--tag` gives its own output folder AND its own notebook
`unscroll-letters-<tag>`). A3 launched (`--sigmas 2 4 8 16 --tag one` -> reports/letters/unassisted_smooth_one).
A1 (`--assist 40 --sigmas 12 16 --tag s1216` -> assisted_smooth_s1216) and A2 (`--assist 80 --sigmas
2 4 8 16 --tag a80` -> assisted80_smooth_a80) queue behind Kaggle's 2-GPU-session limit (retry loop).
1d chart DONE: build_evaluation.py `reading_cost` + evaluate.html #costsection read those folders
automatically.
A3 DONE + fetched: one click, no help -> trace median 125 µm, 19% within 40 µm, narrower patch
(417 cols vs 786). Agreement σ2/4/8/16 = 0.45/0.48/0.40/0.42 (vs 10-click 0.21/0.43/0.63). Images show
blobs, NO letters. The 0.45 is averaged over a different, smaller area: never call it "one click beats
ten". Page table now has a "Trace on the sheet" column plus that caveat. A1/A2 pushed 13:54/14:29.
2026-11-03: NEW PAGE viewer/story.html ("Story", linked from every nav + a home card + README): a
5-chapter explainer for a 20-year-old, all on real data: crush-a-scroll spiral -> real slice; scrub
50 real slices (data.js frames, 28 Oct trace) with a live distance meter; ink drift slider
(eval-data shift, steady vs wobbly); 74-dot "run the test" (compare.new_clicks); true/false flip
cards. Verified in the browser at 1280 px, no console errors. Mention it in the video/submission.
2026-11-03 12:30 KAGGLE ACCOUNT SWITCHED: ~/.kaggle/access_token is now jayachandiran2605 (user's
choice: "Use the new one"; old token kept as access_token.jaya260589.bak, do not touch). The A1/A2
runs under jaya260589 are unreachable; weights+code datasets are being re-uploaded under the new
account (push-data), then A1 and A2 are relaunched there. kaggle_run.py reads the username from the
API, so slugs are now jayachandiran2605/unscroll-letters-{s1216,a80}.
2026-11-03 20:50: still blocked ("Maximum batch GPU session count of 2": the new account's own two
notebooks). User said "Keep waiting": retry loop running (200 tries x 5 min, s1216 then a80).
DONE meanwhile: README cost-of-reading bullet, submission evidence rows (one-click run, Story page),
tests (146 passed, 1 skipped), test count in submission.md. If A1/A2 never run before the deadline,
ship with the one-click + ten-click rows; the page and README already say more runs are queued.
2026-11-03 23:50: A1 DONE + fetched (assisted_smooth_s1216): same trace (identical distances),
σ12 0.725, σ16 0.726, snap 0 both; images look like σ8 (one broad stroke, no letters) -> PLATEAU
~0.1 below official 0.83. build_evaluation folds it into the smoothing section (Home/Eval now say
0.21 -> 0.73); README, submission row, Story flip card, new test (147 passed, 1 skipped) updated.
A2 (a80) pushed 23:36, RUNNING; watcher running. Independent review done (6.8/10 today): top
actions are public repo, video, headline chart + dev-set tie with flow stated, permission
evidence in docs/, back or soften "bit-identical"/"4.8 GB". Also: .gitignore `reports/*.json`.
2026-11-04 01:30: STEP 1 COMPLETE. A2 fetched (assisted80_smooth_a80): 5 clicks, covers only
165/400 slices, 56 µm median, 29% on the sheet, agreement 0.58/0.62/0.60/0.54 (σ2/4/8/16), maps
show no strokes. Cost table now has "Slices covered"; caveat says the scores are NOT a ranking.
README, submission row, test (148 passed, 1 skipped). No Kaggle jobs or loops running. (The old
retry loop pushed duplicate v2s of s1216/a80 at 00:08/00:23; harmless, identical code.)
NEXT is step 2 (messaging) / step 3 (video) / step 4 (ship), plus the review's fixes.
(Historical) check `kaggle kernels status jayachandiran2605/unscroll-letters-a80`,
push any that never started, `fetch --letters --tag <tag>` each, rebuild, inspect PNGs, then README/tests.
Goal: one chart — human clicks per 300 slices (x) vs ink agreement of Unscroll's OWN trace with the
official ink map on the known-text sheet (y), best smoothing per point, official surface 0.83 as the
ceiling. Claim stays "measured path towards reading", never "new text".
1a. Fix output naming in scripts/letters_kaggle.py first (bugs found 2026-11-02):
    - `OUT = OUT / "assisted"` ignores N: use `OUT / f"assisted{a.assist_every}"` for N != 40
      (keep "assisted" for 40 so existing results stay valid).
    - `OUT.parent / f"{OUT.name}_smooth"` with no --assist writes reports/letters_smooth, OUTSIDE the
      synced reports/letters folder: make it `OUT / "smooth"` when OUT is the letters root.
    - add `--tag` so a new sigma set never overwrites assisted_smooth/ (e.g. OUT name + f"_{tag}").
1b. Runs (one kernel slug runs one version at a time: run sequentially, or add a slug suffix in
    scripts/kaggle_run.py `_job` for letters, e.g. unscroll-letters-<tag>):
    - A1 stronger smoothing, same trace: `--letters --assist 40 --sigmas 12 16 --tag s1216`  (~50 min)
    - A2 half the clicks: `--letters --assist 80 --sigmas 2 4 8 16`                          (~40 min)
    - A3 one click only:  `--letters --sigmas 2 4 8 16`                                       (~100 min)
    Sequence each: `kaggle_run.py stage <flags>`; `push-data <flags> --only code`; wait for
    `kaggle datasets status jaya260589/unscroll-code` = ready; `push-kernel <flags>`; watch
    `status`; then `fetch --letters`.
1c. Look at the ink PNGs before trusting numbers (blur can inflate agreement; snap offset varies).
1d. Add to Evaluation page (scripts/build_evaluation.py bundle + viewer/evaluate.html section after
    #smoothsection) a "Cost of reading" chart with hover + table view; add a Home/summary line
    only if the one-click result is meaningful. Update README results + docs/submission.md
    evidence table; add a test in tests/test_compare.py; run the suite.

### 2. Messaging for "no new text" (cheap, do even if step 1 is weak)
- First 20 s of the video + submission description: "Reading needs a trace that stays on the right
  sheet, steadily, for 300+ slices. Unscroll measures and closes that gap — here is how far it got."
- Keep Predict / Tamil to one line; full-page ink map is the Grand Prize model's output (label it).

### 3. Video script (docs/video-script.md) — rewrite around 4 beats, label what is not ours
stakes -> watch a trace slip + the fix -> 50% vs 25% on 74 new clicks and beats optical flow ->
Muse Wonder agent catches its own wrong re-seed (Best Agentic Use-Case) -> wobble vs ink and
smoothing (0.21 -> 0.63, plus cost-of-reading if done) -> close. Keep the "Hackathon rule" block.

### 4. Ship
- Ask the user for an empty PUBLIC GitHub repo URL; then `git add -A` (dry run first with
  `git add -n .`), commit (no Claude Co-Authored-By line), push. external/sam3 stays out.
- Submission form (docs/submission.md has paste-ready text): "Built during the hackathon" paragraph,
  Best Agentic Use-Case, Vesuvius permission screenshot, video link; check links signed out.
- After submitting: user rotates Kaggle token + API key; ask about _archive/; optional X post.

# <<< end of RESUME HERE >>>

# Update 2026-11-01 — more scrolls + full page
Read & Translate: picker over all published Scroll 1 columns (−8 whole, −7, −2, −1;
33 lines, CC BY-NC 4.0, data/reading/passages via scripts/build_passages.py; Spark
translation drafts marked unreviewed). Spark line reading of those columns: 79%
character error, 0/33 exact (scripts/read_columns.py). Other scrolls (PHerc. 172,
139, 1667): links only (images not licensed for reuse). Full page: GP ink model over
whole segment 20230827161847 on Kaggle (reports/ink_gp_full, 8.4 min); view rotated
270 + mirrored + 6 deg level; Spark reads 4 lines, unscored. Tests 135/1 skip.

# Update 2026-11-01 — triage (judge review: impact needs a workload number)
scripts/triage.py -> reports/triage/triage.json: reference-free signals (click offset,
ridge contrast, first-slice method disagreement, coverage, length jump) on 60 seeded
traces. v1 a-priori rule: dev AUC 0.84, held-out 0.52 (failed, reported). v2 = mean rank
of dev-selected signals (dev AUC > 0.6): dev 0.88, held-out 0.66; held-out review 73%
catches all wrong-layer traces, 63% catches 80% (Spark reviewer 90%). Ridge contrast is
the steadiest signal (0.70/0.78). Per-slice signals do not track error within a trace
(Spearman -0.17..0.08), so NO per-slice ribbon. Evaluation page: "Which traces to check
first" chart + tiles + table; seed table "Check order" column (in-sample). Tests 129/1 skip.
Video 2:20-2:35 now shows triage.

# Update 2026-10-31 ~21:30 — Predict page (user: complete a sentence from a fragment, accurately)
viewer/predict.html + POST /api/predict/complete. src/agents/complete.py: word-level
lookup in data/predict/corpus.json (5,998 sentences, Diogenes Laertius Greek from
Perseus CC BY-SA, built by scripts/build_corpus.py, + PHerc. Paris. 4 col. -8), tolerant
of accents/typos/cut edge words/Latin letters; Muse Spark conjecture only if not
attested (position start/middle/end enforced, must contain fragment). Eval
(scripts/eval_complete.py -> reports/predict/eval.json): lookup 99.3% of 900 fragments,
890/890 "attested" correct; Spark alone 1/40 exact, 16% missing words. Tests
tests/test_complete.py (9).

# Update 2026-10-31 ~20:30 — three features (user: "proceed with the suggestions, Kaggle GPU")
1. 3D-normal unrolling: scripts/unroll3d.py (grid from traced curves, smoothed,
   normals from 3D tangents, trilinear layer sampling, reference-free depth/side
   "snap" by ink-map std). Synthetic: ink corr 0.948 vs 0.568 for 2D normals.
   letters_kaggle.py adds control_own_normals + ours3d; run on Kaggle
   (unscroll-letters, assisted, 18:06-19:00) -> reports/letters/assisted/.
   RESULT: control with OUR 3D normals ink corr 0.82 (letters visible; normals
   agree 0.996). Our assisted trace + 3D unroll: corr 0.21, snap -8 vx (edge),
   no letters. Gap = trace accuracy. Pages + docs updated.
2. Agent self-verification: grow_agent.py tools review_seam (numbers + Muse
   Spark verdict, images reports/agent/seam_*.png) and exclude_run
   (reports/flatten/excluded_<seed>.json). Review session
   reports/agent/20231210121321_20261031T180912.jsonl excluded reseed_6159
   (181 vx jump, length x4.4) and descendants -> 58 verified slices. Docs no
   longer say 58->97.
3. Muse Voice: src/agents/voice.py, /api/voice/transcribe in workbench.py
   (same-origin, WAV checks), viewer/voice.js push-to-talk on index.html.
   Parser 12/12, API round trip 200. User should try the real microphone.
README / submission / video script mention voice. Tests 116 passed, 1 skipped.

# Update 2026-10-31 ~19:00 — polish
Launch test: all 4 pages load, 0 broken images/links, 0 console errors; offline
Spark replay works. Server now sends Cache-Control: no-store on everything.
README (results, Muse Wonder, letters, Compute), Home (Wonder in the flow,
"Reading letters" card, 4-column cards), evaluate.html #section links work.
docs/submission.md = portal text, evidence table, licences, user checklist.
docs/video-script.md = final 3-minute shot list. Secrets check: no keys outside
.env / ~/.kaggle. Tests 116 passed, 1 skipped. Pending: user OK to move scratch
into _archive/ (list in chat); user records video and submits Tue Nov 3.

# Update 2026-10-31 ~18:00 — assisted letter run + product
Assisted (10 simulated clicks, 1/40 slices): 278 slices, 32% within 5 vx, median
7.0 vx (unassisted 18%, 15.7) — still no letters on our surface. Control reads
letters. Explanation + next research step in docs/ink-research.md. Evaluation
page has "Reading letters" (official / control / ours x2) built from
reports/letters{,/assisted}. Experiments stop here; remaining: polish, docs,
launch test, submission package, video (user records Tue Nov 3).

# Update 2026-10-30 ~14:40 — letter demo v3 result
Notebook unscroll-letters v3 (103.5 min): window rows 3650-4650 x cols 3000-3400
of segment 20230827161847 (z 10656-11068). Calibration r=0.977. CONTROL (official
surface resampled from Zarr) ink corr 0.83 with official -> read path validated.
OURS: 362/~400 slices traced with 6 re-seeds, but only 18% within 5 vx of the
official surface (median 15.7 vx): mostly a neighbouring sheet; ink map on our
surface is saturated blobs, not letters (do not claim a reading). Images:
reports/letters/view_official_control.png, view_ours.png. Proposed next:
"assisted" mode (one click per ~40 slices from the official surface, labelled
as simulated scholar input) OR stop and polish.

# Update 2026-10-30 ~12:20
- Held-out DONE: walk 36.2% -> 48.5% within 5 vx (median), 8.1 vs 14.7 vx,
  but 234 vs 285 slices traced. In RESULTS.md, README, Evaluation, Home, video.
- Wonder agent DONE: 58 -> 97/100 slices via 3 re-seeds (logs reports/agent/);
  re-seeded curves are much longer (sheet_..._merged is 1275 cols wide):
  possibly another sheet beyond 6149, unverified (no mesh there). Say so.
- Letter-scale demo RUNNING: `kaggle_run.py --letters` (notebook unscroll-letters,
  scripts/letters_kaggle.py): segment 20230827161847 .ppm + OME-Zarr chunks
  (volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr, uint8 128^3), control
  = official surface sampled from Zarr, ours = Unscroll trace + re-seeds, GP ink.
  Results -> reports/letters/ (summary.json, official_/control_/ours_ ink+ct PNGs).

# Update 2026-10-30 ~12:00
- Ink: switched to the scroll-trained Grand Prize model (docs/ink-research.md).
  `kaggle_run.py --ink-gp` reproduces legible letters on segment 20230827161847
  (reports/ink_gp/val_pred.png); our 0.47 mm strip shows none (too small).
  Fragment training (`--ink`) is parked as a benchmark.
- Held-out run finished on Kaggle 11:36; local copy was wiped by a concurrent
  fetch (fixed: per-notebook kaggle/output/<slug>/). Re-fetch was running.
- Wonder agent running (log reports/agent/20231210121321_20261030T113950.jsonl):
  re-seed at 6159 gained 6161-6162 only; second re-seed at 6162 in progress.
- NEXT big item (proposed): letter-scale demo — pick a released Scroll 1 segment
  with known ink, stream the needed slices on Kaggle (no local transfer), trace it
  with Unscroll from a click on that sheet, unroll ±32 layers, run the GP model.

# FINAL PRODUCT PLAN in progress — see docs/final-product-plan.md (2026-10-30)

Done today (afternoon): node test skips with reason (suite 116 passed, 1 skipped);
crop jobs sync tracker run logs; 100-slice stop diagnosed as SAM-side (motion
verified 0.947 at 6159->6160); Muse Wonder patch-growing agent
`scripts/grow_agent.py` (tool calling verified; tools tested with --status);
Spark QA notes `scripts/qa_report.py` (15/15 pass the number check; on the
Evaluation page); historical banner on the surface workbench; ink training
script `scripts/ink_kaggle.py` (2.5D ResNet-34 U-Net, train 1(+2) / val 3,
F0.5/F1/AUC, applied to `reports/flatten/surface65_20231210121321`) wired as
`kaggle_run.py --ink` (CPU-tested). Kaggle queue: held-out run (seeded v6) ->
Wonder agent auto-starts after it (background task; log reports/agent/).
BLOCKED on user: accept https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/rules
Then: `kaggle_run.py stage --ink`, `push-data --only code`, `push-kernel --ink`,
`status --ink`, `fetch --ink` (reports/ink_kaggle/{f1,f12}/metrics.json).

# START HERE (saved 2026-10-29 ~14:45 IST, nothing running)

Deadline: Wed Nov 4, 9 PM ET = Thu Nov 5, 7:30 AM IST. Target: submit Tue Nov 3.
State: real-data evaluation done for both first-slice modes (DP baseline and
walk); results in `reports/seeded_eval/RESULTS.md`, scores in `score.json` /
`score_walk.json`, curves in `curves/` / `curves_walk/` (all local).

PROGRESS 2026-10-30: steps 1 and 2 DONE.
- Step 1 (Spark reviewer): `scripts/spark_review.py`, results in
  `reports/spark_review/RESULTS.md`. Spark never accepted a wrong-layer trace
  (0/26) but flagged 13/16 and 14/16 good traces (review load 90–93%), so it is
  a safe second opinion, NOT a workload saving. Do not claim otherwise.
- Step 2 (before/after): `scripts/before_after.py` -> `reports/before_after/*.gif`;
  new workbench page `viewer/evaluate.html` built by `scripts/build_evaluation.py`
  (seeded results, animations, Spark results), linked from index/read pages.
  Serve with `.venv\Scripts\python.exe scripts/workbench.py` (or the
  `unscroll-workbench` entry in `.claude/launch.json`), open /evaluate.html.
- Correction: seeds on layer (whole-trace median <= 6 vx) are 6 -> 10, not 8 -> 10.
Step 3 IN PROGRESS (2026-10-30):
- 4 cached CT slices were truncated (6151, 6152, 6154, 6155: an early-closing
  server stream saved as complete). `scripts/fetch_scroll.py` now verifies the
  byte count; re-fetched, all 100 slices = 127,722,864 bytes.
- `scripts/make_crop.py`: 2560 px crop of slices 6100-6199 around seed
  20231210121321 (offset 2213,1146; 1.3 GB) -> `data/cache/crop_20231210121321`.
- `scripts/kaggle_run.py --crop ID`: crop job (dataset unscroll-crop-ID,
  notebook unscroll-crop-trace, 100-slice walk trace); `fetch --crop ID` writes
  volume-coordinate curves to `reports/flatten/curves_ID_kaggle`.
- `scripts/flatten_trace.py`: 1-voxel arc-length flattening anchored at the
  click; test on 20-slice walk -> `reports/flatten/sheet_1210_w20` (320 x 20 vx).
- Evaluation page has an "Unrolled papyrus surface" section (build_evaluation.py
  bundles every reports/flatten/sheet_*).
Also done 2026-10-30 (demo push):
- Held-out seeds: `scripts/seeded_eval.py heldout` -> `seeds_heldout.json` (15
  new clicks ~700 vx from the main ones, same meshes). Kaggle seeded job now
  runs `trace --seed-set heldout --first-frame dp walk`; outputs
  `curves_heldout/`, `curves_walk_heldout/`; score with
  `seeded_eval.py score --seed-set heldout --first-frame dp|walk`.
  NOT yet run (queue after the crop job: stage, push-data --only code, push-kernel).
- `viewer/home.html` story landing page; launcher opens /home.html; Home links
  on all pages. README top rewritten (results, Meta models, how to open).
- `docs/video-script.md`: 3-minute shot list + voice-over + do-not-say list.
Step 3 DONE: 100-slice walk trace of 20231210121321 on Kaggle (crop job):
traced 59/100 slices (lost after 6158); vs mesh on 6100-6149: 87.3% within
5 vx, median 1.7 vx. Unrolled 320 x 60 vx (2.53 x 0.47 mm), crossed papyrus
fibres, on the Evaluation page. Section added to seeded_eval/RESULTS.md.
Held-out run IN PROGRESS: Kaggle seeded notebook v6 (pushed ~10:07 IST
2026-10-30), 15 held-out seeds x {dp, walk}. When it ends:
`kaggle_run.py fetch`, then `seeded_eval.py score --seed-set heldout
--first-frame dp` and `... walk`, add a held-out section to RESULTS.md, README
and the Evaluation page (build_evaluation.py), and quote it in the video.

WORKING PLAN (agreed with the user 2026-10-29; on "resume", start at step 1).
Goal: lift the weakest rubric criteria (Tooling 25%, Demo 25%). Self-estimate
now ~6.4/10 weighted; target ~7.5-8 with steps 1-3.
1. (Fri Oct 30) **Muse Spark trace reviewer** — Meta model inside the core loop.
   Render each seed's per-slice overlay (CT crop + trace; NO mesh shown to the
   model), ask Muse Spark 1.3 (structured output) "does the trace follow one
   continuous sheet, or cross/jump to a neighbour?" with a confidence. Score its
   verdicts against the mesh-based truth (on layer = median dist <= 6 vx) for
   both DP and walk runs: precision/recall of "wrong layer" flags. Impact
   number: "auto-accepts X% of seeds correctly; experts review Y%". API key in
   `.env` (META_MODEL_API_KEY); existing client in `src/agents/spark.py`.
   Never use `-contributor` model variants. Record raw responses for replay.
2. (Fri Oct 30) **Before/after view** — split-screen animation over 20 slices on
   real CT: DP trace (red) jumping layers vs walk trace staying on the released
   mesh (green), for 20231210121321 / 20230701020044 / 20230627170800, plus the
   one regression 20230702185753 shown honestly. Put it in the workbench.
3. (Sat Oct 31) **Flattened real papyrus surface** from the best walk trace
   (20231106155351, 96% within 5 vx; or 20231210121321): flatten with
   `src/flatten/`, show it in the workbench. Run ink only as an extra; do not
   claim letters unless clearly legible.
4. (Sat/Sun, half day) **Held-out check** of `--first-frame walk`: new seeds
   from other meshes/slices (e.g. slices 6120-6149 already cached, or seeds at
   other points of the same meshes), scored the same way. Report even if worse.
5. (Sun Nov 1) Feature freeze; one-command launch test (`Launch Unscroll.cmd`).
6. (Mon Nov 2) README, RESULTS, judge walkthrough (`docs/hackathon-demo.md`),
   architecture diagram, cached demo assets; impact sentence from step 1.
7. (Tue Nov 3) 3-minute video script + shot list for the user to record; user
   submits on the Cerebral Valley portal. Wed Nov 4 is buffer only.
Avoid: chasing new letters with the weak ink model (F1 0.54); over-claiming the
55% figure (not held out until step 4).

Commands (run from D:\Meta\unscroll):
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider --basetemp=.test-temp-local   # 116 pass + node failure
.\.venv\Scripts\python.exe scripts/seeded_eval.py score --first-frame walk                       # re-score walk run
.\.venv\Scripts\python.exe scripts/kaggle_run.py stage; .\.venv\Scripts\python.exe scripts/kaggle_run.py push-data --only code; .\.venv\Scripts\python.exe scripts/kaggle_run.py push-kernel   # new GPU run (TRACE_ARGS in kaggle_run.py)
.\.venv\Scripts\python.exe scripts/kaggle_run.py status; .\.venv\Scripts\python.exe scripts/kaggle_run.py fetch
```
Kaggle: user `jaya260589`, token in `~/.kaggle/access_token` (user kept the
token that was pasted in chat; they may regenerate it). Private datasets
`unscroll-{scroll1-slices,sam31-weights,code}`, notebook `unscroll-seeded-trace`
(v5 = walk). About 3 of ~30 weekly GPU hours used.
Local GPU (8 GB) works again after the attention fix, but is often shared with
the user's other jobs (D:\ML Competition) — prefer Kaggle for batch runs.
Vesuvius team OK'd data use (user report); keep their reply with the submission.

# RESULT: first-slice walk (2026-10-29, Kaggle v5)

`--first-frame walk` (local ridge walk from the click, `src/trace/image_ridge.py
_walk`): within 5 vx median 36.7% -> 55.1%, median per-seed mean distance
12.0 -> 7.5 vx, 10/15 seeds on layer (was 6; whole-trace median <= 6 vx); 12 improve, 20230702185753
regresses (47% -> 2%). Not held out. Re-run: `python scripts/seeded_eval.py
score --first-frame walk`. Tests: 116 passed + `node` failure. Plan: Fri
workbench with seeded results, Sat flatten+ink best patch, Sun freeze, Mon docs,
Tue Nov 3 video + submit.

# RESULT: seeded real-data evaluation done (2026-10-29)

`reports/seeded_eval/RESULTS.md`: 15 seeds x 20 slices on Kaggle T4 (282/300
traced). 8/15 seeds stay on the clicked mesh's layer (median 3–7 vx); 7/15 sit
on a neighbouring layer from slice 6100 onward (10–39 vx) with little z drift.
Within 5 vx: median 36.7%. Next: enforce first-slice curve within 2 vx of the
click (re-prompt otherwise) and re-run via `scripts/kaggle_run.py`.

# Kaggle run + attention memory fix (2026-10-29)

Seeded tracing now runs on Kaggle (T4 16 GB): `scripts/kaggle_run.py
{stage,push-data,push-kernel,status,fetch}`; private datasets
jaya260589/unscroll-{scroll1-slices,sam31-weights,code}, notebook
jaya260589/unscroll-seeded-trace. Code ships as `unscroll_code.tgz.bin`
because Kaggle auto-extracts .tar.gz (and nested .gz, breaking SAM's BPE file).

Root cause of the slice-6106 stall: without flash attention the vendored
SAM 3 decoder used math SDPA, which materializes the full score matrix; once
memory attention spans several past frames one call needs ~4.8 GB (Kaggle T4:
OOM at z=6106 for every seed; 8 GB laptop: silent spill to system RAM).
`external/sam3/sam3/model/decoder.py` now attends in query chunks
(`_unscroll_sdpa`, UNSCROLL-tagged): bit-identical to full SDPA on CPU fp32 and
CUDA bf16/fp16 (max abs diff 0.0, 173 chunks). Suite: 114 passed + `node` failure.

# Data license (2026-10-29)

User reports asking the Vesuvius Challenge team on Discord whether a Meta
hackathon entry may use the CC BY-NC Scroll 1 data, and that they said it's
okay. Keep a link or screenshot of that reply with the submission files; still
cite EduceLab-Scrolls / Parsons et al. (2023) and the scrollprize.org data terms.

# PAUSED 2026-10-28 ~19:05 — resume here

Seeded real-data run paused by the user before any seed finished (0/15 traced).
Resume (needs the GPU to itself; other GPU jobs made SAM crawl):
```powershell
.\.venv\Scripts\python.exe scripts/seeded_eval.py trace   # restarts unfinished seeds cleanly
.\.venv\Scripts\python.exe scripts/seeded_eval.py score   # -> reports/seeded_eval/score.json
```
Speed findings (py-spy): (1) `src/trace/thin.py` skeletonize now thins only the
mask bounding box — identical output on 40 random masks, 7x faster on a large
blob; (2) SAM 3.1's `_prepare_backbone_feats` runs the unused text-grounding
detector on each new frame chunk (`run_backbone_and_detection`), causing
multi-minute stalls — skipping it would need a vendored change, not done yet;
(3) `--trim-memory` (opt-in, off by default) bounds tracker memory. Suite: 114
passed + 1 pre-existing `node` failure.

# Checkpoint: reference-seeded real-data evaluation (2026-10-28)

`scripts/seeded_eval.py` clicks ON released Scroll 1 meshes (identity transform,
volume 20230205180739), traces with the unchanged `src.trace.sam3_tracker`, and
scores each trace against its own mesh (exact triangle distance; other meshes
within 2 vx of the seed count as the same sheet). `seeds` found 15 seeds among 20
compatible meshes covering z 6100–6149; seed overlays: `reports/seeded_eval/seed_check.png`.
Tracing is GPU-bound (~1 min/real slice, 97% util, 7.8/8 GB), so traces use
20 slices (6100–6119). Resume: `python scripts/seeded_eval.py trace` (skips
finished seeds), then `python scripts/seeded_eval.py score` -> `reports/seeded_eval/score.json`.

Synthetic ground-truth generator added (`src/synth/scroll.py`, `scripts/synth_{make,score,overlay}.py`,
5 tests in `tests/test_synth.py`; suite 113 passed + 1 pre-existing `node` failure).
On regular synthetic sheets (1024 px window of a radius-1500 scroll) the tracker
traced only 4/50 (crush 0, 2) and 12/50 (crush 4) slices: SAM 3.1 returned empty
masks (`missing_sam_guide`) after ~4 slices; traced points were ~48–51% on the
clicked sheet, 35–45% on a neighbour. Synthetic stripes lack real CT texture, so
treat this as a stress test, not an accuracy estimate. The "Missing keys" log
lines are harmless: the final full-model load misses only 64 `freqs_cis` buffers.

# Previous checkpoint: CPU OCR investigation

110 tests pass. CPU Tesseract grc with enhanced grayscale and PSM13 added to demo. Normalized CER84.62%, strict93.48%, exact0/5; 0.3–0.6s; 56.45MiB peak process RAM. Modest normalized gain; strict worsens. All failed experiments retained. See reports/reading/OCR-INVESTIGATION.md. Original Spark baselines and user browser drafts preserved.

# Latest checkpoint: Greek image recognition

Read & Translate now has live and recorded Spark image-only predictions linked to five regions, explicit human acceptance, correction history and provenance export. 103 tests pass. Recorded baseline: 23.28 seconds, 95.38% normalized CER, 0/5 exact lines. Browser repeat: 26.14 seconds, 87.69% normalized CER, 0/5 exact lines. Accuracy insufficient; no success claim. See reports/reading/RECOGNITION.md and docs/hackathon-demo.md. CPU cropping / remote inference; no local GPU required. Existing user draft namespace preserved.

# Unscroll — Resume Checkpoint

## Latest: Read & Translate (2026-10-28)

Integrated `viewer/read.html`: official published Greek passage (PHerc. Paris.
4, col. −8, ll. 10–14 excerpt), five image-linked lines, immutable scholarly
source plus versioned human corrections, English/Tamil editable drafts,
stale-translation tracking, local persistence and JSON evidence export.
Real Muse Spark 1.3 English/Tamil request succeeded in 13.73 s; recorded replay
is bundled, and corrected-text live translation worked through the local UI.
No local GPU needed. **92 tests passed**, browser QA and source/export integrity
checks passed. See `reports/reading/RESULTS.md` and updated hackathon walkthrough.
This is a separate published reading example, not text recovered from our CT
experiment. Model and editorial translations remain unreviewed drafts.

## Current checkpoint: independent surface evaluation and workbench (2026-10-28)

The older checkpoint below is historical. Start with `README.md` and
`reports/mesh_reference/RESULTS.md`. Tests: **77 passed**. Open the verified
workbench with `.\.venv\Scripts\python.exe scripts/workbench.py --open`.

Mesh coordinates were already volume XYZ; the original reference was selected
for z overlap but occupied a different XY region. New validation rejects
mismatched volume IDs/dimensions and local-index z, uses exact triangle
distances, and keeps one mesh identity for the whole experiment.

Frozen corrected 50-slice trace vs released candidate `20231210121321`:
mean distance 20.899 -> 10.679 voxels; within 2 voxels 2.950% -> 6.951%.
Ridge proxy remains 80.502%, continuity/kink passes 49/50. **Surface accuracy
is not established**: original click is 12.107 voxels from the candidate;
target identity is unconfirmed. Catalog: 256 entries, 4 unresolved, 1 candidate
excluded for wrong volume. Do not use historical G1 or ridge PASS as accuracy.

`viewer/` is a 5.23 MB offline replay with CT/mesh overlays, error chart,
flattened intensity, 3D preview, review-marker exports and predicted OBJ export.
CPU validation took 78.49 s and peaked at 492.93 MiB; no GPU used. No new ink
reading or translation claim. See `docs/hackathon-demo.md` for the pitch and
highest-priority next experiments.

---

# Historical checkpoint (translation slice done)

## How to resume
```powershell
cd D:\Meta\unscroll
.\.venv\Scripts\python.exe -m pytest tests/ -q   # expect 20 passed
.\.venv\Scripts\python.exe scripts/demo.py --offline  # expect GOLDEN OK
```
Key live paths: `.venv/` (torch 2.5.1+cu121), `checkpoints/sam3.1_multiplex.pt` (3.26GB),
`.env` (META_MODEL_API_KEY), `data/cache/{frag1,frag3,scroll1_6100_6200,scroll1_segments}/`.

## DONE (verified this session)
- **SAM 3.1 local tracing** (`src/trace/sam3_tracker.py`): config-C prompt
  (click + ridge-snapped tangent ±60 positives, tight normal ±15/30/50 negatives),
  direct tracker.propagate (multiplex merge drops point-only objects — bypassed),
  full-res ridge snap. Vendored fixes in `external/sam3` (all `UNSCROLL`-tagged):
  init-kwargs filter (`sam3_base_predictor.py`), SDPA flash→math fallback
  (`decoder.py`), bool-argsort cast (`sam3_multiplex_base.py`).
- **Tier B PASS**: `reports/gono_verdict.md` — 50/50 slices ridge_frac2>=0.5,
  mean 0.785 (min 0.609, max 0.919), chance ~0.26. Curves: `curves_gono/`.
  Mesh-reference G1 KNOWN-BROKEN (obj/volume offset ~330vx).
- **ridge_eval fixed**: exact 2px-dilation lookup (was stride-10 cdist, invalid).
- **Flatten**: `anchor.py` open-segment shift (was circular roll); pilot
  `sheet_pilot/` (8×512×9 surface + flat preview, coherent texture).
- **Ink CNN**: `src/ink/{surface,dataset,model,train}.py`. frag1/frag3 zsurf cached.
  20-epoch run: best val_f1=0.5446, AUC~0.70 (`reports/ink20/best.pt`, `history.json`).
  WEAK — faint-signal problem; top resume risk (see below).
- **Spark agents** (`src/agents/spark.py`): Greek read, hieroglyph mode, translation,
  offline stubs; live test OK (χαῖρε κόσμε → Hello world).
- **Viewer/demo assets**: `scripts/render_frames.py` (pilot frames in `demo/frames_pilot/`),
  `demo/viewer.html` + `scripts/build_viewer.py`. Viewer load UNVERIFIED (no browser/Node).
- **Translation slice (2026-10-28)**: `src/agents/translate.py` (6 langs,
  fluent/gloss, offline stubs), `src/agents/harness.py` (harness2-style
  Propose-Probe-Compose + lessons.jsonl), `scripts/translate_demo.py`,
  `scripts/freeze_golden.py` + `scripts/demo.py --offline` (GOLDEN OK),
  viewer translation panel, golden/ frozen. Report: `reports/translation_demo.md`.
  Tests: 20 passed (11 old + 9 new).

## PENDING (in order)
1. Ink: diagnose/boost (threshold tuning — F1@0.5 volatile while AUC 0.70;
   try F1-optimal threshold, more epochs/patches, deeper 2D head, label-alignment check).
2. Flatten 50-slice gono sheet: `flatten_sheet.py --curves curves_gono ... --out sheet_gono`,
   then ink inference on scroll surface (depth-9 vs trained-16 mismatch: resample or retrain half_width=8).
3. Spark Greek read on real sheet + hieroglyph validation.
4. Rebuild viewer for gono frames; README; `demo.py --offline`; freeze golden/ + SHA256SUMS.

## Notes
- GPU: RTX 4060 laptop 8GB. Tracker ~7.4GB; ink train ~1.5GB. Never run both at once.
- `day2_gono.py` now drives sam3 + ridge scoring (was sam2 + mesh).
- Debug scripts `scripts/sam3_debug*.py`, `sam31_*.log`, `sam3_noise*.log` — scratch, safe to delete.
- Env: venv python = `.\.venv\Scripts\python.exe`; run from `D:\Meta\unscroll`.
