# Submission form, field by field

Portal: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/hackathon/submit
(the "Submit Project" dialog). Fields below are in the form's order, with the
exact answer to paste or pick. Read on 2026-11-04 while signed in as jayaudaykmar.

Rules for whoever fills it in, person or agent: fill every "ANSWER" field as
written here; pick the radio options marked here; leave the fields marked
"USER DECIDES" for the user; never tick the permission checkbox or press Submit
unless the user says so in that conversation.

**Deadline:** Wed Nov 4, 9:00 PM ET (Thu Nov 5, 7:30 AM IST).

## Before filling: two links still missing

- [ ] **3-Minute Demo Video URL.** Upload `demo/video/unscroll-demo.mp4`
      (YouTube unlisted or similar), check it plays signed out, paste the link
      into field 7 below.
- [ ] Optional: X post link for the Most Viral Track (field 4). Leave blank if none.

## Fields

### 1. Team Name
ANSWER: `Unscroll`

### 2. Team Members
Already shows jayaudaykmar (Jayachandiran Udayakumar). No one to add.

### 3. Which track are you submitting to? (required)
PICK: **Indie Developer Track**

### 4. Which problem statement did you build towards? (required)
PICK: **Push the Frontier**

### 5. Most Viral Track: X post link
USER DECIDES. Leave blank unless an X post exists.

### 6. Project Description
ANSWER (paste as one block):

```
Unscroll: scholar-guided tracing of the Herculaneum scrolls, measured against the sheets people traced by hand.

The Herculaneum scrolls were carbonized by Vesuvius in 79 AD and can only be read from CT scans, by following one papyrus sheet through thousands of slices where neighbouring sheets are crushed together. The Vesuvius Challenge's 2026 open-problems list says no method yet traces a complete, correct surface automatically, and names a trace switching onto a neighbouring sheet among the failures it sees most. A trace that switches reads the wrong page, and nothing warns you.

Unscroll turns one click on a CT slice into a SAM 3.1 trace and scores it, with exact point-to-triangle distances, against layers released by the Vesuvius Challenge community. That paired test found the slice where traces go wrong, the first one, and a local walk from the click fixes it. On 74 clicks never used while building the fix, the share of each trace within 40 microns of the hand-traced sheet doubled, from 25% to 50% (better on 50 clicks, worse on 19, p = 0.0002). As a control, a classic optical-flow tracer from the same start scores 4.5 points lower; on the 15 development clicks that comparison is a tie, and we say so.

A Muse Wonder agent extends traces by itself: it inspects a trace, re-seeds where tracking stops, launches Kaggle GPU runs, scores and unrolls the result, and checks each seam with continuity numbers plus a Muse Spark visual verdict. It caught its own first re-seed jumping to another sheet and excluded three runs, with reasons logged.

What stands between a trace and readable text is also measured. Wobble from slice to slice, not distance, destroys the ink. Smoothing Unscroll's own trace on a sheet with known Greek text raises agreement with the official ink map from 0.21 to 0.73, against 0.83 for the official surface: broad strokes, not letters. A cost-of-reading test does this for one, five and ten scholar clicks. Unscroll has not read any new text; it measures exactly how far from reading a trace is.

Everything in the repository was built during the hackathon (Oct 27 to Nov 4). Used, not built, and credited: SAM 3.1 and the Muse models (Meta), the 2023 Vesuvius Grand Prize ink model (MIT), and the Scroll 1 CT and human-traced layers (EduceLab-Scrolls / Vesuvius Challenge, CC BY-NC 4.0, used with the team's permission). Tests: 150 passed. Interactive Story page, no install: https://jayaudaykmar26589.github.io/unscroll/viewer/story.html
```

### 7. Public GitHub Repository
ANSWER: `https://github.com/Jayaudaykmar26589/unscroll`

### 8. 3-Minute Demo Video
ANSWER: the uploaded video's URL (see "Before filling"). The file is
`demo/video/unscroll-demo.mp4`, 2:55.

### 9. Describe your use of Meta products to build the submitted project
ANSWER (paste as one block):

```
SAM 3.1 (open weights, run on a Kaggle T4 and locally): tracks the clicked papyrus sheet through the CT slice stack as if it were a video. Unscroll adds prompt design, a CT-ridge refinement and the first-slice fix, and ships four fixes to the SAM 3.1 code as a patch: an exact query-chunked attention path (bit-identical to full attention in a test) that removes a multi-GB allocation which ran out of memory on a 16 GB GPU, a CPU fallback where Triton is unavailable on Windows, an init-argument filter and a boolean sort cast.

Muse Wonder (Meta Model API, tool calling): the patch-growing agent. Given tools to inspect a trace, propose a re-seed, run a Kaggle trace, score, unroll, review a seam and exclude a run, it extended a 100-slice trace, then used the seam check (continuity numbers plus a Muse Spark visual verdict) to find that its first re-seed had jumped to another sheet, and excluded that run and the two built on it. Every decision is logged in reports/agent/.

Muse Spark 1.3 (Meta Model API): reviews traces from CT images (it never accepted a wrong-layer trace, 0 of 26, but flags most good ones too, so it is a safe second opinion rather than a workload saving); writes plain-language QA notes for scholars in which every number is checked automatically against the data; gives the visual verdict in the agent's seam check; and drafts readings and English and Tamil translations of a published Greek passage for a person to accept or correct.

Muse Voice Transcribe (Meta Model API): hands-free review in the surface workbench, push-to-talk commands such as "next", "slice 20", "worst", "mark".

Muse Image was deliberately not used: generated scroll imagery would mislead. No -contributor model variants were used.
```

### 10. Why are Meta's models important to your solution?
ANSWER (paste as one block):

```
SAM 3.1 is the tracer. Its video tracking is what lets one click on one CT slice become a trace through the following slices without any training on scroll data; Unscroll's job was to measure where it fails and fix the slice where it does. The whole benchmark result, 25% to 50% of each trace staying on the sheet, is SAM 3.1 plus a first-slice fix, and the patch that made it run on a 16 GB GPU is part of the submission.

Muse Wonder is what makes the system agentic rather than a script: it decides where to re-seed, spends a limited budget of GPU runs, and, given a seam check, catches and documents its own mistakes. Muse Spark adds the visual judgement a scholar would otherwise supply, in the seam check and the trace review, and writes the number-checked notes that make the results readable by a papyrologist. Muse Voice keeps a reviewer's hands free. Together they cover the loop from click to verified trace to plain-language report.
```

### 11. Provide feedback from your experience using partner technologies
ANSWER: `Not used: all GPU work ran on Kaggle (T4) and Meta Model API.`

### 12. How likely are you to build with Meta Model API? (required)
USER DECIDES (Very likely / Somewhat likely / Neutral / Somewhat unlikely / Very unlikely).

### 13. How would you describe your overall opinion of Meta Model API? (required)
USER DECIDES (Very favorable / Somewhat favorable / Neutral / Somewhat unfavorable / Very unfavorable).

### 14. Permission checkbox
"By accepting, the submitter(s) grants Meta and Cerebral Valley permission to use,
publish, share…" USER DECIDES. Read it in the form first; the agent never ticks it.

### 15. How likely are you to recommend the Meta Model API (0–5)
USER DECIDES.

### 16. How likely are you to recommend Muse Code (0–5)
USER DECIDES. (Muse Code was not used in this project.)

### 17. Biggest obstacle while building
USER DECIDES. Options include: access or account setup; documentation or code
examples; connecting to tools; wrong or unhelpful model answers; the model not
following instructions; a missing model capability; slow responses; errors or
outages; rate limits; Muse Code issues; something else; no obstacles.
For reference, the obstacles recorded in RESUME.md were compute and memory
(the SAM 3.1 attention allocation on a 16 GB GPU, Kaggle's two-session limit),
not the API.

### 18. Compared with other AI models for the same work, Meta's models performed…
USER DECIDES (Much better / Somewhat better / About the same / Somewhat worse /
Much worse / couldn't compare / haven't used others).

### 19. When choosing an AI model or coding tool, what matters most?
USER DECIDES (quality; speed; price; long context; agent abilities; reliability;
easy integration; fits workflow; data privacy; open weights; rate limits; other).

### 20. What makes it hardest to switch model or tool?
USER DECIDES.

### 21. In production, what would most likely stop you using Meta's models?
USER DECIDES (includes "We don't plan to take this project to production").

### 22. Submit button
Never pressed by the agent. The user presses it, or says "submit" in that conversation.

## After submitting (user)

- Rotate the Kaggle token and the Meta Model API key.
- Decide whether scratch files move into `_archive/`.
